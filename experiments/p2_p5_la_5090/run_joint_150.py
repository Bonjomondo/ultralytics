"""Run only the missing C2+LA 150e arm, then evaluate and time trained weights.

Existing C0/C2 runs are read-only references. No speed screen blocks training.
The controller is intended for a persistent tmux session; all runtime output is
server-only under a fresh root, with atomic status/progress and phase logs.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1/5090-rtdetr-r18-c2-150e-s0/weights/best.pt'
BASELINE_SHA = 'ac2b02c45596bb816c4a42799d2b97ff02d62a4dc153bc5816502f02e936a5d6'
NAME = '5090-c2-la-150e-s0'


def read(path):
    return json.loads(path.read_text())


def progress(root, state):
    run = root / NAME
    training = read(run / 'experiment.json') if (run / 'experiment.json').exists() else {}
    if training:
        state['completed_epochs'] = training.get('completed_epochs', 0)
        state['training_status'] = training.get('status')
        state['latest_internal_map'] = training.get('latest_internal_map')
    text = ['# C2＋局部注意力150轮从零训练', '',
            f"队列状态：{state['status']}；阶段：{state.get('phase')}；训练：{state.get('completed_epochs', 0)}/150。", '',
            '仅训练C2＋LA；已有C0/C2不重训。seed0、batch8、640、FP32、原AdamW配方、全量训练集。',
            '训练结束自动执行548图原生FP32 COCO评估和历史C2/新C2＋LA同场测速；FP16仅性能，不代表FP16 AP。', '']
    if training.get('best_internal_map') is not None:
        text.append(f"目前最佳训练内部mAP：{100*training['best_internal_map']:.3f}%；不等同正式COCO AP。")
    (root / 'progress.md').write_text('\n'.join(text) + '\n')


def run(args):
    from experiments.decoder_budget_5090.train_5090 import git, now, sha256_file, write_atomic
    from experiments.p2_p5_la_5090.model import MODELS, build_model
    from experiments.p2_p5_la_5090.train import source_hashes
    from experiments.r18_p2_sfi.train import tensor_digest
    root = args.root.resolve()
    if not args.deployment_python.is_file() or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError('Missing deployment Python or historical baseline identity mismatch')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    shutil.copyfile(args.data, root / 'data.yaml')
    state = {'schema': '5090.joint150.queue.v1', 'status': 'running', 'phase': 'prepare',
             'started_utc': now(), 'root': str(root), 'train_run': str(root / NAME),
             'epochs_requested': 150, 'variant': 'c2-la', 'seed': 0,
             'source_branch': git('branch', '--show-current'), 'source_commit': git('rev-parse', 'HEAD'),
             'source_sha256': source_hashes(), 'annotations': str(args.annotations.resolve()),
             'images': str(args.images.resolve()), 'data_yaml_sha256': sha256_file(root / 'data.yaml'),
             'annotation_sha256': sha256_file(args.annotations),
             'reused_baseline': {'path': str(BASELINE), 'sha256': BASELINE_SHA},
             'new_training': ['c2-la'], 'retrain_existing': False,
             'speed_screen_blocks_training': False, 'phases': [], 'child_pid': None}

    def save():
        state['updated_utc'] = now()
        progress(root, state)
        write_atomic(root / 'status.json', state)

    def frozen():
        if (source_hashes() != state['source_sha256']
                or sha256_file(root / 'data.yaml') != state['data_yaml_sha256']
                or sha256_file(args.annotations) != state['annotation_sha256']):
            raise RuntimeError('Frozen execution source or input changed')

    def stage(name, command):
        frozen()
        phase = {'name': name, 'command': command, 'status': 'running', 'started_utc': now(),
                 'log': str(root / 'logs' / (name + '.log'))}
        state['phase'] = name
        state['phases'].append(phase)
        with Path(phase['log']).open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save()
            while True:
                try:
                    code = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    save()
        state['child_pid'] = None
        phase.update(returncode=code, finished_utc=now(), status='completed' if code == 0 else 'failed')
        save()
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; see {phase["log"]}')
        frozen()

    save()
    try:
        audit_script = ROOT / 'docs/experiment_reports/2026-10-24_5090_baselines/audit_assets_5090.py'
        stage('data-audit', [sys.executable, str(audit_script), '--output', str(root / 'asset_audit.json')])
        audit = read(root / 'asset_audit.json')
        old = read(audit_script.parent / 'raw/asset_audit_5090.json')
        assert audit['status'] == 'passed' and audit['coco']['sha256'] == old['coco']['sha256']
        assert all(audit[k] == old[k] for k in ('label_content_sha256', 'image_inventory_sha256'))
        model = build_model(str(MODELS['c2-la']))
        plan = {'variant': 'c2-la', 'epochs': 150, 'fraction': 1, 'seed': 0,
                'pretrained': False, 'resume': False, 'checkpoint_initialization': None,
                'initialization': model.ablation_initialization, 'full_initial_state_sha256': tensor_digest(model.state_dict()),
                'config_sha256': sha256_file(MODELS['c2-la']), 'source_sha256': state['source_sha256'],
                'reused_baseline': state['reused_baseline'], 'new_training': ['c2-la']}
        write_atomic(root / 'plan.json', plan)
        del model
        stage('train-150e', [sys.executable, '-u', '-m', 'experiments.p2_p5_la_5090.train',
                            '--variant', 'c2-la', '--data', str(root / 'data.yaml'), '--project', str(root),
                            '--name', NAME, '--epochs', '150', '--seed', '0', '--fraction', '1'])
        training = read(root / NAME / 'experiment.json')
        assert training['status'] == 'completed' and training['completed_epochs'] == 150
        assert all(training['initialization'][k] == plan['initialization'][k]
                   for k in ('shared_state_sha256', 'p2_slot_sha256', 'p5_slot_sha256'))
        stage('evaluate-full548', [sys.executable, '-u', '-m', 'experiments.p2_p5_la_5090.evaluate',
                                  '--variant', 'c2-la', '--run', str(root / NAME), '--annotations', str(args.annotations.resolve()),
                                  '--images', str(args.images.resolve()), '--output', str(root / 'evaluation')])
        stage('benchmark-trained', [str(args.deployment_python.resolve()), '-u', '-m',
                                    'experiments.p2_p5_la_5090.run_joint_150', '--stage', 'benchmark', '--root', str(root)])
        final_report(root)
        state.update(status='completed', phase='finished', completed_utc=now())
        save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise


def benchmark(root):
    import fcntl
    import torch
    from experiments.decoder_budget_5090.profile_runtime_5090 import (
        DEFAULT_DATA, configure, environment, load_input, measure, sha, stats, write,
    )
    from experiments.decoder_budget_5090.screen_decoder_perf_5090 import PredictionOnly, ready_paths
    from experiments.p2_p5_la_5090.model import inspect_model
    with open('/tmp/zhj-uav-5090-gpu0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure()
        output = root / 'benchmark'
        output.mkdir(exist_ok=False)
        evaluation = read(root / 'evaluation/summary.json')
        weights = {'c2': BASELINE, 'c2-la': Path(evaluation['checkpoint']['path'])}
        assert sha(BASELINE) == BASELINE_SHA
        assert sha(weights['c2-la']) == evaluation['checkpoint']['sha256']
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                x, identity = load_input(DEFAULT_DATA)
                functions, conditions, keep = {}, {}, []
                for arm, path in weights.items():
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    model = (saved.get('ema') or saved['model']).float().eval().cuda()
                    inspect_model(model, arm)
                    fns, records, objects = ready_paths(PredictionOnly(model), [x], ['images'], output / arm,
                                                       True, ['eager_fp32', 'trt_fp32', 'trt_fp16'])
                    keep.extend(objects)
                    for backend, record in records.items():
                        key = arm + '/' + backend
                        record['checkpoint'] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
                        conditions[key] = record
                    functions.update({arm + '/' + k: v for k, v in fns.items()})
                for fn in functions.values():
                    for _ in range(150):
                        fn()
                torch.cuda.synchronize()
                for index in range(20):
                    order = list(functions)
                    random.Random(20261026 + index).shuffle(order)
                    for key in order:
                        block = measure(functions[key], 10, 50)
                        block['round'] = index + 1
                        conditions[key]['blocks'].append(block)
                for key in functions:
                    condition = conditions[key]
                    for clock in ('cuda', 'wall'):
                        condition[clock + '_stats_ms'] = stats([v for b in condition['blocks'] for v in b[clock + '_ms']])
                    condition['status'] = 'completed'
                write(output / 'assessment.json', {'status': 'completed' if len(functions) == 6 else 'partial_backends_failed',
                      'environment': environment('cuda:0'), 'input': identity, 'conditions': conditions,
                      'protocol': {'rounds': 20, 'samples_per_round': 50, 'warmup': 150,
                                   'scope': 'GPU-resident B1 640 model forward; no preprocessing/transfer/postprocessing',
                                   'precision': 'Native FP32 AP evaluated separately; TensorRT/FP16 timing does not establish FP16 AP',
                                   'baseline': 'Reused historical C2; no retraining'}})
            stream.synchronize()


def final_report(root):
    from experiments.decoder_budget_5090.report_decoder_screen_5090 import reduction
    native = read(root / 'evaluation/summary.json')
    old = read(ROOT / 'docs/experiment_reports/2026-10-25_5090_decoder_screen/raw/depth-eval-c2.json')
    assert native['scope'] == 'full_val' and len(native['protocol']['image_ids']) == 548
    assert native['protocol']['annotations_sha256'] == old['formal_protocol']['annotation_sha256']
    timing = read(root / 'benchmark/assessment.json')
    report = root / 'report'
    report.mkdir(exist_ok=False)
    rows = [{'variant': arm, **metrics} for arm, metrics in
            [('c2 historical150e', old['conditions']['d3']['metrics']), ('c2-la new150e', native['metrics'])]]
    with (report / 'accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# C2＋局部注意力从零150轮结果', '',
             '新训练仅C2＋LA；C2沿用已完成的同配方150轮seed0基线。不是新四组正交实验，不计算交互项。', '',
             '| 模型 | 原生FP32正式AP | APs | APm | APl |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + row['variant'] + ' | ' + ' | '.join(f'{100*row[k]:.3f}' for k in ('AP', 'APs', 'APm', 'APl')) + ' |')
    lines += ['', 'AP及差值使用百分点；原生FP32精度与TensorRT/FP16性能分开，不宣称已验证FP16 AP。', '',
              '| 后端 | C2/ms | C2＋LA/ms | 延迟下降 | 配对95%区间 |', '|---|---:|---:|---:|---:|']
    for backend in ('eager_fp32', 'trt_fp32', 'trt_fp16'):
        a, b = [timing['conditions'][arm + '/' + backend] for arm in ('c2', 'c2-la')]
        if all(v['status'] == 'completed' for v in (a, b)):
            effect = reduction(a, b)
            lines.append(f"| {backend} | {a['cuda_stats_ms']['mean']:.4f} | {b['cuda_stats_ms']['mean']:.4f} | {effect['latency_reduction_percent']:.2f}% | [{effect['ci95_low']:.2f}%, {effect['ci95_high']:.2f}%] |")
        else:
            lines.append(f'| {backend} | 执行失败/不完整，见benchmark记录 | — | — | — |')
    lines += ['', '单seed、150轮固定预算，尚不自动证明收敛；训练曲线/health/源码/输入/checkpoint身份保存在run中，原始计时在benchmark中。', '']
    (report / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('run', 'benchmark'), default='run')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--deployment-python', type=Path)
    args = parser.parse_args()
    if args.stage == 'benchmark':
        benchmark(args.root.resolve())
    else:
        if not all((args.data, args.annotations, args.images, args.deployment_python)):
            parser.error('run requires data, annotations, images and deployment-python')
        run(args)
