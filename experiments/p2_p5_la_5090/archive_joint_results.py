"""Archive the completed C2+LA training and repaired deployment benchmark.

Collect requires server results. Render/verify require only tracked text and the
standard library; weights, exported engines, predictions and full logs stay on
the server. Historical C2 is reused, not presented as a new factorial matrix.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from experiments.decoder_budget_5090.report_decoder_screen_5090 import reduction

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'docs/experiment_reports/2026-10-27_5090_c2_la_150e_results'
START = ROOT / 'docs/experiment_reports/2026-10-26_5090_c2_la_150e_launch'
BASE = ROOT / 'docs/experiment_reports/2026-10-25_5090_decoder_screen/raw/depth-eval-c2.json'
OLD_REPORT = ROOT / 'reports/5090/experiments.json'
OLD_CATEGORY = ROOT / 'docs/experiment_reports/2026-10-26_5090_fresh_decoder_results/raw/category_ap.csv'
BASE_SHA = 'ac2b02c45596bb816c4a42799d2b97ff02d62a4dc153bc5816502f02e936a5d6'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = path.resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(args):
    run, retry = args.run.resolve(), args.retry.resolve()
    status = read(run / 'status.json')
    train = read(run / '5090-c2-la-150e-s0/experiment.json')
    evaluation = read(run / 'evaluation/summary.json')
    original = read(run / 'benchmark/assessment.json')
    benchmark = read(retry / 'benchmark/assessment.json')
    assert status['source_sha256'] == train['source_sha256'] == evaluation['training_source_sha256']
    assert status['status'] == train['status'] == evaluation['status'] == 'completed'
    assert original['status'] == 'partial_backends_failed'
    assert all(original['conditions'][arm+'/'+backend]['status'] == 'failed'
               for arm in ('c2', 'c2-la') for backend in ('trt_fp32', 'trt_fp16'))
    raw = args.output / 'raw'
    raw.mkdir(parents=True, exist_ok=False)
    compact_status = copy.deepcopy(status)
    compact_status.pop('source_sha256')
    compact_eval = copy.deepcopy(evaluation)
    compact_eval.pop('training_source_sha256')
    compact_benchmark = copy.deepcopy(benchmark)
    for condition in compact_benchmark['conditions'].values():
        for block in condition.get('blocks', []):
            block.pop('wall_ms', None)
    write(raw / 'status.json', compact_status)
    write(raw / 'training.json', train)
    write(raw / 'evaluation.json', compact_eval)
    write(raw / 'benchmark.json', compact_benchmark)
    shutil_path = run / '5090-c2-la-150e-s0/results.csv'
    (raw / 'training_curve.csv').write_bytes(shutil_path.read_bytes())
    health = [json.loads(line) for line in (run / '5090-c2-la-150e-s0/health.jsonl').read_text().splitlines()]
    fields = list(dict.fromkeys(key for row in health for key in row))
    write_csv(raw / 'training_health.csv', [{key: row.get(key) for key in fields} for row in health])
    original_command = next(p['command'] for p in status['phases'] if p['name'] == 'benchmark-trained')
    retry_log = args.retry_log.read_text(errors='replace')
    files = [run / 'status.json', run / '5090-c2-la-150e-s0/experiment.json',
             run / '5090-c2-la-150e-s0/results.csv', run / '5090-c2-la-150e-s0/health.jsonl',
             run / 'logs/train-150e.log', run / 'logs/evaluate-full548.log', run / 'logs/benchmark-trained.log',
             run / 'evaluation/summary.json', run / 'evaluation/predictions.json', run / 'benchmark/assessment.json',
             retry / 'benchmark/assessment.json', args.retry_log]
    files.extend(Path(info['path']) for info in train['checkpoints'].values())
    write(raw / 'validation.json', {'original_benchmark': artifact(run / 'benchmark/assessment.json'),
         'original_status': original['status'], 'original_failed_conditions': {
             key: {'status': value['status'], 'error': value.get('error')}
             for key, value in original['conditions'].items() if value['status'] != 'completed'},
         'root_cause': 'The controller resolves the deployment venv Python symlink before launch, selecting the base interpreter without TensorRT. The retry invokes the venv executable path directly.',
         'original_command': original_command, 'retry_executable': benchmark['environment']['executable'],
         'retry_status': benchmark['status'], 'retry_log': artifact(args.retry_log),
         'retry_log_warning_counts': {'fp16_clipped_constants': retry_log.count('contains out-of-range weights when cast to Half'),
                                      'fp16_layernorm': retry_log.count('Detected layernorm nodes in FP16')},
         'source_before_retry': artifact(run / 'evaluation/summary.json'),
         'scope': 'Both AP values are native FP32 COCO; TensorRT FP16 speed is not FP16 accuracy validation.',
         'server_only': [artifact(p) for p in files]})


def attach_fp16(archive, path, replay_path):
    result=read(path)
    assert result['status']=='completed' and result['protocol']['scope']=='full_val'
    replay=read(replay_path)
    assert replay['status']=='completed'
    notes=read(archive/'raw/validation.json')
    notes['fp16_coco_summary']=artifact(path)
    notes['fp16_independent_replay']=artifact(replay_path)
    items=notes['server_only']+[artifact(path),artifact(replay_path),*[artifact(Path(v['predictions']['path'])) for v in result['models'].values()]]
    notes['server_only']=list({v['path']:v for v in items}.values())
    notes['fp16_evaluation_script_sha256']=sha(ROOT/'experiments/p2_p5_la_5090/evaluate_trt_fp16_5090.py')
    notes['scope']='Native FP32 and the specific TensorRT FP16 engine paths were each evaluated on the full validation set.'
    write(archive/'raw/trt_fp16_coco.json',result)
    write(archive/'raw/trt_fp16_replay.json',replay)
    write(archive/'raw/validation.json',notes)


def inputs(archive):
    raw = archive / 'raw'
    return (read(raw/'status.json'), read(raw/'training.json'), read(raw/'evaluation.json'),
            read(raw/'benchmark.json'), read(BASE), read_csv(raw/'training_curve.csv'),
            read_csv(raw/'training_health.csv'))


def validate(archive):
    status, train, ev, benchmark, old, curve, health = inputs(archive)
    assert status['status'] == 'completed' and all(p['returncode'] == 0 for p in status['phases'])
    assert train['status'] == 'completed' and train['variant'] == 'c2-la' and train['completed_epochs'] == 150
    assert train['seed'] == 0 and train['fraction'] == 1 and train['role'] == 'fixed_budget_screen'
    assert not train['recipe']['pretrained'] and not train['recipe']['resume']
    assert train['initialization']['source'].startswith('from scratch') and train['initialization']['shared_state_verified']
    assert train['architecture']['p2_injection'] and train['architecture']['p5_na7_rpb']
    assert (train['architecture']['queries'],train['architecture']['decoder_layers'],train['architecture']['points_per_level']) == (300,3,4)
    assert len(curve) == len(health) == 150
    assert [int(float(r['epoch'])) for r in curve] == [int(r['epoch']) for r in health] == list(range(1,151))
    for row in curve:
        assert all(math.isfinite(float(v)) for v in row.values())
    for row in health:
        for k in ('p2_first_weight_grad_abs','p5_qkv_grad_abs','p5_rpb_grad_abs'):
            assert float(row[k]) > 0
        assert all(float(v) == 1 for k,v in row.items() if k.endswith('_finite'))
    assert ev['status'] == 'completed' and ev['scope'] == 'full_val' and len(ev['protocol']['image_ids']) == 548
    assert ev['training_fraction'] == 1 and ev['training_epochs'] == 150 and ev['seed'] == 0
    assert ev['protocol']['dtype'] == 'float32' and not ev['protocol']['fused']
    assert ev['protocol']['formal_max_dets'] == [1,10,100]
    assert ev['protocol']['annotations_sha256'] == old['formal_protocol']['annotation_sha256']
    assert ev['protocol']['image_ids'] == old['formal_protocol']['image_ids']
    assert ev['checkpoint'] == train['checkpoints']['best'] and ev['model_state_unchanged']
    assert old['checkpoint']['sha256'] == BASE_SHA == status['reused_baseline']['sha256']
    plan = read(START/'raw/plan.json')
    for k in ('shared_state_sha256','p2_slot_sha256','p5_slot_sha256'):
        assert train['initialization'][k] == plan['initialization'][k]
    assert train['source_sha256'] == plan['source_sha256']
    for name, digest in train['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert train['data_yaml_sha256'] == sha(START/'raw/data.yaml')
    assert read(START/'raw/asset_audit.json')['coco']['sha256'] == ev['protocol']['annotations_sha256']
    assert benchmark['protocol']['rounds'] == 20 and benchmark['protocol']['samples_per_round'] == 50
    conditions = benchmark['conditions']
    assert len(conditions) == 6
    for arm in ('c2','c2-la'):
        for backend in ('eager_fp32','trt_fp32','trt_fp16'):
            item = conditions[arm+'/'+backend]
            expected = BASE_SHA if arm == 'c2' else ev['checkpoint']['sha256']
            assert item['checkpoint']['sha256'] == expected
            if item['status'] == 'completed':
                assert item['output_check']['all_finite'] and len(item['blocks']) == 20
                samples = [v for block in item['blocks'] for v in block['cuda_ms']]
                assert len(samples) == 1000 and all(v > 0 and math.isfinite(v) for v in samples)
                assert math.isclose(statistics.fmean(samples),item['cuda_stats_ms']['mean'],abs_tol=1e-12)
                if backend.startswith('trt'):
                    assert item['changed_input_output']['all_finite']
    assert len(ev['per_class']) == 10
    assert math.isclose(statistics.fmean(v['AP'] for v in ev['per_class'].values()),ev['metrics']['AP'],abs_tol=1e-12)
    fp16=read(archive/'raw/trt_fp16_coco.json')
    assert fp16['status']=='completed' and fp16['protocol']['scope']=='full_val' and fp16['protocol']['images']==548
    assert fp16['protocol']['image_ids']==ev['protocol']['image_ids']
    assert fp16['protocol']['annotation_sha256']==ev['protocol']['annotations_sha256']
    assert fp16['protocol']['formal_max_dets']==ev['protocol']['formal_max_dets']
    assert fp16['protocol']['category_ids']==ev['protocol']['category_ids']
    assert fp16['protocol']['conf']==ev['protocol']['conf']==.001
    assert fp16['protocol']['prediction_max_det']==ev['protocol']['prediction_max_det']==300
    assert fp16['first_input_sha256']==benchmark['input']['input_sha256']
    replay=read(archive/'raw/trt_fp16_replay.json')
    assert replay['status']=='completed'
    for arm in ('c2','c2-la'):
        candidate=fp16['models'][arm]
        item=conditions[arm+'/trt_fp16']
        assert item['status']=='completed' and candidate['checkpoint']==item['checkpoint']
        assert candidate['engine']==next(a for a in item['engine']['server_artifacts'] if a['path'].endswith('.engine'))
        assert candidate['prediction_count']>0 and len(candidate['per_class'])==10
        assert math.isclose(statistics.fmean(v['AP'] for v in candidate['per_class'].values()),candidate['metrics']['AP'],abs_tol=1e-12)
        assert replay['models'][arm]['gt_count']==38759
        assert replay['models'][arm]['prediction_sha256']==candidate['predictions']['sha256']
        assert all(abs(replay['models'][arm]['metrics'][k]-v)<1e-12 for k,v in candidate['metrics'].items())
    notes = read(archive/'raw/validation.json')
    assert notes['original_status'] == 'partial_backends_failed' and len(notes['original_failed_conditions']) == 4
    assert benchmark['status'] == notes['retry_status']
    assert notes['fp16_evaluation_script_sha256']==sha(ROOT/'experiments/p2_p5_la_5090/evaluate_trt_fp16_5090.py')
    print('Verified completed 150e training, source/init/checkpoint identity, native and TRT FP16 full548 COCO, health and paired timing.')


def render(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    status, train, ev, benchmark, old, curve, health = inputs(archive)
    base_metrics, new_metrics = old['conditions']['d3']['metrics'], ev['metrics']
    params = read(ROOT/'docs/experiment_reports/2026-10-26_5090_p2_p5_la_preflight/raw/benchmark.json')['models']
    rows=[]
    for arm,metrics in (('c2',base_metrics),('c2-la',new_metrics)):
        rows.append({'variant':arm,'parameters':params[arm]['parameters_unfused'],**metrics,
                     'delta_AP_pp':100*(metrics['AP']-base_metrics['AP']),
                     'delta_APs_pp':100*(metrics['APs']-base_metrics['APs'])})
    write_csv(output/'accuracy.csv',rows)
    fp16=read(archive/'raw/trt_fp16_coco.json')
    fp16_rows=[]
    base_fp16=fp16['models']['c2']['metrics']
    for arm in ('c2','c2-la'):
        metrics=fp16['models'][arm]['metrics']
        fp16_rows.append({'variant':arm,**metrics,'delta_AP_pp':100*(metrics['AP']-base_fp16['AP']),
                          'delta_APs_pp':100*(metrics['APs']-base_fp16['APs'])})
    write_csv(output/'accuracy_trt_fp16.csv',fp16_rows)
    timing=[]
    for backend in ('eager_fp32','trt_fp32','trt_fp16'):
        a,b=[benchmark['conditions'][arm+'/'+backend] for arm in ('c2','c2-la')]
        item={'backend':backend,'c2_status':a['status'],'c2_la_status':b['status'],
              'c2_error':a.get('error'),'c2_la_error':b.get('error')}
        if a['status'] == b['status'] == 'completed':
            paired = reduction(a,b)
            for key in ('latency_reduction_percent','ci95_low','ci95_high'):
                paired[key] = round(paired[key], 9)  # Stable across Python sum implementations.
            item.update(c2_mean_ms=a['cuda_stats_ms']['mean'], c2_la_mean_ms=b['cuda_stats_ms']['mean'],
                        c2_p95_ms=a['cuda_stats_ms']['p95'],c2_la_p95_ms=b['cuda_stats_ms']['p95'],
                        c2_fps=1000/a['cuda_stats_ms']['mean'],c2_la_fps=1000/b['cuda_stats_ms']['mean'],
                        **paired)
        else:
            item.update(dict.fromkeys(('c2_mean_ms','c2_la_mean_ms','c2_p95_ms','c2_la_p95_ms','c2_fps','c2_la_fps',
                                       'latency_reduction_percent','ci95_low','ci95_high','paired_rounds')))
        timing.append(item)
    write_csv(output/'latency.csv',timing)
    baseline_categories={r['category']:r for r in read_csv(OLD_CATEGORY) if r['variant']=='c2-q300-p4-d3'}
    categories=[]
    for name in baseline_categories:
        before,after=baseline_categories[name],ev['per_class'][name]
        categories.append({'category':name,'gt_count':before['gt_count'],'c2_AP':float(before['AP']),
                           'c2_la_AP':after['AP'],'delta_AP_pp':100*(after['AP']-float(before['AP'])),
                           'c2_APs':float(before['APs']) if before['APs'] else None,'c2_la_APs':after['APs']})
    write_csv(output/'category_comparison.csv',categories)
    old_record=next(r for r in read(OLD_REPORT)['records'] if r.get('checkpoint_sha256',{}).get('best')==BASE_SHA)
    combined=[]
    for name,source in (('c2',old_record['curve']),('c2-la',curve)):
        combined.extend({'variant':name, **r} for r in source)
    write_csv(output/'training_curves.csv',combined)
    best=max(curve,key=lambda r:float(r['metrics/mAP50-95(B)']))
    gain=100*(float(curve[149]['metrics/mAP50-95(B)'])-float(curve[124]['metrics/mAP50-95(B)']))
    peak=max(float(r['train_peak_allocated_mib']) for r in health)
    duration=sum(float(r['train_seconds']) for r in health)/3600
    lines=['# C2＋P5局部注意力：从零150轮训练结果', '',
           '分支 `5090-detrscope-query-budget`。用户要求仅补跑缺少的组合模型；C0/C2沿用已有150轮结果。组合150轮、完整548图COCO评估于2026-10-27 04:29（北京时间）完成。', '',
           '**本次seed0、150轮下，组合未超过已有C2：原生FP32正式AP下降0.490点，TensorRT FP16正式AP下降0.537点。** 后端部署精度与速度使用修复虚拟环境路径后的实际验证，不以最初未成功的TensorRT结果冒充。', '',
           '## 正式精度', '',
           '原生未融合FP32，完整548图、38759标注框、batch2、640、conf=.001、每query取最高类、无NMS/坐标舍入，COCO bbox maxDets=[1,10,100]。下表AP乘100；差值是百分点。', '',
           '| 指标 | 旧C2 150e | 新C2＋LA 150e | 差/点 |', '|---|---:|---:|---:|']
    for key in ('AP','AP50','AP75','APs','APm','APl','AR100','ARs'):
        lines.append(f'| {key} | {100*base_metrics[key]:.3f} | {100*new_metrics[key]:.3f} | {100*(new_metrics[key]-base_metrics[key]):+.3f} |')
    lines += ['', '小目标AP略降，ARs增加0.115点；召回和精度反映不同量，不能据ARs上升认定AP改善。大目标AP下降3.474点，数量及类别构成可能影响该指标，单seed不足以归因。完整12项指标见[accuracy.csv](accuracy.csv)。', '',
              '类别方面，自行车AP增加0.916点，truck下降1.567点，awning-tricycle下降2.049点；其余类别见[category_comparison.csv](category_comparison.csv)。样本量和随机训练波动都会影响类别结果。', '',
              '## TensorRT FP16完整验证集精度', '',
              '使用同场测速的实际TensorRT FP16引擎，读取同一548张原图和同一标注，batch1、输入输出FP32、引擎可选择混合精度层；原生FP32评估为batch2。', '',
              '| 指标 | C2 TensorRT FP16 | C2＋LA TensorRT FP16 | 差/点 |', '|---|---:|---:|---:|']
    for key in ('AP','AP50','AP75','APs','APm','APl'):
        before,after=base_fp16[key],fp16['models']['c2-la']['metrics'][key]
        lines.append(f'| {key} | {100*before:.3f} | {100*after:.3f} | {100*(after-before):+.3f} |')
    lines += ['', '具体引擎的FP16路径已完成正式COCO评估，详细指标见[accuracy_trt_fp16.csv](accuracy_trt_fp16.csv)。独立DETRScope评估器重放预测，12项整体指标均在1e-12内重现；见[重放记录](raw/trt_fp16_replay.json)。它与原生FP32使用不同执行路径及batch大小，数值变化不应全部归因于精度转换。', '',
              '## 训练后同场测速', '',
              '训练后各自best.pt；真实验证图、GPU常驻batch1、640、纯forward，20轮随机交错、每轮50个CUDA事件样本，150次预热。Eager是未融合FP32，TensorRT先完成Conv/BN等融合。时间不含图像预处理、数据传输和检测后处理。', '',
              '| 后端 | C2/ms | C2＋LA/ms | 延迟下降 | 配对95%区间 |', '|---|---:|---:|---:|---:|']
    for row in timing:
        if row['c2_mean_ms'] is None:
            lines.append(f"| {row['backend']} | — | — | 未完成 | — |")
        else:
            lines.append(f"| {row['backend']} | {row['c2_mean_ms']:.4f} | {row['c2_la_mean_ms']:.4f} | {row['latency_reduction_percent']:+.2f}% | [{row['ci95_low']:.2f}%, {row['ci95_high']:.2f}%] |")
    fp16_latency=next(r for r in timing if r['backend']=='trt_fp16')
    if fp16_latency['latency_reduction_percent'] is not None:
        lines += ['', f"本次TensorRT FP16的组合模型延迟变化为{-fp16_latency['latency_reduction_percent']:+.2f}%（正值表示变慢），与实测精度一起看，没有出现同时提精度和降延迟的证据。"]
    lines += ['', '延迟下降=100×(1−组合/C2)，负数表示变慢；95%区间按20个配对轮次bootstrap，不代表不同输入、重新构建引擎或不同设备的波动。FPS为纯forward延迟倒数，见[latency.csv](latency.csv)。', '',
              '第一次自动部署测速因控制器将虚拟环境Python符号链接解析成基础解释器，导致找不到TensorRT；顶层队列虽完成，原部署记录明确为`partial_backends_failed`。随后直接使用虚拟环境入口补测，原失败日志和结果仍保留服务器，原因和SHA见[验证记录](raw/validation.json)。', '',
              'TensorRT构建日志的FP16常量裁剪及LayerNorm相关警告按实际出现次数记录在验证文件；两个实际引擎均完成了有限输出、变更输入检查和完整验证集COCO评估。', '',
              '## 训练与解释边界', '',
              f"新组合从零训练：150轮、seed0、batch8、640、FP32、AdamW lr=1e-4、lrf=1、warmup3、weight_decay=1e-4、mixup=.2、workers0、无预训练或旧checkpoint续训。训练部分累计{duration:.2f}小时，PyTorch训练峰值allocated {peak:.0f} MiB；不等同GPU总占用。", '',
              f"内部mAP最佳值出现在第{int(float(best['epoch']))}轮；125→150轮增加{gain:.3f}点，最终学习率仍为1e-4。训练后期继续增长，因此此结论对应150轮固定预算，不证明充分收敛。150轮健康记录中P2/QKV/RPB梯度均非零、框更新有限。", '',
              '![训练曲线和精度延迟比较](figures/summary.svg)', '',
              '历史C2采用同一数据、配方和seed，但不是本轮同期新训；实验仅有组合与历史基线，不能估计完整2×2交互项。固定seed未保证严格确定性，多seed结果缺失，0.490点AP差不能视作统计显著性结论。', '',
              '## 可读证据与复现', '',
              '- [训练曲线：新组合与历史C2](training_curves.csv)、[完整健康记录](raw/training_health.csv)、[训练状态和checkpoint路径/SHA](raw/training.json)。',
              '- [原生FP32正式COCO评估](raw/evaluation.json)、[TensorRT FP16正式COCO评估](raw/trt_fp16_coco.json)、[训练后同场CUDA原始样本](raw/benchmark.json)、[执行与失败原因](raw/validation.json)。',
              '- [启动时计划、冻结数据及复现命令](../2026-10-26_5090_c2_la_150e_launch/README.md)；[前期未训练结构预检](../2026-10-26_5090_p2_p5_la_preflight/README.md)。',
              '- 服务器保留权重、原始预测、完整日志和ONNX/TensorRT engine；路径、大小与SHA在验证记录或engine元数据中，不是云端运行依赖。', '',
              '从仓库文本离线生成和校验，不读取GPU、数据集或服务器权重：', '',
              '```bash','python3 -I docs/experiment_reports/2026-10-27_5090_c2_la_150e_results/verify.py',
              'python3 -m experiments.p2_p5_la_5090.archive_joint_results --output /tmp/5090-joint-final-replay','```','']
    lines += ['GPU复现TensorRT FP16正式AP（需要保留的两个engine、对应benchmark记录、权重及数据；使用新输出目录）：', '',
              '```bash',
              'PYTHONPATH="$PWD:$PWD/../detrscope/src" /tmp/zhj-uav-profiler-20261025-venv/bin/python -m experiments.p2_p5_la_5090.evaluate_trt_fp16_5090 \\',
              '  --benchmark runs/p2_p5_la_5090/5090-c2-la-trained-timing-20261027-r1/benchmark/assessment.json \\',
              '  --training runs/p2_p5_la_5090/5090-c2-la-150e-20261026-s0-r1/5090-c2-la-150e-s0/experiment.json \\',
              '  --native-evaluation runs/p2_p5_la_5090/5090-c2-la-150e-20261026-s0-r1/evaluation/summary.json \\',
              '  --annotations /绝对路径/VisDrone2019-DET_val_coco.json --images /绝对路径/images/val \\',
              '  --output runs/p2_p5_la_5090/5090-c2-la-trt-fp16-coco-new', '```', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='5090-c2-la-150e-20261027'
    curves=read_csv(output/'training_curves.csv')
    accuracy={r['variant']:r for r in read_csv(output/'accuracy.csv')}
    latency=next(r for r in read_csv(output/'latency.csv') if r['backend']=='trt_fp16')
    figure,axes=plt.subplots(1,2,figsize=(10.7,4),layout='constrained')
    for arm,label,color in (('c2','C2','#3565a4'),('c2-la','C2 + LA','#dd7138')):
        data=[r for r in curves if r['variant']==arm]
        axes[0].plot([float(r['epoch']) for r in data],[100*float(r['metrics/mAP50-95(B)']) for r in data],label=label,color=color)
        value=latency['c2_mean_ms'] if arm=='c2' else latency['c2_la_mean_ms']
        if value:
            axes[1].scatter(float(value),100*float(accuracy[arm]['AP']),color=color,label=label,s=45)
    axes[0].set(xlabel='Epoch',ylabel='Internal mAP50-95 (%)',title='Training curves, seed 0')
    axes[1].set(xlabel='TensorRT FP16 forward latency (ms)',ylabel='Native FP32 COCO AP (%)',title='Measured separately')
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    target=output/'figures/summary.svg'
    target.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(target,metadata={'Date':None})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(figure)


def verify_manifest(archive):
    for line in (archive/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    for name in ('README.md','REPORT.md'):
        for target in re.findall(r'\]\(([^)]+)\)',(archive/name).read_text()):
            if not target.startswith(('http:','https:')):
                assert (archive/target.split('#',1)[0]).exists(),target
    print('Verified tracked-only manifest and report links.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path)
    p.add_argument('--verify',action='store_true')
    p.add_argument('--run',type=Path)
    p.add_argument('--retry',type=Path)
    p.add_argument('--retry-log',type=Path)
    p.add_argument('--fp16-eval',type=Path)
    p.add_argument('--fp16-replay',type=Path)
    p.add_argument('--plots',action='store_true')
    args=p.parse_args()
    if args.run:
        if not all((args.retry,args.retry_log,args.output)):
            p.error('collect requires --retry, --retry-log and --output')
        collect(args)
        args.archive=args.output
    if args.fp16_eval:
        if not args.fp16_replay:
            p.error('--fp16-eval requires --fp16-replay')
        attach_fp16(args.archive,args.fp16_eval,args.fp16_replay)
    validate(args.archive)
    if args.verify:
        verify_manifest(args.archive)
    else:
        target=args.output or args.archive
        render(args.archive,target)
        if args.plots:
            plots(target)
