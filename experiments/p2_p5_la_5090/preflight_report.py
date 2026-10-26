"""Archive and assess the 5090 factorial preflight; offline verification uses stdlib.

Untrained architecture timings and a one-epoch engineering smoke test are never
reported as trained accuracy evidence. Large server artifacts remain in place.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('c0', 'c2', 'la', 'c2-la')
DEFAULT = ROOT / 'docs/experiment_reports/2026-10-26_5090_p2_p5_la_preflight'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def artifact(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reduction(before, after, mode, scope):
    def blocks(model):
        return {b['round']: statistics.fmean(b['timings_ms'][scope])
                for b in model['blocks'] if b['mode'] == mode}
    a, b = blocks(before), blocks(after)
    assert a.keys() == b.keys()
    ids = sorted(a)
    point = 100 * (1 - sum(b.values()) / sum(a.values()))
    rng = random.Random(20261026)
    bootstrap = []
    for _ in range(4000):
        sample = rng.choices(ids, k=len(ids))
        bootstrap.append(100 * (1 - sum(b[i] for i in sample) / sum(a[i] for i in sample)))
    bootstrap.sort()
    return {'latency_reduction_percent': point, 'ci95_low': bootstrap[100], 'ci95_high': bootstrap[3899],
            'faster_round_fraction': sum(b[i] < a[i] for i in ids) / len(ids), 'paired_rounds': len(ids)}


def collect(args):
    raw = args.output / 'raw'
    raw.mkdir(parents=True, exist_ok=False)
    benchmark = read(args.benchmark)
    smoke = read(args.smoke / 'experiment.json')
    health = [json.loads(line) for line in (args.smoke / 'health.jsonl').read_text().splitlines()]
    audit = read(args.audit)
    old_audit = read(ROOT / 'docs/experiment_reports/2026-10-24_5090_baselines/raw/asset_audit_5090.json')
    assert audit['status'] == 'passed'
    for key in ('label_content_sha256', 'image_inventory_sha256'):
        assert audit[key] == old_audit[key]
    assert audit['coco']['sha256'] == old_audit['coco']['sha256']
    test_log = args.tests.read_text()
    assert re.search(r'22 passed.*9 subtests passed', test_log)
    changed = {}
    for name, digest in benchmark['source_sha256'].items():
        current = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if current != digest:
            assert name in ('experiments/p2_p5_la_5090/evaluate.py', 'experiments/p2_p5_la_5090/test_experiment.py'), name
            changed[name] = {'benchmark_sha256': digest, 'current_sha256': current,
                             'reason': 'Manual predictor imgsz initialized before accepted reload evaluation; regression test added after benchmark. Neither file is executed by architecture timing.'}
    write(raw / 'benchmark.json', benchmark)
    write(raw / 'smoke.json', smoke)
    write(raw / 'smoke_health.json', health)
    write(raw / 'smoke_evaluation.json', read(args.smoke_evaluation))
    write(raw / 'asset_audit.json', audit)
    (raw / 'smoke_curve.csv').write_bytes((args.smoke / 'results.csv').read_bytes())
    previous = read(args.initial_benchmark)
    write(raw / 'validation.json', {
        'source_design_commit': 'ac73897a16d3eeb3ae0c46996985af962d7f6a8f',
        'branch': '5090-detrscope-query-budget', 'pull_result': 'git pull --ff-only: Already up to date',
        'tests': {'passed': 22, 'subtests_passed': 9, 'log': artifact(args.tests), 'summary': test_log.splitlines()[-1]},
        'post_benchmark_source_changes': changed,
        'failed_reload_evaluation': {'log': artifact(args.failed_evaluation_log),
                                    'error': 'TypeError: predictor.imgsz was None in native pre_transform; corrected before accepted re-run; no checkpoint changes'},
        'input_layout_fix': 'Real-image input now contiguous NCHW, matching the native evaluator and historical 5090 input hash.',
        'initial_layout_run': {'artifact': artifact(args.initial_benchmark),
                               'status': 'superseded for acceptance; retained on server',
                               'summary': {k: v['summary'] for k, v in previous['models'].items()}},
        'server_only': [artifact(p) for p in (args.benchmark, args.smoke / 'experiment.json',
                       args.smoke / 'health.jsonl', args.smoke / 'results.csv', args.smoke_evaluation)]
                       + list(smoke['checkpoints'].values()),
        'full_matrix_status': 'not_started: fails prespecified native FP32 speed gate; no claim about AP or TensorRT speed'})


def validate(archive):
    raw = archive / 'raw'
    b, s, e = [read(raw / name) for name in ('benchmark.json', 'smoke.json', 'smoke_evaluation.json')]
    assert b['status'] == 'completed' and b['role'] == 'untrained_architecture_screen_no_AP'
    assert b['input']['contiguous'] and b['input']['stride'] == [1228800, 409600, 640, 1]
    assert b['input']['tensor_sha256'] == '883789210db2b7f51d7a34c75b61b504637c17d8200c616eef69478cb0ec7bed'
    assert b['protocol']['rounds'] == 10 and b['protocol']['samples_per_round'] == 100
    assert len(b['round_order']) == 10 and set(b['models']) == set(ARMS)
    corrections = read(raw / 'validation.json')['post_benchmark_source_changes']
    assert set(corrections) == {'experiments/p2_p5_la_5090/evaluate.py', 'experiments/p2_p5_la_5090/test_experiment.py'}
    for name, digest in b['source_sha256'].items():
        expected = digest
        if name in corrections:
            assert corrections[name]['benchmark_sha256'] == digest
            expected = corrections[name]['current_sha256']
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for model in b['models'].values():
        assert model['model_state_unchanged'] and model['hooks_remaining'] == 0 and model['output_exact_with_hooks']
        assert model['architecture']['queries'] == 300 and model['architecture']['decoder_layers'] == 3
        assert model['architecture']['points_per_level'] == 4
        assert len(model['blocks']) == 20
        for mode, scopes in (('bare', ('whole',)), ('staged', ('whole', 'backbone', 'p5', 'neck_encoder', 'head_decoder'))):
            for scope in scopes:
                values = [v for block in model['blocks'] if block['mode'] == mode for v in block['timings_ms'][scope]]
                assert len(values) == 1000 and all(v > 0 for v in values)
                assert abs(statistics.fmean(values) - model['summary'][mode][scope]['mean_ms']) < 1e-12
    initials = {arm: b['models'][arm]['identity']['initialization'] for arm in ARMS}
    assert len({v['shared_state_sha256'] for v in initials.values()}) == 1
    for key, pairs in (('p2_slot_sha256', (('c0', 'la'), ('c2', 'c2-la'))),
                       ('p5_slot_sha256', (('c0', 'c2'), ('la', 'c2-la')))):
        assert all(initials[a][key] == initials[c][key] for a, c in pairs)
    assert s['status'] == 'completed' and s['variant'] == 'c2-la' and s['completed_epochs'] == 1
    assert s['role'] == 'engineering_smoke' and s['fraction'] == .03 and s['effective_args']['imgsz'] == 640
    assert s['initialization']['shared_state_sha256'] == initials['c2-la']['shared_state_sha256']
    h = read(raw / 'smoke_health.json')
    assert len(h) == 1
    for key in ('p2_first_weight_grad_abs', 'p5_qkv_grad_abs', 'p5_rpb_grad_abs'):
        assert h[0][key] > 0
    assert all(v == 1 for k, v in h[0].items() if k.endswith('_finite'))
    assert e['status'] == 'completed' and e['scope'] == 'subset_gate' and len(e['protocol']['image_ids']) == 8
    assert e['checkpoint'] == s['checkpoints']['best'] and e['model_state_unchanged']
    assert read(raw / 'validation.json')['tests']['passed'] == 22
    print('Verified paired initialization, 24000 CUDA intervals, input/source identity, real 5090 training and 8-image reload evaluation.')


def render(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    b = read(archive / 'raw/benchmark.json')
    rows = []
    for arm in ARMS:
        m = b['models'][arm]
        rows.append({'variant': arm, 'parameters': m['parameters_unfused'],
                     'whole_ms': m['summary']['bare']['whole']['mean_ms'],
                     'p5_ms': m['summary']['staged']['p5']['mean_ms'],
                     'neck_encoder_ms': m['summary']['staged']['neck_encoder']['mean_ms'],
                     'instrumentation_overhead_percent': m['instrumentation_overhead_percent']})
    comparisons = []
    for before, after in (('c0', 'c2'), ('c0', 'la'), ('c2', 'c2-la'), ('la', 'c2-la'), ('c0', 'c2-la')):
        for mode, scope in (('bare', 'whole'), ('staged', 'neck_encoder'), ('staged', 'p5')):
            comparisons.append({'before': before, 'after': after, 'scope': scope,
                                **reduction(b['models'][before], b['models'][after], mode, scope)})
    main = {r['scope']: r for r in comparisons if r['before'] == 'c2' and r['after'] == 'c2-la'}
    passed = (main['whole']['latency_reduction_percent'] >= 1 and main['neck_encoder']['latency_reduction_percent'] >= 3
              and all(r['faster_round_fraction'] >= .8 and r['ci95_low'] > 0 for r in (main['whole'], main['neck_encoder'])))
    csv_write(output / 'comparison.csv', rows)
    csv_write(output / 'paired_comparisons.csv', comparisons)
    write(output / 'assessment.json', {'native_speed_gate_passed': passed, 'requirements': {
        'whole_reduction_percent_min': 1, 'neck_encoder_reduction_percent_min': 3, 'faster_round_fraction_min': .8},
        'observed': main, 'next_action': 'full_matrix_eligible' if passed else 'defer_full_matrix; diagnose implementation first',
        'accuracy_gate': 'not_evaluated: architecture timing and one-epoch smoke cannot establish AP',
        'deployment': 'TensorRT/Orin not evaluated; native FP32 failure does not establish deployment failure'})
    s = read(archive / 'raw/smoke.json')
    h = read(archive / 'raw/smoke_health.json')[0]
    lines = ['# 5090 P2＋P5局部注意力：设计审查与GPU预检', '',
             '分支 `5090-detrscope-query-budget`；远程方案提交 `ac73897a`，已执行快进拉取并确认最新。', '',
             '**结论：四组正交设计合理，当前NAIFI实现未通过自身预定的5090原生FP32速度门槛，本轮暂不启动四组150轮。** 已完成测试、两轮结构测速及组合真实GPU短跑/重载评估，未把短跑AP当成精度证据。', '',
             '## 设计审查', '',
             '四组为C0、C2、LA、C2＋LA；固定300Q/每尺度4点/3层Decoder，分别控制P2注入与P5替换，能测主效应和交互项。公共参数及两个分支槽分别配对随机初始化，实际Trainer沿用相同初始化器；不存在读取旧checkpoint后伪装从零训练。', '',
             'NAIFI同时替换注意力、二维FFN和归一化，因此该设计测试整个既有模块，不能单独归因于局部窗口。150轮/seed0只是筛选预算；方案已要求同预算、多seed及收敛复验。正式精度使用548图原生FP32 COCO协议，和内部mAP、VisDrone挑战赛指标分开。', '',
             '本机修复后22项测试、9项子测试通过。修复两个执行问题：真实图像测速输入改为与原生评估相同的连续NCHW，保存stride和输入SHA；手动构造的评估Predictor补齐imgsz，回归测试覆盖原生预处理真实图片尺寸。未修复时重载评估会报错；失败日志/初次测速保留服务器路径、SHA和摘要。以下只使用修正后接受的结果。', '',
             '## 修正后10轮交错测速', '',
             'RTX5090 D v2，batch1、640、未融合FP32，TF32关闭，GPU常驻同一真实验证图。四模型各150次预热，10轮随机交错，每块10次预热＋100次采样。整网无标记测量；P5和Neck/Encoder来自独立带标记测量，P5包含在Neck内。CUDA事件跨度含CPU提交间隙，不是纯kernel忙碌时间。', '',
             '| 结构 | 整网/ms | P5/ms | Neck/Encoder/ms |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['variant']} | {row['whole_ms']:.4f} | {row['p5_ms']:.4f} | {row['neck_encoder_ms']:.4f} |")
    lines += ['', '组合相对C2（负下降表示变慢）：', '',
              '| 范围 | 延迟下降 | 配对95%区间 | 更快轮次 | 预定下降门槛 |', '|---|---:|---:|---:|---:|']
    for scope in ('whole', 'neck_encoder', 'p5'):
        r = main[scope]
        target = '1%' if scope == 'whole' else '3%' if scope == 'neck_encoder' else '单独报告'
        lines.append(f"| {scope} | {r['latency_reduction_percent']:+.2f}% | [{r['ci95_low']:.2f}%, {r['ci95_high']:.2f}%] | {r['faster_round_fraction']*100:.0f}% | {target} |")
    lines += ['', '4000次确定性bootstrap按10个配对轮次抽样，不把轮内100次当作独立实验。区间只描述本次计时噪声，不覆盖输入、硬件、编译后端或重新训练差异。所有输出在添加标记前后逐位一致，模型状态未变、hooks清除、分段总和核对通过。完整CUDA样本和source/environment身份随仓库提供。', '',
              '速度负面结果定位到P5模块增加耗时；其他主干与Decoder基本一致。这与`F.unfold`邻域展开、访存及多算子开销可能有关，但此轮没有kernel profiler证据，不能把具体瓶颈归因当作已证实。AIFI在20×20 P5上的全局注意力也未必比该局部实现慢。没有TensorRT/Orin数据，不外推其部署速度。', '',
              '## 真实数据GPU短跑', '',
              f"组合从零跑1轮、fraction=.03、batch8、640、FP32，训练完成，训练计时{s.get('train_seconds_total', sum(x.get('train_seconds', 0) for x in read(archive / 'raw/smoke_health.json'))):.2f}秒。训练内验证使用完整548图，另对自身best.pt完成8图原生FP32重载COCO链路检查。", '',
              f"P2/QKV/RPB采样梯度均非零：{h['p2_first_weight_grad_abs']:.6g} / {h['p5_qkv_grad_abs']:.6g} / {h['p5_rpb_grad_abs']:.6g}；框更新有限，best/last/resume保存与optimizer/EMA恢复资产检查通过。短跑只证明工程链路，不判断最终精度。", '',
              '数据审计确认train6471图/343200框、val548图/38759框，十类顺序、标签内容SHA、图片路径/大小清单SHA及COCO标注SHA与既有5090基线一致。图片清单不是逐图片内容哈希，保留这个边界。', '',
              '## 本轮决策', '',
              '原方案要求先速度筛选：相对C2整网至少下降1%、Encoder至少下降3%、至少80%配对轮同向。当前条件未通过，因此不消耗约一天GPU运行四组长周期训练。这个结论只否定“当前原生FP32实现已经满足预定加速线”，不否定局部注意力的精度潜力或优化后部署潜力。', '',
              '继续此方向时，先诊断并优化NAIFI执行路径，或单独定义TensorRT路径的验收；经过新的速度预检后再按原四组同预算从零训练。若目标改为仅研究精度，也应明确接受当前速度回退，不能沿用“加速组合”的表述。', '',
              '## 可读证据与复现', '',
              '- [四组绝对耗时](comparison.csv)、[全部配对差值](paired_comparisons.csv)、[机器可读门槛判断](assessment.json)。',
              '- [原始CUDA计时与环境/源码](raw/benchmark.json)、[真实GPU训练状态](raw/smoke.json)、[训练健康](raw/smoke_health.json)、[短跑曲线](raw/smoke_curve.csv)、[8图重载检查](raw/smoke_evaluation.json)。',
              '- [数据审计](raw/asset_audit.json)、[测试摘要及初次布局运行、服务器产物路径/SHA](raw/validation.json)。权重、预测及完整日志仍仅在服务器。',
              '- [原始实验设计](../../experiments/p2_p5_la_5090/README.md)（仓库路径见本归档README），不使用历史4090或旧C2精度冒充本次结果。', '',
              '从Git文本离线重建/校验，无需GPU、torch、SDK或服务器资产：', '',
              '```bash', 'python3 -m experiments.p2_p5_la_5090.preflight_report --verify',
              'python3 -m experiments.p2_p5_la_5090.preflight_report --output /tmp/5090-la-preflight-replay', '```', '']
    # Repository-relative link from the dated report directory.
    lines = [line.replace('../../experiments/p2_p5_la_5090/README.md', '../../../experiments/p2_p5_la_5090/README.md') for line in lines]
    (output / 'REPORT.md').write_text('\n'.join(lines))


def verify_manifest(archive):
    for line in (archive / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    for name in ('README.md', 'REPORT.md'):
        for target in re.findall(r'\]\(([^)]+)\)', (archive / name).read_text()):
            if not target.startswith(('http:', 'https:')):
                assert (archive / target.split('#', 1)[0]).exists(), target
    print('Verified tracked-only manifest and links.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--initial-benchmark', type=Path)
    parser.add_argument('--smoke', type=Path)
    parser.add_argument('--smoke-evaluation', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--tests', type=Path)
    parser.add_argument('--failed-evaluation-log', type=Path)
    args = parser.parse_args()
    if args.benchmark:
        assert all((args.initial_benchmark, args.smoke, args.smoke_evaluation, args.audit, args.tests, args.failed_evaluation_log, args.output))
        collect(args)
        args.archive = args.output
    validate(args.archive)
    if args.verify:
        verify_manifest(args.archive)
    else:
        render(args.archive, args.output or args.archive)
