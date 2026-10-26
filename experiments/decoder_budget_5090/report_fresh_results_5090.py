"""Offline final fresh-budget analysis and verification; stdlib except optional plots."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from experiments.decoder_budget_5090.report_decoder_screen_5090 import reduction

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'docs/experiment_reports/2026-10-26_5090_fresh_decoder_results'
BASE = 'c2-q300-p4-d3'
ORDER = [BASE, 'c2-q300-p2-d3', 'c2-q200-p2-d3', 'c2-q300-p4-d2', 'c2-q300-p4-d1']


def read(path):
    return json.loads(path.read_text())


def csv_read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inputs(archive):
    live = read(ROOT / 'reports/5090/fresh_budgets/experiments.json')
    old = read(ROOT / 'reports/5090/experiments.json')
    depth = read(ROOT / 'docs/experiment_reports/2026-10-25_5090_decoder_screen/raw/depth-eval-c2.json')
    baseline = next(r for r in old['records'] if r.get('checkpoint_sha256', {}).get('best') == depth['checkpoint']['sha256'])
    records = {r['state']['variant']: r for r in live['records'] if r['state']['role'] == 'fresh_150e'}
    paired = read(archive / 'raw/paired_benchmark.json')
    return live, baseline, depth, records, paired


def validate(archive):
    live, baseline, depth, records, paired = inputs(archive)
    assert live['queue']['status'] == 'completed' and len(records) == 4
    assert all(p['returncode'] == 0 for p in live['queue']['phases'])
    startup = read(ROOT / 'docs/experiment_reports/2026-10-25_5090_fresh_decoder_training/raw/initialization_plan.json')
    sources = [r['state']['source_sha256'] for r in records.values()]
    assert all(s == sources[0] for s in sources)
    for name, digest in sources[0].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert baseline['epochs'] == 150 and len(baseline['curve']) == 150
    for key, r in records.items():
        s, e = r['state'], r['evaluation']
        assert s['status'] == e['status'] == 'completed' and s['epochs_completed'] == 150
        assert len(r['curve']) == 150 and [int(x['epoch']) for x in r['curve']] == list(range(1, 151))
        assert not s['pretrained'] and not s['resume'] and s['full_initial_state_verified']
        assert s['initialization']['source_checkpoint'] is None
        assert s['common_state_sha256'] == startup['common_state_sha256']
        assert s['initialization']['full_initial_state_sha256'] == startup['variants'][key]['initialization']['full_initial_state_sha256']
        assert s['checkpoints']['best'] == e['checkpoint']
        assert e['scope'] == 'full_val' and e['protocol']['images'] == 548 and e['gt_denominator'] == 38759
        assert e['protocol']['annotation_sha256'] == depth['formal_protocol']['annotation_sha256']
        assert e['protocol']['formal_max_dets'] == [1, 10, 100] and e['protocol']['dtype'] == 'float32'
        assert e['protocol']['fused'] is False and e['model_state_unchanged'] and e['repeated_forward_exact']
        for row in r['curve']:
            assert all(math.isfinite(float(v)) for v in row.values())
    health = csv_read(archive / 'raw/training_health.csv')
    for variant, r in records.items():
        rows = [v for v in health if v['variant'] == variant]
        assert len(rows) == 150
        for row in rows:
            assert float(row['p2_first_weight_grad_abs']) > 0
            for i in range(r['evaluation']['architecture']['decoder_layers']):
                assert float(row[f'd{i+1}_offset_weight_grad_abs']) > 0
                assert float(row[f'd{i+1}_delta_finite']) == 1
    assert paired['status'] == 'completed' and len(paired['conditions']) == 15
    assert len(paired['round_order']) == paired['protocol']['rounds'] == 20
    assert all(set(order) == set(paired['conditions']) for order in paired['round_order'])
    for key, condition in paired['conditions'].items():
        variant = condition['variant']
        checkpoint = depth['checkpoint'] if variant == BASE else records[variant]['evaluation']['checkpoint']
        assert condition['checkpoint'] == checkpoint
        assert len(condition['blocks']) == 20 and {b['round'] for b in condition['blocks']} == set(range(1, 21))
        samples = [x for b in condition['blocks'] for x in b['cuda_ms']]
        assert len(samples) == 1000 and all(math.isfinite(x) and x > 0 for x in samples)
        assert math.isclose(statistics.fmean(samples), condition['cuda_stats_ms']['mean'], abs_tol=1e-12)
        assert math.isclose(statistics.median(samples), condition['cuda_stats_ms']['median'], abs_tol=1e-12)
        assert sorted(samples)[950] == condition['cuda_stats_ms']['p95']
        if variant != BASE and condition['engine']:
            assert condition['engine'] == records[variant]['evaluation']['benchmark']['backends'][condition['backend']]['engine']
        if variant == BASE and condition['engine']:
            archived = read(ROOT / 'docs/experiment_reports/2026-10-25_5090_decoder_screen/raw/depth-c2-model.json')
            assert condition['engine'] == archived['groups']['d3/model']['backends'][condition['backend']]['engine']
    for name, digest in paired['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'experiments/decoder_budget_5090' / Path(name).name).read_bytes()).hexdigest() == digest
    category = csv_read(archive / 'raw/category_ap.csv')
    assert len(category) == 50
    for variant in ORDER:
        rows = [r for r in category if r['variant'] == variant]
        metric = depth['conditions']['d3']['metrics'] if variant == BASE else records[variant]['evaluation']['metrics']
        assert len(rows) == 10 and math.isclose(statistics.fmean(float(r['AP']) for r in rows), metric['AP'], abs_tol=1e-12)
    transitions = csv_read(archive / 'raw/gt_transitions.csv')
    for variant in ORDER[1:]:
        for row in (r for r in transitions if r['variant'] == variant and r['group'] == 'all'):
            assert int(row['gt_count']) == 38759
            assert sum(int(row[k]) for k in ('stable_tp', 'stable_fn', 'lost', 'gained')) == 38759
            assert int(row['gained']) - int(row['lost']) == int(row['net_tp'])
    print('Verified 4x150 epochs, fresh initialization/source identity, 548-image FP32 AP, 600 health rows, 15000 paired CUDA samples, category AP and GT totals.')


def render(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    live, baseline, depth, records, paired = inputs(archive)
    conditions = paired['conditions']
    base_metrics = depth['conditions']['d3']['metrics']
    rows, curves = [], []
    for variant in ORDER:
        if variant == BASE:
            metric = base_metrics
            curve = baseline['curve']
            parameters = 20135032
            state = None
        else:
            r = records[variant]
            metric, curve, state = r['evaluation']['metrics'], r['curve'], r['state']
            parameters = r['evaluation']['architecture']['parameters']
        best = max(curve, key=lambda x: float(x['metrics/mAP50-95(B)']))
        row = {'variant': variant, 'parameters': parameters, **metric,
               'delta_AP_pp': 100 * (metric['AP'] - base_metrics['AP']),
               'delta_APs_pp': 100 * (metric['APs'] - base_metrics['APs']),
               'best_internal_map': float(best['metrics/mAP50-95(B)']), 'best_epoch': int(float(best['epoch'])),
               'internal_gain_epoch125_to150_pp': 100 * (float(curve[149]['metrics/mAP50-95(B)']) - float(curve[124]['metrics/mAP50-95(B)'])),
               'train_seconds': None if state is None else state['train_seconds']}
        for backend in ('eager_fp32', 'trt_fp32', 'trt_fp16'):
            b = conditions[variant + '/' + backend]
            row[backend + '_ms'] = b['cuda_stats_ms']['mean']
            row[backend + '_p95_ms'] = b['cuda_stats_ms']['p95']
            row[backend + '_forward_fps'] = 1000 / b['cuda_stats_ms']['mean']
            row.update({backend + '_' + k: v for k, v in reduction(conditions[BASE + '/' + backend], b).items()})
        rows.append(row)
        curves.extend({'variant': variant, **r} for r in curve)
    csv_write(output / 'comparison.csv', rows)
    # Source curves already live in tracked JSONs; this projection eases direct comparison.
    csv_write(output / 'training_curves.csv', curves)
    health = csv_read(archive / 'raw/training_health.csv')
    category = csv_read(archive / 'raw/category_ap.csv')
    transitions = csv_read(archive / 'raw/gt_transitions.csv')
    lines = ['# C2 四组从零训练：150轮结果分析', '',
             '分支 `5090-detrscope-query-budget`；队列 `5090-c2-fresh-four-20261025-s0-r1`，2026-10-26 14:20:17（北京时间）全部完成。四组各150轮，训练与自动评估阶段全部返回0。', '',
             '**同预算下没有出现兼顾明显加速与基本无损精度的方案。** 300Q减采样点最接近原精度，但TRT FP16只省约0.013 ms；减Query的额外收益很小；从零训练D2/D1的AP损失明显大于直接截断三层已训练模型。', '',
             '## 统一对照', '',
             '精度为548张验证图上的原生未融合FP32正式COCO bbox AP，表内乘100；AP差为百分点。延迟为五模型同场20轮随机交错、每轮50次的GPU常驻batch1、640纯forward，FP16 AP尚未评估。AP与FP16速度是两项独立测量，不能视为已验证的FP16精度。', '',
             '| 模型 | 正式AP | AP差/点 | APs | TRT FP16/ms | 延迟下降 | 配对95%区间 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['variant']} | {100*r['AP']:.3f} | {r['delta_AP_pp']:+.3f} | {100*r['APs']:.3f} | {r['trt_fp16_ms']:.4f} | {r['trt_fp16_latency_reduction_percent']:.2f}% | [{r['trt_fp16_ci95_low']:.2f}%, {r['trt_fp16_ci95_high']:.2f}%] |")
    lines += ['', '完整12项COCO指标、三执行路径均值/P95/FPS/配对区间和参数量见 [comparison.csv](comparison.csv)。FPS仅为1000/ms的纯forward倒数。', '',
              '## 如何理解四个结果', '',
              '1. **300Q、P4→P2：精度代价最小，速度收益也小。** AP由19.517降到19.263，损失0.254点；APs由12.653升到12.739（+0.087点），APm基本持平，但APl由34.665降到29.954（−4.711点）。说明总体AP变化较小并不等于所有尺度无损；单seed不足以确认小目标收益。适合作为低优先级候选，当前收益不足以支撑多组长周期训练。', '',
              '2. **200Q＋P2：当前不划算。** 相对原模型AP损失2.984点；相对300Q/P2又损失2.730点，而FP16延迟额外仅下降0.85%（约0.0072 ms）。APs也从12.739降到11.109。这组差异在P2固定时隔离了Query预算变化；没有新训练200Q/P4，因此不能在新训练条件下独立量化“只减Query”的效应。', '',
              '3. **D2：有真实加速，但从零训练没有保住精度。** FP16约快10.26%，AP损失2.135点（约10.94%相对下降），APs损失1.259点。若以近乎无损为目标，该配方未通过；若允许精度换约10%forward延迟，可保留作候选。后续优先研究训练配方/监督恢复，而不能直接宣称D2是已恢复精度的方案。', '',
              '4. **D1：是更激进的取舍。** FP16延迟下降21.34%，AP损失4.570点，APs损失3.239点；当前更适合严格时延预算下的候选，不宜作为默认替换。', '',
              '## 为什么不能沿用截断模型时的判断', '',
              '| 层数 | 旧D3 checkpoint提前退出AP | 对应结构从零训练AP | 从零训练−提前退出/点 |',
              '|---|---:|---:|---:|']
    for d in (2, 1):
        old_ap = depth['conditions'][f'd{d}']['metrics']['AP']
        new_ap = next(r['AP'] for r in rows if r['variant'] == f'c2-q300-p4-d{d}')
        lines.append(f'| D{d} | {100*old_ap:.3f} | {100*new_ap:.3f} | {100*(new_ap-old_ap):+.3f} |')
    lines += ['', '提前退出的D1/D2来自按三层Decoder训练的同一个网络；重训模型只有1/2层，训练时也减少了相应Decoder辅助损失和迭代更新。二者不是同一学习过程。源码 `ultralytics/models/utils/loss.py` 对各辅助层损失求和，物理减少层数会改变监督结构；这能解释实验并非等价，但尚不能证明它是全部精度差异的原因。未做蒸馏、权重迁移或损失重标定对照。', '',
              '之前采样筛选使用随机兼容模块，得到2.79%/4.96%下降；本次采用各自真正训练后的权重及对应已构建引擎，统一复测为1.47%/2.31%。参数值、引擎tactic与测量场景均可能影响差异，不能把旧筛选数字当成本次收益。', '',
              '## 训练预算与收敛边界', '',
              '所有模型150轮，seed0、FP32/AMP关闭、batch8、640、AdamW、lr=1e-4、lrf=1、warmup3、weight_decay=1e-4、nbs64、mixup=.2、workers0，无预训练、无旧checkpoint初始化或续训。新四组共用随机参数SHA一致，实际trainer初态逐字节核验通过。', '',
              '| 模型 | 最佳内部mAP/% | 最佳轮 | 125→150内部mAP增量/点 | 训练计算/h | 训练峰值/MiB |',
              '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        h = [float(v['train_peak_allocated_mib']) for v in health if v['variant'] == r['variant']]
        hours = '历史对照' if r['train_seconds'] is None else f"{r['train_seconds']/3600:.2f}"
        memory = '—' if not h else f'{max(h):.0f}'
        lines.append(f"| {r['variant']} | {100*r['best_internal_map']:.3f} | {r['best_epoch']} | {r['internal_gain_epoch125_to150_pp']:+.3f} | {hours} | {memory} |")
    lines += ['', '训练计算时间是health累计的训练部分，不含全部验证/导出；峰值为PyTorch allocated，非nvidia-smi总显存。内部mAP用于选best，和正式COCO AP口径不同。600轮health记录中P2和各实际Decoder层offset梯度均非零，采样框更新有限，没有NaN/Inf或静默漏训证据。', '',
              '**150轮只是相同预算，尚未证明充分收敛。** 四组最后25轮内部mAP仍增长0.94–1.58点，最佳轮集中149/150；学习率最终仍为1e-4。原基线也有后期增长。这意味着不能把当前差距全部归结为不可恢复的结构容量损失，也不能凭上升曲线承诺延长训练一定恢复。仅单seed且未开启严格确定性算法，精度差异没有多seed置信区间。', '',
              '![内部训练曲线与AP延迟取舍](comparison.svg)', '',
              '## 类别与同GT证据', '',
              '保存的五组原生FP32预测重新运行COCOeval，12项整体指标与既有结果均在1e-12内一致；以下类别AP由同一次precision数组提取（IoU=.50:.05:.95、area=all、maxDets=100）。', '',
              '| 类别 | 原C2 | 300Q/P2 | 200Q/P2 | D2 | D1 |', '|---|---:|---:|---:|---:|---:|']
    for c in sorted({int(r['category_id']) for r in category}):
        matching = {r['variant']: r for r in category if int(r['category_id']) == c}
        lines.append('| ' + matching[BASE]['category'] + ' | ' + ' | '.join(f"{100*float(matching[v]['AP']):.3f}" for v in ORDER) + ' |')
    lines += ['', '按原标注ID比较同一GT的正式COCO匹配，IoU=.5、maxDets=100：', '',
              '| 模型 | 新找回GT | 丢失GT | 净TP变化 | 持续TP | 持续FN |', '|---|---:|---:|---:|---:|---:|']
    for v in ORDER[1:]:
        r = next(r for r in transitions if r['variant'] == v and r['group'] == 'all' and float(r['iou']) == .5)
        lines.append(f"| {v} | {r['gained']} | {r['lost']} | {int(r['net_tp']):+d} | {r['stable_tp']} | {r['stable_fn']} |")
    lines += ['', '300Q/P2在pedestrian、people、bicycle类别分别增加0.462/0.394/0.794个AP点，truck下降1.710点；总AP接近原值掩盖了类别差异。200Q/P2、D2、D1的10个类别AP均下降，仍需多seed判断可重复性。300Q/P2在IoU=.5净找回118个GT而AP略降，说明单一阈值召回和整体AP并不等价。', '',
              '同GT净变化可定位召回得失，不能替代基于置信度排序的AP。完整10个IoU阈值及类别/互斥尺度分组见 [GT表](raw/gt_transitions.csv)，类别/尺度AP见 [类别表](raw/category_ap.csv)。GT尺度分组来自原图面积并使用all-area匹配，不等同COCO各area重新匹配的APs/APm/APl。', '',
              '## 测速与可复现边界', '',
              'RTX5090 D v2、PyTorch2.14.0+cu130、TensorRT10.14.1.48.post1，关闭TF32，固定同一真实验证图输入。五模型各自best.pt及已有ONNX/engine的SHA校验通过，复用原引擎、不重建tactic。三后端共15条件，20轮随机交错、每轮50次，150次预热。保存15000次CUDA样本及wall分组摘要。95%区间按20个配对轮次bootstrap4000次，只反映本次计时噪声，不覆盖多图片、引擎重新构建、多次独立训练或端到端流水线。', '',
              'Eager FP32下减点/减Query没有可辨加速；D2/D1则分别下降约6.18%/12.32%。这与前期算子诊断一致：减少局部Decoder采样预算无法消除Backbone/Neck卷积和提交开销。TRT FP16的约10%/21%节省只适用于纯forward；没有把它换算成端到端检测加速。', '',
              '历史原C2来自独立的同配方150轮seed0训练，本次按用户要求只重训四个候选，没有同期重训第五个P4/D3基线；其checkpoint固定后重新同场测速。四组训练记录的Git commit因队列期间归档提交而不同，但冻结源码哈希映射完全相同；[启动冻结清单](../2026-10-25_5090_fresh_decoder_training/raw/frozen_sources.json)和最终记录共同证明代码内容一致。', '',
              '## 证据与复现', '',
              '- [最终队列、完整训练曲线与自动正式评估原始计时](../../../reports/5090/fresh_budgets/experiments.json)：唯一完整新训练快照；避免在归档重复拷贝。',
              '- [五模型汇总](comparison.csv)、[完整必要曲线投影](training_curves.csv)、[600轮健康记录](raw/training_health.csv)、[同场CUDA原始样本](raw/paired_benchmark.json)。',
              '- [服务器文件路径、大小、SHA256](raw/server_artifacts.json)：权重/原始日志/预测/COCO大匹配表/ONNX/engine仍仅服务器保存，不是云端相对依赖。',
              '- [前期逐层FP32评估](../2026-10-25_5090_decoder_screen/raw/depth-eval-c2.json)、[原基线完整训练记录](../../../reports/5090/experiments.json)、[新训练启动/复现命令](../2026-10-25_5090_fresh_decoder_training/README.md)。', '',
              '训练集6471图/343200框，验证548图/38759框。标注SHA `6f8b8686864f1485ab172de9339f7049593f74766f6654c6e980a6c00d810371`，数据根路径见启动报告。最终best.pt路径与SHA同时记录在最终快照、同场计时和服务器清单内。', '',
              '离线从仓库文本重建汇总并验证（不访问服务器资产）：', '',
              '```bash', 'python3 -m experiments.decoder_budget_5090.report_fresh_results_5090 --verify',
              'python3 -m experiments.decoder_budget_5090.report_fresh_results_5090 --output /tmp/5090-final-replay',
              '# 可选：安装matplotlib的环境可加 --plots 重建SVG。', '```', '',
              '新同场测速复现（需外部GPU/权重/已构建engine，必须用新output目录）：', '',
              '```bash', 'PYTHONPATH="$PWD:$PWD/../detrscope/src" /tmp/zhj-uav-profiler-20261025-venv/bin/python -m experiments.decoder_budget_5090.benchmark_fresh_comparison_5090 \\',
              '  --root runs/decoder_retrain_5090/5090-c2-fresh-four-20261025-s0-r1 \\',
              '  --baseline runs/diagnostics/5090-depth-perf-c2-20261025-final/assessment.json \\',
              '  --output runs/diagnostics/5090-fresh-paired-new', '```', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    return rows, curves


def plots(output, rows, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = '5090-fresh-results-20261026'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
    for variant in ORDER:
        curve = [r for r in curves if r['variant'] == variant]
        label = variant.removeprefix('c2-').upper()
        line, = axes[0].plot([float(r['epoch']) for r in curve], [100*float(r['metrics/mAP50-95(B)']) for r in curve], label=label)
        row = next(r for r in rows if r['variant'] == variant)
        axes[1].scatter(row['trt_fp16_ms'], 100*row['AP'], color=line.get_color())
        axes[1].annotate(label, (row['trt_fp16_ms'], 100*row['AP']), xytext=(-7, -13 if variant == BASE else 7), textcoords='offset points', ha='right', fontsize=8)
    axes[0].set(xlabel='Epoch', ylabel='Internal mAP50-95 (%)', title='Same 150-epoch budget; seed 0')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='TensorRT FP16 forward latency (ms)', ylabel='Native FP32 formal COCO AP (%)', title='AP and latency measured separately', xlim=(.65, .9))
    for axis in axes:
        axis.grid(alpha=.25)
    fig.savefig(output / 'comparison.svg', metadata={'Date': None})
    svg = output / 'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def verify_manifest(archive):
    for line in (archive / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    for document in ('README.md', 'REPORT.md'):
        for target in re.findall(r'\]\(([^)]+)\)', (archive / document).read_text()):
            if not target.startswith(('http:', 'https:')):
                assert (archive / target.split('#', 1)[0]).exists(), target
    print('Verified tracked-only manifest and report links.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    validate(args.archive)
    if args.verify:
        verify_manifest(args.archive)
    else:
        target = args.output or args.archive
        rows, curves = render(args.archive, target)
        if args.plots:
            plots(target, rows, curves)
