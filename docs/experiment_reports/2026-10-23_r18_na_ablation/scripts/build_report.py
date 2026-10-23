"""Validate benchmark evidence and build the consolidated Chinese experiment report."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

REPORT = Path(__file__).resolve().parents[1]
VARIANTS = ['r0', 'r1', 'r2a', 'r2b', 'r2c']
TZ = ZoneInfo('Asia/Shanghai')


def load(relative):
    return json.loads((REPORT / relative).read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def write_csv(name, rows):
    with (REPORT / 'tables' / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    benchmark = load('raw/inference_20261023/benchmark.json')
    validation = load('provenance/validation.json')
    scope = load('provenance/scope.json')
    queue = load('raw/queue_status.json')
    records = {(r['variant'], r['precision']): r for r in benchmark['records']}
    assert set(records) == {(v, p) for v in VARIANTS for p in ['fp32', 'fp16']}
    assert queue['status'] == 'completed'
    assert queue['completed_training'] == ['r2a', 'r2b', 'r2c']
    assert queue['completed_evaluation'] == ['r2a', 'r2b', 'r2c']
    metrics = {v: load(f'raw/accuracy/{v}_formal_metrics.json') for v in VARIANTS}
    for v in VARIANTS:
        assert metrics[v]['protocol'] == metrics['r0']['protocol']
        for precision in ['fp32', 'fp16']:
            r = records[v, precision]
            assert r['checkpoint_sha256'] == validation['variants'][v]['checkpoint_sha256']
            assert r['finite_output_check_passed']
            assert r['runtime_parameter_dtype'] == ('torch.float32' if precision == 'fp32' else 'torch.float16')
            for name, samples in [('forward_wall', 450), ('forward_gpu', 450), ('pipeline_wall', 192)]:
                timings = r['raw_timings_ms'][name]
                assert len(timings) == samples and r[name]['samples'] == samples
                assert all(math.isfinite(x) and x > 0 for x in timings)
                assert abs(statistics.mean(timings) - r[name]['mean_ms']) < 1e-8
                assert abs(statistics.median(timings) - r[name]['median_ms']) < 1e-8
                assert abs(1000 / statistics.mean(timings) - r[name]['fps']) < 1e-8
    recipe_keys = ['epochs', 'batch', 'imgsz', 'seed', 'amp', 'workers', 'optimizer', 'lr0', 'lrf',
                   'weight_decay', 'momentum', 'nbs', 'warmup_epochs', 'warmup_bias_lr', 'mixup',
                   'mosaic', 'close_mosaic', 'patience', 'deterministic', 'data']
    recipes = {v: yaml.safe_load((REPORT / f'raw/training/{v}_args.yaml').read_text()) for v in VARIANTS}
    recipe_rows = [{'field': key, **{v: recipes[v].get(key) for v in VARIANTS}} for key in recipe_keys]
    differences = [row for row in recipe_rows if len({str(row[v]) for v in VARIANTS}) != 1]
    write_csv('training_recipe.csv', recipe_rows)
    accuracy_rows = []
    merged_rows = []
    delta_rows = []
    for v in VARIANTS:
        m = metrics[v]['metrics']
        accuracy_rows.append({'variant': v, **{k + '_percent': value * 100 for k, value in m.items()}})
        for precision in ['fp32', 'fp16']:
            r = records[v, precision]
            merged_rows.append({'variant': v, 'precision': precision, 'parameters': r['parameters_before_fusion'],
                                'parameters_fused': r['parameters_fused'],
                                **{key + '_percent': value * 100 for key, value in m.items()},
                                'accuracy_precision': 'fp32; fp16 accuracy not evaluated',
                                'forward_mean_ms': r['forward_wall']['mean_ms'],
                                'forward_median_ms': r['forward_wall']['median_ms'],
                                'forward_p95_ms': r['forward_wall']['p95_ms'],
                                'forward_fps': r['forward_wall']['fps'],
                                'forward_gpu_mean_ms': r['forward_gpu']['mean_ms'],
                                'pipeline_mean_ms': r['pipeline_wall']['mean_ms'],
                                'pipeline_p95_ms': r['pipeline_wall']['p95_ms'],
                                'pipeline_fps': r['pipeline_wall']['fps'],
                                'forward_peak_allocated_mib': r['forward_peak_allocated_mib'],
                                'pipeline_peak_allocated_mib': r['pipeline_peak_allocated_mib'],
                                'checkpoint_sha256': r['checkpoint_sha256']})
        baseline, r = records['r0', 'fp32'], records[v, 'fp32']
        delta_rows.append({'variant': v,
                           'AP_delta_pp': (m['AP'] - metrics['r0']['metrics']['AP']) * 100,
                           'APs_delta_pp': (m['APs'] - metrics['r0']['metrics']['APs']) * 100,
                           'parameters_change_percent': (r['parameters_before_fusion'] / baseline['parameters_before_fusion'] - 1) * 100,
                           'forward_fps_change_percent': (r['forward_wall']['fps'] / baseline['forward_wall']['fps'] - 1) * 100,
                           'forward_latency_change_percent': (r['forward_wall']['mean_ms'] / baseline['forward_wall']['mean_ms'] - 1) * 100,
                           'forward_memory_change_percent': (r['forward_peak_allocated_mib'] / baseline['forward_peak_allocated_mib'] - 1) * 100})
    write_csv('accuracy_metrics.csv', accuracy_rows)
    write_csv('all_results.csv', merged_rows)
    write_csv('deltas_vs_r0_fp32.csv', delta_rows)
    write_csv('summary_fp32.csv', [row for row in merged_rows if row['precision'] == 'fp32'])
    protocol = benchmark['protocol']
    env = benchmark['environment']
    assert env['gpu_compute_processes_before_start'] == ''
    assert protocol['batch'] == 1 and protocol['input_shape'] == [1, 3, 640, 640]
    assert len(protocol['images']) == 64
    done = datetime.fromisoformat(queue['finished_utc']).astimezone(TZ).strftime('%Y-%m-%d %H:%M:%S')
    accuracy_table = table(['模型', 'AP (%)', 'AP50 (%)', 'AP75 (%)', 'APs (%)', 'APm (%)', 'APl (%)', 'AR100 (%)'],
                           [[v.upper(), *[f"{metrics[v]['metrics'][k] * 100:.3f}" for k in ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR100']]] for v in VARIANTS])
    def performance_table(precision):
        rows = []
        for v in VARIANTS:
            r = records[v, precision]
            rows.append([v.upper(), f"{r['forward_wall']['mean_ms']:.3f}", f"{r['forward_wall']['p95_ms']:.3f}",
                         f"{r['forward_wall']['fps']:.2f}", f"{r['pipeline_wall']['mean_ms']:.3f}",
                         f"{r['pipeline_wall']['fps']:.2f}", f"{r['forward_peak_allocated_mib']:.2f}"])
        return table(['模型', '纯网络均值 ms', '纯网络 P95 ms', '纯网络 FPS', '处理流程均值 ms', '处理流程 FPS', '网络峰值分配 MiB'], rows)
    delta_table = table(['模型', 'ΔAP 百分点', 'ΔAPs 百分点', '参数变化 %', 'FPS 变化 %', '延迟变化 %', '显存变化 %'],
                        [[row['variant'].upper(), *[f"{row[k]:+.3f}" for k in ['AP_delta_pp', 'APs_delta_pp', 'parameters_change_percent', 'forward_fps_change_percent', 'forward_latency_change_percent', 'forward_memory_change_percent']]] for row in delta_rows])
    model_rows = [
        ['R0', 'P5 全局 AIFI', '保留', '无'], ['R1', 'Decoder 的 P4 支路新增 5×5 局部细化', '保留', '无'],
        ['R2a', 'P5 AIFI 替换为 7×7 NAIFI', '替换', '无'], ['R2b', 'P5 AIFI 替换为 7×7 NAIFI', '替换', '有，392 参数'],
        ['R2c', 'P5 AIFI 替换为 11×11 NAIFI', '替换', '有，968 参数'],
    ]
    parameters_table = table(['模型', '训练结构参数量', '融合后参数量', 'CSV 中最佳 epoch', '完成轮数'],
                             [[v.upper(), f"{records[v, 'fp32']['parameters_before_fusion']:,}",
                               f"{records[v, 'fp32']['parameters_fused']:,}", validation['variants'][v]['best_epoch_by_training_csv'], 150] for v in VARIANTS])
    recipe_text = table(['训练项', *[v.upper() for v in VARIANTS]], [[row['field'], *[row[v] for v in VARIANTS]] for row in recipe_rows if row['field'] != 'data'])
    history_rows = []
    history_batches = load('raw/history/summary.json') + [
        {'label': '2026-10-23 本报告统一重测', 'records': benchmark['records']}]
    for batch in history_batches:
        rr = {(r['variant'], r['precision']): r for r in batch['records']}
        history_rows.append([batch['label'], *[f"{rr[v, 'fp32']['forward_wall']['fps']:.2f}" if (v, 'fp32') in rr else '未测' for v in VARIANTS]])
    history_table = table(['数据批次', 'R0 FPS', 'R1 FPS', 'R2a FPS', 'R2b FPS', 'R2c FPS'], history_rows)
    checkpoint_table = table(['模型', '权重路径', 'SHA256'], [[v.upper(), f"`{validation['variants'][v]['checkpoint']}`", f"`{validation['variants'][v]['checkpoint_sha256']}`"] for v in VARIANTS])
    d = {row['variant']: row for row in delta_rows}
    r1_extra = records['r1', 'fp32']['forward_wall']['mean_ms'] - records['r0', 'fp32']['forward_wall']['mean_ms']
    k11_mem_extra = records['r2c', 'fp32']['forward_peak_allocated_mib'] - records['r2b', 'fp32']['forward_peak_allocated_mib']
    recipe_status = '所核对的关键训练项全部相同。' if not differences else '存在以下已记录差异：' + json.dumps(differences, ensure_ascii=False)
    report_text = f'''# RT-DETR R18 消融实验结果与推理基准报告

报告日期：2026-10-23（Asia/Shanghai）。项目：`{scope['project']}`。

## 1. 本轮结论与完成状态

R0、R1、R2a、R2b、R2c 均完成 150 轮训练；R2 队列的三组正式评估全部成功，结束时间为 **{done}（北京时间）**。本报告在空闲 RTX 4090 上重新测量五组 × FP32/FP16 共 10 个配置，并保存脚本、日志与逐次计时。

R2b 的正式 AP 最高，为 **{metrics['r2b']['metrics']['AP']*100:.3f}%**，相对 R0 增加 **{d['r2b']['AP_delta_pp']:.3f} 个百分点**，但小目标 APs 下降 **{-d['r2b']['APs_delta_pp']:.3f} 个百分点**。R1 的 APs 最高，为 **{metrics['r1']['metrics']['APs']*100:.3f}%**，但总体收益小且推理更慢。R2c 的总体 AP 低于 R0，扩大窗口在本轮未带来总体精度收益，FP32 峰值显存还明显增大。

以上仅为 **seed=0、150 轮** 的筛选结果，未验证多种子稳定性。精度是正式 FP32 COCOeval 数值；FP16 只测性能并检查输出有限值，没有重测 FP16 检测精度。

## 2. 实验范围和模型结构

本次授权范围是核验现有训练与评估、补齐性能测量并归档报告。不新增训练，不改变模型实现、训练配置或已有权重。旧结果和训练输出继续保留在原位置；汇报资料集中于本目录。

{table(['模型', '主要改动', 'R0 的 P5 全局注意力', '相对位置偏置'], model_rows)}

{parameters_table}

参数量使用融合前的训练结构统计，供架构比较；实际推理统一融合可融合层，融合后参数量另列。R2b/R2c 相对 R0 分别仅增加 392/968 个 RPB 参数，不能称为参数压缩。R2 是本项目定义的 encoder adaptation 消融，不声称是其他论文的完全复现。

## 3. 训练、精度和效率的统一口径

### 3.1 训练与精度

{recipe_status} 数据集为 VisDrone2019-DET，验证集 548 张图、38,759 个 GT、10 类；保存权重为各组 `best.pt`。完整训练项见 [训练配置对照](tables/training_recipe.csv)。

{recipe_text}

正式精度采用相同原生融合 FP32 Validator 导出、相同 COCO GT、`pycocotools.COCOeval`、`maxDets=[1,10,100]`、IoU=0.50:0.05:0.95。五份 protocol 逐字段一致，GT 注释 SHA256 为 `{metrics['r0']['protocol']['annotation_sha256']}`。原生 Validator 内部 mAP 与正式 COCO AP 分开保存，不能混用；本报告主精度表只用正式 COCO AP。

### 3.2 性能条件

- GPU：RTX 4090；驱动和测试前状态：`{env['gpu']}`。
- 软件：PyTorch `{env['torch']}`、CUDA runtime `{env['cuda_runtime']}`、cuDNN `{env['cudnn']}`、Ultralytics `{env['ultralytics']}`。
- PyTorch eager，eval + inference_mode，融合模型，batch=1，固定 640×640；没有 TensorRT、torch.compile 或 batch>1 吞吐测试。
- CPU 逻辑核固定为 {env['cpu_affinity']}，PyTorch {env['torch_threads']} 线程，OpenCV 1 线程；TF32 关闭，cuDNN benchmark 开启；没有锁定 GPU 时钟。
- 启动前无 GPU compute 进程。此次所有训练已经结束，无需暂停训练。
- 每配置热身 50 次，纯网络测量 3×150=450 次，保留均值、中位数、P95、标准差及逐次数据；配置顺序用固定种子打乱并记录在 protocol 中。
- 处理流程使用同一组 64 张已解码真实验证图片，重复 3 轮，共 192 帧；置信度 0.25、max_det=300。

纯网络延迟从 GPU 常驻输入开始，使用 `perf_counter` 加 CUDA 同步；同时保存 CUDA Event 耗时。处理流程包含原生缩放/归一化、传入 GPU、前向、原生后处理和检测框回传 CPU。处理流程 **不包含磁盘读取、JPEG 解码、绘图、模型加载**。FPS=1000/平均逐帧延迟，不能解释为多帧流水线或最大批量吞吐。

显存为 benchmark 进程的 PyTorch peak allocated（MiB），包含模型和中间张量，不包含 CUDA 上下文、其他进程，也不同于 `nvidia-smi` 整卡占用。reserved 数据保留在原始 JSON；报告主表使用 allocated。

## 4. 正式精度结果

{accuracy_table}

完整 12 项 COCO 指标见 [accuracy_metrics.csv](tables/accuracy_metrics.csv)，原始结果见 [raw/accuracy](raw/accuracy/)。AP/APs 等值的差异使用“百分点”，不混写为相对百分比。

## 5. 推理性能结果

### 5.1 FP32

{performance_table('fp32')}

### 5.2 FP16

{performance_table('fp16')}

FP16 在当前原生 PyTorch 实现上未普遍加速；该现象不能推及 TensorRT 或其他设备。每个配置的运行时权重 dtype 均已核验，模型输出均为有限值。性能表中的 FP16 数值不代表 FP16 精度已与 FP32 等价。

### 5.3 相对 R0 的精度与效率变化

{delta_table}

延迟/显存正数表示增加，FPS 正数表示提高。最小的 R2 组间差异仅来自本次测量，不据此宣称 RPB 本身可以加速。详表见 [all_results.csv](tables/all_results.csv)；其中 AP 字段始终来自 FP32 正式评估，即使该行的性能精度为 FP16。

## 6. 对结果的分析

### 6.1 R1：新增支路带来成本，小目标 AP 有信号

R1 保留原 P5 AIFI，在 Decoder 的 P4 分支新增 `LayerNorm2d → 降维 → NA5 → 升维 → 门控残差`。640 输入时 P4 有 40×40=1,600 个位置。门控系数无论大小，当前代码都会计算整条支路，不存在按系数跳过计算的逻辑。

本轮 R1 比 R0 多 **{r1_extra:.3f} ms/图**，FPS 变化 **{d['r1']['forward_fps_change_percent']:+.2f}%**，与新增工作量一致。新增权重仅 132,097 个，FP32 约 0.50 MiB；但 K/V 的 `unfold` 展开可合计产生约 39.1 MiB 临时张量。整网峰值取决于各张量的存活时段，不能把该值直接加到基线峰值上。

R1 总体 AP 仅提高 {d['r1']['AP_delta_pp']:.3f} 个百分点，APs 提高 {d['r1']['APs_delta_pp']:.3f} 个百分点。如果目标是小目标精度，这提供了后续验证的线索；如果重点是总体精度和速度，本轮收益相对额外开销偏小。单种子结果不能证明稳定改善。

### 6.2 R2a/R2b：保持参数规模，整体有小幅速度收益

R2 以 P5 邻域注意力替换全局注意力，其他主体保持一致。P5 只有 20×20=400 个位置；k=7 时每位置的邻域为 49。注意力交互项从 400 降到 49，对应该子项理论减少 87.75%，但 QKV/输出投影、FFN、backbone、neck 和 Decoder 没有按同比例减少。

R2a 的参数量与 R0 完全相同，R2b 仅多 392 个参数。相对 R0，本轮 FP32 FPS 分别变化 **{d['r2a']['forward_fps_change_percent']:+.2f}% / {d['r2b']['forward_fps_change_percent']:+.2f}%**。这是包含注意力、位置编码、归一化和 FFN 实现变化的整段替换净效果；没有逐算子 profiler 证据，不能将全部加速归于邻域注意力交互减少。

R2b 相对 R2a 的 AP 变化为 **{(metrics['r2b']['metrics']['AP']-metrics['r2a']['metrics']['AP'])*100:+.3f} 个百分点**，APs 变化 **{(metrics['r2b']['metrics']['APs']-metrics['r2a']['metrics']['APs'])*100:+.3f} 个百分点**。这次加入 RPB 并未带来小目标改善，不能将它描述为全面优于无 RPB。

### 6.3 R2c：扩大窗口没有换来总体精度收益，FP32 显存代价明显

7×7 到 11×11 将每位置邻居从 49 增至 121，即约 2.47 倍。参数仅因 RPB 多 576 个，但展开的 K/V 和其他中间张量会增大。本轮 R2c 比 R2b 的 FP32 网络峰值分配多 **{k11_mem_extra:.2f} MiB**；它相对 R0 的 AP 变化为 **{d['r2c']['AP_delta_pp']:+.3f} 个百分点**。

因此，本轮没有证据支持为总体精度继续扩大这个窗口。它在 APl 上的收益仍需单独看待，不能掩盖总 AP 和小目标 AP 的下降。这里讨论的是筛选结果，不是对所有训练长度和种子的否定。

### 6.4 FP16、峰值显存和理论 FLOPs 的解释边界

模型权重字节减少，不保证峰值显存或延迟同幅下降。当前实现包含 `unfold`、`einsum`、手写归一化、逐尺度 `grid_sample` 等操作；不同精度也可能采用不同工作区与计算算法。它们是值得检查的候选原因，尚未通过逐算子剖析确认哪一项主导 FP16 的变化。

既有 THOP 统计缺少完整全局注意力计数，自定义函数算子也不能默认被完整覆盖，不能将日志中的 GFLOPs 作为三组注意力实现的严格公平比较。本报告以同协议实测延迟、FPS、显存和参数量为主要效率证据。

## 7. 历史数据与本次数据的关系

{history_table}

主表仅使用 **本报告目录内此次统一重测** 的五组数据，不将昨天三组结果与今天剩余两组拼接。云端历史摘要保存在 [raw/history/summary.json](raw/history/summary.json)，其中记录原始文件路径、大小和 SHA256。历史逐次原始 JSON 仅保留于服务器，不随 Git 重复上传。跨批次绝对时间会变化，未锁时钟，也没有对变化做逐项因果分析；主要判断应使用同批次对照。

训练 CSV 的 `time` 是各运行记录的墙钟时间，包含不同运行环境、恢复和暂停等影响。尤其 R2b 第 80 轮附近曾包含约 65 秒的基准测试暂停，不能直接用训练总时长证明注意力实现更快。

## 8. 证据、发现和复现路径

| 证据 | 文件 | 支持的发现 | 核验或复现路径 |
| --- | --- | --- | --- |
| E1 | `raw/queue_status.json`、`raw/training/*_results.csv` | 五组训练完成，R2 三组评估完成 | 核对 150 行、最终 epoch=150、队列 completed 列表 |
| E2 | `raw/accuracy/*_formal_metrics.json` | 同协议 COCO AP/APs 对照 | 比较 protocol 与 GT 哈希，读取 metrics；原评估代码在 sources |
| E3 | `raw/inference_20261023/benchmark.json`、`raw/benchmark_20261023.log` | 10 个配置的性能与显存结果 | 对照 checkpoint 哈希、dtype、逐次计时与汇总统计 |
| E4 | `sources/rtdetr_na.py`、模型 YAML | R1 新增支路、R2 替换 P5，窗口展开带来中间张量 | 阅读 ResidualLocalRefine / NeighborhoodAttention2d / NAIFI 和相应 YAML |
| E5 | `provenance/validation.json`、`source_files.json`、`MANIFEST.sha256` | 权重身份、原训练源码一致性及归档完整性 | 运行本节校验命令，重新生成汇总表 |

### 8.1 权重身份

{checkpoint_table}

没有复制或移动权重，报告通过原始绝对路径及 SHA256 引用它们。归档的训练源码与原队列 `source_sha256.json` 逐项相符。

### 8.2 在原服务器复现实测

先确保 GPU 无其他计算任务。下列命令使用新的输出目录，不覆盖本报告实测数据：

```bash
cd '{REPORT}'
/home/zjnu/anaconda3/envs/UAV_CV/bin/python scripts/benchmark_inference.py --output '{REPORT}/raw/reproduction_20261023'
```

该脚本会在检测到其他 GPU compute 进程时退出。复现目录已存在时也会退出，避免覆盖。重复复现需要使用一个新的目录名。复现性能不等于重跑完整精度；正式精度的原评估实现和 checkpoint 身份已保存在 sources、raw/accuracy、provenance。

重新核验汇总表和报告：

```bash
cd '{REPORT}'
/home/zjnu/anaconda3/envs/UAV_CV/bin/python scripts/build_report.py
sha256sum -c MANIFEST.sha256
```

## 9. 目前可采取的下一步

若优先总 AP 与单帧速度，R2b 可作为下一阶段候选，但增益只有约 0.11 个百分点；应先验证多种子再扩大训练预算。若优先 APs，R1 的信号更值得检查，不过需要解决额外支路的开销。R2c 暂不显示总体优势，不应仅因窗口更大就继续增加预算。任何新增训练均未由本报告自动启动。

若继续追求部署效率，应先做逐算子耗时与内存分析，再考虑窗口展开、mask/位置编码缓存和算子融合；这些属于后续工作，本轮没有实施，也不声称已有对应加速效果。

## 10. Git 追踪与网页端阅读

本目录遵循项目 [AGENTS.md](../../../AGENTS.md) 的轻量归档原则。网页端从 README 和正文报告进入，关键精度/效率表、150 轮训练曲线、配置、源码、校验信息和一份当前完整逐次计时均随 Git 提供。历史逐次数据改为紧凑摘要；权重、数据集、逐图预测、完整训练日志、压缩快照、修复备份继续保留在服务器，由路径和哈希引用。

`MANIFEST.sha256` 仅校验随 Git 提供的文件，从云端检出后可直接运行。服务器额外清单保存为 `MANIFEST.server.sha256`，不随 Git 上传。只使用 Git 文件即可重建本报告；重新运行 GPU 基准仍需要原服务器上的数据集、权重和相应环境。

## 11. 归档质量检查

- [x] 五组均有 150 轮训练记录和匹配的最终权重身份。
- [x] 五份正式评估协议逐字段一致；内部 mAP 和正式 AP 分开。
- [x] 五组 × 两种精度的 10 个性能配置完整，每组 450 次前向、192 帧处理流程。
- [x] dtype、有限输出、逐次均值/中位数与 FPS 公式均已核验。
- [x] 原始记录、源代码、日志、配置、权重哈希和历史数据均可定位。
- [x] 单种子、FP16 精度未验证、无时钟锁定及无逐算子剖析等限制已说明。
- [x] 全部新汇报资料集中于此目录，原训练/评估产物未搬移或覆盖。
'''
    report_name = '2026-10-23_实验结果-rtdetr-r18-report.md'
    (REPORT / report_name).write_text(report_text)
    readme = f'''# R18 消融实验汇报目录

R0、R1、R2a、R2b、R2c 的训练、正式精度与 FP32/FP16 推理性能汇总。R2 队列于 {done}（北京时间）完成。

- [完整中文报告]({report_name})：结果、分析、测试边界、证据和复现命令。
- [FP32 主对照表](tables/summary_fp32.csv)、[全部结果](tables/all_results.csv)、[正式精度](tables/accuracy_metrics.csv)、[相对 R0 变化](tables/deltas_vs_r0_fp32.csv)。
- [本次基准原始数据](raw/inference_20261023/benchmark.json)、[测试协议](raw/inference_20261023/protocol.json)、[运行日志](raw/benchmark_20261023.log)。
- [基准脚本](scripts/benchmark_inference.py)、[报告生成与校验脚本](scripts/build_report.py)。
- `raw/accuracy/`、`raw/training/`：精度与训练记录快照；`raw/history/`：之前批次的紧凑摘要，历史逐次原始 JSON 仅保留于服务器。
- `sources/`：模型配置与相关实现快照；`provenance/`：范围、验证、原始文件位置及哈希；`MANIFEST.sha256`：归档文件校验清单。

主性能对照只使用此次同批次重测；所有 AP 来自 FP32 正式评估，FP16 仅测性能。原权重与训练目录保持原位置，通过报告中的路径和哈希引用。
'''
    (REPORT / 'README.md').write_text(readme)
    qa = {'validated_at': datetime.now(TZ).isoformat(), 'training_and_accuracy_variants': VARIANTS,
          'benchmark_configurations': len(records), 'forward_samples_total': 450 * len(records),
          'pipeline_samples_total': 192 * len(records), 'all_checkpoint_hashes_match': True,
          'all_accuracy_protocols_match': True, 'all_runtime_dtypes_match': True,
          'all_timing_summaries_match_raw_samples': True, 'training_recipe_differences': differences}
    (REPORT / 'provenance/report_validation.json').write_text(json.dumps(qa, indent=2, ensure_ascii=False) + '\n')
    links = []
    import re
    for file in [REPORT / 'README.md', REPORT / report_name]:
        for target in re.findall(r'\]\(([^)]+)\)', file.read_text()):
            if '://' not in target:
                assert (file.parent / target).exists(), target
                links.append(target)
    assert 'TODO' not in report_text and 'placeholder' not in report_text
    manifest = []
    cloud_files = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', '.'], cwd=REPORT).decode().split('\0')
    cloud_files = sorted(set(name for name in cloud_files if name and Path(name).name not in ['MANIFEST.sha256', 'MANIFEST.server.sha256']))
    for name in cloud_files:
        file = REPORT / name
        manifest.append(hashlib.sha256(file.read_bytes()).hexdigest() + '  ' + name)
    (REPORT / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    server_manifest = [hashlib.sha256(file.read_bytes()).hexdigest() + '  ' + str(file.relative_to(REPORT))
                       for file in sorted(REPORT.rglob('*')) if file.is_file() and file.name != 'MANIFEST.server.sha256']
    (REPORT / 'MANIFEST.server.sha256').write_text('\n'.join(server_manifest) + '\n')
    print(json.dumps({'report': str(REPORT / report_name), 'archive_files': len(manifest), 'verified_links': len(links),
                      'qa': qa, 'summary': [{k: row[k] for k in ['variant', 'AP_percent', 'APs_percent', 'forward_mean_ms', 'forward_fps', 'forward_peak_allocated_mib']} for row in merged_rows if row['precision'] == 'fp32']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
