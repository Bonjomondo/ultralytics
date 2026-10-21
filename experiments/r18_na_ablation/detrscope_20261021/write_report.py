"""Generate the Chinese experiment report from exported evidence tables."""
from collections import Counter
import csv
from pathlib import Path
from detrscope.core.io import read_json

p=Path(__file__).resolve().parent
c=read_json(p/'compatibility.json');metrics=read_json(p/'formal_evaluation.json')['metrics']
native=read_json(p/'native_val_internal_metrics.json')['metrics'];nativeformal=read_json(p/'native_val_formal_coco.json')['metrics']
summary=read_json(p/'baseline_summary.json');replay=read_json(p/'offline_replay.json')
base=read_json(p/'probe_sample_coco.json')['metrics'];zero=read_json(p/'zero_d3_sample_coco.json')['metrics']
gtchanges=read_json(p/'zero_d3_formal_gt_changes.json')
r0=read_json(p/'r0_progress_snapshot.json')
rows=lambda name:list(csv.DictReader((p/name).open()))
perclass=rows('per_class_coco.csv');stages=rows('gt_stage_means.csv')
activity=rows('bbox_hidden_activity_sample.csv');residual=rows('na_residual_sample.csv')
variation=rows('bbox_output_variation.csv')
rates=[r for r in summary['statistics'] if r['family']=='gt_object' and r['class_id']=='all' and r['object_size']=='all'
       and r['event'] in ('localization_drop','classification_drop','localization_gain','classification_gain','query_switch')]
rate_table='\n'.join(f"| {r['transition']} | {r['event']} | {r['numerator']} / {r['denominator']} | {r['rate']*100:.3f}% |" for r in rates)
class_table='\n'.join(f"| {r['class_name']} | {r['gt_count']} | {float(r['AP'])*100:.3f} | {float(r['APs'])*100:.3f} |" for r in perclass)
stage_table='\n'.join(f"| D{r['layer']} | {r['valid_gt_assignments']} | {float(r['mean_iou']):.6f} | {float(r['mean_gt_class_probability']):.6f} |" for r in stages)
change_table='\n'.join(f"| {r['iou']:.2f} | {r['counts']['lost']} | {r['counts']['gained']} | {r['counts']['stable_tp']} | {r['counts']['stable_fn']} |" for r in gtchanges['thresholds'] if abs(r['iou']-.5)<1e-6 or abs(r['iou']-.75)<1e-6)
metric_table='\n'.join(f"| {k} | {metrics[k]*100:.3f} |" for k in ('AP','AP50','AP75','APs','APm','APl','AR100'))
perturb='\n'.join(f"| {k} | {base[k]*100:.3f} | {zero[k]*100:.3f} | {(zero[k]-base[k])*100:+.3f} |" for k in ('AP','AP50','AP75','APs'))
act=[float(r['relu_positive_fraction']) for r in activity];res=[float(r['na_residual_rms_ratio']) for r in residual]
text=f'''# RT-DETR-R18 + P4 NA5 真实 checkpoint / CUDA 验收报告

日期：2026-10-21（Asia/Shanghai）。主机：`zjnu-Precision-3660`。实验：`r18_na_ablation/rtdetr-r18-r1-150e-s0`。

## 当前结论

**真实权重、RTX 4090、完整 548 张 VisDrone 验证图的 detrscope 观测、分析、正式 COCOeval 和离线重建已完成。** 全量 274 个 batch 的原生/观测/恢复预测及 RNG 一致性门禁通过，模型状态 SHA 不变。相同原始记录二次重建的 {replay['analysis_artifacts_compared']} 个分析文件 SHA 全等，预测 JSON、COCOeval 结果、HTML/Markdown 报告亦一致。8 图恒等干预结果全等，正常及异常退出均移除 Hook。[兼容性证据](compatibility.json)、[契约检查](contract_checks.json)、[重建核验](offline_replay.json)。

正式 **AP={metrics['AP']*100:.3f}%、AP50={metrics['AP50']*100:.3f}%、APs={metrics['APs']*100:.3f}%**。只评估 R1；R0 在导出快照时到第 {r0['last_logged_metrics']['epoch']} / 150 轮，仍在训练，不能据此宣称 NA 带来结构收益。原 R0/R1 共有初始化的 533 项 SHA 一致，完成后具备开展同预算单 seed 对照的基础；稳定收益还需多 seed。[R0 快照](r0_progress_snapshot.json)、[原始初始化](initialization_original.json)。

## 数据、权重与运行身份

- 权重：`runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/weights/best.pt`，SHA256 `bd3e84698c32771ff49ee9e2d3c02feb968ec4c7aada5e993416a5e12f5b16a9`。`last.pt` 与之字节相同，未重复当作两组实验。文件已剥离训练状态，`epoch=-1`；150 轮完成状态来自完整训练 CSV。保存模型为 EMA，非第 150 轮即时优化器权重。
- 模型源码：`10ce3bdab8030e73561d1ec2315e519289f1c002`；201 个 Python 源文件 SHA 钉住。`PinnedR18NAAdapter` 复用 detrscope 原生 collector 与同一套分析算法；不关闭/冒充官方 wheel 身份门禁。SDK：`0c8624a32bde8fbac5cd6a59e6f2a544225b5051` / v0.2.0，源码未修改。[源码 pin](source_pin.json)、[适配器](source_adapter.py)。
- 环境：`/home/zjnu/anaconda3/envs/UAV_CV/bin/python`，Python 3.11.17、torch 2.1.2+cu121、RTX 4090。FP32、batch=2、640×640 stretch、RGB /255；未融合的原始网络用于轨迹采集。首 2 张图与真实 RTDETRDataset、RTDETRPredictor 的预处理及原生后处理逐值一致，全部记录最终 stage 与原生输出逐值一致。
- GT：现有 `VisDrone2019-DET_val_coco.json`，548 图 / 38,759 注释 / 10 类；当前原生 YOLO 验证标签数量完全相同，按类别和坐标排序的最大归一化差为 5e-7，仅为标签舍入精度。本报告基于这份转换后 GT，不声称采用原始 VisDrone 官方忽略区域评测。[GT 核查](gt_label_audit.json)。
- 输出：全量 Query / GT / Reference / bbox delta；前 8 图额外采集 P4、NA 输出、D3 第二回归隐藏层 Feature，以及 3 层 deformable cross-attention / Sampling。未采集反向梯度、NA 自身 attention matrix 或自注意力。单独诊断进程采集峰值已分配 CUDA 显存约 {c['peak_cuda_allocated_mib']:.1f} MiB。

## 正式 COCO 指标与内部日志口径

正式协议为 pycocotools bbox COCOeval，IoU=.50:.05:.95、maxDets=[1,10,100]，每 Query 取类别 argmax，conf>0.001，按分数排序最多保留 300 框/图，不做 NMS。像素坐标由原始尺寸还原。数字以百分数给出。

| 指标 | 数值（%） |
| --- | ---: |
{metric_table}

[完整 12 指标](formal_evaluation.json)、[逐类别](per_class_coco.csv)、[逐图](per_image_coco.csv)。

| 类别 | GT 数 | AP（%） | APs（%） |
| --- | ---: | ---: | ---: |
{class_table}

对同一 checkpoint，独立、未修改的原生 Validator 再跑全量得到内部 mAP50-95={native['metrics/mAP50-95(B)']*100:.3f}%、mAP50={native['metrics/mAP50(B)']*100:.3f}%，与训练第 150 轮 CSV 的 20.695% / 35.170% 接近。将该 Validator **自己的预测 JSON**交给同协议 COCOeval 后，AP={nativeformal['AP']*100:.3f}%，与记录导出的 AP 差 {(nativeformal['AP']-metrics['AP'])*100:+.4f} 个百分点；native 路径有融合和坐标/分数舍入，两个导出不是逐值相同的产物。[原生内部结果](native_val_internal_metrics.json)、[原生预测正式评估](native_val_formal_coco.json)、[交叉核对](native_vs_recorded_coco.json)。

因此，约 2 个百分点的主要差距存在于内部指标与 COCOeval 之间，不能归因 detrscope 改变了预测。maxDets=300 的补充 AP=18.731%，仅比标准结果高约 0.029 个百分点，也解释不了该差距。**内部匹配/AP 汇总与 COCOeval 的具体差异尚未完全分解；不将内部 mAP 冒充正式 AP。**[maxDets 敏感性](maxdets_sensitivity.json)。

## 逐层直接观测

下表使用无 confidence/IoU 门槛的 Hungarian GT 责任分配，代价为 `(1-IoU)+0.25*(1-pGT)`。每层重新分配，Query 可切换；均值不等同检测 AP、召回率或固定 Query 改善。

| 层 | 有效 GT 责任数 | 平均 IoU | 平均 GT 类概率 |
| --- | ---: | ---: | ---: |
{stage_table}

共 168,171 条固定 Query/GT 对轨迹、413,780 个相邻事件；80 个未形成有效相邻对的转换被排除，各指标使用实际有效分母。

| 转换 | 事件 | 分子 / 有效 GT 对分母 | 比例 |
| --- | --- | ---: | ---: |
{rate_table}

退化/增益阈值分别为 IoU 或 pGT 变化超过 0.02。D2→D3 同时存在定位下降 14.374%、定位增益 16.670%、类别概率下降 19.353% 和类别概率增益 35.240%。总体均值改善不能掩盖局部退化；Query 切换本身也不是错误。[完整类别/尺度分层](baseline_transition_stats.csv)、[12 个示例轨迹](diagnostic_cases.json)。

回归头没有出现“所有 Query 输出同一 bbox delta”的现象：全 548 图、3 层共 1,644 个 image-layer 记录的精确常量计数为 **0**；D3 每图 Query 中心化最大绝对变化最小仍为 {min(float(r['query_centered_max_abs']) for r in variation if r['layer_id']=='3'):.6f}。8 图 D3 第二隐藏层 ReLU 正值比例为 {min(act)*100:.3f}%–{max(act)*100:.3f}%，不是全零隐藏激活。[全量输出变化](bbox_output_variation.csv)、[隐藏激活采样](bbox_hidden_activity_sample.csv)。这是当前样本/权重的观测，不外推为完整训练过程中从未发生退化。

EMA NA γ={c['checkpoint_gamma']['model.23.gamma']:.8f}；8 图 NA 残差相对原 P4 的 L2/RMS 比例为 {min(res)*100:.3f}%–{max(res)*100:.3f}%，该分支并非恒等输出。训练最后一次健康采样来自 live 权重，γ={c['training_last_health']['gamma']:.8f}；不能将 live 日志与 EMA checkpoint 当作同一权重。[NA 残差](na_residual_sample.csv)、[训练健康日志](training_health.jsonl)。

## 限定条件下的冻结权重干预

只在图像 ID 1–8 上，将 D3 bbox delta 乘 0；不重新训练，不关闭 NA，不改类别读出。先通过 ×1 恒等门禁，并在干预后恢复原生行为。

| 指标 | 原生 8 图（%） | D3 delta×0（%） | Δ（百分点） |
| --- | ---: | ---: | ---: |
{perturb}

| COCO IoU | Lost | Gained | Stable TP | Stable FN |
| --- | ---: | ---: | ---: | ---: |
{change_table}

这些 Lost/Gained 来自同 GT、同协议 COCOeval 正式匹配；通用 comparison 文件仍保留其外部预测绑定限制声明，本实验预测来源通过导出脚本、原始 manifest 与 SHA 另行记录。仅 8 张、按图像 ID 取前段，存在选择限制；不能把 AP50 的局部增加或 AP 的局部下降升级为训练机制、全量收益或删除 D3 的建议。[同 GT 变化](zero_d3_formal_gt_changes.json)、[跨条件分组](comparison_group_stats.csv)。

## 运行问题与证据边界

未绑定 CPU 的离线尝试发生两次 Python TypeError；启用 debug allocator/dev mode 但未绑定 CPU 的尝试发生段错误。失败过程及不完整 analysis 目录原样留在 `logs/detrscope_r18_{{full,offline,debug}}_20261021.log` 和原 run 下，不作为有效统计读取。最终在 **CPU 24–27 + PYTHONMALLOC=debug + Python -X dev** 下完成全量分析和第二次重建，并通过全部 SHA/统计/正式评估一致性检查；底层异常根因未确证，不据此诊断硬件故障或模型机制。[运行记录](analysis_runtime.json)。

当前 UAV_CV 的 torch 2.1.2 低于 SDK 发布元数据要求的 torch>=2.6；本次为该环境中源码执行的实际路径验证，不代表 torch 2.1 全 API 或正式安装兼容承诺。没有升级/替换训练环境，没有停止 R0，也没有新增训练。

## 下一步待验证假设

1. R0 完成相同 150 轮后，使用相同数据、COCOeval 和采集协议比较 R0/R1，检验 NA 是否改善小目标、类别混淆与逐层退化；当前仅知道初始化配对。
2. 将 D2→D3 的固定 Query 退化与 GT 责任切换分开，用逐类别/尺度/密度和正式 Lost/Gained 检查哪些群体值得扩大干预；8 图结果不够给出架构结论。
3. 若要解释内部 mAP 与 COCOeval 差距，固定同一份预测继续拆解匹配与 AP 汇总，而不是修改模型或挑选较高数字。

## 复现与协同分析

[复现说明](README.md)、[网页端 ChatGPT 上下文](CHATGPT_CONTEXT.md)。轻量文件由 Git 追踪；权重、图片、完整 tensor/预测/轨迹、HTML 和日志保留在服务器原 run 目录。原始文件与精简导出的 SHA 见 [artifact_index.json](artifact_index.json) 和各条件 manifest。后续分析请引用图像 ID / GT ID，不跨模型对齐 Query ID。
'''
(p/'REPORT.md').write_text(text,encoding='utf8')
print('REPORT WRITTEN',len(text))
