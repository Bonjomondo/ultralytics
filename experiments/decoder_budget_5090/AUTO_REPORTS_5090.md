# 5090 实验自动汇总

参考 SO-DETR 的 `sodetr.reporting.experiments` 工作方式，扫描本分支原生训练、Query Sweep 记录，自动生成 `reports/5090/experiments.{md,csv,json,xlsx}` 和 `reports/5090/plots/`。汇总器不加载模型、权重、CUDA或逐图预测，不触发训练与评估。

## 自动触发

`queue_5090.py` 每次更新状态时刷新报告，覆盖阶段开始、完成、失败、门禁通过和整个队列结束。默认扫描实验根目录的父目录，将各批5090实验一起汇总；可用 `--report-dir` 指定输出位置。报告失败会写入当前队列 `status_5090.json` 的 `reporting` 字段和控制日志，不会掩盖训练状态或终止正在运行的训练。

训练过程中的每轮细节仍由 `experiment_5090.json` 和 `results.csv` 记录。汇总的自动刷新点是队列阶段变化；需要查看某个训练轮次的即时汇总时可手动刷新。已经完成的历史队列可直接扫描回填，不需要重新执行它的冻结源码门禁，也不改写旧run。

## 单命令重生成

从仓库根目录运行：

```bash
bash experiments/decoder_budget_5090/update_reports_5090.sh
```

或显式选择输入、输出：

```bash
/home/ubuntu/miniconda3/envs/UAV_CV/bin/python \
  experiments/decoder_budget_5090/report_5090.py \
  --runs-dir runs/decoder_budget_5090 --output reports/5090
```

依赖为Python≥3.10、Matplotlib和openpyxl；本机现有UAV_CV环境已经提供。Shell入口可通过 `UAV_REPORT_PYTHON` 使用其他解释器。所有输出通过文件锁与临时目录生成后逐文件原子替换，重复执行更新报告，不修改原始实验。

## 内容与口径

- 主结果只收录正式训练角色、完整验证集、完成且身份核验通过的条件。8图接入门禁与1轮短跑保留在实验说明/结构化文件中，失败/中断后残留的AP标记为 `stale`，不作为有效结果显示。
- AP、AP50、AP75、APs、APm、APl和12项COCO指标；同一checkpoint的300Q与其他预算的ΔAP/ΔAPs。不同队列用包含批次名的 `run_id` 区分，不合并同名run，不跨checkpoint计算Query差值。
- 原生观测/卸载恢复门禁、checkpoint/manifest/覆盖分析身份核验、训练轮数与最佳内部mAP轮次；训练内部Precision/Recall/mAP与正式COCOeval分开。
- 未融合FP32 batch1整网forward平均/中位/P95延迟、GPU FPS、峰值allocated显存，以及Encoder Coverage@IoU.5和正确类Coverage。主结果表也直接显示平均GPU延迟与FPS。FPS由保存的原始样本计算为 `1000 / mean_ms`，不取中位延迟倒数、不平均逐次FPS；没有有效样本时显示为空。采集batch2与计时batch1分别标注，不包含传输/预处理/后处理；Coverage不当作正式召回或AP。
- 训练内部mAP曲线、Query AP/APs曲线和Query计时曲线。仅扫描保存的数据，不把小样本结论外推到全量，不将单次细小时延差当作稳定加速。

Markdown/Excel以百分数显示检测指标；JSON/CSV保存0–1原始值，差值单位为百分点。COCO的 `-1` 指标显示为空，不当作负AP或零AP。所有仅服务器保存的路径在Markdown内以文字列出，图表使用报告内的相对链接。

## 无服务器数据重生成

`experiments.json` 包含完整训练曲线、评估摘要及其保存身份。拿到报告文件后，可在没有数据、权重、run或DETRScope的机器上重新输出其余格式：

```bash
/home/ubuntu/miniconda3/envs/UAV_CV/bin/python \
  experiments/decoder_budget_5090/report_5090.py \
  --from-json reports/5090/experiments.json --output /tmp/5090-report-replay
```

该模式保留快照时间，不假装检查远端服务器的实时状态。原实验的冻结源码SHA保持不变；后续队列增加了自动报告功能，需以新代码新建实验，不通过改写旧冻结清单恢复训练。


## 分段延迟归档接入

2026-10-25的C0/C2分段评估保存在[基线报告](../../docs/experiment_reports/2026-10-24_5090_baselines/README.md#5090分段延迟与瓶颈评估)。刷新时由 `stage_latency_5090.py` 校验已归档JSON的SHA、执行脚本SHA、五轮三预算双模式的完整性、逐次分段之和、统计摘要与等价/恢复门禁；仅关联best checkpoint SHA完全匹配的已完成正式训练。没有匹配训练时不显示该实验的分段结果，部分匹配/重复身份/证据损坏会写入 `collection_errors`，不影响其他有效AP的展示。

分段结果自动进入Markdown、`experiments.json` 的 `stage_assessment`、`stage_latency.csv`（30行）及Excel的 `Stage latency` 页。CSV/JSON的 `fraction_of_staged_forward` 为0–1占比；`instrumentation_overhead_percent` 已是百分数，`0.53`表示0.53%。各段五轮均值范围与250次样本数单独保留；完整逐次样本仅归档一份，不重复塞进汇总快照。旧快照没有该字段时正常生成空的分段CSV/Excel页。

这是一张固定真实输入、FP32 batch1未融合路径上的独立诊断，不替换原Query Sweep整网计时，不表示端到端或纯GPU忙碌时间。报告重生成只读取归档证据，不自动发起GPU测试；`--from-json` 完全使用已保存的分段摘要，不访问原始归档或服务器资产。
