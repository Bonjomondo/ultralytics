# R0 / R1 完成版对照

两组均已完成 150 轮，seed=0，完整 VisDrone 验证集 548 图 / 38,759 GT。R0 的最后 checkpoint 保存时间为主机本地 2026-10-21 18:53:59；训练日志正常结束，非中断结果。

## 当前结论

训练 CSV 内部 mAP50-95：R0 **20.593%**，R1 **20.695%**，R1−R0 **+0.102 个百分点**；mAP50 分别为 **34.982% / 35.170%**，差 **+0.188 个百分点**。这不是正式 COCOeval 数值。

下表使用相同原生 Validator 导出及标准 COCOeval 协议。R0 新增原生验证；R1 复用先前原生预测 JSON，但重新核验 checkpoint、源码 SHA、预测身份并重算，正式指标与历史结果逐值一致。R1 历史报告 18.702% 使用未融合的轨迹导出，本次使用与 R0 相同的原生融合导出，所以取其 18.705% 作为对照，避免混用路径。

| 正式指标 | R0（%） | R1（%） | R1−R0（百分点） |
| --- | ---: | ---: | ---: |
| AP | 18.650 | 18.705 | +0.055 |
| AP50 | 32.371 | 32.426 | +0.055 |
| AP75 | 18.181 | 18.183 | +0.002 |
| APs | 12.219 | 12.616 | +0.397 |
| APm | 26.305 | 26.087 | -0.217 |
| APl | 29.587 | 30.129 | +0.543 |
| AR100 | 28.434 | 28.286 | -0.148 |

本次正式 AP 差 +0.055 个百分点，总体差异很小；APs 差 +0.397，但 APm 差 −0.217、ARs 差 −0.034 个百分点，表现存在群体和指标上的取舍，不能概括为全面改善。

共有初始化 533 项 SHA 一致；两组关键训练配方一致，R1 曾从第 36 轮 checkpoint 续跑。以上仅是单 seed 完成版结果，不能据此宣称稳定收益或训练因果。此前“R0 尚未完成”的记录应按历史快照阅读；最新状态以此目录为准。

## 正式同 GT 匹配变化

| IoU | Lost（R0 TP→R1 FN） | Gained | Stable TP | Stable FN |
| --- | ---: | ---: | ---: | ---: |
| 0.50 | 2380 | 2227 | 23432 | 10720 |
| 0.75 | 2457 | 2369 | 13157 | 20776 |

数据来自同一份 COCO GT、相同评估协议和注释 ID；不是 Hungarian 诊断责任。完整 GT ID 列表保留在 `/home/zjnu/文档/ZHJ_WorkSpace/ultralytics/runs/detect/r18_na_ablation/completed_pair_evaluation/gt_changes.json`，未提交大体积原始预测。

## 证据与复现

- [正式对照表](formal_metrics_comparison.csv)、[R0 正式结果](r0_formal_metrics.json)、[R1 正式结果](r1_formal_metrics.json)、[身份与训练配方核验](verification.json)。
- [R0 训练曲线](r0_training_metrics.csv)、[R0 健康日志](r0_training_health.jsonl)、[R0 原生内部指标](r0_native_internal.json)。
- [R1 训练曲线](../detrscope_20261021/training_metrics.csv)、[R1 既有逐层诊断](../detrscope_20261021/REPORT.md)。本次未新增 R0 Decoder 轨迹采集，不声称已完成两组完整 Debug 机制比较。

在仓库根目录使用 UAV_CV，传入新的 `--raw` 输出目录，避免覆盖证据，再通过 tmux 启动。脚本将刷新本目录的生成表，复现后先检查 Git diff：

```bash
cd /home/zjnu/文档/ZHJ_WorkSpace/ultralytics
/home/zjnu/anaconda3/bin/tmux new-session -d -s r18-pair-eval \
  'PYTHONPATH=$PWD:/home/zjnu/文档/ZHJ_WorkSpace/detrscope/src PYTHONMALLOC=debug CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 24-27 /home/zjnu/anaconda3/envs/UAV_CV/bin/python -X dev -u experiments/r18_na_ablation/completed_pair/evaluate_pair.py --raw "$PWD/runs/detect/r18_na_ablation/completed_pair_reproduction" >> logs/r18_completed_pair_eval.log 2>&1'
```

环境为已有 UAV_CV / PyTorch 2.1.2+cu121；SDK v0.2 源码运行的环境限制与此前报告相同。没有新增训练，没有覆盖旧 R1 验收产物。网页端 ChatGPT 分析时优先读取本报告、对照 CSV 和 verification.json，再按需查看旧逐层诊断；请区分内部指标与正式 COCOeval。
