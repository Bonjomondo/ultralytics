# R18 消融实验汇报目录

R0、R1、R2a、R2b、R2c 的训练、正式精度与 FP32/FP16 推理性能汇总。R2 队列于 2026-10-23 03:15:41（北京时间）完成。

- [完整中文报告](2026-10-23_实验结果-rtdetr-r18-report.md)：结果、分析、测试边界、证据和复现命令。
- [FP32 主对照表](tables/summary_fp32.csv)、[全部结果](tables/all_results.csv)、[正式精度](tables/accuracy_metrics.csv)、[相对 R0 变化](tables/deltas_vs_r0_fp32.csv)。
- [本次基准原始数据](raw/inference_20261023/benchmark.json)、[测试协议](raw/inference_20261023/protocol.json)、[运行日志](raw/benchmark_20261023.log)。
- [基准脚本](scripts/benchmark_inference.py)、[报告生成与校验脚本](scripts/build_report.py)。
- `raw/accuracy/`、`raw/training/`：精度与训练记录快照；`raw/history/`：之前批次的紧凑摘要，历史逐次原始 JSON 仅保留于服务器。
- `sources/`：模型配置与相关实现快照；`provenance/`：范围、验证、原始文件位置及哈希；`MANIFEST.sha256`：归档文件校验清单。

主性能对照只使用此次同批次重测；所有 AP 来自 FP32 正式评估，FP16 仅测性能。原权重与训练目录保持原位置，通过报告中的路径和哈希引用。
