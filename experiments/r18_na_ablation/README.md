# R18 NA ablation 结果索引

**最新完成结果：[R0/R1 同协议正式对照](completed_pair/REPORT.md)。两组均完成 150 轮；R1−R0 AP +0.055pp、APs +0.397pp，仅为单 seed 结果。**

设计与训练命令见 [RT-DETR-R18 P4 NA 实验](../rtdetr_r18_na.md)。本目录只保存适合 Git / 网页端协同分析的结论、配置副本、统计表和复现源码；权重与完整原始轨迹位于 `runs/detect/r18_na_ablation/`。

| 日期 | 实验 | 完成范围 | 结论入口 |
| --- | --- | --- | --- |
| 150轮完成后 | R0/R1，seed 0 | 同 GT / 同原生导出 / 同 COCOeval；配置与初始化核验 | [完成版报告](completed_pair/REPORT.md) |
| 2026-10-21 | R1，150 轮，seed 0，EMA | CUDA / 548 图真实观测、COCOeval、离线重建；8 图探针和冻结 D3 bbox 干预 | [报告](detrscope_20261021/REPORT.md) · [ChatGPT 上下文](detrscope_20261021/CHATGPT_CONTEXT.md) |

早期 R1 验收时 R0 尚未完成 150 轮，状态留在带时间戳的快照中；不能把部分训练结果当作等预算 NA 对照。完成后的同预算结果已另存 completed_pair，保持既有证据可复查。
