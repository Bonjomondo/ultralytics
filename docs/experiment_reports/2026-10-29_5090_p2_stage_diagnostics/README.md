# 三 seed P2 阶段诊断

[完整报告](REPORT.md) · [逐层AP差](paired_stage_deltas.csv) · [覆盖差](paired_coverage_deltas.csv) · [同GT群体追踪](advantage_cohorts.csv) · [资产审计](audit.json)

seed1/2 全量诊断已完成，seed0同协议证据复用；本次无重训。seed1的相对AP优势在D3消失但覆盖仍正，seed2候选改善不稳且D2覆盖转负。六模型D3原始预测与历史评估逐项一致。

离线校验：`python verify.py`。图表：`python plot.py`（需matplotlib）。权重/完整预测/原始记录仍仅保留服务器。
