> 2026-10-29更新：seed2已完成，三对最终AP差为+0.855/−0.054/−0.187pp；完整跨seed阶段诊断见[新报告](../2026-10-29_5090_p2_stage_diagnostics/REPORT.md)。本目录10月28日的启动/seed1快照保留历史身份。

# P2 多 seed 验证、同 GT 诊断和 D2 早退部署验证

分支 `5090-detrscope-query-budget`，起点 `8584f1e4`。更新于2026-10-28：本目录记录seed1最终结果、既有同GT/逐层诊断和D2部署验证。[执行计划](PLAN.md) · [完整结果分析](REPORT.md) · [运行入口](../../../experiments/p2_stability_5090/README.md)。

C0/C2 seed1从零150轮及完整548图正式评估已于10月28日10:52结束。seed1 AP为C0 18.882105、C2 18.828508，差−0.053597pp；连同seed0，两对差值均值+0.400767pp、样本标准差0.642568pp，尚未通过跨seed稳定性门槛。APs两对均正。用户授权的seed2队列已于10月28日12:36启动，C2保留候选身份。[seed1最终指标](seed1_results/formal_metrics.csv)、[配对统计](seed1_results/paired_statistics.csv)、[seed2启动核验](raw/seed2_launch_preflight_20261028.json)。

本目录的 `raw/launch_status.json`、`raw/training_status.json` 是历史启动快照；`seed1_results/status.json`为本次完成快照，服务器任务 `runs/p2_stability_5090/5090-p2-stability-20261027-r1/status.json` 为实时状态。未完成seed的AP不填写为结果。

紧凑证据：`seed1_results/`含正式精度、分类别尺度指标、完整训练曲线/逐轮健康、训练/评估receipt与只读核验结果；`same_gt/`和`deployment/`为既有seed0结果，`raw/`保留启动证据。GT、完整预测、权重、TensorRT引擎、完整日志保留在服务器，来源路径和SHA见对应receipt。`MANIFEST.sha256`仅覆盖本目录归档文件，可用 `sha256sum -c MANIFEST.sha256` 校验；`verify.py`可脱离服务器数据和GPU核验归档。
