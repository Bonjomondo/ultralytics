# P2-SFI 原始训练曲线与验证图表

分支：`feat/r18-p2-sfi`。C0–C3为已完成150轮；接续训练仍在运行时，CSV明确标记为快照。

CSV为训练目录results.csv的逐字节归档，150轮文件复用已有Git记录。PNG从原始训练产物直接复制，无重绘或压缩改写。
PR/P/R/F1、混淆矩阵是框架原生验证口径，不能与正式COCOeval AP混用；results.png为各轮训练/验证指标。

| 实验 | 阶段轮数 / 累计轮数 | CSV | PR | 训练曲线 | P / R / F1 | 混淆矩阵 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c0 | 150 / 150 | [results.csv](../2026-10-24_r18_p2_sfi_audit/raw/c0_training_curve.csv) | [PR](figures/c0/BoxPR_curve.png) | [results](figures/c0/results.png) | [P](figures/c0/BoxP_curve.png) / [R](figures/c0/BoxR_curve.png) / [F1](figures/c0/BoxF1_curve.png) | [原始](figures/c0/confusion_matrix.png) / [归一化](figures/c0/confusion_matrix_normalized.png) | 已完成 |
| c1 | 150 / 150 | [results.csv](../2026-10-24_r18_p2_sfi_audit/raw/c1_training_curve.csv) | [PR](figures/c1/BoxPR_curve.png) | [results](figures/c1/results.png) | [P](figures/c1/BoxP_curve.png) / [R](figures/c1/BoxR_curve.png) / [F1](figures/c1/BoxF1_curve.png) | [原始](figures/c1/confusion_matrix.png) / [归一化](figures/c1/confusion_matrix_normalized.png) | 已完成 |
| c2 | 150 / 150 | [results.csv](../2026-10-24_r18_p2_sfi_audit/raw/c2_training_curve.csv) | [PR](figures/c2/BoxPR_curve.png) | [results](figures/c2/results.png) | [P](figures/c2/BoxP_curve.png) / [R](figures/c2/BoxR_curve.png) / [F1](figures/c2/BoxF1_curve.png) | [原始](figures/c2/confusion_matrix.png) / [归一化](figures/c2/confusion_matrix_normalized.png) | 已完成 |
| c3 | 150 / 150 | [results.csv](../2026-10-24_r18_p2_sfi_audit/raw/c3_training_curve.csv) | [PR](figures/c3/BoxPR_curve.png) | [results](figures/c3/results.png) | [P](figures/c3/BoxP_curve.png) / [R](figures/c3/BoxR_curve.png) / [F1](figures/c3/BoxF1_curve.png) | [原始](figures/c3/confusion_matrix.png) / [归一化](figures/c3/confusion_matrix_normalized.png) | 已完成 |
| c2_continuation | 350 / 500 | [results.csv](raw/c2_continuation/results.csv) | [PR](figures/c2_continuation/BoxPR_curve.png) | [results](figures/c2_continuation/results.png) | [P](figures/c2_continuation/BoxP_curve.png) / [R](figures/c2_continuation/BoxR_curve.png) / [F1](figures/c2_continuation/BoxF1_curve.png) | [原始](figures/c2_continuation/confusion_matrix.png) / [归一化](figures/c2_continuation/confusion_matrix_normalized.png) | 已完成 |
| c0_continuation | 350 / 500 | [results.csv](raw/c0_continuation/results.csv) | [PR](figures/c0_continuation/BoxPR_curve.png) | [results](figures/c0_continuation/results.png) | [P](figures/c0_continuation/BoxP_curve.png) / [R](figures/c0_continuation/BoxR_curve.png) / [F1](figures/c0_continuation/BoxF1_curve.png) | [原始](figures/c0_continuation/confusion_matrix.png) / [归一化](figures/c0_continuation/confusion_matrix_normalized.png) | 已完成 |

[文件来源、轮次和SHA256](artifacts.json)。每次采集只保留一份当前快照，Git保存已提交历史。
原生PNG通常在训练结束时生成；运行中尚未生成的图表不以旧实验图替代。训练batch拼图、验证样本图、权重、数据集、逐图预测及完整日志只留服务器。

更新归档：`PYTHONPATH=. python docs/experiment_reports/export_p2_sfi_curves.py`。`--watch`每60秒更新一次，接续队列完成或失败后采集最后一次并退出。
自动采集只更新归档文件，不自动提交或推送；新增图表按本目录的Git例外规则纳入后续提交。
