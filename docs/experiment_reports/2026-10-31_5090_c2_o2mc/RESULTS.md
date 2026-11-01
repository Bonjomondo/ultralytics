# C2 + O2M-C：150轮单 seed 结果

分支 `5090-c2-o2mc-20261031`；基点 `f7038be175fab17d9f82ca021500a9b6a6fbd0d7`，实际源码身份见 [training.json](results/training.json)。
完整6471图训练、548图标准COCO评估。精度训练FP32，checkpoint原生保存为FP16 EMA后重载FP32；不是VisDrone官方挑战分数。

| 指标 | C2基线 | C2+O2M-C | 差值pp |
|---|---:|---:|---:|
| AP | 19.516709 | 20.630194 | +1.113486 |
| AP50 | 33.494803 | 34.960458 | +1.465654 |
| AP75 | 19.147566 | 20.288813 | +1.141246 |
| APs | 12.652582 | 13.510098 | +0.857516 |
| APm | 26.698442 | 29.151891 | +2.453449 |
| APl | 34.665103 | 30.618709 | -4.046394 |

训练曲线覆盖 150 轮；最佳内部mAP50–95=0.227550。内部验证用于选择best，表中为best完整正式COCO AP。
收敛检查：`{"best_gain_last50": 0.02779999999999999, "converged": false, "criterion": "after >=150 additional epochs; <=0.1pp best gain over 50 epochs and abs(mean20-previous20)<=0.1pp", "epochs": 150, "mean_gain_last20_vs_previous20": 0.00808600000000001, "tolerance": 0.001}`；完成150轮不自动表示收敛。

[完整训练曲线](results/training_curve.csv) · [每轮资源和健康](results/health.jsonl) · [分配与梯度诊断](results/aux_diagnostics.jsonl) · [分类别和尺度指标](results/evaluation.json)。

该比较只回答seed0、150轮预算下辅助监督是否改善C2。不能推断多seed稳定性，未做C0-Aux，不能宣称C2与辅助监督协同。
训练诊断面积是640输入坐标下的面积；COCO APs按原图标注面积，二者分母不能混用。每轮第三个batch的梯度只是抽样。

权重、完整预测、源码快照与完整日志仅保存在服务器：
`/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/c2_o2mc_5090/5090-c2-o2mc-20261031-s0-r2`。best/last/resume路径、大小及SHA见training.json；部署原生权重为服务器 `evaluation/native-c2.pt`。

设计与复现见 [PLAN.md](PLAN.md)。
