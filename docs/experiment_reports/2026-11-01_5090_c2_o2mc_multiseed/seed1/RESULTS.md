# C2 + O2M-C：150轮单 seed 结果

分支 `5090-c2-o2mc-20261031`；基点 `f7038be175fab17d9f82ca021500a9b6a6fbd0d7`，实际源码身份见 [training.json](results/training.json)。
完整6471图训练、548图标准COCO评估。精度训练FP32，checkpoint原生保存为FP16 EMA后重载FP32；不是VisDrone官方挑战分数。

| 指标 | C2基线 | C2+O2M-C | 差值pp |
|---|---:|---:|---:|
| AP | 18.828508 | 20.168050 | +1.339542 |
| AP50 | 32.698191 | 34.446420 | +1.748229 |
| AP75 | 18.331520 | 19.988529 | +1.657009 |
| APs | 12.157114 | 12.940761 | +0.783646 |
| APm | 26.622759 | 28.321076 | +1.698318 |
| APl | 32.460732 | 35.251625 | +2.790893 |

训练曲线覆盖 150 轮；最佳内部mAP50–95=0.221720。内部验证用于选择best，表中为best完整正式COCO AP。
收敛检查：`{"best_gain_last50": 0.026240000000000013, "converged": false, "criterion": "after >=150 additional epochs; <=0.1pp best gain over 50 epochs and abs(mean20-previous20)<=0.1pp", "epochs": 150, "mean_gain_last20_vs_previous20": 0.009246000000000087, "tolerance": 0.001}`；完成150轮不自动表示收敛。

[完整训练曲线](results/training_curve.csv) · [每轮资源和健康](results/health.jsonl) · [分配与梯度诊断](results/aux_diagnostics.jsonl) · [分类别和尺度指标](results/evaluation.json)。

该比较只回答seed1、150轮预算下辅助监督是否改善C2。三seed统计见上级目录；未做C0-Aux，不能宣称C2与辅助监督协同。
训练诊断面积是640输入坐标下的面积；COCO APs按原图标注面积，二者分母不能混用。每轮第三个batch的梯度只是抽样。

权重、完整预测、源码快照与完整日志仅保存在服务器：
`/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/c2_o2mc_5090/5090-c2-o2mc-multiseed-20261101-r1/seed1`。best/last/resume路径、大小及SHA见training.json；部署原生权重为服务器 `evaluation/native-c2.pt`。

设计与复现见 [PLAN.md](PLAN.md)。
