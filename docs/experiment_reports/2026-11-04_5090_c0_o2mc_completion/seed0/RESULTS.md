# C0 + O2M-C：150轮单 seed 结果

分支 `5090-c2-o2mc-20261031`；基点 `47876f20ce47cd73289a8258b7ab5b8985ba86c8`，实际源码身份见 [training.json](results/training.json)。
完整6471图训练、548图标准COCO评估。精度训练FP32，checkpoint原生保存为FP16 EMA后重载FP32；不是VisDrone官方挑战分数。

| 指标 | C0基线 | C0+O2M-C | 差值pp |
|---|---:|---:|---:|
| AP | 18.661578 | 19.874558 | +1.212980 |
| AP50 | 32.500877 | 34.034764 | +1.533887 |
| AP75 | 18.066354 | 19.685443 | +1.619089 |
| APs | 12.166368 | 12.755094 | +0.588726 |
| APm | 25.960278 | 27.866972 | +1.906694 |
| APl | 29.230754 | 33.328195 | +4.097441 |

训练曲线覆盖 150 轮；最佳内部mAP50–95=0.221270。内部验证用于选择best，表中为best完整正式COCO AP。
收敛检查：`{"best_gain_last50": 0.025609999999999994, "converged": false, "criterion": "after >=150 additional epochs; <=0.1pp best gain over 50 epochs and abs(mean20-previous20)<=0.1pp", "epochs": 150, "mean_gain_last20_vs_previous20": 0.008672999999999959, "tolerance": 0.001}`；完成150轮不自动表示收敛。

[完整训练曲线](results/training_curve.csv) · [每轮资源和健康](results/health.jsonl) · [分配与梯度诊断](results/aux_diagnostics.jsonl) · [分类别和尺度指标](results/evaluation.json)。

该比较只回答seed0、150轮预算下辅助监督是否改善C0。三seed统计见上级目录；交互效应须在四组同协议结果齐全后计算。
训练诊断面积是640输入坐标下的面积；COCO APs按原图标注面积，二者分母不能混用。每轮第三个batch的梯度只是抽样。

权重、完整预测、源码快照与完整日志仅保存在服务器：
`/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/c0_o2mc_5090/5090-c0-o2mc-seeds012-20261104-r1/seed0`。best/last/resume路径、大小及SHA见training.json；部署原生权重为服务器 `evaluation/native-c0.pt`。

设计与复现见 [PLAN.md](PLAN.md)。
