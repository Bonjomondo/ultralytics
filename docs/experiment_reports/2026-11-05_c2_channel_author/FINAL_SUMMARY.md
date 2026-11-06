# C2 / Scalar / ChannelGate 最终筛选结果

2026-11-07已完成三组各150轮seed0训练、独立COCOeval及同场测速。ChannelGate未通过预设晋级条件，本次实验按方案完成。Git仓库为Bonjomondo/ultralytics，分支feat/r18-c2-channelgate-author。

## 正式COCO结果

同一converted VisDrone val548、maxDets=[1,10,100]。以下精度均为百分数，差值为百分点。

| 结构 | AP | APs | AP75 | APm | APl | ΔAP vs C2 | GPU中位ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| c2 | 25.606 | 17.322 | 25.840 | 35.294 | 41.406 | 0.000 | 7.133 |
| scalar | 26.224 | 17.758 | 26.608 | 36.284 | 42.256 | 0.618 | 7.140 |
| channel | 25.954 | 17.469 | 26.563 | 36.293 | 39.639 | 0.348 | 7.146 |

Scalar在本轮AP最高，相对C2增加0.618pp、APs增加0.436pp。Channel相对C2的AP增加0.348pp，但APs只增加0.147pp，未达到0.3pp门槛，APl下降1.767pp，超过0.5pp容许降幅。Channel相对Scalar的AP低0.270pp、APs低0.289pp、APl低2.616pp。精确决定使用未四舍五入数值，见[decision.json](decision.json)。

## 训练预算、恢复与解释边界

三组均从同一全新C2初始化开始，seed0、150轮、patience0、workers0、batch/nbs4、640、FP32训练、AdamW恒定1e-4。共有参数哈希及第一批图像、标签、损失完全一致。Scalar新增1个参数，Channel新增128个参数，门控倍率为2*sigmoid(w)，初始为1。两种门控均为跨样本、空间共享的静态参数。

C2发生两次训练中断，分别从完整第50轮和第98轮raw模型、EMA及优化器恢复。恢复张量校验通过，采样与增强序列可能改变。Scalar与Channel均在一次正式训练尝试内完成150轮。本轮结论限于单seed、150轮筛选，C2的恢复轨迹也限制了严格配对解释。历史350轮AP27.77/28.72属于不同训练预算。本次没有触发条件内的350轮多seed阶段。此前随机崩溃根因仍待确认。

归档曾因Matplotlib生成SVG的行尾空格触发Git检查而停止，修复只涉及本报告生成SVG的Git格式规则。见[恢复授权记录](user_resume_20261106.json)、[第一次恢复审计](raw/screen_c2_150e_s0/recovery_audit.json)、[第98轮恢复校验](raw/screen_c2_150e_s0/restore_epoch98.json)、[归档修复](archive_repair_20261106.json)。

## 原生mAP与测速

| 结构 | 原生mAP50-95 % | 正式COCO AP |
|---|---:|---:|
| c2 | 28.026 | 25.606 |
| scalar | 28.656 | 26.224 |
| channel | 28.568 | 25.954 |

原生mAP来自作者Validator，对应best checkpoint选择及原生曲线。正式AP来自独立CPU COCOeval，使用同一checkpoint输出的548图、每图300条query预测，再按COCO maxDets100汇总。上述两列采用不同评估口径。

测速为同一4090、同进程、随机交错5轮，每结构500个计时样本，640/batch1、fused、GPU常驻输入、CUDA event同步。权重为FP32，matmul TF32关闭，cuDNN TF32允许。耗时范围为网络前向。各组中位时间变化均小于0.2%，不作微小差异显著性结论。

## Alpha与可复查证据

| 结构 | 最后轮alpha均值 | 标准差 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| c2 | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| scalar | 1.139974 | 0.000000 | 1.139974 | 1.139974 |
| channel | 1.043079 | 0.043165 | 0.975229 | 1.154868 |

各组保留150轮完整CSV及128列Alpha。C2和Scalar在Alpha CSV中广播到128列，实际可学习门控参数分别为0和1。Channel的128列对应128个独立可学习倍率。每轮前三批次的特征/梯度诊断来自服务器完整diagnostics.jsonl，云端保留逐轮摘要及源文件SHA。

- [最终逐项审计及GT/权重/预测/曲线/benchmark SHA](final_audit.json)
- [完整正式对照表](comparison.csv)
- [原始逐次计时](benchmark_screen.json)
- [队列完成记录](final_queue_status.json)
- c2的[完整训练曲线CSV](raw/screen_c2_150e_s0/results.csv)、[Alpha CSV](raw/screen_c2_150e_s0/alpha.csv)、[诊断摘要](raw/screen_c2_150e_s0/diagnostics.csv)、[正式指标](raw/screen_c2_150e_s0/metrics.json)、[原生曲线图](figures/screen_c2_150e_s0/results.png)、[Alpha图](figures/screen_c2_150e_s0/alpha_history.png)、[PR图](figures/screen_c2_150e_s0/PR_curve.png)、[混淆矩阵](figures/screen_c2_150e_s0/confusion_matrix.png)。
- scalar的[完整训练曲线CSV](raw/screen_scalar_150e_s0/results.csv)、[Alpha CSV](raw/screen_scalar_150e_s0/alpha.csv)、[诊断摘要](raw/screen_scalar_150e_s0/diagnostics.csv)、[正式指标](raw/screen_scalar_150e_s0/metrics.json)、[原生曲线图](figures/screen_scalar_150e_s0/results.png)、[Alpha图](figures/screen_scalar_150e_s0/alpha_history.png)、[PR图](figures/screen_scalar_150e_s0/PR_curve.png)、[混淆矩阵](figures/screen_scalar_150e_s0/confusion_matrix.png)。
- channel的[完整训练曲线CSV](raw/screen_channel_150e_s0/results.csv)、[Alpha CSV](raw/screen_channel_150e_s0/alpha.csv)、[诊断摘要](raw/screen_channel_150e_s0/diagnostics.csv)、[正式指标](raw/screen_channel_150e_s0/metrics.json)、[原生曲线图](figures/screen_channel_150e_s0/results.png)、[Alpha图](figures/screen_channel_150e_s0/alpha_history.png)、[PR图](figures/screen_channel_150e_s0/PR_curve.png)、[混淆矩阵](figures/screen_channel_150e_s0/confusion_matrix.png)。

本目录MANIFEST.sha256覆盖随仓库提供的报告和数据。完整权重、完整逐图预测、训练日志、diagnostics.jsonl与源码冻结快照保存在服务器运行目录 /home/zjnu/文档/ZHJ_WorkSpace/ultralytics/runs/c2_channel_author_20261105，各大文件的路径和SHA由审计与元数据引用。
