# 四组从零重训：实验身份与启动验收

2026-10-25（Asia/Shanghai），分支 `5090-detrscope-query-budget`。本报告记录启动验收，不把正在运行的任务写成已完成训练。

用户要求将先前速度筛选中的四组全部独立重新训练：300Q/每尺度2点/3层、200Q/每尺度2点/3层、300Q/每尺度4点/2层、300Q/每尺度4点/1层。四组均基于C2，完整预算各150轮、seed0。D1/D2是真正的1/2层架构，新模型没有后续Decoder与预测头。

每组由YAML重新构造，`pretrained=False`、`resume=False`。随机共用初值由内存中的全新C2参考模型生成；未从旧训练checkpoint读取参数，也不从1轮门禁权重继续。新采样维度保留原生零offset/attention矩阵与环形bias初始化。共用参数逐字节相同，四组实际trainer模型的完整初值SHA均与预先计划一致。

训练配方保持原5090基线：FP32、AMP关闭、batch8、640、AdamW、lr0=1e-4、lrf=1、weight_decay=1e-4、warmup3、warmup_bias_lr=0、nbs64、mixup0.2、workers0、patience0。分支构建seed=1000，延续原C2规则；CPU/CUDA实际初始seed另外记录，不承诺跨设备逐值训练确定性。

数据为VisDrone训练6471张/343200框、验证548张/38759框。新审计通过，与原基线标签内容SHA、图片文件名/大小清单以及正式COCO标注SHA相同。图片库存摘要不等同于逐图片像素哈希。标注SHA为 `6f8b8686864f1485ab172de9339f7049593f74766f6654c6e980a6c00d810371`。

四组均通过：

- 原生模型结构与query/point投影维度检查，真实loss/backward、P2与offset非零梯度。
- 1轮真实数据门禁，fraction=.01，batch8 FP32；内部验证仍覆盖548张。
- best/last/resume checkpoint保存、8张真实图重载前向与COCO导出，重复前向逐值一致，模型state不变。
- 实际trainer完整随机初值与初始化计划一致，四组共用参数SHA一致。
- 10项专项测试；包括拒绝旧权重初始化、已有run防覆盖、单组失败仍继续其他请求组，以及源码变化全局停止。

单层健康检查使用Encoder框作为前态，避免原先“最后层减倒数第二层”的检查越界。这只改变诊断采样，没有改训练损失或网络数学结构。

完整四组已进入 `5090-c2-fresh-four-150e-20261025-s0-r1` tmux会话串行运行。单卡不并行抢占。每组完成150轮后，自动对新best.pt跑完整548图正式未融合FP32 COCOeval及Eager FP32/TensorRT FP32/FP16部署计时。正式AP、APs与训练内部mAP分开，FP16性能不代表已通过FP16精度验收。150轮完成不自动等同于收敛。

门禁指标仅作工程验收，完整训练尚未完成，当前没有可以代替最终结果的四组正式AP表。当前训练进度及完整曲线在本地每轮自动更新；网页端Git内容是最近一次推送的快照。若单组训练或评估失败，会保留记录并继续其余组；冻结源码/输入改变则停止整个队列。

[初始化与参数量](raw/initialization_plan.json) · [真实门禁与梯度](raw/gates_summary.json) · [数据审计](raw/asset_audit.json) · [测试结果](raw/validation.json) · [首次完整训练状态](raw/first_training_status.json) · [队列启动快照](raw/launch_status.json)。这些文本是云端提供的证据；JSON内的checkpoint路径只引用服务器资产。

[实时本地进度/离线曲线快照](../../../reports/5090/fresh_budgets/experiments.md) · [复现命令、路径与边界](README.md)。权重、原始图像、完整预测、导出引擎、环境和完整日志继续保留在服务器，未纳入本次Git归档。
