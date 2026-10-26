# 5090 P2＋P5-LA 实验设计

分支`5090-detrscope-query-budget`，设计起点`11c5346`。状态：实现和CPU验证；**没有新组合的5090训练、COCO AP或CUDA计时结果**。

[完整实验方案与复现命令](../../../experiments/p2_p5_la_5090/README.md) · [组合模型配置](../../../ultralytics/cfg/models/rt-detr/rtdetr-resnet18-p2-sfi-c2-p5-naifi.yaml) · [验证记录](VALIDATION.md)

四组固定300Q/4采样点/3层Decoder，比较C0、C2、P5-NA7/RPB、C2＋P5-NA7/RPB。四组共有参数、两组P2及两组P5-LA各自逐位配对；全新150轮seed0筛选，再按结果进行多seed/长预算复验。结构与训练后测速均分别测P5、Neck/Encoder、无标记整网，防止历史4090速度与5090新精度混算。

本提交只包含配置、执行代码、设计文档和CPU验证证据。数据集、训练权重、预测及导出模型没有随本设计提交。
