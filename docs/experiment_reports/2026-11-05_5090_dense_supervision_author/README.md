# 5090 作者配方层次化密集监督：阶段A

新分支`5090-rtdetrv3-dense-supervision-20261105`，先seed0，额外seed最后补。旧协议结果已在原分支完成归档。

本队列进行工程门禁、两组小样本验收、C2前150轮最佳权重复评，并补跑C0/C0+O2M-C/C2+O2M-C三组150轮。全部完整548图正式评估后形成新配方2×2汇总。当前不是最终精度结论。

[冻结计划、配置和复现入口](../../../experiments/rtdetrv3_dense_supervision_5090/README.md) · [数据迁移核验](../../../experiments/rtdetrv3_dense_supervision_5090/reference/data_verification.json) · [C2复用检查点来源](../../../experiments/rtdetrv3_dense_supervision_5090/reference/c2_checkpoint_origin.json)。

当前执行边界为阶段A的150轮筛选与归档；350轮扩展、O2M-T、MGSA和组合由结果决定，尚未接入自动队列。新研究alpha1、batch4/nbs4、作者GT与引擎；旧150轮alpha0.5结果不能直接并入此表。

2026-11-05启动验收：两种结构的真实模型门禁、两组32图训练/16图保存重载及COCOeval全部通过。辅助头移除前后预测逐位一致。C2前150轮内最佳（epoch145）已在5090完成548图复评，AP26.085202、APs18.036855、AR100 35.780225；这是150轮预算对照，不是350轮AP28.72。C0补充150轮已启动，后续依次自动运行两组alpha1辅助实验。

[启动验收证据](preflight/completed_stage_receipts.json) · [真实模型门禁](preflight/gates.json) · [冻结源码SHA](preflight/source_pin.json) · [复用C2完整结果](a1_c2_reused/coco_metrics.json)。辅助头来源为原`experiments/c2_o2mc_5090/model.py`的PyTorch适配（参考RT-DETRv3的PP-YOLOE头与分配器）；本实验未实现其余两种机制。
