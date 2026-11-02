# 5090 RT-DETR-R18 C2 + O2M-C

2026-10-31，Asia/Shanghai；工作分支 `5090-c2-o2mc-20261031`。本轮只运行C2加CNN辅助监督，单seed0、150轮从零筛选。已有C2多seed结果有波动，不预设辅助监督有效。

<!-- RUN_STATUS_BEGIN -->
150轮训练和完整正式评估已完成，见 [最终结果](RESULTS.md)。
<!-- RUN_STATUS_END -->

- [实验设计、冻结协议及复现](PLAN.md)
- [实现与命令](../../../../experiments/c2_o2mc_5090/README.md)
- [本地官方RT-DETRv3参考文件及SHA](../../../../experiments/c2_o2mc_5090/UPSTREAM.md)
- [启动环境、配方、基线权重SHA](raw/launch.json) · [实际训练入口与初始化](raw/training_start.json) · [源码SHA清单](raw/source_sha256.json)
- [全数据集审计](raw/asset_audit.json) · [CPU门禁日志](raw/cpu_gates.txt) · [GPU smoke训练与checkpoint](raw/smoke_training.json) · [8图评估及去头逐位等价](raw/smoke_evaluation.json)
- [smoke分配与梯度证据](raw/smoke_aux_diagnostics.jsonl) · [两次工程失败与处理边界](raw/engineering_failures.json) · [实际launch.sh](raw/launch.sh) · [启动阶段快照](raw/launch_stages.json)
- [首轮健康、梯度及best/last快照SHA](raw/first_epoch.json)：训练160.45秒，训练峰值allocated显存9087MiB，首轮曲线及诊断完整。best/last仍随训练更新，此处SHA只标识首轮快照。

主路径参数20,135,032，训练辅助头参数1,330,410。eval/导出完全跳过辅助计算；去头后的原生模型参数与初始C2同量。辅助头零初始化导致首步特征梯度为0，后续步骤已核验P2梯度连通。smoke仅65图/1轮，8图AP为0；这是保存/评估工程门禁，不是精度筛选结论。

服务器运行根目录为 `/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/c2_o2mc_5090/5090-c2-o2mc-20261031-s0-r2/`。实时进度读取其中 `status.json` 和 `training/5090-c2-o2mc-150e-s0/experiment.json`，日志为 `logs/full-train.log`。权重、完整逐图预测、完整日志、源码快照只留服务器。报告中的小型JSON/CSV/JSONL为云端可读证据；完整评估后自动生成 `RESULTS.md` 与 `results/`，不自动提交或推送Git。
