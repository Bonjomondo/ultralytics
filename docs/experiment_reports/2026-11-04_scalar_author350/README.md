# C2标量门控：作者引擎350轮与RT-DETR基线对照

任务分支：`feat/r18-c2-gated-fusion`。将150轮筛选中表现最好的标量门控迁入SO-DETR作者Ultralytics 8.0.201引擎，从头训练，比较历史B0及C2。

## 对照关系

| 模型 | 配方 | 已有COCO AP | 作用 |
|---|---|---:|---|
| B0：原RT-DETR-R18 | 作者引擎，350轮，seed0 | 27.773 | 用户指定的论文重建基线 |
| C2：P2注入 | 同一作者引擎和训练配方 | 28.722 | 区分P2注入与新增门控的收益 |
| C2＋Scalar | 同一作者引擎和训练配方，新训练 | 待完成 | 本次实验 |

本次不沿用现代引擎150轮的19.42 AP或其已训练权重。只保留结构假设：第18层将P2下采样分支乘以 `2*sigmoid(logit)`，再按原顺序 `[P4up,P2down,P3]` Concat；其余结构、decoder和损失均不变。logit=0，初始倍率=1，新增1个参数，总参数20,135,289。

## 训练与验证

350轮上限、patience40，与历史B0/C2一致；如触发早停，报告实际轮数，不称固定350轮已完成。640、batch4、nbs4、accumulate1、FP32、AdamW、恒定LR1e-4、weight_decay1e-4、mixup0.2、mosaic1、workers8、seed0、无预训练。旧引擎参数名warmup_epochs=2000在该作者实现中表示2000个warmup batch，保留原语义。

初始化重放历史构造器，公共527项张量哈希必须匹配B1的初始化receipt。门控不消耗额外随机初始化，P2使用原C2构造路径。历史B0/B1未保存完整初始模型，因此不能声称与历史训练逐位等价；CUDA grid_sample反向也不保证确定性。

采用独立目录，不改历史B0/C2源码、日志或权重：

```text
/home/zjnu/文档/ZHJ_WorkSpace/reproductions/c2-scalar-author-20261104
```

源码树由已有作者引擎复制，新增门控模块并修改parser；训练数学不变，checkpoint保存改为FP32 CPU暂存、标准库pickle、fsync、重新加载校验及原子替换，避免旧dill默认路径。保留raw模型、EMA、优化器、scaler、scheduler、stopper及RNG；当前入口拒绝默认EMA续训，若中断须审计raw-state恢复。

训练后自动导出548图FP32完整300-query预测，单独CPU进程COCOeval，主表maxDets100，补充maxDets300和类别/尺度指标。评估数据为转换VisDrone val，不等于官方挑战赛协议。预测workers0是评估执行调整；训练保留workers8。

将B0、C2及本次最佳权重同进程随机交错测速：640、batch1、fused FP32、CUDA events、5×100次。报告参数量、逐次耗时、median/p95；不含传输/预后处理，不外推Orin/TensorRT。

## 执行与证据

本目录实验说明配套源码在 `experiments/c2_scalar_author_20261104/`；`prepare.py`依赖服务器现有历史C2目录创建独立运行环境，拒绝覆盖既有目录。

```bash
cd /home/zjnu/文档/ZHJ_WorkSpace/ultralytics
PYTHONPATH=. /home/zjnu/anaconda3/envs/UAV_CV/bin/python experiments/c2_scalar_author_20261104/prepare.py
cd /home/zjnu/文档/ZHJ_WorkSpace/reproductions/c2-scalar-author-20261104
PYTHONPATH=. /home/zjnu/anaconda3/envs/UAV_CV/bin/python preflight.py
PYTHONPATH=. /home/zjnu/anaconda3/envs/UAV_CV/bin/python train_scalar.py --smoke
```

首次启动前必须完成预测/独立评估/测速短跑并生成smoke_verification.json，再从仓库运行 `experiments/c2_scalar_author_20261104/start.py`。该脚本检查空闲GPU、源码哈希和重复任务，通过tmux `c2-scalar-author-20261104`启动控制器。

证据：[迁移与初始化检查](preflight.json)、[真实数据短跑](smoke_verification.json)、[配方](experiment_spec.json)、[启动信息](launch.json)。短跑只验证运行链路，不提供精度结论。最终比较写入RESULTS.md、comparison.csv，分别报告scalar−B0和scalar−C2。

完整权重、日志、逐图预测与作者源码副本仅保存在服务器；Git归档必要源码补丁、配置、指标、曲线与紧凑证据。最终报告由控制器自动保存，Git同步可能需要本机认证。

## 第130轮后的恢复记录

第131轮训练时DataLoader worker SIGSEGV，保留完整130轮FP32检查点。已验证raw模型、EMA及优化器逐张量恢复，scaler/stopper/RNG恢复；训练workers8改0后继续第131轮至原350轮上限/patience40。随机增强与加载顺序会随worker变化，不能称逐位等价连续训练。恢复检查点、旧状态与原日志均保留。详见recovery_receipt.json和recovery_verification.json；恢复源码recover_worker0.py独立于原冻结源码。

## 最终结果

350轮训练和548图预测完成；独立COCO评估第一次SIGSEGV后以同一GT/预测重跑成功，三组同场测速完成。完整结果见[RESULTS.md](RESULTS.md)与[comparison.csv](comparison.csv)。

Scalar AP=28.038、APs=19.840；对B0 AP+0.265pp、APs+0.359pp；对C2 AP−0.684pp、APs−0.135pp。本次没有体现新增门控相对C2的精度收益。单seed、worker恢复和checkpoint精度差异限制见final_result.json。训练预算完成，不能直接宣称充分收敛。

完整350轮原始曲线见raw/scalar/results.csv；figures/为框架原生曲线，不能替代正式COCO AP。CPU评估故障栈见coco_failure_trace.txt，重跑日志见coco_retry_evidence.txt；没有重新训练或重生成预测。
