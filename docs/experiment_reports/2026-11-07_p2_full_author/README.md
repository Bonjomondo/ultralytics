# RT-DETR-R18完整P2四尺度实验

本轮在4090上运行B0、C2和P2-Full三组各350轮seed1实验。当前状态见[启动信息](launch.json)、服务器queue_status.json与[结果表](RESULTS.md)。短跑数据用于验证代码链路。

Git分支为 `feat/r18-p2-full-author`，源码位于 `experiments/p2_full_author_20261107`。作者后端复用仓库中 `experiments/c2_channel_author_20261105/backend`，来源为SO_DETR作者提交148d5972bbaec298b3975ca00c00d91abbc6657c的重建RT-DETR-R18实现。启动时将依赖后端与本轮代码一起冻结并逐文件校验。

## 三组结构

| 组别 | P2路径 | Decoder输入stride | 参数量 |
|---|---|---|---:|
| B0 | 原始三尺度neck | 8/16/32 | 20,094,584 |
| C2 | 骨干P2下采样至stride8，注入P3融合路径 | 8/16/32 | 20,135,288 |
| P2-Full | P3语义上采样与P2融合，再自底向上融合P3/P4/P5 | 4/8/16/32 | 22,182,552 |

P2-Full在640输入下有160/80/40/20四个输出。P2 projection输出128通道，融合后的四个尺度输出均256通道。AIFI仍在P5。Decoder为300 queries、3层、8头、每头每尺度4采样点。P2参与初始query选择及解码器cross-attention。

为保留原P3/P4/P5先验，四尺度decoder采用首层grid_size0.025，按层加倍，对应P2/P3/P4/P5归一化wh为0.025/0.05/0.1/0.2。预检断言四尺度去除P2后的anchor及valid mask与原始三尺度逐元素相等。三尺度调用作者原始anchor函数。

四尺度特征位置数34,000，三尺度8,400。位置数增幅不直接代表整网耗时增幅。参数量、FLOPs和实测延迟分别归档。

## 配对与训练

三组均从头训练seed1，保存完整未训练B0状态于服务器。共有骨干逐张量配对，按语义映射复制兼容neck和decoder参数。C2的P2投影及P3融合输入语义发生变化，其相关层独立初始化。Full的新P2分支独立初始化，原三尺度decoder投影按尺度对应到四尺度的后三级；offset与attention的已有尺度行按头/采样点映射。完整复制映射、扩展映射及新层列表见preflight.json及各run/initialization.json。

结构不同，初始输出允许不同。三组首批训练图像与标签哈希要求相同，分别保存损失。模型构造期间同时隔离CPU/CUDA RNG，避免额外层初始化改变训练随机流。

| 配方项 | 生效值 |
|---|---|
| 引擎 | 作者后端Ultralytics8.0.201，ResNet18-vd |
| 数据 | VisDrone train6471 / val548，10类 |
| 预算 | 各350完整轮，patience0 |
| seed / workers | 1 / 0 |
| 输入 / batch / nbs | 640 / 4 / 4，accumulate1 |
| 优化器 / LR / decay | AdamW / 恒定1e-4 / 1e-4 |
| warmup | 参数名warmup_epochs=2000，作者实现语义为2000 batch |
| Mosaic / MixUp | 1.0 / 0.2，close_mosaic0 |
| 精度 / 预训练 | FP32 / 从头训练 |

三组预算从开始就设350轮。150、250、350轮分别保存当轮last和截至该预算原生fitness选出的best，连同完整raw模型、EMA、优化器、scheduler、scaler、stopper和RNG。训练连续完成350轮后评估这些节点，因此150/250轮正式评估结果会在对应结构350轮训练结束后产生。

本队列先完成这三组单seed实验，后续独立消融、其他seed以及Orin部署测试依据结果另行安排。历史seed0的27.77/28.72 AP作为背景，主结构差值使用本轮同条件对照。

## 正式评估与诊断

每个节点的last与best独立加载FP32 EMA，作者RTDETRValidator输出548图、每图300个argmax query，无NMS、坐标不取整。保留完整164400条预测并校验GT/权重/预测SHA。GT固定为converted VisDrone val，SHA为6f57aa4b42d64ee7b99b8a6a5b159ad19cba2eacf00bd86f23314966614bf4d8。

独立CPU进程运行pycocotools.COCOeval，主表maxDets=[1,10,100]，补充maxDets=[1,10,300]。maxDets300的AP从precision数组明确计算，避免官方summarize对AP默认maxDets100的硬编码；补充召回名称使用AR300。原生mAP及原生曲线单独标注。

Tiny recall按作者验证实际使用的640×640方形缩放，取缩放后目标框短边分为[0,8)、[8,16)、[16,32)、[32,inf)。基于COCO逐类别、逐图的一对一匹配，统计IoU0.50与0.75召回，忽略GT按COCO规则排除。此项为补充召回诊断，保留各桶GT数量。

为每条预测保留初始query来源stride。预测按最终分类分数重排时同步重排来源标签，并在tiny recall中汇总匹配框的初始来源。来源反映query初始化，后续跨尺度注意力会混合信息，不能把它解释成某尺度的唯一因果贡献。

训练每轮记录前三批次的query来源计数、P2投影裁剪前梯度范数、原生指标、耗时和峰值显存。云端保留逐轮摘要CSV，完整JSONL留服务器并归档SHA。

最终同进程随机交错5轮，每模型500个计时样本，640/batch1、fused FP32、GPU常驻输入、CUDA event同步。matmul TF32关闭，cuDNN TF32允许。计时范围为网络前向，Orin性能需另行实测。

## 验证与恢复

- [结构、初始化、640前向反向与锚框验证](preflight.json)
- [三组真实短跑、预测、独立COCO及测速](smoke_verification.json)
- [两轮短跑在第一轮停止后恢复至第二轮](resume_verification.json)
- [实际milestone保存路径及last/best哈希核验](milestone_saving_verification.json)
- [密集batch4前向反向显存验证](dense_memory_probe.json)
- [Tiny分桶、匹配、query标签与maxDet300合成测试](tiny_recall_verification.json)
- [实际实验规格](experiment_spec.json)、[数据身份](dataset_identity.json)

Full的密集样本测试选取训练集中标注最多的四张图，原图目标数902/511/462/431，增强后共4206个目标。前向反向通过，峰值分配显存约10.92GiB。这是单批容量验证。

启动器检查源码已提交、GPU空闲、无重复tmux后冻结脚本与已追踪后端。运行目录为 `/home/zjnu/文档/ZHJ_WorkSpace/ultralytics/runs/p2_full_author_20261107`，tmux为 `p2-full-author-20261107`。各阶段核验冻结manifest，完成阶段通过done列表去重。

训练每阶段累计最多两次尝试，失败后先备份last/best与状态，再从完整raw检查点恢复。恢复时核对raw、EMA和优化器张量，并恢复scheduler/scaler/stopper/RNG。采样器重建可能改变数据流，任何恢复须在结果解释中披露。评估和测速最多三次尝试。超过30分钟无日志/心跳推进视为故障，按同一有界策略处理。此前服务器随机原生崩溃原因仍待确认，恢复验证仅证明状态恢复链路。

控制器每25轮保存运行中完整CSV快照，逐组训练和评估完成后归档曲线、PR/P/R/F1和混淆矩阵。服务器保留权重、完整预测、完整日志、源码冻结副本。Git保存脚本、配置、完整训练CSV、诊断摘要、指标、图表和SHA清单。服务器无GitHub认证，提交通过本机认证同步推送。
