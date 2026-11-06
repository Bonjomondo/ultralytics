# C2可学习通道门控：作者引擎配对实验

**2026-11-07全部完成：三组各150轮、正式COCO评估和同场测速完成。AP为C2 25.606、Scalar 26.224、Channel 25.954。Channel未通过预设晋级条件，实验按方案结束。见[最终报告](FINAL_SUMMARY.md)和[最终审计](final_audit.json)。**

状态以 `launch.json`、服务器 `queue_status.json` 和本目录RESULTS.md为准。短跑只验证代码链路，不是检测性能证据。

Git分支：`feat/r18-c2-channelgate-author`。实验源码及完整作者后端位于 `experiments/c2_channel_author_20261105/`；不再依赖服务器未入Git的源码副本。作者后端约1.3 MB，由原27.77/28.72 AP实验使用的Ultralytics 8.0.201适配实现复制；来源、原始逐文件哈希及修改范围见backend/PROVENANCE.md、author_backend_original_manifest.json。原始作者仓库提交为148d5972bbaec298b3975ca00c00d91abbc6657c。

## 第一阶段：三组150轮严格配对

| 组别 | C2第18层操作 | 新增参数 | 初始倍率 |
|---|---|---:|---|
| C2 | 原始Concat | 0 | 1 |
| Scalar | P2的128个通道共享 `2*sigmoid(w)` | 1 | 1 |
| ChannelGate | P2各通道独立 `2*sigmoid(w_c)` | 128 | 全部1 |

只改变下采样P2的倍率，保持 `[P4_up,P2_down,P3]` 的Concat顺序与后续RepC3不变。ChannelGate是跨样本、跨空间共享的静态通道参数，不是SE/CBAM式动态注意力。后续可学习卷积可吸收这种缩放，因此研究假设是优化/参数化收益，不预设表达能力或精度必然提升。

三组使用同一份全新随机C2状态，逐张量复制全部533项共有参数，包括P2分支；只额外初始化零门控logits。种子0/1/2的完整未训练C2状态另存在服务器。预检验证640整网输出逐元素相同；真实短跑还验证第一批图像、标签和损失完全相同，而非只检查权重名称。

训练引擎严格为作者后端：ResNet18-vd、普通IoU质量目标/GIoU回归、AIFI、三尺度decoder、300 queries、三层decoder。不是完整SO-DETR增强模型，也不是现代8.4.30引擎。

| 设置 | 生效值 |
|---|---|
| 数据 | 同一VisDrone train6471/val548、10类 |
| 输入 / batch / nbs | 640 / 4 / 4 |
| 梯度累积 | 固定1，每个微批次更新一次 |
| 优化器 / LR / decay | AdamW / 恒定1e-4 / 1e-4 |
| Warmup | 作者参数名warmup_epochs=2000，实际语义为2000 batch |
| Mosaic / MixUp | 1.0 / 0.2，close_mosaic=0 |
| 精度 / 初始化 | FP32、无预训练、seed0 |
| 预算 / 早停 | 每组完整150轮，patience=0关闭早停 |
| workers | 三组统一0，从开始即一致，规避已发生的worker SIGSEGV |

workers0与历史B0/C2的workers8存在执行差异，历史AP27.77/28.72仅作背景。本轮比较使用重新训练的同条件C2，不混用历史指标。实际 `args.yaml` 会随每组完整归档；脚本检查batch4、FP32和accumulate1，不能静默改配方。

## 诊断与评估

每轮保存全部128个Alpha（Scalar/C2在CSV中广播并明确标注）、均值、标准差、最小/最大值、低于0.9/高于1.1的通道数和接近0/2的饱和数量。训练前三批次记录门控/P2梯度范数（裁剪前FP32）及P2门控前后特征RMS。完整Alpha变化写入alpha.csv并绘制PNG/SVG；不把通道编号直接解释为确定语义。

原生验证mAP仅用于训练和best选择。正式评估重新加载完整FP32 EMA，以作者Validator输出548图、每图300个argmax query、无NMS、未四舍五入坐标，断言完整164400条预测覆盖。单独CPU进程运行未改动的pycocotools.COCOeval，maxDets=[1,10,100]，报告AP/AP50/AP75/APs/APm/APl及类别指标。GT SHA固定为 `6f57aa4b42d64ee7b99b8a6a5b159ad19cba2eacf00bd86f23314966614bf4d8`。这是转换VisDrone GT，不是官方挑战赛评测。

三组完成后同进程随机交错5轮，每模型每轮100次，50次初始预热。640、batch1、fused FP32、GPU常驻输入、CUDA event同步；记录逐次样本和median/p95。THOP不一定计入门控乘法，另外标注新增128×80×80次逐元素乘法。耗时不含传输、预后处理，不代表Orin或TensorRT性能。

## 晋级决定与第二阶段

150轮筛选后，ChannelGate相对本轮新C2须同时满足：

- AP≥+0.3pp，APs≥+0.3pp；
- AP75/APm/APl各自下降不超过0.5pp；
- 同场GPU前向中位耗时增幅≤10%。

使用未四舍五入值判断。另报告ChannelGate−Scalar差值；不假定通道级一定优于统一标量。工程门槛不代表统计显著性。

**通过则自动进入第二阶段**：C2与ChannelGate每组从头训练完整350轮，seed0/1/2，每个seed配对初始化，workers0、patience0及其他配方一致。每个seed独立评估并做同场两模型测速，最后报告三seed均值、样本标准差和配对差值。不会把150轮权重warm start成350轮，也不会用历史28.722 AP作为唯一对照。

**未通过则完成筛选并保留负面结果**，不自动扩展空间门控或继续消耗350轮预算。

## 启动、恢复与归档

运行根目录：`/home/zjnu/文档/ZHJ_WorkSpace/ultralytics/runs/c2_channel_author_20261105`。

首次准备已入Git的后端和配置后，预检/短跑示例（完整数据与UAV_CV环境需另行提供）：

```bash
cd /home/zjnu/文档/ZHJ_WorkSpace/ultralytics/experiments/c2_channel_author_20261105
export PYTHONPATH="$PWD/backend:$PWD"
/home/zjnu/anaconda3/envs/UAV_CV/bin/python preflight.py
/home/zjnu/anaconda3/envs/UAV_CV/bin/python train.py --arm channel --smoke
```

三组真实短跑、独立评估、测速与raw-state恢复检查通过并提交源码后，运行 `start.py`。启动器要求目标分支、干净工作区、空闲GPU，冻结已提交实验源码和后端到runs/.../source，并记录commit与逐文件SHA。正式进程只从冻结副本加载后端，每阶段核对源码，不受当前工作区后续编辑影响。tmux会话：`c2-channel-author-20261105`。

检查状态：`runs/c2_channel_author_20261105/queue_status.json`、各run的status.json、logs/。启动器拒绝重复队列；`start.py --resume-controller`仅供核查后恢复父控制器，依靠done列表避免重跑已完成阶段。

保存完整FP32 raw/EMA/优化器、scheduler、scaler、stopper及Python/NumPy/Torch RNG；标准库pickle、CPU暂存、fsync、重载校验后原子替换。训练失败最多自动尝试一次从本run完整last检查点恢复，并先保存故障检查点；初始无检查点则停止。恢复时逐张量校验raw模型、EMA、优化器，记录恢复epoch；采样顺序可能改变，不称逐位等价。单个训练进程30分钟没有日志/状态推进会终止并按同一策略处理。

独立评估和测速失败最多重试三次，保留各次日志；不会因此重训或偷偷改评估算法。仍失败则停止并等待排查。

控制器逐组归档完整results.csv（与服务器原文件逐字节一致）、实际配置、初始化和首批配对记录、Alpha与诊断CSV、正式指标、native图表和原始测速样本，并仅提交本报告目录。GitHub推送由本机认证完成；完整权重、数据集、逐图预测、完整日志及源码快照只留服务器。MANIFEST.sha256只列出本目录实际纳入Git的归档文件。

离线重建结果和图表（只依赖Git归档数据、NumPy和Matplotlib，不需要GPU/权重）：

```bash
python experiments/c2_channel_author_20261105/report.py --report docs/experiment_reports/2026-11-05_c2_channel_author
```

证据入口：[结构与初始化预检](raw/preflight.json)、[三组真实短跑](raw/smoke_verification.json)、[raw-state恢复验证](raw/resume_verification.json)、[实验规格](experiment_spec.json)。用户原始方案保存在raw/user_design.txt，属于设计输入，不作为实际结果。

## 运行中恢复记录：C2第51轮

C2在第51轮反向传播遇到PyTorch CUDA内部断言（CUDAGuardImpl.h:25），控制器自动从完整第50轮检查点恢复。原始模型、EMA和优化器逐张量验证通过，训练配方及workers0未更改。采样顺序可能随恢复改变，因此后续只能视为同初始化、同预算的恢复对照，不称逐位等价的完整配对轨迹。筛选结论及晋级解释须保留这个限制；若晋级，350轮各组仍从头配对训练。详见[恢复审计](raw/screen_c2_150e_s0/recovery_audit.json)。
