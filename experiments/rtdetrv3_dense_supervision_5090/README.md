# 5090 作者配方：层次化密集正样本监督

项目5090-RTDETRv3-DS-01。旧协议四组seed0补齐已完成并推送至`5090-c2-o2mc-20261031`；新研究分支为`5090-rtdetrv3-dense-supervision-20261105`。本目录仅实现阶段A，先seed0，seed1/2最后验证。

## 本次自动队列

1. 真实C0/C2工程检查：初始化、alpha=0主损失等价、ATSS/TAL分配、三尺度梯度、推理不执行辅助头、去辅助头逐位等价、保存重载。
2. C0+O2M-C、C2+O2M-C各32图1轮训练，16图评估及COCOeval。
3. 复用4090 C2前150轮内最优检查点（实际epoch145），在5090完整548图重新评估。
4. 串行训练C0、C0+O2M-C、C2+O2M-C各150轮，逐组正式评估并归档。
5. 汇总2×2交互、尺度/类别AP及Recall，写入350轮候选筛选结果。队列在150轮科学判断点结束；350轮扩展、O2M-T、MGSA、组合实验尚未接入此执行器。

任何门禁、非有限loss、提前结束或子进程错误均停止队列并保留日志。关闭聊天不会停止tmux。此次不覆盖历史run、不自动启动其他seed，不自动push运行期间新结果。

## 固定训练与评估

使用4090作者引擎的独立源文件副本`author_engine/ultralytics`；主网络/主损失未修改。每个进程核验实际导入路径，禁止混入主仓库同名包。直接使用UAV_CV解释器。`reference/c0_args.yaml`与`c2_args.yaml`是原始配方，`train.py`只改数据/输出路径与阶段停止边界。

640输入，scratch无预训练，FP32/AMP关闭，batch4/nbs4、累积1、workers8、AdamW lr1e-4恒定、wd1e-4、mosaic1、mixup0.2、关闭mosaic轮数0、seed0、deterministic warn_only、原作者EMA。`warmup_epochs=2000`在该引擎中实际为2000个全局batch，bias起始lr0.1。TF32采用作者PyTorch默认：matmul关闭、cuDNN开启，并记录实测值。训练仍配置350轮/patience40，150轮完成后由回调暂停，保留完整optimizer/EMA与best/last；提前在150轮前停止视作筛选预算未完成，不伪装150轮结果。150轮是预算比较，不宣称收敛。

O2M-C沿用已验证的PP-YOLOE风格适配实现，输入主decoder的P3/P4/P5，C2不直接给辅助头新增P2。aux alpha=1，前30轮ATSS、之后TAL；VFL+2.5GIoU+0.5DFL。主损失包含原encoder、decoder及denoising项。辅助头隔离seed4000，不影响原主网络构造RNG；C2复制同seed C0的527个共有张量，符合4090原C2初始化回放SHA。4090原C0初始权重未归档，历史回放证明的边界保留。

数据使用独立`reproductions/rtdetrv3_dense_supervision_5090/data`标签/缓存，图片只读引用旧图片目录。当前4090与5090逐文件图片和标签SHA完全一致，详见`reference/data_verification.json`。原训练记录只保留标签/文件名身份，历史图片像素SHA不存在，不能声称验证了不存在的历史清单。训练标签原始343205项，作者loader会去重，与旧数据处理产生的343200项需区分。GT SHA为`6f57aa4b42d64ee7b99b8a6a5b159ad19cba2eacf00bd86f23314966614bf4d8`，不得混用旧5090评估GT。

正式评估：作者RTDETRValidator/AutoBackend默认融合、FP32（从原生FP16 EMA重载），batch4、640 stretch/RGB归一化、全部300个argmax query、无置信度过滤/无NMS/不舍入、原图坐标、类别0..9映射1..10；548图，独立CPU pycocotools，主表maxDets=[1,10,100]，附加maxDet300。这是转换GT上的COCO评估，不是VisDrone挑战赛分数。记录AP/AP50/AP75/APs/APm/APl、10类与AR100、训练资源、机制抽样，以及无探针GPU纯forward FP32延迟；部署FP16/INT8和Orin结果待后续真实测量，不填估计数。

## 运行和归档

队列入口：`/home/ubuntu/miniconda3/envs/UAV_CV/bin/python -u experiments/rtdetrv3_dense_supervision_5090/controller.py`，从仓库根启动；launch.sh位于run根目录并保存实际环境变量。

运行根目录：`runs/rtdetrv3_dense_supervision_5090/author-stageA-s0-20261105-r1`。`queue_status.json`含实际阶段、PID、命令和退出状态。`source_pin.json`冻结本目录所有必要源码/配置SHA，每个阶段先验pin。完整权重、预测、大日志和原始逐文件数据清单只留服务器。紧凑结果、曲线、诊断、配置、环境、SHA和对照表归档到`docs/experiment_reports/2026-11-05_5090_dense_supervision_author`。

原始完整研究计划见[原请求](../../docs/experiment_reports/2026-11-04_5090_c0_o2mc_completion/REQUESTED_STUDY.md)。旧150轮alpha0.5结果只作为动机证据，不能占用本alpha1作者配方的实验单元。
