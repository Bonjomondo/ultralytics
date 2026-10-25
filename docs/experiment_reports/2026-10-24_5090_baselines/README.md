# 5090 C0/C2 基线与 Query 预算实验

2026-10-24，Asia/Shanghai。分支 `5090-detrscope-query-budget`。本次在 RTX 5090 D v2 上重建 C0/C2 的同机对照，实验与日志名称均以 `5090-` 开头；4090历史结果继续单列。

当前状态：**C0/C2各150轮训练、8图接入门禁、548图四档Query评估与计时全部完成，队列于2026-10-25 10:04（Asia/Shanghai）结束。** 最新结果由[自动实验汇总](../../../reports/5090/experiments.md)提供，同时生成CSV、JSON、Excel和曲线；[自动汇总操作说明](../../../experiments/decoder_budget_5090/AUTO_REPORTS_5090.md)。

## 为什么先重建基线

本机没有 RT-DETR C0/C2 旧权重，已有 SO-DETR 权重的架构和训练协议不同。故复用原 C0/C2 的配方及共有参数初始化规则，从零开始重建同机、同预算的150轮基线。短跑权重只用于验证训练、保存加载和诊断链，不进入正式模型对照。

真实数据来自本工作区 `SO_DETR/datasets/VisDrone2019-DET-YOLO`：6471张训练图、343200框；548张验证图、38759框。当前COCO标注SHA为 `6f8b8686864f1485ab172de9339f7049593f74766f6654c6e980a6c00d810371`，与旧4090 RT-DETR记录不同。已核对本机图像/标签完整及COCO对YOLO坐标误差<=5e-7；现有证据仍不能证明两份历史COCO语义完全一致。本次冻结本地GT，只计算同一5090、同一GT下的配对差异。

## 顺序与晋级门禁

1. C0、C2分别跑1轮、训练集fraction=0.01、640、batch8、FP32的真实数据短跑；各自验证集仍使用完整548图。要求loss有限、健康记录有限、两组共有初始化SHA相同、C2支路确有梯度、best/last/resume checkpoint齐全。
2. 对两份短跑权重各做固定8张真实验证图、300/250/200/150Q完整Decoder重执行，验证观测等价、GT/输入身份、COCO导出、离线分析和报告。这些指标仅作工程验收。
3. 门禁全部通过后，顺序训练 `5090-rtdetr-r18-c0-150e-s0` 与 `5090-rtdetr-r18-c2-150e-s0`。每组完成后先8图接入检查，再548图Query Sweep及无observer GPU计时。
4. 根据真实Coverage/AP/APs/延迟取舍再选择少量预算；本队列不自动加入采样mask、新结构或多seed。

完整训练沿用旧研究的150e、seed0、AdamW、lr1e-4、nbs64、FP32、workers0、batch8、同初始化规则。框架checkpoint保存FP16 EMA并在评估恢复FP32；保留未strip的resume.pt，但不声称checkpoint跨硬件逐比特恢复。

## 执行与产物

新增入口：[训练](../../../experiments/decoder_budget_5090/train_5090.py)、[串行队列](../../../experiments/decoder_budget_5090/queue_5090.py)。所有run、日志均以5090开头。长任务放在独立tmux会话，任何阶段失败即停，不覆盖失败目录。

正式Query评估使用本次诊断协议：未融合FP32、原生stretch/RGB/255、原生Predictor后处理、不舍入COCO JSON、maxDets=[1,10,100]。它与旧4090 fused/舍入validator协议分开记录。

权重、完整原始tensor、逐图预测和完整日志只留服务器；仓库保留命令、源码身份、紧凑状态/健康检查与阶段指标。运行中状态是本次观察快照，不替代服务器上的最新status_5090.json。

## 已通过的真实数据门禁

| 检查 | C0 | C2 |
| --- | --- | --- |
| 1轮、fraction=.01、batch8 FP32训练及完整548图验证 | 通过 | 通过 |
| best/last/resume checkpoint保存和加载 | 通过 | 通过 |
| 同一8图四档300/250/200/150Q完整Decoder、观测等价与离线报告 | 通过 | 通过 |
| 原生输出、RNG及卸载恢复 | 全batch通过 | 全batch通过 |
| 配对初始化 | 两组共有SHA完全相同 | P2支路梯度非零 |

[数据审计](raw/asset_audit_5090.json) · [训练门禁](raw/training_gates_5090.json) · [真实图像Query门禁](raw/query_gates_5090.json) · [门禁回执](raw/gates_passed_5090.json) · [实际环境](raw/environment_5090.json) · [测试日志](raw/pytest_5090.log)。其中1轮模型的检测指标仅验收工程链，不能与150轮架构结果比较。

首次C0短跑在保存阶段因缺少polars失败；完整目录保留，[失败及修复记录](raw/failed_attempt_5090.json)。随后只在本次临时venv安装polars==2.0.0，并在新`-r1`目录重跑全部门禁通过，未改变共享UAV_CV环境或旧4090源码。

## 5090分段延迟与瓶颈评估

固定一张真实验证图；未融合FP32、batch1、640，先预热150次，再做5轮交错顺序，每条件每轮10次预热/50次计时。分段与无标记原生forward分别采样；每条件每模式250次。此处为独立测量，不替换原Query Sweep的200次整网计时。
同一次forward内设置CUDA Event边界，分段之和逐次核对整网；包含CPU提交导致的CUDA流空隙，不等于纯GPU内核忙碌时间，不含传输/预处理/后处理。

| 阶段（300Q） | C0 ms | C0占比 | C2 ms | C2占比 |
|---|---:|---:|---:|---:|
| Backbone（层0–4） | 2.919 | 39.6% | 2.921 | 39.3% |
| Neck/Encoder（层5–23） | 2.554 | 34.7% | 2.609 | 35.1% |
| Head输入投影与Query选择 | 0.363 | 4.9% | 0.359 | 4.8% |
| Decoder与预测头 | 1.501 | 20.4% | 1.511 | 20.3% |
| 输出拼装 | 0.030 | 0.4% | 0.032 | 0.4% |

| 模型 | Q | 无标记整网 ms | 分段整网 ms | 标记开销 | Decoder与预测头 ms |
|---|---:|---:|---:|---:|---:|
| C0 | 300 | 7.329 | 7.368 | 0.53% | 1.501 |
| C0 | 200 | 7.314 | 7.349 | 0.48% | 1.487 |
| C0 | 150 | 7.319 | 7.349 | 0.42% | 1.489 |
| C2 | 300 | 7.394 | 7.431 | 0.50% | 1.511 |
| C2 | 200 | 7.389 | 7.419 | 0.40% | 1.497 |
| C2 | 150 | 7.387 | 7.426 | 0.52% | 1.508 |

Backbone为最大单块（约39%），Neck/Encoder约35%，两者合计约74%；Decoder与预测头约20%。Query从300降到150时，Decoder段也未明显变快，不能用Query减半推算整网计算量或延迟减半。若仅假设Decoder段耗时减半、其余不变，整网耗时约下降10%，这只是占比估算。
本实验只改变Query；Decoder仍为3层、8头、3层级、各层级4个采样点，未减少采样点或实施采样mask。固定特征图上的Backbone、Neck/Encoder和value投影等工作不随Query缩减同比下降。
模块分布尚不能判定算力、访存或CPU提交哪个是根因；需要算子级分析及目标部署路径复测。固定单图结果不代表全验证集时延分布；FP16、融合、TensorRT或Orin需重新测量。输出/RNG等价、模型state不变、预算恢复和0残留hook均通过；checkpoint与输入哈希为归档时回溯核验，详见身份记录。

[分段CSV](../../../reports/5090/stage_latency.csv) · Excel的Stage latency页 · [逐次原始计时](raw/stage_latency_5090.json) · [身份与复现来源](raw/stage_latency_identity_5090.json)

### 测量边界、身份与复现

测量于2026-10-25 11:50:17（Asia/Shanghai）开始。输入为COCO中按ID排序的第一张验证图（ID=1，`0000001_02999_d_0000005.jpg`），按原生RGB/stretch/255预处理至640×640，4个CPU线程，关闭TF32与cuDNN benchmark。五轮中每轮将三档预算×两种计时模式以固定种子 `20261025 + round_index` 打乱。没有测250Q的分段耗时，不从其他预算插值补齐。

边界依次设在整网入口、`model.model[4]`之后、`model.model[23]`之后、`head.decoder`之前/之后及整网出口。四个临时hook只记录Event；forward内部不插入同步，整网出口Event同步后读取。六个Event预先创建/预热后复用。每次核对五段之和与整网相差不超过0.02ms；阶段统计的分母为带标记的整网时间，不能拿独立缓存子模块的时间相加归因。各段五轮均值范围、250个逐次样本、GPU时钟/温度与完整环境均保存在原始JSON。

300Q下，C0的Backbone五轮均值为2.9176–2.9208ms，Neck/Encoder为2.5508–2.5600ms；C2对应为2.9198–2.9233ms、2.6056–2.6123ms。分段标记开销在本次六个模型/预算条件下约0.40%–0.53%。小于百分之一的整网差异不作为已证明的稳定加速；该测量未做多输入分布、目标设备能耗或算子根因验证。

原始计时文件记录了checkpoint路径和运行时等价/恢复检查，但没有在计时循环内记录权重、输入或源码哈希。[归档身份记录](raw/stage_latency_identity_5090.json)明确区分了事后核验：归档时读取两份best.pt并核对已完成训练的SHA，核验图像/标注及相关模型源码当前字节；执行脚本逐字节归档。它不声称这些事后哈希是运行时采集。C0/C2的权重SHA分别为 `adeaec17109a3b3ac6b8a6e521fb6cc3c2bdb3cd124ad6eedecc930d35a03d2e`、`ac2b02c45596bb816c4a42799d2b97ff02d62a4dc153bc5816502f02e936a5d6`。

[实际执行源码](../../../experiments/decoder_budget_5090/profile_stages_5090.py)保留本机固定ROOT、数据和权重路径。复测依赖本报告列明的服务器资产及DETRScope环境；迁移机器先调整路径，并保留修改后的源码身份。从本仓库根目录运行（会新建带5090前缀的诊断目录，不训练）：

```bash
PYTHONPATH="$PWD:$PWD/../detrscope/src" \
YOLO_CONFIG_DIR=/tmp/zhj-uav-5090-config \
  /tmp/zhj-uav-5090-venv/bin/python -m experiments.decoder_budget_5090.profile_stages_5090
```

原始服务器目录（不作为云端必须存在的链接）：`runs/diagnostics/5090-stage-latency-20261025T115017/`。初步独立子模块/算子探查目录 `runs/diagnostics/5090-query-timing-20261025T113627/`仅留服务器；其中缓存子模块与受profiler影响的整网时间不用于上述最终分段归因。权重、图像、逐图预测及大tensor均未纳入本次提交。

### 对后续研究的影响

当前证据支持优先评估Backbone与Neck/Encoder的轻量化机会，并在实际目标执行路径上复测。Query预算仍需结合AP/APs与密集场景候选覆盖选择，现阶段不能把减少Query写成已获得显著延迟收益。采样点预算是尚未实现的独立实验变量；后续应固定Query，逐项改变采样点，先验证输出形状、有限性和调试记录，再训练/评估AP与APs并重测各段耗时。减少采样点也不减少上述固定特征提取和全部value投影开销，实际收益待测。

自动汇总会校验原始逐次计时、摘要、样本数、轮次、分段之和与恢复检查，按同一已完成训练的checkpoint SHA关联；相关权重不匹配或记录不完整时不冒充当前结果。分段摘要嵌入自动汇总JSON，支持离线重新生成Markdown/CSV/Excel；不自动重跑GPU测量。

## 原始产物位置

实验根目录（仅服务器）：

```text
/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1/
  status_5090.json
  5090-rtdetr-r18-c0-150e-s0/
  5090-rtdetr-r18-c2-150e-s0/
  logs_5090/5090-*.log
```

当时的tmux会话：`5090-r18-c0c2-full150-20261024-r1`，队列已正常退出。控制日志为仓库下 `logs/5090-c0c2-full150-20261024-r1.log`；每阶段详细日志和命令写入`status_5090.json`。启动观察见[历史启动快照](raw/launch_status_5090.json)，不将该历史快照改写为完成结果；最新结果由自动汇总重新扫描取得。

复现时先选择不存在的新root并运行门禁，再继续该root：

```bash
UAV_PYTHON=/tmp/zhj-uav-5090-venv/bin/python
EXPERIMENT_ROOT="$PWD/runs/decoder_budget_5090/5090-c0c2-baselines-新实验名"
"$UAV_PYTHON" experiments/decoder_budget_5090/queue_5090.py \
  --root "$EXPERIMENT_ROOT" --sdk ../detrscope --gate-only
"$UAV_PYTHON" experiments/decoder_budget_5090/queue_5090.py \
  --root "$EXPERIMENT_ROOT" --sdk ../detrscope --continue-after-gates
```

上述长任务需置于tmux并将输出直接重定向文件；`--continue-after-gates`核验先前门禁回执和源文件SHA后才继续，不接受未完成/被改动的门禁。新机器需使用Python3.11与实际依赖环境，临时venv路径是本机执行证据。

启动时专项测试共 **58项通过**，对应上方历史测试日志；2026-10-25补充分段归档与自动报告后，当前专项测试为 **94项通过**。云端检出后可运行 `python3 docs/experiment_reports/2026-10-24_5090_baselines/verify_5090.py` 校验提交的源码及门禁证据，不需要服务器权重、数据或GPU。

自动汇总升级了队列及其测试。原实验冻结SHA保持不变，历史门禁核验从本地Git的启动版本 `d11de6e4` 读取这两个文件的原始字节，其余文件继续核验当前归档；不静默重写冻结身份。自动报告的独立JSON可脱离Git、服务器资产和DETRScope重新渲染。

当前报告验证命令（CPU；第一项需要本地DETRScope源码，后两项只需要本仓库与Git历史）：

```bash
PYTHONPATH="$PWD:$PWD/../detrscope/src" YOLO_CONFIG_DIR=/tmp/zhj-uav-5090-config \
  /tmp/zhj-uav-5090-venv/bin/python -m pytest experiments/decoder_budget_5090/tests -q
python3 docs/experiment_reports/2026-10-24_5090_detrscope/verify.py
python3 docs/experiment_reports/2026-10-24_5090_baselines/verify_5090.py
```
