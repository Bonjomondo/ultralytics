# 面向无人机密集小目标检测的 RT-DETR / SO-DETR 轻量化优化研究构想

**研究方向：** 基于诊断驱动的 Query Selection 与多尺度 Deformable Attention 稀疏采样优化  
**数据集：** VisDrone2019-DET  
**候选架构：** RT-DETR R18、RT-DETR R18 + P2 融合、SO-DETR R18-NoFFT  
**研究阶段：** 研究构想与实验设计，轻量化方案尚待验证

## 一、研究背景与动机

无人机航拍场景具有小目标多、密集遮挡、背景复杂、尺度变化大的特点，且最终希望部署在 Orin NX 等边缘平台。当前研究已在 SO-DETR / UAV-DETR 的特征融合与注意力模块上进行探索，并建立了 Encoder/Decoder 逐层诊断系统。下一阶段拟研究 **Decoder 计算冗余**：固定 300 个 Object Queries 是否必要？MSDeformAttn 的采样点是否可以减少？

RT-DETRv2 的采样点消融表明，降低采样数量有可能仅造成较小的检测精度损失。但其 COCO 结果不能直接推广到 VisDrone 的密集小目标场景。希望结合现有 Debug 系统先分析冗余，再进行针对性压缩，而非盲目网格搜索。

## 二、架构与特征流

### 2.1 RT-DETR 的 Encoder、Top-K 和 Decoder

以 640×640 输入、256 维特征为例：

| 特征层 | 分辨率 | Token 数 |
|---|---|---:|
| P3/8 | 80×80 | 6400 |
| P4/16 | 40×40 | 1600 |
| P5/32 | 20×20 | 400 |
| 合计 | — | **8400** |

Flatten + Concat 得到 `[B, 8400, 256]` 的 Encoder Tokens，而非 8400 个最终 Object Queries。分类评分与 Top-K 从中选择 300 个候选位置；在 `learnt_init_query=False` 的实现中，选中位置的 Encoder 特征初始化 Content Queries，对应 Proposal 初始化 Reference Boxes。完整 8400 个 Token 仍作为 Cross-Attention 的特征来源。

R18 当前重点研究配置使用 **3 层 Decoder**。每层包含：

- **Self-Attention：** Object Queries 之间交互，更新候选特征；部分计算随 Query 数量的平方增长。
- **MSDeformAttn Cross-Attention：** 根据 Query、Reference Box 预测偏移和权重，从多尺度特征中稀疏采样、聚合信息。
- **FFN 与预测头：** 更新 Query 表征、类别与边界框。

### 2.2 SO-DETR 的 P2 特征融合

SO-DETR **确实使用 P2**，但并未直接把完整 P2/4（160×160）送进 Decoder。其路径为：

`P2 (160×160) → SPDConv 下采样 (80×80) → 与 P3 及 FPN 特征 Concat → DDF / DDFNoFFT → RepC3 → Fused P3 → PAN → Fused P4/P5 → Decoder`

因此 Decoder 仍接收 P3/P4/P5，总计 **8400** 个 Encoder Tokens。若直接把 P2 作为第四尺度输入，数量才会变为 `25600+6400+1600+400=34000`，会显著增加候选评分、Value Projection 和访存成本，当前不建议这样做。

### 2.3 三条候选路线

| 路线 | P2 注入 | 特点 | 用途 |
|---|---|---|---|
| RT-DETR R18 C0 | 无 | 结构简洁，标准基线 | 主研究对照 |
| RT-DETR R18 P2-C2 | P2 stride-2 Conv | 轻量高分辨率融合 | 研究 P2 与压缩互补性 |
| SO-DETR R18-NoFFT | P2 SPDConv + DDFNoFFT | 无 FFT 但保留额外融合，Debug 基础成熟 | 跨架构验证 |

NoFFT 并不等于原版 RT-DETR：它移除了 DDF 的 FFT/IFFT 频域路径，但仍保留 P2 注入及空间融合模块。

## 三、已有实验与证据边界

### 3.1 RT-DETR R18 P2-SFI（150 Epoch，单 Seed 筛选）

| 配置 | P2 融合 | AP | APs |
|---|---|---:|---:|
| C0 | 无 | 18.705 | 12.104 |
| C1 | SPDConv | 18.830 | 12.254 |
| C2 | 3×3 stride-2 Conv | **19.408** | **12.933** |
| C3 | SPDLite | 18.416 | 12.035 |

C2 相比 C0：AP +0.704 pp，APs +0.829 pp。该结果提示简单 P2 注入值得继续验证，但还需确认完整收敛、多 Seed 稳定性，以及融合通道变化等混杂因素。

### 3.2 SO-DETR-NoFFT

已有 NoFFT B0 及采样分配实验。在此前 150 Epoch、seed0 的记录中：

| 配置 | AP | APs |
|---|---:|---:|
| NoFFT B0 | 27.412 | 19.227 |
| NoFFT-SP `[4,6,2]` | 27.026 | 18.835 |

`[4,6,2]` 与 `[4,4,4]` **总采样点数相同**，只是跨尺度重分配，不能作为“采样点减半”的证据。不同仓库、训练协议下的绝对 AP 也不能直接比较架构优劣。

## 四、研究方向 A：Query Selection 压缩与优化

### 4.1 问题与假设

固定 300 Queries 可能包含重复目标候选、背景候选和低价值候选；但密集场景中，降低 Query 数量也可能导致遮挡小目标在进入 Decoder 前被排除。

初步测试：`Nq ∈ {300, 250, 200, 150}`。

在使用 Encoder 特征初始化 Query 的实现中，可先用同一 checkpoint 改变 Top-K 数量，**重新执行完整 Decoder**，进行无需重新训练的快速评估；不能直接截断原 300 Query 的最终预测。

### 4.2 关键诊断指标

**Top-K GT Coverage：**

\[
\operatorname{Coverage}(K,\tau)=\frac{1}{N_{GT}}\sum_j\mathbf{1}\left[\max_{i\in TopK}\operatorname{IoU}(P_i,GT_j)\ge\tau\right]
\]

建议使用 IoU 阈值 0.3/0.5/0.75，分别报告全部目标、各类别、小目标、遮挡目标和拥挤区域的覆盖率，并考虑类别正确的覆盖率。

还应统计：

- **Query Rank Distribution：** Top-100、100～200、200～300 分别新增覆盖哪些 GT？
- **Query Redundancy：** 候选之间的重复与 GT 身份关系；不能简单把高重叠相邻目标当作冗余。
- **Decoder Contribution：** 哪些低分 Query 在 D1～D3 恢复为有效目标？哪些 Query 被其他候选替代？
- **边际收益：** 每增加 50 个 Query，带来的 Coverage/AP 改善与延迟成本。

若单纯减少 Query 导致漏检，再研究质量感知 Top-K、尺度感知候选预算、空间多样性与冗余感知选择；是否采用动态 Query 预算由证据决定。

## 五、研究方向 B：MSDeformAttn 采样点压缩

### 5.1 计算预算

设 Query 数 `Nq`、Heads 数 `H`、每 Head 各尺度采样点总数 `K`、Decoder 层数 `L`：

\[
S=N_q\times H\times K\times L
\]

当前 R18 示例：`Nq=300, H=8, L=3, [4,4,4]`，总采样位置数为 **86,400**。这是采样位置数，不是 GFLOPs，也不代表端到端延迟。

| 方案 | P3/P4/P5 | 每 Head 总点数 |
|---|---|---:|
| 基准 | `[4,4,4]` | 12 |
| 均匀减半 | `[2,2,2]` | 6 |
| P3 优先 | `[3,2,1]` | 6 |
| P4 优先 | `[2,3,1]` | 6 |
| 激进压缩 | `[2,1,1]` | 4 |

**采样预算压缩**（总点数减少）与 **采样预算重分配**（总点数相同）必须分开研究。P3 优先是候选假设，不是确定最优。

### 5.2 Debug 诊断与反事实干预

复用 V2 的 Sampling Locations、Offsets、Attention Weights，增加：

- 每尺度/每 Head/每点的权重与几何分布。
- 采样点与 GT 中心、目标边缘、相邻 GT 的关系。
- 按目标大小、密度、遮挡程度统计采样贡献。
- 采样点之间的特征相似性与空间重复。
- 逐层 D1/D2/D3 的采样敏感度和目标状态变化。

不能把低 Attention Weight、框外采样直接认定为无用，因为上下文和低权重信息也可能重要。

在 V1 增加采样点级 Mask 干预：保留指定采样点、屏蔽其他点、重新归一化保留权重，并重新执行后续 Decoder，比较 GT 状态与最终 AP。

**重要边界：** Mask 干预用于快速分析精度敏感性，通常不真正节省算力。正式压缩必须调整采样算子及 Offset/Weight 投影的实际维度，并微调或重新训练。对于 4D Reference Box，Offset 归一化与点数有关，直接改变 `n_points` 可能改变采样坐标，不能将两种实验等价。

## 六、研究方向 C：Query 与 Sampling 联合压缩

| Queries | 采样点配置 | 3 层总采样次数 | 较基准减少 |
|---:|---|---:|---:|
| 300 | `[4,4,4]` | 86,400 | 0% |
| 200 | `[4,4,4]` | 57,600 | 33.3% |
| 300 | `[2,2,2]` | 43,200 | 50.0% |
| 200 | `[3,2,1]` | 28,800 | 66.7% |
| 150 | `[2,2,2]` | 21,600 | 75.0% |

核心候选：**200 Queries + `[3,2,1]`**，理论采样位置数减少 66.7%，且 Query 减少还会影响 Self-Attention、FFN、预测头的部分计算。但整个网络 FLOPs 与延迟绝不会按同一比例下降，必须测量。

联合压缩的主要风险有两类：Query 变少导致 **候选覆盖丢失**；采样点变少导致 **已有候选定位与分类能力下降**。Debug 必须区分这两种错误来源。

## 七、研究方向 D：P2 融合与 Decoder 压缩的互补性

研究假设：P2 注入改善 P3 的空间细节，可能提升 Encoder 初始 Proposal 质量，使较少 Query 仍能覆盖目标，或使较少采样点仍能获取有效定位信息。

需在 **C0（无 P2）**、**C2（轻量 P2）**、**SO-DETR-NoFFT（P2 + DDFNoFFT）** 分别测量：

- Top-K Coverage 随 Query 数量下降的曲线。
- AP/APs 随采样预算下降的曲线。
- 逐层定位恢复能力与密集/遮挡分组表现。
- 相同采样预算下的真实延迟与显存。

P2 融合提升 AP 并不自动证明它能缓解 Decoder 压缩损失；需通过配对消融验证。

## 八、现有 Debug 系统的改造

### V0 Observation

复用 Encoder/Decoder 轨迹、GT 匹配和状态迁移；新增 `query_budget_sweep`、`topk_gt_coverage`、`query_rank_redundancy` 和按类别/尺度/拥挤度的边际覆盖分析。

### V1 Controlled Intervention

复用 Self/Cross-Attn、FFN、Reference 更新等干预能力；新增 `sampling_point_mask`、尺度级预算干预、逐层干预和完整 Decoder 重执行。Query 数量变化应作为完整推理配置而不是最终预测截断。

### V2 Mechanism Probing

复用真实采样位置、Attention Weight、Query 特征及几何统计；新增 `per_level_importance`、`per_point_redundancy`、`sampling_budget_sensitivity`、按 GT 分组的损害/恢复统计。

### 自动搜索框架

构建 **Diagnosis-Guided Decoder Budget Search**：

1. 固定 checkpoint、验证集与评估协议。
2. 记录 Encoder Top-K 与 Decoder 诊断数据。
3. 离线筛选 Query 数量。
4. 通过采样点 Mask 反事实分析敏感度。
5. 生成少量真实轻量化配置。
6. 短周期微调筛选，再进行完整训练。
7. 以 AP、APs、GT Coverage、Decoder/整网延迟形成 Pareto 排序。

目标是将 Debug 从解释工具升级为**可复用的诊断驱动自动调参工具**，而非用 Debug 直接替代正式训练。

## 九、建议实验路线

**阶段 0：基线审计。** 完成 C0/C2 接续收敛；统一 COCOeval、输入、Seed、训练预算和计时口径；核对 Decoder 层数及采样实现；迁移 Debug 并验证有无 Hook 的输出等价性。

**阶段 1：Query Sweep。** `300/250/200/150`，使用已有 checkpoint 重新执行完整 Decoder；记录 AP、APs、GT Coverage、候选冗余、延迟；挑选 1～2 个预算。

**阶段 2：Sampling Sweep。** 先 V2 诊断，再 V1 Mask 反事实；测试 `[4,4,4]`、`[2,2,2]`、`[3,2,1]`、`[2,3,1]`、`[2,1,1]`；对真正缩点结构微调验证。RT-DETR 分支可能需要移植 NoFFT 分支的按尺度独立点数实现。

**阶段 3：联合优化。** 优先尝试 200Q + `[2,2,2]`、200Q + `[3,2,1]`、250Q + `[3,2,1]`、150Q + `[2,2,2]`，实际候选由前两阶段结果决定。

**阶段 4：必要时改进 Top-K。** 只有确认减少 Query 导致特定类别/尺度 GT 覆盖不足，才引入质量感知、尺度预算或空间多样性机制。

**阶段 5：完整训练与多 Seed。** 对最优 2～3 个候选采用一致训练协议，记录 AP50、AP50:95、APs、分类/尺度/密集分组结果和多 Seed 波动。

**阶段 6：部署。** 验证 PyTorch FP32/FP16、ONNX、TensorRT 与 Orin NX 的精度、延迟、显存和端到端吞吐，确认实际算子确实减少工作量。

## 十、量化目标与研究问题

### 初步目标（不是已有结果）

- Query 从 300 尝试降至 200～250。
- 每 Head 总采样点从 12 尝试降至约 6。
- Decoder 采样位置数减少约 50%～70%。
- 以 AP50:95 下降不超过 **0.3 个百分点**、APs 下降不超过 **0.5 个百分点** 作为初步筛选目标，后续根据实验波动调整。
- 获得可复现的 Decoder 及整网推理加速，最终在 Orin NX 上验证。

### 研究问题

- **RQ1：** 固定 300 Queries 是否存在可量化冗余？如何在减少 Query 时保持密集目标 GT Coverage？
- **RQ2：** 不同尺度、不同 Decoder 层的采样贡献是否不均衡？能否在小预算下优化分配？
- **RQ3：** P2→P3 高分辨率融合是否能缓解 Query/采样压缩造成的小目标损失？
- **RQ4：** 能否利用 GT 状态迁移、反事实干预和逐层诊断，比盲目网格搜索更高效地选择预算？
- **RQ5：** 方法能否跨 RT-DETR、SO-DETR-NoFFT 迁移，并在真实部署中产生收益？

## 十一、风险与实验纪律

1. 低分 Query 可能在后续 Decoder 恢复，不能简单视为无效。
2. 高 IoU 相邻框可能对应不同密集 GT，不能粗暴去冗余。
3. Attention Weight 和几何位置不等于因果贡献，需配合干预。
4. Mask 反事实不等于真实减点结构，更不等于真实加速。
5. 采样次数减少比例不等于 FLOPs 或端到端延迟减少比例。
6. 不同仓库、训练轮数、初始化和评估口径下的绝对 AP 不可直接比较。
7. 先验证固定小预算；只有证据支持时再引入动态 Query/采样控制模块。

## 十二、当前结论与下一步

**暂定主线：** `RT-DETR R18 C0/C2 → Debug 诊断 → Query 压缩 → Sampling 压缩 → 联合优化 → 完整训练 → SO-DETR-NoFFT 迁移验证 → Orin NX 部署`。

其中 P2 融合、改进 Top-K、动态采样都应作为独立变量逐一验证，不宜一开始全部叠加。

**一句话概括：** 利用已有 DETR Debug 系统，从候选覆盖、采样贡献和逐层误差传播三个角度定位冗余，再联合优化 Query 数量与多尺度采样预算，在尽量保持 P2 高分辨率小目标检测优势的同时，降低 RT-DETR / SO-DETR 的 Decoder 计算开销。

## 参考代码与资料

- SO-DETR：<https://github.com/Bonjomondo/SO_DETR>
- SO-DETR NoFFT：<https://github.com/Bonjomondo/SO_DETR/tree/feat/nofft-local-attention>
- SO-DETR Debug：<https://github.com/Bonjomondo/SO_DETR/tree/feat/detr-debug-v0>
- RT-DETR R18 P2-SFI：<https://github.com/Bonjomondo/ultralytics/tree/feat/r18-p2-sfi>
- P2-SFI 实验结果：<https://github.com/Bonjomondo/ultralytics/blob/feat/r18-p2-sfi/experiments/r18_p2_sfi/corrected_20261024/RESULTS.md>
