# RTX 5090 实验方案：C2 + RT-DETRv3 层次化密集正样本监督

**项目编号：** 5090-RTDETRv3-DS-01
**研究对象：** RT-DETR-R18 / VisDrone
**主要设备：** RTX 5090 24GB
**基础结构：** C2（P2 → P3 特征注入）
**核心参考：** RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision

---

## 1. 研究目标

本实验旨在研究 RT-DETRv3 提出的三种辅助监督机制：

1. O2M-C：CNN-based One-to-Many Auxiliary Branch。
2. O2M-T：Transformer-based One-to-Many Dense Supervision。
3. MGSA：Multi-Group Self-Attention Perturbation。

研究这三种机制能否改善 VisDrone 密集小目标检测，并分析它们与 C2 特征增强结构之间的关系。

最终希望回答四个问题：

- O2M-C、O2M-T、MGSA 分别能带来多少性能提升？
- 哪个模块最适合 VisDrone？
- 三个模块是否存在互补或负向干扰？
- 能否在不增加最终推理结构复杂度的情况下提升 AP？

### 研究假设

H1：O2M-C 能够增强 Encoder 的特征学习能力。

H2：O2M-T 能够增加 Decoder 获得的正样本监督数量。

H3：MGSA 能够通过不同组的 Self-Attention 扰动，改善 Query 的正样本分配多样性。

H4：C2 的高分辨率细节增强与密集监督机制可能存在协同作用。

以上均为待验证假设，不预设实验一定取得正收益。

---

## 2. 基础模型与实验约束

### 2.1 基础模型

以现有 C2 为主要实验基线：

- Backbone：ResNet18。
- 特征融合：现有 C2 / P2 注入结构。
- Encoder：保持现有结构。
- Query 数量：300。
- Decoder：保持现有 3 层配置。
- 检测类别：VisDrone 10 类。
- 原始检测监督：Hungarian Matching / O2O。
- 训练配方：采用已经验证的作者配方 350 epoch 训练配置。

同时保留 C0 作为对照模型。

### 2.2 必须固定的变量

所有对照实验尽量保持以下设置一致：

- 数据集及训练、验证划分。
- 输入分辨率。
- Backbone 预训练权重。
- Batch Size 和有效 Batch Size。
- Optimizer。
- Learning Rate 和 Scheduler。
- Weight Decay。
- 数据增强策略。
- Epoch 数量。
- EMA 配置。
- Seed。
- 验证与评估代码。

不得因为增加辅助分支而顺手修改其他网络结构或训练超参数。

如果辅助模块导致显存不足，优先采用合理的梯度累积方案，并记录实际有效 Batch Size。

### 2.3 现有基线参考

此前作者配方 350 epoch 实验：

| 模型 | AP |
|---|---:|
| C0 | 27.77 |
| C2 | 28.72 |
| C2 增益 | +0.95 |

这两组结果作为已有参考。

正式比较之前，必须核对代码版本、训练配置、Seed 和评估协议是否一致。如果不一致，需要补充匹配对照，不能直接将不同协议的结果拼接。

---

## 3. 三个模块的实现设计

### 3.1 O2M-C：Encoder CNN 一对多辅助监督

**研究目的：** 增强 Encoder 特征表达。

按照原论文思路，使用 PP-YOLOE 风格的 CNN 辅助检测头。

输入：

- Encoder 输出 P3。
- Encoder 输出 P4。
- Encoder 输出 P5。

对于 C2：

P2 通过现有注入路径增强 P3，但 O2M-C 不直接增加原始 P2 输入分支。

辅助头采用：

- ATSS：训练早期正样本分配。
- TaskAligned：后续正样本分配。
- VFL：分类损失。
- DFL：边界框分布回归。

损失：

L_total = L_O2O + α × L_O2M-C

原论文默认 α=1。

实验要求：

1. 保留原始 O2O Decoder。
2. O2M-C 只负责辅助监督。
3. 梯度能够从辅助头反向传播到 Encoder。
4. 推理时移除 O2M-C。
5. 记录正样本数量、各类别分配情况及损失变化。

**重点研究：** O2M-C 是否能够改善小目标和密集目标的特征学习。

### 3.2 O2M-T：Transformer 一对多辅助监督

**研究目的：** 增强 Decoder 的正样本监督。

实现思路：

1. 使用现有 Query Selection 生成 300 个 Object Queries。
2. 保留原始 O2O Decoder 分支。
3. 增加参数共享的 Transformer 一对多监督分支。
4. 将每个 GT 复制 m 份。
5. 将扩展后的 GT 集合与 Decoder Query 进行匹配。
6. 使用原有检测损失计算辅助分支损失。

原论文默认：

m = 4

损失：

L_total = L_O2O + γ × L_O2M-T

原论文默认 γ=1。

实现要求：

- 共享 Decoder 参数，而不是简单复制一套完整 Decoder。
- 保留原始 Hungarian Matching。
- O2M-T 使用独立的一对多匹配过程。
- 明确区分主分支和辅助分支的预测及损失。
- 推理时移除辅助监督分支。

**重点研究：** 多个 Query 对同一 GT 提供监督，是否能改善 VisDrone 的密集目标检测。

### 3.3 MGSA：多组 Self-Attention 扰动

**研究目的：** 增强 Decoder Query 之间的交互多样性。

按照原论文：

1. 通过 Query Selection 生成多组 Query。
2. 为每组生成对应的随机扰动 Mask。
3. 将 Mask 作用于 Self-Attention。
4. 不同组经过共享 Decoder 参数进行处理。
5. 各组分别执行原始 O2O Hungarian Matching。
6. 对各组 O2O Loss 求平均。

核心计算：

W_i = Softmax(M_i × (Q_i K_i^T))

V_i' = W_i V_i

原论文默认扰动组数量：

N = 3

MGSA 的主要特点：

- 共享 Decoder 权重。
- 不改变原始检测任务定义。
- 每组仍然进行 O2O 匈牙利匹配。
- 通过不同的 Self-Attention 扰动，产生不同的 Query 表达与正样本分配。

注意：MGSA 不等同于 O2M-T。前者改变 Query 的交互方式，后者改变 Decoder 的标签匹配方式。

**重点研究：** 不同组是否真的产生了更丰富的正样本分配，而不只是增加计算量。

---

## 4. 实验矩阵

### 阶段 A：O2M-C 独立贡献验证

这是整个项目的第一优先级。

采用 2×2 消融设计：

| 编号 | 模型 | P2 注入 | O2M-C |
|---|---|---|---|
| A0 | C0 | 否 | 否 |
| A1 | C2 | 是 | 否 |
| A2 | C0 + O2M-C | 否 | 是 |
| A3 | C2 + O2M-C | 是 | 是 |

训练安排：

- 先进行 150 epoch 筛选。
- A0/A1 优先复用已有匹配训练的中间 checkpoint。
- 如果没有匹配的 150 epoch checkpoint，补充短程对照。
- A2/A3 使用完全一致的 O2M-C 实现和损失权重。

如果筛选结果有效，进入 350 epoch 正式训练。

核心分析：

Δ_C0 = AP(A2) - AP(A0)

Δ_C2 = AP(A3) - AP(A1)

交互效应：

I_C2,O2M-C = Δ_C2 - Δ_C0

如果交互效应为正，说明 C2 与 O2M-C 可能存在正向协同。

特别关注 AP-small、Recall 和各类别 AP。

### 阶段 B：O2M-T 独立验证

| 编号 | 模型 |
|---|---|
| B0 | C2 |
| B1 | C2 + O2M-T |

配置：

- Query = 300。
- GT 复制倍数 m=4。
- Decoder 参数共享。
- 原始 O2O 分支保留。
- 初始 γ=1，按照原论文默认设置。

先进行 150 epoch 筛选。

如果取得稳定正收益，再进行 350 epoch 正式训练。

如果显存、训练速度或损失规模出现异常，先排查实现与归一化，不要直接改变多个超参数。

### 阶段 C：MGSA 独立验证

| 编号 | 模型 |
|---|---|
| C0 | C2 |
| C1 | C2 + MGSA |

配置：

- MGSA 组数 N=3。
- 每组 300 Queries。
- Decoder 参数共享。
- 每组独立 O2O Hungarian Matching。
- 各组 O2O Loss 求平均。

先进行 150 epoch 筛选。

如果结果有效，再考虑组数消融：

| 组数 | 说明 |
|---|---|
| 2 | 较少扰动组 |
| 3 | 原论文默认配置 |
| 4 | 更多扰动组 |

不要在第一次实验就同时改变组数和其他训练参数。

### 阶段 D：有效模块组合

根据 A、B、C 的结果决定是否组合。

优先考虑：

| 编号 | 模型 |
|---|---|
| D0 | C2 + O2M-C + O2M-T |
| D1 | C2 + O2M-C + MGSA |
| D2 | C2 + O2M-T + MGSA |
| D3 | C2 + O2M-C + O2M-T + MGSA |

组合实验仍然保留原始 O2O 主检测监督。

如果某个独立模块没有有效收益，且没有明确机制证据，不必强行把它加入完整组合。

先进行 150 epoch 筛选，再对最优组合进行 350 epoch 正式训练。

---

## 5. 训练顺序与 GPU 执行计划

RTX 5090 按以下顺序执行：

| 顺序 | 实验 | Epoch | 优先级 |
|---|---|---:|---|
| 1 | C0 + O2M-C | 150 | 最高 |
| 2 | C2 + O2M-C | 150 | 最高 |
| 3 | C0 + O2M-C | 350 | 正式验证 |
| 4 | C2 + O2M-C | 350 | 正式验证 |
| 5 | C2 + O2M-T | 150 | 高 |
| 6 | C2 + MGSA | 150 | 高 |
| 7 | 最优模块组合 | 150 | 中 |
| 8 | 最优组合 | 350 | 最终验证 |

说明：

- 现有 C0/C2 350 epoch 结果可以作为参考，但需要先核验训练配置。
- A2、A3 的正式训练是优先保障任务。
- B、C 两条支线不必全部强制进行 350 epoch。
- 如果 150 epoch 筛选没有明确收益，先检查收敛曲线和模块实现，再决定是否继续。
- 最终候选模型应使用至少两个额外 Seed 进行稳定性验证，优先对比相同 Seed 下的 Baseline 和改进模型。

---

## 6. 重点记录的机制指标

不能只记录最终 AP。

### O2M-C

记录：

- 每张图片正样本数量。
- 不同类别的正样本分布。
- Small / Medium / Large 目标的正样本数量。
- ATSS 与 TaskAligned 的匹配变化。
- Encoder 各尺度特征的梯度范数。
- O2M-C 和 O2O Loss 的相对大小。

### O2M-T

记录：

- 每个 GT 匹配到的 Query 数量。
- 一对多匹配成功率。
- 不同目标尺寸的匹配数量。
- O2M-T 与 O2O Loss 的比例。
- Decoder 参数是否真正共享。

### MGSA

记录：

- 各组 Query 的正样本数量。
- 不同组 Hungarian 匹配结果的差异。
- 各组预测框的 IoU 分布。
- 不同组 O2O Loss。
- Self-Attention Mask 的统计分布。
- 不同组 Query 特征的相似性。

特别注意：如果不同组最终得到几乎相同的匹配结果，MGSA 的扰动机制可能没有发挥预期作用。

---

## 7. 评价指标

### 精度指标

必须记录：

- AP。
- AP50。
- AP75。
- AP-small。
- AP-medium。
- AP-large。
- 10 个类别的 AP。
- Recall。

优先关注：

1. AP。
2. AP-small。
3. Recall。
4. AP75。

### 训练效率

记录：

- 每轮训练耗时。
- 显存峰值。
- GPU 利用率。
- 训练吞吐量。
- Loss 收敛曲线。
- 训练稳定性。

辅助监督虽然不增加最终推理分支，但可能增加训练耗时和显存。

### 部署指标

训练完成后，移除所有辅助分支，只导出原有 C0/C2 推理结构。

对比：

- 参数量。
- GFLOPs。
- 4090 TensorRT FP16 延迟。
- Orin NX TensorRT FP16 延迟。
- Orin NX TensorRT INT8 延迟。
- 完整视频检测 FPS。

确认辅助模块没有进入最终推理图。

---

## 8. 实验判断标准

### 第一类：有效

满足：

- 350 epoch 下 AP 有稳定提升。
- 多 Seed 下增益方向一致。
- 没有明显损害 AP-small 或 Recall。
- 推理结构和延迟基本不变。

进入下一阶段。

### 第二类：可能有效，需要进一步研究

例如：

- AP 提升较小。
- AP-small 明显提升，但 AP-large 下降。
- 早期收敛更快，但最终 AP 差异不大。
- 不同 Seed 之间波动明显。

需要增加机制分析或调整对应模块，不立即认定失败。

### 第三类：无效或负向

例如：

- AP 持续下降。
- 小目标 Recall 明显退化。
- 训练出现梯度冲突。
- MGSA 各组几乎没有差异。
- 辅助分支造成明显训练不稳定。

停止扩大组合，先排查原因。

---

## 9. 最终交付物

每个实验都必须保存：

1. 模型 YAML / 配置文件。
2. Git Commit ID。
3. 完整训练命令。
4. Seed 和环境信息。
5. 训练日志。
6. 最优 checkpoint。
7. 最终验证 JSON。
8. 类别 AP 表格。
9. 参数量、GFLOPs 和延迟报告。
10. Loss 和机制诊断图表。

统一输出目录建议：

```text
experiments/
└── rtdetrv3_dense_supervision_5090/
    ├── configs/
    │   ├── c0.yaml
    │   ├── c2.yaml
    │   ├── c0_o2mc.yaml
    │   ├── c2_o2mc.yaml
    │   ├── c2_o2mt.yaml
    │   ├── c2_mgsa.yaml
    │   └── c2_full.yaml
    ├── runs/
    ├── checkpoints/
    ├── evaluations/
    ├── diagnostics/
    └── reports/
        ├── experiment_plan.md
        ├── ablation_summary.md
        ├── mechanism_analysis.md
        └── deployment_comparison.md
```

---

## 10. 最终研究目标

最终希望得到以下实验结论：

1. C2 是否能够稳定提升 VisDrone 检测精度。
2. O2M-C 是否能独立改善 Encoder 特征表达。
3. O2M-T 是否能改善 Decoder 的正样本监督。
4. MGSA 是否能改善 Query 之间的正样本分配多样性。
5. 三种辅助监督机制是否存在协同作用。
6. 哪种组合能够取得最好的精度与训练成本平衡。

**最终部署原则：只保留有效的结构改进，所有不必要的辅助监督分支在推理时移除。**

本项目不是为了完整照搬 RT-DETRv3，而是以 C2 为基础，研究 RT-DETRv3 的层次化密集正样本监督在 VisDrone 小目标检测场景下的适用性，并确定最优的训练监督组合。
