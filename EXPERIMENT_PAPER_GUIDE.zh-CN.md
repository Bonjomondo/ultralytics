# Ultralytics 实验与论文复现指南（实战版）

这份文档面向“要做实验并写论文”的场景，目标是：

1. 实验可复现（别人按你的步骤能跑出同量级结果）
2. 结果可追溯（每个结论都能对应到具体配置和日志）
3. 写作可落地（表格、图、消融、对比实验都能直接组织）

适用任务：目标检测、实例分割、分类、姿态估计、OBB。

## 1. 推荐工作流总览

1. 固定环境和版本（Python、PyTorch、ultralytics 提交哈希）
2. 清洗并冻结数据集划分（train/val/test 不再变化）
3. 先跑 baseline（官方预训练权重 + 默认超参数）
4. 再做改进方法与消融（每次只改一个变量）
5. 多随机种子重复实验（建议 3~5 次）
6. 汇总统计（均值、标准差、速度、参数量、FLOPs）
7. 导出可复现包（代码、配置、日志、权重、命令）

## 2. 环境搭建（建议）

以下示例在 Windows PowerShell 下可直接执行。

### 2.1 新建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2.2 安装依赖

如果你在这个仓库内做改动并希望可编辑安装：

```powershell
pip install -e .
```

如果只是使用发行版：

```powershell
pip install -U ultralytics
```

### 2.3 记录环境用于论文附录

```powershell
python -V
pip freeze > artifacts\env\requirements-lock.txt
```

建议同时记录：

- GPU 型号、显存、驱动版本
- CUDA 版本
- PyTorch 版本
- 当前仓库提交号（`git rev-parse HEAD`）

## 3. 数据集组织与配置

## 3.1 推荐目录结构

```text
project_root/
  datasets/
    mydata/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
      mydata.yaml
```

## 3.2 检测任务 YAML 模板（mydata.yaml）

```yaml
path: datasets/mydata
train: images/train
val: images/val
test: images/test

names:
  0: class_a
  1: class_b
  2: class_c
```

关键点：

1. `train/val/test` 一旦确定，不要在论文周期内变更
2. 类别顺序固定，避免中途改 `names`
3. 建议留出独立 `test`，不要用 `val` 代替最终测试

## 4. Baseline：先跑通再优化

先确定一个“可复现 baseline”，例如检测任务：

```powershell
yolo detect train data=datasets/mydata/mydata.yaml model=yolo11n.pt imgsz=640 epochs=100 batch=16 device=0 seed=42 project=runs_paper name=baseline_yolo11n_s42
```

训练结束后，核心产物通常在对应目录：

- `weights/best.pt`
- `results.csv`
- `results.png`
- 各种 PR 曲线、混淆矩阵图

## 5. 论文常用实验设计

## 5.1 主结果（Main Results）

与你的对比方法在同一数据集、同一输入尺寸、同一评价指标上比较：

- 检测常用：mAP50、mAP50-95、FPS 或 latency
- 分割常用：mAP(box)、mAP(mask)
- 分类常用：Top-1、Top-5
- 姿态常用：mAP pose

## 5.2 消融实验（Ablation）

原则：一次只改一个变量。

示例维度：

1. 模块开关：`with_x=True/False`
2. 损失函数：`loss_a` vs `loss_b`
3. 输入尺寸：`imgsz=640/800/1024`
4. 数据增强策略：开/关 mosaic、mixup、copy-paste

推荐做法：

- 保持同一随机种子集合（如 42、43、44）
- 汇报均值和标准差，而不是单次最好结果

## 5.3 多种子重复实验

如果你的结果指标是 $m_i$，共跑 $n$ 次：

$$
\mu = \frac{1}{n}\sum_{i=1}^{n} m_i, \quad
\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(m_i-\mu)^2}
$$

论文中建议写成：

- `mAP50-95 = 51.2 ± 0.3`

## 6. 一组可直接复用的命令模板

## 6.1 训练

```powershell
yolo detect train data=datasets/mydata/mydata.yaml model=yolo11s.pt imgsz=640 epochs=300 batch=16 device=0 workers=8 seed=42 project=runs_paper name=exp1
```

## 6.2 验证

```powershell
yolo detect val data=datasets/mydata/mydata.yaml model=runs_paper/exp1/weights/best.pt split=val imgsz=640 device=0
```

## 6.3 测试集评估

```powershell
yolo detect val data=datasets/mydata/mydata.yaml model=runs_paper/exp1/weights/best.pt split=test imgsz=640 device=0
```

## 6.4 推理示例图（论文可视化）

```powershell
yolo detect predict model=runs_paper/exp1/weights/best.pt source=datasets/mydata/images/test imgsz=640 conf=0.25 save=True project=runs_paper name=pred_vis
```

## 6.5 导出部署模型（补充实验）

```powershell
yolo export model=runs_paper/exp1/weights/best.pt format=onnx imgsz=640
```

## 7. 结果管理建议（非常重要）

## 7.1 命名规范

实验名建议包含：

- 模型规模（n/s/m/l/x）
- 数据集版本
- 关键改动
- 随机种子

示例：

- `y11s_v2_cbam_s42`
- `y11s_v2_cbam_s43`
- `y11s_v2_baseline_s42`

## 7.2 建议目录

```text
runs_paper/
  baseline_yolo11n_s42/
  baseline_yolo11n_s43/
  methodA_yolo11n_s42/
artifacts/
  env/
    requirements-lock.txt
  tables/
    main_results.csv
    ablation.csv
  figures/
    pr_curve.png
    qualitative_cases.png
```

## 7.3 不要只保留 best.pt

建议同时保存：

1. 完整训练日志（包括 `results.csv`）
2. 关键配置（命令行参数或 yaml）
3. 失败实验记录（避免重复踩坑）

## 8. 论文写作对齐清单

投稿前你至少要能回答：

1. 你的 baseline 是什么，是否公平对比？
2. 所有对比方法是否使用相同数据划分与评价协议？
3. 每个结论是否有实验支撑（表格或图）？
4. 是否报告了速度/参数量/FLOPs（不仅是精度）？
5. 是否有消融与误差分析（失败案例）？
6. 是否提供复现信息（代码、模型、命令、随机种子）？

## 9. 论文表格模板（可直接粘到 Markdown）

## 9.1 主结果表

```markdown
| Method | Input Size | Params (M) | FLOPs (G) | mAP50 | mAP50-95 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (YOLO11n) | 640 | 2.6 | 6.5 | 68.1 | 45.7 | 210 |
| Ours | 640 | 3.0 | 7.4 | 70.2 | 48.1 | 190 |
```

## 9.2 消融表

```markdown
| ID | Module A | Module B | Augment X | mAP50-95 |
|---|---|---|---|---:|
| A0 | - | - | - | 45.7 |
| A1 | ✓ | - | - | 46.5 |
| A2 | ✓ | ✓ | - | 47.4 |
| A3 | ✓ | ✓ | ✓ | 48.1 |
```

## 10. 常见坑与规避

1. 数据泄漏：训练集与测试集重复图片或近重复图片
2. 对比不公平：你的方法用了更大输入尺寸或更久训练，但没说明
3. 只汇报最好一次：缺少多种子统计，结论不稳
4. 指标口径混用：有的用 val，有的用 test，导致表格不可比
5. 忘记固定随机种子：同配置结果波动大，难复现

## 11. 最小可复现实验包（建议作为开源附录）

至少包含：

1. 训练与评估命令（脚本或 README）
2. 数据集下载/组织说明（不含受限数据本体）
3. 模型权重（或下载链接）
4. 实验日志与结果表生成脚本
5. 环境锁定文件（`requirements-lock.txt`）

---

如果你愿意，我可以下一步直接帮你生成两样东西：

1. 一个可直接批量跑 3 个随机种子的 PowerShell 脚本
2. 一个自动汇总 `runs_paper` 结果为 `main_results.csv` 的 Python 脚本
