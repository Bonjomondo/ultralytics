# RT-DETR VisDrone Baseline 实验说明

## 1. 实验目的

使用 Ultralytics 官方 RT-DETR-L 作为 Transformer 检测器 baseline，在完全相同的数据划分和训练设置下比较两种输入分辨率：

| 实验编号 | 模型 | 预训练权重 | 输入尺寸 | 4090 建议 batch | 主要目的 |
| --- | --- | --- | ---: | ---: | --- |
| R1 | RT-DETR-L | `rtdetr-l.pt` | 640 | 8 | 标准分辨率 baseline |
| R2 | RT-DETR-L | `rtdetr-l.pt` | 960 | 2 | 观察高分辨率对小目标检测的提升 |

选择 RT-DETR-L 而不是 RT-DETR-X，是因为 L 已经是有代表性的官方 RT-DETR baseline，同时更适合单张 RTX 4090 完成 960 分辨率训练。两组实验只改变 `imgsz` 和受显存约束的物理 batch，其余参数保持一致。

## 2. 数据与公平性要求

- 服务器若已有当前实验使用的 `datasets/VisDrone-local.yaml`，两组实验继续统一使用该文件。
- 新环境若没有本地数据配置，可使用仓库内的 `ultralytics/cfg/datasets/VisDrone.yaml`，并先确认其中的 `path` 指向服务器上的 VisDrone 数据集。
- 使用相同的 train/val 划分，不额外合并验证集或测试集。
- 两组都从同一个 COCO 预训练权重 `rtdetr-l.pt` 开始。
- 固定 `seed=0`，记录代码版本、CUDA、PyTorch 和 GPU 型号。
- 保持 `nbs=64`。Ultralytics 会根据物理 batch 自动进行梯度累积，使 640 和 960 的优化尺度尽量一致。
- 不开启多尺度训练，确保实验只比较固定的 640 和 960 输入。
- 训练阶段使用相同的 300 epochs、数据增强和验证设置，与现有 YOLO11-SAC 实验口径对齐。

> RT-DETR 使用的 `grid_sample` 不完全支持确定性训练，因此命令中明确设置 `deterministic=False`。固定随机种子仍能减少随机差异，但不能保证逐位复现。

## 3. 推荐训练参数

| 参数 | 设置 | 说明 |
| --- | --- | --- |
| `epochs` | 300 | 与现有 VisDrone 主实验一致 |
| `patience` | 100 | 避免过早停止，同时保留早停保护 |
| `optimizer` | AdamW | 更符合 Transformer 检测器的常用优化方式 |
| `lr0` | 0.0001 | RT-DETR 微调的保守初始学习率 |
| `weight_decay` | 0.0001 | AdamW 权重衰减 |
| `warmup_epochs` | 3 | 降低训练初期不稳定风险 |
| `amp` | True | 4090 上使用混合精度以降低显存和加速 |
| `nbs` | 64 | 统一名义 batch，自动梯度累积 |
| `workers` | 8 | 根据服务器 CPU 和存储速度可调至 4 或 16 |
| `cache` | False | 默认不占用大量内存；内存充足可改为 `disk` |
| `close_mosaic` | 10 | 最后 10 个 epoch 关闭 Mosaic |
| `max_det` | 300 | 与现有实验保持相同评估口径 |

这里不建议直接使用 `optimizer=auto`，因为它会根据训练迭代数自动选择优化器和学习率，不利于跨分辨率复现实验。

## 4. 训练命令

### R1：RT-DETR-L，640

```bash
yolo detect train \
  model=rtdetr-l.pt \
  data=datasets/VisDrone-local.yaml \
  imgsz=640 \
  epochs=300 \
  patience=100 \
  batch=8 \
  nbs=64 \
  optimizer=AdamW \
  lr0=0.0001 \
  lrf=0.01 \
  weight_decay=0.0001 \
  warmup_epochs=3 \
  amp=True \
  deterministic=False \
  seed=0 \
  device=0 \
  workers=8 \
  multi_scale=0.0 \
  close_mosaic=10 \
  max_det=300 \
  project=runs/rtdetr_baseline \
  name=rtdetr_l_visdrone_img640_s0
```

### R2：RT-DETR-L，960

```bash
yolo detect train \
  model=rtdetr-l.pt \
  data=datasets/VisDrone-local.yaml \
  imgsz=960 \
  epochs=300 \
  patience=100 \
  batch=2 \
  nbs=64 \
  optimizer=AdamW \
  lr0=0.0001 \
  lrf=0.01 \
  weight_decay=0.0001 \
  warmup_epochs=3 \
  amp=True \
  deterministic=False \
  seed=0 \
  device=0 \
  workers=8 \
  multi_scale=0.0 \
  close_mosaic=10 \
  max_det=300 \
  project=runs/rtdetr_baseline \
  name=rtdetr_l_visdrone_img960_s0
```

Windows PowerShell 中可将行尾的 `\` 改为反引号，服务器 Linux shell 可直接使用以上命令。

以上命令沿用已有服务器实验的 `datasets/VisDrone-local.yaml`。如果服务器上不存在该文件，请将两条训练命令和后续验证命令中的 `data` 统一替换为已经校正数据根目录的 `ultralytics/cfg/datasets/VisDrone.yaml`。

## 5. 4090 显存处理

建议先各训练 1 个 epoch 检查显存和数值稳定性，再启动完整实验。

- 640：从 `batch=8` 开始；OOM 时降为 `batch=4`。
- 960：从 `batch=2` 开始；OOM 时降为 `batch=1`。
- 不要为了塞入更大 batch 同时改动分辨率、模型或增强参数。
- 降低物理 batch 后继续保留 `nbs=64`，由框架进行梯度累积。
- 若日志出现 `NaN`、匈牙利匹配异常或 loss 突然失效，优先将 `amp=False` 重跑；这是 RT-DETR 混合精度训练需要特别检查的问题。

显存占用还会受 PyTorch/CUDA 版本、数据增强和同卡其他进程影响，因此 batch 数值应视为 24 GB RTX 4090 的起始建议，而不是硬性保证。

## 6. 统一验证与测速

训练完成后，使用各自的 `best.pt` 在对应分辨率上重新验证：

```bash
yolo detect val model=runs/rtdetr_baseline/rtdetr_l_visdrone_img640_s0/weights/best.pt data=datasets/VisDrone-local.yaml imgsz=640 batch=1 device=0 half=True max_det=300 plots=True

yolo detect val model=runs/rtdetr_baseline/rtdetr_l_visdrone_img960_s0/weights/best.pt data=datasets/VisDrone-local.yaml imgsz=960 batch=1 device=0 half=True max_det=300 plots=True
```

精度和速度应分开汇报。测速时两组都使用：

- 同一张 RTX 4090；
- `batch=1`；
- FP16；
- 相同软件环境；
- 相同验证集；
- 不包含模型加载和首次 CUDA 初始化时间。

## 7. 结果记录表

| 实验 | imgsz | 物理 batch | 最佳 epoch | mAP50-95 | mAP50 | mAP75 | mAP-S | Precision | Recall | 推理时间 ms/image | 峰值显存 GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 RT-DETR-L | 640 | 8/实际值 |  |  |  |  |  |  |  |  |  |
| R2 RT-DETR-L | 960 | 2/实际值 |  |  |  |  |  |  |  |  |  |

VisDrone 以密集小目标为主，重点观察 `mAP50-95`、`mAP-S` 和 Recall。预计 960 更有利于小目标，但计算量和显存会显著增加；最终结论应同时报告精度、速度和显存，而不能只比较 mAP。

## 8. 可选重复实验

如果该 baseline 要写入论文主表，建议在主实验完成后使用 `seed=0, 1, 2` 重复两组实验，并报告均值和标准差。算力有限时，可以先完成单种子筛选，再只对最终采用的设置补三种子实验。
