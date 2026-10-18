# 论文严格复刻实验：RT-DETR baseline（VisDrone2019-DET）

本分支 `paper-repro-rtdetr-visdrone` 从 `baseline-visdrone` 拉出，用于**严格复刻论文**中
RT-DETR baseline 的训练配方，验证能否跑出论文里的指标。

## 1. 复刻目标

论文（Zhang et al., *SO-DETR*）Table I 的 RT-DETR baseline 行：

| 模型 | Backbone | 论文 Params | 论文 GFLOPs | 论文 AP | 论文 AP50 |
| --- | --- | ---: | ---: | ---: | ---: |
| RT-DETR-R18 | ResNet18 | 20.0M | 57.3 | 26.7 | 44.6 |
| **RT-DETR-R50** | **ResNet50** | **42.0M** | **129.9** | **28.4** | **47.0** |

本 fork 的 `ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml` 实测 **42.9M** 参数，
与论文 RT-DETR-R50 / 原始 `rtdetr-r50vd` 一致 → **以 RT-DETR-R50 为忠实复刻对象**。

> R18 未纳入：原始 RT-DETR-R18 的 backbone 是 ResNet18（BasicBlock），而本 fork 的
> `ResNetLayer/ResNetBlock` 只实现了 Bottleneck 形态，缺少 `BasicBlock` 模块，无法忠实复刻。

## 2. 严格配方（论文口径）

来自论文正文（350 epochs / batch 4 / patience 40）+ SO-DETR 仓库 V1.0 baseline 规程：

| 参数 | 值 |
| --- | --- |
| epochs | 350 |
| batch | 4 |
| patience | 40 |
| imgsz | 640 |
| optimizer | AdamW |
| lr0 | 1e-4 |
| lrf | 1.0（恒定学习率） |
| weight_decay | 1e-4 |
| warmup_epochs | 3 |
| nbs | 64（自动梯度累积） |
| mixup | 0.2 |
| close_mosaic | 0 |
| seed | 0 |
| max_det | 300 |
| amp | True |
| deterministic | False（`grid_sample` 不支持确定性） |

## 3. 与现有 `baseline-visdrone` 的差别

| 参数 | `baseline-visdrone`（旧） | 本分支（论文口径） |
| --- | --- | --- |
| epochs | 300 | **350** |
| batch | 8 | **4** |
| patience | 100 | **40** |
| lrf | 0.01 | **1.0** |
| mixup | 0.0 | **0.2** |
| close_mosaic | 10 | **0** |
| 模型 | RT-DETR-L | **RT-DETR-R50** |

其中 `lrf=1.0`（恒定 LR）是与旧口径最本质的差异。

## 4. 关键保真度说明（预训练权重）

论文的 RT-DETR baseline 使用 **COCO 预训练**权重（"adheres to the default parameters
from the official implementations"）。但 **Ultralytics 只提供 HGNetv2 版本的 COCO 权重**
（`rtdetr-l.pt`、`rtdetr-x.pt`），**没有 ResNet RT-DETR 的 COCO 权重**。因此：

- 默认：RT-DETR-R50 **从零训练**（`--pretrained` 未指定）。
- 若要严格对齐论文初始化：用 `--pretrained <path/to.pt>` 指定转换好的官方权重。

> 这项差异会影响最终能否达到论文的 AP；从零训练通常会偏低。

## 5. 运行

```bash
cd /home/zjnu/文档/ZHJ_WorkSpace/ultralytics

# 冒烟测试（几分钟）
PYTHONPATH=$PWD python train_paper_repro_visdrone.py --epochs 1 --fraction 0.03 --name smoke

# 完整严格复刻（RT-DETR-R50，论文配方）
PYTHONPATH=$PWD python train_paper_repro_visdrone.py --model r50

# 若拿到 COCO 预训练权重
PYTHONPATH=$PWD python train_paper_repro_visdrone.py --model r50 --pretrained /path/to/rtdetr_r50.pt
```

输出目录：`runs/paper_repro/rtdetr-r50-paper-350e-s0/`。

## 6. 结果记录表

| 实验 | 模型 | 预训练 | epochs | 最佳 epoch | mAP50-95 | mAP50 | P | R |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 论文 | RT-DETR-R50 | COCO | 350 | - | 0.284* | 0.470* | - | - |
| 本复刻 | RT-DETR-R50 | 无/从零 | 350 | | | | | |

（*论文以百分数报告，此处换算为小数。）

## 7. 备注

- 评测口径：论文报告 COCO 风格 AP/AP50；Ultralytics `val()` 的 `mAP50-95` 即 COCO AP，
  若要完全对齐可用 pycocotools 的 `COCOeval` 复核 `best.pt`。
- RT-DETR + AMP 偶发 NaN，如出现可 `--amp False` 重跑。