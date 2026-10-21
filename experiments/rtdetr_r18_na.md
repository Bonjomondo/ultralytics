# RT-DETR-R18 的 P4-NA5 实验

本分支在纯 RT-DETR-R18 的最终融合 P4 到 Decoder 的支路加入零初始化门控的局部注意力，先运行 R1。

## 启动 R1

在仓库根目录、已安装本项目依赖的 Python 环境中运行。默认数据配置指向服务器上已有的 VisDrone 数据集；本地运行时用 `--data` 指定本地数据配置。

依赖以根目录 `pyproject.toml` 为准，包括 checkpoint 保存所需的 `polars`。

先验证训练和验证流程：

```bash
PYTHONPATH="$PWD" python train_rtdetr_r18_na.py \
  --variant r1 --device 0 --epochs 1 --fraction 0.01 \
  --batch 2 --workers 2 --name r1-smoke
```

正式筛选：

```bash
PYTHONPATH="$PWD" python train_rtdetr_r18_na.py \
  --variant r1 --device 0 --name rtdetr-r18-r1-150e-s0
```

默认训练 150 epoch、640 输入、batch 8、seed 0、AdamW（betas 0.9/0.999）、学习率及 weight decay 均为 1e-4；AMP 关闭、bias warmup LR 为 0、early stopping 关闭。沿用现有流程的 nbs=64、warmup=3 epoch、mixup=0.2、constant LR；梯度累积会使有效 batch 与物理 batch 不同。150 epoch 用于筛选，不能假定已经收敛。

## 结构和配对初始化

R18 backbone 使用 ResNet-vd 的三层 3×3 stem，以及 `[2,2,2,2]` 个 BasicBlock，输出通道为 128/256/512。融合块 expansion=0.5，P5 保留 AIFI，Decoder 为三层。这里适配的是本仓库的 Ultralytics 训练与算子实现，训练协议和权重格式不宣称与官方实现逐项等同。

参考：[官方 R18 配置](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml)、[官方 ResNet-vd 实现](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/nn/backbone/presnet.py)。

```text
融合 P4（第19层） ────────────────────────────────────────┐
                 └ LN → 256→128 → NA5/4 heads → 128→256 → γ → 加法 → Decoder
                 └ 原有下采样 → 融合 P5 → Decoder
```

`ResidualLocalRefine` 不含 FFN，R1 关闭 RPB，γ 初始为零。注意力使用带有效邻域 mask 的 F.unfold 实现；它是固定中心窗口、边缘裁剪的局部注意力，没有使用 NATTEN/FlashAttention。它不会修改 P4 到 P5 的原融合路径。

R0 的第23层是参数为空的 Identity，R1 的同一层是 NA，Decoder 都位于第24层。专用 Trainer 在真正创建训练模型时，从固定 seed 的 R0 复制所有共有参数，避免新增模块消耗随机数后造成 Decoder 初值变化。该初始化仅由本实验训练入口保证；直接使用普通 `RTDETR(yaml).train()` 不保证这项配对。

`initialization.json` 保存共有状态 SHA256、seed 和初始化来源。相同 seed、类别数、代码版本和初始化策略的 R0/R1，其共有状态 SHA256 应相同。

当前默认从零训练。`--pretrained` 只接受与本实验 R18 backbone 和模块命名兼容的本地 Ultralytics `.pt`；它向指定结构加载权重，不替换成 checkpoint 自带结构，也不直接兼容官方 Paddle/PyTorch state_dict。

## 查看训练健康状况

训练目录中的 `health.jsonl` 每个 epoch 写入：

| 字段 | 含义 |
| --- | --- |
| `gamma` | epoch 结束时的门控值 |
| `gamma_grad_abs` / `na_qkv_grad_abs` | 前三个训练 batch 的梯度绝对值均值；AMP 开启时为缩放后的梯度 |
| `last_bbox_relu_active_ratio` | 最后一层回归 MLP 的最后隐藏层 ReLU 激活比例 |
| `last_bbox_delta_logit_abs` | 最后一层回归头原始 logit 更新量绝对值均值 |
| `last_bbox_step_abs` | 最后两层 Decoder 输出框的绝对变化均值 |
| `train_seconds` | 训练耗时，不含验证和保存 checkpoint |
| `train_peak_allocated_mib` | 训练期间 PyTorch 的峰值已分配显存，不含验证 |
| `mean_train_seconds_first5` | 第五轮记录的前五轮平均训练耗时 |

γ=0 时首次反传的 NA 权重梯度为零是预期行为；要结合 γ 更新后的梯度判断。回归头的最后线性层本来也是零初始化，首次出现零更新量不能直接判断为失活。非有限诊断值记录为 null，需要结合训练 loss 排查。

## 补跑 R0 和正式评估

```bash
PYTHONPATH="$PWD" python train_rtdetr_r18_na.py \
  --variant r0 --device 0 --name rtdetr-r18-r0-150e-s0

PYTHONPATH="$PWD" python eval_coco.py \
  --weights runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/weights/best.pt \
  --data configs/visdrone-local.yaml --device 0
```

实际训练输出目录以启动日志中的 save_dir 为准。AP/APs 使用相同 COCOeval 脚本和设置；R1 单独训练只能验证可训练性与成本，结构增益要与同协议 R0 比较。有效候选再做完整周期及多 seed 验证。

中断后可恢复同一变体：

```bash
PYTHONPATH="$PWD" python train_rtdetr_r18_na.py --variant r1 \
  --resume runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/weights/last.pt
```

## 运行代码检查

```bash
PYTHONPATH="$PWD" python -m unittest discover -s tests -p test_rtdetr_r18_na.py -v
```

检查覆盖注意力边缘语义、γ 从零开始的梯度、R18 BasicBlock、R0/R1 共有初始化、P4 支路隔离、真实检测 loss 的反传、checkpoint 回读以及 CLI 参数。原有 `train_paper_repro_visdrone.py` 也修复了 `--amp False` 的解析和预训练权重绕过自定义 YAML 的问题，其训练配方默认值保持不变。

2026-10-20 验证记录：在 RTX 4090、Python 3.11.17、PyTorch 2.1.2+cu121 环境中，10 项代码检查通过。R1 使用 65 张训练图、完整 548 张验证图、640 输入、batch 8、FP32 完成一轮训练及 checkpoint 保存；训练峰值已分配显存为 7277 MiB。γ 从 0 更新到约 1.30e-5，诊断采样中的最后回归隐藏层激活比例约 0.515。该小样本验证用于确认流程和梯度，不用于比较检测精度。R1 的未融合模型有 20,226,425 个参数，其中 NA 分支新增 132,097 个。

## 2026-10-21：完成版 R1 的 detrscope 真实验收

在 UAV_CV / RTX 4090 上，R1 150 轮 EMA checkpoint 已完成全量 548 图 / 38,759 GT 的观测、逐层分析及正式 COCOeval；274 个 batch 的原生/观测/恢复输出逐值一致，模型状态与 Hook 清理核验通过。原始数据独立重建的 557 个分析文件 SHA、预测 JSON、统计结果及报告均一致。内部探针、恒等/非恒等干预另覆盖 8 图。

正式 COCOeval AP/AP50/APs = **18.702% / 32.431% / 12.610%**。原生 Validator 内部 mAP=20.699%，而将其自身预测交给 COCOeval 得到 AP=18.705%；内部指标与正式 AP 应分别记录，差异尚未完全拆解。全量 3 层回归 delta 均有 Query 变化；8 图 D3 隐藏激活非全零，不能套用其他模型的 bias-only 结论。

验收时 R0 仍在训练，未用其未完成 checkpoint 对照 R1，也未据此宣称 NA 有稳定结构收益。共有初始化 SHA 已核对。详细结论、边界、失败进程记录与后续假设集中在 [实验报告](r18_na_ablation/detrscope_20261021/REPORT.md)；网页端协同从 [ChatGPT 上下文](r18_na_ablation/detrscope_20261021/CHATGPT_CONTEXT.md) 开始。大文件仍保留在原 run 目录，轻量证据和复现源码纳入 Git。
