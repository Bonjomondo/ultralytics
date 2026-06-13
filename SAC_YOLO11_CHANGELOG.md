# SAC-YOLO11 实验 Changelog

本文档用于记录 YOLO11s 集成 UAV-DETR Semantic Alignment Calibration（SAC）的实现、验证、训练环境、
已知问题和实验约束。后续分析实验或向 AI 提供上下文时，应优先引用本文档，并结合具体训练目录中的
`args.yaml`、`results.csv`、`weights/best.pt` 和 `weights/last.pt`。

## 当前版本

- 实验版本：`V1.0`
- 日期：`2026-06-13`
- Git 分支：`SAC_YOLOV11`
- Git commit：`f9f3e3d96f6ea8b957f6452acd029f8648283367`
- Ultralytics 基线：`v8.4.30`
- 基线模型：YOLO11s Detect
- 参考论文：`UAV-DETR: Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery`
- 目标数据集：VisDrone2019-DET

## V1.0 变更

### 1. 新增 FFM

文件：`ultralytics/nn/modules/block.py`

新增 `FFM`（Frequency-Focused Modulation）：

- 使用两个 `1x1 Conv2d` 分别生成空间分支和频域分支。
- 频域分支使用 `torch.fft.fft2` 和 `torch.fft.ifft2`。
- FFT 计算显式转为 FP32，结果再恢复输入 dtype，以提高 AMP 训练稳定性。
- 使用可学习参数 `alpha` 和 `beta` 融合频域增强结果与原始输入。
- `alpha` 初始化为 0，`beta` 初始化为 1，因此 FFM 初始接近恒等映射。

### 2. 新增 SemanticAlignmentCalibration

文件：`ultralytics/nn/modules/block.py`

新增 `SemanticAlignmentCalibration`：

- 输入为 `[P3, P5]`，其中 P3 提供高分辨率空间信息，P5 提供深层语义信息。
- 将 P5 投影到 P3 通道数，并双线性上采样到 P3 尺寸。
- 使用 FFM 和门控卷积融合空间域与频域语义特征。
- 使用分组偏移和 `grid_sample` 分别校准空间特征与语义特征。
- 默认分组数为 2，P3 通道数必须可被分组数整除。
- 偏移卷积最后一层使用零初始化，使初始采样网格为恒等网格。
- 使用 `align_corners=False`，像素偏移归一化系数为 `(2/W, 2/H)`。
- 使用 `padding_mode="border"`，避免边界采样引入大面积零值。
- 偏移通道布局保持为：全部空间组偏移、全部语义组偏移、两个融合权重。

原始参考代码中的类名为 `SemanticAlignmenCalibration`，本项目修正为
`SemanticAlignmentCalibration`。

### 3. 新增 SACDetect

文件：`ultralytics/nn/modules/head.py`

新增继承自标准 `Detect` 的 `SACDetect`：

- Detect 输入仍为 `[P3, P4, P5]`。
- 检测前使用 P5 对 P3 执行 SAC 校准。
- P4 和 P5 检测分支保持原 YOLO11 结构。
- 标准 Detect 参数名和层索引保持不变，便于迁移 `yolo11s.pt` 检测头权重。
- 新增可学习标量 `sac_scale`，初始化为 0。
- 实际 P3 输出为：

```text
P3_out = P3 + sac_scale * (SAC(P3, P5) - P3)
```

因此加载 YOLO11s 预训练权重后，模型初始行为与标准 YOLO11s 一致；训练过程中再逐步启用 SAC。

### 4. 新增 YOLO11-SAC 模型配置

文件：`ultralytics/cfg/models/11/yolo11-sac.yaml`

- Backbone 和 PAN-FPN 保持标准 YOLO11。
- 最后一层由 `Detect` 替换为 `SACDetect`。
- 检测输入层索引为 `[16, 19, 22]`，分别对应 P3、P4、P5。
- 使用文件名 `yolo11s-sac.yaml` 时，Ultralytics 会解析统一配置
  `yolo11-sac.yaml` 并选择 `s` 规模。

### 5. 注册模型模块

修改文件：

- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`

注册以下模块，使 YAML 解析器能够构建模型：

- `FFM`
- `SemanticAlignmentCalibration`
- `SACDetect`

### 6. 修复 MuSGD 参数分组

文件：`ultralytics/engine/trainer.py`

原始逻辑：

```python
if param.ndim >= 2 and use_muon:
```

修复后：

```python
if param.ndim in {2, 4} and use_muon:
```

原因：

- Muon 的 Newton-Schulz 更新仅支持二维矩阵。
- 四维卷积核会在更新前展平为二维矩阵。
- SAC 的 `alpha` 和 `beta` 为三维参数 `[C, 1, 1]`。
- 原逻辑错误地将三维参数分配给 Muon，导致：

```text
AssertionError: assert len(G.shape) == 2
```

修复后，三维 SAC 参数由 MuSGD 中的普通 SGD 分支更新，二维和四维权重继续使用 Muon。

## 模型结构与初始化

- YOLO11s-SAC 参数量：`10,268,609`
- 输出步长：`[8, 16, 32]`
- 检测尺度：P3、P4、P5
- SAC 作用位置：最终 P3 检测特征与最终 P5 检测特征之间
- SAC 初始残差系数：`sac_scale = 0`
- 标准 YOLO11s YAML 与 SAC YAML 加载相同预训练权重后，初始化输出最大差值：`0.0`

首次显式执行：

```python
model = YOLO("ultralytics/cfg/models/11/yolo11s-sac.yaml")
model.load("yolo11s.pt")
```

本地验证日志为 `Transferred 499/527 items`。未迁移项为新增 SAC 状态。训练器内部再次构建模型时可能显示
`Transferred 521/527 items`，这是已加载自定义模型向训练模型再次迁移的日志，不代表 SAC 权重来自
YOLO11s。

## 数据集状态

数据目录：

```text
datasets/VisDrone/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

检查结果：

| Split | Images | Labels |
|---|---:|---:|
| train | 6471 | 6471 |
| val | 548 | 548 |
| test | 1610 | 1610 |

- 标注框总数：`457,066`
- 类别 ID：`0-9`
- 非法标注行：`0`
- 空标签文件：`0`
- 图片与标签缺失配对：`0`
- Ultralytics 扫描时发现 4 个重复框，已自动去重，不阻塞训练。

类别：

```text
0 pedestrian
1 people
2 bicycle
3 car
4 van
5 truck
6 tricycle
7 awning-tricycle
8 bus
9 motor
```

服务器数据 YAML 使用绝对路径，避免官方 `VisDrone.yaml` 根据全局 `datasets_dir` 误判数据不存在并重新下载：

```yaml
path: /public/home/zhaohaojun/jhupload/SAC_YOLO/ultralytics/datasets/VisDrone
train: images/train
val: images/val
test: images/test
```

VisDrone 的三个 ZIP 文件不参与训练。确认解压数据完整后可删除以释放约 `1.81 GiB`；仅在需要保留原始
数据备份或重新转换标注时有用。`train.cache` 和 `val.cache` 建议保留。

## 已验证项目

- SAC 独立前向输出尺寸正确。
- 梯度能够回传至 P3 和 P5 输入。
- YOLO11s-SAC YAML 能正常解析和构建。
- 训练模式可生成 box 和 class 原始预测。
- 标准 YOLO11s 预训练参数键与 SACDetect 的标准 Detect 参数兼容。
- `sac_scale=0` 时，加载相同权重后的标准模型与 SAC 模型输出一致。
- MuSGD 修复后，Muon 参数组只包含二维或四维参数。
- FFM 的三维 `alpha/beta` 不再进入 Muon 参数组。
- SAC 模型反向传播和一次 `MuSGD.step()` 已通过。
- 远端 RTX 4090 环境 AMP 检查通过。

## 训练环境

已观察到的远端环境：

```text
OS/容器：Linux
Python：3.10.13
PyTorch：2.5.1+cu124
GPU：NVIDIA GeForce RTX 4090, 24217 MiB
Ultralytics：8.4.30
```

必须从仓库根目录运行：

```text
~/jhupload/SAC_YOLO/ultralytics
```

并确保导入的是本地源码：

```bash
python -m pip install -e .
python -c "import ultralytics; print(ultralytics.__file__)"
```

正确路径应指向：

```text
.../SAC_YOLO/ultralytics/ultralytics/__init__.py
```

## 推荐训练命令

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11s-sac.yaml")
model.load("yolo11s.pt")

model.train(
    data="datasets/VisDrone-local.yaml",
    epochs=300,
    imgsz=640,
    batch=8,
    device=0,
    workers=4,
    deterministic=False,
    name="yolo11s_sac_visdrone",
)
PY
```

使用已修复的 `trainer.py` 时，`optimizer="auto"` 可以继续选择 MuSGD。若远端代码尚未同步 MuSGD 修复，
临时使用：

```python
optimizer="SGD"
```

## 已遇到的问题

### 错误：`KeyError: 'SACDetect'`

原因：调用了 `site-packages` 中的官方 Ultralytics，而不是当前修改后的仓库源码。

处理：

```bash
python -m pip install -e .
```

### 错误：`ImportError: cannot import name 'YOLO' from 'ultralytics'`

原因：在仓库上一级 `~/jhupload/SAC_YOLO` 执行，目录名与 Python 包名冲突。

处理：进入真正的仓库根目录。

```bash
cd ~/jhupload/SAC_YOLO/ultralytics
```

### 数据集被错误地重新下载

原因：官方 `VisDrone.yaml` 的 `path: VisDrone` 相对于 Ultralytics 全局 `datasets_dir` 解析，而服务器数据
存放在仓库内部。

处理：使用带绝对 `path` 的 `datasets/VisDrone-local.yaml`。

### MuSGD 在第一个 epoch 崩溃

错误：

```text
AssertionError
zeropower_via_newtonschulz5:
assert len(G.shape) == 2
```

原因：三维 `alpha/beta` 被错误分配给 Muon。

处理：将 Muon 参数筛选条件改为 `param.ndim in {2, 4}`，或临时指定 `optimizer="SGD"`。

### `grid_sampler_2d_backward_cuda` 非确定性警告

该信息是警告而非错误。SAC 使用 `grid_sample`，其 CUDA backward 没有确定性实现。建议实验训练设置：

```python
deterministic=False
```

并记录随机种子。需要严格复现实验时，应保留相同 CUDA、PyTorch、GPU、batch size 和数据加载设置，
但仍不能保证 `grid_sample` CUDA backward 逐位一致。

## 已知限制

- SAC 使用 `torch.fft`，ONNX/TensorRT 导出尚未验证，可能需要替换或分解 FFT 算子。
- `grid_sample` 可能限制部分部署后端。
- 当前只验证了 Detect 任务，没有为 Segment、Pose 或 OBB 接入 SAC。
- 当前尚无完整 300 epoch 的精度、速度和显存结论。
- `sac_scale` 初始化为 0 时，SAC 内部参数首个优化步骤的梯度会被残差门控抑制；首步主要更新
  `sac_scale`，之后 SAC 分支逐步获得有效梯度。
- 当前实现只校准 P3，未对 P4 做额外语义对齐。

## 后续实验建议

每组实验至少记录：

- Git commit。
- 模型 YAML。
- 初始化权重。
- optimizer、lr、batch、imgsz、epochs。
- 随机种子与 `deterministic`。
- GPU、PyTorch、CUDA 版本。
- 参数量、训练显存和单 epoch 时间。
- `mAP50-95`、`mAP50`、`mAP75`。
- `mAP_s`、`mAP_m`、`mAP_l`，重点关注小目标指标。
- 各类别 AP，特别是 pedestrian、people、bicycle 和 motor。
- 推理延迟和 FPS。

建议消融：

1. 标准 YOLO11s。
2. YOLO11s + SAC，但关闭 FFM。
3. YOLO11s + FFM，但关闭偏移校准。
4. 完整 YOLO11s-SAC。
5. `sac_scale` 初始化为 `0`、`0.1` 和 `1.0`。
6. SAC groups 为 `1`、`2` 和 `4`。
7. SGD、AdamW 和修复后的 MuSGD。
8. 输入尺寸 `640`、`960` 和 `1280`。

## 提供给 AI 的最小上下文

```text
项目基于 Ultralytics v8.4.30，在 YOLO11s Detect 上集成 UAV-DETR 的 Semantic Alignment
Calibration。实现位于 block.py，包含 FFM 和 SemanticAlignmentCalibration；head.py 中新增
SACDetect，在最终 Detect 前使用 P5 校准 P3，P4/P5 保持不变。SACDetect 继承标准 Detect，
并使用初始化为 0 的 sac_scale 做残差插值，因此加载 yolo11s.pt 后初始输出与标准 YOLO11s
一致。模型配置为 ultralytics/cfg/models/11/yolo11-sac.yaml，使用
yolo11s-sac.yaml 名称选择 s scale。

VisDrone 数据为 6471 train、548 val、1610 test，10 类，YOLO 标签已转换完成。远端环境是
Python 3.10.13、PyTorch 2.5.1+cu124、RTX 4090、Ultralytics 8.4.30。

曾出现 MuSGD AssertionError，因为原 trainer.py 将 ndim>=2 的参数全部送入 Muon，而 FFM
alpha/beta 是三维参数。已将条件修复为 param.ndim in {2, 4}，三维参数使用普通 SGD。
SAC 使用 torch.fft 和 grid_sample，AMP 已通过，但 ONNX/TensorRT 尚未验证，
grid_sample CUDA backward 非确定性，训练建议 deterministic=False。

当前版本 commit 为 f9f3e3d96f6ea8b957f6452acd029f8648283367。完整 300 epoch 结果尚未产生。
```

## 实验结果追加模板

```markdown
### EXP-YYYYMMDD-序号

- Git commit:
- 目的:
- 相对基线的唯一改动:
- Model:
- Weights:
- Data:
- Device:
- Epochs:
- Image size:
- Batch:
- Optimizer:
- LR:
- Seed:
- Deterministic:
- Parameters:
- Peak GPU memory:
- Training time:
- Best epoch:
- mAP50-95:
- mAP50:
- mAP75:
- mAP_s:
- mAP_m:
- mAP_l:
- Per-class AP:
- Inference latency/FPS:
- 结论:
- 异常与备注:
```
