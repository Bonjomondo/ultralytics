# SAC-YOLO11 实验 Changelog

本文档用于记录 YOLO11s 面向 UAV 小目标检测的结构改造，包括 SAC、P2 检测层、频率关注下采样、
频域增强 Neck、轻量语义对齐和小目标回归损失，以及对应的验证、训练环境、已知问题和实验约束。
后续分析实验或向 AI 提供上下文时，应优先引用本文档，并结合具体训练目录中的 `args.yaml`、
`results.csv`、`weights/best.pt` 和 `weights/last.pt`。

## 当前版本

- 实验版本：`V2.0`
- 日期：`2026-06-14`
- Git 分支：`SAC_YOLOV11`
- Git commit：当前改动尚未提交；V1.0 基线 commit 为 `f9f3e3d96f6ea8b957f6452acd029f8648283367`
- Ultralytics 基线：`v8.4.30`
- 基线模型：YOLO11s Detect
- 参考论文：`UAV-DETR: Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery`
- 目标数据集：VisDrone2019-DET

## V2.0 变更：YOLO11-UAV 小目标完整结构

V2.0 新增独立模型 `yolo11-uav.yaml`，不覆盖 V1.0 的 `yolo11-sac.yaml`。两套配置可以继续独立复现：

- V1.0：标准 P3/P4/P5 YOLO11s，仅在 Detect 前使用 P5 校准 P3。
- V2.0：P2/P3/P4/P5 四尺度检测，并在 Backbone、Neck、特征对齐和回归损失中加入小目标设计。

### 1. 新增 P2 检测层

文件：`ultralytics/cfg/models/11/yolo11-uav.yaml`

检测尺度由：

```text
P3 / 8
P4 / 16
P5 / 32
```

扩展为：

```text
P2 / 4
P3 / 8
P4 / 16
P5 / 32
```

当输入为 `640x640` 时，对应特征图为：

```text
P2: 160x160
P3: 80x80
P4: 40x40
P5: 20x20
```

P2 分支由最终 P3 top-down 特征再次上采样，与 Backbone C2 特征对齐融合后生成。

### 2. 新增 FrequencyFocusedDownsample

文件：`ultralytics/nn/modules/block.py`

新增 `FrequencyFocusedDownsample`（FFD），用于替换部分 `stride=2 Conv`：

```text
输入
├── 3x3 stride=2 Conv 主分支
├── MaxPool + 1x1 Conv 显著响应分支
└── 1x1 Conv + FFT 高通响应 + AvgPool 频域分支
        ↓
      Concat
        ↓
     1x1 Conv
```

设计细节：

- FFT/IFFT 显式使用 FP32，输出再恢复输入 dtype。
- 高频掩码由二维径向频率生成，中心低频权重低、边缘高频权重高。
- `frequency_scale` 初始化为 0，训练初期先依赖普通投影，之后逐步引入高频残差。
- 主 `3x3 stride=2 Conv` 的参数键保持为 `conv/bn`，可迁移标准 YOLO11 对应下采样权重。
- 支持 Ultralytics 的 Conv-BN fuse；加载预训练权重后，融合前后最大输出差值小于 `4e-7`。

放置位置：

- Backbone：P2→P3、P3→P4。
- Neck：P2→P3、P3→P4 的 bottom-up 回流。
- P4→P5 仍使用标准 `Conv stride=2`，避免所有下采样都替换造成过大计算开销。

### 3. 新增 FrequencyEnhancedFusion

文件：`ultralytics/nn/modules/block.py`

新增 `FrequencyEnhancedFusion`，放置在 Neck 每次融合后的 `C3k2` 之后：

- 局部分支使用深度卷积提取空间纹理。
- 频域分支先将通道压缩到约 `1/4`，执行 FFT 高通增强，再投影回原通道数。
- 输出使用残差形式：

```text
output = input + scale * (local_feature + frequency_feature)
```

- `scale` 初始化为 0，因此模块初始为恒等映射，便于稳定加载预训练特征。
- 使用压缩频域通道而非全通道双投影，以控制参数量和 FFT 成本。

### 4. 新增 LightweightAlignedConcat

文件：`ultralytics/nn/modules/block.py`

Top-down Neck 的三个普通 `Concat` 被替换为 `LightweightAlignedConcat`：

- 输入为 top-down 语义特征和同尺寸 lateral 空间特征。
- 两路特征先压缩通道，用轻量卷积预测二维采样偏移和融合门控。
- 使用 `grid_sample` 对 top-down 特征进行空间校准。
- lateral 特征投影到 top-down 通道后，参与语义修正。
- 最终仍按通道拼接，不改变后续 `C3k2` 的使用方式。
- `align_scale` 初始化为 0，初始输出等价于普通 `Concat`；首个优化阶段先学习开启对齐分支。

Bottom-up Neck 保留普通 `Concat`，避免每条路径都使用 `grid_sample`。

### 5. 新增 UAVDetect 和小目标回归损失

文件：

- `ultralytics/nn/modules/head.py`
- `ultralytics/utils/loss.py`
- `ultralytics/nn/tasks.py`

新增 `UAVDetect`：

- 检测计算仍继承标准 `Detect`。
- 输入为 `[P2, P3, P4, P5]`。
- 同时作为模型标记，使 `DetectionModel` 只对该模型启用 `UAVDetectionLoss`。
- 标准 YOLO11、SACDetect、Segment、Pose 和 OBB 的损失逻辑不受影响。

新增 `SmallObjectBboxLoss` 和 `UAVDetectionLoss`：

- 保留原 TaskAlignedAssigner、分类损失和 DFL 框架。
- 将标准 CIoU 回归项替换为 Inner-SIoU。
- 默认 Inner-IoU 缩放比例：`inner_iou_ratio=0.7`。
- 对目标框按像素几何尺寸增加连续权重：

```text
small_weight = 1 + small_box_gain * exp(-sqrt(width * height) / small_box_scale)
```

- 默认 `small_box_gain=0.5`、`small_box_scale=32.0`。
- 同一小目标权重同时作用于 Inner-SIoU 和 DFL。
- 这些参数保存在 `yolo11-uav.yaml` 顶层，可直接做消融。

### 6. 模型解析与模块注册

修改文件：

- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`

已注册：

- `FrequencyFocusedDownsample`
- `FrequencyEnhancedFusion`
- `LightweightAlignedConcat`
- `UAVDetect`

YAML 解析器会：

- 自动按 `n/s/m/l/x` 宽度缩放 FFD 和频域融合模块。
- 为 `LightweightAlignedConcat` 注入两路输入通道数。
- 为 `UAVDetect` 注入 P2/P3/P4/P5 的通道数和标准 Detect 参数。

### 7. V2.0 模型结构与参数量

YOLO11s-UAV 默认 80 类模型：

- 参数量：`11,519,904`
- 输出步长：`[4, 8, 16, 32]`
- 检测尺度：P2、P3、P4、P5

VisDrone 10 类模型：

- YOLO11s baseline：`9,431,662`
- YOLO11s-UAV：`11,471,048`
- 新增参数：`2,039,386`
- 参数增幅：约 `21.62%`

加载 `yolo11s.pt` 的本地日志：

```text
Transferred 264/873 items from pretrained weights
```

迁移比例低于 V1.0 SAC 的原因是 Neck 层级和检测尺度发生了结构性变化；Backbone 未替换层以及 FFD 的主下采样
卷积仍可迁移。新 P2 Head、频域分支、对齐分支和四尺度 Detect 参数从初始化状态开始训练。

### 8. 已完成验证

- FFD 输出尺寸正确，梯度可回传，`frequency_scale` 能获得非零梯度。
- FrequencyEnhancedFusion 初始保持残差输入，`scale` 能获得非零梯度。
- LightweightAlignedConcat 输出通道等于两路输入通道之和，`align_scale` 能获得非零梯度。
- YOLO11s-UAV YAML 可解析，最终 Head 为 `UAVDetect`。
- 四个检测输出步长为 `[4, 8, 16, 32]`。
- `64x64` 测试输入的四尺度特征尺寸为 `16x16`、`8x8`、`4x4`、`2x2`。
- `UAVDetectionLoss` 能自动选择 `SmallObjectBboxLoss`。
- Inner-SIoU 对相同框返回 1，目标框发生空间偏移后相似度下降且保持有限值。
- 合成小目标 batch 的 box/cls/dfl loss 均为有限值，完整反向传播通过。
- 加载预训练权重后，模型 Conv-BN fuse 前后输出最大差值小于 `4e-7`。
- V1.0 的 SAC 前向、梯度和 MuSGD 参数分组测试继续通过。

测试文件：

```text
tests/test_yolo11_uav.py
tests/test_sac.py
```

当前 `uav` Conda 环境未安装 `pytest`，因此本地使用 Python 直接调用测试函数；全部测试通过。

### 9. 推荐训练命令

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11s-uav.yaml")
model.load("./yolo11s.pt")

model.train(
    data="datasets/VisDrone-local.yaml",
    epochs=300,
    imgsz=640,
    batch=8,
    device=0,
    workers=4,
    deterministic=False,
    name="yolo11s_uav_visdrone",
)
```

建议第一轮先保持与 baseline 完全相同的训练参数，只改变模型 YAML 和模型专用 bbox loss，以便测量完整 V2.0
结构相对标准 YOLO11s 的总体收益。

### 10. V2.0 已知限制

- P2 检测层会显著增加高分辨率预测点数量、显存占用和 NMS 压力。
- FFT 和 `grid_sample` 的 ONNX/TensorRT 兼容性尚未验证。
- `grid_sample` CUDA backward 仍可能非确定，训练建议 `deterministic=False`。
- HFF 与对齐模块的残差系数初始化为 0，内部增强参数首步梯度会被门控抑制，先由残差系数学习开启分支。
- FFD 并非整体恒等替换：虽然高频残差从 0 开始且主卷积可迁移，但池化/投影分支与最终融合层是新增参数。
- 当前 Inner-SIoU 和小目标权重为工程实现，尚未通过 VisDrone 多 seed 消融确认最佳超参数。
- 尚未得到 V2.0 的完整训练精度、峰值显存、推理延迟、FPS、mAP_s/mAP_m/mAP_l 和逐类 AP。

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
