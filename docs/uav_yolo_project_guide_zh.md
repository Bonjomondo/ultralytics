# Ultralytics 项目教程：无人机小目标检测 + 轻量化部署

本文面向这样的目标：

- 想快速改 YOLO 架构
- 想先把实验尽快跑起来
- 后面要导出 ONNX / TensorRT，部署到无人机或边缘端
- 希望尽量少改代码，把精力放在论文创新点上

本文基于当前仓库版本 `ultralytics==8.4.30`。

## 1. 先说结论：这个项目适不适合你

非常适合。

原因很直接：

- 训练、验证、预测、导出、部署后端都在一个仓库里
- 模型结构主要由 YAML 驱动，很多结构改动不用大改 Python
- 已经内置了小目标相关的 `P2` 模型配置
- 已经内置了 `VisDrone.yaml`、`DOTAv1.yaml`、`xView.yaml` 等空天/小目标相关数据集入口
- 已经内置了 ONNX、TensorRT、OpenVINO、NCNN、RKNN 等导出和推理后端

如果你的目标是“先跑通一个强 baseline，再加一个论文模块，再顺手导出部署”，这个仓库就是比较省事的路线。

## 2. 项目整体结构

仓库顶层目录可以先这样理解：

```text
ultralytics/
├─ ultralytics/          核心源码
├─ docs/                 官方文档源码
├─ examples/             推理/导出/跨语言调用示例
├─ tests/                测试
├─ docker/               各类 Dockerfile
├─ README.md             英文说明
├─ README.zh-CN.md       中文说明
└─ pyproject.toml        包配置、依赖、CLI 入口
```

`ultralytics/` 这个源码目录又可以再拆成几层：

```text
ultralytics/
├─ cfg/                  配置系统
│  ├─ __init__.py        CLI 入口：yolo train/val/predict/export
│  ├─ default.yaml       全局默认超参数
│  ├─ datasets/          数据集 YAML
│  ├─ models/            模型 YAML
│  └─ trackers/          跟踪器配置
├─ engine/               统一训练/验证/推理/导出引擎
├─ models/               按任务封装的 Trainer/Validator/Predictor
├─ nn/                   真正的网络定义与推理后端
├─ data/                 数据集、增强、DataLoader
├─ utils/                loss、metrics、NMS、工具函数
├─ trackers/             ByteTrack / BoT-SORT
├─ solutions/            各类现成应用方案
├─ hub/                  Ultralytics HUB 相关
└─ optim/                优化器扩展
```

## 3. 每个部分分别是什么功能

下面这张表是最实用的“改代码导航图”。

| 路径 | 作用 | 你什么时候会改它 |
| --- | --- | --- |
| `pyproject.toml` | 依赖、打包、CLI 脚本入口 `yolo` | 安装方式、开发模式 |
| `ultralytics/cfg/__init__.py` | 命令行入口，解析 `yolo train ...` 这类命令 | 一般不改，想理解 CLI 流程时看 |
| `ultralytics/cfg/default.yaml` | 默认训练/验证/导出参数 | 了解常用超参数时看 |
| `ultralytics/cfg/datasets/*.yaml` | 数据集定义 | 训练自定义数据集时一定会改 |
| `ultralytics/cfg/models/*.yaml` | 模型结构蓝图 | 改 backbone/neck/head 时最先改 |
| `ultralytics/engine/model.py` | `YOLO()` 统一 API，串联 train/val/predict/export | 想看总流程时看 |
| `ultralytics/engine/trainer.py` | 通用训练框架 | 只有做深度训练逻辑改造时才会改 |
| `ultralytics/engine/validator.py` | 通用验证框架 | 一般不改 |
| `ultralytics/engine/predictor.py` | 通用推理框架 | 自定义预测流程时改 |
| `ultralytics/engine/exporter.py` | ONNX / TensorRT / OpenVINO 导出 | 做部署和排查导出问题时看 |
| `ultralytics/models/yolo/detect/train.py` | 检测任务 Trainer | 检测训练细节优先看这里 |
| `ultralytics/models/yolo/detect/val.py` | 检测任务 mAP 评估 | 看评估流程、保存结果时看 |
| `ultralytics/models/yolo/detect/predict.py` | 检测任务后处理 | 改 NMS、结果封装时看 |
| `ultralytics/models/yolo/model.py` | `YOLO/YOLOWorld/YOLOE` 任务映射 | 理解任务分发时看 |
| `ultralytics/nn/tasks.py` | 从 YAML 解析并搭建模型，加载权重 | 改网络结构时最核心文件 |
| `ultralytics/nn/modules/conv.py` | 基础卷积模块 | 自定义轻量模块时会改 |
| `ultralytics/nn/modules/block.py` | C2f、C3k2、SPPF、PSA 等骨干块 | 论文模块最常加在这里 |
| `ultralytics/nn/modules/head.py` | Detect/Segment/Pose/OBB head | 改检测头时会改 |
| `ultralytics/data/build.py` | 构建 DataLoader / dataset | 一般不改 |
| `ultralytics/data/dataset.py` | YOLODataset 定义 | 改标签读取逻辑、样本组织时改 |
| `ultralytics/data/augment.py` | Mosaic、MixUp、LetterBox 等增强 | 改增强策略时改 |
| `ultralytics/utils/loss.py` | 检测/分割/姿态 loss | 创新点在损失函数时改 |
| `ultralytics/utils/tal.py` | 正负样本分配、anchor 点逻辑 | 创新点在 assigner 时改 |
| `ultralytics/utils/metrics.py` | mAP 等评价指标 | 一般不改 |
| `ultralytics/utils/nms.py` | NMS 后处理 | 改后处理时改 |
| `ultralytics/nn/autobackend.py` | 自动切换 PyTorch / ONNX / TensorRT 推理后端 | 做部署时重点看 |
| `ultralytics/nn/backends/*` | 各后端推理实现 | 排查 ONNX/TensorRT 推理问题时看 |
| `examples/` | 各平台和语言示例 | 导出后落地时参考 |
| `tests/` | 回归测试 | 你改了核心结构后建议补测 |

## 4. 这个项目的整体架构

可以把它看成 5 层：

```text
命令层
  yolo train / yolo val / yolo predict / yolo export

调度层
  ultralytics/cfg/__init__.py
  ultralytics/engine/model.py
  ultralytics/models/yolo/model.py

任务层
  ultralytics/models/yolo/detect/*.py
  ultralytics/models/yolo/segment/*.py
  ultralytics/models/yolo/pose/*.py

网络层
  ultralytics/nn/tasks.py
  ultralytics/nn/modules/*.py
  ultralytics/cfg/models/*.yaml

数据与部署层
  ultralytics/data/*.py
  ultralytics/engine/exporter.py
  ultralytics/nn/autobackend.py
  ultralytics/nn/backends/*.py
```

### 4.1 训练调用链

以 `yolo train model=yolo11n.pt data=VisDrone.yaml` 为例：

```text
yolo 命令
-> ultralytics/cfg/__init__.py
-> ultralytics.models.yolo.model.YOLO
-> ultralytics.engine.model.Model.train()
-> ultralytics.models.yolo.detect.train.DetectionTrainer
-> ultralytics.engine.trainer.BaseTrainer
-> ultralytics.data.build / dataset / augment
-> ultralytics.nn.tasks.parse_model()
-> ultralytics.nn.modules.*
-> ultralytics.utils.loss.py
-> ultralytics.models.yolo.detect.val.DetectionValidator
```

### 4.2 模型搭建调用链

```text
模型 YAML
-> ultralytics/nn/tasks.py: yaml_model_load()
-> ultralytics/nn/tasks.py: parse_model()
-> ultralytics/nn/modules/conv.py
-> ultralytics/nn/modules/block.py
-> ultralytics/nn/modules/head.py
```

也就是说：

- 改结构，第一选择通常是改 `cfg/models/*.yaml`
- YAML 里用到的新模块，才需要补到 `nn/modules/*.py`
- 模块加完后，还要在 `nn/modules/__init__.py` 和 `nn/tasks.py` 里暴露出来

### 4.3 导出部署调用链

```text
model.export(...)
-> ultralytics/engine/exporter.py
-> 导出为 onnx / engine / openvino ...

部署推理
-> ultralytics/nn/autobackend.py
-> ultralytics/nn/backends/onnx.py
-> ultralytics/nn/backends/tensorrt.py
-> 统一返回结果给 predictor
```

这也是为什么这个仓库很适合“训练和部署一体化”。

## 5. 你这个场景下，最推荐的使用思路

你的目标不是做一个大而全的平台，而是：

1. 先跑出无人机小目标检测 baseline
2. 再做一个尽量少改代码的结构创新
3. 最后导出 ONNX / TensorRT 上机

所以最推荐走下面这条路线。

### 路线 A：最快 baseline

适合先确认数据、环境、训练链路没有问题。

```bash
yolo train model=yolo11n.pt data=VisDrone.yaml imgsz=960 epochs=100 batch=16 device=0
```

特点：

- 几乎零学习成本
- 训练、验证、导出都很稳
- 适合作为第一条能跑通的 baseline

### 路线 B：更适合小目标的 baseline

适合你的无人机场景，更推荐。

```bash
yolo train model=ultralytics/cfg/models/26/yolo26n-p2.yaml pretrained=yolo26n.pt data=VisDrone.yaml imgsz=1024 epochs=150 batch=16 device=0
```

为什么推荐它：

- `P2` 检测头会保留更高分辨率特征，对小目标更友好
- 还是 YAML 驱动，代码改动很少
- 后续导出 ONNX / TensorRT 也顺

如果你想继续用 `v8` 风格，也可以用：

```bash
yolo train model=ultralytics/cfg/models/v8/yolov8n-p2.yaml pretrained=yolov8n.pt data=VisDrone.yaml imgsz=1024 epochs=150 batch=16 device=0
```

## 6. 先把实验跑起来：详细上手教程

### 6.1 安装

建议在仓库根目录使用开发模式安装。

只做训练和推理：

```bash
pip install -e .
```

后面要导出 ONNX / TensorRT：

```bash
pip install -e ".[export]"
```

如果你还想加常见增强库：

```bash
pip install -e ".[extra]"
```

### 6.2 先做一次烟雾测试

先确认仓库能正常推理：

```bash
yolo predict model=yolo11n.pt source=ultralytics/assets/bus.jpg device=0
```

如果这一步能跑通，说明：

- CLI 入口正常
- 权重下载正常
- 推理流程正常

### 6.3 先跑内置的无人机相关数据集

这个仓库内置了：

- `ultralytics/cfg/datasets/VisDrone.yaml`
- `ultralytics/cfg/datasets/DOTAv1.yaml`
- `ultralytics/cfg/datasets/xView.yaml`

其中对你最直接的是 `VisDrone.yaml`。

第一次训练时，如果本地没有数据，它会按 YAML 里的 `download` 段自动下载并转换成 YOLO 格式。

直接跑：

```bash
yolo train model=yolo11n.pt data=VisDrone.yaml imgsz=960 epochs=100 batch=16 device=0
```

### 6.4 如果你用自己的无人机数据集

你只需要先准备一个数据集 YAML，例如 `datasets/my_uav.yaml`：

```yaml
path: C:/datasets/my_uav
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: car
  2: bicycle
```

目录建议这样放：

```text
my_uav/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/
```

标签格式就是标准 YOLO 检测格式：

```text
class x_center y_center width height
```

### 6.5 跑你的第一个正式 baseline

推荐先跑两组：

组 1，普通轻量 baseline：

```bash
yolo train model=yolo11n.pt data=datasets/my_uav.yaml imgsz=960 epochs=100 batch=16 device=0 project=runs/uav name=yolo11n_baseline
```

组 2，小目标 baseline：

```bash
yolo train model=ultralytics/cfg/models/26/yolo26n-p2.yaml pretrained=yolo26n.pt data=datasets/my_uav.yaml imgsz=1024 epochs=150 batch=16 device=0 project=runs/uav name=yolo26n_p2_baseline
```

建议你先别急着改结构，先把这两组跑出来。很多论文创新点最后其实都是在和这个 baseline 比。

### 6.6 训练结果会保存到哪里

默认在 `runs/` 下，例如：

```text
runs/
└─ detect/
   └─ train/
      ├─ args.yaml
      ├─ results.csv
      ├─ results.png
      ├─ train_batch0.jpg
      └─ weights/
         ├─ best.pt
         └─ last.pt
```

你最常用的是：

- `weights/best.pt`：最佳权重
- `args.yaml`：这次实验的参数快照
- `results.csv`：每个 epoch 的指标
- `results.png`：训练曲线

### 6.7 验证和推理

验证：

```bash
yolo val model=runs/uav/yolo26n_p2_baseline/weights/best.pt data=datasets/my_uav.yaml device=0
```

推理图片：

```bash
yolo predict model=runs/uav/yolo26n_p2_baseline/weights/best.pt source=demo.jpg device=0
```

推理视频：

```bash
yolo predict model=runs/uav/yolo26n_p2_baseline/weights/best.pt source=demo.mp4 device=0
```

## 7. 如果你想“少改代码但能写论文”，最好的切入点

### 7.1 零代码创新：先改 YAML，不改 Python

这是最推荐的第一步。

你可以先做这些变化：

- 换成 `P2` 结构
- 调整 neck 的通道数
- 增加或减少某一层 block 的重复次数
- 替换某些 block 类型
- 调整 head 的输入层级

这类改动通常只需要改 `ultralytics/cfg/models/*.yaml`。

### 7.2 低代码创新：加一个自定义模块

如果论文一定要有“新模块”，尽量只加一个 block，不要一开始就大改训练框架。

最小改动路径如下：

1. 在 `ultralytics/nn/modules/block.py` 或 `conv.py` 中定义模块
2. 在 `ultralytics/nn/modules/__init__.py` 里导出模块
3. 在 `ultralytics/nn/tasks.py` 里把模块 import 进去
4. 在模型 YAML 里使用这个模块

也就是说，常见情况下你只要动 4 个地方：

- `ultralytics/nn/modules/block.py`
- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/cfg/models/你的模型.yaml`

### 7.3 中代码创新：改 loss 或 assigner

如果你的创新点在训练机制：

- 改 loss：看 `ultralytics/utils/loss.py`
- 改正负样本分配：看 `ultralytics/utils/tal.py`

但这条路的风险更大：

- 调参成本高
- 训练更容易不稳定
- 导出虽然通常不受影响，但论文复现会更麻烦

如果你现在最优先的是“先把实验跑起来”，建议先别碰这里。

## 8. 针对无人机小目标检测的实战建议

### 8.1 结构上优先做什么

优先级建议如下：

1. 先用 `P2` 头
2. 提高 `imgsz`
3. 再考虑轻量 block
4. 最后再碰 loss / assigner

原因：

- 无人机小目标最直接的问题通常是目标在高层特征图里太小
- `P2` 和更大的输入尺寸通常比“花哨模块”更稳定
- 这条路线对部署也更友好

### 8.2 参数上优先试什么

建议先试这些，而不是一下子改太多：

- `imgsz=960` 或 `imgsz=1024`
- `epochs=150` 左右
- `close_mosaic=10` 或 `20`
- `batch` 按显存调整

示例：

```bash
yolo train model=ultralytics/cfg/models/26/yolo26n-p2.yaml pretrained=yolo26n.pt data=datasets/my_uav.yaml imgsz=1024 epochs=150 batch=16 close_mosaic=10 device=0
```

### 8.3 轻量化部署时要提前注意什么

如果你后面一定要导出 ONNX / TensorRT，那么前面改结构时要注意：

- 尽量使用标准卷积、深度可分离卷积、Concat、Upsample 这类友好算子
- 少引入非常规算子
- 自定义模块最好由 PyTorch 基本算子组成
- 每加一个新模块，就尽快做一次 ONNX 导出测试

一句话：训练能跑不算结束，能导出才算这个创新点真正可用。

## 9. ONNX / TensorRT 导出教程

### 9.1 导出 ONNX

```bash
yolo export model=runs/uav/yolo26n_p2_baseline/weights/best.pt format=onnx imgsz=1024 dynamic=True simplify=True
```

导出后可以直接验证：

```bash
yolo predict model=runs/uav/yolo26n_p2_baseline/weights/best.onnx source=demo.jpg
```

也可以直接做验证，看看导出后精度有没有明显掉：

```bash
yolo val model=runs/uav/yolo26n_p2_baseline/weights/best.onnx data=datasets/my_uav.yaml batch=1
```

### 9.2 导出 TensorRT

```bash
yolo export model=runs/uav/yolo26n_p2_baseline/weights/best.pt format=engine imgsz=1024 half=True dynamic=True device=0
```

导出后推理：

```bash
yolo predict model=runs/uav/yolo26n_p2_baseline/weights/best.engine source=demo.jpg device=0
```

### 9.3 如果你要做 INT8

```bash
yolo export model=runs/uav/yolo26n_p2_baseline/weights/best.pt format=engine imgsz=1024 int8=True batch=8 data=datasets/my_uav.yaml device=0
```

注意：

- INT8 校准最好在最终部署设备或同类设备上做
- 校准数据要尽量接近真实飞行场景
- INT8 不一定白赚，速度会上去，但精度可能掉

## 10. 你应该怎么学习这个项目

如果你直接从头硬啃整个仓库，会很累。推荐按下面顺序学。

### 第 1 阶段：先会用

目标：能训练、验证、预测、导出。

先只掌握这几个命令：

- `yolo train`
- `yolo val`
- `yolo predict`
- `yolo export`

### 第 2 阶段：先学“配模型”，再学“写模型”

先读：

- `ultralytics/cfg/models/*.yaml`
- `ultralytics/cfg/datasets/*.yaml`

因为这一步最接近你真正做实验时的动作。

### 第 3 阶段：看清训练主链

按这个顺序读：

1. `ultralytics/engine/model.py`
2. `ultralytics/models/yolo/model.py`
3. `ultralytics/models/yolo/detect/train.py`
4. `ultralytics/engine/trainer.py`

学完这几份，你就知道训练是怎么串起来的。

### 第 4 阶段：再去看网络拼装

按这个顺序读：

1. `ultralytics/nn/tasks.py`
2. `ultralytics/nn/modules/__init__.py`
3. `ultralytics/nn/modules/block.py`
4. `ultralytics/nn/modules/head.py`

学完这部分，你就知道：

- YAML 是怎么被解析成模型的
- 自定义模块该挂到哪里
- head 是怎么接多尺度特征的

### 第 5 阶段：最后再看数据增强和部署

按这个顺序读：

1. `ultralytics/data/dataset.py`
2. `ultralytics/data/augment.py`
3. `ultralytics/engine/exporter.py`
4. `ultralytics/nn/autobackend.py`
5. `ultralytics/nn/backends/onnx.py`
6. `ultralytics/nn/backends/tensorrt.py`

这时你就能把“训练”和“部署”连起来看。

## 11. 给你的推荐实验节奏

如果你现在就准备开始，我建议按下面这个节奏推进。

### 第一步

先跑通：

```bash
yolo train model=yolo11n.pt data=VisDrone.yaml imgsz=960 epochs=100 batch=16 device=0
```

### 第二步

再跑一个小目标 baseline：

```bash
yolo train model=ultralytics/cfg/models/26/yolo26n-p2.yaml pretrained=yolo26n.pt data=VisDrone.yaml imgsz=1024 epochs=150 batch=16 device=0
```

### 第三步

复制一份自己的 YAML，只改一个创新点。

比如：

- 把某个 `C3k2` 换成你自己的轻量 block
- 只在 neck 改一个模块
- 不要同时改 backbone、neck、head、loss

### 第四步

每做完一个结构改动，都立即验证三件事：

1. 能不能正常训练
2. 指标有没有提升
3. 能不能导出 ONNX

### 第五步

最后再做 TensorRT 和设备部署。

## 12. 一句话总结

对你这个“无人机小目标检测 + 轻量化部署 + 尽量少改代码”的目标，最优策略不是一开始猛改 Python，而是：

- 先用这个仓库自带的训练和导出链路
- 先跑 `YOLO11n` 或 `YOLO26n-P2` baseline
- 优先在 `cfg/models/*.yaml` 上做结构创新
- 只有确实需要时，再往 `nn/modules/*.py` 里加自定义模块
- 每次改动都顺手验证 ONNX / TensorRT 导出

这样最省代码，也最贴近论文和部署的双重目标。
