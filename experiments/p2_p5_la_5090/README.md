# 5090：P2 小目标特征＋P5 局部注意力

分支：`5090-detrscope-query-budget`。设计基于 `11c5346`，状态为**配置和执行工具已准备，尚未在5090训练或测速**。历史4090 R2b和5090 C2结果只用于提出假设，不填写为新组合的结果。

目标：C2 的 P2→P3 空间注入改善小目标表示；P5 用已有 R2b 的 `NAIFI(k=7, RPB=true)` 替代 AIFI，检查是否降低 P5、整个 Neck/Encoder 和整网延迟。复用现有 `F.unfold` 局部注意力，不新增 NATTEN 依赖。当前 NAIFI 是整个 P5 编码块的替换，包括局部注意力、二维 FFN 和归一化；本实验评估这个既有块，不能把收益全部归因于窗口大小。

## 四组正交对照

| 参数名 | P2→P3 | P5 | 模型配置 |
|---|---|---|---|
| `c0` | 原P3投影 | AIFI | [原始R18](../../ultralytics/cfg/models/rt-detr/rtdetr-resnet18.yaml) |
| `c2` | C2，3×3 stride2 Conv | AIFI | [C2](../../ultralytics/cfg/models/rt-detr/rtdetr-resnet18-p2-sfi-c2.yaml) |
| `la` | 原P3投影 | NA7＋RPB | [R2b](../../ultralytics/cfg/models/rt-detr/rtdetr-resnet18-p5-naifi.yaml) |
| `c2-la` | C2，3×3 stride2 Conv | NA7＋RPB | [新组合](../../ultralytics/cfg/models/rt-detr/rtdetr-resnet18-p2-sfi-c2-p5-naifi.yaml) |

新组合只改 C2 的层6（AIFI→NAIFI），保留层14的P2注入和其余连线。Decoder仍接P3/P4/P5三尺度；没有添加stride4的第四路Decoder输入。全部固定300 Query、每尺度4个采样点、8头、3层Decoder。这里的P2表示特征层，不是之前采样点实验名中的`p2`。

比较：`c2−c0`测P2；`la−c0`测LA；`c2-la−c2`测P2存在时LA的增量；`c2-la−la`测LA存在时P2的增量。交互项为`joint−c2−la+c0`。AP/APs用百分点；延迟用ms和相对变化，不能直接相加历史百分比。

## 公平性与预算

- 第一阶段四组均从零150轮，seed0；batch8、640、FP32、AdamW、lr=1e-4且lrf=1、wd=1e-4、warmup3、bias warmup LR=0、nbs64、mixup0.2、close_mosaic0、patience0、workers0。继承原R18配方；`deterministic=False`，grid_sample等CUDA算子存在非确定性，固定seed不等于严格逐位复现。
- [初始化器](model.py)按稳定层身份复制参数。四组共有层完全相同；P2槽由seed+1000独立参考模型提供，LA槽由seed+2000提供。两个P2组、两个LA组分别逐位匹配。实际Trainer重建时使用同一初始化器，保存共有层/P2/P5哈希。新C0/C2与旧初始化器保持一致；LA的新配对初值不声称复现旧4090 R2b的全部随机权重。
- 默认重跑四组，形成同源、同机、同预算主表；旧C0/C2可作历史参考，不静默混入新主表。150轮是筛选预算。候选有希望后四组seed1/2复验；最终结论需增加训练预算或证明平台期，所有比较组采用一致预算。不要只延长组合组后与150轮基线比较。
- 数据沿用VisDrone2019-DET：6471张训练图、548张val、十类及原顺序。训练显式传本机YAML；评估显式传COCO GT与val图片目录。GT忽略区域/类别处理沿用既有转换，COCO AP不等同VisDrone挑战赛官方评测。

## 先测5090速度，再决定是否投入完整训练

在仓库根目录、原5090环境运行；该环境需原项目依赖、pytest和pycocotools。以下均为待用户在服务器执行的命令，本次未启动服务器任务。

```bash
python -m pytest experiments/p2_p5_la_5090/test_experiment.py -q -o addopts=''

python -m experiments.p2_p5_la_5090.benchmark \
  --output runs/p2_p5_la_5090/architecture-screen-s0
```

不传checkpoint时是配对随机初始化的**结构测速，不产生AP**。默认同一seed随机输入，真实输入可加`--image /绝对路径/val图片.jpg`。batch1、640、未融合FP32、TF32关闭，150次预热，至少5轮随机交错，每轮每条件10次预热＋100次计时；同时测无标记整网和带标记分段，检查输出一致、模型状态不变及hook清除。

输出`benchmark.json`保留全部逐次CUDA Event样本、每轮顺序、环境、源码哈希、参数量及标记开销。`p5`是层6本体；`neck_encoder`是层5–23，包含P5，**不能再将P5重复相加**；另记录Backbone和Head/Decoder。CUDA流耗时包含CPU提交导致的空隙，不是纯kernel忙碌时间；不含预处理、传输、后处理。结构测速与训练后的测速分开保存。

若LA只降低P5本体却未降低Encoder或整网，不判定整体加速。若这里变慢，先排查unfold/访存/算子实现，避免把4090上的约5%延迟收益外推到5090。当前执行器只测原生FP32；TensorRT FP32/FP16及Orin NX需对各自导出模型重测，并评估对应精度，不能从此结果宣称部署加速。

## 单组冒烟与完整串行运行

先将路径改为5090本机实际路径，不使用4090配置里的历史绝对路径：

```bash
DATA=/绝对路径/5090-data.yaml
GT=/绝对路径/annotations/VisDrone2019-DET_val_coco.json
VAL_IMAGES=/绝对路径/images/val
PROJECT="$PWD/runs/p2_p5_la_5090/5090-factorial-150e-s0"

# 工程检查，1轮/3%训练数据；不是结构精度结果。
python -m experiments.p2_p5_la_5090.train --variant c2-la \
  --data "$DATA" --project "$PWD/runs/p2_p5_la_5090/smoke" \
  --name 5090-c2-la-smoke-s0 --epochs 1 --fraction 0.03

# 四组各150轮 → 各自best全量评估 → 四checkpoint同场测速 → 文本汇总。
bash experiments/p2_p5_la_5090/run_matrix.sh \
  "$DATA" "$GT" "$VAL_IMAGES" "$PROJECT" 150 0
```

运行目录必须全新；遇到训练/评估失败立即停止，不覆写旧实验。各进程用现有5090 GPU0文件锁，防止与遵守同一锁的旧任务同时使用GPU；它不检查所有外部GPU程序，测速前需确保GPU空闲。脚本不自动提交、不自动推送、不自动续训。`resume.pt`保留原生未剥离优化器状态，当前新实验入口只支持从零训练；中断后不要用best.pt伪装精确resume，先按原生恢复协议核对后再恢复。

可单独训练任一组，例如优先调试组合；正式归因仍需要完整四组：

```bash
python -m experiments.p2_p5_la_5090.train --variant c2-la \
  --data "$DATA" --project "$PROJECT" --name 5090-c2-la-150e-s0
```

## 精度协议与判据

各组选择**自身训练日志内部mAP最好的best.pt**，不看正式val AP反复挑checkpoint。完整548图，原生未融合FP32、batch2、640 scale_fill、conf0.001、每Query取最高类、无NMS、最多输出300框；正式COCOeval maxDets=[1,10,100]。内部mAP与正式AP分别记录；APs/ARs用于检查小目标，逐类别AP/APs用于检查是否仅某类获益。预测留服务器且不舍入JSON坐标。FP16 checkpoint存储后转回FP32不恢复已丢弃的存储精度，各组沿用相同checkpoint机制。

预先采用以下**工程筛选线**，不是统计显著性标准：

1. 组合相对C0的APs正向，AP不下降；相对C2，AP下降不超过0.3pp、APs下降不超过0.2pp。
2. 组合相对C2，Encoder平均延迟至少下降3%、无标记整网至少下降1%；至少80%的配对轮次同向，收益超过轮间波动。P5本体另报，不用局部加速冒充整网收益。
3. 满足前两条后进行seed1/2和更长预算复验；若小目标精度明显受损，即使速度提高也只列为速度/精度折中，不宣称“同时改善”。

每轮前三个batch记录P2权重、P5 QKV/RPB梯度及D1–D3回归健康；有非有限值、无分支梯度、预算不完整、初始化不匹配时拒绝标为完成。矩阵结束后报告器核对同seed、预算、GT、训练配方、共有初值和checkpoint身份，输出主表及交互项。

## 结果保存与独立阅读

每组run含`experiment.json`、`initialization.json`、`health.jsonl`、`results.csv`、`weights/{best,last,resume}.pt`；评估目录含`summary.json`和服务器专用`predictions.json`；`paired-timing/benchmark.json`含原始计时，`report/`含REPORT.md、comparison.csv、effects.json。

正式结果归档到`docs/experiment_reports/`新日期目录并更新[索引](../../docs/experiment_reports/README.md)：提交自包含报告、必要完整训练曲线、健康与初始化摘要、环境/配方、数据和源码身份、checkpoint路径/大小/SHA256、紧凑原始计时以及仅包含随Git提供文件的MANIFEST。大权重、预测、图片、完整日志、导出engine和运行锁留服务器。不能只提交生成的三张汇总表就声称证据完整。

本次设计与验证记录：[报告入口](../../docs/experiment_reports/2026-10-26_5090_p2_p5_la_design/README.md)。
