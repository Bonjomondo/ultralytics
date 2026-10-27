# C2＋P5局部注意力：从零150轮训练结果

分支 `5090-detrscope-query-budget`。用户要求仅补跑缺少的组合模型；C0/C2沿用已有150轮结果。组合150轮、完整548图COCO评估于2026-10-27 04:29（北京时间）完成。

**本次seed0、150轮下，组合未超过已有C2：原生FP32正式AP下降0.490点，TensorRT FP16正式AP下降0.537点。** 后端部署精度与速度使用修复虚拟环境路径后的实际验证，不以最初未成功的TensorRT结果冒充。

## 正式精度

原生未融合FP32，完整548图、38759标注框、batch2、640、conf=.001、每query取最高类、无NMS/坐标舍入，COCO bbox maxDets=[1,10,100]。下表AP乘100；差值是百分点。

| 指标 | 旧C2 150e | 新C2＋LA 150e | 差/点 |
|---|---:|---:|---:|
| AP | 19.517 | 19.027 | -0.490 |
| AP50 | 33.495 | 32.846 | -0.649 |
| AP75 | 19.148 | 18.442 | -0.705 |
| APs | 12.653 | 12.390 | -0.263 |
| APm | 26.698 | 26.613 | -0.086 |
| APl | 34.665 | 31.191 | -3.474 |
| AR100 | 29.639 | 29.150 | -0.490 |
| ARs | 21.031 | 21.146 | +0.115 |

小目标AP略降，ARs增加0.115点；召回和精度反映不同量，不能据ARs上升认定AP改善。大目标AP下降3.474点，数量及类别构成可能影响该指标，单seed不足以归因。完整12项指标见[accuracy.csv](accuracy.csv)。

类别方面，自行车AP增加0.916点，truck下降1.567点，awning-tricycle下降2.049点；其余类别见[category_comparison.csv](category_comparison.csv)。样本量和随机训练波动都会影响类别结果。

## TensorRT FP16完整验证集精度

使用同场测速的实际TensorRT FP16引擎，读取同一548张原图和同一标注，batch1、输入输出FP32、引擎可选择混合精度层；原生FP32评估为batch2。

| 指标 | C2 TensorRT FP16 | C2＋LA TensorRT FP16 | 差/点 |
|---|---:|---:|---:|
| AP | 19.476 | 18.939 | -0.537 |
| AP50 | 33.531 | 32.705 | -0.826 |
| AP75 | 18.978 | 18.360 | -0.618 |
| APs | 12.581 | 12.303 | -0.278 |
| APm | 26.682 | 26.575 | -0.107 |
| APl | 34.772 | 30.976 | -3.796 |

具体引擎的FP16路径已完成正式COCO评估，详细指标见[accuracy_trt_fp16.csv](accuracy_trt_fp16.csv)。独立DETRScope评估器重放预测，12项整体指标均在1e-12内重现；见[重放记录](raw/trt_fp16_replay.json)。它与原生FP32使用不同执行路径及batch大小，数值变化不应全部归因于精度转换。

## 训练后同场测速

训练后各自best.pt；真实验证图、GPU常驻batch1、640、纯forward，20轮随机交错、每轮50个CUDA事件样本，150次预热。Eager是未融合FP32，TensorRT先完成Conv/BN等融合。时间不含图像预处理、数据传输和检测后处理。

| 后端 | C2/ms | C2＋LA/ms | 延迟下降 | 配对95%区间 |
|---|---:|---:|---:|---:|
| eager_fp32 | 7.4104 | 7.4782 | -0.91% | [-0.96%, -0.87%] |
| trt_fp32 | 2.3555 | 2.4125 | -2.42% | [-2.44%, -2.40%] |
| trt_fp16 | 0.8611 | 0.9317 | -8.20% | [-8.24%, -8.15%] |

本次TensorRT FP16的组合模型延迟变化为+8.20%（正值表示变慢），与实测精度一起看，没有出现同时提精度和降延迟的证据。

延迟下降=100×(1−组合/C2)，负数表示变慢；95%区间按20个配对轮次bootstrap，不代表不同输入、重新构建引擎或不同设备的波动。FPS为纯forward延迟倒数，见[latency.csv](latency.csv)。

第一次自动部署测速因控制器将虚拟环境Python符号链接解析成基础解释器，导致找不到TensorRT；顶层队列虽完成，原部署记录明确为`partial_backends_failed`。随后直接使用虚拟环境入口补测，原失败日志和结果仍保留服务器，原因和SHA见[验证记录](raw/validation.json)。

TensorRT构建日志的FP16常量裁剪及LayerNorm相关警告按实际出现次数记录在验证文件；两个实际引擎均完成了有限输出、变更输入检查和完整验证集COCO评估。

## 训练与解释边界

新组合从零训练：150轮、seed0、batch8、640、FP32、AdamW lr=1e-4、lrf=1、warmup3、weight_decay=1e-4、mixup=.2、workers0、无预训练或旧checkpoint续训。训练部分累计5.80小时，PyTorch训练峰值allocated 14029 MiB；不等同GPU总占用。

内部mAP最佳值出现在第150轮；125→150轮增加1.263点，最终学习率仍为1e-4。训练后期继续增长，因此此结论对应150轮固定预算，不证明充分收敛。150轮健康记录中P2/QKV/RPB梯度均非零、框更新有限。

![训练曲线和精度延迟比较](figures/summary.svg)

历史C2采用同一数据、配方和seed，但不是本轮同期新训；实验仅有组合与历史基线，不能估计完整2×2交互项。固定seed未保证严格确定性，多seed结果缺失，0.490点AP差不能视作统计显著性结论。

## 可读证据与复现

- [训练曲线：新组合与历史C2](training_curves.csv)、[完整健康记录](raw/training_health.csv)、[训练状态和checkpoint路径/SHA](raw/training.json)。
- [原生FP32正式COCO评估](raw/evaluation.json)、[TensorRT FP16正式COCO评估](raw/trt_fp16_coco.json)、[训练后同场CUDA原始样本](raw/benchmark.json)、[执行与失败原因](raw/validation.json)。
- [启动时计划、冻结数据及复现命令](../2026-10-26_5090_c2_la_150e_launch/README.md)；[前期未训练结构预检](../2026-10-26_5090_p2_p5_la_preflight/README.md)。
- 服务器保留权重、原始预测、完整日志和ONNX/TensorRT engine；路径、大小与SHA在验证记录或engine元数据中，不是云端运行依赖。

从仓库文本离线生成和校验，不读取GPU、数据集或服务器权重：

```bash
python3 -I docs/experiment_reports/2026-10-27_5090_c2_la_150e_results/verify.py
python3 -m experiments.p2_p5_la_5090.archive_joint_results --output /tmp/5090-joint-final-replay
```

GPU复现TensorRT FP16正式AP（需要保留的两个engine、对应benchmark记录、权重及数据；使用新输出目录）：

```bash
PYTHONPATH="$PWD:$PWD/../detrscope/src" /tmp/zhj-uav-profiler-20261025-venv/bin/python -m experiments.p2_p5_la_5090.evaluate_trt_fp16_5090 \
  --benchmark runs/p2_p5_la_5090/5090-c2-la-trained-timing-20261027-r1/benchmark/assessment.json \
  --training runs/p2_p5_la_5090/5090-c2-la-150e-20261026-s0-r1/5090-c2-la-150e-s0/experiment.json \
  --native-evaluation runs/p2_p5_la_5090/5090-c2-la-150e-20261026-s0-r1/evaluation/summary.json \
  --annotations /绝对路径/VisDrone2019-DET_val_coco.json --images /绝对路径/images/val \
  --output runs/p2_p5_la_5090/5090-c2-la-trt-fp16-coco-new
```
