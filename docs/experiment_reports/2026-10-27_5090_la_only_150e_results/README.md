# 5090 LA-only：150轮正式对照结果

分支 `5090-detrscope-query-budget`。LA-only从零150轮已完成；完整548图原生FP32和TensorRT FP16 COCO评估、四模型同场测速均完成。已有C0、C2、C2＋LA没有重新训练。

[完整分析](REPORT.md) · [四组指标](comparison.csv) · [效应与交互](effects.csv) · [同场延迟](latency_effects.csv) · [完整训练曲线](training_curves.csv)

| 模型 | 原生FP32 AP | 原生FP32 APs | TensorRT FP16 AP | TensorRT FP16延迟 |
|---|---:|---:|---:|---:|
| C0 | 18.662 | 12.166 | 18.601 | 0.8544 ms |
| C2 | 19.517 | 12.653 | 19.494 | 0.8672 ms |
| LA-only | 18.311 | 12.086 | 18.280 | 0.9138 ms |
| C2＋LA | 19.027 | 12.390 | 18.942 | 0.9180 ms |

本次结果：单独加LA（C0→LA）原生AP下降 **0.350点**，TensorRT FP16纯forward延迟上升 **6.96%**；给LA加P2（LA→C2＋LA）原生AP增加 **0.715点**。C2同时比两组带LA模型精度高、延迟低。

AP差为百分点。延迟为训练后四模型同场、batch1/640/GPU常驻纯forward，不含前后处理。TensorRT FP16的AP是同一548图上对实际引擎正式评估所得；四份原始预测都经独立COCO评估器重放核对。原生FP32使用batch2、FP16引擎使用batch1，精度路径分开报告。

![训练曲线及TensorRT FP16精度延迟](figures/summary.svg)

四组采用相同数据、seed0、150轮配方，LA-only与组合的P5及共有参数初值逐字节配对；历史三组独立训练且未严格确定性。**只有一个seed，交互项和小幅AP差不能视为统计显著；LA-only后25轮内部mAP仍增加1.009点，不能断言已充分收敛。**

- [原生与FP16分类别/尺度指标](category_metrics.csv)、[完整效应与交互](effects.csv)、[延迟配对区间](latency_effects.csv)。
- [原始12000次CUDA样本](raw/benchmark.json)、[FP16实际别名映射和结果](raw/fp16-c0-la.json)、[另一对FP16结果](raw/fp16-c2-joint.json)、[四模型独立COCO重放](raw/replay.json)。
- [新训练状态、源码及checkpoint路径/SHA](raw/training.json)、[正式FP32评估](raw/evaluation.json)、[150轮健康](raw/training_health.csv)、[服务器资产路径/大小/SHA](raw/validation.json)。权重、逐图预测、图片、完整日志及导出模型仅服务器保存。
- [启动计划、数据审计、初始化证据与复现命令](../2026-10-27_5090_la_only_150e_launch/README.md) · [先前C2＋LA 150轮](../2026-10-27_5090_c2_la_150e_results/README.md)。

离线校验和重建（不需GPU、数据集或服务器权重）：

```bash
python3 -I docs/experiment_reports/2026-10-27_5090_la_only_150e_results/verify.py
python3 -m experiments.p2_p5_la_5090.archive_la_only_results --output /tmp/5090-la-final-replay
# 安装matplotlib后可加 --plots 重建SVG。
```

`MANIFEST.sha256`只列随Git提供的文件。GPU复测依赖已说明的服务器数据、checkpoint、TensorRT环境和执行脚本。
