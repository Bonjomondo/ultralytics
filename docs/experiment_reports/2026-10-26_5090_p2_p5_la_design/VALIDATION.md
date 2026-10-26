# 实现验证

2026-10-26；CPU环境Python 3.12.14、torch 2.14.1+cpu、torchvision 0.29.1+cpu。没有CUDA设备，未运行5090训练或延迟基准。

[结构化验证证据](raw/validation.json)。相关测试：**21 passed、9 subtests passed（23.09s）**，包含新实验7项以及既有R18-NA/P2-SFI回归。

```bash
python -m pytest experiments/p2_p5_la_5090/test_experiment.py \
  tests/test_rtdetr_r18_na.py tests/test_rtdetr_p2_sfi.py -q -o addopts=''
bash -n experiments/p2_p5_la_5090/run_matrix.sh
python -m compileall -q experiments/p2_p5_la_5090
```

验证覆盖：四组结构与固定Decoder预算；共有参数和P2/P5槽配对初值；C0/C2与旧初始化器一致；实际Trainer重建；组合前向、完整检测loss反向与P2/QKV/RPB有效梯度；checkpoint保存恢复；COCO类别语义映射、精确检测及空预测；不配对/子集报告拒绝。

另以8张合成训练图、2张合成验证图、batch8、128分辨率跑通1轮真实CPU训练入口。仅在测试内覆盖5090设备检查及device/imgsz，正式入口仍固定5090/640。实际训练callbacks、健康记录、best/last/resume及恢复资产完整性检查通过。合成图指标不用于研究结论。测试临时权重不提交。

GPU边界：CUDA Event分段和交错基准入口尚未在5090执行，完整548图评估及TensorRT/Orin性能尚未验证。下一步先运行设计中的结构测速和单组GPU冒烟，再启动正式矩阵。
