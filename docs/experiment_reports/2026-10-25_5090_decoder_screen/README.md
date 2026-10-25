# 5090 第二/第三优先级：采样点速度筛选与 Decoder 早退

2026-10-25，Asia/Shanghai；分支 `5090-detrscope-query-budget`。阅读 [自包含报告](REPORT.md)。本次只做推理性能与现有权重评估，没有训练。

- [Sampling Points 全路径性能](sampling_performance.csv)、[单变量/联合压缩配对比较](sampling_comparisons.csv)
- [D1/D2/D3 正式 COCO AP/AR](depth_ap.csv)、[真实早退耗时](depth_performance.csv)
- [C0 正式 GT 迁移](raw/c0-formal_gt_transitions.csv)、[C2 正式 GT 迁移](raw/c2-formal_gt_transitions.csv)
- [C0 逐层轨迹诊断](raw/c0-diagnostic_transitions.csv)、[C2 逐层轨迹诊断](raw/c2-diagnostic_transitions.csv)
- [归档索引](raw/index.json)、[源码 pin](raw/source_pin.json)、[失败与警告记录](raw/execution_notes.json)

采样点实验：300Q/p4、300Q/p2、200Q/p4、200Q/p2、300Q/p1；p表示每尺度点数，三个尺度使用相同值。整网与独立 MSDeformAttn 均测 Eager FP32、融合 Graph FP32/FP16、TensorRT FP32/FP16。所有点数组（包括p4）使用同规则随机兼容的3层Cross-Attention，故只分析速度，不分析其AP。

深度实验：保留原C0/C2训练权重与300Q/[4,4,4]，原生eval_idx真正跳过后续层；D1/D2/D3使用各自分类头。全部548图核验早退输出与D3观测中间层逐值一致，原D3正式指标精确重放。FP32正式AP与只测性能的FP16分开标记，诊断GT责任变化与正式COCO TP/FN分开解释。

验收：全部正式条件完成，共65000次CUDA计时；两模型各548图的原生观测/恢复、D1/D2/D3逐值等价、D3原有12项指标重放及模型state/预算恢复均通过。17项专项测试通过，包含错误预算、NaN、重复轮次、丢失engine输入、错误checkpoint、早退未验收等拒绝检查；报告在隔离标准库进程内可校验。[验证摘要](raw/validation.json)。AP/AR CSV保留0–1值，报告表格转换为百分数，精度差值使用百分点。

原模型训练设置、数据划分及checkpoint身份见 [5090基线报告](../2026-10-24_5090_baselines/README.md)；运行时选择依据见 [第一优先级报告](../2026-10-25_5090_runtime_profile/README.md)。本次GPU输入为原先同一张验证图，640×640、batch1；精度为完整548张验证图、batch2、未融合FP32、原生RTDETRPredictor、不舍入、COCOeval maxDets=[1,10,100]。标注SHA为 `6f8b8686864f1485ab172de9339f7049593f74766f6654c6e980a6c00d810371`。

复测依赖原服务器的权重、数据和相邻 `../detrscope/src`。训练权重没有随本次提交上传，源路径、大小/SHA、输入身份及实际环境见各 `raw/identity-*.json` 与 `raw/depth-eval-*.json`。性能环境复用第一优先级的 `/tmp/zhj-uav-profiler-20261025-venv`（torch 2.14.0+cu130、TensorRT 10.14.1.48.post1），CPU测试环境为 `/tmp/zhj-uav-5090-venv`；共享训练环境未修改。

```bash
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
export YOLO_CONFIG_DIR=/tmp/zhj-uav-5090-config
UAV_SCREEN_PYTHON=/tmp/zhj-uav-profiler-20261025-venv/bin/python

# 每模型独立进程，GPU任务串行运行，输出必须为新目录。
for variant in c2 c0; do
  "$UAV_SCREEN_PYTHON" -m experiments.decoder_budget_5090.screen_decoder_perf_5090 sampling \
    --variant "$variant" --output "runs/diagnostics/5090-sampling-${variant}-new"
done
for variant in c2 c0; do
  "$UAV_SCREEN_PYTHON" -m experiments.decoder_budget_5090.screen_decoder_perf_5090 depth \
    --variant "$variant" --output "runs/diagnostics/5090-depth-perf-${variant}-new"
done
for variant in c2 c0; do
  "$UAV_SCREEN_PYTHON" -m experiments.decoder_budget_5090.screen_decoder_eval_5090 \
    --variant "$variant" --expected-commit 50bc059fbdecf61267dd48dfa21c1b33548ed075 \
    --output "runs/diagnostics/5090-depth-eval-${variant}-new"
done
```

可用 `--limit 8` 先做深度工程门禁；该子集指标不能放入正式AP表。性能的 `--cases`、`--rounds`、`--repeats`用于短跑；正式归档要求完整条件、10轮、每轮50次。每个后端先150次预热，每块再10次预热。报告使用按轮次配对的bootstrap区间，保留500次CUDA原始计时；完整CPU wall逐次数据保存在服务器assessment，仓库保留其块级与总统计。

离线报告重建与校验只用Python标准库，不依赖GPU、torch、SDK、权重或服务器数据：

```bash
python3 -I experiments/decoder_budget_5090/report_decoder_screen_5090.py --verify
python3 -I experiments/decoder_budget_5090/report_decoder_screen_5090.py
PYTHONPATH="$PWD:$PWD/../detrscope/src" YOLO_CONFIG_DIR=/tmp/zhj-uav-5090-config \
  /tmp/zhj-uav-5090-venv/bin/python -m pytest \
  experiments/decoder_budget_5090/tests/test_decoder_screen_5090.py -q
```

`MANIFEST.sha256`只列云端提供的文本、源码与报告；先验证再重建，不用重建掩盖未知数据变化。服务端原始assessment、COCO逐GT匹配、完整预测与逐query tensor、ONNX/engine、全部日志只保存路径/大小/哈希引用，仍保留在服务器。采样短跑、导出失败目录没有删除，也不混入最终性能表。

本次服务器结果到新紧凑归档的复现命令如下；`execution_notes.json`作为当前实验必要的短失败摘要明确保留并传入，不读取旧服务器日志即可重建该摘要。新归档应放在新目录，不能覆盖既有证据。

```bash
python3 experiments/decoder_budget_5090/archive_decoder_screen_5090.py \
  --performance \
    runs/diagnostics/5090-sampling-c2-20261025-r3/assessment.json \
    runs/diagnostics/5090-sampling-c0-20261025-final/assessment.json \
    runs/diagnostics/5090-depth-perf-c2-20261025-final/assessment.json \
    runs/diagnostics/5090-depth-perf-c0-20261025-final/assessment.json \
  --evaluation \
    runs/diagnostics/5090-depth-eval-c2-20261025-final/summary.json \
    runs/diagnostics/5090-depth-eval-c0-20261025-final/summary.json \
  --notes docs/experiment_reports/2026-10-25_5090_decoder_screen/raw/execution_notes.json \
  --validation docs/experiment_reports/2026-10-25_5090_decoder_screen/raw/validation.json \
  --output /tmp/5090-decoder-archive-new
```
