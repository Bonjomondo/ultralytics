# 5090：四组 Decoder 预算从零重新训练

2026-10-25（Asia/Shanghai）；分支 `5090-detrscope-query-budget`。**四组门禁已通过，四组完整150轮任务已进入持久串行队列；当前是启动快照，不是训练完成报告。**

[自包含启动报告](REPORT.md) · [证据校验脚本](verify.py)

| 结构 | Query | 每尺度采样点 | 实际Decoder层数 | 正式训练预算 |
|---|---:|---|---:|---:|
| C2-Q300-P2-D3 | 300 | [2,2,2] | 3 | 150轮 |
| C2-Q200-P2-D3 | 200 | [2,2,2] | 3 | 150轮 |
| C2-Q300-P4-D2 | 300 | [4,4,4] | 2 | 150轮 |
| C2-Q300-P4-D1 | 300 | [4,4,4] | 1 | 150轮 |

每组从对应YAML构建新模型，`pretrained=False`、`resume=False`。没有读取旧C0/C2训练checkpoint，没有旧权重微调或截断旧模型。1/2层模型物理删除了后续Decoder及预测头，减少采样点也真实改变offset/weight投影维度。四份配置在 [configs目录](../../../experiments/decoder_budget_5090/configs)。

训练沿用原C2配方：seed0、FP32/AMP关闭、batch8、640、AdamW、lr=1e-4、lrf=1、weight_decay=1e-4、warmup3、warmup_bias_lr=0、nbs64、mixup0.2、workers0、patience0、无预训练。数据为6471张训练图/343200框、548张验证图/38759框；与原5090 C0/C2数据的标签内容SHA、图片文件/大小清单及COCO标注SHA一致。图片清单不是逐图片内容哈希，数据审计保留该边界。

为保持可比，共用参数来自**在内存中按seed0重新生成的随机C2参考模型**；新维度使用原生初始化。该步骤没有载入旧权重。分支构建seed=1000，保持原C2初始化习惯；实际训练启动时同时记录CPU与CUDA初始seed。四组共用参数逐字节相同，并对实际trainer模型核验完整初始state SHA，详见 [初始化计划](raw/initialization_plan.json) 与 [真实门禁](raw/gates_summary.json)。

所有四组均完成真实数据1轮、fraction=.01短跑、完整548图内部验证、best/last/resume保存、8图checkpoint重载正式COCO链路检查；损失有限、P2与各层offset梯度非零，实际训练初值符合计划。单层健康检查以Encoder框为前态，不引用不存在的前一个Decoder层。门禁AP只是工程结果，不作为架构精度比较。10项专项测试通过，包括真实loss/backward、拒绝旧权重初始化、防覆盖已有run、单组失败继续其余组及源码变化全局停止。

每组完整训练结束后，队列自动用新best.pt完成548图原生未融合FP32正式COCO AP（maxDets=[1,10,100]，conf=.001，不舍入）及Eager FP32/TensorRT FP32/FP16的batch1 GPU常驻forward测速；FP16只测性能，不能直接沿用FP32 AP。保存未strip的resume.pt供后续恢复，不在本任务中续用旧实验权重。完成150轮代表预算完成，尚不自动证明收敛。

- [本地自动进度与完整训练曲线](../../../reports/5090/fresh_budgets/experiments.md) · [可离线重建的JSON](../../../reports/5090/fresh_budgets/experiments.json)
- [启动时队列状态](raw/launch_status.json) · [首个正式训练状态](raw/first_training_status.json)
- [门禁摘要](raw/gates_summary.json) · [门禁回执](raw/gates.json) · [数据审计](raw/asset_audit.json) · [测试摘要](raw/validation.json)
- [冻结源码/输入身份](raw/frozen_sources.json) · [训练数据配置](raw/data.yaml)

本地进度文件每轮和每阶段自动刷新；Git网页端看到的是最近推送的快照，不是实时服务器状态。当前入口只承诺已启动，后续最终checkpoint/正式AP/部署耗时以队列保存的完成记录为准。

服务器实验根目录：`runs/decoder_retrain_5090/5090-c2-fresh-four-20261025-s0-r1/`。实时队列为其中的 `status.json`，逐组训练记录为各run的 `experiment.json`、`results.csv` 和 `health.jsonl`，完整日志在 `logs/`。权重、数据集、预测、ONNX/engine和完整日志仍留服务器。

持久会话：`5090-c2-fresh-four-150e-20261025-s0-r1`；控制日志：`/tmp/5090-c2-fresh-four-150e-20261025-s0-r1.log`。单卡串行执行，普通单组失败会记录后继续其余组，源码或冻结输入变化则停止队列。锁为 `/tmp/zhj-uav-5090-gpu0.lock`。

复现时使用新实验根目录，避免覆盖已有训练：

```bash
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
export YOLO_CONFIG_DIR=/tmp/zhj-uav-5090-config
UAV_FRESH_PYTHON=/tmp/zhj-uav-5090-venv/bin/python

# 默认先完成四组门禁，再自动依次执行四组150轮和正式评估。
"$UAV_FRESH_PYTHON" -u -m experiments.decoder_budget_5090.queue_fresh_budgets_5090 \
  --root runs/decoder_retrain_5090/5090-c2-fresh-four-new

# 可先--gate-only；仅门禁完整通过的同一目录允许--continue-after-gates。
# 部署评估环境为 /tmp/zhj-uav-profiler-20261025-venv/bin/python。

# 从仓库JSON重建进度表，不依赖GPU、SDK、数据或权重。
python3 experiments/decoder_budget_5090/report_fresh_budgets_5090.py \
  --from-json reports/5090/fresh_budgets/experiments.json --output /tmp/5090-fresh-report-replay
python3 docs/experiment_reports/2026-10-25_5090_fresh_decoder_training/verify.py
```

`MANIFEST.sha256`只列本启动归档与本次源码/配置，不列服务器文件，也不绑定会持续更新的本地进度表。云端校验只读仓库文本，外部COCO标注由审计SHA摘要引用；GPU复测依赖明确列出的外部数据、SDK和环境。
