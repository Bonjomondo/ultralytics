# 三 seed P2 阶段诊断（只推理）

复用 C0/C2 的150轮seed0/1/2 best，保持640、batch2、FP32、300Q与三层Decoder。`depth.py`是原`screen_decoder_eval_5090.py`的独立适配副本：增加显式checkpoint/训练seed/历史评估绑定，补齐原生predictor的imgsz，逐图证明真实早退等价，完整D3重放原指标。原训练与诊断脚本未改动。

从仓库根目录、使用已验证的UAV_CV继承venv运行。长任务放tmux，stdout/stderr重定向至文件。队列独占GPU锁、拒绝覆盖输出；先四组8图检查，再四组548图推理，最后离线汇总。seed0复用历史同协议raw。所有权重/标注/source pin/输入和GT身份必须通过校验，不会调用任何训练入口。

```bash
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
STAGE_RUN="runs/diagnostics/p2-stage-replay-$(date +%Y%m%d-%H%M%S)"
/tmp/zhj-uav-5090-venv/bin/python -u -m experiments.p2_stage_diagnostics_5090.queue --output "$STAGE_RUN"
/tmp/zhj-uav-5090-venv/bin/python -u -m experiments.p2_stage_diagnostics_5090.cohort --analysis "$STAGE_RUN/analysis"
/tmp/zhj-uav-5090-venv/bin/python -u -m experiments.p2_stage_diagnostics_5090.metric_details --analysis "$STAGE_RUN/analysis"
/tmp/zhj-uav-5090-venv/bin/python -m experiments.p2_stage_diagnostics_5090.package --root "$STAGE_RUN" --output "docs/experiment_reports/p2-stage-replay-$(date +%Y%m%d-%H%M%S)"
```

`cohort.py`按相同GT追踪跨模型覆盖优势状态；`metric_details.py`离线分解逐IoU/类别AP，并计算100/300预算敏感性；`package.py`核验原始D3预测列表、checkpoint、源码pin与SDK，归档紧凑CSV/JSON，不复制权重或大数组。COCO.loadRes会给内存检测记录补字段；原始预测哈希须在补字段前按同一规范比较。

本次结果入口：[2026-10-29报告](../../docs/experiment_reports/2026-10-29_5090_p2_stage_diagnostics/README.md)。候选覆盖是多对多、正式COCO匹配是一对一、SDK Hungarian轨迹是诊断配对，三者分开报告；跨模型不能按query编号对应目标。观测结果用于阶段定位，不宣称单一因果机制。
