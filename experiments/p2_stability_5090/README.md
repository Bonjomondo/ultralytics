# 5090 P2 稳定性与同 GT 诊断

当前主线为 C0/C2 配对 seed1/2、已有四模型的同 GT 与 D1–D3 诊断、C2 D2/D3 的 TensorRT FP16 完整 AP。[实验计划与实际结果入口](../../docs/experiment_reports/2026-10-27_5090_p2_stability/README.md)。

在 ultralytics 仓库根目录运行，使用原 5090 环境。默认执行 C0-S1→评估→C2-S1→评估；所有新训练从零150轮。每个 seed 共有权重与 AIFI 初值逐位核验，数据标签/图片清单/COCO标注身份与 seed0 对照核验。训练保留原 recipe、FP32、batch8、640、workers0、AdamW、恒定学习率和原增强。调用已有训练/正式评估入口，不改历史训练代码。

```bash
# 只看当前状态，不用GPU。
/tmp/zhj-uav-5090-venv/bin/python -m experiments.p2_stability_5090.queue --status

# 默认只补一对seed1；相同命令再次执行会跳过已完成的训练及评估。
/tmp/zhj-uav-5090-venv/bin/python -u -m experiments.p2_stability_5090.queue --seeds 1

# 后续补seed2；先前完成的seed1不会重训。
/tmp/zhj-uav-5090-venv/bin/python -u -m experiments.p2_stability_5090.queue --seeds 1 2

# 无GPU的匹配/队列检查。
/tmp/zhj-uav-5090-venv/bin/python -m pytest experiments/p2_stability_5090/test_stability.py -q -o addopts=''
```

默认任务目录 `runs/p2_stability_5090/5090-p2-stability-20261027-r1/`：`status.json`、逐阶段 `logs/`、训练 `training/5090-{c0,c2}-150e-s{1,2}/`、正式评估 `evaluations/`、配对结果 `paired_seeds.csv` 和 `paired_summary.json`。根目录有控制器锁；训练/评估另遵守 `/tmp/zhj-uav-5090-gpu0.lock`。独占GPU，测速与训练不要同时运行。

这是**按阶段续跑**：完成的训练和评估可直接复用；如果某组150轮训练中途退出，目录和 last.pt/优化器均保留，入口会停止并要求核对恢复资产，不覆盖目录，不从 best.pt 偷换成新训练。现有 FactorialTrainer 仅接受从零训练，不能直接给它加 `--resume`。恢复该组需先核对未剥离 checkpoint、epoch、优化器、EMA、曲线和健康记录，再实现相应原生恢复入口；CUDA随机数/数据顺序未完整保存，恢复也不能称为逐位连续。用户可以先让当前一对跑完，后续再执行seed2。

队列固定源码和数据身份，运行中不要改其冻结的 Python 源码或数据。后续若确需修复，先记录变更，核对是否影响实验协议，再安排新的运行身份。报告和实验计划可继续更新。

已有模型的无训练任务可独立重放到新目录：

```bash
/tmp/zhj-uav-5090-venv/bin/python -u -m experiments.p2_stability_5090.same_gt \
  --output runs/p2_stability_5090/same-gt-new

/tmp/zhj-uav-profiler-20261025-venv/bin/python -u -m experiments.p2_stability_5090.trt_early_exit \
  --output runs/p2_stability_5090/c2-d2d3-trt-fp16-new
```

同GT使用COCO逐图逐类别、按分数贪心匹配，以GT ID比较，包含IoU .5/.75、conf .001/.05/.25及每类别100/300检测预算；300只作诊断。分组含原图sqrt(area)、COCO尺度、类别、图像GT数量、边缘距离、遮挡和截断。另有C0/C2逐层AP和初始Top-K覆盖摘要；跨模型不按query编号认定同一目标。

FP16任务使用原C2三层checkpoint的既有D2/D3早退引擎，核验checkpoint和engine SHA，完整548图。复用双引擎评估器时明确别名 `c2→d3`、`c2-la→d2`，最终summary改回真实深度名称。对照表中的延迟来自同一引擎此前的测量，精度为本次新测；本次不声称重新测速或Orin结果。

权重、完整预测、引擎、日志和实时锁/PID只留服务器；紧凑统计、计划、启动快照和完整必要训练曲线按项目归档规则保存。训练完成后需另更新最终结果，启动快照不能作为最终精度证据。
