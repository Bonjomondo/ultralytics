# 5090 LA-only：从零150轮补充对照

分支 `5090-detrscope-query-budget`；任务 `5090-la-only-150e-20261027-s0-r1`。已核对现有5090记录：C0、C2、C2＋LA均有完整150轮结果，**没有同配方、从零训练的LA-only完整结果**。本次只新训LA-only；本页是启动快照，不是最终精度报告。

[实际训练计划与配对初值](raw/plan.json) · [启动队列](raw/status.json) · [真实训练状态](raw/training_status.json) · [首轮健康](raw/first_epoch_health.json) · [数据审计](raw/asset_audit.json) · [数据配置](raw/data.yaml)

LA-only使用原C0的P3投影，**不带P2注入**；P5使用与组合模型相同的NAIFI(k7、RPB)，Decoder固定300 Query、每尺度4采样点、8头、3层。两个LA模型的P5槽与共有参数初始SHA完全一致；LA-only的P3槽与C0初始化一致。对应[模型配置](../../../ultralytics/cfg/models/rt-detr/rtdetr-resnet18-p5-naifi.yaml)、[初始化器](../../../experiments/p2_p5_la_5090/model.py)及[持久控制器](../../../experiments/p2_p5_la_5090/run_la_only_150.py)。

从零训练，seed0、全量训练集6471图/343200框、150轮、batch8、640、FP32/AMP关闭、AdamW lr=1e-4、lrf=1、weight_decay=1e-4、warmup3、bias warmup LR=0、nbs64、mixup=.2、workers0、patience0。`pretrained=False`、`resume=False`；没有读取历史训练权重初始化新模型。数据标签SHA、图片路径/大小清单SHA及标注SHA在实际启动前与已有5090实验核对一致；图片清单不是逐图内容哈希。

历史权重只作完成后的对照，不重新训练：

| 组别 | 既有best.pt SHA256 | 原生FP32 COCO AP |
|---|---|---:|
| C0 | `adeaec17109a3b3ac6b8a6e521fb6cc3c2bdb3cd124ad6eedecc930d35a03d2e` | 18.662 |
| C2 | `ac2b02c45596bb816c4a42799d2b97ff02d62a4dc153bc5816502f02e936a5d6` | 19.517 |
| C2＋LA | `1a3866ebc5246748d8c359acb30129baa688566c41cfe3be5ce6a05baab4833f` | 19.027 |
| LA-only | 新训练中 | 待完成 |

三个历史C0/C2/C2＋LA并非本次同时重训；本次启动前已核验其checkpoint SHA、150轮、seed、数据、相关源码和配对初值。它们使用相同预算，但独立训练且未启用严格确定性算法，单seed结果没有统计显著性保证。

## 持久执行

后台依次执行数据审计、唯一缺失组LA-only从零150轮、自身best.pt原生未融合FP32的完整548图COCO AP、四个实际训练checkpoint的同场GPU测速，以及TensorRT FP16可用时的完整548图精度评估。最后自动输出服务器 `report/REPORT.md`、`accuracy_fp32.csv`、`effects.json` 与 `timing_effects.json`。

主对照包含 `LA−C0`、`C2＋LA−C2`、`C2＋LA−LA` 和 `C2−C0`；交互项计算为 `C2＋LA−C2−LA＋C0`。AP差值用百分点；FP32精度、FP16精度及各自延迟分别报告。150轮是同预算筛选，训练是否收敛由曲线判断。

测速使用已有三组和新LA-only的best.pt，不修改已有权重；Eager FP32/TensorRT FP32/TensorRT FP16共12条件，20轮随机交错、每条件每轮50次，batch1、640、GPU常驻纯forward。FP16评估按两对模型执行，复用现有双引擎评估器：`fp16-benchmark-view-c0-la.json` 明确记录别名 `c2→实际C0`、`c2-la→实际LA`；`fp16-benchmark-view-c2-joint.json` 使用本名。输出必须通过映射读取，避免混淆模型身份。

自动流程为每个阶段保存日志和状态；若部署后端不可用，会保留已完成的150轮与正式FP32 AP并明确记录缺失。它使用虚拟环境Python入口本身，避免此前解析软链接导致TensorRT不可见。源码或数据身份在训练中变化会使控制器停止；不要在任务运行时更改冻结的源码。

服务器任务根目录：`runs/p2_p5_la_5090/5090-la-only-150e-20261027-s0-r1/`。

- 实时：根目录 `status.json`、`progress.md`；训练目录 `5090-la-only-150e-s0/experiment.json`、`results.csv`、`health.jsonl`。
- 模型：训练目录 `weights/{best,last,resume}.pt`；resume保留优化器和EMA恢复资产。
- 正式FP32评估：`evaluation/summary.json`；四模型测速：`benchmark/assessment.json`；FP16：`fp16-evaluation-c0-la/summary.json` 和 `fp16-evaluation-c2-joint/summary.json`。
- 每阶段完整日志位于 `logs/`。持久会话 `5090-la-only-150e-20261027-s0-r1`，控制日志 `/tmp/5090-la-only-150e-20261027-s0-r1-controller.log`。

权重、图片、预测、导出引擎、完整日志和实时状态仅留服务器；当前Git归档为启动快照。完成后按项目规则另归档完整曲线、指标、参数/延迟、关键原始计时、checkpoint与源码身份，更新汇报入口。

复现必须使用新任务根目录、同一外部数据及三份历史checkpoint：

```bash
PYTHONPATH="$PWD:$PWD/../detrscope/src" /tmp/zhj-uav-5090-venv/bin/python -u \
  -m experiments.p2_p5_la_5090.run_la_only_150 \
  --root runs/p2_p5_la_5090/5090-la-only-150e-new \
  --data /绝对路径/本机数据.yaml \
  --annotations /绝对路径/VisDrone2019-DET_val_coco.json \
  --images /绝对路径/images/val \
  --deployment-python /tmp/zhj-uav-profiler-20261025-venv/bin/python
```

无需GPU/数据/权重即可检查本快照：

```bash
python3 -I docs/experiment_reports/2026-10-27_5090_la_only_150e_launch/verify.py
```

`MANIFEST.sha256`仅包含随Git提供的文件。此前的组合[150轮结果](../2026-10-27_5090_c2_la_150e_results/README.md)、[未训练结构预检](../2026-10-26_5090_p2_p5_la_preflight/README.md)均可独立阅读；旧4090实验不混入本次5090主对照。
