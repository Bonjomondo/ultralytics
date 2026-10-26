# C2＋局部注意力：仅补跑150轮从零训练

分支 `5090-detrscope-query-budget`；实验 `5090-c2-la-150e-20261026-s0-r1`。按用户“先完成150轮、已有的不重跑”的明确要求，**仅新训练C2＋LA，已经进入后台150轮流程**。本页为启动快照，不是最终精度报告；此前FP32速度预检不再作为长训练启动条件。

- [训练计划与初值身份](raw/plan.json)、[启动队列快照](raw/status.json)、[实际训练状态快照](raw/training_status.json)、[数据审计](raw/asset_audit.json)、[数据配置](raw/data.yaml)。
- 首轮已经完成，P2/QKV/RPB梯度均有效；[首轮健康记录](raw/first_epoch_health.json)。这只是启动健康证据，不作为训练完成精度结果。
- [持久流程实现](../../../experiments/p2_p5_la_5090/run_joint_150.py)、[组合模型配置](../../../ultralytics/cfg/models/rt-detr/rtdetr-resnet18-p2-sfi-c2-p5-naifi.yaml)、[原实验设计](../../../experiments/p2_p5_la_5090/README.md)。
- [前期GPU预检](../2026-10-26_5090_p2_p5_la_preflight/README.md)是历史筛选记录；本次用户明确要求先取得训练完成结果，任务执行不会因该记录中的速度回退而停止。

训练固定seed0、全量6471图/343200框、150轮、batch8、640、FP32/AMP关闭、AdamW lr=1e-4、lrf=1、weight_decay=1e-4、warmup3、bias warmup LR=0、nbs64、mixup=.2、workers0。模型按新YAML和配对随机初始化器重新构建，`pretrained=False`、`resume=False`；不加载旧模型或1轮短跑权重作为学生初值。Decoder固定300Q、每尺度4点、3层；P5为NAIFI(k7＋RPB)，保留C2的P2→P3注入。

已有C0/C2的150轮训练不重复。本次后续主对照使用历史同配方seed0 C2 best.pt：SHA256 `ac2b02c45596bb816c4a42799d2b97ff02d62a4dc153bc5816502f02e936a5d6`。它只作为评估/测速基线，不用于初始化新模型。单个新组合加历史基线不构成新的四组正交实验，不计算交互项；单seed/150轮也不自动证明收敛。

后台依次执行：

1. 数据审计、源码/标注/YAML冻结、随机初值身份保存。
2. C2＋LA全量从零150轮，逐轮保留内部曲线和P2/QKV/RPB、Decoder健康记录。
3. 自身best.pt的完整548图原生未融合FP32正式COCO AP、尺度与逐类别指标，maxDets=[1,10,100]、conf=.001、无NMS/坐标舍入。
4. 历史C2和新C2＋LA训练后同场测速：Eager FP32、TensorRT FP32/FP16，20轮交错、每轮50次。部署后端失败会明确标为不完整，保留已完成的精度与其他后端结果；FP16只测性能，不声称FP16 AP已验证。
5. 自动生成服务器端 `report/REPORT.md` 和 `accuracy.csv`。完成后再归档最终精度、曲线及紧凑测速证据。

服务器任务根目录：`runs/p2_p5_la_5090/5090-c2-la-150e-20261026-s0-r1/`。

- 实时状态：根目录 `status.json` 和 `progress.md`；子进程PID按阶段更新。
- 训练目录：`5090-c2-la-150e-s0/`；其中 `experiment.json`、`results.csv`、`health.jsonl` 为实际运行证据。
- 权重：训练目录 `weights/{best,last,resume}.pt`；resume保留未剥离optimizer/EMA，仅作恢复资产。
- 日志：根目录 `logs/{data-audit,train-150e,evaluate-full548,benchmark-trained}.log`。
- 评估：`evaluation/summary.json`；部署测速：`benchmark/assessment.json`。
- 持久tmux会话：`5090-c2-la-150e-20261026-s0-r1`；控制日志：`/tmp/5090-c2-la-150e-20261026-s0-r1-controller.log`。

以上路径仅服务器保存，不要求云端存在；数据、权重、预测、完整日志和导出引擎不提交。Git只保存启动时静态快照，训练进度以服务器实时状态为准。源码/输入改变或训练/正式评估失败会明确记录失败并停止后续阶段，不覆写或自动重新开始已有训练。

同样流程的复现命令（需GPU、外部数据与历史基线；必须换全新root）：

```bash
PYTHONPATH="$PWD:$PWD/../detrscope/src" /tmp/zhj-uav-5090-venv/bin/python -u \
  -m experiments.p2_p5_la_5090.run_joint_150 \
  --root runs/p2_p5_la_5090/5090-c2-la-150e-new \
  --data /绝对路径/本机数据.yaml \
  --annotations /绝对路径/VisDrone2019-DET_val_coco.json \
  --images /绝对路径/images/val \
  --deployment-python /tmp/zhj-uav-profiler-20261025-venv/bin/python
```

云端校验只读取仓库文本：

```bash
python3 -I docs/experiment_reports/2026-10-26_5090_c2_la_150e_launch/verify.py
```

`MANIFEST.sha256`只包含随仓库提供的文件；冻结源码哈希、历史基线身份和数据标注SHA均在启动证据中保留。已有22项测试/9项子测试、真实GPU短跑及重载评估证据沿用前期预检；本次不重复运行旧基线训练或额外速度门槛。
