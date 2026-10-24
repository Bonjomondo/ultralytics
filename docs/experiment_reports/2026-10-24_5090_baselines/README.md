# 5090 C0/C2 基线与 Query 预算实验

2026-10-24，Asia/Shanghai。分支 `5090-detrscope-query-budget`。本次在 RTX 5090 D v2 上重建 C0/C2 的同机对照，实验与日志名称均以 `5090-` 开头；4090历史结果继续单列。

当前状态：**真实数据短跑及8图四档Query门禁全部通过，C0的150轮正式训练已完成第1轮并进入第2轮，loss及健康指标有限、checkpoint正常；C2和后续评估按队列顺序自动执行。** 完整训练和正式评估完成前，不报告结构收益、压缩收益或加速比例。

## 为什么先重建基线

本机没有 RT-DETR C0/C2 旧权重，已有 SO-DETR 权重的架构和训练协议不同。故复用原 C0/C2 的配方及共有参数初始化规则，从零开始重建同机、同预算的150轮基线。短跑权重只用于验证训练、保存加载和诊断链，不进入正式模型对照。

真实数据来自本工作区 `SO_DETR/datasets/VisDrone2019-DET-YOLO`：6471张训练图、343200框；548张验证图、38759框。当前COCO标注SHA为 `6f8b8686864f1485ab172de9339f7049593f74766f6654c6e980a6c00d810371`，与旧4090 RT-DETR记录不同。已核对本机图像/标签完整及COCO对YOLO坐标误差<=5e-7；现有证据仍不能证明两份历史COCO语义完全一致。本次冻结本地GT，只计算同一5090、同一GT下的配对差异。

## 顺序与晋级门禁

1. C0、C2分别跑1轮、训练集fraction=0.01、640、batch8、FP32的真实数据短跑；各自验证集仍使用完整548图。要求loss有限、健康记录有限、两组共有初始化SHA相同、C2支路确有梯度、best/last/resume checkpoint齐全。
2. 对两份短跑权重各做固定8张真实验证图、300/250/200/150Q完整Decoder重执行，验证观测等价、GT/输入身份、COCO导出、离线分析和报告。这些指标仅作工程验收。
3. 门禁全部通过后，顺序训练 `5090-rtdetr-r18-c0-150e-s0` 与 `5090-rtdetr-r18-c2-150e-s0`。每组完成后先8图接入检查，再548图Query Sweep及无observer GPU计时。
4. 根据真实Coverage/AP/APs/延迟取舍再选择少量预算；本队列不自动加入采样mask、新结构或多seed。

完整训练沿用旧研究的150e、seed0、AdamW、lr1e-4、nbs64、FP32、workers0、batch8、同初始化规则。框架checkpoint保存FP16 EMA并在评估恢复FP32；保留未strip的resume.pt，但不声称checkpoint跨硬件逐比特恢复。

## 执行与产物

新增入口：[训练](../../../experiments/decoder_budget_5090/train_5090.py)、[串行队列](../../../experiments/decoder_budget_5090/queue_5090.py)。所有run、日志均以5090开头。长任务放在独立tmux会话，任何阶段失败即停，不覆盖失败目录。

正式Query评估使用本次诊断协议：未融合FP32、原生stretch/RGB/255、原生Predictor后处理、不舍入COCO JSON、maxDets=[1,10,100]。它与旧4090 fused/舍入validator协议分开记录。

权重、完整原始tensor、逐图预测和完整日志只留服务器；仓库保留命令、源码身份、紧凑状态/健康检查与阶段指标。运行中状态是本次观察快照，不替代服务器上的最新status_5090.json。

## 已通过的真实数据门禁

| 检查 | C0 | C2 |
| --- | --- | --- |
| 1轮、fraction=.01、batch8 FP32训练及完整548图验证 | 通过 | 通过 |
| best/last/resume checkpoint保存和加载 | 通过 | 通过 |
| 同一8图四档300/250/200/150Q完整Decoder、观测等价与离线报告 | 通过 | 通过 |
| 原生输出、RNG及卸载恢复 | 全batch通过 | 全batch通过 |
| 配对初始化 | 两组共有SHA完全相同 | P2支路梯度非零 |

[数据审计](raw/asset_audit_5090.json) · [训练门禁](raw/training_gates_5090.json) · [真实图像Query门禁](raw/query_gates_5090.json) · [门禁回执](raw/gates_passed_5090.json) · [实际环境](raw/environment_5090.json) · [测试日志](raw/pytest_5090.log)。其中1轮模型的检测指标仅验收工程链，不能与150轮架构结果比较。

首次C0短跑在保存阶段因缺少polars失败；完整目录保留，[失败及修复记录](raw/failed_attempt_5090.json)。随后只在本次临时venv安装polars==2.0.0，并在新`-r1`目录重跑全部门禁通过，未改变共享UAV_CV环境或旧4090源码。

## 当前运行位置

实验根目录（仅服务器）：

```text
/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1/
  status_5090.json
  5090-rtdetr-r18-c0-150e-s0/
  5090-rtdetr-r18-c2-150e-s0/       # 轮到C2时创建
  logs_5090/5090-*.log
```

持久tmux会话：`5090-r18-c0c2-full150-20261024-r1`。控制日志为仓库下 `logs/5090-c0c2-full150-20261024-r1.log`；每阶段详细日志和命令写入`status_5090.json`。启动观察见[状态快照](raw/launch_status_5090.json)，该快照不实时更新。

复现时先选择不存在的新root并运行门禁，再继续该root：

```bash
UAV_PYTHON=/tmp/zhj-uav-5090-venv/bin/python
EXPERIMENT_ROOT="$PWD/runs/decoder_budget_5090/5090-c0c2-baselines-新实验名"
"$UAV_PYTHON" experiments/decoder_budget_5090/queue_5090.py \
  --root "$EXPERIMENT_ROOT" --sdk ../detrscope --gate-only
"$UAV_PYTHON" experiments/decoder_budget_5090/queue_5090.py \
  --root "$EXPERIMENT_ROOT" --sdk ../detrscope --continue-after-gates
```

上述长任务需置于tmux并将输出直接重定向文件；`--continue-after-gates`核验先前门禁回执和源文件SHA后才继续，不接受未完成/被改动的门禁。新机器需使用Python3.11与实际依赖环境，临时venv路径是本机执行证据。

本次全部专项测试共 **58项通过**。云端检出后可运行 `python3 docs/experiment_reports/2026-10-24_5090_baselines/verify_5090.py` 校验提交的源码及门禁证据，不需要服务器权重、数据或GPU。
