# 5090：先完善 DETRScope，再筛选 Decoder 预算

分支：`5090-detrscope-query-budget`。起点为 `origin/feat/r18-p2-sfi` 的 `6d1d4d1badbecf47970243685a1d712ad3e31559`。本目录承接[原始研究构想](../../docs/experiment_reports/2026-10-24_5090_detrscope/RESEARCH_PLAN.md)，将第一步限定为 **源码身份校验、观测等价性、Top-K GT Coverage 和完整 Decoder Query 预算重执行**。CPU和5090四档合成模型工程验收已通过，结果与环境见[本次报告](../../docs/experiment_reports/2026-10-24_5090_detrscope/README.md)。

## 当前事实与可复用基础

- 第二/第三优先级：[Sampling Points速度筛选与Decoder深度报告](../../docs/experiment_reports/2026-10-25_5090_decoder_screen/README.md)。五组采样预算同时测独立Attention/整网；真实D1/D2/D3早退比较完整验证集AP、同GT状态迁移和实际延迟；不训练。

- 7.4 ms的第一优先级诊断已独立归档：[算子与运行时报告](../../docs/experiment_reports/2026-10-25_5090_runtime_profile/README.md)。包括CPU/CUDA timeline、算子归属、FP16/BF16、Conv-BN/RepConv融合、CUDA Graph和部署路径；该次归档未开展Sampling Points实验，新的筛选见上方第二/第三优先级报告。

- [C0/C2 原始150轮正式结果](../r18_p2_sfi/corrected_20261024/RESULTS.md)来自4090：C0 AP/APs 为18.705/12.104%，C2为19.408/12.933%。它们是单seed筛选，末期仍上涨；C2同时改变P2接入和P3融合通道，不能仅归因高分辨率信息。原服务器另有[分阶段收敛训练记录](../../docs/experiment_reports/2026-10-24_r18_p2_sfi_convergence/README.md)，当前Git快照状态为running，不能推定已完成。
- 本fork已有[4090真实权重验收](../r18_na_ablation/detrscope_20261021/README.md)：R18+P4 NA5、548张真实验证图、38,759个转换后GT，完整观测和正式COCOeval通过；内部探针/干预只覆盖8图。它使用显式源码SHA桥接，不是SDK官方wheel适配器。
- 相邻独立 `detrscope` 库的发布验收使用合成CPU模型；上述真实checkpoint验收属于本fork的实验插件。两项证据分别记录，不能相互替代，也不能自动证明新C0/C2或可变Query已通过。
- 新clone不包含历史 `runs/` 和C0/C2权重。本机相邻 `SO_DETR/datasets/` 已有真实验证图像与COCO标注，可共享核对SHA；下一阶段缺少的是本fork C0/C2的明确权重及其身份信息。
- 实际GPU为RTX 5090 D v2 24GB，driver 580.178.04，Python 3.11.16、torch 2.14.0+cu130、cuDNN 92400、compute capability 12.0。隔离执行内曾无法访问CUDA，宿主机GPU实际验收已经通过，已排除驱动不可用的初步误判。CPU/GPU合成smoke均通过，实际AP和加速仍待真实权重验证。

## 第一阶段实现与验收

| 组件 | 目的 | 必须通过的检查 |
| --- | --- | --- |
| `source_adapter.py` | 继承已有源码桥接经验，显式冻结本fork源码与允许的Query预算 | 核验源码路径、commit和Python文件SHA；拒绝未知源码/不支持的模型配置；明确标记源码fork而非wheel身份 |
| `coverage.py` | 对Encoder已选Top-K候选计算GT覆盖及排名边际收益 | 同图GT身份与类别映射一致；IoU .3/.5/.75；类无关和正确argmax类别分别统计；报告GT/图像分母、无效与ignore/crowd处理 |
| `run.py` | 提供 `preflight`、`pin`、`smoke`、`sweep` 入口 | 缺输入/驱动时预检失败退出；显式建立源码pin；CPU小型原生模型验证hook等价和可变Query真实执行；真实sweep绑定权重/GT/协议并保存完整性证据 |

验收按以下顺序进行；前一道未通过，不解释下一道的模型统计。

1. **环境与资产审计。** 记录分支/commit、源码SHA、Python/torch/CUDA/驱动、实际GPU型号、输入尺寸、dtype、batch、预处理、checkpoint/标注SHA、类别映射及图像顺序。缺失权重或CUDA必须明确报出；`5090`分支名不是硬件检测结果。
2. **原生观测等价。** 同一模型、输入和RNG状态下比较bare/安装collector/卸载collector后的完整原生输出；模型参数、buffer和模式保持一致，异常退出也应清理hook。只把实际通过的模型和设备列为已验收范围。
3. **Coverage正确性。** 用已知几何与类别的合成样本覆盖空GT、空候选、不同类别、相邻密集GT及阈值端点。允许同一候选几何覆盖多个GT，明确其为多对多候选覆盖，不冒充一对一检测召回。Top-K分组必须按真实Encoder得分排名，不能把query slot顺序当跨模型目标身份；记录预算边界同分，保存前缀分析不冒充独立Top-K重执行。
4. **完整Query重执行。** 对 `300/250/200/150` 改变原生Top-K数量，再执行完整Decoder，校验Encoder和每个实际执行Decoder层的query维度。禁止截断原300Q最终预测代替预算实验。预算恢复与失败清理同样验收。
5. **真实小样本复验。** 有权重和5090后，先选固定真实图像，核对数据预处理、GT映射、最终stage与原生输出、标准300Q正式指标重放，以及开启/关闭观测等价。只有这些通过，才运行完整验证集。

已实现的Coverage分组为类别、原图面积及密度。面积优先使用标注 `area`，缺省用原图bbox面积；small `<1024`、medium `[1024,9216)`、large `>=9216`，属于互斥诊断分组，不声称复刻COCOeval面积端点包含规则。密度为“另一个有效GT中心距不超过原图对角线5%”，不限制类别；ignore/crowd从覆盖分母和邻居集合排除。覆盖IoU阈值为 `.3/.5/.75`，正确类以候选argmax类别判断；汇总通过累加GT计数/分母得到，零分母留空，不能平均逐图比例。没有可靠遮挡标注时不生成遮挡组。

每档保存 `coverage.json`、GT分母聚合的 `coverage_summary.json/csv`及DETRScope `analyze/report`。真实 `sweep`还校验各Q的预处理输入、GT和顺序哈希一致。输出目录拒绝覆盖；相关正常/异常清理及no-overwrite行为纳入专项测试。

## 操作命令

从本仓库根目录执行。Python使用现有 `UAV_CV` 的完整路径，`PYTHONPATH`同时包含本fork及相邻DETRScope源码；不使用系统默认Python。输出进入本地 `runs/decoder_budget_5090/`，不要提交原始tensor/预测。

```bash
export PYTHONPATH="$PWD:../detrscope/src"
UAV_PYTHON=/home/ubuntu/miniconda3/envs/UAV_CV/bin/python

"$UAV_PYTHON" experiments/decoder_budget_5090/run.py --sdk ../detrscope pin \
  --expected-commit 6d1d4d1badbecf47970243685a1d712ad3e31559 \
  --output runs/decoder_budget_5090/source_pin.json

"$UAV_PYTHON" experiments/decoder_budget_5090/run.py --sdk ../detrscope preflight \
  --device cuda:0 --expected-gpu 5090 \
  --output runs/decoder_budget_5090/preflight.json

"$UAV_PYTHON" experiments/decoder_budget_5090/run.py --sdk ../detrscope smoke \
  --pin runs/decoder_budget_5090/source_pin.json \
  --output runs/decoder_budget_5090/smoke

"$UAV_PYTHON" experiments/decoder_budget_5090/run.py --sdk ../detrscope smoke \
  --pin runs/decoder_budget_5090/source_pin.json --device cuda:0 \
  --output runs/decoder_budget_5090/smoke-cuda
```

`preflight`在检查真实资产时另传 `--checkpoint /实际C0或C2/best.pt --annotations /实际val_coco.json --images /实际val/images`；输入缺失、GPU不可访问或型号不符必须失败退出。`smoke`默认CPU，另可指定CUDA；使用小型原生 `DetectionModel`、随机权重和合成GT，验证各Query完整重跑、hook一致性及离线Coverage，输出不用于报告AP收益。再次运行须选新输出目录。

有真实权重后，先用新目录运行8图预检。下面路径是待替换的外部资产，当前clone不提供这些权重：

```bash
"$UAV_PYTHON" experiments/decoder_budget_5090/run.py --sdk ../detrscope sweep \
  --pin runs/decoder_budget_5090/source_pin.json \
  --checkpoint /实际C0/best.pt \
  --annotations /实际val_coco.json --images /实际val/images \
  --device cuda:0 --expected-gpu 5090 --limit 8 --batch 2 \
  --output runs/decoder_budget_5090/c0-smoke
```

预算默认 `300/250/200/150`，`--limit 0`表示完整验证集。小样本通过后才去掉8图限制，C0/C2使用独立目录。`--benchmark`可选，使用50次预热、200次计时，测量无observer的未融合FP32整网forward；当前不启动完整训练。长任务使用 `tmux` 并将输出直接重定向到文件，避免终端管道。

## 第二阶段：C0/C2 Query Sweep

拿到C0/C2的明确checkpoint和SHA后，在同一5090、相同数据与评估协议下依次执行300/250/200/150Q。先完成每个模型自身300Q基线，不把4090历史结果当本次执行结果。优先使用相同训练预算的checkpoint；150轮和分阶段warm start收敛权重分开标明。

每个预算报告Encoder Top-K Coverage、正确类Coverage、排名区间新增覆盖、各类/原图尺度/已定义密度组的GT分母，并保留同GT的E→D1→D2→D3覆盖变化。最后层覆盖下降既可能来自候选集合，也可能来自分类/定位与排序，不能只看单一几何指标归因。

当前 `sweep`始终改变 `head.num_queries`并重跑完整原生Decoder，从本次raw对应的最终query生成COCO预测/评估；保存标注、权重、源码、协议哈希。固定COCO JSON、类别映射和 `COCOeval bbox`，明确 `maxDets=[1,10,100]`、预测上限及置信阈值。它采用**未融合、不舍入**的新诊断协议，不能与旧fused/舍入validator指标直接相减；同5090、每个checkpoint的300Q结果才是本次预算对照。历史validator口径的独立正式复评仍待后续完成。

Coverage是诊断指标，AP是检测指标，不能直接相减得到“Decoder丢失召回”。真实权重AP/APs及不确定性分析属于后续验收，未运行不得填入CPU报告。

计时另用**不安装observer**的路径，记录预热/计时次数、同步方式、batch、dtype、fused状态、输入输出范围、median/p95与显存。Decoder forward、整网GPU forward、端到端延迟分别标明；采集和CPU smoke耗时均不作为加速结果。所有新比较在同一5090复测，4090表格保留为历史参考。

## 后续路线与进入条件

| 阶段 | 进入条件 | 本阶段回答的问题 |
| --- | --- | --- |
| Query筛选 | 第一阶段工程门禁及真实300Q基线通过 | 低预算损失来自Top-K覆盖不足，还是已有候选后续退化？选1–2个预算 |
| 采样mask反事实 | Query结果可信，真实采样位置/权重采集通过 | 哪些层/尺度/点对现有候选敏感？保留点权重重新归一化并完整重执行后续层 |
| 真实缩点结构 | mask提示存在可利用预算，算子/投影维度实现完成 | `[2,2,2]`、`[3,2,1]`等配置微调或重训后是否保精度、真加速？ |
| 联合与跨架构验证 | 单变量Query和缩点各自有证据 | 少量组合是否优于独立压缩？再迁移SO-DETR NoFFT |
| 多seed与部署 | 少数候选通过同协议筛选 | 收益是否稳定，ONNX/TensorRT/Orin NX是否真正获益？ |

不能把低attention weight或框外采样直接认定为冗余。mask通常仍计算全部投影与采样，不代表节省计算；真实缩点必须改变offset/weight投影及采样算子的实际工作量。4D reference box使用 `offset / n_points * reference_wh * 0.5`，改变 `n_points` 还会改变坐标，mask与真实缩点不是等价干预。`[4,6,2]`和`[4,4,4]`总点数相同，只能称重分配。采样位置数减少也不等于FLOPs或整网延迟同比减少。

同seed跨设备不保证逐值重复；原生CUDA `grid_sample` 反向的确定性设置需按实际版本记录，不能只凭 `deterministic=True` 承诺确定训练。本次优先推理诊断，不同时加入新KD、局部注意力或动态Query模块。

## 产物与Git边界

按[仓库约定](../../AGENTS.md)保留脚本、源码pin、紧凑环境/门禁JSON、Coverage汇总CSV及主报告。权重、图像、完整预测和逐query/采样tensor留在服务器，报告写明绝对路径和SHA；不删除历史原件。失败/未完成结果记录状态，不能并入正式结果表。后续每次真实执行只更新本说明和本次报告，避免增生独立进度快照。

本机验证图像位于 `SO_DETR/datasets/VisDrone2019-DET-YOLO/images/val`（相对工作区），548张均存在。现有COCO标注含38,759个GT，但SHA与旧RT-DETR R0记录不同；这可能包含格式或转换语义差异，尚未核对，不能静默替代旧GT来复现历史AP。真实实验前需同时确认权重及标注身份。
