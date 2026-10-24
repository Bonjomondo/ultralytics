# 5090分支：DETRScope第一阶段工程验收

日期：2026-10-24（Asia/Shanghai）。分支：`5090-detrscope-query-budget`；起点：`origin/feat/r18-p2-sfi`，commit `6d1d4d1badbecf47970243685a1d712ad3e31559`。

**第一阶段CPU与5090合成模型工程验收通过。** 已实现源码身份桥接、Top-K Coverage和300/250/200/150Q完整Decoder重执行，下一步待C0/C2真实权重到位后复评。[实施路线与验收条件](../../../experiments/decoder_budget_5090/README.md) · [原研究构想](RESEARCH_PLAN.md)。本报告不宣称新的检测精度、5090速度收益或训练收敛结果。

## 已有证据及本次环境

| 项目 | 已知状态 | 解释范围 |
| --- | --- | --- |
| 旧C0/C2筛选 | 4090、seed0、150轮；AP/APs为18.705/12.104%和19.408/12.933% | [原始正式指标](../../../experiments/r18_p2_sfi/corrected_20261024/RESULTS.md)；单seed且末期上涨，不作收敛或多seed结论 |
| 旧DETRScope真实验收 | 本fork R18+P4 NA5在4090完成548图观测/COCOeval，8图probe与干预 | [原始报告](../../../experiments/r18_na_ablation/detrscope_20261021/REPORT.md)；显式源码桥接，不能直接推广至C0/C2及可变Query |
| SDK独立库发布验收 | 合成模型、CPU | 工程契约证据；不等同于本fork真实checkpoint验收 |
| 当前clone权重 | Git不包含历史 `runs/` 和C0/C2权重 | 不能在本次环境直接复算其AP；新工程输出可另建本地 `runs/` |
| 当前真实数据 | 相邻 `SO_DETR/datasets/`已有验证图像及COCO标注 | [资产核验](raw/asset_audit.json)；当前缺的是本fork C0/C2权重 |
| 当前GPU | RTX 5090 D v2 24GB、driver 580.178.04、CUDA 13.0、capability 12.0 | 宿主机GPU实际验收通过；旧隔离内CUDA不可用不代表驱动故障 |
| 当前软件 | Python 3.11.16、torch 2.14.0+cu130、cuDNN 92400 | [完整环境预检](raw/preflight.json)；不推定与旧4090环境逐值等价 |
| 新完整训练 | 未启动 | 第一阶段无完整训练预算 |

原4090[分阶段收敛报告](../2026-10-24_r18_p2_sfi_convergence/README.md)在本次Git快照中为running。该报告指向原服务器工作，不能据此推断本次clone存在进程或权重；新的5090执行结果单独归档。

## 已完成的工程验证

在 `experiments/decoder_budget_5090/` 增加源码SHA适配器、Top-K Coverage分析和 `preflight/pin/smoke/sweep` 命令入口。可变Query修改 `head.num_queries`后完整执行原生Decoder，逐层检查实际query维度。命令见[操作说明](../../../experiments/decoder_budget_5090/README.md#操作命令)；源码身份见[pin](raw/source_pin.json)及[实验源码SHA](raw/experiment_sources.json)。缺失输入、不可访问GPU或型号不符时预检失败退出。

| 验收 | 实际范围与结果 | 证据 |
| --- | --- | --- |
| CPU合成smoke | 2图随机权重小型原生 `DetectionModel`；300/250/200/150Q各执行3层；原生/观测/恢复输出及RNG逐值一致；模型state和预算恢复，残留hook=0 | [CPU结果](raw/cpu_smoke.json) |
| 5090合成smoke | 同上四档、2图、3层；GPU原生输出及RNG等价通过，state/预算恢复，残留hook=0 | [GPU结果](raw/gpu_smoke.json) |
| CPU专项测试 | 43项通过，覆盖Coverage口径、源码门禁、预算恢复、CLI及端到端/no-overwrite行为 | [CPU日志](raw/pytest_cpu.log) |
| CUDA端到端测试 | 5项通过 | [CUDA日志](raw/pytest_cuda.log)、[验证汇总](raw/validation.json) |

这里“逐值一致”只比较同设备、同输入下安装/卸载观测的路径，不声称CPU与GPU预测逐值相同。随机权重与合成GT用于检查真实模型执行链，没有已训练C0/C2权重，也没有测量真实数据AP或加速。

Coverage已经支持类别、互斥原图面积组和密度组；密度定义为另一个非ignore/crowd GT中心处于原图对角线5%以内，类别不限。所有Coverage分母排除ignore/crowd，聚合CSV累加GT计数后计算比例；零分母留空。离线Encoder前缀统计与实际改变Q后重跑完整Decoder分别标明，避免将两者等价。`sweep`核验跨Q输入/GT/顺序哈希并保存每档 `analyze/report`，已有输出目录拒绝覆盖。

真实C0/C2 checkpoint、548图AP和无observer延迟均待权重到位，工程测试通过不能替代这些结果。`sweep`从本次最终query生成COCO预测，采用未融合、不舍入的新协议；历史fused/舍入validator结果单列，不能直接相减。先在同5090为每个checkpoint重建300Q基线，历史validator口径的正式复评另行完成。

紧凑工程证据、测试日志和原研究构想随仓库提供，[MANIFEST.sha256](MANIFEST.sha256)用于校验。完整raw tensor、预测和逐query结果保留在本地 `runs/decoder_budget_5090/`，不入Git；本次没有下载或提交训练权重。

云端检出后可运行 `python3 docs/experiment_reports/2026-10-24_5090_detrscope/verify.py`，只用标准库核验已提交文件、源码SHA、证据结构和报告链接，不访问服务器tensor/权重。归档pin保留原机器的绝对位置；迁移机器时需显式重新建pin。

## 后续最小执行顺序

1. 取得C0/C2权重与SHA，核对本机已有COCO GT和图像，冻结源码/软件/硬件/预处理/评估协议；保持旧4090记录可读。
2. 同一5090先做300Q真实小样本预检及全量基线，核对无hook/有hook原生输出、GT身份、最终stage和正式指标重放。
3. 分别重新执行300/250/200/150Q完整Decoder，比较类无关/正确类Coverage和E→D1→D2→D3变化；正式AP另用COCOeval，计时另走无observer路径。
4. 据此选择1–2个预算，再做采样mask；有证据后才实现真实缩点、微调/重训和少量联合配置。

mask、真实缩点及加速分开验收。4D reference中的 `n_points` 同时影响采样坐标；采样位置数下降不等于整网延迟同比下降。当前不同时扩展KD、动态Query或其他结构。

本机验证图像位于 `SO_DETR/datasets/VisDrone2019-DET-YOLO/images/val`（相对工作区），548张均存在。现有COCO标注含38,759个GT，但SHA与旧RT-DETR R0记录不同；这可能包含格式或转换语义差异，尚未核对，不能静默替代旧GT来复现历史AP。真实实验前需同时确认权重及标注身份。
