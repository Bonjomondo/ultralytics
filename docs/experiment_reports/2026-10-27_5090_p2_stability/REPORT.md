# P2 配对 seed1 最终结果、同 GT 诊断与 D2 部署验证

更新于2026-10-28（Asia/Shanghai）。**C0/C2 seed1 的150轮训练和完整548图正式COCO评估已于今天10:52全部结束。seed1 的C2总体AP没有提升，seed0的增益尚未跨seed稳定复现。** 用户授权的seed2队列已于12:36启动，当前先训练C0。[后续计划与执行命令](PLAN.md)。

## Seed1 最终精度与跨 seed 判断

下表均为原生未融合FP32、完整验证集、COCO maxDets=[1,10,100]。AP列使用百分数，差值为百分点（pp）。C0为基础结构，C2加入P2-SFI；两者均为300 queries、三层Decoder，不含P5局部注意力。

| Seed | C0 AP | C2 AP | AP差pp | C0 APs | C2 APs | APs差pp |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 18.661578 | 19.516709 | +0.855130 | 12.166368 | 12.652582 | +0.486214 |
| 1 | 18.882105 | 18.828508 | −0.053597 | 11.894779 | 12.157114 | +0.262335 |

| 指标 | Seed0差pp | Seed1差pp | 配对差均值pp | 样本标准差pp | 最小/最大pp |
|---|---:|---:|---:|---:|---:|
| AP | +0.855130 | −0.053597 | +0.400767 | 0.642568 | −0.053597 / +0.855130 |
| APs | +0.486214 | +0.262335 | +0.374274 | 0.158306 | +0.262335 / +0.486214 |
| AP50 | +0.993926 | +0.096637 | +0.545282 | 0.634480 | +0.096637 / +0.993926 |
| AP75 | +1.081212 | +0.051183 | +0.566197 | 0.728341 | +0.051183 / +1.081212 |

seed1总体AP基本持平且略低。相较seed0，C0自身提高0.220527pp，C2下降0.688201pp，配对收益缩小0.908728pp。AP50/AP75虽略正，不能替代IoU .50:.95的总体AP判断。当前只有两个训练seed，均值不是显著性结论；配对AP差的标准差大于均值，收益对训练随机性敏感。

小目标APs在两个seed中都为正，是比总体AP更一致的初步信号，仍需更多重复验证。seed1的APm为26.732139→26.622759（−0.109380pp），APl为30.698593→32.460732（+1.762139pp）；不能把总体结果归因于单一尺度。分类别AP为5类改善、5类退化：pedestrian +0.911050、tricycle +0.418957、people +0.252182；awning-tricycle −0.806893、truck −0.691732、bus −0.492076。seed0中van/truck/awning-tricycle/bus均正，seed1均转负，说明既有seed0同GT分组结论不能直接外推为跨seed机制。[完整正式指标](seed1_results/formal_metrics.csv)、[配对统计](seed1_results/paired_statistics.csv)、[分类别/尺度指标](seed1_results/category_metrics.csv)。

预先约定的工程门槛是“三对AP差全部为正且均值至少+0.3pp”。**当前门槛未通过，且seed1为负意味着补完seed2也无法满足原来的全正条件。** 不应事后更换门槛并称为通过。补seed2仍可量化三对差值、观察APs是否继续为正，并判断是否值得设计新的验证方案；目前C2保留为候选，不升级为已验证的稳定改进。

![seed1内部训练曲线与两次seed正式AP/APs差值](seed1_results/summary.svg)

## 当前执行到哪一步

| 任务 | 当前状态 |
|---|---|
| P0-1：C0-S1/C2-S1从零150轮及正式评估 | 全部完成；C0评估05:01:44结束，C2评估10:52:06结束 |
| P0-2：四模型同GT诊断 | 已完成，使用既有seed0预测 |
| P0-3：Encoder及D1–D3逐层汇总 | 已完成，使用既有seed0观测和预测 |
| P0-4：C0-S2/C2-S2 | 10月28日12:36启动；先C0训练、正式评估，再C2训练、正式评估 |
| P1-1：C2 D2/D3完整TensorRT FP16 AP | 已完成，使用既有seed0引擎 |
| P1-2：Orin NX部署测速 | 待目标设备执行 |
| P2：NAIFI/NATTEN微基准、长训或D2恢复 | 尚未执行，后两项仍为条件任务 |

seed1队列按预设正常退出，全部阶段返回码为0，完成快照为`seed1_completed_seed2_pending`。seed1全量训练从10月27日23:11:37至10月28日10:52:06，含评估约11小时40分；每组训练约5小时50分。seed2新队列于10月28日12:36启动，实际进度看服务器`status.json`；启动时C0已进入第一轮并占用GPU。[seed2启动核验](raw/seed2_launch_preflight_20261028.json)。以下既有同GT与部署结果均来自seed0，不能当作seed1新增结果。[seed1完成状态快照](seed1_results/status.json)。

## P2 的收益与损失

以下为原生FP32、完整548图、38759个有效GT，conf≥.001、每图每类别maxDets100、COCO按分数贪心匹配的状态变化。GT ID跨模型固定，匹配的query可以改变。这是固定阈值下的召回诊断，不是AP差值。

| 比较 | IoU | 找回GT | 丢失GT | 净找回 | 召回变化pp |
|---|---:|---:|---:|---:|---:|
| C0→C2 | .50 | 2326 | 2215 | +111 | +.286 |
| C0→C2 | .75 | 2552 | 2356 | +196 | +.506 |
| LA→C2+LA | .50 | 2418 | 2243 | +175 | +.452 |
| LA→C2+LA | .75 | 2521 | 2278 | +243 | +.627 |
| C2→C2+LA | .50 | 2286 | 2432 | −146 | −.377 |
| C2→C2+LA | .75 | 2424 | 2534 | −110 | −.284 |

两条P2路径总体均正向，LA加入C2后总体负向；具体找回目标有较大差异。IoU .5下两条P2路径共同找回438个GT，各自找回2326/2418个；IoU .75下共同483个，各自2552/2521个。这是同GT身份交集，不是因果效应估计。[输入身份和交集](same_gt/summary.json)。

| C0→C2尺度组 | GT数量 | IoU .50净找回 | IoU .75净找回 |
|---|---:|---:|---:|
| small | 26575 | +60 | +114 |
| medium | 11116 | +60 | +80 |
| large | 1068 | −9 | +2 |

IoU .50下，van净+70、car+36、awning-tricycle+27；pedestrian−23、people−12、motor−27。LA→C2+LA的pedestrian净+103，people净−79。因此“P2主要找回同一群密集行人”仍未被当前数据证实；尺度和类别分布应分别报告。C2→C2+LA的损失较集中于people（净−116）和car（−70），同时存在van（+37）和motor（+31）的改善。各组分母、遮挡/截断/边缘/密集分组及共有检出框的IoU与分数变化见[完整分组表](same_gt/groups.csv)。这些召回变化不能单独解释各类别AP的全部变化。

C0→C2在conf .001/.05/.25、maxDets100/300、IoU .50/.75的12个诊断条件下，净找回都为正；范围+67至+219个GT。这只说明本seed0下上述阈值敏感性中方向保持，跨seed稳定性由新训练回答。300预算只用于诊断，正式COCO AP继续使用100。

## 改善出现在哪些阶段

| 原生FP32正式AP | C0 | C2 | C2−C0 pp |
|---|---:|---:|---:|
| D1 | 16.747 | 17.752 | +1.005 |
| D2 | 18.201 | 18.639 | +.437 |
| D3 | 18.662 | 19.517 | +.855 |

Encoder Top-300在IoU .50下，类别正确覆盖率为C0 65.386%、C2 65.977%；IoU .75下为37.222%、37.888%。C2的候选覆盖和D1已经存在正差异，后续层间差值并非单调增长。它支持进一步研究P2对候选质量及Decoder更新的影响，尚不能确定唯一机制。[Encoder覆盖](same_gt/encoder_topk_coverage.csv)、[逐层正式AP与尺度指标](same_gt/native_decoder_stages.csv)。

C2 D2→D3在IoU .50下找回744个GT、丢失603个、净+141；其中small净+102/26575、medium+41/11116、large−2/1068。IoU .75下找回846、丢失724、净+122。第三层既有修复也有退化，针对D2精度恢复可从这些GT分组着手。[全部逐层状态迁移](same_gt/overall.csv)。

本次复用了既有逐层预测和观测一致性验证。原始固定query/GT重匹配轨迹仍在服务器 `runs/diagnostics/5090-depth-eval-{c0,c2}-20261025-final/` 的记录与诊断汇总中；本次新增表按GT重匹配。跨模型相同query编号不作同目标身份依据。

## D2 早退的真实 TensorRT FP16 精度

同一个C2三层best checkpoint，原TensorRT引擎SHA逐项核验；本次重跑完整548图FP16精度。延迟列复用**这些完全相同引擎**此前5090 GPU常驻纯forward计时，并明确保留其历史身份。

| 输出 | 原生FP32 AP | 本次TensorRT FP16 AP | FP16 APs | 历史同引擎forward ms |
|---|---:|---:|---:|---:|
| D2真实早退 | 18.639 | 18.616 | 11.994 | .770305 |
| D3 | 19.517 | 19.489 | 12.573 | .868049 |

FP16 D2相对D3损失.873 AP、.578 APs，历史forward约降低11.26%。这组数据补上了FP16完整精度缺口；本次没有重新测速，也没有Orin数据，不能外推视频端到端收益。[对照CSV](deployment/comparison.csv)、[精度/引擎/环境/完整协议](deployment/summary.json)。FP16允许引擎保留FP32层，I/O为FP32，不宣称每层都是FP16。

双引擎评估复用旧入口，其临时别名显式为`c2→D3`、`c2-la→D2`，新summary已转换为真实`d2/d3`名称。没有训练新的D2网络，没有重建引擎；checkpoint、ONNX、engine与新预测路径/大小/SHA都保留于receipt。[显式引擎视图](deployment/benchmark-view.json)。

## Seed1 训练健康与证据核验

原始数据标签、图像路径/大小清单及COCO标注SHA与seed0一致，训练集6471图、343200框，验证548图、38759框。C0/C2的seed1和seed2配对初始化已在CPU核验；两组seed1真实GPU smoke各1轮、1%训练数据完成，共有层初始化SHA相同，健康记录有限，C2的P2梯度非零。smoke不是结构精度结果。

后台持久会话 `5090-p2-stability-s1-20261027` 已完成C0-S1全量150轮→正式评估→C2-S1全量150轮→正式评估。两组各150行训练曲线、150行健康记录连续完整，曲线与采样健康数值有限，各阶段bbox delta finite为1；C2每轮采样的P2首层权重梯度均非零，最小0.000201841。没有发现记录所覆盖范围内的数值发散或P2分支未参与训练。健康探针仅覆盖每轮前三个训练batch，不能声称检查了全部batch的激活和梯度。

C0/C2按内部mAP选出的最佳轮均为150，best与last逐文件SHA也分别一致。第125→150轮内部mAP为C0 19.562→20.937、C2 19.880→20.722，均继续改善；当前曲线不足以断言150轮已充分收敛。C2在若干中间轮领先，到最终轮被C0超过，不能只挑中间轮作为结构获胜证据。内部mAP与上表正式COCO AP口径不同，不混算。checkpoint由原生训练器保存FP16 EMA，正式评估加载后转FP32计算；训练本身AMP关闭。[全部曲线](seed1_results/training_curves.csv)、[全部逐轮采样健康记录](seed1_results/training_health.csv)。

本次只读核验254个冻结源码文件、数据YAML、COCO标注SHA、两组实际初始化与配方、训练receipt到评估receipt的绑定、全部best/last/resume文件哈希及正式预测哈希，均通过。原始训练receipt及正式评估receipt已归档为`seed1_results/training-{c0,c2}.json`和`evaluation-{c0,c2}.json`，其中含完整协议、执行环境、服务器checkpoint路径/大小/SHA及源码身份。[核验记录](seed1_results/audit.json)。权重、完整预测、日志和引擎仍仅保留服务器。

当前运行是起点commit加未提交新执行代码，不能只用旧commit定位本轮队列。[初始化计划](raw/initializations.json)、[冻结计划](raw/plan.json)、[数据审计](raw/asset_audit.json)。`raw/launch_status.json`、`raw/training_status.json`和`raw/first_epoch_health.json`保留为历史启动快照，seed1最终状态以`seed1_results/`为准。seed2启动前核对了冻结的训练/评估源码及数据身份。

归档复现（只读服务器资产；输出目录必须不存在）：

```bash
PYTHONPATH="$PWD:$PWD/../detrscope/src" /tmp/zhj-uav-5090-venv/bin/python \
  docs/experiment_reports/2026-10-27_5090_p2_stability/archive_seed1.py \
  --output /tmp/p2-seed1-analysis-new
```

seed2已按原配方执行，完成后按原门槛报告并评估后续验证。300轮长训需C0/C2配对设计，不能从当前仍上升的曲线推定P2长训必然获益。Orin测速仍作为独立速度线推进。

CPU匹配与命令协议测试4项通过，覆盖重复检测、crowd排除、100/300预算前缀、置信度、IoU门槛与空预测。既有原生训练代码未修改。[测试记录](raw/tests.txt)。
