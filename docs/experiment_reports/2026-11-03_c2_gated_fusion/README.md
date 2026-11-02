# 4090：C2门控与特征融合筛选

状态：四组短跑与548图预测/独立评估链路已通过；正式队列启动信息见raw/launch.json，正式结果尚未产生。分支 `feat/r18-c2-gated-fusion`。

本轮以本仓库 `feat/r18-p2-sfi` 的C2实现为基线，研究P2细节注入是否需要自适应调节。5090的O2M-C/O2M-T/MGSA不在本队列内。

## 四组实验

保持P2下采样、P3/P4/P5路径、AIFI、300 queries和三层decoder。只替换第15层Concat，保留输入顺序和所有原C2权重键。

| arm | 变更 | 新增参数 | 初始状态 |
|---|---|---:|---|
| c2 | 原C2，同条件重新训练 | 0 | 原始Concat |
| scalar | P2分支乘 `2*sigmoid(logit)` | 1 | logit=0，倍率=1 |
| spatial | 三路各取通道均值/最大值，6→8的1×1卷积，再3×3输出空间门控 | 129 | 最后一层全零，倍率=1 |
| fusion | 三路各乘 `3*softmax(logits)` 后Concat | 3 | logits全零，三路倍率均为1 |

空间门控以三路上下文生成单通道空间权重，只作用于P2。倍率范围(0,2)既允许抑制也允许增强；采用2倍sigmoid是为了初始严格等价C2，避免普通sigmoid零初始化把P2直接减半。三路加权是独立融合消融，不与空间门控叠加。保留原Concat+RepC3，不把不同通道的P2/P3直接相加。

所有组从随机初始化训练，无预训练/教师/辅助监督。逐张量复制同一个未训练C2的全部参数，包括P2下采样分支；仅门控新增参数独立初始化。640整网初始输出要求与C2逐元素完全相同。

## 配方与评估

- 每组150轮，seed0；640、batch8、workers0、FP32、AdamW、lr1e-4、恒定末端LR、nbs64、warmup3轮、warmup_bias_lr=0、patience0，采用本仓库既有150轮筛选配方。
- 这与历史 `reproductions` 的作者引擎配方不同：本轮C2为20,135,032参数；历史28.72 AP的C2为20,135,288参数。框架、构造器、有效batch及warmup不同，禁止把两套绝对指标当结构消融。
- `args.yaml`为每组实际生效设置；`initialization.json`记录共有张量SHA。
- 原生mAP用于训练中选择best。最终加载best做548图FP32预测，conf0保留每图300个argmax query，无NMS；独立CPU pycocotools.COCOeval，maxDets=[1,10,100]，报告AP/AP50/AP75/APs/APm/APl及十类指标。
- GT为converted VisDrone val，SHA固定为 `6f57aa4b42d64ee7b99b8a6a5b159ad19cba2eacf00bd86f23314966614bf4d8`；这不是VisDrone挑战赛官方评测。
- 四组完成后，同进程随机交错5轮，每模型每轮100次；640、batch1、fused PyTorch FP32、GPU常驻输入、CUDA event同步。保存逐次耗时、参数量与FLOPs。排除传输、预后处理，不外推TensorRT或Orin。
- 训练每轮记录前三batch门控均值/方差/饱和比例、门控与P2梯度、encoder及三层decoder活性/方差、训练耗时与显存；不只观察最终AP。

预设晋级门槛：相对同场C2，AP和APs均至少+0.3pp；AP75/APm/APl各下降不超过0.5pp；前向中位延迟增幅≤10%。用于筛选，不代表统计显著性。150轮结束后，有希望的结构再做C2/候选同预算350轮和多seed；本轮不自动叠加未经验证的模块。

## 启动与状态

在服务器仓库根目录，以 `/home/zjnu/anaconda3/envs/UAV_CV/bin/python` 运行，必须使用本仓库PYTHONPATH：

```bash
PYTHONPATH=. /home/zjnu/anaconda3/envs/UAV_CV/bin/python experiments/c2_gated_20261103/prepare_smoke.py
PYTHONPATH=. /home/zjnu/anaconda3/envs/UAV_CV/bin/python experiments/c2_gated_20261103/preflight.py
PYTHONPATH=. /home/zjnu/anaconda3/envs/UAV_CV/bin/python experiments/c2_gated_20261103/train.py --arm spatial --smoke
# 完成四组短跑、完整评估链验证、审查并提交源码后再启动唯一队列：
PYTHONPATH=. /home/zjnu/anaconda3/envs/UAV_CV/bin/python experiments/c2_gated_20261103/launch.py
```

启动器检查空闲GPU、干净工作区、目标分支和重复队列，复制已提交源码到server-only `runs/detect/c2_gated_20261103/source`。所有子进程从该冻结副本运行，每阶段重验源码SHA；tmux会话为 `c2-gated-20261103`。

队列依次运行c2→scalar→spatial→fusion，每组训练后正式预测与评估，最后同场测速。任一阶段失败停止队列，保留失败状态、日志和权重；不自动回退、重复训练或停止其他GPU进程。

状态：服务器仓库下 `runs/detect/c2_gated_20261103/queue_status.json`；各组 `status.json`；日志 `logs/{arm}-{phase}.log`。只检查GPU空闲不足以断言成功。

## 权重保存与证据

自定义训练器用标准库pickle，CPU暂存FP32 raw模型、EMA、优化器、调度器、scaler、early-stopper与RNG，fsync并重新加载验证后原子替换last/best。完整权重不strip，不覆盖旧实验。新环境未调用旧dill路径，但这不代表原环境SIGSEGV已彻底根治。

当前入口拒绝隐式续训或覆盖既有权重；检查点保留了恢复所需状态，若中断需先核查并实现明确的raw-state恢复，不能把EMA warm start称为无缝恢复。

逐组完成后将完整CSV、诊断、实际配置、正式指标、框架原生曲线归档到本目录 `raw/`、`figures/`；逐图预测、权重、完整日志与源码副本只留服务器。每完成一组，控制器仅提交本报告目录并推送本任务分支；如分支改变或有其他暂存工作则停止自动提交并保留本地产物。

启动记录：[正式队列](raw/launch.json) · [冻结源码身份](raw/source_identity.json) · [首次字体初始化修复](raw/startup_font_receipt.json)。

阅读：[预检证据](raw/preflight.json) · [实验配方](raw/experiment_spec.json)。完整结果将写入本目录RESULTS.md和comparison.csv，未完成前不把短跑结果当收益。

证据链：共享初始化与整网等价验证→排除初始P2缩放和随机共有权重差异→再用同条件150轮独立评估检查结构收益。真实短跑→验证反传、验证、保存和加载链路→通过后启动正式队列；短跑不构成精度证据。
