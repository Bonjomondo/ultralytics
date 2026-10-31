# 实验计划：先验证 P2，再解释收益，速度线验证 D2

创建于2026-10-27，更新于2026-10-28（Asia/Shanghai）。C2为候选，跨seed稳定性尚未通过；NAIFI暂作支线。[seed1最终结果](REPORT.md)。下列状态应与本目录证据及服务器实时状态一起阅读。

| 顺序 | 任务 | 本轮安排 | 完成判据 |
|---|---|---|---|
| P0-1 | C0-S1、C2-S1，各从零150轮 | 已完成，10月28日10:52正常退出 | 全量6471图、150轮曲线/健康完整，配对初始化核验，各自best完整548图正式COCO评估 |
| P0-2 | 四模型同GT诊断 | 已完成 | 38759个有效GT，IoU .5/.75、类别/尺寸/密集/遮挡/截断/边缘分组；100/300预算、三档conf敏感性 |
| P0-3 | C0/C2 Encoder和D1–D3诊断 | 已复用既有观测及预测完成新汇总 | 候选Coverage、逐层AP、逐GT阶段迁移；严格区分固定query与重匹配视角 |
| P0-4 | C0-S2、C2-S2，各从零150轮 | 10月28日12:36启动；C0→评估→C2→评估串行 | 汇总三对seed的配对差值及波动 |
| P1-1 | C2 D2/D3 TensorRT FP16完整AP | 已完成 | 原三层checkpoint，真实D2早退，完整548图，相同标注/预处理/后处理/maxDets100 |
| P1-2 | Orin NX部署测速 | 待目标设备上执行 | 固定功耗/时钟、TensorRT构建、温度、预热；forward、预处理、传输、后处理和视频端到端分别测量 |
| P2-1 | NAIFI/no-RPB/NATTEN微基准 | 待前两优先级后执行 | P5真实形状、k7、FP32和可运行低精度；核对边界与RPB语义，不能称为等价实现加速 |
| P2-2 | C0/C2长训，或D2恢复 | 条件任务，尚未启动 | 根据多seed及Orin结果择一，组间预算/配方一致 |

## 训练协议与证据

保持原seed0配方：640输入、batch8、150轮、FP32/AMP关闭、AdamW、lr0=1e-4、lrf=1、weight_decay=1e-4、warmup3、warmup_bias_lr=0、nbs64、mixup=.2、close_mosaic=0、patience0、workers0、pretrained=False。相同数据划分、类别顺序和正式评估入口。实际完整参数由训练receipt保存，原训练代码未修改。

对每个seed，C0/C2共有层、AIFI槽及原C0全量参考初始化SHA完全相同；C2的P2槽使用独立seed+1000。两个架构的随机数消耗与非确定CUDA算子仍可能使训练轨迹不同，不声称严格逐位可复现。

开跑前保存数据标注SHA、标签内容SHA、图片路径/大小清单SHA、源码SHA和配置；图片清单不等于逐图像素SHA。每组保存完整曲线、每轮健康、最佳内部mAP选择规则、best/last/未剥离resume checkpoint路径及SHA。正式AP采用未融合原生FP32、batch2、scale_fill、conf=.001、每query最高类、无NMS、最多300输出，COCO maxDets=[1,10,100]。每类别100与全图100不同；诊断300不混入标准AP。

| Seed | C0 AP | C2 AP | C2−C0 pp |
|---|---:|---:|---:|
| 0，既有结果 | 18.661578 | 19.516709 | +0.855130 |
| 1，新训练 | 18.882105 | 18.828508 | −0.053597 |
| 2，训练已启动 | 待测 | 待测 | 待算 |

逐seed先计算AP/APs/AP50/AP75差，再汇报差值均值、样本标准差、最小/最大；分类别结果保留在正式评估summary。预设工程门槛：三对AP差均正、平均至少+0.3pp，C2继续作为主线；这不是统计显著性门槛。APs若不稳定提升，不宣称专门改善小目标。

目前两对AP差均值+0.400767pp、样本标准差0.642568pp；seed1总体AP为负，原全正门槛未通过，补seed2也不能改变这一事实。两对APs差为+0.486214/+0.262335pp，属于初步一致信号。[最终配对表](seed1_results/paired_seeds.csv)、[全指标统计](seed1_results/paired_statistics.csv)。

## 分批执行与监控

仓库根目录为 `/home/ubuntu/文档/ZHJ_UAV/ultralytics`。宿主机GPU可用，隔离沙箱未暴露GPU设备；GPU任务需在宿主机运行。

```bash
# seed1一对已完成；重复此命令会复用完成阶段。
/tmp/zhj-uav-5090-venv/bin/python -u -m experiments.p2_stability_5090.queue --seeds 1

# seed2已通过以下同一队列启动（实际本次传 --seeds 2），已完成seed不再训练。
/tmp/zhj-uav-5090-venv/bin/python -u -m experiments.p2_stability_5090.queue --seeds 1 2

# 查看状态、当前训练轮数与日志路径。
/tmp/zhj-uav-5090-venv/bin/python -m experiments.p2_stability_5090.queue --status
```

seed2持久会话 `5090-p2-stability-s2-20261028`，控制器日志 `logs/controller-seed2.log`。任务根目录 `runs/p2_stability_5090/5090-p2-stability-20261027-r1/`，分组日志在 `logs/`，训练在 `training/`，评估在 `evaluations/`，配对表 `paired_seeds.csv`。启动前重新审计标签内容、图片路径/大小清单、COCO标注及冻结源码，并核对配对初始化和环境，结果见[启动核验](raw/seed2_launch_preflight_20261028.json)、[数据审计](raw/seed2_asset_audit_20261028.json)。本轮小样本smoke使用每组1轮/1%训练数据，只验证工程流程，不能作为结构精度结果。

支持阶段续跑，不冒充任意训练点的精确恢复。若一组中断，保留所有资产，核对checkpoint/优化器/EMA/epoch/CSV/health后再恢复该组；现有从零入口会拒绝覆盖。不要删除目录重跑并沿用旧实验身份，不要用best.pt重启后声称连续训练。源码被冻结，运行时不得修改其Python文件；文档可以继续更新。

## 诊断解释和后续决策

同GT分别统计找回、丢失、共有检出和共有漏检，并保留分母；原图sqrt(area)分档<8/8–16/16–32/≥32px，密集度按每图有效GT数<50/50–100/≥100，边缘距离按GT框到最近图边的像素距离。遮挡/截断使用已有标注字段。置信度与IoU差只在共有检出上取均值，不把匹配变化当作固定query移动。

本轮同时比较C0→C2、LA→C2+LA、C2→C2+LA，以及C0/C2各自D1→D2→D3。Encoder覆盖率和D1已有差别，不能据此直接断言P2的唯一因果作用阶段；固定query/采样点级机制仍可后续深入。DETRScope已有固定query和按GT重匹配的轨迹报告保留在既有深度诊断目录，不以跨模型相同query编号建立目标对应。

Orin阶段应复核D2/D3各自FP16 AP并重新构建目标平台engine。5090 engine不可作为跨设备部署包，5090的0.770/0.868ms不外推为Orin端到端收益。记录板卡/JetPack/CUDA/TensorRT、nvpmodel功耗模式、jetson_clocks设置、温度/频率、batch1/640、构建精度与工作区；至少预热150次、多轮交错。端到端测实际输入视频及逐阶段开销，不能把GPU常驻forward当系统FPS。

若D2在Orin有实际价值，再比较原三层checkpoint直接早退、D2针对性微调、相同checkpoint及预算的D3继续训练；记录训练步数和耗时，与从零D2分开报告。若P2多seed稳定，再配对从零300轮C0/C2；调度方案必须两组一致。NAIFI/NATTEN只先微基准，不自动新增四模型150轮训练。
