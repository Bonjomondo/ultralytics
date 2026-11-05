# 原150轮C0辅助监督补齐与后续研究计划

2026-11-04，用户最新顺序：先补齐5090-c2-o2mc-20261031分支缺少的C0+O2M-C，再新建分支跑作者配方新方案。原协议seed0补齐已完成，后续作者配方在新分支独立推进。

## 已完成且复用

| 实验 | 协议 | seed | 复用方式 |
|---|---|---|---|
| C0 | 旧配方150轮、FP32 batch8/nbs64 | 0/1/2 | 不重训，核验初始状态和best SHA后同环境复评 |
| C2 | 同上 | 0/1/2 | 复用正式评估 |
| C2+O2M-C | 同上，aux0.5 | 0/1/2 | 复用已归档完整结果，AP差+1.113486/+1.339542/+1.252266pp |
| C0+O2M-C | 同上，aux0.5 | 0已完成；1/2延后 | seed0训练、正式评估和归档完成 |

原C2三seed报告：[入口](../2026-11-01_5090_c2_o2mc_multiseed/README.md)。官方Paddle RT-DETRv3 VisDrone最终AP25.640552来自另一个仓库与预训练/调度协议，不占用上表任何单元。

## 本次冻结配方

完全沿用旧C2+O2M-C的模型/损失实现：主结构改为C0，300 queries、3层decoder、每尺度4点；aux0.5，ATSS前30轮，之后TAL，VFL/GIoU/DFL权重1/2.5/0.5，不修改原O2O含encoder/decoder/denoising主损失。C0不是移除辅助头的C2，须使用原C0 YAML。

UAV_CV直接解释器，固定150轮，640 scale_fill，micro batch8、warmup后累积8、有效batch64，FP32/无AMP/TF32，AdamW常量lr1e-4、weight_decay1e-4、warmup3轮、bias warmup0、mixup0.2、workers0、从零、无预训练。seed0/1/2与各自原C0初始state_dict、r0 SHA、checkpoint SHA逐一匹配；实测三个seed门禁已通过。辅助头独立seed4000+seed。BatchNorm仍是micro batch统计。

每组依次数据审计→C0工程门禁→65图1轮训练与保存/重载→8图COCO及去辅助头等价检查→正式150轮→best完整548图评估→原C0基线同环境复评→归档。失败保留证据并停止队列。实验root拒绝覆盖，不自动续训。

正式评估采用原native未融合FP32，batch2/640/conf0.001，无NMS，最多300预测，COCO maxDets=[1,10,100]，原图annotation area。三个新结果与对应C2归档的protocol必须严格一致，才生成factorial.csv和逐类交互分析。主指标AP/AP50/AP75/APs/APm/APl/AR100，所有差为百分点。

交互项I=(C2+Aux−C2)−(C0+Aux−C0)。不凭单seed或仅正均值宣称显著协同。保留完整150轮曲线、逐轮健康/梯度、正式逐类尺度指标、环境/配置/源码/GT/权重SHA、启动命令和门禁日志；best/last/resume、原生去辅助头权重和完整预测留服务器。每轮第三batch梯度诊断是抽样，不称全量机制证明。训练完成不等于收敛。

## 后续分支：作者配方研究（尚未启动）

原始方案见REQUESTED_STUDY.md。待旧协议2×2补齐归档后，再创建独立研究分支和冻结执行计划，按A→B/C→有效组合顺序推进。350轮是否扩大由同协议150轮筛选决定，不直接跨协议复用旧C2辅助收益。

4090已核实来源：/home/zjnu/文档/ZHJ_WorkSpace/reproductions/rtdetr-r18-sodetr-paper-20261027（C0 AP27.7733055）及rtdetr-r18-c2-author-recipe-20261028（C2 AP28.7223432）。两者作者引擎C2层号19/22/25，不是旧5090层号16/23/22。作者配方batch4/nbs4，warmup按全局batch索引2000、bias0.1、恒定lr1e-4、350轮；新方案aux1。C0未保留150轮权重，C2有epoch150快照。评估GT SHA6f57aa…也与旧5090的6f8b86…不同，不能直接拼指标。

作者引擎参考已复制到5090相邻reproductions/rtdetrv3_dense_supervision_5090，仅为迁移准备，未启动。数据迁移准备中；旧实验绝不使用这些标签/GT或替换共享数据。后续需核验数据语义、精确配方、初始化、评估和warmup实现后才能决定复用checkpoint或补基线。

## 状态和复现

当前代码：experiments/c0_o2mc_5090；原C2实现按字节保留。本报告归档于原分支；2026-11-05按用户要求提交并推送，确切提交由Git记录。运行目录runs/c0_o2mc_5090/5090-c0-o2mc-seeds012-20261104-r1。查看queue_status.json以及seedN/status.json。独立tmux与文件重定向，关闭聊天不会停止。

完成后各seed保留results，主表RESULTS.md、factorial.csv、factorial_statistics.json自动更新。原数据、权重、大日志和源码快照只留服务器，SHA清单仅覆盖紧凑归档。

## 最新执行修订

用户要求多seed留到最后。seed0已于2026-11-05 04:29（北京时间）完成，所有阶段退出码为0；seed1/2不自动接续。独立监督器已归档2×2结果并结束旧调度。方案后续仍先做单seed筛选与正式训练，多seed验证最后补。
