# C2 + O2M-C 三seed配对结果

seed0、seed1、seed2均完成150轮训练和完整548图正式评估。相对同seed C2基线的AP平均提升+1.235pp，样本标准差0.114pp；三组均正向，中小目标改善一致，大目标波动，三组均尚未收敛。

[结果汇总](RESULTS.md) · [分析与验收](ANALYSIS.md) · [配对CSV](paired_seeds.csv) · [配对统计](paired_statistics.json) · [逐类比较](class_comparison.csv) · [健康汇总](health_summary.csv)

[冻结协议与复现](PLAN.md) · [完成状态](runtime/queue_status.json) · [seed0](seed0/RESULTS.md) · [seed1](seed1/RESULTS.md) · [seed2](seed2/RESULTS.md)

实际分支`5090-c2-o2mc-20261031`；每组实际源码commit、未提交源码SHA、数据/初始化/权重身份见对应results/training.json和plan.json。复现代码在[实验目录](../../../experiments/c2_o2mc_5090/README.md)，多seed入口为multiseed_queue.py；按协议选择新的run目录，不覆盖历史结果。

权重、完整预测、大日志、源码快照与恢复资产只留服务器；本目录紧凑证据支持离线阅读、比较及重新生成汇总。
