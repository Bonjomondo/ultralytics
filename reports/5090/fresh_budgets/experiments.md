# 5090 四组从零训练

快照时间（UTC）：`2026-10-25T08:38:56.805027+00:00`；队列状态：`running`。

150e、seed0、batch8、640、AdamW、FP32；无pretrained、无旧权重微调、无旧checkpoint续训。

此文件在每轮和阶段切换时本地自动刷新；Git云端只显示最近推送的快照，不是实时服务器状态。工程短跑指标不进入正式模型比较。

| 实验 | 角色 | 训练状态 | 已完成轮数 | 内部mAP50–95 % | 正式AP % | APs % | TRT FP16 ms |
|---|---|---|---:|---:|---:|---:|---:|
| c2-q200-p2-d3 | engineering_gate | completed | 1/1 | 0.000 | — | — | — |
| c2-q300-p2-d3 | fresh_150e | running | 4/150 | 0.399 | — | — | — |
| c2-q300-p2-d3 | engineering_gate | completed | 1/1 | 0.000 | — | — | — |
| c2-q300-p4-d1 | engineering_gate | completed | 1/1 | 0.000 | — | — | — |
| c2-q300-p4-d2 | engineering_gate | completed | 1/1 | 0.000 | — | — | — |

正式AP与训练内部指标分开；正式AP为原生未融合FP32 COCOeval，TensorRT FP16只测性能。完整必要训练曲线保存在同目录experiments.json，checkpoint/原始日志/逐图预测仅服务器保存。
