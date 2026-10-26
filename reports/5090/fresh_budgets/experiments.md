# 5090 四组从零训练

快照时间（UTC）：`2026-10-26T06:20:17.530891+00:00`；队列状态：`completed`。

150e、seed0、batch8、640、AdamW、FP32；无pretrained、无旧权重微调、无旧checkpoint续训。

此文件在每轮和阶段切换时本地自动刷新；Git云端只显示最近推送的快照，不是实时服务器状态。工程短跑指标不进入正式模型比较。

| 实验 | 角色 | 训练状态 | 已完成轮数 | 内部mAP50–95 % | 正式AP % | APs % | TRT FP16 ms |
|---|---|---|---:|---:|---:|---:|---:|
| c2-q200-p2-d3 | fresh_150e | completed | 150/150 | 19.232 | 16.533 | 11.109 | 0.8438 |
| c2-q200-p2-d3 | engineering_gate | completed | 1/1 | 0.000 | — | — | — |
| c2-q300-p2-d3 | fresh_150e | completed | 150/150 | 21.190 | 19.263 | 12.739 | 0.8549 |
| c2-q300-p2-d3 | engineering_gate | completed | 1/1 | 0.000 | — | — | — |
| c2-q300-p4-d1 | fresh_150e | completed | 150/150 | 16.946 | 14.947 | 9.414 | 0.6810 |
| c2-q300-p4-d1 | engineering_gate | completed | 1/1 | 0.000 | — | — | — |
| c2-q300-p4-d2 | fresh_150e | completed | 150/150 | 19.330 | 17.381 | 11.394 | 0.7773 |
| c2-q300-p4-d2 | engineering_gate | completed | 1/1 | 0.000 | — | — | — |

正式AP与训练内部指标分开；正式AP为原生未融合FP32 COCOeval，TensorRT FP16只测性能。完整必要训练曲线保存在同目录experiments.json，checkpoint/原始日志/逐图预测仅服务器保存。
