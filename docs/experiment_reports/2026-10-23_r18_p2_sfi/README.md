> 更新：四组150轮与正式评估已完成；[2026-10-24核查与完整训练证据](../2026-10-24_r18_p2_sfi_audit/README.md)。以下保留当时启动快照。

# RT-DETR-R18 P2-SFI：实验设计与启动记录

2026-10-23，分支 `feat/r18-p2-sfi`。本页为启动快照，尚无完整150轮精度结论。

重新训练C0基线，比较C1普通SPD、C2步长卷积、C3轻量SPD。四组都不含DDF/FFT，保持300 queries、3层、P3/P4/P5 Decoder。

[完整设计与复现命令](../../../experiments/r18_p2_sfi/README.md) · [真实数据预检查](../../../experiments/r18_p2_sfi/preflight.json) · [冻结源码SHA](../../../experiments/r18_p2_sfi/source_sha256.json)。

14项测试通过；四组1%训练数据/1轮短跑完成，batch8/640/FP32，支路梯度有限且非零；548图正式评估及Encoder/D1–D3健康汇总链路通过。C3短跑精度不用于结构收益判断。

| 组别 | 参数量 | THOP GFLOPs@640 |
| --- | ---: | ---: |
| C0 | 20,094,328 | 58.317 |
| C1 | 20,356,216 | 61.666 |
| C2 | 20,135,032 | 58.834 |
| C3 | 20,086,712 | 58.219 |

四组共同初始化SHA一致，C0完整初始权重SHA与历史R0一致。现有R18配方：150轮、seed0、AdamW1e-4、batch8/nbs64、3轮预热、bias预热LR0、FP32。CUDA非确定性与历史R0一致。

按C0→C3→C2→C1顺序串行训练；每组结束后自动正式COCOeval、每类别/尺度指标、全val回归健康汇总和同口径FP32延迟测试。失败或源码变化停止队列。

服务器tmux为 `p2-sfi-20261023`，训练目录为 `/home/zjnu/文档/ZHJ_WorkSpace/ultralytics/runs/detect/r18_p2_sfi_20261023`，状态在 `experiments/r18_p2_sfi/queue_status.json`。可变进程状态/锁、完整日志、模型权重、预测和源码压缩快照仅在服务器保留；本页不将这些当作云端已有文件。

新测正式结果由 `experiments/r18_p2_sfi/evaluate.py` 生成 `c*_formal_metrics.json`、`formal_metrics_comparison.csv` 和 `RESULTS.md`，完成后应按仓库规则归档更新。当前Git仅提供设计、源码、源码SHA和预检查。

测试维护说明：旧NA测试在干净起点917109ac同样有两项失败，原因是新增variant-only元数据后仍要求整份字典相等。本次仅修正测试断言：共同元数据一致，trainer与同variant比较；未改旧NA模型或训练配方。
