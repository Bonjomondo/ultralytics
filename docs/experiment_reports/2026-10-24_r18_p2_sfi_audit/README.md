# P2-SFI 后处理核查与150轮结果

2026-10-24，分支 `feat/r18-p2-sfi`；输入提交 `38e574cc`。

后处理确有排序后仍使用旧顺序score掩码的问题，已改为对排序后的pred[:,4]做阈值过滤。
四组各548图、164400个Query，在conf=0.001下均无Query被过滤；每组修复前后预测JSON SHA256完全相同，正式COCO指标也完全相同。
因此该bug不影响本次四个已选best权重的验证结果。未保存的历史epoch无法追溯逐Query分数，不能据此宣称所有训练期间best选择均完全不受影响。

[逐组核查摘要](raw/postprocess_comparison.json) · [修复后正式结果](../../../experiments/r18_p2_sfi/corrected_20261024/RESULTS.md) · [完整指标表](../../../experiments/r18_p2_sfi/corrected_20261024/formal_metrics_comparison.csv)

| 组别 | 正式AP (%) | APs (%) | 内部best轮次 | 内部best mAP (%) | 最后两段10轮均值 (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| c0 | 18.705 | 12.104 | 150 | 20.699 | 20.173 → 20.534 |
| c1 | 18.830 | 12.254 | 149 | 20.918 | 20.545 → 20.815 |
| c2 | 19.408 | 12.933 | 150 | 21.375 | 20.771 → 21.225 |
| c3 | 18.416 | 12.035 | 150 | 20.394 | 19.788 → 20.231 |

四组末期仍上涨；150轮尚不能视为收敛。正式AP与内部mAP口径不同，不混用。C2保留+0.704pp AP、+0.829pp APs。

## 证据与边界

- raw/ 中四组各包含完整150轮CSV曲线、实际配置JSON与原始初始化记录；共有初始化SHA已交叉校验一致。
- [训练摘要、best轮次和权重SHA](raw/training_summary.json)；权重及完整原始日志只留服务器。
- [类别×尺度GT数量与覆盖图数](raw/gt_category_scale_counts.csv)，尺度端点按COCO面积区间包含，1024/9216边界可重叠；另列非crowd/非ignore数量。
- [类别/尺度增量](raw/c2_vs_c0_class_deltas.json)：bus/van约占净APs增量84%；这是类别平均AP增量分解，不能解释为目标数量或图像贡献。尚未做逐图重采样归因。
- C0/C2同时改变P2接入与P3融合通道，机制仍需独立控制实验；本轮不新增结构、多seed或部署实验。
- 延迟为4090 batch1 fused PyTorch FP32 GPU forward，50次预热/200次计时，不含传输/预后处理，不代表Orin部署。

## 接续训练

按用户要求先运行seed0的C2和C0至平台期。原150轮last.pt已被框架strip_optimizer，优化器状态和epoch被清除，权重为FP16保存的EMA。
本次从各自第150轮last权重重新建立AdamW/EMA，并进行相同3轮预热；属于分阶段warm start，不是无缝恢复或从零连续500轮。
其余配方保持一致：640、batch8/nbs64、FP32、AdamW、恒定1e-4、seed0、原增强。
每组至少追加150轮；之后若最近50轮最高内部mAP相比更早最高值增加≤0.1pp，且最近20轮均值与前20轮均值差绝对值≤0.1pp，标记经验平台期。最多追加350轮（累计预算上限500轮）；触顶不自动宣称收敛。
每50轮同时保留last和当时best；最终另存未清理优化器的resume.pt。若独立平台停止预算不同，报告必须列出轮数，并利用公共50轮检查点作等预算对比。
脚本：`experiments/r18_p2_sfi/continue_train.py`；队列：`convergence_queue.py`。结束自动正式评估并生成最终报告。

原始150轮报告和旧指标保留，修复后结果另存corrected_20261024；原始权重不会覆盖。

复生成本页：PYTHONPATH=. python experiments/r18_p2_sfi/archive_audit.py（仅需Git内原始表格，无GPU/服务器数据）。--collect为服务器证据采集模式，依赖外部权重/GT。

验证：[19项测试及真实数据接续短跑记录](raw/verification.json)。短跑仅检查管线，不参与精度比较。
