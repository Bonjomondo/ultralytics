# 网页端 ChatGPT 协同入口

本目录记录 2026-10-21 在 `zjnu-Precision-3660`、`UAV_CV`、RTX 4090 上，对已完成 150 轮的 RT-DETR-R18 + P4 NA5（R1）进行的真实 checkpoint / VisDrone 验收。优先阅读 [REPORT.md](REPORT.md)，再按问题读下面的原始表。

## 建议提供给 ChatGPT 的上下文

> 请分析本目录的 REPORT.md、formal_evaluation.json、per_class_coco.csv、gt_stage_means.csv、baseline_transition_stats.csv、bbox_output_variation.csv、training_metrics.csv、training_health.jsonl 和 diagnostic_cases.json。区分正式 COCOeval 指标、无置信度门槛的 Hungarian 诊断责任、冻结权重干预与训练因果。R1 的 best.pt 与 last.pt 字节相同；它们是训练结束剥离保存的 EMA 模型。当前只有 R1 完成 150 轮，R0 进度见 r0_progress_snapshot.json，不要将其作为等训练预算的完成版对照。先找直接观测支持的问题，再提出需要 R0 完成后验证的假设。

## 文件用途

| 问题 | 文件 |
| --- | --- |
| CUDA / 原生行为 / Hook 清理是否可靠 | `compatibility.json`、`contract_checks.json`、`offline_replay.json` |
| 正式 AP 和类别/逐图表现 | `formal_evaluation.json`、`per_class_coco.csv`、`per_image_coco.csv` |
| Decoder 逐层变化及分母 | `gt_stage_means.csv`、`baseline_transition_stats.csv`、`baseline_summary.json` |
| 回归输出是否仍依赖 Query | `bbox_output_variation.csv`、`bbox_hidden_activity_sample.csv` |
| NA 实际残差与内部探针 | `na_residual_sample.csv`、`probe_statistics.csv` |
| 冻结 D3 bbox delta 的局部影响 | `probe_sample_coco.json`、`zero_d3_sample_coco.json`、`zero_d3_formal_gt_changes.json`、`comparison_group_stats.csv` |
| 代表性退化轨迹 | `diagnostic_cases.json`（按图像 ID 顺序选择，非随机抽样） |
| 训练曲线与 live 参数健康 | `training_metrics.csv`、`training_health.jsonl`、`training_args.yaml` |
| 初始化是否配对、源码是否一致 | `initialization_original.json`、`r0_progress_snapshot.json`、`source_pin.json`、`artifact_index.json` |
| 如何复现 | `reproduce.sh`、`run_validation.py`、`source_adapter.py`、`check_contract.py`、`export_results.py` |

Attention/Sampling 探针只覆盖前 8 张图的 Decoder deformable cross-attention；并未观测 NA 自身的 attention matrix、自注意力或反向梯度。Feature 探针和干预同样仅覆盖这 8 张图。38,759 个 GT 的全量轨迹与正式 COCOeval 则覆盖 548 张图。

完整 tensor / prediction / trajectory 文件保留在服务器的 `runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/detrscope_20261021/`，GitHub 不保存这些大文件。需要某个 GT 的详细全 Query 记录时，用 `image_id` / `gt_id` 请求从该目录导出；不要跨模型对齐 Query ID。
