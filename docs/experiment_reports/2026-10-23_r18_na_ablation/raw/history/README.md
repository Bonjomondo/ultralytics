# 历史基准摘要

云端只保留 [summary.json](summary.json)：每批次的环境、模型/精度汇总和原文件路径/大小/SHA256。历史逐次计时 JSON 继续保留于服务器同目录，由 Git 忽略；当前正式基准的逐次计时仍完整追踪在 `../inference_20261023/benchmark.json`。

报告生成只依赖 summary.json，不依赖被忽略的历史原始文件。
