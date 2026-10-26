# 5090：P2＋P5局部注意力设计审查与GPU预检

分支 `5090-detrscope-query-budget`，远程设计提交 `ac73897a`；已拉取确认最新，并在RTX5090 D v2执行。本轮是**结构性能预检和真实GPU工程短跑，不是四组150轮精度结果**。

[完整审查与报告](REPORT.md) · [四组耗时](comparison.csv) · [配对区间](paired_comparisons.csv) · [门槛判断](assessment.json) · [原始设计](../../../experiments/p2_p5_la_5090/README.md)

四组正交设计及配对初始化合理，但当前NAIFI实现未通过自身预定的原生FP32速度门槛：

| 相对C2 | C2 | C2＋LA | 组合变化 |
|---|---:|---:|---:|
| 整网无标记 | 7.2882 ms | 7.4155 ms | 慢1.75% |
| Neck/Encoder | 2.5922 ms | 2.6846 ms | 慢3.56% |
| P5模块 | 0.2880 ms | 0.4052 ms | 慢40.72% |

10轮中没有一轮组合更快。P5包含在Neck/Encoder内，不能重复相加；分段有独立标记开销，整网使用无标记测量。batch1、640、GPU常驻未融合FP32，不含前后处理；仅随机初始化结构测速，不产生AP。不能据此宣称TensorRT/Orin也变慢。

已完成：22项测试＋9项子测试；四组10轮交错测速、24000个CUDA区间；组合模型真实VisDrone数据1轮/3%训练、完整548图训练内验证；新best.pt的8图原生FP32重载评估。P2/QKV/RPB梯度有效、框更新有限、best/last/resume保存正常。短跑精度不作结构比较。

修复两个真实执行问题：真实图像测速入口补连续NCHW及输入身份；手动构造的评估Predictor补imgsz，新增真实尺寸预处理回归测试。修复前日志、第一轮测速和部分归档尝试继续留服务器，路径/哈希及精简摘要见[验证记录](raw/validation.json)；接受的证据只来自修复后复测/重载。

按原方案“先筛速度”的规则，**四组150轮尚未启动**。下一步应先诊断局部注意力实现或明确另一条部署路径的验收目标；此结果不否定其精度潜力。

- [完整CUDA样本/环境/源码](raw/benchmark.json)、[数据审计](raw/asset_audit.json)。
- [真实短跑状态与checkpoint路径/SHA](raw/smoke.json)、[健康记录](raw/smoke_health.json)、[曲线](raw/smoke_curve.csv)、[8图重载检查](raw/smoke_evaluation.json)。
- 服务器根目录：`runs/p2_p5_la_5090/5090-preflight-20261026-s0-r1/`；测速：`runs/p2_p5_la_5090/5090-architecture-screen-20261026-s0-r2/`。权重、预测、完整日志及运行状态仍仅留服务器。

从仓库文本即可离线重建报告和校验，不读取服务器资产：

```bash
python3 -m experiments.p2_p5_la_5090.preflight_report --verify
python3 -m experiments.p2_p5_la_5090.preflight_report --output /tmp/5090-la-preflight-replay
```

`MANIFEST.sha256`仅包含随Git提供的文件。评估脚本和回归测试在结构测速后修复，前后哈希及原因明确记在验证记录中；测速所执行的模型、初始化器和基准代码内容保持可核验。
