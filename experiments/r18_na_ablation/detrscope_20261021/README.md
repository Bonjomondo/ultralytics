# R18 + P4 NA5：detrscope 真实权重验收

[实验结论](REPORT.md) · [网页端 ChatGPT 入口](CHATGPT_CONTEXT.md) · [总体实验设计](../../rtdetr_r18_na.md)

2026-10-21 在 `/home/zjnu/文档/ZHJ_WorkSpace/ultralytics`，使用 `UAV_CV` 与 RTX 4090，验证已经完成 150 轮的 `rtdetr-r18-r1-150e-s0`。完整观测与 COCOeval 覆盖 548 张真实验证图、38,759 个转换后 GT；内部探针和冻结权重干预覆盖前 8 张图。R0 仍在训练，本目录不提供等预算 R0/R1 性能结论。

## 文件与 Git 边界

这里保存复现源码、源码 SHA、配置、训练曲线副本、精简统计、逐类/逐图 COCO 表及代表性 GT 轨迹。完整预测、Tensor、逐图全 Query 轨迹、HTML 报告和失败日志保留在原 run 下及 `logs/`，不提交权重或图片。

原始记录：

```text
runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/detrscope_20261021/
  baseline/                # 全量观测与正式评估
  probe_sample/            # 8 图 Feature / cross-attention / Sampling
  identity_sample/         # 8 图 D3 bbox delta ×1
  zero_d3_sample/          # 8 图 D3 bbox delta ×0
  offline_replay/          # 同一原始记录的独立离线重建
  zero_d3_comparison/      # 同 GT 对照报告
```

## 复现

依赖相邻 `detrscope` 仓库的 `0c8624a32bde8fbac5cd6a59e6f2a544225b5051` 源码。`source_adapter.py` 显式校验本项目 201 个 Python 源文件 SHA，继承原生 hook collector，复用全部核心分析算法；它是本实验的源码适配器，不冒充官方 wheel 适配器。

脚本会激活 `UAV_CV`，仍使用 Python 完整路径。选择新的原始输出目录以保留旧证据，长任务须放 tmux 中且只重定向到日志：

```bash
cd /home/zjnu/文档/ZHJ_WorkSpace/ultralytics
/home/zjnu/anaconda3/bin/tmux new-session -d -s r18-debug-replay \
  'bash experiments/r18_na_ablation/detrscope_20261021/reproduce.sh \
    /home/zjnu/文档/ZHJ_WorkSpace/ultralytics/runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/detrscope_reproduction \
    >> logs/detrscope_r18_reproduction.log 2>&1'
```

`reproduce.sh` 先重新采集，再离线分析/评估/核对重建，最后执行契约检查和导出表。它会更新本目录的生成统计；复现后先检查 Git diff。`run_validation.py offline` 可恢复已有且 SHA 完整的分析；未完成分析会另行保留，不能当作完成结果读取。

环境范围：本机 `UAV_CV` 是 Python 3.11.17 / PyTorch 2.1.2+cu121；SDK 发布元数据要求 torch>=2.6。本次通过源码执行验证具体路径，不能据此声称 torch 2.1 全 API 受支持。初次未绑定 CPU 的离线进程出现 Python TypeError / 段错误；复现命令固定离线 CPU 24–27，并启用 Python debug allocator/dev mode，具体尝试与重建结论见 REPORT.md。没有升级训练环境或调整正在运行的 R0。
