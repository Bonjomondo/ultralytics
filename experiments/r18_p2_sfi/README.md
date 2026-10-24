> 2026-10-24更新：C0–C3均已完成150轮；后处理修复后的四组预测SHA和COCO指标保持一致。[完整核查与训练证据](../../docs/experiment_reports/2026-10-24_r18_p2_sfi_audit/README.md)。现按seed0接续C2/C0至经验平台期，不跑多seed。

# RT-DETR-R18 P2-SFI：C0–C3 配对实验

2026-10-23，Asia/Shanghai。仓库 `Bonjomondo/ultralytics`，分支 `feat/r18-p2-sfi`，起点 `917109ac`。
这是新增纯卷积 P2→P3 融合的筛选实验；SO-DETR NoFFT 仅提供设计参考，不复用其训练权重、DDF 或损失。

## 预注册矩阵

| 组别 | 第14层 | 第15层输入 | 研究用途 |
| --- | --- | --- | --- |
| C0 | 原 P3 1×1 Conv：128→256 | P4↑256 + P3投影256 | 新训基线 |
| C1 | P2 SpaceToDepth + 3×3 Conv：256→128 | P4↑256 + P2↓128 + 原始P3 128 | 普通SPD融合 |
| C2 | P2 3×3 stride=2 Conv：64→128 | 同C1 | 普通下采样对照 |
| C3 | P2 SpaceToDepth + PW256→64 + DW3×3 + PW64→128 | 同C1 | 轻量候选 |

融合后均为512通道，进入相同RepC3；P3/P4/P5均为256通道，300 queries、3层Decoder、P5原AIFI保持一致。无FFT/IFFT、复数参数、DDF/abs增强、局部自注意力或新增门控。

C1–C3之间只改变P2下采样实现。相对C0，同时将P3的128→256投影替换为“原始P3 128 + P2支路128”，因此比较的是完整融合替换方案；不能把差异全部归因于P2信息本身。若候选有效，后续需容量/融合控制确认机制。

理论P2卷积MACs（80×80，不计BN/激活/重排/融合）：C1=1.8874368G，C2=0.4718592G，C3=0.1609728G。它们不是全网MACs或实测加速比。

## 固定训练协议

在同一张RTX 4090 / UAV_CV（Python3.11、torch2.1.2+cu121）上顺序执行；完整VisDrone训练集和548图验证集。
150轮，seed0，640，batch8，nbs64，AdamW，lr0=1e-4，lrf=1，weight_decay=1e-4，momentum=.9，FP32/AMP=false，workers0，patience0，cache=false；mosaic1、mixup.2、close_mosaic0。

继承现有R18的3轮预热，bias起始LR=0。此仓库以 `max(round(warmup_epochs * len(train_loader)),100)` 计算预热batch索引，**不采用SO-DETR自定义的2000全局batch预热**。nbs64使梯度累积逐步达到8。deterministic=false与历史R18一致；共有初始张量一致不表示CUDA训练逐比特可复现。

四组均从YAML初始化，无backbone/COCO预训练、无detector checkpoint迁移。baseline按seed0构造，候选新增支路使用seed1000；通过稳定层号复制所有共有张量。第14层因语义改变不作为共有参数复制；每组initialization.json记录共同SHA和例外列表。实际trainer重建也使用此逻辑。

重新训练C0用于本轮同环境比较；历史R0的AP18.650%、APs12.219%只作参考，SO-DETR的绝对AP不作本轮基线。

## 执行与产物

顺序 **C0 → C3 → C2 → C1**，优先获得轻量候选结果；每组150轮后立即正式评估和基准，再进入下一组。任何失败/非有限loss、batch自动变化、源码SHA变化或训练不足150轮都会停止队列。

```bash
cd /home/zjnu/文档/ZHJ_WorkSpace/ultralytics
/home/zjnu/anaconda3/bin/tmux new-session -d -s p2-sfi-20261023 \
  'cd /home/zjnu/文档/ZHJ_WorkSpace/ultralytics; export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 MPLBACKEND=Agg; /home/zjnu/anaconda3/envs/UAV_CV/bin/python -u experiments/r18_p2_sfi/run_queue.py >> logs/p2-sfi-queue-20261023.log 2>&1'
```

- 源码：[训练](train.py)、[评估/性能基准](evaluate.py)、[队列](run_queue.py)。
- 服务器状态：`experiments/r18_p2_sfi/queue_status.json`；该可变进程状态不提交Git。
- 服务器训练产物：`runs/detect/r18_p2_sfi_20261023/rtdetr-r18-c{0,1,2,3}-150e-s0/`。
- 每组日志：`logs/p2-sfi-c{0,1,2,3}-{training,evaluation}-20261023.log`。
- 每组完成后生成 `c*_formal_metrics.json`、`formal_metrics_comparison.csv`、`RESULTS.md`；初始文档不代表已有完整训练结果。
- `source_sha256.json`冻结全部模型/训练/评估源码和配置；源码压缩快照仅在服务器run父目录保存。
- 权重、优化器、完整日志、逐图预测留服务器；正式结果保留checkpoint/GT/预测SHA。按AGENTS规则归档紧凑曲线和结果后再提交。

## 健康与精度评估

每轮前三个真实训练batch记录Encoder、D1/D2/D3末层输入的正激活比例、Query间方差、回归delta幅值/方差、有限性，以及P2支路梯度和训练峰值显存。新增诊断切片只取最后300个matching queries，排除前置denoising queries；兼容字段last_bbox_*仍是旧记录口径。

初始bbox输出投影为零，因此初始化的delta方差为零属于设计行为，不能单凭首批零输出判定回归头失活。健康日志是抽样观测，不单独证明训练机制。正式原生验证另外观测best EMA的全部548图，逐层汇总隐藏激活和delta方差，并核验覆盖图数。

正式精度使用原生fused FP32 Validator导出，再执行同GT、同pycocotools版本的COCOeval bbox、maxDets=[1,10,100]；模型保留300 queries。JSON原生框/分数舍入口径与已有R0一致。内部mAP仅用于曲线和best.pt选择，与正式AP分开报告。

GPU性能统一batch1、640、fused PyTorch FP32，50次预热、200次CUDA Event计时，报告中位/均值/P95及原始计时。仅测GPU forward，不含传输/预处理/后处理，不代表Orin NX/TensorRT延迟。THOP GFLOPs单独标注其算子统计范围。

## 预先设定的晋级门槛

相对本轮C0：AP≥+0.3pp、APs≥+0.5pp；AP75/APm/APl各不低于基线超过0.5pp；FP32同口径中位延迟增加≤20%。这是工程筛选门槛，不是显著性结论。

首轮不自动增加C4/C5门控、长周期或多seed。通过筛选后，再安排seed0/1/2配对长周期与部署验证；失败结果及各类别取舍也须保留。
