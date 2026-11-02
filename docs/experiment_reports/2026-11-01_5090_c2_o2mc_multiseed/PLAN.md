# C2 + O2M-C：seed1/seed2冻结协议

2026-11-01，Asia/Shanghai。沿用[seed0协议](../2026-10-31_5090_c2_o2mc/PLAN.md)，只改变随机种子及其派生初始化。原模型、损失、数据、增强、优化器和正式评估保持不变。现有seed0代码与228项源码SHA通过核验，新增脚本仅提供多seed调度、身份门禁及归档。

## 执行顺序

单5090依次执行seed1和seed2。每组先审计6471训练图/548验证图的数据身份，运行原CPU门禁，再用65图训练1轮并保存/重载checkpoint，做8图COCO及去辅助头逐位等价检查；全部通过后从零跑正式150轮，不复用smoke或历史best权重。正式完成后评估best完整548图，再用相同入口复评对应历史C2基线。任何门禁失败保留目录与日志并停止队列，不改配方重试。

## 固定配置

原C2模型；300 queries、3层Decoder、每尺度4采样点。640 scale_fill、RGB/255，FP32，AMP/TF32关闭，micro batch8，warmup后累积8次，有效batch64，BatchNorm仍为micro batch。AdamW lr1e-4，lrf1，weight_decay1e-4，warmup3轮/2427 iterations，mixup0.2，workers0，150轮固定预算，不自动延长。

辅助权重0.5，VFL/GIoU/DFL权重1/2.5/0.5，前30轮ATSS，第31轮起TAL，其余参数与原recipe.yaml一致。P2初始化seed=seed+1000，辅助头seed=seed+4000。实际主路径初始state_dict与历史同seed基线重建结果逐项/哈希完全一致，并核验历史初始化及best checkpoint SHA。

直接使用UAV_CV解释器；完整环境、实际args、cwd、PYTHONPATH、tmux、PID、命令、源码/配置/GT/数据SHA均归档。历史seed1/2基线训练环境与本轮的差异如实保留，同一UAV_CV复评基线精度。

## 评估与完成条件

best由内部mAP50–95选择。正式COCO bbox使用原生未融合FP32、batch2、640、conf0.001、每query最高类别、无NMS、最多300输出，标准maxDets=[1,10,100]；原图annotation area划分尺度。不是VisDrone官方挑战分数。先检查候选/基线协议完全相同，再比较AP/AP50/AP75/APs/APm/APl及逐类别指标。

150轮、150条健康和150条辅助诊断均齐全才可完成；保留best/last/resume、未剥离优化器和EMA恢复资产。到达预算不代表收敛，按原平台期标准报告。单独保存原生去辅助头模型和逐位等价证据，不据结构相同宣称部署速度。

## 结果与证据

各seed归档包含完整150轮CSV、逐轮健康/分配/梯度JSONL、实际训练参数、训练/正式评估/基线复评JSON、数据审计、初始化和权重SHA、smoke与CPU门禁记录、SHA清单。权重、完整预测、完整大日志和源码快照留服务器，通过receipt引用，不纳入Git。seed0紧凑证据复制到本目录seed0以便三seed汇总不依赖服务器大文件；原归档不变。

| 证据 | 检查结论 | 路径 |
|---|---|---|
| 队列状态和启动凭据 | 哪个seed已开始/完成/失败 | runtime/queue_status.json、runtime/queue_plan.json、runtime/launch.sh |
| 每组训练与评估 | 150轮完整性、best绑定、同协议对照 | seed1/results、seed2/results |
| 配对CSV/统计JSON | 三seed差值、均值、样本标准差及正向次数 | paired_seeds.csv、paired_statistics.json |

不以三seed当作统计显著性证明；没有C0-Aux，不能宣称结构与监督协同。重点同时保留大目标退化和少样本类别分母。

## 查看与复现

从仓库根目录查看运行状态：

```bash
cat runs/c2_o2mc_5090/5090-c2-o2mc-multiseed-20261101-r1/queue_status.json
cat runs/c2_o2mc_5090/5090-c2-o2mc-multiseed-20261101-r1/seed1/status.json
```

实际启动命令保存在runtime/launch.sh；它使用独立tmux和文件重定向。运行目录严格拒绝覆盖和自动续训。中断后先审计准确checkpoint中的epoch/optimizer/EMA，再单独处理。

仅用紧凑归档重新生成三seed汇总（需本仓库及现有Python依赖，不读取服务器权重/完整预测）：

```bash
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
/home/ubuntu/miniconda3/envs/UAV_CV/bin/python -c 'from pathlib import Path; from experiments.c2_o2mc_5090.multiseed_support import summarize; summarize(Path("docs/experiment_reports/2026-11-01_5090_c2_o2mc_multiseed"))'
cd docs/experiment_reports/2026-11-01_5090_c2_o2mc_multiseed
sha256sum -c MANIFEST.sha256
```

不自动提交或推送Git。
