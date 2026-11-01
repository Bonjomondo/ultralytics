# C2 + O2M-C：第一阶段实验协议

## 研究问题与范围

在相同原生C2结构、150轮预算下，训练期Encoder密集辅助监督能否改善VisDrone总体AP和小目标AP？本轮只做C2-Aux seed0。已有C2 seed0作为配对初始化参考，训练不加载其checkpoint；结束后在同一UAV_CV中重新正式评估该基线权重。C0-Aux及多seed尚未运行，因此不能给出监督与结构的协同结论，也不能以单seed替代稳定性验证。

## 固定结构和损失

原配置 `ultralytics/cfg/models/rt-detr/rtdetr-resnet18-p2-sfi-c2.yaml` 保持不变：P2通过model.14的3×3 stride2卷积注入P3，P3融合RepC3为model.16，Decoder输入model.[16,23,22]，均256通道，对应stride8/16/32；300 queries、3层decoder、8 heads、每尺度4采样点。不给P2额外接头。

CNN辅助头使用各尺度独立的ESE分类/回归stem；分类保留输入残差，回归预测4×17档非负距离分布。零初始化预测卷积，分类prior=.01，回归bias=1，ESE权重std=.001、bias=0。辅助头接收原Decoder输入特征，保持梯度连接。

`Ltotal = Lnative_RTDETR + 0.5 × (LVFL + 2.5×LGIoU + 0.5×LDFL)`，其中 `Lnative_RTDETR` 表示原生RT-DETR全部主损失（含原有encoder、各decoder层及denoising项）。分类VFL alpha=.75/gamma=2；定位包含GIoU，不能只实现VFL+DFL后称为PP-YOLOE损失。三项均除以 `max(sum(assigned_scores),1)`，GIoU/DFL按对应质量加权。

前30轮（zero-based epoch0–29）ATSS：每尺度最近9点、5×stride方形anchor、候选anchor IoU均值+无偏标准差阈值，中心在GT内，目标质量为detach预测IoU。第31轮起TAL：top13、score¹×IoU⁶，按每GT最大alignment和最大IoU归一化。争用点按官方最高IoU规则消歧；不强制极小GT匹配，记录未分配GT。30轮来自本地官方配置，是本次固定选择，未声称这是论文对150轮规定的切换点。

实现是本地PyTorch适配，不是完整RT-DETRv3复现；不加入O2M-T或MGSA。参考身份和与Paddle的实现差异见 [UPSTREAM.md](../../../experiments/c2_o2mc_5090/UPSTREAM.md)。

## 初始化、环境和训练

保留原C2配对初始化：C0参考seed0，共有状态严格复制；P2槽来自seed1000的C2，辅助头独立seed4000，不改变主路径初始化或外部RNG。启动时核对历史C2共有SHA、C0完整参考SHA及P2初始化seed；另保存原生C2全状态SHA和aux状态SHA。不加载任何预训练/teacher/既有best权重。

直接使用 `/home/ubuntu/miniconda3/envs/UAV_CV/bin/python`。实测Python3.11、torch2.14.0+cu130、torchvision0.29.0+cu130、numpy1.26.4、CUDA13.0、cuDNN92400；实际sys.executable/prefix/base_prefix和本地ultralytics/DETRScope路径写入plan与training receipt。首次工程验收缺少polars，在失败进程退出后仅安装polars1.34.0及对应runtime，未升级torch等现有依赖。没有安装pytest，工程门禁可由同一解释器直接运行测试模块。

沿用C2配方：640 stretch/scale_fill输入、batch8、单5090、FP32（AMP/TF32关闭）、AdamW lr1e-4、lrf1、weight_decay1e-4、momentum=.9、warmup3 epoch、warmup_bias_lr0、nbs64、mixup=.2、close_mosaic0、workers0、150轮、patience0、pretrained=False。其他增强和全部实际参数保存到args.yaml及receipt，不按论文默认值猜测。

全量训练6471图，每轮809个iteration；warmup实际为 `max(round(3×809),100)=2427` iterations。梯度累积warmup时按iteration由1插值到8，之后micro batch8×accumulation8=effective batch64；BatchNorm仍使用micro batch统计。调度器按epoch推进，lrf1即常量终点倍率。沿用trainer全模型梯度裁剪max_norm=10，加入aux参数会影响组合梯度的裁剪尺度，属于联合训练的一部分。保留best/last及未剥离resume.pt，最终原生trainer储存FP16 EMA/optimizer，重载FP32不恢复保存时丢失的精度。

数据使用服务器相邻SO_DETR下既有VisDrone2019-DET-YOLO。开跑前审计全部图片头/标签内容、类别顺序、COCO与YOLO坐标关系；核对与seed0基线的label content SHA和image路径/大小inventory SHA。图片清单不是像素SHA，不改写数据集图像、标签或GT。

## 工程门禁、诊断与评价

CPU门禁：已知多正样本/TAL质量、争用点、空GT、ATSS和TAL损失、主路径初始化完全相同、aux_weight0时主损失逐位相同、梯度回流P2、eval/导出跳过aux、原生去头输出逐位相同、checkpoint保存重载。预测卷积零初始化使第一个step的辅助特征梯度为0，梯度回流在优化后验证，不以第一步的零梯度误判断路。

GPU门禁：同640/batch8/FP32，65张训练图1轮及完整548图训练期验证，保存best/last/resume，再用best做8图标准COCO工程评估及原生去头输出等价。小样本AP不是结构精度结果。任何失败使用新root重验，旧目录、日志、源码身份保留。

每轮第三个训练batch抽样：每GT正样本数、未分配GT、P3/P4/P5正样本数、按resize后面积的GT/正样本分布、main/aux损失及加权比；在P2下采样卷积、P3 RepC3第一卷积及融合P3特征上计算main与加权aux梯度范数/余弦。探针用torch.autograd.grad，不修改参数，不写入.grad；正式训练随后执行组合损失反传。余弦抽样不等于全量机制证明，也不是SDK在线梯度功能。

best按内部mAP50–95选择，训练曲线保留150轮。最终独立正式评估：原生未融合FP32，548图，batch2/640/scale_fill，conf.001、每query取最高类并排序、无NMS、最多300预测、COCO maxDets=[1,10,100]；报告AP/AP50/AP75/APs/APm/APl、每类别和尺度。COCO APs使用原图annotation area，与训练resize面积分组不同。AP范围0–1，表中乘100，提升为百分点。转换数据集COCO评估不是VisDrone官方挑战分数。

原生去头模型严格核对主路径state_dict，并复制原生主模块树，保留不在state_dict内的Decoder anchors/valid_mask/shapes缓存。真实输入上逐位核验输出，部署checkpoint不包含aux模块/参数。FP16保存后缓存anchors已量化，不能重建FP32 anchors后仍声称与加载checkpoint逐位相同；该情形已有回归门禁。此实验不做observer计时，不用结构相同代替实测部署速度；TensorRT/Orin性能暂不扩展。

收敛观察最少150轮，最近50轮最佳改善≤.001，连续20轮窗口均值变化绝对值≤.001才标记plateau；本轮固定150轮预算不自动延长，不把到达上限称作收敛。若未收敛，按原样报告，后续延长须配套对照。

## 启动与留存

主流程为 `python -m experiments.c2_o2mc_5090.run pipeline --root <新目录>`，须用上文规定的UAV_CV解释器。独立tmux执行 `bash launch.sh >> controller.log 2>&1`；launch.sh保留cwd、PYTHONPATH、解释器和退出码，status.json保留控制器/子PID、阶段命令、日志及起止。不使用tee。

正式目录为服务器 `runs/c2_o2mc_5090/5090-c2-o2mc-20261031-s0-r2/`，tmux `5090-c2-o2mc-s0-20261031-r2`。首次失败工程目录为 `runs/c2_o2mc_5090/5090-c2-o2mc-engineering-20261031-r1/`（缺polars）；第二次为 `runs/c2_o2mc_5090/5090-c2-o2mc-20261031-s0-r1/`（去头时未保留缓存anchors）。两个目录与原始证据均保留，均未进入150轮训练。正式源码冻结后不修改Python/YAML。训练和评估结束自动生成本报告目录的results与RESULTS.md，只归档紧凑文本；权重、完整预测、完整日志、源码快照不进Git。当前没有提交或推送授权。

拿到归档后可仅用随仓库的紧凑结果重生成报告，不访问服务器大文件：

```bash
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
/home/ubuntu/miniconda3/envs/UAV_CV/bin/python -m experiments.c2_o2mc_5090.archive \
  --render-only --output docs/experiment_reports/2026-10-31_5090_c2_o2mc
# 在报告目录运行sha256sum -c MANIFEST.sha256检查已提供文件。
```
