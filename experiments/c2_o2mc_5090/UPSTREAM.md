# 官方实现参考身份

参考 [clxia12/RT-DETRv3](https://github.com/clxia12/RT-DETRv3)，Apache-2.0；论文 [RT-DETRv3, WACV 2025](https://arxiv.org/abs/2409.08475)。本目录是针对本地Ultralytics C2的PyTorch改写，未运行Paddle实现，也不是完整RT-DETRv3复现。

2026-10-31直接读取服务器 `../RT-DETRv3`，HEAD `349e7d99a5065e7b684118912e6a74178d4f4625`。该仓库另有本地改动，因此下面按实际文件SHA注明参考身份。只读参考，未修改官方仓库。

| 相对官方仓库路径 | SHA256 |
|---|---|
| `ppdet/modeling/heads/ppyoloe_head.py` | `6ac36fe00f32e9a8d0d559ac168d0a9aaa0ea8046f9e13b9017829b04233b4e6` |
| `ppdet/modeling/assigners/atss_assigner.py` | `0f725d402c42f518d463fc0eb6cd91bf2bfd86889b9830dd73f2331a4450368b` |
| `ppdet/modeling/assigners/task_aligned_assigner.py` | `4ced1d0a06ee48b29e0a5b5fe057320eaa962809078ae0076636bbeecf978c36` |
| `configs/rtdetrv3/_base_/rtdetrv3_r50vd.yml` | `92946fa58dc2c10acf32fa51ffca2df98fa7d359697d24a2ba22eb6612d9d79b` |

移植内容：ESE分类/回归stem、分类残差、零初始化预测卷积、17档距离分布、ATSS→TAL、VFL/GIoU/DFL及按assigned score总和归一化。ATSS争用点按全部GT最高anchor IoU消歧，TAL按全部GT最高预测IoU消歧，保持参考实现行为；不强制为极小GT补配正样本。

本地适配：逐图匹配降低密集场景内存，logits版BCE提升数值稳定性，原生P3/P4/P5顺序、Ultralytics Conv/BN和GIoU算子。浮点eps、初始化实现和CUDA tie选择不保证与Paddle逐位相同。辅助权重0.5是本实验选择；30轮切换与官方配置同值，但针对150轮从零训练的有效性仍待验证。
