# P2完整四尺度实验结果

当前已归档数据如下。正式指标为converted VisDrone val548、COCO maxDets100。

| arm | budget | checkpoint | AP | APs | AP75 | ΔAP vs B0 |
|---|---:|---|---:|---:|---:|---:|

本队列为seed1单seed实验。本轮只训练P2-Full，B0/C2采用历史seed0结果作参考；初始化和workers差异限制因果解释。
当轮last和截至该预算原生fitness选出的best分开评估，原生mAP不代替正式COCO AP。
Tiny recall采用640方形缩放后短边分桶，COCO按类别匹配、忽略GT排除，query来源层属于初始化来源诊断。

| historical model | AP | APs |
|---|---:|---:|
| historical_b0 | 27.773 | 19.481 |
| historical_c2 | 28.722 | 19.975 |

历史对照完整配方、checkpoint SHA和比较限制见historical_references.json。
