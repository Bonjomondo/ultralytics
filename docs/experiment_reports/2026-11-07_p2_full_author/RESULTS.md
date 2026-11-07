# P2完整四尺度实验结果

当前已归档数据如下。正式指标为converted VisDrone val548、COCO maxDets100。

| arm | budget | checkpoint | AP | APs | AP75 | ΔAP vs B0 |
|---|---:|---|---:|---:|---:|---:|

本队列为seed1单seed实验。三种架构共有骨干及语义兼容层配对，四尺度decoder形状变更层有单独初始化记录。
当轮last和截至该预算原生fitness选出的best分开评估，原生mAP不代替正式COCO AP。
Tiny recall采用640方形缩放后短边分桶，COCO按类别匹配、忽略GT排除，query来源层属于初始化来源诊断。
