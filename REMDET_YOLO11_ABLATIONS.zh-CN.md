# YOLO11n RemDet/MFR 消融实验清单

本组配置以 `yolo11n.yaml` 为 baseline，每次只增加一个实验因素，建议同一数据集、输入尺寸、epoch、batch、seed 下对比。

| ID | 配置 | 变量 | 目的 |
|---|---|---|---|
| A0 | `yolo11n.yaml` 或 `yolo11n.pt` | baseline | YOLO11n 原始结果 |
| A1 | `ultralytics/cfg/models/11/yolo11n-remdet-ced.yaml` | CED-like 下采样 | 验证下采样是否减少小目标信息损失 |
| A2 | `ultralytics/cfg/models/11/yolo11n-remdet-gffn.yaml` | GatedFFN 替换 backbone C3k2 | 验证门控乘法表达能力 |
| A3 | `ultralytics/cfg/models/11/yolo11n-remdet-ced-gffn.yaml` | CED-like + GatedFFN | 验证组合收益 |
| A4 | `ultralytics/cfg/models/11/yolo11n-remdet-ced-gffn-p2.yaml` | 轻量 P2 检测头 | 提升远距离小目标召回 |
| A5 | `ultralytics/cfg/models/11/yolo11n-remdet-ced-gffn-p2-meca.yaml` | MECA 注意力 | 判断注意力收益是否覆盖额外成本 |

示例训练命令：

```powershell
yolo detect train data=VisDrone-local.yaml model=ultralytics/cfg/models/11/yolo11n-remdet-ced.yaml imgsz=640 epochs=300 batch=16 device=0 seed=42 project=runs_remdet_ablation name=A1_ced_s42
```

建议每个配置至少跑同一组 seed：

```powershell
42, 43, 44
```

结果表建议记录：

| ID | Params(M) | FLOPs(G) | mAP50 | mAP50-95 | APS/small | FPS |
|---|---:|---:|---:|---:|---:|---:|
| A0 | | | | | | |
| A1 | | | | | | |
| A2 | | | | | | |
| A3 | | | | | | |
| A4 | | | | | | |
| A5 | | | | | | |
