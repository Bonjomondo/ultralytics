# P2完整四尺度实验

在SO-DETR作者代码衍生的Ultralytics8.0.201后端中，对比原始RT-DETR-R18、现有C2/P2注入和完整四尺度P2。正式队列为三组各350轮、seed1，记录150/250/350轮的last与截至该预算原生fitness选出的best。

源码依赖 `../c2_channel_author_20261105/backend` 中已被Git追踪的作者后端。`models.py`在本实验进程内为解析器注册stride-aware decoder，启动器将脚本、配置及这个后端一起冻结到运行目录。完整源码身份以source_identity.json为准。

在仓库根目录使用服务器UAV_CV Python环境执行。以下命令供复现准备阶段使用，正式运行的目录应由start.py一次性创建。

```bash
cd experiments/p2_full_author_20261107
export AUTHOR_P2_REPO=/home/zjnu/文档/ZHJ_WorkSpace/ultralytics
export PYTHONPATH="$PWD"
export YOLO_AUTOINSTALL=false
python preflight.py
python train.py --arm full --smoke
python verify.py
python dense_probe.py
```

实际项目入口和完整评估口径见 `../../docs/experiment_reports/2026-11-07_p2_full_author/README.md`。启动器还要求tiny recall合成匹配测试证据通过、已提交源码、空闲GPU和无重复队列。

```bash
python start.py
```

经过故障审计后可使用 `python start.py --resume-controller` 恢复父控制器。每个训练阶段最多累计两次尝试，评估/测速最多三次。已完成阶段通过queue_status.json的done列表跳过。
