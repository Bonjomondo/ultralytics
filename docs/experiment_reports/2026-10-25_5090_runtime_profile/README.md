# 5090 第一优先级：算子与运行时诊断

2026-10-25（Asia/Shanghai）；分支 `5090-detrscope-query-budget`。阅读 [完整报告](REPORT.md)，原始逐次计时与源码身份见 [raw/identity.json](raw/identity.json)。本次不训练、不改 Query 或 Sampling Points。

训练配方、数据划分和原FP32正式COCO AP见 [原始C0/C2基线报告](../2026-10-24_5090_baselines/README.md)。本次只测性能，原有AP不能自动转用到新的低精度或部署路径。

- [整网运行时对比 CSV](runtime.csv)
- [C2 算子 CUDA 时间](operators-c2.csv)、[模块时间](layers-c2.csv)、[模块×算子交叉表](layer-operators-c2.csv)
- [可悬停的紧凑 Timeline](figures/timeline.svg)、[逐 kernel 文本](raw/timeline-c2.csv)、[CPU 算子和模块时间](raw/cpu-c2.csv)
- [C0 逐次计时与检查](raw/runtime-c0.json)、[C2 逐次计时与检查](raw/runtime-c2.json)
- [C2 ONNX/TensorRT 同轮部署对比](raw/deployment-c2.json)

云端随仓库提供源码、250次/条件的原始计时、3次/路径的profiler摘要、逐kernel紧凑timeline、CPU时间表、输出检查及报告。完整Chrome trace、checkpoint、ONNX、TensorRT engine、inspector和安装环境只保留服务器；绝对路径、大小/SHA记录在JSON中，不是云端必须存在的链接。探索失败和跨进程测量差异见完整报告；不重复上传探索批次大数组。

复测使用原有 `UAV_CV` 派生的临时环境，训练环境未修改。依赖原生C0/C2 best.pt和同一验证图/标注，路径及SHA见identity。CPU/CUDA采样和无profiler计时相互独立。每个模型建议独立进程，避免跨模型常驻状态影响：

```bash
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
export YOLO_CONFIG_DIR=/tmp/zhj-uav-5090-config
UAV_PROFILE_PYTHON=/tmp/zhj-uav-5090-venv/bin/python

# 各自使用新目录，拒绝覆盖；仅推理，不训练。
"$UAV_PROFILE_PYTHON" -m experiments.decoder_budget_5090.profile_runtime_5090 \
  --models c0 --output runs/diagnostics/5090-runtime-c0-new
"$UAV_PROFILE_PYTHON" -m experiments.decoder_budget_5090.profile_runtime_5090 \
  --models c2 --output runs/diagnostics/5090-runtime-c2-new

# 部署依赖安装在独立 /tmp 环境，不改变共享训练环境。
/home/ubuntu/miniconda3/envs/UAV_CV/bin/python -m venv --system-site-packages /tmp/zhj-uav-profiler-new
/tmp/zhj-uav-profiler-new/bin/python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  onnxruntime-gpu==1.30.0 tensorrt-cu13==10.14.1.48.post1
/tmp/zhj-uav-profiler-new/bin/python -m experiments.decoder_budget_5090.profile_deployment_5090 \
  --models c2 --output runs/diagnostics/5090-deployment-c2-new
```

本次PyPI的TensorRT占位源码包在metadata阶段停滞，原版10.14.1.48也缺少匹配的Python 3.11绑定；改用NVIDIA官方索引的post1 wheel，库包SHA为 `f55d59e9f93ebe0967c4bc108fb4068e74cdbc50bef3e6c9936e92f21cf11352`。最终下载/校验/安装日志在服务器 `/tmp/zhj-uav-profiler-20261025-trt-wheel.log`、绑定日志在 `/tmp/zhj-uav-profiler-20261025-trt-bindings.log`，早期失败日志仍保留。本机实际部署环境为 `/tmp/zhj-uav-profiler-20261025-venv`。GPU复测需宿主机CUDA访问，沙箱内不可用时不能替代为CPU性能。

从服务器结果生成新紧凑归档（新目录）及离线报告：

```bash
python3 experiments/decoder_budget_5090/archive_runtime_5090.py \
  --runtime c0=runs/diagnostics/5090-runtime-profile-20261025-r3/assessment.json \
            c2=runs/diagnostics/5090-runtime-profile-20261025-c2-final/assessment.json \
  --deployment runs/diagnostics/5090-deployment-profile-20261025-c2-stream-final/assessment.json \
  --notes docs/experiment_reports/2026-10-25_5090_runtime_profile/raw/backend_notes.json \
  --output /tmp/5090-runtime-archive-new

# 当前仓库报告离线重建与校验：无需GPU、权重、数据、torch或第三方库。
python3 experiments/decoder_budget_5090/report_runtime_5090.py
python3 -I experiments/decoder_budget_5090/report_runtime_5090.py --verify
```

`MANIFEST.sha256`只列随仓库提供的文件。先核验已有manifest再重建；重建会更新生成文件及manifest，不能用重建掩盖未知输入变化。

验证：本次新增的12项专项检查通过，覆盖kernel相关性归属、重叠区间去重、单次Graph发射、缺CUDA证据拒绝，以及原始计时/Graph输出/CPU回退等损坏证据的拒绝。标准库隔离进程校验和原有两个5090归档校验均通过。没有重跑训练。

```bash
PYTHONPATH="$PWD" YOLO_CONFIG_DIR=/tmp/zhj-uav-5090-config \
  /tmp/zhj-uav-5090-venv/bin/python -m pytest \
  experiments/decoder_budget_5090/tests/test_runtime_profile_5090.py -q
```

方法参考：[PyTorch Profiler及开销说明](https://docs.pytorch.org/docs/stable/profiler.html)、[CUDA Graph运行时分析](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)、[TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html)、[ONNX CUDA provider与stream/I/O配置](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。
