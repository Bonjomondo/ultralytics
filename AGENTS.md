# 项目协作与实验归档规则

## Python 环境与进程约定

- 通用规则统一遵守 [总实验规则](../.agent/AGENTS.md)。新实验直接使用 conda `UAV_CV`，解释器 `/home/ubuntu/miniconda3/envs/UAV_CV/bin/python`；禁止系统 Python，不能自行以独立或继承 venv 替代。
- 从本仓库根目录启动，设置 `PYTHONPATH="$PWD:$PWD/../detrscope/src"`，核对 `sys.executable`、`sys.prefix`、`sys.base_prefix`、框架版本、`ultralytics.__file__` 与 `detrscope.__file__`。不能混用 SO_DETR 的同名实现或 pip 上游 wheel。
- 依赖或部署组件缺失时先核对兼容性，在 UAV_CV 内按通用规则处理；不能因 TensorRT 等组件缺失，默认切回历史 venv。健康长任务期间不升级依赖、切换解释器或改写冻结源码，文档整理不触发重训。
- 长任务使用独立 `tmux` 与 `>> 日志 2>&1`，不接 `tee` 到可能断开的终端 pty；进程和实验身份记录遵守总规则。
- 历史环境记录（仅用于定位已启动/冻结实验）：`/tmp/zhj-uav-5090-venv/bin/python` 和 `/tmp/zhj-uav-profiler-20261025-venv/bin/python` 曾由 UAV_CV 通过 `venv --system-site-packages` 创建。2026-10-27 核验 Python 3.11.16、torch 2.14.0+cu130、torchvision 0.29.0、numpy 1.26.4；部署 venv 另有 `tensorrt-cu13==10.14.1.48.post1`。这些是历史事实，不构成新实验的默认入口或当前版本保证；已启动冻结流程按原协议完成。

## DETRScope：功能、边界与调用

SDK源码为工作区相邻 `../detrscope/src/detrscope`，当前本地版本v0.2.0。入口说明见SDK的 `README.md`、`docs/adapters.md`；本fork的实际桥接与实验入口见 [experiments/decoder_budget_5090/README.md](experiments/decoder_budget_5090/README.md)。使用前按实际代码核对能力，不把历史研究系统所有V0/V1/V2功能当作已迁入SDK。

### 已实现的能力

- `Debugger.observe → analyze → report`：保存模型/源码/输入/GT身份、Encoder候选与实际执行的Decoder层框、类别、Reference及Query身份，生成离线HTML/Markdown报告。
- Query/GT轨迹：固定Query、按GT重新匹配、候选覆盖与逐层改善/退化；跨模型按同图同GT比较IoU、正确类别margin和分组统计。不同模型的相同query编号不代表同一目标。
- 探针：明确路径的模块特征，以及原生Deformable Cross-Attention投影对应的attention权重、offsets和采样位置重建。`probes="auto"`按适配器能力选择，features仍需显式模块路径。
- 干预：指定已执行Decoder层的原生bbox head `bbox_delta`缩放；运行前后强制验证scale=1恒等、观测等价和hook卸载恢复。
- 正式评估：`CocoEvaluator`执行标准COCO bbox AP和真实GT匹配；`compare_gt`及CLI `compare-gt`输出Lost/Gained。默认maxDets=[1,10,100]，API的AP范围0–1，报告百分点时乘100。
- 本fork扩展：`coverage.py`计算Encoder Top-K GT Coverage/排名增益；`run.py sweep`改变实际Top-K Query预算并重跑完整Decoder；`screen_decoder_eval_5090.py`复用观测做真实D1/D2/D3早退和正式评估。性能另走无observer路径。

### 能力限制与身份校验

- SDK目前不采集self-attention、不提供在线反向梯度/梯度干预，也未实现Reference/任意Query/head/point mask干预；存在schema不等于已有运行时功能。专项研究代码若另有实现，要明确脚本及验证范围。
- 模型须已加载、`eval()`、eager非export；调用者负责设备、FP32及预处理。当前批协议使用显式stretch/scale_fill、RGB/255；GT为normalized cxcywh，保留原始image/category/annotation ID和原图面积、ignore/crowd信息。
- 普通观测默认核验原生/挂hook/卸载后的输出、RNG及模型状态；早退后的层不补跑。早层类别辅助读出不自动等同独立训练的浅层模型。
- 本fork使用 [PinnedRTDETRAdapter](experiments/decoder_budget_5090/source_adapter.py) 和 `create_source_pin`，校验实际仓库/祖先commit/源码SHA/SDK SHA。SDK原生 `UltralyticsRTDETRAdapter`只接受其支持的官方wheel身份，不用改forward或跳过校验来适配本fork；SO-DETR另用 `SODeTRAdapter`。
- 跨模型比较必须核对输入tensor SHA、预处理、GT身份/坐标及类别映射。Hungarian诊断、Coverage与正式COCO TP/FN/AP分开；外部预测与观测模型的绑定需单独证明。
- 采集/分析/比较使用新输出目录，禁止覆盖旧证据；已有analysis可重生成report，不能重复覆盖analyze。先小样本工程验收，再做全量正式评估；GPU计时不能带observer。

### 本fork可复用命令

以下从本仓库根目录运行。`-new`输出名是示例；实际每次选尚不存在的目录，不覆盖或重写历史source pin。启动前按总规则核验 UAV_CV 中的实际依赖与源码身份，不凭历史 venv 记录推定依赖可用。

```bash
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
UAV_PYTHON=/home/ubuntu/miniconda3/envs/UAV_CV/bin/python

# 只检查导入身份，不运行训练或GPU计算。
"$UAV_PYTHON" -c 'import sys, torch, ultralytics, detrscope; print(sys.executable, sys.prefix, sys.base_prefix); print(torch.__version__); print(ultralytics.__file__); print(detrscope.__file__)'

# 为本fork与SDK创建显式源码pin；祖先commit是该实验分支的已知基点。
"$UAV_PYTHON" experiments/decoder_budget_5090/run.py --sdk ../detrscope pin \
  --expected-commit 6d1d4d1badbecf47970243685a1d712ad3e31559 \
  --output runs/decoder_budget_5090/source-pin-new.json

# CPU合成工程检查：真实原生模型类、随机权重，验证观测与完整Query重执行。
"$UAV_PYTHON" experiments/decoder_budget_5090/run.py --sdk ../detrscope smoke \
  --pin runs/decoder_budget_5090/source-pin-new.json \
  --device cpu --output runs/decoder_budget_5090/smoke-new

# 真实C2权重8图检查；默认预算300/250/200/150，去掉limit才跑全量。
"$UAV_PYTHON" experiments/decoder_budget_5090/run.py --sdk ../detrscope sweep \
  --pin runs/decoder_budget_5090/source-pin-new.json \
  --checkpoint "$PWD/runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1/5090-rtdetr-r18-c2-150e-s0/weights/best.pt" \
  --annotations "$PWD/../SO_DETR/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json" \
  --images "$PWD/../SO_DETR/datasets/VisDrone2019-DET-YOLO/images/val" \
  --device cuda:0 --expected-gpu 5090 --limit 8 --batch 2 \
  --output runs/decoder_budget_5090/c2-sweep-new
```

`sweep`会自动采集、分析、正式评估并保存各预算的结果；raw run目录形如 `c2-sweep-new/q300`。只有raw的独立run可用下面analyze，已经分析过的run直接report。下面路径为待替换的已保存模型记录目录，不是新的模型训练目录。

```bash
"$UAV_PYTHON" -m detrscope adapters
"$UAV_PYTHON" -m detrscope analyze /绝对路径/仅含raw的run
"$UAV_PYTHON" -m detrscope report /绝对路径/已完成analysis的run
"$UAV_PYTHON" -m detrscope compare /绝对路径/模型A记录 /绝对路径/模型B记录 \
  --output runs/decoder_budget_5090/compare-new
"$UAV_PYTHON" -m detrscope coco --annotations /绝对路径/GT.json \
  --predictions /绝对路径/COCO检测结果.json --output /绝对路径/正式评估-new.json
"$UAV_PYTHON" -m detrscope compare-gt /绝对路径/基线正式评估.json \
  /绝对路径/候选正式评估.json --output /绝对路径/GT变化-new.json
```

代码接入时，先用 `create_source_pin(repo, pin_path, expected_commit=...)` 创建新pin，再构造 `PinnedRTDETRAdapter(model, repo=repo, pin=pin_path, class_names=names, category_ids=category_ids)`；之后 `Debugger(model, adapter, output_dir=新目录)`，检查 `capabilities()`，调用 `observe(val_loader, probes="auto")`、`analyze()`、`report()`。`val_loader`必须产出SDK的 `DetectionBatch`，普通训练batch不能未经转换直接传入。bbox干预入口为 `Debugger.intervene(val_loader, InterventionSpec(layer_id=1, scale=0.0), probes="auto")`，也使用独立新目录。

完整raw tensor、预测、注意力/采样大数组只留服务器，归档保留诊断分母、紧凑分组表、协议、哈希与必要报告。当前P2任务的同GT/逐层复用入口见 [experiments/p2_stability_5090/README.md](experiments/p2_stability_5090/README.md)；其中离线COCO状态分析不等于重新执行DETRScope在线观测。

## Git 追踪目标

在控制提交到云端的仓库体积的同时，保留足以供网页端 ChatGPT 独立阅读、比较和分析实验的必要代码、数据和报告。不能只留下结论，也不能把所有运行产物一股脑提交。优先使用可直接阅读的 Markdown、CSV、JSON、YAML 和源码，不以 ZIP/TAR、Base64 或截图代替必要的文本数据。

## 应提交的内容

- 模型实现、训练/评估/基准脚本、配置及复现命令。
- `docs/experiment_reports/README.md` 索引及按日期/实验命名的汇报目录；每个目录必须有入口 README、自包含报告及清楚的证据链接。
- 分析所需的关键指标与对照表：精度、尺度/类别指标（如实验产生且分析需要）、参数量、延迟、FPS、显存、基线差异、完整必要训练曲线。保留负面结果、失败原因及解释边界。
- 数据划分、评估协议、随机种子、训练设置、软件/硬件环境、最终 checkpoint 路径与 SHA256、源码身份信息。区分内部 mAP 与正式 COCO AP、百分点与相对百分比、FP32 精度与仅测性能的 FP16。
- 一份有必要且体积可控的当前实验原始证据，例如逐次基准计时 JSON、简短基准日志。相同证据不重复提交多份；Git 本身可以去重相同内容，仍应避免无必要的历史副本。
- 适合阅读的紧凑图表；必要分析数据应有文本表格，不只存在于图中。

## 只留服务器的内容

- 数据集、原始图片/视频、完整逐图预测、框/轨迹/注意力/特征大数组。
- 模型权重、优化器/中间 checkpoint、导出模型、环境、缓存、完整训练日志。
- 源码打包快照、修复备份、重复历史基准原始数据、运行锁/PID 等临时状态。

这些内容不为减小 Git 提交而从服务器删除。需要引用时，提交它们的原路径、大小/哈希、生成方式和必要统计摘要；网页端理解结论所需的数据必须另外保留为可读表格。历史原始数据优先保留紧凑摘要，不把每个历史批次的逐次大数组重复上传。

## 体积检查与可读性

- 默认以单个新增文本/数据文件约 1 MiB、单次新增/修改内容约 5 MiB 为检查提示，不作为机械删除证据的硬限制。超过时先检查重复数据、逐图输出及可用的摘要/外部引用；确有必要的大文件应在提交说明中交代用途和体积。
- 不使用笼统的 `*.json` / `*.csv` 忽略规则丢掉关键实验数据；对确需追踪的短日志、汇总 CSV 和报告图表使用明确的例外。
- 报告链接尽量使用仓库相对路径。区分“云端随仓库提供”与“仅服务器保存”的文件；不能把仅服务器文件写成云端必须存在的相对链接。
- 云端 `MANIFEST.sha256` 只列入随仓库提供的文件。从 Git 检出后，报告生成、数据校验和报告内链接检查必须能够在不访问服务器独有大文件的情况下完成；GPU 复测可依赖文档中明确列出的外部数据集/权重。
- 报告应说明当前分支、实验身份和阅读入口，方便网页端 ChatGPT 选对仓库/分支。此项目当前汇报入口为 `docs/experiment_reports/README.md`。

## 提交与推送

以下是用户已授权提交/推送时的操作流程；文档修改本身不触发自动提交或推送。

- 先检查状态、差异、已有代理说明、远端和当前分支；只暂存本次相关代码与归档文件，不把无关改动或服务器产物夹入提交。
- 明确列出暂存路径，检查暂存文件类型、体积及敏感信息；运行 `git diff --cached --check`、适当的源码/报告校验，检查没有误纳权重、数据集或压缩备份。
- 已通过完整训练/评估和基准的代码，应保留对应验证证据；仅改归档规则或报告时，不为形式重复启动昂贵训练。
- 提交前确认必要文件确实被 Git 追踪。报告脚本使用的紧凑历史摘要、输入数据、索引和校验清单都要包括在内。
- 推送当前任务分支，处理远端更新时保留双方工作；未经明确要求，不强制推送、不改写历史、不擅自合并到默认分支。
- 完成后说明分支、commit、推送结果、主要追踪/排除内容和实际提交体积；不要把服务器本地仍存在的忽略文件称为已删除。
