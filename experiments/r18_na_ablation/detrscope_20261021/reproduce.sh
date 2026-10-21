#!/usr/bin/env bash
set -euo pipefail
REPO="${R18_REPO:-/home/zjnu/文档/ZHJ_WorkSpace/ultralytics}"
SDK="${DETRSCOPE_REPO:-/home/zjnu/文档/ZHJ_WorkSpace/detrscope}"
RAW="${1:-$REPO/runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/detrscope_20261021}"
TRACKED="$REPO/experiments/r18_na_ablation/detrscope_20261021"
source /home/zjnu/anaconda3/etc/profile.d/conda.sh
conda activate UAV_CV
cd "$REPO"
export PYTHONPATH="$REPO:$SDK/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONMALLOC=debug
PYTHON=/home/zjnu/anaconda3/envs/UAV_CV/bin/python
ARGS=(--repo "$REPO" --sdk "$SDK" --tracked "$TRACKED" --raw "$RAW"
  --checkpoint "$REPO/runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/weights/best.pt"
  --gt /home/zjnu/文档/ZHJ_WorkSpace/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json
  --images /home/zjnu/文档/ZHJ_WorkSpace/datasets/VisDrone2019-DET-YOLO/VisDrone2019-DET-val/images --batch 2)
"$PYTHON" -u "$TRACKED/run_validation.py" collect "${ARGS[@]}"
taskset -c 24-27 "$PYTHON" -X dev -u "$TRACKED/run_validation.py" offline "${ARGS[@]}"
taskset -c 24-27 "$PYTHON" -X dev "$TRACKED/check_contract.py" --repo "$REPO"
taskset -c 24-27 "$PYTHON" -X dev "$TRACKED/export_results.py" --repo "$REPO" --raw "$RAW"
taskset -c 24-27 "$PYTHON" -X dev "$TRACKED/check_native_val.py"
taskset -c 24-27 "$PYTHON" -X dev "$TRACKED/finalize_evidence.py" "$RAW"
