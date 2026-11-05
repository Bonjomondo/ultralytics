#!/usr/bin/env bash
set -euo pipefail
cd '/home/ubuntu/文档/ZHJ_UAV/ultralytics'
export PYTHONPATH='/home/ubuntu/文档/ZHJ_UAV/ultralytics/experiments/rtdetrv3_dense_supervision_5090/author_engine:/home/ubuntu/文档/ZHJ_UAV/ultralytics/experiments/rtdetrv3_dense_supervision_5090'
export YOLO_CONFIG_DIR='/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/rtdetrv3_dense_supervision_5090/author-stageA-s0-20261105-r1/.yolo'
export YOLO_AUTOINSTALL=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MPLBACKEND=Agg WANDB_MODE=disabled COMET_MODE=DISABLED
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
exec '/home/ubuntu/miniconda3/envs/UAV_CV/bin/python' -u '/home/ubuntu/文档/ZHJ_UAV/ultralytics/experiments/rtdetrv3_dense_supervision_5090/controller.py'
