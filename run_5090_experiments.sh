#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash run_5090_experiments.sh
#   RUN_AGGRESSIVE_BS12=0 bash run_5090_experiments.sh
#   RUN_AEROTINY=0 bash run_5090_experiments.sh
#   FORCE_FRESH=1 bash run_5090_experiments.sh

PROJECT="${PROJECT:-runs/das_yolo}"
# Ultralytics detect training saves under runs/detect/<project>/<name>
SAVE_ROOT="runs/detect/${PROJECT}"
LOG_DIR="${PROJECT}/_batch_logs"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/train_batch_${TIMESTAMP}.log"

# 1 = run optional 1280_bs12 experiment, 0 = skip
RUN_AGGRESSIVE_BS12="${RUN_AGGRESSIVE_BS12:-1}"
# 1 = include AeroTiny-ONX experiments, 0 = skip
RUN_AEROTINY="${RUN_AEROTINY:-1}"
# 1 = ignore DONE markers and resume checkpoints, force fresh runs
FORCE_FRESH="${FORCE_FRESH:-0}"

# Shared training knobs
DATA_CFG="${DATA_CFG:-VisDrone.yaml}"
EPOCHS="${EPOCHS:-150}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
AEROTINY_MODEL="${AEROTINY_MODEL:-yolo11s-aerotiny-onx.yaml}"

# Batch size knobs (override with env vars if needed)
BASELINE_BATCH="${BASELINE_BATCH:-16}"
THRF_1024_BATCH="${THRF_1024_BATCH:-4}"
THRF_1280_BATCH="${THRF_1280_BATCH:-8}"
THRF_1280_BS12_BATCH="${THRF_1280_BS12_BATCH:-12}"
AEROTINY_1024_BATCH="${AEROTINY_1024_BATCH:-6}"
AEROTINY_1280_BATCH="${AEROTINY_1280_BATCH:-4}"
AEROTINY_1280_BS6_BATCH="${AEROTINY_1280_BS6_BATCH:-6}"

mkdir -p "${LOG_DIR}"

strip_ansi_and_cr() {
    sed -u -r 's/\x1B\[[0-9;]*[[:alpha:]]//g; s/\r//g'
}

log_line() {
    echo "$1" | tee -a "${LOG_FILE}"
}

has_done_marker() {
    local name="$1"
    shopt -s nullglob
    local logs=("${LOG_DIR}"/train_batch_*.log)
    shopt -u nullglob

    [[ ${#logs[@]} -gt 0 ]] || return 1
    grep -hE "\[DONE\] ${name}( |$)" "${logs[@]}" >/dev/null 2>&1
}

is_completed_experiment() {
    local name="$1"
    shopt -s nullglob
    local exp_dirs=("${SAVE_ROOT}/${name}"*)
    shopt -u nullglob

    has_done_marker "${name}" || return 1

    local exp_dir
    for exp_dir in "${exp_dirs[@]}"; do
        [[ -d "${exp_dir}" ]] || continue
        if [[ -s "${exp_dir}/weights/best.pt" || -s "${exp_dir}/weights/last.pt" ]]; then
            return 0
        fi
    done

    return 1
}

find_resume_checkpoint() {
    local name="$1"
    shopt -s nullglob
    local exp_dirs=("${SAVE_ROOT}/${name}"*)
    shopt -u nullglob

    [[ ${#exp_dirs[@]} -gt 0 ]] || return 1

    local sorted_dirs=()
    mapfile -t sorted_dirs < <(printf '%s\n' "${exp_dirs[@]}" | sort -V)

    local exp_dir
    for exp_dir in "${sorted_dirs[@]}"; do
        [[ -d "${exp_dir}" ]] || continue
        if [[ -s "${exp_dir}/weights/last.pt" ]]; then
            echo "${exp_dir}/weights/last.pt"
        fi
    done | tail -n 1

    if [[ ${PIPESTATUS[1]} -ne 0 ]]; then
        return 1
    fi
}

resume_checkpoint_loadable() {
    local ckpt="$1"
    CKPT_PATH="${ckpt}" python - <<'PY' >/dev/null 2>&1
import os
import torch

ckpt = os.environ["CKPT_PATH"]
torch.load(ckpt, map_location="cpu")
PY
}

check_torch_cuda_compat() {
    DEVICE_ARG="${DEVICE}" python - <<'PY'
import os
import sys
import torch

device = os.environ["DEVICE_ARG"].strip().lower()
if device in {"cpu", "mps"}:
    sys.exit(0)

if not torch.cuda.is_available():
    print("CUDA is not available in current PyTorch runtime.")
    sys.exit(2)

props = torch.cuda.get_device_properties(0)
arch = f"sm_{props.major}{props.minor}"
arch_list = set(torch.cuda.get_arch_list())

if arch not in arch_list:
    print(
        f"Incompatible PyTorch CUDA build for current GPU: need {arch}, "
        f"but torch supports {sorted(arch_list)}. "
        f"torch={torch.__version__}, cuda={torch.version.cuda}."
    )
    sys.exit(3)
PY
}

if ! command -v yolo >/dev/null 2>&1; then
    log_line "[ERROR] 'yolo' command not found. Activate your environment first."
    exit 1
fi

if ! check_torch_cuda_compat; then
    log_line "[ERROR] PyTorch/CUDA runtime is incompatible with configured device=${DEVICE}."
    log_line "[ERROR] Please upgrade to a GPU-compatible PyTorch build before running batch training."
    exit 1
fi

if [[ "${RUN_AEROTINY}" == "1" && ! -f "ultralytics/cfg/models/11/yolo11-aerotiny-onx.yaml" ]]; then
    log_line "[ERROR] Missing ultralytics/cfg/models/11/yolo11-aerotiny-onx.yaml"
    log_line "[ERROR] Please ensure AeroTiny-ONX YAML exists before running batch experiments."
    exit 1
fi

log_line "===== Batch training started at $(date '+%F %T') ====="
log_line "Project: ${PROJECT}"
log_line "Save root: ${SAVE_ROOT}"
log_line "Log: ${LOG_FILE}"
log_line "RUN_AGGRESSIVE_BS12=${RUN_AGGRESSIVE_BS12}"
log_line "RUN_AEROTINY=${RUN_AEROTINY}, AEROTINY_MODEL=${AEROTINY_MODEL}"
log_line "FORCE_FRESH=${FORCE_FRESH}"
log_line "DATA_CFG=${DATA_CFG}, EPOCHS=${EPOCHS}, WORKERS=${WORKERS}, DEVICE=${DEVICE}"
log_line "BASELINE_BATCH=${BASELINE_BATCH}, THRF_1024_BATCH=${THRF_1024_BATCH}, THRF_1280_BATCH=${THRF_1280_BATCH}, THRF_1280_BS12_BATCH=${THRF_1280_BS12_BATCH}"
log_line "AEROTINY_1024_BATCH=${AEROTINY_1024_BATCH}, AEROTINY_1280_BATCH=${AEROTINY_1280_BATCH}, AEROTINY_1280_BS6_BATCH=${AEROTINY_1280_BS6_BATCH}"

declare -a EXPERIMENTS=(
"final_baseline_1024|yolo train model=yolo11s.yaml pretrained=yolo11s.pt data=${DATA_CFG} imgsz=1024 epochs=${EPOCHS} batch=${BASELINE_BATCH} optimizer=AdamW lr0=0.001 lrf=0.01 weight_decay=0.0005 close_mosaic=15 cos_lr=True amp=True workers=${WORKERS} device=${DEVICE} exist_ok=True project=${PROJECT} name=final_baseline_1024"
"final_thrf_nwd_1024|yolo train model=yolo11s-thrf-p2.yaml pretrained=yolo11s.pt data=${DATA_CFG} imgsz=1024 epochs=${EPOCHS} batch=${THRF_1024_BATCH} optimizer=AdamW lr0=0.001 lrf=0.01 weight_decay=0.0005 close_mosaic=15 cos_lr=True amp=True workers=${WORKERS} nwd=0.3 nwd_tau=12.8 device=${DEVICE} exist_ok=True project=${PROJECT} name=final_thrf_nwd_1024"
# "final_thrf_nwd_1280|yolo train model=yolo11s-thrf-p2.yaml pretrained=yolo11s.pt data=${DATA_CFG} imgsz=1280 epochs=${EPOCHS} batch=${THRF_1280_BATCH} optimizer=AdamW lr0=0.001 lrf=0.01 weight_decay=0.0005 close_mosaic=15 cos_lr=True amp=True workers=${WORKERS} nwd=0.3 nwd_tau=12.8 device=${DEVICE} exist_ok=True project=${PROJECT} name=final_thrf_nwd_1280"
)

if [[ "${RUN_AEROTINY}" == "1" ]]; then
    EXPERIMENTS+=(
"final_aerotiny_onx_nwd_1024|yolo train model=${AEROTINY_MODEL} pretrained=yolo11s.pt data=${DATA_CFG} imgsz=1024 epochs=${EPOCHS} batch=${AEROTINY_1024_BATCH} optimizer=AdamW lr0=0.001 lrf=0.01 weight_decay=0.0005 close_mosaic=15 cos_lr=True amp=True workers=${WORKERS} nwd=0.3 nwd_tau=12.8 device=${DEVICE} exist_ok=True project=${PROJECT} name=final_aerotiny_onx_nwd_1024"
"final_aerotiny_onx_nwd_1280|yolo train model=${AEROTINY_MODEL} pretrained=yolo11s.pt data=${DATA_CFG} imgsz=1280 epochs=${EPOCHS} batch=${AEROTINY_1280_BATCH} optimizer=AdamW lr0=0.001 lrf=0.01 weight_decay=0.0005 close_mosaic=15 cos_lr=True amp=True workers=${WORKERS} nwd=0.3 nwd_tau=12.8 device=${DEVICE} exist_ok=True project=${PROJECT} name=final_aerotiny_onx_nwd_1280"
)
fi

if [[ "${RUN_AGGRESSIVE_BS12}" == "1" ]]; then
    EXPERIMENTS+=(
"final_thrf_nwd_1280_bs12|yolo train model=yolo11s-thrf-p2.yaml pretrained=yolo11s.pt data=${DATA_CFG} imgsz=1280 epochs=${EPOCHS} batch=${THRF_1280_BS12_BATCH} optimizer=AdamW lr0=0.001 lrf=0.01 weight_decay=0.0005 close_mosaic=15 cos_lr=True amp=True workers=${WORKERS} nwd=0.3 nwd_tau=12.8 device=${DEVICE} exist_ok=True project=${PROJECT} name=final_thrf_nwd_1280_bs12"
)
    if [[ "${RUN_AEROTINY}" == "1" ]]; then
        EXPERIMENTS+=(
"final_aerotiny_onx_nwd_1280_bs6|yolo train model=${AEROTINY_MODEL} pretrained=yolo11s.pt data=${DATA_CFG} imgsz=1280 epochs=${EPOCHS} batch=${AEROTINY_1280_BS6_BATCH} optimizer=AdamW lr0=0.001 lrf=0.01 weight_decay=0.0005 close_mosaic=15 cos_lr=True amp=True workers=${WORKERS} nwd=0.3 nwd_tau=12.8 device=${DEVICE} exist_ok=True project=${PROJECT} name=final_aerotiny_onx_nwd_1280_bs6"
)
    fi
fi

TOTAL="${#EXPERIMENTS[@]}"
IDX=0

for item in "${EXPERIMENTS[@]}"; do
    IDX=$((IDX + 1))
    IFS='|' read -r NAME CMD <<< "${item}"

    EXP_DIR="${SAVE_ROOT}/${NAME}"

    if [[ "${FORCE_FRESH}" != "1" ]] && is_completed_experiment "${NAME}"; then
        log_line "[$IDX/$TOTAL] [SKIP] ${NAME} completed (checkpoint + DONE marker found across ${EXP_DIR}*)"
        continue
    fi

    ORIGINAL_CMD="${CMD}"
    RUN_CMD="${ORIGINAL_CMD}"
    RESUME_CKPT=""
    if [[ "${FORCE_FRESH}" != "1" ]]; then
        RESUME_CKPT="$(find_resume_checkpoint "${NAME}" || true)"
    fi
    if [[ -n "${RESUME_CKPT}" ]]; then
        if resume_checkpoint_loadable "${RESUME_CKPT}"; then
            # Resume interrupted run from checkpoint instead of creating a new indexed folder.
            RUN_CMD="yolo train resume model=${RESUME_CKPT}"
            log_line "[$IDX/$TOTAL] [RESUME] ${NAME} from ${RESUME_CKPT}"
        else
            log_line "[$IDX/$TOTAL] [WARN] resume checkpoint is not loadable, fallback to fresh train command"
            RUN_CMD="${ORIGINAL_CMD}"
        fi
    elif [[ "${FORCE_FRESH}" == "1" ]]; then
        log_line "[$IDX/$TOTAL] [FRESH] ${NAME} forced fresh run (skip/resume disabled)"
    fi

    log_line "[$IDX/$TOTAL] [START] ${NAME} at $(date '+%F %T')"
    log_line "[$IDX/$TOTAL] [CMD] ${RUN_CMD}"

    # Run one experiment at a time; stop the whole batch on first failure.
    set +e
    eval "${RUN_CMD}" 2>&1 | tee >(strip_ansi_and_cr >> "${LOG_FILE}")
    STATUS=${PIPESTATUS[0]}
    set -e

    if [[ ${STATUS} -ne 0 ]]; then
        log_line "[$IDX/$TOTAL] [FAIL] ${NAME} exit code: ${STATUS}"
        log_line "===== Batch stopped at $(date '+%F %T') ====="
        exit ${STATUS}
    fi

    log_line "[$IDX/$TOTAL] [DONE] ${NAME} at $(date '+%F %T')"
done

log_line "===== Batch finished successfully at $(date '+%F %T') ====="