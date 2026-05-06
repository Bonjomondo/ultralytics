#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash run_remdet_5090_ablation.sh --dry-run
#   bash run_remdet_5090_ablation.sh --smoke
#   EPOCHS=50 IMGSZ=640 BATCH=32 SEEDS=42 RUN_SUFFIX=quick50 FORCE_FRESH=1 bash run_remdet_5090_ablation.sh
#   EPOCHS=300 IMGSZ=960 BATCH=16 SEEDS=42,43,44 RUN_SUFFIX=final bash run_remdet_5090_ablation.sh
#
# Notes:
#   - Run from a Linux machine with the training environment activated.
#   - The script calls the local repo's Ultralytics CLI through PYTHON, not PATH's yolo binary.
#   - PROJECT_DIR is converted to an absolute path before being passed to Ultralytics.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

DATA="${DATA:-VisDrone-local.yaml}"
EPOCHS="${EPOCHS:-300}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
PROJECT_DIR="${PROJECT_DIR:-runs/remdet_ablation_5090}"
SEEDS="${SEEDS:-42,43,44}"
EXPERIMENTS="${EXPERIMENTS:-A0,A1,A2,A3,A4,A5}"
PYTHON="${PYTHON:-python}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
FORCE_FRESH="${FORCE_FRESH:-0}"
NO_RESUME="${NO_RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ENV_CHECK="${SKIP_ENV_CHECK:-0}"
CHECK_ENV_IN_DRY_RUN="${CHECK_ENV_IN_DRY_RUN:-0}"

OPTIMIZER="${OPTIMIZER:-AdamW}"
LR0="${LR0:-0.001}"
LRF="${LRF:-0.01}"
MOMENTUM="${MOMENTUM:-0.937}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-15}"
PATIENCE="${PATIENCE:-0}"

usage() {
    cat <<'EOF'
Usage:
  bash run_remdet_5090_ablation.sh [options]

Options:
  --data PATH              Dataset YAML. Default: VisDrone-local.yaml
  --epochs N               Training epochs. Default: 300
  --imgsz N                Image size. Default: 960
  --batch N                Batch size. Default: 16
  --workers N              DataLoader workers. Default: 8
  --device ID              CUDA device, e.g. 0 or 0,1. Default: 0
  --project-dir PATH       Output root. Default: runs/remdet_ablation_5090
  --seeds LIST             Seeds, comma or space separated. Default: 42,43,44
  --experiments LIST       Experiment IDs. Default: A0,A1,A2,A3,A4,A5
  --python PATH            Python executable. Default: python
  --run-suffix TEXT        Append suffix to each run name.
  --optimizer NAME         Optimizer. Default: AdamW
  --lr0 VALUE              Initial LR. Default: 0.001
  --lrf VALUE              Final LR fraction. Default: 0.01
  --force-fresh            Disable skip/resume. Adds fresh timestamp if no run suffix is set.
  --no-resume              Do not resume from existing last.pt.
  --dry-run                Print commands without training. Skips environment check by default.
  --skip-env-check         Skip local import and CUDA checks.
  --smoke                  Set epochs=1 imgsz=640 batch=2 seeds=42 run_suffix=smoke.
  -h, --help               Show this help.

Environment variables with matching uppercase names are also supported.
Examples:
  bash run_remdet_5090_ablation.sh --dry-run --epochs 1 --imgsz 640 --batch 2 --seeds 42
  bash run_remdet_5090_ablation.sh --smoke
  EPOCHS=300 IMGSZ=960 BATCH=16 SEEDS=42,43,44 RUN_SUFFIX=final bash run_remdet_5090_ablation.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)
            DATA="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --experiments)
            EXPERIMENTS="$2"
            shift 2
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --run-suffix)
            RUN_SUFFIX="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --lr0)
            LR0="$2"
            shift 2
            ;;
        --lrf)
            LRF="$2"
            shift 2
            ;;
        --force-fresh)
            FORCE_FRESH=1
            shift
            ;;
        --no-resume)
            NO_RESUME=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-env-check)
            SKIP_ENV_CHECK=1
            shift
            ;;
        --smoke)
            EPOCHS=1
            IMGSZ=640
            BATCH=2
            SEEDS=42
            RUN_SUFFIX="${RUN_SUFFIX:-smoke}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${DRY_RUN}" == "1" && "${CHECK_ENV_IN_DRY_RUN}" != "1" ]]; then
    SKIP_ENV_CHECK=1
fi

abs_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${REPO_ROOT}" "${path}"
    fi
}

PROJECT_DIR_ABS="$(abs_path "${PROJECT_DIR}")"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${PROJECT_DIR_ABS}/_batch_logs"
LOG_FILE="${LOG_DIR}/remdet_ablation_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

strip_ansi_and_cr() {
    sed -u -r 's/\x1B\[[0-9;]*[[:alpha:]]//g; s/\r//g'
}

log_line() {
    echo "$1" | tee -a "${LOG_FILE}"
}

normalize_list() {
    local raw="$1"
    raw="${raw//,/ }"
    # shellcheck disable=SC2086
    echo ${raw}
}

print_command() {
    local -a cmd=("$@")
    printf '%q ' "${cmd[@]}"
    printf '\n'
}

check_required_files() {
    if [[ ! -f "${DATA}" ]]; then
        log_line "[ERROR] Data config not found: ${DATA}"
        exit 1
    fi

    local def id slug model pretrained
    for def in "${SELECTED_EXPERIMENT_DEFS[@]}"; do
        IFS='|' read -r id slug model pretrained <<< "${def}"
        if [[ "${model}" == *.yaml && ! -f "${model}" ]]; then
            log_line "[ERROR] Model config not found: ${model}"
            exit 1
        fi
    done
}

check_local_ultralytics_import() {
    REPO_ROOT="${REPO_ROOT}" PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" - <<'PY'
import os
import sys
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"]).resolve()

try:
    import ultralytics
    from ultralytics.cfg import entrypoint  # noqa: F401
    from ultralytics.nn.modules import CED, GatedFFN, MECA  # noqa: F401
except Exception as exc:
    print(f"Ultralytics import check failed: {exc}")
    sys.exit(2)

pkg = Path(ultralytics.__file__).resolve()
if repo not in pkg.parents:
    print(f"Ultralytics import is not from this repo: {pkg}")
    print(f"Expected it under: {repo}")
    sys.exit(3)

print(f"Ultralytics import OK: {pkg}")
PY
}

check_torch_cuda_compat() {
    DEVICE_ARG="${DEVICE}" "${PYTHON}" - <<'PY'
import os
import sys
import torch

device = os.environ["DEVICE_ARG"].strip().lower()
if device in {"cpu", "mps"}:
    print(f"device={device}; CUDA check skipped.")
    sys.exit(0)

if not torch.cuda.is_available():
    print("CUDA is not available in current PyTorch runtime.")
    sys.exit(2)

first = device.split(",")[0].replace("cuda:", "")
index = int(first) if first.isdigit() else 0
if index >= torch.cuda.device_count():
    print(f"Requested CUDA device {index}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")
    sys.exit(2)

props = torch.cuda.get_device_properties(index)
arch = f"sm_{props.major}{props.minor}"
arch_list = set(torch.cuda.get_arch_list())
print(f"CUDA device {index}: {torch.cuda.get_device_name(index)} ({arch}); torch={torch.__version__}; cuda={torch.version.cuda}")

if arch not in arch_list:
    print(
        f"Incompatible PyTorch CUDA build for current GPU: need {arch}, "
        f"but torch supports {sorted(arch_list)}."
    )
    sys.exit(3)
PY
}

checkpoint_loadable() {
    local ckpt="$1"
    "${PYTHON}" - "${ckpt}" <<'PY' >/dev/null 2>&1
import sys
import torch

torch.load(sys.argv[1], map_location="cpu", weights_only=False)
PY
}

run_complete() {
    local run_dir="$1"
    local expected_epochs="$2"
    local best="${run_dir}/weights/best.pt"
    local csv="${run_dir}/results.csv"

    [[ -s "${best}" && -s "${csv}" ]] || return 1

    "${PYTHON}" - "${csv}" "${expected_epochs}" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
expected_epochs = int(sys.argv[2])

try:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(1)
    epoch_key = next((k for k in rows[0].keys() if k and k.strip() == "epoch"), None)
    if epoch_key is None:
        sys.exit(1)
    last_epoch = int(float(rows[-1][epoch_key]))
except Exception:
    sys.exit(1)

sys.exit(0 if last_epoch >= expected_epochs - 1 else 1)
PY
}

make_run_name() {
    local id="$1"
    local slug="$2"
    local seed="$3"
    local name="${id}_${slug}_img${IMGSZ}_s${seed}"

    if [[ -n "${RUN_SUFFIX}" ]]; then
        name="${name}_${RUN_SUFFIX}"
    elif [[ "${FORCE_FRESH}" == "1" ]]; then
        name="${name}_fresh_${TIMESTAMP}"
    fi

    printf '%s\n' "${name}"
}

run_yolo() {
    local -a cmd=("$@")
    local rendered
    rendered="$(print_command "${cmd[@]}")"
    log_line "[CMD] ${rendered}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}" 2>&1 | tee >(strip_ansi_and_cr >> "${LOG_FILE}")
    local status=${PIPESTATUS[0]}

    return "${status}"
}

ALL_EXPERIMENT_DEFS=(
    "A0|baseline|yolo11n.pt|"
    "A1|ced|ultralytics/cfg/models/11/yolo11n-remdet-ced.yaml|yolo11n.pt"
    "A2|gffn|ultralytics/cfg/models/11/yolo11n-remdet-gffn.yaml|yolo11n.pt"
    "A3|ced_gffn|ultralytics/cfg/models/11/yolo11n-remdet-ced-gffn.yaml|yolo11n.pt"
    "A4|ced_gffn_p2|ultralytics/cfg/models/11/yolo11n-remdet-ced-gffn-p2.yaml|yolo11n.pt"
    "A5|ced_gffn_p2_meca|ultralytics/cfg/models/11/yolo11n-remdet-ced-gffn-p2-meca.yaml|yolo11n.pt"
)

read -r -a SEED_ARRAY <<< "$(normalize_list "${SEEDS}")"
read -r -a REQUESTED_EXPERIMENTS <<< "$(normalize_list "${EXPERIMENTS}")"

declare -A REQUESTED_MAP=()
for id in "${REQUESTED_EXPERIMENTS[@]}"; do
    REQUESTED_MAP["${id}"]=0
done

SELECTED_EXPERIMENT_DEFS=()
for def in "${ALL_EXPERIMENT_DEFS[@]}"; do
    IFS='|' read -r id slug model pretrained <<< "${def}"
    if [[ -n "${REQUESTED_MAP[${id}]+x}" ]]; then
        REQUESTED_MAP["${id}"]=1
        SELECTED_EXPERIMENT_DEFS+=("${def}")
    fi
done

for id in "${!REQUESTED_MAP[@]}"; do
    if [[ "${REQUESTED_MAP[${id}]}" != "1" ]]; then
        log_line "[ERROR] Unknown experiment id: ${id}"
        exit 2
    fi
done

if [[ "${#SEED_ARRAY[@]}" -eq 0 || "${#SELECTED_EXPERIMENT_DEFS[@]}" -eq 0 ]]; then
    log_line "[ERROR] Empty seed list or experiment list."
    exit 2
fi

check_required_files

if [[ "${SKIP_ENV_CHECK}" != "1" ]]; then
    if ! check_local_ultralytics_import 2>&1 | tee >(strip_ansi_and_cr >> "${LOG_FILE}"); then
        log_line "[ERROR] Local Ultralytics import check failed. Activate the correct environment and run: pip install -e ."
        exit 1
    fi

    if ! check_torch_cuda_compat 2>&1 | tee >(strip_ansi_and_cr >> "${LOG_FILE}"); then
        log_line "[ERROR] PyTorch/CUDA runtime is incompatible with device=${DEVICE}."
        exit 1
    fi
fi

log_line "===== RemDet YOLO11n ablation batch started at $(date '+%F %T') ====="
log_line "Repo: ${REPO_ROOT}"
log_line "Log: ${LOG_FILE}"
log_line "Python: ${PYTHON}"
log_line "Project dir: ${PROJECT_DIR_ABS}"
log_line "Data=${DATA}, Epochs=${EPOCHS}, ImgSz=${IMGSZ}, Batch=${BATCH}, Workers=${WORKERS}, Device=${DEVICE}"
log_line "Seeds=${SEED_ARRAY[*]}, Experiments=${REQUESTED_EXPERIMENTS[*]}"
log_line "Optimizer=${OPTIMIZER}, lr0=${LR0}, lrf=${LRF}, momentum=${MOMENTUM}, weight_decay=${WEIGHT_DECAY}, patience=${PATIENCE}"
log_line "ForceFresh=${FORCE_FRESH}, NoResume=${NO_RESUME}, DryRun=${DRY_RUN}, SkipEnvCheck=${SKIP_ENV_CHECK}, RunSuffix=${RUN_SUFFIX:-<none>}"

TOTAL=$(( ${#SEED_ARRAY[@]} * ${#SELECTED_EXPERIMENT_DEFS[@]} ))
IDX=0

for seed in "${SEED_ARRAY[@]}"; do
    for def in "${SELECTED_EXPERIMENT_DEFS[@]}"; do
        IDX=$((IDX + 1))
        IFS='|' read -r id slug model pretrained <<< "${def}"

        name="$(make_run_name "${id}" "${slug}" "${seed}")"
        run_dir="${PROJECT_DIR_ABS}/${name}"
        last="${run_dir}/weights/last.pt"

        if [[ "${DRY_RUN}" != "1" && "${FORCE_FRESH}" != "1" ]] && run_complete "${run_dir}" "${EPOCHS}"; then
            log_line "[${IDX}/${TOTAL}] [SKIP] ${name} is already complete."
            continue
        fi

        if [[ "${DRY_RUN}" != "1" && "${FORCE_FRESH}" != "1" && "${NO_RESUME}" != "1" && -s "${last}" ]] && checkpoint_loadable "${last}"; then
            log_line "[${IDX}/${TOTAL}] [RESUME] ${name} from ${last}"
            cmd=(
                "${PYTHON}"
                "-c"
                "from ultralytics.cfg import entrypoint; entrypoint()"
                "detect"
                "train"
                "resume=True"
                "model=${last}"
                "imgsz=${IMGSZ}"
                "batch=${BATCH}"
                "workers=${WORKERS}"
                "device=${DEVICE}"
                "patience=${PATIENCE}"
                "val=True"
            )
        else
            if [[ "${FORCE_FRESH}" == "1" ]]; then
                log_line "[${IDX}/${TOTAL}] [FRESH] ${name}"
            else
                log_line "[${IDX}/${TOTAL}] [START] ${name}"
            fi

            cmd=(
                "${PYTHON}"
                "-c"
                "from ultralytics.cfg import entrypoint; entrypoint()"
                "detect"
                "train"
                "data=${DATA}"
                "model=${model}"
                "imgsz=${IMGSZ}"
                "epochs=${EPOCHS}"
                "batch=${BATCH}"
                "optimizer=${OPTIMIZER}"
                "lr0=${LR0}"
                "lrf=${LRF}"
                "momentum=${MOMENTUM}"
                "weight_decay=${WEIGHT_DECAY}"
                "close_mosaic=${CLOSE_MOSAIC}"
                "cos_lr=True"
                "amp=True"
                "workers=${WORKERS}"
                "device=${DEVICE}"
                "seed=${seed}"
                "project=${PROJECT_DIR_ABS}"
                "name=${name}"
                "exist_ok=True"
                "plots=True"
                "patience=${PATIENCE}"
                "save=True"
                "val=True"
            )

            if [[ -n "${pretrained}" ]]; then
                cmd+=("pretrained=${pretrained}")
            fi
        fi

        set +e
        run_yolo "${cmd[@]}"
        status=$?
        set -e

        if [[ "${status}" -ne 0 ]]; then
            log_line "[${IDX}/${TOTAL}] [FAIL] ${name} exit code: ${status}"
            log_line "===== Batch stopped at $(date '+%F %T') ====="
            exit "${status}"
        fi

        log_line "[${IDX}/${TOTAL}] [DONE] ${name} at $(date '+%F %T')"
    done
done

log_line "===== RemDet YOLO11n ablation batch finished at $(date '+%F %T') ====="
