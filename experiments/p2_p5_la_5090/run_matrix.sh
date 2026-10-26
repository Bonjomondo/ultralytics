#!/usr/bin/env bash
# Serial from-scratch matrix; uses a fresh project directory and stops on any failure.
set -euo pipefail
if (( $# < 4 || $# > 6 )); then
  echo "Usage: bash $0 DATA_YAML VAL_COCO_JSON VAL_IMAGES NEW_PROJECT [EPOCHS=150] [SEED=0]" >&2
  exit 2
fi
cd "$(dirname "$0")/../.."
data=$(realpath "$1")
annotations=$(realpath "$2")
images=$(realpath "$3")
project=$(realpath -m "$4")
epochs=${5:-150}
seed=${6:-0}
python_bin=${PYTHON:-python}
if [[ ! "$epochs" =~ ^[0-9]+$ || ! "$seed" =~ ^[0-9]+$ ]] || (( epochs < 150 )); then
  echo "The full matrix needs EPOCHS>=150 and SEED>=0; use train.py for smoke tests." >&2
  exit 2
fi
if [[ -e "$project" ]]; then
  echo "Project already exists; choose a new directory: $project" >&2
  exit 2
fi
mkdir -p "$project"
evaluations=()
checkpoints=()
for arm in c0 c2 la c2-la; do
  name="5090-${arm}-${epochs}e-s${seed}"
  "$python_bin" -m experiments.p2_p5_la_5090.train --variant "$arm" --data "$data" \
    --project "$project" --name "$name" --epochs "$epochs" --seed "$seed"
  "$python_bin" -m experiments.p2_p5_la_5090.evaluate --variant "$arm" --run "$project/$name" \
    --annotations "$annotations" --images "$images" --output "$project/$name-eval"
  evaluations+=("$project/$name-eval/summary.json")
  checkpoints+=("$project/$name/weights/best.pt")
done
"$python_bin" -m experiments.p2_p5_la_5090.benchmark --checkpoints "${checkpoints[@]}" \
  --output "$project/paired-timing" --rounds 10
"$python_bin" -m experiments.p2_p5_la_5090.report --evaluations "${evaluations[@]}" \
  --benchmark "$project/paired-timing/benchmark.json" --output "$project/report"
