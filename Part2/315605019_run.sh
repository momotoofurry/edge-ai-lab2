#!/bin/bash
# stdout carries only the final results; everything else goes to logs/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

python -m pip install -r requirements.txt > logs/pip.log 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# case  rate  wbits
CASES=(
  "1 0.75 4"
  "2 0.81 3"
  "3 0.50 8"
  "4 0.85 3"
)

for spec in "${CASES[@]}"; do
  read -r c rate wbits <<< "$spec"
  out=$(python run_eval.py --case "$c" --rate "$rate" --wbits "$wbits" 2> "logs/case${c}.log")
  ppl=$(sed -n 's/^ppl: //p' <<< "$out")
  ratio=$(sed -n 's/^comp_rate: //p' <<< "$out")
  echo "Case $c"
  echo "Compression Ratio: $ratio"
  printf 'PPL: %.2f\n' "$ppl"
done
