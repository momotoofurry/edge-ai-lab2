#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

python -m pip install -r requirements.txt > logs/pip.log 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# thresholds.json was derived once, offline, on Llama-3.1-8B (see README)
python run_ptq.py --mode adaptquant --thresholds thresholds.json 2> logs/adaptquant.log
