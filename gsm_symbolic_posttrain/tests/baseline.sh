#!/usr/bin/env bash
# Run manually in the verifier environment to measure the untouched base model.
set -euo pipefail

mkdir -p /logs/verifier
COUNT="${EVAL_INSTANCES_PER_TEMPLATE:-5}"
python /tests/prepare_eval.py \
  --output /tmp/gsm_symbolic_eval.jsonl \
  --instances-per-template "$COUNT"
python /tests/evaluate.py \
  --eval-data /tmp/gsm_symbolic_eval.jsonl \
  --base-model /models/base \
  --train-data /data/gsm8k_train.jsonl \
  --reward-json /logs/verifier/baseline.json \
  --responses-jsonl /logs/verifier/baseline_responses.jsonl
