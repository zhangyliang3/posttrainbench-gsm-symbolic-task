#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier
COUNT="${EVAL_INSTANCES_PER_TEMPLATE:-5}"

python /tests/prepare_eval.py \
  --output /tmp/gsm_symbolic_eval.jsonl \
  --instances-per-template "$COUNT"

python /tests/evaluate.py \
  --eval-data /tmp/gsm_symbolic_eval.jsonl \
  --base-model /models/base \
  --adapter /logs/artifacts/adapter \
  --train-data /data/gsm8k_train.jsonl \
  --dev-data /data/gsm8k_dev.jsonl \
  --batch-size "${EVAL_BATCH_SIZE:-8}" \
  --require-submission-evidence "${REQUIRE_SUBMISSION_EVIDENCE:-1}" \
  --reward-json /logs/verifier/reward.json \
  --overlap-report /logs/verifier/overlap_report.json \
  --responses-jsonl /logs/verifier/responses.jsonl
