#!/usr/bin/env bash
set -euo pipefail

python /solution/train_reference.py \
  --base-model /models/base \
  --train-data /data/gsm8k_train.jsonl \
  --output /logs/artifacts/adapter
