# Separate verifier image. Harbor copies the agent's adapter via artifacts.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_ENDPOINT=https://hf-mirror.com

RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    peft==0.13.2 \
    accelerate==1.1.1 \
    datasets==3.1.0 \
    huggingface-hub==0.26.2 \
    requests==2.32.3

RUN mkdir -p /models/base /data /tests /logs/verifier /logs/artifacts

RUN --mount=type=cache,id=novix-hf-cache,target=/root/.cache/huggingface \
    --mount=type=cache,id=novix-model-weight,target=/model-cache python - <<'PY'
import hashlib
import shutil
import time
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

snapshot_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    revision="4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
    ignore_patterns=["model.safetensors"],
    max_workers=1,
)
shutil.copytree(snapshot_path, "/models/base", dirs_exist_ok=True)

url = "https://modelscope.cn/models/Qwen/Qwen2.5-Math-1.5B/resolve/master/model.safetensors"
weight = Path("/model-cache/model.safetensors")
expected_size = 3087467144
expected_sha256 = "f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b"
for attempt in range(10):
    offset = weight.stat().st_size if weight.exists() else 0
    if offset == expected_size:
        break
    try:
        with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                          stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            if offset and response.status_code != 206:
                offset = 0
            with weight.open("ab" if offset else "wb") as output:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
        print(f"model weight: {weight.stat().st_size}/{expected_size} bytes", flush=True)
    except requests.RequestException as exc:
        print(f"download retry {attempt + 1}: {exc}", flush=True)
        time.sleep(min(5 * (attempt + 1), 30))
if not weight.exists() or weight.stat().st_size != expected_size:
    raise RuntimeError("model weight download incomplete")
digest = hashlib.sha256()
with weight.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_sha256:
    raise RuntimeError("model weight SHA256 mismatch")
shutil.copy2(weight, "/models/base/model.safetensors")
PY

RUN --mount=type=cache,id=novix-hf-cache,target=/root/.cache/huggingface python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

rows = load_dataset(
    "openai/gsm8k", "main", split="train",
    revision="740312add88f781978c0658806c59bc2815b9866",
)
assert len(rows) == 7473, len(rows)
for filename, start, stop in (("gsm8k_train.jsonl", 0, 7223), ("gsm8k_dev.jsonl", 7223, 7473)):
    with Path("/data", filename).open("w", encoding="utf-8") as f:
        for original_index, row in enumerate(rows.select(range(start, stop)), start=start):
            f.write(json.dumps({"source_index": original_index, "question": row["question"], "answer": row["answer"]}, ensure_ascii=False) + "\n")
PY

COPY . /tests
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
WORKDIR /workspace
