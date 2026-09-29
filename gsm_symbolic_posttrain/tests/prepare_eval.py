"""Select a reproducible subset from pinned GSM-Symbolic source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

VARIANTS = ("main", "p1", "p2")
EXPECTED_FULL_COUNTS = {"main": 5000, "p1": 5000, "p2": 2500}
DATA_REVISION = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
DATA_URL = f"https://hf-mirror.com/datasets/apple/GSM-Symbolic/resolve/{DATA_REVISION}/{{variant}}/test.jsonl"
EXPECTED_SHA256 = {
    "main": "945055837b232bd9b56827b1e3a5aa019075f29e8ba4669d9a9df8b4439a291f",
    "p1": "5478fe81b9cdc5d9447514ca68b0dfbb6cb01b7227fefadf143469b103b592ac",
    "p2": "00496e6c36bd746d56b28cffcd9385f78f0568a5fe1543650ddf9b55d398251c",
}


def select_rows(raw_text: str, variant: str, instances_per_template: int) -> list[dict]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if not 1 <= instances_per_template <= 50:
        raise ValueError("instances_per_template must be between 1 and 50")

    records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    if len(records) != EXPECTED_FULL_COUNTS[variant]:
        raise ValueError(f"{variant}: expected {EXPECTED_FULL_COUNTS[variant]} source records, got {len(records)}")

    selected = []
    seen = set()
    for row in records:
        template_id, instance = row["id"], row["instance"]
        if template_id >= 50 or instance >= instances_per_template:
            continue
        key = (template_id, instance)
        if key in seen:
            raise ValueError(f"duplicate {variant} key {key}")
        seen.add(key)
        answer = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        try:
            float(answer)
        except ValueError as exc:
            raise ValueError(f"invalid gold for {variant} {key}: {answer}") from exc
        selected.append(
            {"variant": variant, "id": template_id, "instance": instance,
             "question": row["question"], "gold": answer}
        )

    expected_keys = {(i, j) for i in range(50) for j in range(instances_per_template)}
    if seen != expected_keys:
        raise ValueError(f"{variant}: missing {len(expected_keys - seen)} selected keys")
    return sorted(selected, key=lambda row: (row["id"], row["instance"]))


def prepare(output: Path, instances_per_template: int) -> dict[str, str]:
    all_rows = []
    source_hashes = {}
    for variant in VARIANTS:
        # The verifier image includes these pinned files so evaluation works offline.
        bundled_source = Path(__file__).parent / "data" / f"{variant}.jsonl"
        if bundled_source.is_file():
            raw_bytes = bundled_source.read_bytes()
        else:
            with urlopen(DATA_URL.format(variant=variant), timeout=120) as response:
                raw_bytes = response.read()
        source_hashes[variant] = hashlib.sha256(raw_bytes).hexdigest()
        if source_hashes[variant] != EXPECTED_SHA256[variant]:
            raise ValueError(f"{variant}: source SHA256 changed unexpectedly")
        all_rows.extend(select_rows(raw_bytes.decode("utf-8"), variant, instances_per_template))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return source_hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/gsm_symbolic_eval.jsonl"))
    parser.add_argument("--instances-per-template", type=int, default=5)
    args = parser.parse_args()
    hashes = prepare(args.output, args.instances_per_template)
    print(json.dumps({"output": str(args.output), "source_sha256": hashes}, indent=2))
