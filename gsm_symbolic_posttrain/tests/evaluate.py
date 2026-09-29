"""Deterministic numeric-answer verifier for the Harbor post-training task."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

VARIANTS = ("main", "p1", "p2")
BASE_MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
BASE_MODEL_REVISION = "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
BASE_MODEL_WEIGHT_SHA256 = "f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b"
FINAL_RE = re.compile(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)")


def extract_final(text: str) -> Decimal | None:
    matches = FINAL_RE.findall(text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def is_correct(response: str, gold: str) -> bool:
    predicted = extract_final(response)
    if predicted is None:
        return False
    try:
        return abs(predicted - Decimal(gold)) <= Decimal("0.000001")
    except InvalidOperation:
        return False


def aggregate(rows: list[dict], responses: list[str]) -> dict[str, float]:
    if len(rows) != len(responses):
        raise ValueError("one response is required for each eval row")
    correct = {variant: 0 for variant in VARIANTS}
    total = {variant: 0 for variant in VARIANTS}
    for row, response in zip(rows, responses):
        variant = row["variant"]
        if variant not in total:
            raise ValueError(f"unknown variant: {variant}")
        total[variant] += 1
        correct[variant] += int(is_correct(response, row["gold"]))
    if len(set(total.values())) != 1 or not all(total.values()):
        raise ValueError(f"expected nonempty and balanced variants, got {total}")
    accuracies = {variant: correct[variant] / total[variant] for variant in VARIANTS}
    return {
        "score": sum(accuracies.values()) / 3,
        "acc_main": accuracies["main"],
        "acc_p1": accuracies["p1"],
        "acc_p2": accuracies["p2"],
        "n_main": float(total["main"]),
        "n_p1": float(total["p1"]),
        "n_p2": float(total["p2"]),
    }


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("empty evaluation file")
    return rows


def load_shots(path: Path, count: int = 4) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(next(handle)) for _ in range(count)]


def make_prompt(question: str, shots: list[dict]) -> str:
    parts = ["Solve each math word problem step by step. End each answer with #### followed by only the final number.\n"]
    for row in shots:
        parts.append(f"Q: {row['question']}\nA: {row['answer'].strip()}\n")
    parts.append(f"Q: {question}\nA:")
    return "\n".join(parts)


class InvalidAdapterSubmission(Exception):
    """The submitted adapter is missing, malformed, or incompatible."""


def invalid_reward(rows: list[dict]) -> dict[str, float]:
    totals = {variant: sum(row.get("variant") == variant for row in rows) for variant in VARIANTS}
    return {
        "score": 0.0,
        "acc_main": 0.0,
        "acc_p1": 0.0,
        "acc_p2": 0.0,
        "n_main": float(totals["main"]),
        "n_p1": float(totals["p1"]),
        "n_p2": float(totals["p2"]),
        "valid_submission": 0.0,
    }


def write_submission_status(path: Path, valid: bool, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"valid_submission": valid, "reason": reason[:1000]}, indent=2) + "\n",
        encoding="utf-8",
    )


def submission_bundle_error(adapter_path: Path) -> str | None:
    if not adapter_path.is_dir():
        return f"missing adapter directory: {adapter_path}"
    if not (adapter_path / "adapter_config.json").is_file():
        return "missing adapter_config.json"
    if not any((adapter_path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        return "missing adapter weights (adapter_model.safetensors or adapter_model.bin)"
    root = adapter_path.parent
    for name in ("run_summary.json", "train_script.py", "train_data.jsonl"):
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            return f"missing or empty submission evidence: {name}"
    try:
        summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
        required = {
            "base_model", "base_revision", "training_data_source", "training_rows",
            "training_method", "seed", "training_steps", "training_elapsed_seconds",
            "training_data_sha256", "training_script_sha256", "development_metrics",
        }
        missing = sorted(required - summary.keys())
        if missing:
            return f"run_summary.json missing fields: {', '.join(missing)}"
        if not isinstance(summary["development_metrics"], dict):
            return "run_summary.json development_metrics must be an object"
        with (root / "train_data.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows or any(not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str) for row in rows):
            return "train_data.jsonl must contain nonempty question and answer strings"
        if int(summary["training_rows"]) != len(rows):
            return f"run_summary training_rows={summary['training_rows']} does not match train_data.jsonl row count={len(rows)}"
        for field, filename in (("training_data_sha256", "train_data.jsonl"), ("training_script_sha256", "train_script.py")):
            actual_hash = hashlib.sha256((root / filename).read_bytes()).hexdigest()
            if summary[field] != actual_hash:
                return f"run_summary {field} does not match {filename}"
    except Exception as exc:
        return f"invalid submission evidence: {type(exc).__name__}: {exc}"
    return None


def _shingles(text: str, size: int = 5) -> set[bytes]:
    tokens = re.findall(r"[a-z]+|\d[\d,]*(?:\.\d+)?", text.lower())
    tokens = ["<number>" if token[0].isdigit() else token for token in tokens]
    if len(tokens) < size:
        tokens = [" ".join(tokens)] if tokens else []
        size = 1
    return {
        hashlib.blake2b(" ".join(tokens[i:i + size]).encode(), digest_size=8).digest()
        for i in range(len(tokens) - size + 1)
    }


def find_near_duplicates(
    training_questions: list[str], reference_rows: list[dict], threshold: float = 0.75,
) -> list[dict[str, float | str]]:
    """Find exact/near duplicates via numeric-normalized five-token shingle Jaccard."""
    train_sets = [_shingles(question) for question in training_questions]
    index: dict[bytes, list[int]] = defaultdict(list)
    for train_index, shingles in enumerate(train_sets):
        for shingle in shingles:
            index[shingle].append(train_index)

    matches = []
    for row in reference_rows:
        candidate = _shingles(str(row.get("question", "")))
        counts: dict[int, int] = defaultdict(int)
        for shingle in candidate:
            for train_index in index.get(shingle, ()):
                counts[train_index] += 1
        for train_index, intersection in counts.items():
            union = len(candidate) + len(train_sets[train_index]) - intersection
            similarity = intersection / union if union else 1.0
            if similarity >= threshold:
                matches.append({
                    "reference_variant": str(row.get("variant", "development")),
                    "reference_id": str(row.get("id", row.get("source_index", "unknown"))),
                    "training_row": str(train_index),
                    "jaccard": round(similarity, 4),
                })
                break
    return matches


def run_inference(
    rows: list[dict], base_path: Path, adapter_path: Path | None,
    train_path: Path, batch_size: int = 8, compute_base: bool = True,
) -> tuple[list[str] | None, list[str] | None]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the full verifier run")
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), torch_dtype=torch.float16, device_map="auto", local_files_only=True
    )
    if adapter_path is not None:
        from peft import PeftConfig, PeftModel

        try:
            if not (adapter_path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"missing adapter_config.json in {adapter_path}")
            config = PeftConfig.from_pretrained(str(adapter_path))
            if config.peft_type.value != "LORA":
                raise ValueError("only a LoRA adapter is accepted")
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        except Exception as exc:
            raise InvalidAdapterSubmission(f"{type(exc).__name__}: {exc}") from exc
    model.eval()
    shots = load_shots(train_path)
    device = next(model.parameters()).device

    def generate_predictions(tag: str) -> list[str]:
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                prompts = [make_prompt(row["question"], shots) for row in batch]
                tokens = tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=3072,
                )
                tokens = {name: value.to(device) for name, value in tokens.items()}
                generated = model.generate(
                    **tokens, max_new_tokens=384, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                input_width = tokens["input_ids"].shape[-1]
                predictions.extend(
                    tokenizer.decode(output[input_width:], skip_special_tokens=True)
                    for output in generated
                )
                done = min(start + len(batch), len(rows))
                if done // 50 > start // 50 or done == len(rows):
                    print(f"{tag}: evaluated {done}/{len(rows)}", flush=True)
        return predictions

    if adapter_path is None:
        return generate_predictions("base"), None
    base_responses = None
    if compute_base:
        with model.disable_adapter():
            base_responses = generate_predictions("base")
    return base_responses, generate_predictions("adapter")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/models/base"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--train-data", type=Path, default=Path("/data/gsm8k_train.jsonl"))
    parser.add_argument("--dev-data", type=Path, default=Path("/data/gsm8k_dev.jsonl"))
    parser.add_argument("--reward-json", type=Path, required=True)
    parser.add_argument("--responses-jsonl", type=Path)
    parser.add_argument("--overlap-report", type=Path, default=Path("/logs/verifier/overlap_report.json"))
    parser.add_argument("--baseline-json", type=Path, default=Path("/tests/base_baseline.json"))
    parser.add_argument("--status-json", type=Path, default=Path("/logs/verifier/submission_status.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--require-submission-evidence", type=lambda value: value.lower() in {"1", "true", "yes"},
        default=True,
    )
    args = parser.parse_args()

    rows = load_rows(args.eval_data)
    baseline = None
    if args.adapter is not None and args.baseline_json.is_file():
        candidate = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        eval_sha256 = hashlib.sha256(args.eval_data.read_bytes()).hexdigest()
        protocol_id = f"gsm-symbolic-v1-4shot-greedy384-batch{args.batch_size}"
        if (
            candidate.get("base_model") == BASE_MODEL_ID
            and candidate.get("base_revision") == BASE_MODEL_REVISION
            and candidate.get("base_weights_sha256") == BASE_MODEL_WEIGHT_SHA256
            and candidate.get("eval_data_sha256") == eval_sha256
            and candidate.get("protocol_id") == protocol_id
            and isinstance(candidate.get("metrics"), dict)
            and all(isinstance(value, (int, float)) for value in candidate["metrics"].values())
            and {"score", "acc_main", "acc_p1", "acc_p2"} <= candidate["metrics"].keys()
        ):
            baseline = candidate["metrics"]
    evidence_audit = "checked_clean"
    if args.adapter is not None and args.require_submission_evidence:
        error = submission_bundle_error(args.adapter)
        if error:
            metrics = invalid_reward(rows)
            args.reward_json.parent.mkdir(parents=True, exist_ok=True)
            args.reward_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            write_submission_status(args.status_json, False, error)
            print(json.dumps(metrics, indent=2), flush=True)
            return
        train_path = args.adapter.parent / "train_data.jsonl"
        training_questions = [row["question"] for row in load_rows(train_path)]
        dev_rows = load_rows(args.dev_data)
        overlap_matches = find_near_duplicates(training_questions, [*rows, *dev_rows])
        args.overlap_report.parent.mkdir(parents=True, exist_ok=True)
        args.overlap_report.write_text(json.dumps({"threshold": 0.75, "matches": overlap_matches}, indent=2) + "\n", encoding="utf-8")
        if overlap_matches:
            reason = f"training data has {len(overlap_matches)} exact/near overlaps with held-out evaluation questions"
            metrics = invalid_reward(rows)
            metrics["overlap_count"] = float(len(overlap_matches))
            args.reward_json.parent.mkdir(parents=True, exist_ok=True)
            args.reward_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            write_submission_status(args.status_json, False, reason)
            print(json.dumps(metrics, indent=2), flush=True)
            return
    elif args.adapter is not None:
        evidence_audit = "not_checked_legacy_trial"
    try:
        base_responses, adapter_responses = run_inference(
            rows, args.base_model, args.adapter, args.train_data, args.batch_size,
            compute_base=baseline is None,
        )
    except InvalidAdapterSubmission as exc:
        metrics = invalid_reward(rows)
        args.reward_json.parent.mkdir(parents=True, exist_ok=True)
        args.reward_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        write_submission_status(args.status_json, False, str(exc))
        print(json.dumps(metrics, indent=2), flush=True)
        return
    metrics = aggregate(rows, adapter_responses if adapter_responses is not None else base_responses)
    if args.adapter is not None:
        metrics["valid_submission"] = float(args.require_submission_evidence)
        metrics["evidence_audit_complete"] = float(args.require_submission_evidence)
        base_metrics = baseline if baseline is not None else aggregate(rows, base_responses)
        metrics["base_score"] = base_metrics["score"]
        metrics["base_acc_main"] = base_metrics["acc_main"]
        metrics["base_acc_p1"] = base_metrics["acc_p1"]
        metrics["base_acc_p2"] = base_metrics["acc_p2"]
        metrics["delta_score"] = metrics["score"] - metrics["base_score"]
        metrics["baseline_reference_used"] = float(baseline is not None)
        write_submission_status(
            args.status_json, bool(args.require_submission_evidence),
            evidence_audit if args.require_submission_evidence else "legacy trial; training evidence was not submitted",
        )
    args.reward_json.parent.mkdir(parents=True, exist_ok=True)
    args.reward_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.responses_jsonl:
        args.responses_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.responses_jsonl.open("w", encoding="utf-8") as handle:
            final_responses = adapter_responses if adapter_responses is not None else base_responses
            for row, response in zip(rows, final_responses):
                handle.write(json.dumps({**row, "response": response,
                                         "correct": is_correct(response, row["gold"])}, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
