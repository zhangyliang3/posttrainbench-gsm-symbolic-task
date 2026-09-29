"""Small semantic checks for scoring, independent of GPU and Harbor."""

import importlib.util
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1] / "gsm_symbolic_posttrain" / "tests"
SPEC = importlib.util.spec_from_file_location("task_evaluate", ROOT / "evaluate.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extract() -> None:
    assert MODULE.extract_final("reasoning\n#### 1,234.5") == 1234.5
    assert MODULE.extract_final("no final marker") is None
    assert MODULE.is_correct("#### 32.500000", "32.5")
    assert not MODULE.is_correct("#### 35", "32.5")


def test_balanced_macro() -> None:
    rows = [
        {"variant": "main", "gold": "1"},
        {"variant": "p1", "gold": "2"},
        {"variant": "p2", "gold": "3"},
    ]
    metrics = MODULE.aggregate(rows, ["#### 1", "#### 2", "#### 4"])
    assert metrics["acc_main"] == 1
    assert metrics["acc_p1"] == 1
    assert metrics["acc_p2"] == 0
    assert abs(metrics["score"] - 2 / 3) < 1e-12


def test_invalid_adapter_is_structured_zero() -> None:
    rows = [{"variant": variant} for variant in ("main", "p1", "p2")]
    reward = MODULE.invalid_reward(rows)
    assert reward["score"] == 0.0
    assert reward["valid_submission"] == 0.0
    assert all(isinstance(value, float) for value in reward.values())


def test_near_duplicate_audit_normalizes_numbers() -> None:
    training = ["A farmer has 12 bags of seed and sells 5 bags each day for 3 days. How many bags remain?"]
    eval_rows = [
        {"variant": "main", "id": 1, "question": "A farmer has 16 bags of seed and sells 7 bags each day for 2 days. How many bags remain?"},
        {"variant": "p1", "id": 2, "question": "Lena reads a book in a quiet room. What does she read?"},
    ]
    matches = MODULE.find_near_duplicates(training, eval_rows)
    assert len(matches) == 1
    assert matches[0]["reference_variant"] == "main"


def test_submission_bundle_contract() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        adapter = root / "adapter"
        adapter.mkdir()
        assert MODULE.submission_bundle_error(adapter) == "missing adapter_config.json"
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        (root / "train_script.py").write_text("print('train')", encoding="utf-8")
        (root / "train_data.jsonl").write_text('{"question":"q","answer":"#### 1"}\n', encoding="utf-8")
        summary = {
            "base_model": "Qwen/Qwen2.5-Math-1.5B",
            "base_revision": "pinned",
            "training_data_source": "GSM8K train",
            "training_rows": 1,
            "training_method": "LoRA",
            "seed": 0,
            "training_steps": 1,
            "training_elapsed_seconds": 1,
            "training_data_sha256": hashlib.sha256((root / "train_data.jsonl").read_bytes()).hexdigest(),
            "training_script_sha256": hashlib.sha256((root / "train_script.py").read_bytes()).hexdigest(),
            "development_metrics": {},
        }
        (root / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        assert MODULE.submission_bundle_error(adapter) is None


if __name__ == "__main__":
    test_extract()
    test_balanced_macro()
    test_invalid_adapter_is_structured_zero()
    test_near_duplicate_audit_normalizes_numbers()
    test_submission_bundle_contract()
    print("scoring checks passed")
