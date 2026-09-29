"""Tiny LoRA reference run for Oracle smoke testing, not an optimized solution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("The reference LoRA run requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model), torch_dtype=torch.float16, device_map="auto", local_files_only=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                   task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]),
    )
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    rows = [json.loads(line) for line in args.train_data.read_text(encoding="utf-8").splitlines()[:args.steps]]
    for step, row in enumerate(rows, 1):
        prompt = f"Solve the math word problem step by step. End with #### and the final number.\nQ: {row['question']}\nA:"
        completion = " " + row["answer"].strip() + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:768]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:768]
        if all(value == -100 for value in labels):
            continue
        inputs = torch.tensor([input_ids], device="cuda")
        targets = torch.tensor([labels], device="cuda")
        loss = model(input_ids=inputs, labels=targets).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 8 == 0:
            print(f"reference step={step} loss={loss.item():.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    summary = {
        "purpose": "Oracle smoke check only; no expected score improvement",
        "base_model": "Qwen/Qwen2.5-Math-1.5B",
        "source": "GSM8K train, first 32 records",
        "steps": len(rows),
        "method": "LoRA SFT, rank 8, q_proj and v_proj",
    }
    Path("/logs/artifacts/run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
