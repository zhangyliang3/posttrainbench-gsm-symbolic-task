# Research task: improve robustness on GSM-Symbolic

You are a post-training research agent. You have **one GPU and a 3,600-second training budget** to adapt the provided **base model** so it answers grade-school math word problems more reliably when names/numbers change and extra conditions are added. The Agent phase has up to 4,800 seconds total, including setup and a bounded development comparison; finish training within the one-hour budget and reserve the remaining time for a concise check. Do not spend it on exhaustive evaluation. Submit a PEFT LoRA adapter and the evidence files listed below. The verifier measures numerical-answer accuracy on fixed subsets of GSM-Symbolic Main, P1 and P2.

## Inputs

- Base model: `/models/base` (`Qwen/Qwen2.5-Math-1.5B`, base version). Use only this model and its own derivatives.
- Training source: `/data/gsm8k_train.jsonl` (the first 7,223 records of the pinned GSM8K train split). Each line has `source_index`, `question` and `answer`; the final answer follows `####`.
- Development source: `/data/gsm8k_dev.jsonl` (the remaining 250 records of that same train split). Use it only to compare the base and candidate adapter; never train on it or synthesize examples from it. The one-hour limit applies to actual training; development checks and submission preparation must fit in the Agent phase's 80-minute total limit.
- Python, PyTorch, Transformers, PEFT, Accelerate and Datasets are installed. One NVIDIA GPU is requested.
- You may synthesize new training examples from the permitted train data and choose your own training procedure, sample mixture, LoRA modules/rank, hyperparameters and checkpoint. Keep a brief record of sources and decisions.

## Boundaries

- Do not train on GSM8K test or any GSM-Symbolic Main/P1/P2 question or answer. Do not download or reconstruct their answer tables.
- Do not train on `/data/gsm8k_dev.jsonl`; keep its questions and answers out of every training or synthetic example.
- Do not replace the base model with an instruction-tuned model or another pretrained checkpoint.
- Do not modify task verification code, score files or environment configuration.
- Agent network access is restricted to the configured DeepSeek API host. Do not attempt to access other public hosts, benchmarks, datasets or model weights.
- Keep the actual training data used, training script and a concise run summary as submission evidence. The verifier checks these files for format and overlap with the held-out development and final evaluation questions.

## Submission

Save all of these in `/logs/artifacts/`:

- `adapter/`: the final LoRA adapter, including `adapter_config.json` and adapter weights readable by `PeftModel.from_pretrained`. Do not merge the full base weights into this directory.
- `train_script.py`: the exact script used for final training.
- `train_data.jsonl`: the exact examples used for final training, one JSON object per line with `question` and `answer` fields. Do not include development examples.
- `run_summary.json`: a JSON object with these keys: `base_model` (`Qwen/Qwen2.5-Math-1.5B`), `base_revision` (`4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`), `training_data_source` (`openai/gsm8k:main@740312add88f781978c0658806c59bc2815b9866#train[0:7223]` plus a description of any derived examples), `training_rows` (must equal the JSONL row count), `training_method`, `seed`, `training_steps`, `training_elapsed_seconds`, `training_data_sha256`, `training_script_sha256`, and `development_metrics` (an object containing base and final-adapter results on `/data/gsm8k_dev.jsonl`). The two SHA256 values must match the submitted files.

Missing evidence, invalid adapter files, or a detected exact/near-duplicate between submitted training questions and held-out/final evaluation questions makes the submission invalid and yields a structured zero reward with a reason. The adapter is the scored artifact; the evidence files make its provenance auditable.

The verifier reloads the original base model, attaches your adapter and uses the **same fixed prompt and greedy decoding protocol** used for the unmodified base. It extracts a final answer in the form `#### <number>`. Missing or invalid final answers count as incorrect. The final task score is the mean of the Main, P1 and P2 accuracies on 250 examples per version; the three individual accuracies are also recorded. Development metrics are for experiment selection only and do not affect the reward. A valid submission is not required to exceed the base score, but improvements should be supported by the same-protocol development comparison.

You are free to research how best to use your limited training budget. The reference solution is only a task-author smoke check, not a training recipe you must follow.
