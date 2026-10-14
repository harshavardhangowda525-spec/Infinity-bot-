# Evaluation report

Every number here was measured. Anything not measured says so.

## Test set (`training/evaluation/test_cases.jsonl`, 24 held-out cases)

| Suite | Cases | How it is scored |
|---|---|---|
| instruction | 4 | regex / exact format |
| factual | 4 | verifiable answer present |
| reasoning | 4 | unseen word problems, answer present |
| uncertainty | 3 | admits it cannot know, or says the made-up book doesn't exist |
| coding | 7 | generated code must pass hidden unit tests in the sandbox |
| error_handling | 2 | generated code must raise the right exception |
| repair (optional `--repair N`) | N | real agent runtime fixes held-out bugs; `completed_verified` counts |

- **Decoding:** greedy, so runs are reproducible. Latency, tokens/s and peak process memory are recorded.
- **Results files:** `training/evaluation/results/<label>.json` and `.md`. They are never
  overwritten, which keeps the baseline safe.

## Measured results

| Model | Overall | Notes |
|---|---|---|
| INFINITY HYPER 5.5 tiny (step 1500, 5.8M params) | **0 / 24** | Every suite 0%. Mean latency 0.96 s, 398 tokens/s, peak memory 594 MB (build machine, CPU) |
| INFINITY HYPER 5.5 — Fine-Tuned (LoRA adapter) | **0 / 24** | Replies to everything with an agent action. Mean latency 0.18 s, peak memory 597 MB |
| Tiny random Qwen2-architecture test model | 0 / 6 (subset) | Pipeline check only. Random weights; meaningless as quality |
| **Qwen2.5-Coder 0.5B Instruct (baseline)** | **not measured** | huggingface.co is blocked in the build environment. Run `evaluate.py --label baseline` on your PC |
| **+ combined adapter** | **not measured** | needs a real training run on your PC |
| **chat-only vs coding-only adapters** | **not measured** | see `docs/training_guide.md` §2 |

**Correction made during the build:** the first version of the `inst-identity` check passed
HYPER 5.5 by accident, because its gibberish contained the word "model". The check now
requires the model to actually name itself. HYPER's score went from 1/24 to 0/24.

## INFINITY HYPER 5.5 vs INFINITY HYPER 5.5 — Fine-Tuned (measured 2026-10-14)

**Setup:**
- Adapter `adapters/infinity-hyper-5-5/hyper-5.5-finetuned`.
- LoRA rank 16, alpha 32 on the qkv, proj, gate, up and down layers: 419,328 trainable parameters.
  The 5,795,072 base parameters stay frozen.
- 1,000 steps on the `runs/sft-v2` tool-use data (12,000 examples), context 512, LM rehearsal 0.25.
- 1,183 s on the build machine's CPU.

**Original model check:**
- All 5 base checkpoint files were sha256-identical before and after training.
- This was checked by the training run and again independently with `sha256sum`.

| Benchmark (same settings for both) | INFINITY HYPER 5.5 | INFINITY HYPER 5.5 — Fine-Tuned |
|---|---|---|
| Tool-use eval loss (SFT eval split) | 6.293 | **0.475** |
| LM perplexity (pretraining test split, lower is better) | 40.15 | 40.33 |
| Protocol: valid JSON actions (200 held-out prompts) | 0% | **99%** |
| Protocol: right tool | 0% | **99%** |
| Protocol: exact action | 0% | **67%** |
| Live repair tasks fixed (30, real runtime) | 0 / 30 | 0 / 30 |
| Mean agent steps per repair task | 0 (blocked at once) | 7.7 (works through tools, runs out of steps) |
| Held-out chat/coding test set (24 cases) | 0 / 24 | 0 / 24 |

**What this means:**
- The adapter taught the original model the bot's action format almost perfectly, while
  general text ability stayed the same (perplexity 40.15 → 40.33).
- It still fixes **no** real bugs.
- It answers every chat question with an agent action, for example
  `{"thought":"Look at the project files.", …}`, because that is all it was trained on.
- These results match the earlier full fine-tune (`docs/FINETUNE_REPORT.md`). The limit is
  the model's size (5.8M parameters), not the training method.

Records: `runs/benchmarks.db` results #10–#15 (the fine-tuned version is stored under the
run `hyper-5.5-tiny+lora:hyper-5.5-finetuned`), plus
`training/evaluation/results/hyper-5.5-tiny.json` and `hyper-5.5-finetuned.json`.

## How to add results

```bat
python training/scripts/evaluate.py --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --label baseline --repair 10
python training/scripts/evaluate.py --model "hf:Qwen/Qwen2.5-Coder-0.5B-Instruct::adapters/open-models/qwen-coder-0.5b-combined" --label combined --repair 10
python training/scripts/evaluate.py --compare baseline combined
```

Paste the comparison into this file.

**Choosing an adapter:**
- Base it on the evaluation, not on training loss.
- Look at regressions as well as gains.
