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
| Tiny random Qwen2-architecture test model | 0 / 6 (subset) | Pipeline check only. Random weights; meaningless as quality |
| **Qwen2.5-Coder 0.5B Instruct (baseline)** | **not measured** | huggingface.co is blocked in the build environment. Run `evaluate.py --label baseline` on your PC |
| **+ combined adapter** | **not measured** | needs a real training run on your PC |
| **chat-only vs coding-only adapters** | **not measured** | see `docs/training_guide.md` §2 |

**Correction made during the build:** the first version of the `inst-identity` check passed
HYPER 5.5 by accident, because its gibberish contained the word "model". The check now
requires the model to actually name itself. HYPER's score went from 1/24 to 0/24.

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
