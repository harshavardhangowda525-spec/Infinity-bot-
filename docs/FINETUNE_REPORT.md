# INFINITY HYPER 5.5 — tool-use fine-tuning report

Run date: 2026-10-12. Hardware: 4-core CPU, no GPU. Every number below was
produced by `infinity bench run` and stored in `runs/benchmarks.db` (test set)
or `runs/selection.db` (checkpoint selection). Checkpoints and databases are
not committed (`runs/` is ignored); the commands at the end reproduce them.

## What was trained

| | Base | SFT v1 | SFT v2 |
|---|---|---|---|
| Starting point | pretrained `hyper-5.5-tiny` step 1500 | base | base |
| Parameters | 5,795,072 | same | same |
| Context window | 256 | 512 (extended) | 512 (extended) |
| Fine-tuning data | — | 2,000 tasks / 10,000 examples, 12 module names | 2,000 tasks / 12,000 examples, generated identifiers |
| Expert sequence | — | test → read → edit → test → final | list_dir → test → read → edit → test → final |
| Rehearsal on pretraining text | — | none | 25% of steps |
| Steps / wall time | — | 1,500 / 25 min | 1,500 / 61 min (shared CPU) |

All training transcripts were produced by replaying expert actions through the
**real** agent runtime: real files, real `python3 -m unittest`, real
permission checks. Only runs the runtime judged `completed_verified` were kept
(4,000 of 4,000). Evaluation tasks use identifiers the model never saw.

## Results (held-out test set, seed 1337)

| Metric | Base | SFT v1 (step 1500) | SFT v2 (step 1000, selected) |
|---|---|---|---|
| Test-split perplexity (pretraining text) | 40.15 | **372,569** (catastrophic forgetting) | **44.19** |
| Valid JSON actions (200 held-out prompts) | 0% | 100% | 100% |
| Right tool chosen | 0% | 100% | 100% |
| Exact action (tool and arguments) | 0% | 60% | 67% |
| Live repair: tests pass after the agent's fix (30 unseen tasks) | **0 / 30** | **0 / 30** | **0 / 30** |

Protocol numbers for v1 and v2 are measured on each version's own held-out
evaluation file (different expert sequences), so they are comparable in
spirit, not item by item.

Checkpoint selection for v2 used a separate task set (seed 4242). All six v2
checkpoints (steps 250–1500) scored 0/30, so the one with the lowest held-out
evaluation loss (step 1000, 0.427) was reported.

## What this shows

1. **The pipeline works.** Fine-tuning taught the model the agent's action
   format and the right tool order: from 0% to 100% valid actions.
2. **Rehearsal fixed forgetting.** v1 destroyed the model's general language
   ability (perplexity 40 → 372,569). v2 kept it (40 → 44).
3. **The model is too small to act on what it sees.** In live runs it
   lists the files, then reads a file name it invents from training-style
   syllables instead of copying the name shown in the listing:

   ```
   1 list_dir {}                                  ok   (shows elukex.py)
   2 run_command python3 -m unittest -q           failed (as expected)
   3 read_file {'path': 'turika.py'}              not a file
   ...  repeats until the 8-step limit → blocked
   ```

   Copying unseen strings from context is something a 5.8M-parameter model
   trained on about 1M tokens of text has not learned. Teacher-forced metrics
   look good, but end-to-end success stays at zero, and the runtime reports it
   as `blocked` rather than pretending.

## Decision

No checkpoint was promoted (`infinity model promote` was not run). v2 is
better than v1 and keeps language ability, but it does not complete any real
task, so the studio keeps loading the newest checkpoint as before.

## What would move the live score

- A larger model (the `hyper-5.5-small` config, about 100M parameters) pretrained
  on far more code. Copying from context ("induction") reliably appears only
  with more capacity and data. See `docs/FEASIBILITY.md` for cost and time.
- More varied expert data, including recovery from errors (for example, after
  "not a file", list the directory again).

## Reproduce

```bash
infinity sft generate --out runs/sft-v2 --train 2000 --eval 100 --context 512
infinity sft train --base runs/hyper-5.5-tiny/checkpoints/step_001500 --data runs/sft-v2 \
    --out runs/hyper-5.5-tiny-sft2 --steps 1500 --context 512 --lm-mix 0.25 --eval-every 250
for s in 000250 000500 000750 001000 001250 001500; do
  infinity bench run --checkpoint runs/hyper-5.5-tiny-sft2/checkpoints/step_$s --suites repair \
      --repair-seed 4242 --store runs/selection.db --sft runs/sft-v2
done
infinity bench run --checkpoint runs/hyper-5.5-tiny-sft2/checkpoints/step_001000 --suites lm,protocol,repair --sft runs/sft-v2
infinity bench list
```
