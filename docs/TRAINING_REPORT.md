# INFINITY HYPER 5.5 — training report: `hyper-5.5-tiny`

Run date: 2026-10-12. Hardware: 4-core x86_64 CPU, no GPU (see `FEASIBILITY.md`).
All values are copied from `runs/hyper-5.5-tiny/train_log.jsonl` and the
checkpoint manifests produced by this run.

Checkpoints are not committed to git: they are 67 MB each, and `runs/` is
ignored. Reproduce them with the commands at the end.

## Setup

| | |
|---|---|
| Config | `configs/hyper-5.5-tiny.json` |
| Parameters | 5,795,072 |
| Tokenizer | INFINITY BPE, vocabulary 4,096, fingerprint `7e487f94e3434858`, trained on the train split only |
| Data | 153 documents after filtering (1 duplicate and 1 too-short file dropped; 0 secrets found) |
| Train / val / test tokens | 1,041,458 / 86,795 / 78,990 (split per document) |
| Steps | 1,500 × 4,096 tokens = 6,144,000 tokens seen (~5.9 passes over the train split) |
| Wall time | 877 s (~7,000 tokens/s) |

## Loss

The initial training loss was 8.365, close to ln(4096) = 8.32, which is what a
uniform guess would score.

| Step | Train loss | Val loss | Tokens seen | Elapsed (s) |
|---|---|---|---|---|
| 250 | 3.978 | 4.554 | 1,024,000 | 140 |
| 500 | 3.731 | 4.108 | 2,048,000 | 281 |
| 750 | 2.837 | 3.845 | 3,072,000 | 432 |
| 1000 | 2.567 | 3.705 | 4,096,000 | 585 |
| 1250 | 2.135 | 3.619 | 5,120,000 | 731 |
| 1500 | 1.975 | 3.621 | 6,144,000 | 877 |

**Reading it.** Validation loss stopped improving after about step 1,250,
while training loss kept falling. The model is memorising the 1M-token
corpus (overfitting). More data is the fix, not more steps.

## Checkpoint verification

`infinity checkpoints --run runs/hyper-5.5-tiny` verified all three checkpoints
(500, 1000, 1500) against their sha256 manifests. Each one reloaded with
`weights_only=True` and generated text.

## Generation samples (step 1500, unedited)

Prompt `def read_config(path):`, seed 1, temperature 0.8:

```python
def read_config(path):
    """Debug file descriptors into a user-specification context."""

def _get_socket(path):
    """Generatedir objects with an object.
    """
    if not isinstance(path, bytes):
        raise TypeError("Invalid paths: %r" % (or os.name == "ascii")
    return path ord(path) in mode or "UTF-8" or "utf-32" or "0" or
```

Prompt `class Stack:`, seed 3, temperature 0.6:

```python
class Stack:
    """A file object that doesn't reference to open()."""

    def __init__(self, f):
        self.close()


class _StoreTrue(
    """Base class for sared methods of a semantics."""

    def __init__(self, f):
        self.file = f
        self.file = f
        ...
```

**What it has learned:** Python layout (indentation, docstrings, `def` and
`class`, `__init__`), standard-library vocabulary and idioms (`isinstance`,
`raise TypeError(...)`).

**What it has not learned:** meaning, correctness, planning, or the agent's
JSON action protocol.

## Agent run with this checkpoint

```
infinity agent --model neural:runs/hyper-5.5-tiny --mode debug --permission auto \
               --task "The add test fails. Fix it."      (workspace: a 2-file demo project)
```

Result: **`blocked`**. The model produced 3 consecutive outputs with no JSON
action, the runtime stopped, and no file was modified. The runtime also
noted that the 256-token context window is too small for the system prompt.
This is the expected and honest outcome for a 5.8M-parameter model trained
on 1M tokens.

## Reproduce

```bash
infinity data prepare --vocab 4096 --out runs/data
infinity train --config configs/hyper-5.5-tiny.json --data runs/data
infinity checkpoints --run runs/hyper-5.5-tiny
infinity generate --checkpoint runs/hyper-5.5-tiny --prompt "def read_config(path):" --seed 1
```

## Next steps toward a useful model

1. A larger licensed corpus (hundreds of millions to billions of tokens) through `data/manifest.json`.
2. The `hyper-5.5-small` config (about 100M parameters, 1,024-token context) on a GPU.
3. Supervised fine-tuning on tool-use transcripts in the action protocol, with a held-out benchmark.
