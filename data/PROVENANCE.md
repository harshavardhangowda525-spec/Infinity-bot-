# Dataset provenance

| Source | License | What is used | Stored in repo? |
|---|---|---|---|
| `infinity-original` | Project-Original (authored for this repo) | `data/corpus/` | Yes |
| `python-stdlib` | PSF-2.0 (Python Software Foundation License) | Top-level `*.py` modules of the local CPython install | No — read locally at prepare time |

Rules enforced by `infinity/core/data.py`:

- A source whose license is not in `ALLOWED_LICENSES` is rejected and listed in `prepared.json`.
- Documents matching secret patterns (private keys, cloud/API tokens, hard-coded passwords) are dropped.
- E-mail addresses are replaced with `<email>`.
- Exact duplicates are removed (sha256 of normalized text).
- Splits are made per document (train/val/test), so no file appears in two splits.
- The tokenizer is trained on the **train split only**.
- `runs/.../prepared.json` and `files.json` record exactly which files went into which split.

To add data: append a source to `data/manifest.json` with a license and origin, then re-run `infinity data prepare`.
Never add data you do not have the right to use.
