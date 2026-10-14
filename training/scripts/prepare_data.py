"""Validate, deduplicate, leak-check and tokenize the datasets -> training/data/ + report.

    python training/scripts/prepare_data.py --config training/configs/default.yaml
"""

import argparse
import json

from _common import ROOT, tokenizer_for

from infinity.hf.config import load_config
from infinity.hf.prepare import prepare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="training/configs/default.yaml")
    ap.add_argument("--tool-tasks", type=int, help="override data.tool_tasks_train")
    a = ap.parse_args()
    over = {"data": {"tool_tasks_train": a.tool_tasks}} if a.tool_tasks is not None else None
    cfg = load_config(ROOT / a.config, over)
    rep = prepare(cfg, ROOT, tokenizer_for(cfg))
    print(json.dumps({k: rep[k] for k in ("train", "val", "duplicates_removed", "leaks_removed_from_train",
                                          "too_long_dropped", "tool_transcripts")}, indent=2))
    fails = len(rep["validation_failures"])
    print(f"validation failures: {fails} (details in {cfg['data']['out_dir']}/report.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
