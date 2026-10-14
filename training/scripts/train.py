"""LoRA fine-tuning of the configured open model. Never touches the base weights.

    python training/scripts/train.py --config training/configs/default.yaml --dry-run
    python training/scripts/train.py --config training/configs/default.yaml --smoke
    python training/scripts/train.py --config training/configs/default.yaml           # full run (asks first)
    python training/scripts/train.py --config training/configs/default.yaml --resume
"""

import argparse
import sys

from _common import ROOT, tokenizer_for

from infinity.hf.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="training/configs/default.yaml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="validate config + data only (no model weights loaded)")
    g.add_argument("--smoke", action="store_true", help="2 steps on 4 examples in a temp folder, then reload check")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint of this run")
    ap.add_argument("--steps", type=int, help="override training.max_steps")
    ap.add_argument("--yes", action="store_true", help="start a full run without the confirmation prompt")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config, {"training": {"max_steps": a.steps}} if a.steps else None)

    from infinity.hf.train import TrainingOOM, plan, train

    print(f"config {a.config} · run {cfg['run_name']} · base {cfg['base_model']}")
    summary = plan(cfg, ROOT, tokenizer_for(cfg))
    if a.dry_run:
        print("dry run OK — nothing was trained")
        return 0
    if not a.smoke and not a.yes:
        print(f"\nThis will run {cfg['training']['max_steps']} optimizer steps "
              f"({summary['sequences_processed']} sequences). On a CPU this can take hours; "
              "progress and ETA are printed every step and Ctrl+C saves a checkpoint.")
        if input("Start training? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not started")
            return 1
    try:
        res = train(cfg, ROOT, resume=a.resume, smoke=a.smoke)
    except TrainingOOM as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
