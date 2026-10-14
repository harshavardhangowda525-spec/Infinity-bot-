"""Evaluate a model on the held-out test cases -> training/evaluation/results/<label>.{json,md}.

    # baseline: the open model before fine-tuning
    python training/scripts/evaluate.py --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --label baseline
    # the same model + a trained adapter
    python training/scripts/evaluate.py --model "hf:Qwen/Qwen2.5-Coder-0.5B-Instruct::adapters/open-models/qwen-coder-0.5b-combined" --label combined
    # INFINITY HYPER 5.5 itself, for reference
    python training/scripts/evaluate.py --model neural:runs/hyper-5.5-tiny --label hyper-tiny
    # compare two saved reports
    python training/scripts/evaluate.py --compare baseline combined
"""

import argparse
import json
import re

from _common import ROOT

RESULTS = ROOT / "training" / "evaluation" / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="hf:<model>[::<adapter>] or neural:<checkpoint>")
    ap.add_argument("--label", help="name for the saved report (e.g. baseline)")
    ap.add_argument("--cases", default="training/evaluation/test_cases.jsonl")
    ap.add_argument("--limit", type=int, help="only the first N cases (quick check)")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--context", type=int, default=4096, help="context budget for hf models")
    ap.add_argument("--repair", type=int, default=0, help="also run N live agent repair tasks (slow)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two saved report labels")
    a = ap.parse_args()

    from infinity.hf.evaluate import compare, load_cases, run_suite, to_markdown

    if a.compare:
        ra, rb = (json.loads((RESULTS / f"{x}.json").read_text(encoding="utf-8")) for x in a.compare)
        md = compare(ra, rb, *a.compare)
        out = RESULTS / f"compare-{a.compare[0]}-vs-{a.compare[1]}.md"
        out.write_text(md, encoding="utf-8")
        print(md)
        print("saved", out)
        return 0
    if not a.model or not a.label:
        ap.error("--model and --label are required (or use --compare)")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.label):
        ap.error("--label may only contain letters, digits, '.', '_' and '-'")

    out = RESULTS / f"{a.label}.json"
    if out.exists():  # never overwrite a saved result (keeps the baseline safe)
        raise SystemExit(f"{out} already exists; choose another --label")
    kind, _, arg = a.model.partition(":")
    if kind == "hf":
        from infinity.hf.model import HFModel

        base, _, adapter = arg.partition("::")
        model = HFModel(base, adapter or None, context_tokens=a.context, base_dir=ROOT)
    else:
        from pathlib import Path

        from infinity.agent.models import load_model

        if kind == "neural" and not Path(arg).is_absolute():
            a.model = f"neural:{ROOT / arg}"
        model = load_model(a.model)
    cases = load_cases(ROOT / a.cases)[: a.limit]
    print(f"evaluating {model.info().get('product')} on {len(cases)} cases…")
    report = run_suite(model, cases, max_new_tokens=a.max_new_tokens, include_repair=a.repair)
    report["label"] = a.label
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / f"{a.label}.md").write_text(to_markdown(report, f"Evaluation: {a.label}"), encoding="utf-8")
    print(f"overall {report['overall']['passed']}/{report['overall']['total']} · saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
