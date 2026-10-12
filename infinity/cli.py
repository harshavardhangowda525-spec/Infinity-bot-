"""INFINITY BOT command-line interface.

    infinity audit
    infinity data prepare  [--manifest data/manifest.json] [--out runs/data] [--vocab 4096]
    infinity train         --config configs/hyper-5.5-tiny.json [--data runs/data] [--steps N] [--no-resume]
    infinity generate      --checkpoint runs/hyper-5.5-tiny --prompt "def "
    infinity checkpoints   --run runs/hyper-5.5-tiny
    infinity agent         --workspace PATH --model neural:runs/hyper-5.5-tiny --mode code --task "..."
    infinity rollback      --workspace PATH --session ID
    infinity memory        show|export|delete-session|delete-note|wipe|trust|untrust ...
    infinity admin         create-owner --email you@example.com | users | grant-plan --email E --plan pro | audit-verify
    infinity studio        [--port 8765] [--no-registration]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path


def cmd_audit(args) -> int:
    from .audit import audit, recommend

    rep = audit()
    rep["recommendation"] = recommend(rep)
    print(json.dumps(rep, indent=2))
    return 0


def cmd_data_prepare(args) -> int:
    from .core.data import prepare

    summary = prepare(args.manifest, args.out, args.vocab, args.seed)
    print(json.dumps({k: summary[k] for k in ("tokenizer", "cleaning", "splits")}, indent=2))
    return 0


def cmd_train(args) -> int:
    from .core.config import RunConfig
    from .core.trainer import Trainer

    cfg = RunConfig.load(args.config)
    stop = threading.Event()

    def handle(sig, frame):
        print("\ninterrupt received — finishing the current step and saving a checkpoint ...")
        stop.set()

    signal.signal(signal.SIGINT, handle)
    trainer = Trainer(cfg, args.data, args.out)
    result = trainer.train(resume=not args.no_resume, max_steps=args.steps, stop_flag=stop)
    print(json.dumps(result, indent=2))
    return 0


def cmd_generate(args) -> int:
    from .core.inference import InferenceEngine, SamplingConfig

    eng = InferenceEngine(args.checkpoint, device=args.device)
    info = eng.info()
    print(f"[{info['product']} · {info['implementation']} · step {info['step']} · {info['parameters']:,} params]",
          file=sys.stderr)
    cfg = SamplingConfig(max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k,
                         top_p=args.top_p, repetition_penalty=args.repetition_penalty, seed=args.seed)
    sys.stdout.write(args.prompt)
    for piece in eng.stream(args.prompt, cfg, allow_special=False):
        sys.stdout.write(piece)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return 0


def cmd_checkpoints(args) -> int:
    from .core.checkpoint import list_checkpoints, verify_checkpoint

    root = Path(args.run) / "checkpoints"
    for c in list_checkpoints(root):
        try:
            verify_checkpoint(Path(c["path"]))
            ok = "verified"
        except Exception as e:
            ok = f"INVALID: {e}"
        print(f"step {c['step']:>7}  {ok:<10}  {json.dumps(c['metrics'])}")
    return 0


def _cli_approver(req) -> bool:
    print("\n" + "=" * 70)
    print(f"APPROVAL NEEDED · {req.tool} · {req.category.value}" + (f" · risk {req.risk.value}" if req.risk else ""))
    print(req.summary)
    for r in req.reasons:
        print(f"  - {r}")
    if req.diff:
        diff = req.diff if len(req.diff) < 6000 else req.diff[:6000] + "\n... (diff truncated)"
        print(diff)
    try:
        ans = input("Allow? [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def cmd_agent(args) -> int:
    from .agent import Agent, Limits, Memory, Mode, PermissionMode, PermissionPolicy, Workspace
    from .agent.models import ModelUnavailable, load_model

    ws = Workspace(Path(args.workspace))
    memory = Memory(args.db) if not args.no_memory else None
    perm = args.permission
    if perm is None:
        trusted = memory.workspace_trust(str(ws.root)) if memory else None
        perm = trusted or PermissionMode.READ_ONLY.value
        if trusted is None:
            print(f"[workspace {ws.root} is not trusted yet — running read-only. "
                  f"Use --permission ask|auto or `infinity memory trust {ws.root} ask`]", file=sys.stderr)
    try:
        model = load_model(args.model)
    except ModelUnavailable as e:
        print(f"INFINITY HYPER 5.5 is not available: {e}", file=sys.stderr)
        return 2
    info = model.info()
    print(f"[{info.get('product')} · {info.get('implementation')}]", file=sys.stderr)
    policy = PermissionPolicy(PermissionMode(perm), approver=_cli_approver if sys.stdin.isatty() else (lambda r: False))
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda s, f: cancel.set())

    def on_event(e):
        t = e.get("type")
        if t == "state":
            extra = f" {e.get('tool')} {json.dumps(e.get('args'))[:160]}" if e.get("tool") else ""
            print(f"· step {e['step']} {e['state']}{extra}", file=sys.stderr)
        elif t == "plan":
            print("· plan: " + " | ".join(e["plan"]), file=sys.stderr)
        elif t == "invalid_output":
            print(f"· step {e['step']} invalid model output: {e['error']}", file=sys.stderr)
        elif t == "output" and args.verbose:
            print("  " + e["line"].rstrip(), file=sys.stderr)
        elif t == "tool_result" and args.verbose:
            print("  " + e["observation"][:600].replace("\n", "\n  "), file=sys.stderr)

    limits = Limits(max_steps=args.max_steps, max_seconds=args.max_seconds, command_timeout=args.command_timeout)
    agent = Agent(model, ws, policy, Mode(args.mode), memory=memory, limits=limits, on_event=on_event, cancel=cancel)
    report = agent.run(args.task)
    print(report.to_text())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return 0 if report.status in ("completed_verified", "completed_unverified", "answered") else 1


def cmd_rollback(args) -> int:
    from .agent.checkpoints import CheckpointStore, list_sessions

    root = Path(args.workspace).resolve()
    if not args.session:
        for s in list_sessions(root):
            print(f"{s['session']}  {s['files']} path(s)")
        return 0
    store = CheckpointStore(root, args.session)
    if not store.index:
        print(f"no checkpoint for session {args.session}", file=sys.stderr)
        return 1
    print("will restore:\n  " + "\n  ".join(store.touched()))
    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1
    for rel in store.restore():
        print(f"restored {rel}")
    return 0


def cmd_memory(args) -> int:
    from .agent import Memory

    mem = Memory(args.db)
    a = args.action
    if a == "show":
        print(json.dumps({"preferences": mem.preferences(), "notes": mem.notes(), "trusted": mem.export()["trusted_workspaces"],
                          "recent_tasks": [{k: t[k] for k in ("id", "task", "mode", "status", "reason")} for t in mem.tasks(20)]},
                         indent=2, default=str))
    elif a == "export":
        Path(args.target).write_text(json.dumps(mem.export(), indent=2, default=str))
        print(f"exported to {args.target}")
    elif a == "delete-session":
        print("deleted" if mem.delete_session(args.target) else "not found")
    elif a == "delete-note":
        print("deleted" if mem.delete_note(int(args.target)) else "not found")
    elif a == "set-preference":
        key, _, value = args.target.partition("=")
        mem.set_preference(key, value)
    elif a == "trust":
        mem.trust_workspace(str(Path(args.target).resolve()), args.value or "ask")
    elif a == "untrust":
        print("removed" if mem.untrust_workspace(str(Path(args.target).resolve())) else "not found")
    elif a == "wipe":
        if input("Delete ALL INFINITY memory? type 'wipe' to confirm: ").strip() == "wipe":
            mem.wipe()
            print("wiped")
    return 0


def cmd_admin(args) -> int:
    import getpass

    from .platform import AuthError, BillingError, Platform

    p = Platform(args.platform_db)
    try:
        if args.action == "create-owner":
            if p.auth.owner_exists():
                print("an owner account already exists", file=sys.stderr)
                return 1
            if not args.email:
                print("--email is required", file=sys.stderr)
                return 2
            pw = getpass.getpass("Owner password (min 10 chars): ")
            if pw != getpass.getpass("Repeat password: "):
                print("passwords do not match", file=sys.stderr)
                return 1
            u = p.auth.create_user(args.email, pw, "owner")
            print(f"owner account created: {u.email} (id {u.id})")
        elif args.action == "users":
            for r in p.db.all("SELECT id, email, role, status, created_at FROM users ORDER BY id"):
                print(f"{r['id']:>4}  {r['role']:<5}  {r['status']:<9}  {r['email']}")
        elif args.action == "grant-plan":
            owner = p.db.one("SELECT id FROM users WHERE role='owner'")
            u = p.auth.by_email(args.email or "")
            if not owner or not u:
                print("need an owner account and an existing --email user", file=sys.stderr)
                return 1
            sub = p.billing.owner_grant(u.id, args.plan, owner["id"], args.reason or "granted from CLI")
            print(f"granted {args.plan} to {u.email} until {sub['current_period_end']:.0f}")
        elif args.action == "audit-verify":
            print(json.dumps(p.audit.verify()))
    except (AuthError, BillingError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_sft(args) -> int:
    if args.action == "generate":
        from .core.tokenizer import BPETokenizer
        from .sft.generate import generate

        tok = BPETokenizer.load(Path(args.tokenizer))
        print(json.dumps(generate(args.out, tok, args.context, args.train, args.eval, args.seed), indent=2))
    else:
        from .sft.finetune import finetune

        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda s, f: stop.set())
        print(json.dumps(finetune(args.base, args.data, args.out, steps=args.steps, context=args.context,
                                  batch_size=args.batch_size, lr=args.lr, stop_flag=stop, lm_mix=args.lm_mix,
                                  lm_data=args.lm_data, eval_every=args.eval_every), indent=2))
    return 0


def cmd_bench(args) -> int:
    from .bench.runner import BenchStore, run

    if args.action == "list":
        for r in BenchStore(args.store).list(args.checkpoint):
            print(f"#{r['id']:<4} {r['suite']:<9} {r['run']} step {r['step']:<6} {json.dumps(r['metrics'])[:160]}")
        return 0
    run(args.checkpoint, args.suites.split(","), args.store, args.data, args.sft, n_repair=args.n_repair,
        n_protocol=args.n_protocol, repair_seed=args.repair_seed)
    return 0


def cmd_model(args) -> int:
    from .bench.runner import BenchStore
    from .core.checkpoint import resolve_checkpoint, verify_checkpoint

    active = Path(args.base) / "runs" / "ACTIVE.json"
    if args.action == "active":
        print(active.read_text() if active.exists() else "no promoted model (studio loads the newest checkpoint)")
        return 0
    ckpt = resolve_checkpoint(args.checkpoint)
    manifest = verify_checkpoint(ckpt)
    results = BenchStore(args.store).list(str(ckpt))
    if not results and not args.force_unbenchmarked:
        print("refusing to promote: no benchmark results recorded for this checkpoint (run `infinity bench run` first)", file=sys.stderr)
        return 1
    if not args.reason:
        print("--reason is required", file=sys.stderr)
        return 2
    record = {"checkpoint": str(ckpt), "step": manifest["step"], "model_sha256": manifest["sha256"]["model.pt"],
              "promoted_at": time.time(), "reason": args.reason, "benchmark_result_ids": [r["id"] for r in results]}
    if active.exists():
        hist = active.with_name("ACTIVE_HISTORY.jsonl")
        with open(hist, "a") as f:
            f.write(active.read_text().strip() + "\n")
    active.write_text(json.dumps(record, indent=2))
    print(f"promoted {ckpt} (previous promotion kept in runs/ACTIVE_HISTORY.jsonl)")
    return 0


def cmd_studio(args) -> int:
    from .server.app import main as studio_main

    return studio_main(args.rest)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="infinity", description="INFINITY BOT · INFINITY HYPER 5.5")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit", help="hardware/software capability report").set_defaults(fn=cmd_audit)

    d = sub.add_parser("data", help="dataset tools")
    dsub = d.add_subparsers(dest="data_cmd", required=True)
    dp = dsub.add_parser("prepare", help="filter, dedupe, split, train tokenizer, tokenize")
    dp.add_argument("--manifest", default="data/manifest.json")
    dp.add_argument("--out", default="runs/data")
    dp.add_argument("--vocab", type=int, default=4096)
    dp.add_argument("--seed", type=int, default=1337)
    dp.set_defaults(fn=cmd_data_prepare)

    t = sub.add_parser("train", help="pretrain INFINITY HYPER 5.5 from scratch")
    t.add_argument("--config", default="configs/hyper-5.5-tiny.json")
    t.add_argument("--data", default="runs/data")
    t.add_argument("--out", default=None)
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--no-resume", action="store_true")
    t.set_defaults(fn=cmd_train)

    g = sub.add_parser("generate", help="generate text from a checkpoint")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=120)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--repetition-penalty", type=float, default=1.1)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--device", default="cpu")
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("checkpoints", help="list and verify checkpoints")
    c.add_argument("--run", required=True)
    c.set_defaults(fn=cmd_checkpoints)

    a = sub.add_parser("agent", help="run INFINITY BOT on a task")
    a.add_argument("--workspace", default=".")
    a.add_argument("--model", required=True, help="neural:<checkpoint|run dir> or scripted:<file.json>")
    a.add_argument("--mode", default="code", choices=["ask", "plan", "review", "code", "debug", "build", "autonomous"])
    a.add_argument("--permission", default=None, choices=["read-only", "ask", "auto"])
    a.add_argument("--task", required=True)
    a.add_argument("--max-steps", type=int, default=25)
    a.add_argument("--max-seconds", type=float, default=900)
    a.add_argument("--command-timeout", type=float, default=120)
    a.add_argument("--db", default=None, help="memory database path (default ~/.infinity/infinity.db)")
    a.add_argument("--no-memory", action="store_true")
    a.add_argument("--json", default=None, help="write the task report as JSON")
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(fn=cmd_agent)

    r = sub.add_parser("rollback", help="restore files changed by an agent session")
    r.add_argument("--workspace", default=".")
    r.add_argument("--session", default=None)
    r.add_argument("--yes", action="store_true")
    r.set_defaults(fn=cmd_rollback)

    m = sub.add_parser("memory", help="inspect / export / delete stored memory")
    m.add_argument("action", choices=["show", "export", "delete-session", "delete-note", "set-preference", "trust", "untrust", "wipe"])
    m.add_argument("target", nargs="?", default="")
    m.add_argument("value", nargs="?", default="")
    m.add_argument("--db", default=None)
    m.set_defaults(fn=cmd_memory)
    sf = sub.add_parser("sft", help="tool-use fine-tuning: generate data with the real runtime, then fine-tune")
    sf.add_argument("action", choices=["generate", "train"])
    sf.add_argument("--tokenizer", default="runs/data/tokenizer.json")
    sf.add_argument("--out", default="runs/sft")
    sf.add_argument("--context", type=int, default=512)
    sf.add_argument("--train", type=int, default=2000)
    sf.add_argument("--eval", type=int, default=100)
    sf.add_argument("--seed", type=int, default=1337)
    sf.add_argument("--base", default="runs/hyper-5.5-tiny")
    sf.add_argument("--data", default="runs/sft")
    sf.add_argument("--steps", type=int, default=1200)
    sf.add_argument("--batch-size", type=int, default=16)
    sf.add_argument("--lr", type=float, default=1e-3)
    sf.add_argument("--lm-mix", type=float, default=0.25, help="fraction of steps that also train on pretraining text (limits forgetting)")
    sf.add_argument("--lm-data", default="runs/data")
    sf.add_argument("--eval-every", type=int, default=250)
    sf.set_defaults(fn=cmd_sft)

    be = sub.add_parser("bench", help="run / list recorded benchmarks")
    be.add_argument("action", choices=["run", "list"])
    be.add_argument("--checkpoint", default=None)
    be.add_argument("--suites", default="lm,protocol,repair")
    be.add_argument("--store", default="runs/benchmarks.db")
    be.add_argument("--data", default="runs/data")
    be.add_argument("--sft", default="runs/sft")
    be.add_argument("--n-repair", type=int, default=30)
    be.add_argument("--n-protocol", type=int, default=200)
    be.add_argument("--repair-seed", type=int, default=1337, help="1337 = reported test set; use another seed for checkpoint selection")
    be.set_defaults(fn=cmd_bench)

    mo = sub.add_parser("model", help="promote a benchmarked checkpoint (explicit, recorded)")
    mo.add_argument("action", choices=["promote", "active"])
    mo.add_argument("--checkpoint", default=None)
    mo.add_argument("--reason", default=None)
    mo.add_argument("--store", default="runs/benchmarks.db")
    mo.add_argument("--base", default=".")
    mo.add_argument("--force-unbenchmarked", action="store_true")
    mo.set_defaults(fn=cmd_model)

    ad = sub.add_parser("admin", help="owner account and platform administration")
    ad.add_argument("action", choices=["create-owner", "users", "grant-plan", "audit-verify"])
    ad.add_argument("--email", default=None)
    ad.add_argument("--plan", default="plus", choices=["plus", "pro", "max"])
    ad.add_argument("--reason", default=None)
    ad.add_argument("--platform-db", default=None)
    ad.set_defaults(fn=cmd_admin)

    st = sub.add_parser("studio", help="start the INFINITY STUDIO dashboard (local only)", add_help=False)
    st.add_argument("rest", nargs=argparse.REMAINDER)
    st.set_defaults(fn=cmd_studio)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "studio":  # forward all options to the studio's own parser
        from .server.app import main as studio_main

        return studio_main(argv[1:])
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
