"""Open-model pipeline: data, LoRA training, evaluation, sandbox and bot integration.

All tests use a tiny randomly initialised Qwen2-architecture model built offline
(``infinity.hf.tiny``). They prove the plumbing works, not that any model is good.
"""

import json
import shutil
import threading
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

from infinity.agent.models import Message, ScriptedModel, load_model
from infinity.hf.chat import TOOL_PREFIX, system_prompt_for, to_hf_messages
from infinity.hf.config import ConfigError, load_config
from infinity.hf.data import dedupe, encode_example, load_jsonl, prompt_key, remove_leaks, validate_example
from infinity.hf.evaluate import compare, run_suite
from infinity.hf.model import HFModel
from infinity.hf.sandbox import extract_code, run_tests
from infinity.hf.tiny import build_tiny_model


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return build_tiny_model(tmp_path_factory.mktemp("tiny") / "model")


@pytest.fixture(scope="module")
def tok(tiny):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny)


def ex(*turns, domain="chat", **kw):
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(turns)]
    return {"domain": domain, "license": "Project-Original", "messages": msgs, **kw}


# ------------------------------------------------------------------ data
def test_validation_catches_malformed_examples():
    assert validate_example(ex("hi", "hello")) == []
    assert any("license" in e for e in validate_example({**ex("hi", "hello"), "license": ""}))
    assert any("domain" in e for e in validate_example({**ex("hi", "hello"), "domain": "poetry"}))
    assert any("empty" in e for e in validate_example(ex("hi", "   ")))
    assert any("last message" in e for e in validate_example(ex("hi")))
    assert any("secret" in e for e in validate_example(ex("my key", "AKIAIOSFODNN7EXAMPLE is it")))
    bad = {"domain": "chat", "license": "x", "messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "b"},
                                                          {"role": "assistant", "content": "c"}]}
    assert any("system message must come first" in e for e in validate_example(bad))
    assert validate_example("nope") == ["not a JSON object"]


def test_load_dedupe_and_leak_removal(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join([json.dumps(ex("Hi", "Hello")), "{broken", json.dumps(ex("hi ", "hello")),
                            json.dumps(ex("What is 2+2?", "4"))]) + "\n")
    res = load_jsonl(f)
    assert len(res.examples) == 3 and res.failures[0]["line"] == 2
    kept, dups = dedupe(res.examples)
    assert dups == 1 and len(kept) == 2
    kept, leaks = remove_leaks(kept, {prompt_key(ex("what is 2+2?", "x"))})
    assert leaks == 1 and len(kept) == 1


def test_tool_role_maps_to_prefixed_user_turn():
    out = to_hf_messages([Message("system", "s"), Message("tool", "exit 0")])
    assert out[1] == {"role": "user", "content": TOOL_PREFIX + "exit 0"}
    with pytest.raises(ValueError):
        to_hf_messages([{"role": "robot", "content": "x"}])


def test_encoding_trains_only_on_assistant_tokens(tok):
    e = ex("hi", "Hello there!", "bye", "Goodbye.")
    ids, labels = encode_example(tok, e, 512)
    assert tok.decode([i for i, l in zip(ids, labels) if l != -100]) == "Hello there!<|im_end|>\nGoodbye.<|im_end|>\n"
    ids, labels = encode_example(tok, {**e, "loss_on": "last_assistant"}, 512)
    assert tok.decode([i for i, l in zip(ids, labels) if l != -100]) == "Goodbye.<|im_end|>\n"
    assert encode_example(tok, e, 5) is None  # too long -> dropped, never truncated silently


def test_open_model_identity_prompt_never_claims_hyper():
    p = system_prompt_for({"family": "open-model", "product": "Qwen2.5-Coder 0.5B Instruct"})
    assert "Qwen2.5-Coder 0.5B Instruct" in p and "not INFINITY HYPER 5.5" in p
    assert "You are INFINITY HYPER 5.5" in system_prompt_for({"product": "INFINITY HYPER 5.5"})


# ------------------------------------------------------- config + pipeline
def _project(tmp_path, tiny):
    root = tmp_path / "proj"
    shutil.copytree("training/configs", root / "training/configs")
    for d in ("chat", "coding", "validation"):
        shutil.copytree(f"datasets/{d}", root / "datasets" / d)
    (root / "training/evaluation").mkdir(parents=True)
    shutil.copy("training/evaluation/test_cases.jsonl", root / "training/evaluation/test_cases.jsonl")
    shutil.copytree(tiny, root / "training/tiny-model")
    return root


def test_config_inheritance_and_validation(tmp_path):
    cfg = load_config("training/configs/coding.yaml")
    assert cfg["data"]["train_domains"] == ["coding", "tool"] and cfg["lora"]["r"] == 16
    bad = tmp_path / "bad.yaml"
    default = Path("training/configs/default.yaml").resolve()
    bad.write_text(f"extends: {default.as_posix()}\ntraining: {{max_steps: 0}}\n")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_prepare_train_resume_and_export(tmp_path, tiny, tok):
    from infinity.hf.prepare import prepare
    from infinity.hf.train import plan, train

    root = _project(tmp_path, tiny)
    # A training example whose prompt is also a held-out test case must be removed.
    leak = ex("What is the capital of France? Answer in one word.", "Paris")
    (root / "datasets/chat/leak.jsonl").write_text(json.dumps(leak) + "\n")
    cfg = load_config(root / "training/configs/tiny-test.yaml",
                      {"data": {"tool_tasks_train": 2, "tool_tasks_val": 1},
                       "training": {"max_steps": 2, "checkpoint_interval": 1, "eval_interval": 1, "eval_examples": 2}})
    rep = prepare(cfg, root, tok, log=lambda m: None)
    assert rep["leaks_removed_from_train"] == 1
    assert rep["train"]["by_domain"]["tool"] > 0 and rep["val"]["examples"] > 0
    assert rep["tool_transcripts"]["train"]["kept"] == 2
    assert (root / "training/data-tiny/report.md").exists()
    train_rows = [json.loads(l) for l in (root / "training/data-tiny/train.jsonl").read_text().splitlines()]
    chat_rows = [r for r in train_rows if r["domain"] == "chat"]
    assert all(r["messages"][0]["role"] == "system" and "— Fine-Tuned" in r["messages"][0]["content"] for r in chat_rows)

    assert plan(cfg, root, tok, log=lambda m: None)["train_examples"] == rep["train"]["examples"]
    smoke = train(cfg, root, smoke=True, log=lambda m: None)
    assert smoke["reload_max_logit_diff"] < 1e-3 and smoke["adapter_effect_max_logit"] > 0
    assert not (root / "adapters").exists()  # smoke runs never write into the project

    first = train(cfg, root, log=lambda m: None)
    ck = root / "training/checkpoints/tiny-test"
    assert (ck / "LATEST").read_text() == "step_000002"
    cfg["training"]["max_steps"] = 3
    second = train(cfg, root, resume=True, log=lambda m: None)
    assert second["steps"] == 3 and (ck / "step_000003").exists() and (ck / "step_000001").exists()
    assert first["adapter"] != second["adapter"]  # an existing export is never overwritten
    manifest = json.loads((root / "adapters/open-models/tiny-test/training_manifest.json").read_text())
    assert "NOT INFINITY HYPER 5.5" in manifest["note"]
    log = [json.loads(l) for l in (root / "training/logs/tiny-test.jsonl").read_text().splitlines()]
    steps = [r for r in log if "event" not in r]
    assert [r["step"] for r in steps] == [1, 2, 3] and all("loss" in r and "lr" in r for r in steps)
    starts = [r for r in log if r.get("event") == "start"]
    assert starts and starts[0]["base_role"].startswith("original") and starts[0]["output"].endswith("— Fine-Tuned")
    assert starts[0]["base_files_sha256"] and all(r["base_unchanged_after_training"] for r in log if r.get("event") == "end")
    assert manifest["merged_into_base"] is False and manifest["base_unchanged_after_training"] is True
    assert second["base_unchanged"] and second["name"] == "Tiny test model (random weights) — Fine-Tuned"

    # The trained adapter loads through the same path the bot uses.
    m = HFModel(str(root / "training/tiny-model"), "adapters/open-models/tiny-test", label="Tiny",
                context_tokens=512, base_dir=root, temperature=0)
    assert m.info()["product"] == "Tiny — Fine-Tuned" and m.info()["variant"] == "fine-tuned"
    from peft import PeftModel

    assert isinstance(m.model, PeftModel) and m.info()["adapter_merged"] is False  # adapter kept separate


def test_training_fails_if_base_files_change(tmp_path, tiny, tok, monkeypatch):
    from infinity.hf import train as T
    from infinity.hf.prepare import prepare

    root = _project(tmp_path, tiny)
    cfg = load_config(root / "training/configs/tiny-test.yaml", {"data": {"tool_tasks_train": 0, "train_domains": ["chat"]},
                                                                 "training": {"max_steps": 1}})
    prepare(cfg, root, tok, log=lambda m: None)
    calls = iter([{"model.safetensors": "aaa"}, {"model.safetensors": "bbb"}])
    monkeypatch.setattr(T, "fingerprint_files", lambda folder: next(calls))
    with pytest.raises(RuntimeError, match="ORIGINAL BASE MODEL FILES CHANGED"):
        T.train(cfg, root, log=lambda m: None)
    assert not (root / "adapters").exists()  # nothing exported from a run that touched the base


def test_oom_is_reported_without_deleting_checkpoints(tmp_path, tiny, tok, monkeypatch):
    import torch

    from infinity.hf import train as T
    from infinity.hf.prepare import prepare

    root = _project(tmp_path, tiny)
    cfg = load_config(root / "training/configs/tiny-test.yaml", {"data": {"tool_tasks_train": 0, "train_domains": ["chat"]},
                                                                 "training": {"max_steps": 1}})
    prepare(cfg, root, tok, log=lambda m: None)
    T.train(cfg, root, log=lambda m: None)
    before = sorted(p.name for p in (root / "training/checkpoints/tiny-test").iterdir())

    def boom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", boom)
    cfg["training"]["max_steps"] = 2
    with pytest.raises(T.TrainingOOM, match="Nothing was deleted"):
        T.train(cfg, root, resume=True, log=lambda m: None)
    assert sorted(p.name for p in (root / "training/checkpoints/tiny-test").iterdir()) == before


# ---------------------------------------------------------------- model
def test_hf_model_adapter_generation_cancel_and_identity(tiny):
    m = load_model(f"hf:{tiny}")
    info = m.info()
    assert info["family"] == "open-model" and "not INFINITY HYPER 5.5" in info["implementation"]
    m.temperature = 0
    c = m.complete([Message("system", "s"), Message("user", "hi")], max_new_tokens=8)
    assert c.tokens_exact and c.tokens_out <= 8 and c.tokens_in > 0
    stop = threading.Event()
    stop.set()
    assert m.complete([Message("user", "hi")], max_new_tokens=50, cancel=stop).tokens_out <= 1


def test_broken_adapter_is_reported(tiny, tmp_path):
    from infinity.agent.models import ModelUnavailable

    (tmp_path / "bad").mkdir()
    with pytest.raises(ModelUnavailable, match="adapter"):
        HFModel(str(tiny), str(tmp_path / "bad"))


# ------------------------------------------------------------- sandbox
def test_sandbox_runs_tests_and_blocks_escapes(tmp_path):
    assert run_tests("def f(x): return 2*x", "assert f(2) == 4")["passed"]
    assert not run_tests("def f(x): return x", "assert f(2) == 4")["passed"]
    assert "timeout" in run_tests("while True: pass", "", timeout=2)["error"]
    assert "blocked" in run_tests("import subprocess; subprocess.run(['echo'])", "")["error"]
    assert "blocked" in run_tests("import socket; socket.getaddrinfo('example.com', 80)", "")["error"]
    outside = tmp_path / "escape.txt"
    assert "blocked" in run_tests(f"open({str(outside)!r}, 'w').write('x')", "")["error"]
    assert not outside.exists()
    assert extract_code("x\n```python\nA = 1\n```\n```js\nB\n```") == "A = 1\n"


def test_evaluation_scores_text_and_code(tmp_path):
    cases = [
        {"id": "f", "suite": "factual", "prompt": "capital of France?", "any": [r"\bparis\b"]},
        {"id": "y", "suite": "instruction", "prompt": "say YES", "fullmatch": r"\W*yes\W*"},
        {"id": "c", "suite": "coding", "prompt": "double", "tests": "assert double(3) == 6"},
        {"id": "c2", "suite": "coding", "prompt": "triple", "tests": "assert triple(3) == 9"},
    ]
    answers = iter(["It is Paris.", "Yes, sure!", "```python\ndef double(x):\n    return 2 * x\n```", "def triple(x): return x"])
    r = run_suite(ScriptedModel(lambda msgs: next(answers)), cases, log=lambda m: None)
    assert [c["passed"] for c in r["cases"]] == [True, False, True, False]
    assert r["overall"] == {"passed": 2, "total": 4, "rate": 0.5}
    assert r["cases"][0]["latency_s"] >= 0 and "tokens_per_s" in r
    better = json.loads(json.dumps(r))
    better["cases"][1]["passed"] = True
    better["cases"][0]["passed"] = False
    md = compare(r, better, "base", "tuned")
    assert "Regressions (1): f" in md and "Newly passing (1): y" in md


# ------------------------------------------------------------- server
def test_server_open_model_switch_fallback_and_chat(tmp_path, tiny):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from infinity.agent import Memory
    from infinity.platform import Platform
    from infinity.server.app import create_app
    from infinity.server.runtime import RuntimeHub

    base = tmp_path / "base"
    (base / "configs").mkdir(parents=True)
    shutil.copytree(tiny, base / "models/tiny")
    (base / "configs/open-models.json").write_text(json.dumps({"models": [
        {"id": "tiny", "label": "Tiny Open Model", "base": "models/tiny", "adapter": None, "context_tokens": 512},
        {"id": "tiny-broken", "label": "Tiny Open Model", "base": "models/tiny", "adapter": "models/missing-adapter",
         "context_tokens": 512},
    ]}))
    plat = Platform(tmp_path / "p.db")
    hub = RuntimeHub(base, Memory(tmp_path / "m.db"))
    hub.set_model(ScriptedModel(["hello from scripted"]), None)
    app = create_app(hub, plat)
    pw = "correct horse battery"
    plat.auth.create_user("owner@example.com", pw, "owner")
    plat.auth.create_user("free@example.com", pw)

    def client(email):
        c = TestClient(app)
        assert c.post("/api/auth/login", json={"email": email, "password": pw}).status_code == 200
        c.headers["x-infinity-token"] = c.get("/api/auth/me").json()["csrf"]
        return c

    owner, free = client("owner@example.com"), client("free@example.com")
    listing = owner.get("/api/models").json()["open_models"]
    assert {e["id"]: e["adapter_present"] for e in listing} == {"tiny": None, "tiny-broken": False}

    assert free.post("/api/model", json={"open_model": "tiny"}).status_code == 403
    assert owner.post("/api/model", json={"open_model": "nope"}).status_code == 404
    assert owner.post("/api/model", json={}).status_code == 422

    # A broken adapter keeps the current model and reports why.
    r = owner.post("/api/model", json={"open_model": "tiny-broken"})
    assert r.status_code == 409 and "kept the current model" in r.json()["detail"]
    st = owner.get("/api/status").json()["model"]
    assert st["implementation"].startswith("scripted") and "switch_error" in st

    r = owner.post("/api/model", json={"open_model": "tiny"})
    assert r.status_code == 200, r.text
    assert r.json()["product"] == "Tiny Open Model" and r.json()["family"] == "open-model"
    assert json.loads((base / "runs/SELECTED_MODEL.json").read_text()) == {"kind": "open", "id": "tiny"}

    # Normal chat works for a free account, is metered, and is not charged.
    r = free.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "max_new_tokens": 16})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"]["product"] == "Tiny Open Model" and body["usage"]["credits_charged"] == 0
    assert body["usage"]["tokens_in"] > 0
    usage = plat.billing.db.all("SELECT source FROM usage_events")
    assert [u["source"] for u in usage] == ["chat"]
    assert free.post("/api/chat", json={"messages": [{"role": "assistant", "content": "x"}]}).status_code == 422
    c = TestClient(app)
    assert c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code in (401, 403)

    # A restart restores the owner's selection.
    hub2 = RuntimeHub(base, Memory(tmp_path / "m2.db"))
    hub2.autoload_latest()
    assert hub2.model_status()["family"] == "open-model"
