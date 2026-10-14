"""LoRA adapters on the original INFINITY HYPER 5.5 model: preservation guarantees."""

import json
import shutil

import pytest
import torch

from infinity.agent.models import NeuralCoreModel, load_model
from infinity.core.checkpoint import save_checkpoint
from infinity.core.config import ModelConfig, RunConfig
from infinity.core.lora import (
    FINETUNED_NAME, AdapterError, LoRAConfig, adapter_state, apply_lora, base_fingerprint, load_with_adapter,
    verify_unchanged,
)
from infinity.core.model import InfinityNeuralCore
from infinity.core.tokenizer import BPETokenizer
from infinity.sft.generate import generate
from infinity.sft.lora_train import lora_finetune


@pytest.fixture(scope="module")
def tok():
    corpus = ['{"thought":"Run the tests.","tool":"run_command","args":{"command":"python3 -m unittest -q"}}',
              "def add_rx(a, b):\n    return a + b\n", "Traceback (most recent call last):\nAssertionError: 1 != 2\n"] * 30
    return BPETokenizer.train(corpus, vocab_size=400)


@pytest.fixture(scope="module")
def base_ckpt(tok, tmp_path_factory):
    d = tmp_path_factory.mktemp("base")
    cfg = RunConfig(name="tiny", model=ModelConfig(vocab_size=tok.vocab_size, context_length=128, d_model=32, n_layers=1,
                                                   n_heads=4, d_ff=64))
    torch.manual_seed(0)
    return save_checkpoint(d / "checkpoints", 10, InfinityNeuralCore(cfg.model), None, cfg, tok, {"loss": 1.0})


@pytest.fixture(scope="module")
def sft(tok, tmp_path_factory):
    d = tmp_path_factory.mktemp("sft")
    generate(d, tok, 192, 8, 4, log=lambda m: None)
    return d


def test_fresh_adapter_is_identity_and_freezes_base():
    torch.manual_seed(0)
    m = InfinityNeuralCore(ModelConfig(vocab_size=50, context_length=16, d_model=16, n_layers=1, n_heads=2, d_ff=32))
    x = torch.randint(0, 50, (1, 8))
    want = m(x)[0]
    wrapped = apply_lora(m, LoRAConfig(r=4, alpha=8))
    assert len(wrapped) == 5  # qkv, proj, gate, up, down
    assert torch.allclose(m(x)[0], want)
    assert all(".lora_" in n for n, p in m.named_parameters() if p.requires_grad)
    assert set(adapter_state(m)) == {n for n, p in m.named_parameters() if p.requires_grad}


def test_lora_training_preserves_original_and_loads_on_top(tok, base_ckpt, sft, tmp_path):
    before = base_fingerprint(base_ckpt)
    r = lora_finetune(str(base_ckpt), sft, "hyper-ft", root=tmp_path, steps=30, context=192, batch_size=8, lr=1e-2,
                      eval_every=15, r=4, alpha=8, log=lambda m: None)
    assert r["base_unchanged"] and base_fingerprint(base_ckpt) == before  # original files bit-identical
    assert r["eval_loss_after"] < r["eval_loss_before"]
    ad = tmp_path / "adapters/infinity-hyper-5-5/hyper-ft"
    conf = json.loads((ad / "adapter_config.json").read_text())
    assert conf["name"] == FINETUNED_NAME and conf["base_model"] == "INFINITY HYPER 5.5"
    assert conf["merged_into_base"] is False and conf["base_unchanged_after_training"] is True
    assert sorted(p.name for p in ad.iterdir()) == ["adapter.pt", "adapter_config.json"]  # adapter only, no base weights
    assert (tmp_path / "training/checkpoints/hyper-ft/step_000015/adapter.pt").exists()
    log = [json.loads(l) for l in (tmp_path / "training/logs/hyper-ft.jsonl").read_text().splitlines()]
    assert log[0]["event"] == "start" and log[0]["base_role"].startswith("original") and log[0]["output"] == FINETUNED_NAME
    assert log[-1] == {**log[-1], "event": "end", "base_unchanged_after_training": True}

    original = NeuralCoreModel(str(base_ckpt))
    tuned = NeuralCoreModel(str(base_ckpt), adapter=str(ad))
    assert original.info()["product"] == "INFINITY HYPER 5.5" and original.info()["variant"] == "original"
    ti = tuned.info()
    assert ti["product"] == FINETUNED_NAME and ti["variant"] == "fine-tuned" and ti["adapter_name"] == "hyper-ft"
    assert "not merged" in ti["implementation"]
    x = torch.randint(0, tok.vocab_size, (1, 12))
    assert not torch.allclose(original.engine.model(x)[0], tuned.engine.model(x)[0])  # the adapter has an effect
    assert load_model(f"neural:{base_ckpt}::{ad}").info()["product"] == FINETUNED_NAME
    assert base_fingerprint(base_ckpt) == before  # loading never writes either

    # Retraining under the same name never overwrites the existing adapter.
    r2 = lora_finetune(str(base_ckpt), sft, "hyper-ft", root=tmp_path, steps=2, context=192, batch_size=8,
                       eval_every=50, r=4, alpha=8, log=lambda m: None)
    assert r2["adapter"] != str(ad) and (ad / "adapter.pt").exists()


def test_adapter_refuses_wrong_base_and_corruption(tok, base_ckpt, sft, tmp_path):
    lora_finetune(str(base_ckpt), sft, "a", root=tmp_path, steps=2, context=192, batch_size=8, eval_every=50, r=2,
                  alpha=4, log=lambda m: None)
    ad = tmp_path / "adapters/infinity-hyper-5-5/a"
    other_cfg = RunConfig(name="other", model=ModelConfig(vocab_size=tok.vocab_size, context_length=128, d_model=32,
                                                          n_layers=1, n_heads=4, d_ff=64))
    torch.manual_seed(99)
    other = save_checkpoint(tmp_path / "other", 1, InfinityNeuralCore(other_cfg.model), None, other_cfg, tok, {})
    with pytest.raises(AdapterError, match="different INFINITY HYPER 5.5 checkpoint"):
        load_with_adapter(other, ad)
    bad = tmp_path / "bad"
    shutil.copytree(ad, bad)
    sd = torch.load(bad / "adapter.pt", weights_only=True)
    torch.save({k: v + 1 for k, v in sd.items()}, bad / "adapter.pt")
    with pytest.raises(AdapterError, match="corrupted"):
        load_with_adapter(base_ckpt, bad)
    with pytest.raises(AdapterError, match="no adapter_config"):
        load_with_adapter(base_ckpt, tmp_path)


def test_verify_unchanged_detects_tampering(base_ckpt, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(base_ckpt, copy)
    before = base_fingerprint(copy)
    verify_unchanged(copy, before)
    with open(copy / "model.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="ORIGINAL MODEL FILES CHANGED"):
        verify_unchanged(copy, before)


def test_server_lists_switches_and_falls_back(tok, base_ckpt, sft, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from infinity.agent import Memory
    from infinity.platform import Platform
    from infinity.server.app import create_app
    from infinity.server.runtime import RuntimeHub

    base = tmp_path / "base"
    ck = base / "runs/hyper/checkpoints"
    shutil.copytree(base_ckpt.parent, ck)
    lora_finetune(str(ck / base_ckpt.name), sft, "ft", root=base, steps=2, context=192, batch_size=8, eval_every=50,
                  r=2, alpha=4, log=lambda m: None)
    broken = base / "adapters/infinity-hyper-5-5/broken"
    shutil.copytree(base / "adapters/infinity-hyper-5-5/ft", broken)
    (broken / "adapter.pt").write_bytes(b"garbage")
    before = base_fingerprint(ck / base_ckpt.name)

    plat = Platform(tmp_path / "p.db")
    hub = RuntimeHub(base, Memory(tmp_path / "m.db"))
    hub.autoload_latest()
    assert hub.model_status()["product"] == "INFINITY HYPER 5.5"
    app = create_app(hub, plat)
    plat.auth.create_user("owner@example.com", "correct horse battery", "owner")
    c = TestClient(app)
    c.post("/api/auth/login", json={"email": "owner@example.com", "password": "correct horse battery"})
    c.headers["x-infinity-token"] = c.get("/api/auth/me").json()["csrf"]

    listed = {a["name"]: a for a in c.get("/api/models").json()["hyper_adapters"]}
    assert set(listed) == {"ft", "broken"} and listed["ft"]["base_present"]
    r = c.post("/api/model", json={"hyper_adapter": "broken"})
    assert r.status_code == 409 and "kept the current model" in r.json()["detail"]
    assert c.get("/api/status").json()["model"]["product"] == "INFINITY HYPER 5.5"  # original still serving
    r = c.post("/api/model", json={"hyper_adapter": "ft"})
    assert r.status_code == 200 and r.json()["product"] == FINETUNED_NAME and r.json()["variant"] == "fine-tuned"
    assert c.post("/api/model", json={"hyper_adapter": "ft", "checkpoint": "x"}).status_code == 422
    # Rollback: the original stays selectable.
    orig = c.get("/api/models").json()["checkpoints"][0]["path"]
    assert c.post("/api/model", json={"checkpoint": orig}).json()["variant"] == "original"
    c.post("/api/model", json={"hyper_adapter": "ft"})
    hub2 = RuntimeHub(base, Memory(tmp_path / "m2.db"))
    hub2.autoload_latest()
    assert hub2.model_status()["product"] == FINETUNED_NAME  # selection survives a restart
    assert base_fingerprint(ck / base_ckpt.name) == before
