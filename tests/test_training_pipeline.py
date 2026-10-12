"""End-to-end: prepare data -> train -> checkpoint -> resume -> reload -> generate."""

import json
import threading

import pytest

torch = pytest.importorskip("torch")

from infinity.core import data as data_mod
from infinity.core.config import ModelConfig, RunConfig, TrainConfig
from infinity.core.inference import InferenceEngine, SamplingConfig
from infinity.core.trainer import Trainer, lr_at


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "corpus").mkdir(parents=True)
    for i in range(12):
        (root / "data" / "corpus" / f"doc{i}.md").write_text(
            f"def function_{i}(x):\n    return x * {i} + {i}\n\n" * 30 + f"Document {i} explains function_{i}.\n"
        )
    (root / "data" / "corpus" / "secret.md").write_text("config\npassword = 'hunter2hunter2'\n" * 5)
    (root / "data" / "corpus" / "dup.md").write_text((root / "data" / "corpus" / "doc0.md").read_text())
    (root / "data" / "bad").mkdir()
    (root / "data" / "bad" / "x.md").write_text("proprietary text " * 50)
    manifest = {
        "sources": [
            {"name": "orig", "license": "Project-Original", "include": ["data/corpus/*.md"]},
            {"name": "bad", "license": "Proprietary", "include": ["data/bad/*.md"]},
        ]
    }
    (root / "data" / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_prepare_filters_and_records_provenance(corpus, tmp_path):
    out = tmp_path / "prepared"
    summary = data_mod.prepare(corpus / "data" / "manifest.json", out, vocab_size=320, seed=1, verbose=False)
    assert summary["cleaning"]["secret_dropped"] == 1
    assert summary["cleaning"]["duplicate_dropped"] == 1
    assert summary["provenance"]["rejected"][0]["name"] == "bad"
    files = json.loads((out / "files.json").read_text())
    all_paths = [f["path"] for split in files.values() for f in split]
    assert len(all_paths) == len(set(all_paths))  # no file in two splits
    assert not any("secret" in p or "/bad/" in p for p in all_paths)
    assert summary["splits"]["train"]["tokens"] > 0


def test_train_resume_reload_generate(corpus, tmp_path):
    out = tmp_path / "prepared"
    data_mod.prepare(corpus / "data" / "manifest.json", out, vocab_size=320, seed=1, verbose=False)
    run = RunConfig(
        name="test-tiny",
        model=ModelConfig(vocab_size=320, context_length=32, d_model=32, n_layers=2, n_heads=4, d_ff=64),
        train=TrainConfig(batch_size=8, max_steps=40, warmup_steps=5, eval_interval=20, eval_batches=2,
                          checkpoint_interval=20, learning_rate=3e-3),
        output_dir=str(tmp_path / "run"),
    )
    logs = []
    r1 = Trainer(run, out, log=logs.append).train(max_steps=20)
    assert r1["final_step"] == 20 and r1["start_step"] == 0
    r2 = Trainer(run, out, log=logs.append).train(max_steps=40)
    assert r2["start_step"] == 20 and r2["final_step"] == 40
    assert any("resumed from" in l for l in logs)

    records = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == list(range(1, 41))
    assert records[-1]["loss"] < records[0]["loss"]  # it actually learned something

    eng = InferenceEngine(str(tmp_path / "run"))
    assert eng.info()["step"] == 40
    a = eng.generate("def function_3(x):\n", SamplingConfig(max_new_tokens=20, seed=7))
    b = eng.generate("def function_3(x):\n", SamplingConfig(max_new_tokens=20, seed=7))
    assert a == b and len(a) > 0
    greedy = eng.generate("def ", SamplingConfig(max_new_tokens=60, temperature=0))
    assert isinstance(greedy, str)


def test_interrupt_saves_resumable_checkpoint(corpus, tmp_path):
    out = tmp_path / "prepared"
    data_mod.prepare(corpus / "data" / "manifest.json", out, vocab_size=320, seed=1, verbose=False)
    run = RunConfig(
        name="t",
        model=ModelConfig(vocab_size=320, context_length=32, d_model=32, n_layers=1, n_heads=4, d_ff=64),
        train=TrainConfig(batch_size=4, max_steps=1000, checkpoint_interval=1000, eval_interval=1000),
        output_dir=str(tmp_path / "run"),
    )
    stop = threading.Event()
    calls = []

    def log(msg):
        calls.append(msg)
        if "step" in msg and "10 " in msg:
            stop.set()

    res = Trainer(run, out, log=log).train(stop_flag=stop)
    assert res["interrupted"] and res["final_step"] < 1000
    assert (tmp_path / "run" / "checkpoints" / f"step_{res['final_step']:06d}" / "manifest.json").exists()


def test_lr_schedule_warmup_and_decay():
    tc = TrainConfig(learning_rate=1.0, warmup_steps=10, max_steps=110, min_lr_ratio=0.1)
    assert lr_at(0, tc) == pytest.approx(0.1)
    assert lr_at(9, tc) == pytest.approx(1.0)
    assert lr_at(110, tc) == pytest.approx(0.1)
    assert lr_at(60, tc) == pytest.approx(0.55)
