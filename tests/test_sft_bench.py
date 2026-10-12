import json
import random

import pytest

torch = pytest.importorskip("torch")

from infinity.bench.runner import BenchStore, bench_protocol, bench_repair
from infinity.agent.models import NeuralCoreModel
from infinity.core.checkpoint import save_checkpoint
from infinity.core.config import ModelConfig, RunConfig
from infinity.core.model import InfinityNeuralCore
from infinity.core.tokenizer import BPETokenizer
from infinity.sft.finetune import encode_example, finetune
from infinity.sft.generate import generate
from infinity.sft.tasks import SYLLABLES, expert_actions, make_tasks


@pytest.fixture(scope="module")
def tok():
    corpus = ['{"thought":"Run the tests.","tool":"run_command","args":{"command":"python3 -m unittest -q"}}',
              "def add_rx(a, b):\n    return a + b\n", "Traceback (most recent call last):\nAssertionError: 1 != 2\n"] * 30
    return BPETokenizer.train(corpus, vocab_size=400)


@pytest.fixture(scope="module")
def base_ckpt(tok, tmp_path_factory):
    d = tmp_path_factory.mktemp("base")
    cfg = RunConfig(name="tiny", model=ModelConfig(vocab_size=tok.vocab_size, context_length=128, d_model=32, n_layers=1, n_heads=4, d_ff=64))
    torch.manual_seed(0)
    return save_checkpoint(d / "checkpoints", 10, InfinityNeuralCore(cfg.model), None, cfg, tok, {"loss": 1.0})


def test_task_splits_are_disjoint_and_deterministic():
    assert not set(SYLLABLES["train"]) & set(SYLLABLES["eval"])
    a = make_tasks("train", 20, 1)
    assert [t.files for t in a] == [t.files for t in make_tasks("train", 20, 1)]
    ev = make_tasks("eval", 20, 1)
    train_fns = {t.function for t in make_tasks("train", 300, 1)}
    assert not train_fns & {t.function for t in ev}
    assert {t.family for t in make_tasks("train", 60, 3)} == {"typo", "operator", "compare"}


def test_expert_transcripts_verify_through_real_runtime(tok, tmp_path):
    s = generate(tmp_path / "sft", tok, 256, 6, 3, log=lambda m: None)
    assert s["splits"]["train"]["kept"] == 6 and s["splits"]["train"]["rejected"] == 0
    rows = [json.loads(l) for l in open(tmp_path / "sft" / "train.jsonl")]
    assert [r["tool"] for r in rows[:6]] == ["list_dir", "run_command", "read_file", "edit_file", "run_command", "final"]
    first = rows[0]["prompt"]
    assert first.startswith("<|system|>INFINITY HYPER 5.5 agent.") and first.endswith("<|assistant|>")
    # step 3's prompt contains the REAL tool output of the failing test run
    assert "FAILED" in rows[2]["prompt"] or "Error" in rows[2]["prompt"]
    # train names are generated: many distinct modules, so copying (not recall) is required
    assert len({t.module_file for t in make_tasks("train", 200, 1)}) > 150


def test_masked_encoding_mirrors_inference_truncation(tok):
    enc = encode_example(tok, "<|user|>" + "x " * 500 + "<|end|><|assistant|>", '{"thought":"t","tool":"list_dir","args":{}}<|end|>', 128)
    ids, labels = enc
    assert len(ids) <= 128
    n_target = sum(1 for l in labels if l != -100)
    assert labels[:-n_target] == [-100] * (len(ids) - n_target)
    assert ids[-1] == tok.token_id("<|end|>")
    assert encode_example(tok, "<|user|>hi<|end|><|assistant|>", "y" * 2000, 128) is None  # completion too long


def test_finetune_creates_new_run_with_lineage(tok, base_ckpt, tmp_path):
    generate(tmp_path / "sft", tok, 192, 8, 4, log=lambda m: None)
    r = finetune(str(base_ckpt), tmp_path / "sft", tmp_path / "out", steps=30, context=192, batch_size=8, lr=3e-3, eval_every=15, log=lambda m: None)
    assert (tmp_path / "out" / "checkpoints" / "step_000015").exists()  # intermediate checkpoint for selection
    assert r["eval_loss_after"] < r["eval_loss_before"]
    m = NeuralCoreModel(r["checkpoint"])
    info = m.info()
    assert info["context_length"] == 192
    assert info["metrics"]["lineage"]["parent_checkpoint"] == str(base_ckpt)
    assert (base_ckpt / "manifest.json").exists()  # parent untouched
    with pytest.raises(FileExistsError):
        finetune(str(base_ckpt), tmp_path / "sft", tmp_path / "out", steps=1, context=192, log=lambda m: None)


def test_benchmarks_record_real_results(tok, base_ckpt, tmp_path):
    generate(tmp_path / "sft", tok, 128, 2, 3, log=lambda m: None)
    model = NeuralCoreModel(str(base_ckpt), temperature=0.0)
    metrics, cfg = bench_protocol(model, tmp_path / "sft" / "eval.jsonl", limit=5)
    assert metrics["examples"] == 5 and 0 <= metrics["valid_action_rate"] <= 1
    rep, _ = bench_repair(model, n=2, max_steps=3)
    assert rep["tasks"] == 2 and sum(rep["outcomes"].values()) == 2
    assert rep["verified_success_rate"] == 0.0  # an untrained model cannot fix anything — and is not reported as if it did
    store = BenchStore(tmp_path / "b.db")
    rid = store.add("protocol", {"checkpoint": str(base_ckpt), "run": "tiny", "step": 10, "model_sha256": "x"}, cfg, metrics, 1.0)
    assert store.list(str(base_ckpt))[0]["id"] == rid
