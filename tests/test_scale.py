"""Large-scale pipeline: sharded data, licence-filtered fetching, multi-process DDP/FSDP2 training."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from infinity.core.data_large import ShardedTokenDataset, canonical_license, fetch_hf_parquet, prepare_large
from infinity.core.inference import InferenceEngine
from infinity.sizing import plan
from infinity.core.config import RunConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def corpus(tmp_path):
    repo = tmp_path / "repo"
    (repo / "data" / "corpus").mkdir(parents=True)
    for i in range(40):
        (repo / "data" / "corpus" / f"d{i}.md").write_text(f"def fn_{i}(x):\n    return x * {i} + {i}\n\n" * 30)
    (repo / "data" / "corpus" / "dup.md").write_text((repo / "data" / "corpus" / "d0.md").read_text())
    (repo / "data" / "corpus" / "key.md").write_text("config\nAKIAABCDEFGHIJKLMNOP\n" * 5)
    rows = [{"id": f"r{i}", "license": lic, "text": f"// file {i}\nfunction f{i}() {{ return {i}; }}\n" * 5}
            for i, lic in enumerate(["mit", "apache-2.0", "gpl-3.0", "isc", "proprietary"] * 4)]
    (repo / "data" / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (repo / "data" / "manifest.json").write_text(json.dumps({"sources": [
        {"name": "files", "license": "Project-Original", "include": ["data/corpus/*.md"]},
        {"name": "rows", "type": "jsonl", "include": ["data/rows.jsonl"]}]}))
    return repo


def test_prepare_large_filters_dedupes_and_shards(corpus, tmp_path):
    s = prepare_large(corpus / "data" / "manifest.json", tmp_path / "prep", 400, val_frac=0.1, test_frac=0.1,
                      shard_bytes=3000, workers=2, log=lambda m: None)
    c = s["cleaning"]
    assert c["duplicate_dropped"] == 1 and c["secret_dropped"] == 1
    assert s["provenance"]["rows"]["documents"] == 12  # gpl-3.0 and proprietary rows rejected per record
    assert len(s["splits"]["train"]["shards"]) > 1 and s["splits"]["train"]["tokens"] > 0
    ds = ShardedTokenDataset(tmp_path / "prep", s["splits"]["train"], 32)
    x, y = ds.batch(8, torch.Generator().manual_seed(0))
    assert x.shape == (8, 32) and torch.equal(x[:, 1:], y[:, :-1])


def test_license_aliases():
    assert canonical_license("mit") == "MIT" and canonical_license("isc") == "ISC"
    assert canonical_license("gpl-3.0") is None and canonical_license("") is None


def test_fetch_hf_parquet_keeps_only_permissive_rows(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"code": ["a = 1\n", "b = 2\n", "c = 3\n", "d = 4\n"], "license": ["mit", "gpl-3.0", "apache-2.0", "mit"],
                      "language": ["Python", "Python", "Python", "Go"], "repo_name": ["r"] * 4, "path": ["a", "b", "c", "d"]})
    f = tmp_path / "x.parquet"
    pq.write_table(table, f)
    prov = fetch_hf_parquet("org/ds", tmp_path / "out", ["mit", "apache-2.0"], ["Python"], files=[str(f)], log=lambda m: None)
    assert prov["rows_kept"] == 2 and prov["licenses_kept"] == {"mit": 1, "apache-2.0": 1}
    rows = [json.loads(l) for l in open(tmp_path / "out" / "part_00000.jsonl")]
    assert {r["text"] for r in rows} == {"a = 1\n", "c = 3\n"}
    with pytest.raises(ValueError):
        fetch_hf_parquet("org/ds", tmp_path / "o2", ["gpl-3.0"], files=[str(f)], log=lambda m: None)


def test_sizing_7b():
    p = plan(RunConfig.load(ROOT / "configs" / "hyper-5.5-7b.json"))
    assert p.params == 7_110_660_096 and p.train_state_gb > 100 and p.min_gpus["A100-80GB"] >= 2


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_two_process_training_and_resume(corpus, tmp_path, strategy):
    prep = tmp_path / "prep"
    prepare_large(corpus / "data" / "manifest.json", prep, 400, val_frac=0.1, test_frac=0.1, shard_bytes=3000, workers=1, log=lambda m: None)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"name": "t", "model": {"vocab_size": 400, "context_length": 32, "d_model": 32, "n_layers": 2,
                                                      "n_heads": 4, "n_kv_heads": 2, "d_ff": 64, "grad_checkpoint": True},
                               "train": {"batch_size": 4, "max_steps": 4, "warmup_steps": 1, "eval_interval": 2, "eval_batches": 1,
                                         "checkpoint_interval": 2, "learning_rate": 0.003}}))
    out = tmp_path / "out"
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    port = "296" + ("1" if strategy == "ddp" else "2")
    for steps in (2, 4):
        r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", port + str(steps),
                            "-m", "infinity.core.dist_train", "--config", str(cfg), "--data", str(prep), "--out", str(out),
                            "--strategy", strategy, "--steps", str(steps)], capture_output=True, text=True, timeout=300, env=env, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
    assert "resumed from" in r.stdout
    eng = InferenceEngine(str(out))
    assert eng.info()["step"] == 4 and eng.manifest["metrics"]["world_size"] == 2


def test_kaggle_notebook_runs_end_to_end_in_smoke_mode(tmp_path, monkeypatch):
    """Executes every code cell of notebooks/kaggle_train_100m.ipynb with a tiny model and local data."""
    nb = json.loads((ROOT / "notebooks" / "kaggle_train_100m.ipynb").read_text())
    monkeypatch.setenv("INFINITY_NOTEBOOK_SMOKE", "1")
    monkeypatch.setenv("INFINITY_WORK", str(tmp_path))
    monkeypatch.setenv("INFINITY_REPO", str(ROOT))
    g: dict = {}
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        exec(compile("".join(cell["source"]), f"cell{i + 1}", "exec"), g)
    assert (tmp_path / "infinity-hyper-5.5-small.tar.gz").stat().st_size > 0
    assert (tmp_path / "runs" / "benchmarks.db").exists()


def test_notebook_is_built_from_source():
    sys.path.insert(0, str(ROOT / "tools"))
    import build_kaggle_notebook

    assert json.loads((ROOT / "notebooks" / "kaggle_train_100m.ipynb").read_text()) == json.loads(json.dumps(build_kaggle_notebook.build()))
