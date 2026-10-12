import json
import math

import pytest

torch = pytest.importorskip("torch")

from infinity.core.checkpoint import CheckpointError, load_checkpoint, save_checkpoint, verify_checkpoint
from infinity.core.config import ModelConfig, RunConfig
from infinity.core.model import InfinityNeuralCore
from infinity.core.tokenizer import BPETokenizer

SMALL = ModelConfig(vocab_size=300, context_length=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def make_model(seed=0, cfg=SMALL):
    torch.manual_seed(seed)
    return InfinityNeuralCore(cfg)


def test_forward_shapes_and_initial_loss():
    m = make_model()
    x = torch.randint(0, SMALL.vocab_size, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, SMALL.vocab_size)
    # Freshly initialized model should be close to uniform: loss ~ ln(V).
    assert abs(loss.item() - math.log(SMALL.vocab_size)) < 0.5


def test_causality_future_tokens_do_not_affect_past():
    m = make_model().eval()
    x = torch.randint(0, SMALL.vocab_size, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 7) % SMALL.vocab_size
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(y)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-5)
    assert not torch.allclose(a[0, 8:], b[0, 8:], atol=1e-5)


def test_manual_and_fused_attention_agree():
    m = make_model().eval()
    x = torch.randint(0, SMALL.vocab_size, (2, 20))
    with torch.no_grad():
        m.set_fused_attention(True)
        a, _ = m(x)
        m.set_fused_attention(False)
        b, _ = m(x)
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("fused", [True, False])
def test_kv_cache_matches_full_forward(fused):
    m = make_model().eval()
    m.set_fused_attention(fused)
    x = torch.randint(0, SMALL.vocab_size, (1, 10))
    with torch.no_grad():
        full, _ = m(x)
        caches = m.new_caches()
        first, _ = m(x[:, :6], caches=caches)
        steps = [first[:, -1]]
        for t in range(6, 10):
            out, _ = m(x[:, t : t + 1], caches=caches)
            steps.append(out[:, -1])
    assert torch.allclose(full[:, 5], steps[0], atol=1e-4)
    for i, t in enumerate(range(6, 10), 1):
        assert torch.allclose(full[:, t], steps[i], atol=1e-4)


def test_context_length_enforced():
    m = make_model()
    with pytest.raises(ValueError):
        m(torch.zeros((1, SMALL.context_length + 1), dtype=torch.long))


def test_weight_tying_and_param_count():
    m = make_model()
    assert m.head.weight is m.embed.weight
    assert m.num_parameters() == sum(p.numel() for p in m.parameters())


def test_can_overfit_tiny_batch():
    m = make_model()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, SMALL.vocab_size, (4, 16))
    y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(150):
        _, loss = m(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2


def _tok():
    return BPETokenizer.train(["hello world, hello agent\n"] * 10, vocab_size=SMALL.vocab_size)


def test_checkpoint_save_load_verify(tmp_path):
    tok = _tok()
    m = make_model()
    run = RunConfig(name="t", model=SMALL)
    path = save_checkpoint(tmp_path, 5, m, None, run, tok, {"loss": 1.0})
    assert (tmp_path / "LATEST").read_text() == path.name
    m2, tok2, run2, tstate, manifest, _ = load_checkpoint(tmp_path)
    assert tstate["step"] == 5 and manifest["identity"]["version"] == "5.5"
    for (k, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k
    assert tok2.fingerprint() == tok.fingerprint()
    with pytest.raises(CheckpointError):
        save_checkpoint(tmp_path, 5, m, None, run, tok, {})  # never overwrite


def test_checkpoint_corruption_detected(tmp_path):
    path = save_checkpoint(tmp_path, 1, make_model(), None, RunConfig(name="t", model=SMALL), _tok(), {})
    with open(path / "model.pt", "ab") as f:
        f.write(b"tamper")
    with pytest.raises(CheckpointError):
        verify_checkpoint(path)


def test_incomplete_checkpoint_rejected(tmp_path):
    path = save_checkpoint(tmp_path, 1, make_model(), None, RunConfig(name="t", model=SMALL), _tok(), {})
    (path / "manifest.json").unlink()
    with pytest.raises(CheckpointError):
        verify_checkpoint(path)


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"name": "x", "model": {"d_model": 32, "n_heads": 4, "bogus": 1}}))
    with pytest.raises(ValueError):
        RunConfig.load(p)
