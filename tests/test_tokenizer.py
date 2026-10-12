import pytest

from infinity.core.tokenizer import BPETokenizer, pretokenize

CORPUS = [
    "def add(a, b):\n    return a + b\n\n",
    "def sub(a, b):\n    return a - b\n\n",
    "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n",
    "The agent reads the file, edits the file, and runs the tests.\n",
] * 20


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=320)


def test_pretokenize_is_lossless():
    text = "  x = foo(1234)  # comment\n\tif y:\n        return 'ok'\r\n ünïcödé 😀"
    assert "".join(pretokenize(text)) == text


def test_roundtrip_ascii_code_and_unicode(tok):
    samples = [
        "def add(a, b):\n    return a + b\n",
        "print('héllo wörld') # 你好 😀",
        "",
        "\x00\x01 weird \u200b bytes",
        "a" * 1000,
    ]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress(tok):
    text = "def add(a, b):\n    return a + b\n"
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_vocab_size_and_special_ids(tok):
    assert tok.vocab_size == 320
    ids = {tok.token_id(t) for t in tok.special_tokens}
    assert len(ids) == len(tok.special_tokens)
    assert max(ids) == tok.vocab_size - 1


def test_special_tokens_not_injected_by_plain_text(tok):
    plain = tok.encode("<|eos|>")
    assert tok.eos_id not in plain
    assert tok.encode("<|eos|>", allow_special=True) == [tok.eos_id]
    assert tok.decode(tok.encode("hi<|end|>", allow_special=True)) == "hi<|end|>"


def test_save_load_roundtrip_and_fingerprint(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    again = BPETokenizer.load(path)
    assert again.merges == tok.merges
    assert again.fingerprint() == tok.fingerprint()
    s = "class Point:\n    pass\n"
    assert again.encode(s) == tok.encode(s)


def test_tampered_file_rejected(tok, tmp_path):
    import json

    path = tmp_path / "tok.json"
    tok.save(path)
    data = json.loads(path.read_text())
    data["merges"][0] = [1, 2]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        BPETokenizer.load(path)


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=300)
    b = BPETokenizer.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges


def test_decode_rejects_out_of_range(tok):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size + 5])
