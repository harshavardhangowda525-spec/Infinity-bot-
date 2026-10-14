"""Build a tiny, randomly initialised Qwen2-architecture model + tokenizer, fully offline.

It is used ONLY to test the open-model pipeline (data, LoRA training, evaluation,
server integration) where huggingface.co cannot be reached. It knows nothing, and
results from it say nothing about model quality.
"""

from __future__ import annotations

from pathlib import Path

CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build_tiny_model(out_dir: str | Path, corpus: str | None = None, seed: int = 0) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = corpus or (Path(__file__).resolve().parents[1] / "agent" / "loop.py").read_text(encoding="utf-8")
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk.train_from_iterator([text], trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>"])
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out)
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = Qwen2ForCausalLM(cfg)
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.save_pretrained(out)
    (out / "TINY_TEST_MODEL.txt").write_text("Random weights for pipeline tests only. Not a usable model.\n")
    return out
