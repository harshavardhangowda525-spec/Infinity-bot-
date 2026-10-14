"""Create training/tiny-model: a random, tiny Qwen2-architecture model for OFFLINE pipeline tests.

    python training/scripts/make_tiny_model.py
    python training/scripts/train.py --config training/configs/tiny-test.yaml --smoke

It knows nothing. Use it to check the pipeline works, never to judge quality.
"""

from _common import ROOT

from infinity.hf.tiny import build_tiny_model

if __name__ == "__main__":
    print("wrote", build_tiny_model(ROOT / "training" / "tiny-model"))
