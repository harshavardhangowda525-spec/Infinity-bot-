"""Training-size planner: parameters, memory, compute and GPU-hours for a config.

Everything except the GPU throughput figures is computed exactly from the
config. Throughput uses each GPU's published dense bf16/fp16 peak times an
assumed utilisation (MFU), shown in the output so it can be judged. Prices
are never assumed: pass ``price_per_gpu_hour`` to get a cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core.config import RunConfig

# Published dense tensor peak (TFLOP/s, bf16/fp16) and an assumed achievable utilisation.
GPUS = {
    "T4": (65.0, 0.30, 16),
    "P100": (18.7, 0.30, 16),
    "RTX4090": (165.0, 0.35, 24),
    "A100-80GB": (312.0, 0.40, 80),
    "H100-80GB": (989.0, 0.40, 80),
}
BYTES_PER_PARAM_TRAIN = 16  # fp32 master weights (4) + Adam moments (8) + bf16 weights/grads (4)


@dataclass
class Plan:
    params: int
    tokens: int
    flops: float
    weights_bf16_gb: float
    train_state_gb: float
    min_gpus: dict[str, int]
    gpu_hours: dict[str, float]

    def to_dict(self) -> dict:
        return self.__dict__


def plan(cfg: RunConfig, tokens: int | None = None) -> Plan:
    n = cfg.model.parameter_count()
    tokens = tokens or 20 * n  # compute-optimal rule of thumb (~20 tokens per parameter)
    flops = 6.0 * n * tokens
    state_gb = n * BYTES_PER_PARAM_TRAIN / 1e9
    min_gpus = {}
    hours = {}
    for name, (peak, mfu, mem) in GPUS.items():
        # sharded training state must fit in ~70% of GPU memory (rest: activations, buffers)
        min_gpus[name] = max(1, -(-int(state_gb * 10) // int(mem * 0.7 * 10)))
        hours[name] = round(flops / (peak * 1e12 * mfu) / 3600, 1)
    return Plan(n, tokens, flops, round(n * 2 / 1e9, 2), round(state_gb, 1), min_gpus, hours)


def render(cfg: RunConfig, p: Plan, price_per_gpu_hour: float | None = None, currency: str = "INR") -> str:
    lines = [
        f"{cfg.name}: {p.params:,} parameters",
        f"  weights (bf16): {p.weights_bf16_gb} GB   training state (weights + Adam): ~{p.train_state_gb} GB",
        f"  training tokens: {p.tokens:,}   compute: {p.flops:.2e} FLOPs (6 x params x tokens)",
        "  GPU                min GPUs   GPU-hours   assumptions" + ("   cost" if price_per_gpu_hour else ""),
    ]
    for name, (peak, mfu, mem) in GPUS.items():
        h = p.gpu_hours[name]
        row = f"  {name:<18} {p.min_gpus[name]:>8}   {h:>9,.1f}   {peak:.0f} TFLOP/s peak x {mfu:.0%} MFU, {mem} GB"
        if price_per_gpu_hour:
            row += f"   {currency} {h * price_per_gpu_hour:,.0f}"
        lines.append(row)
    lines.append("  Wall-clock time ~= GPU-hours / number of GPUs (plus failed runs and evaluation).")
    if not price_per_gpu_hour:
        lines.append("  Pass --price-per-gpu-hour to estimate cost with your provider's real price.")
    return "\n".join(lines)
