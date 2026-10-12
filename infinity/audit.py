"""Hardware and software capability audit (all values measured on this machine)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def _mem_total_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
        return round(int(out) / 1024**3, 1)
    except Exception:
        return None


def _tool_version(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (cp.stdout or cp.stderr).strip().splitlines()[0]
    except Exception:
        return "installed (version unknown)"


def audit(path: str = ".") -> dict:
    disk = shutil.disk_usage(path)
    report = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _mem_total_gb(),
        "disk_free_gb": round(disk.free / 1024**3, 1),
        "python": sys.version.split()[0],
        "torch": None,
        "accelerator": "none",
        "gpus": [],
        "tools": {
            "git": _tool_version(["git", "--version"]),
            "node": _tool_version(["node", "--version"]),
            "npm": _tool_version(["npm", "--version"]),
            "docker": _tool_version(["docker", "--version"]),
            "blender": _tool_version(["blender", "--version"]),
        },
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            report["accelerator"] = "cuda"
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                report["gpus"].append({"name": p.name, "vram_gb": round(p.total_memory / 1024**3, 1)})
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            report["accelerator"] = "mps"
    except ImportError:
        pass
    return report


def recommend(report: dict) -> dict:
    """Pick a starting model size from measured resources.

    Rough training-cost rule of thumb: ~6 * params * tokens FLOPs.
    A modern 4-core CPU sustains on the order of 1e10-1e11 FLOP/s in practice
    for these small matmuls; a consumer GPU 1e13+.
    """
    acc = report.get("accelerator")
    vram = max((g["vram_gb"] for g in report.get("gpus", [])), default=0)
    if acc == "cuda" and vram >= 20:
        return {"config": "configs/hyper-5.5-small.json", "why": f"CUDA GPU with {vram} GB VRAM"}
    if acc in ("cuda", "mps"):
        return {"config": "configs/hyper-5.5-small.json", "why": f"{acc} accelerator available"}
    return {"config": "configs/hyper-5.5-tiny.json", "why": "CPU only — tiny prototype is the practical size"}
