"""Outils de profilage : chronométrage, mémoire GPU, comptage de paramètres, infos machine."""
from __future__ import annotations

import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import psutil
import torch
from torch import nn


@dataclass
class ParamStats:
    total: int
    trainable: int

    @property
    def ratio(self) -> float:
        return 100.0 * self.trainable / self.total


def count_parameters(model: nn.Module) -> ParamStats:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParamStats(total, trainable)


@dataclass
class Timer:
    """Accumule des durées nommées (en secondes)."""
    records: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.records[name] = self.records.get(name, 0.0) + time.perf_counter() - start


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2**20


def machine_info() -> dict[str, str]:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": _cpu_name(),
        "cpu_cores": f"{psutil.cpu_count(logical=False)} physiques / {psutil.cpu_count()} logiques",
        "ram_gb": f"{psutil.virtual_memory().total / 2**30:.1f}",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_mem_gb=f"{props.total_memory / 2**30:.1f}",
            cuda=str(torch.version.cuda),
            compute_capability=f"{props.major}.{props.minor}",
        )
    return info


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "inconnu"
