"""Monitoreo de memoria con límite duro. Invariante Sección 0/4 del MD."""
from __future__ import annotations
import psutil
import torch


class MemoryBudgetExceeded(RuntimeError):
    pass


class MemoryMonitor:
    """Rastrea pico de RAM (y VRAM si hay CUDA). Falla si se excede el presupuesto."""

    def __init__(self, ram_budget_bytes: int, vram_budget_bytes: int | None = None):
        self.ram_budget = int(ram_budget_bytes)
        self.vram_budget = int(vram_budget_bytes or ram_budget_bytes)
        self.proc = psutil.Process()
        self.peak_ram = 0
        self.peak_vram = 0

    def rss(self) -> int:
        return self.proc.memory_info().rss

    def vram(self) -> int:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0

    def check(self, phase: str = "") -> dict:
        rss = self.rss()
        vrm = self.vram()
        self.peak_ram = max(self.peak_ram, rss)
        self.peak_vram = max(self.peak_vram, vrm)
        if rss > self.ram_budget:
            raise MemoryBudgetExceeded(
                f"[TBC][MEM-FAIL] fase={phase} RSS={rss/1e9:.3f}GB > presupuesto={self.ram_budget/1e9:.3f}GB"
            )
        if torch.cuda.is_available() and vrm > self.vram_budget:
            raise MemoryBudgetExceeded(
                f"[TBC][MEM-FAIL] fase={phase} VRAM={vrm/1e9:.3f}GB > presupuesto={self.vram_budget/1e9:.3f}GB"
            )
        return {"rss": rss, "vram": vrm, "peak_ram": self.peak_ram, "peak_vram": self.peak_vram}

    def summary(self) -> dict:
        return {
            "peak_ram_bytes": self.peak_ram,
            "peak_ram_gb": round(self.peak_ram / 1e9, 3),
            "peak_vram_bytes": self.peak_vram,
            "peak_vram_gb": round(self.peak_vram / 1e9, 3),
            "ram_budget_gb": round(self.ram_budget / 1e9, 3),
        }
