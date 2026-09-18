"""TBCConfig — configuración central del compilador. Sección 18/24 del MD."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


def parse_budget(s: str | int) -> int:
    """Convierte '6GB'/'512MB'/int -> bytes."""
    if isinstance(s, int):
        return s
    s = str(s).strip().upper().replace(" ", "")
    mult = 1
    if s.endswith("GB"):
        mult = 1024 ** 3
        num = s[:-2]
    elif s.endswith("MB"):
        mult = 1024 ** 2
        num = s[:-2]
    elif s.endswith("KB"):
        mult = 1024
        num = s[:-2]
    elif s.endswith("B"):
        num = s[:-1]
    else:
        num = s
    return int(float(num) * mult)


@dataclass
class TBCConfig:
    vram_budget: str | int = "6GB"
    ram_budget: str | int = "6GB"
    group_size: int = 32
    beam_size: int = 4
    epsilon_target: float = 0.05
    epsilon_schedule: List[float] = field(default_factory=lambda: [0.10, 0.08, 0.06, 0.05])
    epsilon_cheap: float = 0.15
    epsilon_medium: float = 0.10
    calibration_samples: int = 512
    calibration_seq_len: int = 64
    max_tokens_per_layer: int = 8192
    packing_format: str = "I2_S_TL2"
    seed: int = 1234
    max_passes_per_layer: int = 2
    candidates_per_group: int = 16
    lambda_A: float = 0.2
    lambda_H: float = 0.3
    lambda_L: float = 0.2
    lambda_KL: float = 0.3
    mode: str = "fast"  # fast | standard | max
    enable_cross_layer_refinement: bool = False
    enable_backtracking: bool = True

    def __post_init__(self):
        self.vram_budget_bytes = parse_budget(self.vram_budget)
        self.ram_budget_bytes = parse_budget(self.ram_budget)
        assert self.group_size in (16, 32, 64, 128), "group_size debe ser 16/32/64/128"
        assert 1 <= self.beam_size <= 16, "beam_size K debe estar en [1,16]"
        if self.mode == "standard":
            self.beam_size = max(self.beam_size, 8)
        elif self.mode == "max":
            self.beam_size = 16

    @property
    def mem_budget_bytes(self) -> int:
        return min(self.vram_budget_bytes, self.ram_budget_bytes)
