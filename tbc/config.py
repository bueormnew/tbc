"""TBCConfig — configuración central del compilador. Sección 18/24 del MD."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


def parse_budget(s: str | int) -> int:
    """Convierte '6GB'/'512MB'/'auto'/int -> bytes.

    'auto' usa el 80% de la memoria disponible del sistema (RAM; y VRAM si
    hay CUDA). Así cualquier usuario compila sin conocer los números.
    """
    if isinstance(s, int):
        return s
    t = str(s).strip().upper().replace(" ", "")
    if t == "AUTO":
        import psutil
        import torch
        if torch.cuda.is_available():
            try:
                free, _ = torch.cuda.mem_get_info()
                return int(free * 0.8)
            except Exception:
                pass
        return int(psutil.virtual_memory().available * 0.8)
    mult = 1
    if t.endswith("GB"):
        mult = 1024 ** 3
        num = t[:-2]
    elif t.endswith("MB"):
        mult = 1024 ** 2
        num = t[:-2]
    elif t.endswith("KB"):
        mult = 1024
        num = t[:-2]
    elif t.endswith("B"):
        num = t[:-1]
    else:
        num = t
    return int(float(num) * mult)


def env_budget(var: str, default: str = "auto") -> str | int:
    """Lee presupuesto desde variable de entorno (ej. TBC_RAM_BUDGET=24GB)."""
    import os
    return os.environ.get(var, default)


def env_dtype(var: str = "TBC_DTYPE", default: str = "float32"):
    """float32 (reproduce resultados publicados) o float16 (modelos grandes)."""
    import os
    import torch
    name = os.environ.get(var, default).lower()
    return {"float32": torch.float32, "fp32": torch.float32,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[name]


@dataclass
class TBCConfig:
    vram_budget: str | int = "auto"
    ram_budget: str | int = "auto"
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
