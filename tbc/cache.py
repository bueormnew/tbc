"""BehaviorCache con MEMORIA FIJA. Secciones 4/5/13 del MD.

Diseño:
- Presupuesto fijo en bytes dado en init. Toda la memoria de cache se
  pre-asigna como slots (listas de longitud fija). Tras `freeze()`,
  cero alloc dinámico: solo se reutilizan slots con política LRU/circular.
- Dos regiones: `summaries` (estadísticos compactos, baratos) y
  `exact` (checkpoints exactos en circular buffer de capacidad fija).
- `ingest` estima bytes y falla con MemoryBudgetExceeded si excede.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
import torch

from .memory import MemoryBudgetExceeded


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


@dataclass
class Summary:
    mean: float
    var: float
    norm_l2: float
    hist: list
    topk_vals: list
    topk_idx: list
    proj: list
    numel: int

    def approx_bytes(self) -> int:
        # compacto: escalares + hist(32) + topk(8) + proj(16)
        return 8 * (3 + 32 + 8 + 8 + 16) + 64


def summarize_tensor(t: torch.Tensor, hist_bins: int = 32, topk: int = 8, proj_dim: int = 16) -> Summary:
    with torch.no_grad():
        f = t.detach().float().flatten()
        mean = float(f.mean().item()) if f.numel() else 0.0
        var = float(f.var(unbiased=False).item()) if f.numel() > 1 else 0.0
        norm = float(torch.linalg.vector_norm(f).item()) if f.numel() else 0.0
        lo, hi = float(f.min().item()) if f.numel() else 0.0, float(f.max().item()) if f.numel() else 1.0
        if hi <= lo:
            hi = lo + 1e-6
        hist = torch.histc(f, bins=hist_bins, min=lo, max=hi).tolist()
        k = min(topk, f.numel())
        if k > 0:
            v, i = torch.topk(f.abs(), k)
            topk_vals, topk_idx = v.tolist(), i.tolist()
        else:
            topk_vals, topk_idx = [], []
        # random projection determinista (seed fija) para resumen lineal
        g = torch.Generator().manual_seed(0)
        r = torch.randn(f.numel(), min(proj_dim, max(1, f.numel())), generator=g)
        proj = (f @ r).tolist() if f.numel() else []
        return Summary(mean, var, norm, hist, topk_vals, topk_idx, proj, f.numel())


class CircularBuffer:
    """Buffer circular de capacidad fija. Slots pre-asignados, sin crecimiento."""

    def __init__(self, capacity: int):
        assert capacity >= 1
        self.capacity = int(capacity)
        self.slots: list = [None] * self.capacity
        self.keys: list = [None] * self.capacity
        self.pos = 0
        self.size = 0
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def write(self, key, value):
        # reutiliza slot circular, nunca hace append/grow tras init
        self.slots[self.pos] = value
        self.keys[self.pos] = key
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def read(self, key):
        for i in range(self.size):
            idx = (self.pos - 1 - i) % self.capacity
            if self.keys[idx] == key:
                return self.slots[idx]
        return None

    def __len__(self):
        return self.size


class HierarchicalCache:
    """Cache jerárquica con presupuesto fijo (Sección 13 del MD)."""

    def __init__(self, budget_bytes: int, exact_slots: int = 16, prefix_slots: int = 8):
        self.budget = int(budget_bytes)
        self.used = 0
        self.frozen = False
        self.exact = CircularBuffer(exact_slots)
        self.prefix = CircularBuffer(prefix_slots)
        self.summaries: OrderedDict = OrderedDict()
        self.max_summaries = 4096  # cota fija de entradas de resumen
        self.evictions = 0

    def freeze(self):
        self.frozen = True
        self.exact.freeze()
        self.prefix.freeze()

    def _charge(self, nbytes: int, what: str):
        if self.used + nbytes > self.budget:
            raise MemoryBudgetExceeded(
                f"[TBC][CACHE-FAIL] {what} necesita {nbytes}B, usado={self.used}B, budget={self.budget}B"
            )
        self.used += nbytes

    def ingest_summary(self, key: str, summary: Summary):
        b = summary.approx_bytes()
        if key in self.summaries:
            return  # ya existe, no duplica memoria
        if len(self.summaries) >= self.max_summaries:
            # evicción LRU dentro del presupuesto fijo
            self.summaries.popitem(last=False)
            self.evictions += 1
            self.used = max(0, self.used - b)
        self._charge(b, f"summary:{key}")
        self.summaries[key] = summary

    def write_exact(self, key: str, tensor: torch.Tensor):
        # Los checkpoints exactos viven en slots circulares pre-asignados:
        # no crecen; si el tensor es mayor que lo visto, se contabiliza una vez.
        b = _tensor_bytes(tensor.detach().cpu())
        # Si la key ya existe, reemplazo sin cargo extra.
        if self.exact.read(key) is not None:
            self.exact.write(key, tensor.detach().cpu().clone())
            return
        # Nuevo exacto: solo cabe si hay presupuesto; si no, evicta el más viejo
        # (el circular buffer sobrescribe, liberando implícitamente).
        if self.used + b > self.budget:
            # sobrescritura circular libera el slot más antiguo: estima su tamaño
            oldest_idx = self.exact.pos % self.exact.capacity
            oldest = self.exact.slots[oldest_idx]
            if oldest is not None and isinstance(oldest, torch.Tensor):
                self.used = max(0, self.used - _tensor_bytes(oldest))
        self._charge(b, f"exact:{key}")
        self.exact.write(key, tensor.detach().cpu().clone())

    def write_prefix(self, key: str, tensor: torch.Tensor):
        b = _tensor_bytes(tensor.detach().cpu())
        if self.prefix.read(key) is not None:
            self.prefix.write(key, tensor.detach().cpu().clone())
            return
        if self.used + b > self.budget:
            oldest_idx = self.prefix.pos % self.prefix.capacity
            oldest = self.prefix.slots[oldest_idx]
            if oldest is not None and isinstance(oldest, torch.Tensor):
                self.used = max(0, self.used - _tensor_bytes(oldest))
        self._charge(b, f"prefix:{key}")
        self.prefix.write(key, tensor.detach().cpu().clone())

    def get_exact(self, key: str):
        return self.exact.read(key)

    def get_prefix(self, key: str):
        return self.prefix.read(key)

    def get_summary(self, key: str):
        return self.summaries.get(key)

    def memory_report(self) -> dict:
        return {
            "budget_bytes": self.budget,
            "used_bytes": self.used,
            "used_gb": round(self.used / 1e9, 6),
            "exact_slots_used": len(self.exact),
            "prefix_slots_used": len(self.prefix),
            "summaries": len(self.summaries),
            "evictions": self.evictions,
            "frozen": self.frozen,
        }
