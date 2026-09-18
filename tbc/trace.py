"""TraceGenerator + SparseCheckpointScheduler. Secciones 3.A/5/6 del MD."""
from __future__ import annotations
import random
import torch
import torch.nn as nn

from .cache import HierarchicalCache, summarize_tensor


class SparseCheckpointScheduler:
    """Muestreo estratificado por profundidad, rotación determinista (Sec 5)."""

    def __init__(self, L: int, p: float = 0.33, strategy: str = "stratified_depth", seed: int = 0):
        assert L >= 1
        self.L = int(L)
        self.p = float(p)
        self.strategy = strategy
        self.seed = seed
        self.k = max(1, int(round(self.L * self.p)))

    def sample(self, run_id: int) -> list[int]:
        rng = random.Random(self.seed + run_id)
        if self.strategy == "stratified_depth":
            # estratos: temprano / medio / tardío
            thirds = [list(range(0, self.L // 3)), list(range(self.L // 3, 2 * self.L // 3)), list(range(2 * self.L // 3, self.L))]
            per = max(1, self.k // 3)
            out: list[int] = []
            for s in thirds:
                if not s:
                    continue
                # rotación determinista + barajado con seed
                rot = (run_id * per) % max(1, len(s))
                ordered = s[rot:] + s[:rot]
                idx = list(ordered)
                rng.shuffle(idx)
                out.extend(sorted(idx[:per]))
            # completar hasta k si faltó por redondeo
            all_ids = list(range(self.L))
            rng.shuffle(all_ids)
            for i in all_ids:
                if len(out) >= self.k:
                    break
                if i not in out:
                    out.append(i)
            return sorted(out[: self.k])
        # fallback uniforme
        ids = list(range(self.L))
        rng.shuffle(ids)
        return sorted(ids[: self.k])

    def coverage_prob(self, K: int) -> float:
        return 1.0 - (1.0 - self.p) ** K


def generate_synthetic_calibration(vocab_size: int, n_samples: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    """Estímulos sintéticos deterministas (Sec 6): coberturas de entropía/longitud."""
    g = torch.Generator().manual_seed(seed)
    # mezcla: mitad uniforme, mitad sesgada (tokens raros vs frecuentes)
    ids = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)
    return ids


class TraceGenerator:
    """Ejecuta M_FP y captura trazas {h_l, A_l, logits} con hooks.

    Uso por capa lineal: registra input (h_{l-1}) y output por cada módulo
    objetivo durante forward reales del modelo completo.
    """

    def __init__(self, model: nn.Module, target_names: list[str], scheduler: SparseCheckpointScheduler, cache: HierarchicalCache):
        self.model = model
        self.target_names = list(target_names)
        self.scheduler = scheduler
        self.cache = cache
        self._captured: dict = {}

    def _hook(self, name: str):
        def fn(mod, inp, out):
            x = inp[0].detach().cpu() if isinstance(inp, tuple) else inp.detach().cpu()
            y = out.detach().cpu() if isinstance(out, torch.Tensor) else out[0].detach().cpu()
            # guarda solo último batch por nombre (ventana rotativa) + summary
            self._captured[name] = {"in": x, "out": y}
            self.cache.ingest_summary(f"{name}.in", summarize_tensor(x))
            self.cache.ingest_summary(f"{name}.out", summarize_tensor(y))
        return fn

    @torch.no_grad()
    def capture(self, batches: list[torch.Tensor], run_id: int = 0, exact_names: list[str] | None = None) -> dict:
        """Ejecuta forwards y guarda exactos solo para `exact_names` (sparse p)."""
        handles = []
        name_to_mod = dict(self.model.named_modules())
        for n in self.target_names:
            if n in name_to_mod:
                handles.append(name_to_mod[n].register_forward_hook(self._hook(n)))
        self.model.eval()
        try:
            for b in batches:
                self._captured = {}
                self.model(b)
                layer_ids = self.scheduler.sample(run_id)
                # mapea layer_ids -> nombres objetivo (por orden)
                sparse_set = {self.target_names[i] for i in layer_ids if i < len(self.target_names)}
                if exact_names is not None:
                    sparse_set = sparse_set.intersection(set(exact_names))
                for name, cap in self._captured.items():
                    if name in sparse_set:
                        self.cache.write_exact(f"{name}.in", cap["in"][0].detach() if cap["in"].dim() > 2 else cap["in"])
                        self.cache.write_exact(f"{name}.out", cap["out"][0].detach() if cap["out"].dim() > 2 else cap["out"])
        finally:
            for h in handles:
                h.remove()
        return self.cache.memory_report()
