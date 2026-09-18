"""SensitivityAnalyzer — S_l por capa/bloque/canal (Sección 15 del MD).

Método empírico honesto: ternariza SOLO la capa l (RTN trivial) y mide el
incremento de error de salida respecto a referencia. Capas cuyo RTN aislado
más degrada -> mayor S_l -> más presupuesto de búsqueda.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .search import initial_ternarization
from .linalg import weight_out_in


class SensitivityAnalyzer:
    def __init__(self, targets: list[tuple[str, nn.Linear]]):
        self.targets = targets  # [(name, linear)]

    @torch.no_grad()
    def compute(self, calibration: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
        """calibration: lista por capa de (X, Y_ref) con X:[N,d_in], Y:[N,d_out].

        Retorna sensitivity_map {name: S_l >= 0}.
        """
        smap: dict[str, float] = {}
        for (name, lin), (X, Y_ref) in zip(self.targets, calibration):
            W = weight_out_in(lin)  # convención [out,in] (soporta Conv1D/GPT-2)
            Wt, alpha = initial_ternarization(W, group_size=32)
            # dequant por grupo con alpha vector
            Y_rtn = X.float() @ dequant_alpha(Wt, alpha, 32).T
            if lin.bias is not None:
                Y_rtn = Y_rtn + lin.bias.detach().float()
            denom = Y_ref.float().norm().item() + 1e-9
            err = (Y_rtn - Y_ref.float()).norm().item() / denom
            # pondera por norma de la capa (capas grandes con mismo error pesan más)
            smap[name] = float(err * (1.0 + 0.1 * torch.log10(torch.tensor(max(1, W.numel()))).item()))
        # normaliza a [0,1]
        mx = max(smap.values()) if smap else 1.0
        if mx > 0:
            smap = {k: v / mx for k, v in smap.items()}
        return smap

    @staticmethod
    def budget_for(s: float) -> dict:
        if s >= 0.66:
            return {"K": 16, "candidates": 256, "tol": 0.05, "label": "very_high"}
        if s >= 0.33:
            return {"K": 8, "candidates": 64, "tol": 0.07, "label": "medium"}
        return {"K": 4, "candidates": 16, "tol": 0.10, "label": "low"}


def dequant_alpha(Wt: torch.Tensor, alpha, group_size: int) -> torch.Tensor:
    """Reconstruye float desde ternario + alphas por grupo (última dim).
    Delega al kernel vectorizado de search (misma matemática)."""
    from .search import dequantize_ternary
    a = alpha if isinstance(alpha, torch.Tensor) else torch.as_tensor(float(alpha))
    if a.dim() == 0:
        ng = (Wt.shape[-1] + group_size - 1) // group_size
        a = a.expand(Wt.shape[0] * ng).clone()
    return dequantize_ternary(Wt, a, group_size)
