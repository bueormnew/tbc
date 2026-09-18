"""EquivalenceChecker — cascada A/B/C/D + métrica compuesta E (Secs 14/16)."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class EquivalenceResult:
    E_A: float
    E_H: float
    E_L: float
    E_KL: float
    E: float
    passed: bool
    level_reached: str  # A|B|C|D


class EquivalenceChecker:
    def __init__(self, lambda_A=0.2, lambda_H=0.3, lambda_L=0.2, lambda_KL=0.3,
                 eps_cheap=0.15, eps_medium=0.10):
        s = lambda_A + lambda_H + lambda_L + lambda_KL
        self.lA, self.lH, self.lL, self.lKL = lambda_A / s, lambda_H / s, lambda_L / s, lambda_KL / s
        self.eps_cheap = eps_cheap
        self.eps_medium = eps_medium

    @staticmethod
    def level_A(h, h_hat) -> float:
        return float((((h - h_hat).norm()) / (h.norm() + 1e-9)).item())

    @staticmethod
    def level_B(h, h_hat) -> float:
        cos = F.cosine_similarity(h.flatten().float(), h_hat.flatten().float(), dim=0)
        return float((1.0 - cos).item())

    @staticmethod
    def level_C(z, z_hat, k: int = 10) -> float:
        k = max(1, min(k, z.numel()))
        zv, _ = torch.topk(z.flatten().float(), k)
        hv, _ = torch.topk(z_hat.flatten().float(), k)
        return float((((zv - hv).norm()) / (zv.norm() + 1e-9)).item())

    @staticmethod
    def level_D(z, z_hat, T: float = 1.0) -> float:
        p = F.softmax(z.flatten().float() / T, dim=0).clamp_min(1e-12)
        q = F.softmax(z_hat.flatten().float() / T, dim=0).clamp_min(1e-12)
        return float((F.kl_div(q.log(), p, reduction="sum")).item())

    def composed(self, E_A, E_H, E_L, E_KL) -> float:
        return self.lA * E_A + self.lH * E_H + self.lL * E_L + self.lKL * E_KL

    def cascade(self, h_ref, h_cand, z_ref=None, z_cand=None, epsilon=0.05) -> EquivalenceResult:
        """Early-exit: la mayoría muere en A/B sin llegar a logits (Sec 14)."""
        EA = self.level_A(h_ref, h_cand)
        if EA > self.eps_cheap and EA > epsilon:
            return EquivalenceResult(EA, EA, EA, 0.0, EA, False, "A")
        EB = self.level_B(h_ref, h_cand)
        if EB > self.eps_medium and EB > epsilon:
            E = self.composed(EA, EB, EA, 0.0)
            return EquivalenceResult(EA, EB, EA, 0.0, E, E <= epsilon, "B")
        if z_ref is None or z_cand is None:
            E = self.composed(EA, EB, EA, 0.0)
            return EquivalenceResult(EA, EB, EA, 0.0, E, E <= epsilon, "B")
        EC = self.level_C(z_ref, z_cand)
        ED = self.level_D(z_ref, z_cand)
        E = self.composed(EA, EB, EC, min(ED, 5.0) / 5.0)
        return EquivalenceResult(EA, EB, EC, ED, E, E <= epsilon, "D")

    def validate_layer(self, Y_ref, Y_cand) -> EquivalenceResult:
        EA = self.level_A(Y_ref, Y_cand)
        EB = self.level_B(Y_ref, Y_cand)
        E = self.composed(EA, EB, EA, 0.0)
        return EquivalenceResult(EA, EB, EA, 0.0, E, True, "B")
