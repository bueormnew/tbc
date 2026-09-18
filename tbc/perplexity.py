"""Perplejidad real FP16 vs RTN vs TBC (Sec 31 del MD)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn


@torch.no_grad()
def perplexity(model: nn.Module, input_ids: torch.Tensor) -> float:
    model.eval()
    nll, ntok = 0.0, 0
    B = input_ids.shape[0]
    for i in range(B):
        ids = input_ids[i : i + 1]
        out = model(ids, labels=ids)
        # transformers devuelve loss = NLL media
        loss = out.loss if hasattr(out, "loss") else out[0]
        nll += float(loss.item()) * ids.numel()
        ntok += ids.numel()
    return math.exp(nll / max(1, ntok))


@torch.no_grad()
def apply_dequantized(model: nn.Module, dq_state: dict[str, torch.Tensor], targets: list[str]) -> dict[str, torch.Tensor]:
    """Sustituye lineales por versión dequantizada (para PPL_TBC honesta en runtime torch).
    Retorna backup para restaurar."""
    mods = dict(model.named_modules())
    backup = {}
    for name in targets:
        mod = mods.get(name)
        if mod is None or name not in dq_state:
            continue
        if hasattr(mod, "weight"):
            backup[name] = mod.weight.detach().clone()
            from .linalg import write_weight_out_in
            with torch.no_grad():
                write_weight_out_in(mod, dq_state[name])
    return backup


@torch.no_grad()
def restore(model: nn.Module, backup: dict[str, torch.Tensor]):
    mods = dict(model.named_modules())
    for name, w in backup.items():
        if name in mods and hasattr(mods[name], "weight"):
            mods[name].weight.copy_(w)
