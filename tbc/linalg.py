"""Convenciones de pesos por tipo de módulo (nn.Linear vs Conv1D de GPT-2).

TBC trabaja internamente SIEMPRE en convención [out, in] con Y = X @ W.T.
- nn.Linear: weight ya es [out, in].
- transformers Conv1D (GPT-2 c_attn/c_proj/c_fc): weight es [in, out];
  se transpone al entrar y al salir. Sin excepciones en el resto del código.
"""
from __future__ import annotations
import torch


def is_conv1d(mod) -> bool:
    return type(mod).__name__ == "Conv1D"


def weight_out_in(mod) -> torch.Tensor:
    W = mod.weight.detach().float()
    if is_conv1d(mod):
        return W.T.clone()
    return W.clone()


def write_weight_out_in(mod, W_oi: torch.Tensor, dtype=None) -> None:
    dtype = dtype or mod.weight.dtype
    with torch.no_grad():
        if is_conv1d(mod):
            mod.weight.copy_(W_oi.T.to(dtype))
        else:
            mod.weight.copy_(W_oi.to(dtype))


def forward_out_in(X: torch.Tensor, W_oi: torch.Tensor, bias=None) -> torch.Tensor:
    Y = X.float() @ W_oi.T.float()
    if bias is not None:
        Y = Y + bias.detach().float()
    return Y
