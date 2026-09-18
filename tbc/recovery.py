"""TBC-R — Recuperación por etapas desde cuantización ternaria extrema.

Pipeline (post-training, presupuesto 6GB, sin reentrenar pesos densos):

  R0 entrada: M_FP + compilado TBC (patrones Ŵ, alphas por grupo, mapa E).
  R1 adaptativo (TBC-A, Sec 35): capas con E > adapt_threshold vuelven a FP
       original. El mapa de imposibilidad decide, no un humano.
  R2 aceptación global: por capa (orden de sensibilidad), se compara NLL de
       modelo completo con patrón RTN vs patrón TBC y se queda el mejor.
       Corrige el desajuste capa-local vs transferencia global.
  R3 destilación de escalas: patrones congelados; se entrenan alphas por grupo
       (+ normas + biases, <0.1% de parámetros) con Adam contra
       L = β·KL(teacher/T) + (1-β)·CE, early-stop en held-out.
  R4 exportación dual: torch (alphas por grupo) + GGUF (colapso honesto a una
       escala f32 por tensor, óptima LS). Se reportan ambas métricas.

Todo honesto: cada etapa debe mejorar (o igualar) la NLL de calibración
held-out; si no, se revierte la etapa.
"""
from __future__ import annotations
import copy
import logging
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compiler import get_ternarize_targets
from .linalg import weight_out_in
from .memory import MemoryMonitor
from .search import dequantize_ternary, initial_ternarization

log = logging.getLogger("tbc-r")


@dataclass
class RecoveryConfig:
    adapt_threshold: float = 0.65
    r2_batches: int = 8
    r2_seq_len: int = 32
    kd_steps: int = 300
    kd_batch_seqs: int = 4
    kd_seq_len: int = 64
    kd_lr_alpha: float = 5e-3
    kd_lr_other: float = 1e-4
    kd_T: float = 2.0
    kd_beta: float = 0.7
    kd_patience: int = 40
    seed: int = 0
    ram_budget: str | int = "6GB"


def _sanitize(name: str) -> str:
    return name.replace(".", "_")


class TernaryStudent(nn.Module):
    """Copia del modelo donde cada lineal objetivo se evalúa como
    Y = X @ dequant(T_frozen, alpha_learnable).T + bias.

    Solo alpha (+normas/biases si se pide) requieren gradiente.
    """

    def __init__(self, model_fp: nn.Module, targets, patterns: dict[str, torch.Tensor],
                 alphas: dict[str, torch.Tensor], group_size: int,
                 promoted: set[str] | None = None,
                 train_norms: bool = True, train_bias: bool = True):
        super().__init__()
        self.model = copy.deepcopy(model_fp).float()
        self.group_size = group_size
        self.patterns: dict[str, torch.Tensor] = {}
        self.alpha = nn.ParameterDict()
        mods = dict(self.model.named_modules())
        for name, mod in targets:
            if promoted and name in promoted:
                continue  # capa promovida: peso FP original intacto
            T = patterns[name].to(torch.int8)
            self.patterns[name] = T  # buffer congelado (no parámetro)
            a = alphas[name].detach().float().reshape(-1).clone()
            self.alpha[_sanitize(name)] = nn.Parameter(a)
            m = mods[name]
            Tbuf = self.patterns[name]
            aP = self.alpha[_sanitize(name)]
            gs = group_size

            def make_hook(Tb, aPm):
                def hook(md, inp, out):
                    X = inp[0].float()
                    Wdq = dequantize_ternary(Tb.float(), aPm, gs)
                    Y = X @ Wdq.T
                    b = md.bias
                    if b is not None:
                        Y = Y + b.float()
                    return Y.to(out.dtype) if isinstance(out, torch.Tensor) else Y
                return hook

            m.register_forward_hook(make_hook(Tbuf, aP))
        # congela todo menos alphas (+normas/biases opcionales)
        for p in self.model.parameters():
            p.requires_grad = False
        if train_norms or train_bias:
            for n, m in self.model.named_modules():
                lname = type(m).__name__.lower()
                is_norm = "norm" in lname or "ln_" in n or n.endswith("norm")
                if is_norm and train_norms and hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
                    m.weight.requires_grad = True
                if train_bias and hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
                    m.bias.requires_grad = True
        for p in self.alpha.parameters():
            p.requires_grad = True

    def learnable(self):
        ps = list(self.alpha.parameters())
        others = [p for p in self.model.parameters() if p.requires_grad]
        return ps, others

    def n_learnable(self) -> int:
        return sum(p.numel() for g in self.learnable() for p in g)


@torch.no_grad()
def eval_nll(model: nn.Module, batches: list[torch.Tensor]) -> float:
    """NLL media por token (eval). batches: [B,T] input_ids."""
    model.eval()
    tot, ntok = 0.0, 0
    for ids in batches:
        out = model(ids, labels=ids)
        loss = out.loss if hasattr(out, "loss") else out[0]
        tot += float(loss.item()) * ids.numel()
        ntok += ids.numel()
    return tot / max(1, ntok)


def make_windows(tok_or_vocab, ids_source: torch.Tensor | None, n: int, seq: int, seed: int,
                 tok=None) -> list[torch.Tensor]:
    """Lotes [1,seq] deterministas desde ids planas o matriz dada."""
    g = torch.Generator().manual_seed(seed)
    if ids_source is not None and ids_source.dim() == 2 and ids_source.shape[1] == seq:
        idx = torch.randperm(ids_source.shape[0], generator=g)[:n]
        return [ids_source[i:i + 1] for i in idx.tolist()]
    flat = ids_source.flatten()
    starts = torch.randperm(max(1, flat.numel() - seq), generator=g)[:n]
    return [flat[s:s + seq].unsqueeze(0) for s in starts.tolist()]


def stage_r1(per_layer_E: dict[str, float], threshold: float) -> set[str]:
    promoted = {n for n, e in per_layer_E.items() if e > threshold}
    log.info("[R1] promovidas a FP %d/%d (E>%.2f): %s",
             len(promoted), len(per_layer_E), threshold, sorted(promoted)[:8])
    return promoted


def stage_r2(model_fp: nn.Module, targets, patterns_tbc: dict, alphas_tbc: dict,
             group_size: int, smap: dict[str, float], calib: list[torch.Tensor],
             promoted: set[str]) -> tuple[dict, dict, dict]:
    """Aceptación global RTN-vs-TBC por capa. Retorna (patterns, alphas, choice)."""
    from .sensitivity import dequant_alpha
    mods = dict(model_fp.named_modules())
    # patrones RTN de referencia
    patterns_rtn, alphas_rtn = {}, {}
    with torch.no_grad():
        for name, mod in targets:
            if name in promoted:
                continue
            Wt, al = initial_ternarization(weight_out_in(mod))
            patterns_rtn[name] = Wt.to(torch.int8)
            alphas_rtn[name] = al.float()
    base = TernaryStudent(model_fp, targets, patterns_tbc, alphas_tbc, group_size, promoted,
                          train_norms=False, train_bias=False)
    base_nll = eval_nll(base.model, calib)
    log.info("[R2] NLL base TBC = %.4f", base_nll)
    final_p, final_a, choice = dict(patterns_tbc), dict(alphas_tbc), {}
    order = sorted([n for n, _ in targets if n not in promoted], key=lambda n: smap.get(n, 0))
    for name in order:
        trial_p = dict(final_p)
        trial_a = dict(final_a)
        trial_p[name] = patterns_rtn[name]
        trial_a[name] = alphas_rtn[name]
        trial = TernaryStudent(model_fp, targets, trial_p, trial_a, group_size, promoted,
                               train_norms=False, train_bias=False)
        nll = eval_nll(trial.model, calib)
        if nll < base_nll - 1e-6:
            final_p, final_a, base_nll = trial_p, trial_a, nll
            choice[name] = "RTN"
        else:
            choice[name] = "TBC"
        del trial
    n_rtn = sum(1 for v in choice.values() if v == "RTN")
    log.info("[R2] NLL final = %.4f (%d capas prefieren RTN)", base_nll, n_rtn)
    return final_p, final_a, choice


def stage_r3(model_fp: nn.Module, targets, patterns: dict, alphas: dict, group_size: int,
             promoted: set[str], train_ids: torch.Tensor, held_ids: torch.Tensor,
             cfg: RecoveryConfig) -> tuple[TernaryStudent, dict]:
    """Destilación de escalas+normas+biases. Retorna (student, stats)."""
    from .config import parse_budget
    mon = MemoryMonitor(parse_budget(cfg.ram_budget), parse_budget(cfg.ram_budget))
    teacher = copy.deepcopy(model_fp).float().eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = TernaryStudent(model_fp, targets, patterns, alphas, group_size, promoted,
                             train_norms=True, train_bias=True)
    log.info("[R3] parámetros entrenables: %d", student.n_learnable())
    a_ps, o_ps = student.learnable()
    opt = torch.optim.Adam([{"params": a_ps, "lr": cfg.kd_lr_alpha},
                            {"params": o_ps, "lr": cfg.kd_lr_other}])
    g = torch.Generator().manual_seed(cfg.seed)
    flat_tr = train_ids.flatten()
    flat_ho = held_ids.flatten()
    held = [flat_ho[i:i + cfg.kd_seq_len].unsqueeze(0)
            for i in range(0, max(1, flat_ho.numel() - cfg.kd_seq_len), cfg.kd_seq_len)][:16]
    best, best_state, bad = float("inf"), None, 0
    hist = []
    t0 = time.time()
    for step in range(cfg.kd_steps):
        student.model.train()
        idx = torch.randperm(max(1, flat_tr.numel() - cfg.kd_seq_len), generator=g)[:cfg.kd_batch_seqs]
        batch = torch.stack([flat_tr[i:i + cfg.kd_seq_len] for i in idx.tolist()])
        with torch.no_grad():
            t_logits = teacher(batch).logits
        s_out = student.model(batch, labels=batch)
        s_logits = s_out.logits
        ce = s_out.loss if hasattr(s_out, "loss") else s_out[0]
        kl = F.kl_div(F.log_softmax(s_logits / cfg.kd_T, dim=-1),
                      F.softmax(t_logits / cfg.kd_T, dim=-1),
                      reduction="batchmean") * (cfg.kd_T ** 2)
        loss = cfg.kd_beta * kl + (1 - cfg.kd_beta) * ce
        opt.zero_grad()
        loss.backward()
        opt.step()
        mon.check(f"kd-step-{step}")
        if (step + 1) % 25 == 0 or step == 0:
            hnll = eval_nll(student.model, held)
            hist.append((step + 1, float(loss.item()), hnll))
            log.info("[R3] step %d loss=%.4f held-NLL=%.4f", step + 1, loss.item(), hnll)
            if hnll < best - 1e-5:
                best, bad = hnll, 0
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()
                              if "alpha" in k or v.requires_grad}
            else:
                bad += 1
                if bad * 25 >= cfg.kd_patience:
                    log.info("[R3] early-stop en step %d (best held-NLL=%.4f)", step + 1, best)
                    break
    if best_state:
        student.load_state_dict(best_state, strict=False)
    stats = {"best_held_nll": best, "history": hist, "time_s": round(time.time() - t0, 1),
             "peak": mon.summary(), "n_learnable": student.n_learnable()}
    return student, stats
