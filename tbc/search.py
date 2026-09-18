"""TernarySearchEngine — búsqueda discreta real (Secciones 8-11/22/26 del MD).

Incluye:
- initial_ternarization (semilla trivial Q)
- alpha analítico L2 por grupo (Sec 22)
- CoordinateSearch discreto Hamming-1 con evaluación incremental Y_base + XΔW
- BeamSearch con K slots fijos pre-asignados
- Branch-and-Bound con cheap bound (norma L2)

PROHIBIDO GPTQ/AWQ como núcleo: todo aquí es código propio de búsqueda.
"""
from __future__ import annotations
import random
import torch
import torch.nn.functional as F


# ---------- Semilla + escalas ----------

def initial_ternarization(W: torch.Tensor, tau_strategy: str = "percentile_33", group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Q(w): +1 si w>τ, 0 si |w|<=τ, -1 si w<-τ. Retorna (Wt int8, alpha_por_grupo)."""
    Wf = W.detach().float()
    a = Wf.abs()
    if tau_strategy == "percentile_33":
        tau = float(torch.quantile(a.flatten(), 0.33).item())
    else:
        tau = float(0.5 * a.mean().item())
    Wt = torch.zeros_like(Wf, dtype=torch.int8)
    Wt[Wf > tau] = 1
    Wt[Wf < -tau] = -1
    alpha = compute_alpha_per_group(Wf, Wt, group_size=group_size)
    return Wt, alpha


def compute_alpha_per_group(W: torch.Tensor, Wt: torch.Tensor, group_size: int) -> torch.Tensor:
    """α*_g = <W_g, Ŵ_g> / ||Ŵ_g||² por fila-grupo (mínimos cuadrados, Sec 22).
    Vectorizado (idéntico matemáticamente al bucle fila×grupo)."""
    Wf, Wtf = W.float(), Wt.float()
    rows, cols = Wf.shape
    ng = (cols + group_size - 1) // group_size
    pad = ng * group_size - cols
    if pad:
        Wf = F.pad(Wf, (0, pad))
        Wtf = F.pad(Wtf, (0, pad))
    Wg = Wf.reshape(rows, ng, group_size)
    Tg = Wtf.reshape(rows, ng, group_size)
    num = (Wg * Tg).sum(-1)
    den = (Tg * Tg).sum(-1)
    alpha = torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.zeros_like(num))
    return alpha.reshape(-1)


def compute_alpha_analytic(X: torch.Tensor, Y_ref: torch.Tensor, XWt: torch.Tensor) -> float:
    """α* global L2: <Y_ref, XŴ>/<||XŴ||²> (Sec 22, costo casi cero)."""
    a = XWt.double().flatten()
    b = Y_ref.double().flatten()
    denom = float((a * a).sum().item())
    if denom < 1e-12:
        return 1.0
    return float((a * b).sum().item() / denom)


def dequantize_ternary(Wt: torch.Tensor, alpha: torch.Tensor, group_size: int) -> torch.Tensor:
    """Vectorizado: reshape (rows, ng, gs) * alpha[:, :, None]."""
    T = Wt.float()
    rows, cols = T.shape
    ng = (cols + group_size - 1) // group_size
    pad = ng * group_size - cols
    if pad:
        T = F.pad(T, (0, pad))
    a = alpha.float().reshape(rows, ng)
    out = (T.reshape(rows, ng, group_size) * a.unsqueeze(-1)).reshape(rows, ng * group_size)
    return out[:, :cols] if pad else out


# ---------- Motor de búsqueda ----------

class CoordinateSearchEngine:
    """Coordinate descent discreto con beam + B&B + eval incremental (Secs 9-11, 26)."""

    def __init__(self, group_size: int = 32, beam_size: int = 4, seed: int = 0):
        self.group_size = int(group_size)
        self.K = int(beam_size)
        self.rng = random.Random(seed)
        # slots fijos pre-asignados del beam (memoria fija)
        self.beam_slots: list = [None] * self.K

    @staticmethod
    def _layer_error(Y_cand: torch.Tensor, Y_ref: torch.Tensor, ref_norm: float | None = None) -> float:
        rn = ref_norm if ref_norm is not None else float(Y_ref.norm().item())
        return float(((Y_cand - Y_ref).norm() / (rn + 1e-9)).item())

    def candidates_hamming1(self, Wt: torch.Tensor, num: int) -> list[torch.Tensor]:
        """Genera `num` mutaciones a Hamming distance 1 (un peso -> otro valor)."""
        rows, cols = Wt.shape
        cands = []
        for _ in range(num):
            c = Wt.clone()
            r = self.rng.randrange(rows)
            cc = self.rng.randrange(cols)
            cur = int(c[r, cc].item())
            choices = [v for v in (-1, 0, 1) if v != cur]
            c[r, cc] = self.rng.choice(choices)
            cands.append(c)
        return cands

    def search_layer(
        self,
        X: torch.Tensor,          # [N, d_in] entradas cacheadas h_{l-1}
        W_fp: torch.Tensor,       # [d_out, d_in] referencia
        bias: torch.Tensor | None,
        Y_ref: torch.Tensor,      # [N, d_out] salida referencia
        num_candidates: int = 64,
        max_passes: int = 2,
        epsilon: float = 0.05,
        cheap_eps: float = 0.15,
        exact_tokens: int = 256,  # submuestra para la evaluación exacta (rápida y representativa)
    ) -> tuple[torch.Tensor, torch.Tensor, float, dict]:
        """Búsqueda greedy + beam por grupos. Retorna (Wt_best, alpha_best, E_best, stats)."""
        Xf = X.detach().float()
        Wf = W_fp.detach().float()
        Yf = Y_ref.detach().float()
        Wt_cur, alpha_cur = initial_ternarization(Wf)
        Wdq = dequantize_ternary(Wt_cur, alpha_cur, self.group_size)
        Y_base = Xf @ Wdq.T
        if bias is not None:
            Y_base = Y_base + bias.detach().float()
        E_best = self._layer_error(Y_base, Yf)
        E_init = E_best
        ref_norm = float(Yf.norm().item())  # hoist: la norma de referencia es constante
        evals = 0
        pruned = 0

        # Beam de K slots fijos: lista de (E, Wt, alpha)
        beam: list[tuple[float, torch.Tensor, torch.Tensor]] = [(E_best, Wt_cur.clone(), alpha_cur.clone())]

        rows, cols = Wt_cur.shape
        base_Y = None   # Y_base cacheada (hoist: una matmul completa por grupo, no por candidato)
        base_key = None  # E de beam[0] con la que se calculó base_Y
        # Saliency para propuestas: |X| media por entrada × |W| (guía discreta,
        # la decisión final sigue siendo verificación conductual E, no la saliency)
        with torch.no_grad():
            col_sal = Xf.abs().mean(dim=0)  # [d_in]
            sal_full = Wf.abs() * col_sal.unsqueeze(0).expand_as(Wf)
            sal_sum = float(sal_full.sum().item())
        for _ in range(max_passes):
            improved = False
            # recorre grupos contiguos de la última dim
            for gs in range(0, cols, self.group_size):
                ge = min(cols, gs + self.group_size)
                # candidatos: mutaciones Hamming-1 muestreadas por saliency en el grupo
                gw = ge - gs
                cands = []
                if sal_sum > 1e-12:
                    probs = (sal_full[:, gs:ge].reshape(-1) + 1e-12)
                    probs = probs / probs.sum()
                    picks = torch.multinomial(probs, num_candidates, replacement=True).tolist()
                else:
                    picks = [self.rng.randrange(rows * gw) for _ in range(num_candidates)]
                for k in picks:
                    r, cc0 = divmod(int(k), gw)
                    cc = cc0 + gs
                    c = beam[0][1].clone()
                    cur = int(c[r, cc].item())
                    c[r, cc] = self.rng.choice([v for v in (-1, 0, 1) if v != cur])
                    cands.append(c)
                for cand in cands:
                    # --- Evaluación incremental: solo ΔW del grupo ---
                    cur = beam[0][1]
                    dW_sign = (cand.float() - cur.float())  # en {-2..2}
                    if bool((dW_sign == 0).all()):
                        continue
                    # alpha rápida: reutiliza alpha del beam (exacta al final)
                    alpha = beam[0][2]
                    # ΔY = X @ (ΔW_dequant)^T ; Y_base cacheada por grupo
                    # (se recalcula solo si beam[0] cambió: base_key)
                    if base_Y is None or base_key != beam[0][0]:
                        Wdq0 = dequantize_ternary(cur, alpha, self.group_size)
                        base_Y = Xf @ Wdq0.T
                        if bias is not None:
                            base_Y = base_Y + bias.detach().float()
                        base_key = beam[0][0]
                    Wdq_cand = None  # solo se calculan en la rama no-Hamming-1
                    # kernel sparse-dense: solo columnas del grupo.
                    # Hamming-1 => rank-1 update exacto (sin matmuls ni dequants):
                    nz = (cand != cur).nonzero()
                    if nz.shape[0] == 1:
                        r0, c0 = int(nz[0, 0].item()), int(nz[0, 1].item())
                        ngroups = (cols + self.group_size - 1) // self.group_size
                        a_val = float(alpha[r0 * ngroups + min(c0 // self.group_size, ngroups - 1)].item())
                        dv = (int(cand[r0, c0].item()) - int(cur[r0, c0].item())) * a_val
                        Y_cand = base_Y.clone()
                        Y_cand[:, r0] += Xf[:, c0] * dv
                    else:
                        Wdq_cand = dequantize_ternary(cand, alpha, self.group_size)
                        Wdq_cur = dequantize_ternary(cur, alpha, self.group_size)
                        dWdq = Wdq_cand - Wdq_cur
                        dY = Xf[:, gs:ge] @ dWdq[:, gs:ge].T
                        Y_cand = base_Y + dY
                    evals += 1
                    # Branch-and-Bound: cheap bound L2
                    cheap = self._layer_error(Y_cand, Yf, ref_norm)
                    if cheap > cheap_eps and cheap > E_best:
                        pruned += 1
                        continue
                    # recalcula alpha analítica exacta para el candidato (barato, cerrado)
                    # evaluación exacta en submuestra (representativa, mucho más rápida)
                    Xe = Xf[:exact_tokens] if Xf.shape[0] > exact_tokens else Xf
                    Ye = Yf[:exact_tokens] if Yf.shape[0] > exact_tokens else Yf
                    alpha_new = compute_alpha_per_group(Wf, cand, self.group_size)
                    Wdq_new = dequantize_ternary(cand, alpha_new, self.group_size)
                    Y_new = Xe @ Wdq_new.T
                    if bias is not None:
                        Y_new = Y_new + bias.detach().float()
                    E_new = self._layer_error(Y_new, Ye, float(Ye.norm().item()))
                    evals += 1
                    # inserta en beam fijo (top-K)
                    beam.append((E_new, cand.clone(), alpha_new.clone()))
                    beam.sort(key=lambda t: t[0])
                    beam = beam[: self.K]
                    if beam[0][0] < E_best - 1e-9:
                        E_best = beam[0][0]
                        improved = True
                    # early exit si ya cumple epsilon
                    if E_best <= epsilon:
                        break
                if E_best <= epsilon:
                    break
            if not improved:
                break

        Wt_best, alpha_best = beam[0][1], beam[0][2]
        # E final SIEMPRE en X completa (la submuestra solo acelera la selección)
        with torch.no_grad():
            Y_final = Xf @ dequantize_ternary(Wt_best, alpha_best, self.group_size).T
            if bias is not None:
                Y_final = Y_final + bias.detach().float()
            E_final = self._layer_error(Y_final, Yf, ref_norm)
        stats = {"E_init": E_init, "E_best": E_final, "evals": evals, "pruned": pruned, "beam_size": len(beam)}
        return Wt_best, alpha_best, E_final, stats


class BeamSearch:
    """Wrapper fino con K fijo para compatibilidad con Sec 10 (slots pre-asignados)."""

    def __init__(self, K: int = 4):
        assert 1 <= K <= 16
        self.K = K
        self.slots: list = [None] * K

    def search(self, scored: list[tuple[float, object]]) -> list[tuple[float, object]]:
        scored = sorted(scored, key=lambda t: t[0])[: self.K]
        self.slots = (scored + [None] * self.K)[: self.K]
        return [s for s in self.slots if s is not None]
