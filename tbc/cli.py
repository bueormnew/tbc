"""TBC CLI: cuantiza y recupera CUALQUIER modelo causal de HuggingFace.

Uso:
  python -m tbc quantize --model Qwen/Qwen2.5-7B --out ./qwen25-7b-tbc
  python -m tbc quantize --model gpt2 --out ./gpt2-tbc --beam 8 --mode standard
  python -m tbc recover  --model gpt2 --tbc ./gpt2-tbc/model.tbc --out ./gpt2-r
  python -m tbc arch-info --arch qwen3

Presupuestos: --ram/--vram aceptan '6GB', '24GB' o 'auto' (=80% disponible).
Sin límites duros por defecto: el usuario elige (ver TBCConfig/env TBC_RAM_BUDGET).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch


def _load_causal(model_id: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=dtype)
    model.to(dtype)
    model.eval()
    return model, tok


def cmd_quantize(a) -> int:
    from tbc.config import TBCConfig, env_dtype
    from tbc.compiler import TBC_COMPILE, get_ternarize_targets, detect_architecture
    from tbc.calib_text import text_windows
    dt = env_dtype()
    log = _log()
    log(f"[cli] cargando {a.model} ({dt}) ...")
    model, tok = _load_causal(a.model, dt)
    arch = detect_architecture(model)
    tgts = get_ternarize_targets(model, arch)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[cli] arch={arch} lineales={len(tgts)} params={n_params}")
    cfg = TBCConfig(
        vram_budget=a.vram, ram_budget=a.ram, group_size=a.group, beam_size=a.beam,
        epsilon_target=a.eps, calibration_samples=a.calib, calibration_seq_len=a.seqlen,
        max_passes_per_layer=a.passes, candidates_per_group=a.candidates,
        mode=a.mode, enable_cross_layer_refinement=a.mode in ("standard", "max"),
        enable_backtracking=True, seed=a.seed)
    n_seq = min(a.calib, 256)
    try:
        calib = text_windows(tok, n_seq, min(a.seqlen, 64), stride=8)
    except Exception as e:
        log(f"[cli] calib texto no disponible ({e}); usando ids aleatorios")
        g = torch.Generator().manual_seed(a.seed)
        calib = torch.randint(0, model.config.vocab_size, (n_seq, min(a.seqlen, 64)), generator=g)
    os.makedirs(a.out, exist_ok=True)
    tbc_model, report = TBC_COMPILE(model, cfg, calib, ckpt_path=os.path.join(a.out, "ckpt.pt"))
    mdir = os.path.join(a.out, "model.tbc")
    os.makedirs(mdir, exist_ok=True)
    torch.save({n: {"W": tbc_model.ternary_weights[n].cpu(), "alpha": tbc_model.scales[n].cpu()}
                for n in tbc_model.ternary_weights}, os.path.join(mdir, "weights.pt"))
    with open(os.path.join(mdir, "manifest.json"), "w") as f:
        json.dump(tbc_model.manifest, f, indent=2, default=str)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(tbc_model.manifest, f, indent=2, default=str)
    print(f"QUANTIZE arch={arch} status={report['status']} E={report['global_E']:.4f} "
          f"peak={tbc_model.manifest['peak_ram_gb']}GB out={a.out}")
    return 0


def cmd_recover(a) -> int:
    from tbc.compiler import get_ternarize_targets, detect_architecture
    from tbc.perplexity import perplexity, apply_dequantized, restore
    from tbc.search import dequantize_ternary
    from tbc.recovery import (RecoveryConfig, stage_r1, stage_r2, stage_r3)
    from tbc.calib_text import text_windows
    log = _log()
    dt = _dtype()
    model, tok = _load_causal(a.model, dt)
    arch = detect_architecture(model)
    targets = get_ternarize_targets(model, arch)
    names = [n for n, _ in targets]
    blob = torch.load(os.path.join(a.tbc, "weights.pt"), map_location="cpu")
    manifest = json.load(open(os.path.join(a.tbc, "manifest.json")))
    patterns = {n: blob[n]["W"].to(torch.int8) for n in names if n in blob}
    alphas = {n: blob[n]["alpha"].float() for n in names if n in blob}
    Emap = {k: float(v) for k, v in manifest.get("layer_tolerances", {}).items()}
    keep = [n for n in names if n in patterns]
    ids = tok("Hello. " * 64, return_tensors="pt")["input_ids"]
    ev = torch.cat([ids[:, i:i + 32] for i in range(0, ids.shape[1], 32)
                    if ids[:, i:i + 32].shape[1] == 32][:4], dim=0)
    dq0 = {n: dequantize_ternary(patterns[n], alphas[n], a.group) for n in keep}
    b0 = apply_dequantized(model, dq0, keep)
    p0 = perplexity(model, ev)
    restore(model, b0)
    rconf = RecoveryConfig(adapt_threshold=a.adapt, kd_steps=a.steps, seed=a.seed)
    promoted = stage_r1(Emap, rconf.adapt_threshold)
    smap = {n: Emap.get(n, 0) for n in names}
    cal = [text_windows(tok, 8, 32, stride=8)[i:i + 1] for i in range(8)]
    p2, a2, choice = stage_r2(model, targets, patterns, alphas, a.group, smap, cal, promoted)
    tr = text_windows(tok, 48, 64, stride=16, offset_seq=0)
    ho = text_windows(tok, 16, 64, stride=16, offset_seq=850)
    student, stats = stage_r3(model, targets, p2, a2, a.group, promoted, tr, ho, rconf)
    p3 = perplexity(student.model, ev)
    os.makedirs(a.out, exist_ok=True)
    rec_al = {}
    for n in keep:
        if n in promoted:
            continue
        key = n.replace(".", "_")
        if key in student.alpha:
            p = student.alpha[key].detach().cpu()
            ng = (patterns[n].shape[1] + a.group - 1) // a.group
            rec_al[n] = p.reshape(-1)[: patterns[n].shape[0] * ng].clone()
    torch.save({"patterns": {k: v.cpu() for k, v in p2.items() if k in keep},
                "alphas_tbc": {k: v.cpu() for k, v in a2.items() if k in keep},
                "alphas_r3": rec_al, "promoted": sorted(promoted), "choice": choice},
               os.path.join(a.out, "recovered.pt"))
    nb = {n: p.detach().cpu().clone() for n, p in student.model.named_parameters()
          if p.requires_grad and "alpha" not in n}
    torch.save(nb, os.path.join(a.out, "norms_bias.pt"))
    n_tern = sum(patterns[n].numel() for n in keep if n not in promoted)
    n_tot = sum(p.numel() for p in model.parameters())
    eff = (n_tern * 1.585 + (n_tot - n_tern) * 32) / n_tot
    json.dump({"model": a.model, "R0": p0, "R3": p3, "eff_bits": round(eff, 3),
               "promoted": sorted(promoted), "stats": stats},
              open(os.path.join(a.out, "recovery.json"), "w"), indent=2)
    print(f"RECOVERY R0={p0:.2f} R3={p3:.2f} eff={eff:.2f}b/p out={a.out}")
    return 0


def cmd_arch_info(a) -> int:
    from tbc.arch import arch_spec, supported_arches
    if a.arch in (None, "list"):
        print("arquitecturas: " + ", ".join(supported_arches()))
        return 0
    s = arch_spec(a.arch)
    print(json.dumps({"arch": a.arch, **{k: (list(v) if isinstance(v, tuple) else v)
                                         for k, v in s.items()}}, indent=2, ensure_ascii=False))
    return 0


def _log():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return logging.getLogger("tbc-cli").info


def _dtype():
    from tbc.config import env_dtype
    return env_dtype()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tbc", description="Ternary Behavioral Compilation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize", help="cuantiza cualquier modelo causal de HF")
    q.add_argument("--model", required=True, help="id HF (ej. Qwen/Qwen2.5-7B, gpt2)")
    q.add_argument("--out", required=True)
    q.add_argument("--beam", type=int, default=4)
    q.add_argument("--mode", default="fast", choices=["fast", "standard", "max"])
    q.add_argument("--ram", default="auto", help="'auto', '6GB', '24GB'... (o TBC_RAM_BUDGET)")
    q.add_argument("--vram", default="auto")
    q.add_argument("--group", type=int, default=32)
    q.add_argument("--eps", type=float, default=0.05)
    q.add_argument("--calib", type=int, default=256)
    q.add_argument("--seqlen", type=int, default=32)
    q.add_argument("--passes", type=int, default=1)
    q.add_argument("--candidates", type=int, default=16)
    q.add_argument("--seed", type=int, default=1234)
    r = sub.add_parser("recover", help="TBC-R: recupera un modelo ternarizado")
    r.add_argument("--model", required=True)
    r.add_argument("--tbc", required=True, help="dir model.tbc")
    r.add_argument("--out", required=True)
    r.add_argument("--group", type=int, default=32)
    r.add_argument("--adapt", type=float, default=0.65)
    r.add_argument("--steps", type=int, default=300)
    r.add_argument("--seed", type=int, default=0)
    ai = sub.add_parser("arch-info", help="spec de arquitectura o lista")
    ai.add_argument("--arch", default="list")
    a = ap.parse_args(argv)
    if a.cmd == "quantize":
        return cmd_quantize(a)
    if a.cmd == "recover":
        return cmd_recover(a)
    return cmd_arch_info(a)


if __name__ == "__main__":
    raise SystemExit(main())
