#!/usr/bin/env python
"""run_recovery.py — TBC-R por etapas. Uso: python run_recovery.py [nanodex|gpt2].

R0 = compilado existente -> R1 adaptativo -> R2 aceptación global ->
R3 destilación de escalas -> tabla por etapa + artefactos.
"""
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("tbc-r-run")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nanodex"
CONF = {
    "nanodex": {"id": "DedeProGames/NanoDex-1M", "arch": "llama", "tbc": "tbc_output/model.tbc",
                "out": "tbc_output_r", "tok_cls": None},
    "gpt2": {"id": "openai-community/gpt2", "arch": "gpt2", "tbc": "tbc_output_gpt2/model.tbc",
             "out": "tbc_output_gpt2_r", "tok_cls": None},
}[MODEL]


def main():
    from tbc.config import TBCConfig
    from tbc.compiler import TBC_COMPILE, get_ternarize_targets, detect_architecture
    from tbc.perplexity import perplexity
    from tbc.recovery import (RecoveryConfig, TernaryStudent, eval_nll, make_windows,
                              stage_r1, stage_r2, stage_r3)

    t_start = time.time()
    OUT = os.path.join(BASE, CONF["out"])
    os.makedirs(OUT, exist_ok=True)
    if MODEL == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
        tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        EVAL = ("Hello, I am a small language model. The quick brown fox jumps over the lazy dog. " * 6).strip()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(CONF["id"], trust_remote_code=True, dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(CONF["id"], trust_remote_code=True)
        EVAL = ("Hello, I am a small language model. The quick brown fox jumps over the lazy dog. " * 6).strip()
    model.float().eval()
    arch = detect_architecture(model)
    targets = get_ternarize_targets(model, arch)
    names = [n for n, _ in targets]
    log.info("[R] modelo=%s arch=%s lineales=%d", MODEL, arch, len(names))

    tbc_blob = torch.load(os.path.join(BASE, CONF["tbc"], "weights.pt"), map_location="cpu")
    manifest = json.load(open(os.path.join(BASE, CONF["tbc"], "manifest.json")))
    patterns = {n: tbc_blob[n]["W"].to(torch.int8) for n in names}
    alphas = {n: tbc_blob[n]["alpha"].float() for n in names}
    Emap = {k: float(v) for k, v in manifest["layer_tolerances"].items()}

    ids = tok(EVAL, return_tensors="pt")["input_ids"]
    wins = [ids[:, i:i + 32] for i in range(0, ids.shape[1], 32) if ids[:, i:i + 32].shape[1] == 32]
    eval_ids = torch.cat(wins[:4], dim=0)

    from tbc.calib_text import text_windows
    calib_full = text_windows(tok, 64, 32, stride=8)
    r2cal = [calib_full[i:i + 1] for i in range(8)]
    tr = text_windows(tok, 48, 64, stride=16, offset_seq=0)
    ho = text_windows(tok, 16, 64, stride=16, offset_seq=850)

    # FP32 puro y R0 TBC
    from tbc.perplexity import apply_dequantized, restore
    from tbc.search import dequantize_ternary
    dq0 = {n: dequantize_ternary(patterns[n], alphas[n], 32) for n in names}
    ppl_fp = perplexity(model, eval_ids)
    b0 = apply_dequantized(model, dq0, names)
    ppl_r0 = perplexity(model, eval_ids)
    restore(model, b0)
    stage = {"FP32": ppl_fp, "R0_TBC": ppl_r0}
    log.info("[R] FP32=%.2f R0_TBC=%.2f", ppl_fp, ppl_r0)

    # R1
    rconf = RecoveryConfig(adapt_threshold=0.65, kd_steps=100 if MODEL == "nanodex" else 300)
    promoted = stage_r1(Emap, rconf.adapt_threshold)
    keep = [n for n in names if n not in promoted]
    b1 = apply_dequantized(model, dq0, keep)
    stage["R1_adapt"] = perplexity(model, eval_ids)
    n_prom = len(promoted)
    restore(model, b1)
    log.info("[R] R1 promovidas=%d PPL=%.2f", n_prom, stage["R1_adapt"])

    # R2
    smap = {n: Emap.get(n, 0) for n in names}
    p2, a2, choice = stage_r2(model, targets, patterns, alphas, 32, smap, r2cal, promoted)
    dq2 = {n: dequantize_ternary(p2[n], a2[n], 32) for n in keep}
    b2 = apply_dequantized(model, dq2, keep)
    stage["R2_accept"] = perplexity(model, eval_ids)
    restore(model, b2)
    log.info("[R] R2 PPL=%.2f", stage["R2_accept"])

    # R3
    student, stats = stage_r3(model, targets, p2, a2, 32, promoted, tr, ho, rconf)
    # vuelca alphas entrenadas + PPL final (aplica al modelo vía hooks del student)
    ppl_r3 = perplexity(student.model, eval_ids)
    stage["R3_distill"] = ppl_r3
    log.info("[R] R3 PPL=%.2f best_held=%.4f learnable=%d", ppl_r3, stats["best_held_nll"], stats["n_learnable"])

    # guarda alphas entrenadas
    rec_alphas = {}
    for n in names:
        if n in promoted:
            continue
        key = n.replace(".", "_")
        if key in student.alpha:
            p = student.alpha[key].detach().cpu()
            ng = (patterns[n].shape[1] + 31) // 32
            rec_alphas[n] = p.reshape(-1)[: patterns[n].shape[0] * ng].clone()
    torch.save({"patterns": {k: v.cpu() for k, v in p2.items()},
                "alphas_tbc": {k: v.cpu() for k, v in a2.items()},
                "alphas_r3": rec_alphas,
                "promoted": sorted(promoted), "choice": choice},
               os.path.join(OUT, "recovered.pt"))
    # normas/biases entrenados viven en student.model: guarda diff mínima (solo esos)
    nb = {n: p.detach().cpu().clone() for n, p in student.model.named_parameters() if p.requires_grad
          and "alpha" not in n}
    torch.save(nb, os.path.join(OUT, "norms_bias.pt"))

    # bits efectivos
    n_tern = sum(patterns[n].numel() for n in keep)
    n_fp = sum(model.state_dict()[n + ".weight"].numel() for n, _ in targets if n in promoted)
    n_total = sum(p.numel() for p in model.parameters())
    eff = (n_tern * 1.585 + n_fp * 32) / n_total
    out = {"model": MODEL, "stages": stage, "promoted": sorted(promoted),
           "r2_rtn": sum(1 for v in choice.values() if v == "RTN"),
           "eff_bits": round(eff, 3), "r3_stats": stats,
           "total_s": round(time.time() - t_start, 1)}
    json.dump(out, open(os.path.join(OUT, "recovery.json"), "w"), indent=2)
    print("RECOVERY " + json.dumps({"stages": stage, "eff_bits": out["eff_bits"]}))
    log.info("[R] fin en %.0fs", time.time() - t_start)


if __name__ == "__main__":
    main()
