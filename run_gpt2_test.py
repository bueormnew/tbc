#!/usr/bin/env python
"""TBC completa en GPT-2 small (124M, Conv1D) dentro de 6GB.

Cuantización TBC real de los 48 Conv1D + PPL FP32 vs RTN vs TBC.
Salidas: tbc_output_gpt2/.
"""
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "tbc_output_gpt2")
os.makedirs(OUT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("tbc-gpt2")

EVAL_TEXT = ("Hello, I am a small language model. The quick brown fox jumps over the lazy dog. " * 6).strip()


def main():
    from tbc.config import TBCConfig
    from tbc.compiler import TBC_COMPILE, get_ternarize_targets
    from tbc.search import initial_ternarization
    from tbc.linalg import weight_out_in, write_weight_out_in
    from tbc.perplexity import apply_dequantized, restore, perplexity
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    t_start = time.time()
    cfg = TBCConfig(
        vram_budget="6GB", ram_budget="6GB", group_size=32, beam_size=4,
        epsilon_target=0.05, epsilon_schedule=[0.05],
        calibration_samples=256, calibration_seq_len=32, max_tokens_per_layer=1024,
        packing_format="I2_S_TL2", max_passes_per_layer=1, candidates_per_group=64,
        mode="fast", enable_cross_layer_refinement=False, enable_backtracking=True,
    )
    log.info("[gpt2] config: %s", cfg)

    log.info("[gpt2] descargando GPT-2 small...")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
    tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model.float().eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("[gpt2] params=%d arch=%s", n_params, type(model).__name__)

    vocab = model.config.vocab_size
    from tbc.calib_text import text_windows
    calib_ids = text_windows(tok, 256, 32, stride=8)  # texto real en distribución

    tbc_model, report = TBC_COMPILE(model, cfg, calib_ids,
                                    ckpt_path=os.path.join(OUT, "ckpt.pt"))
    log.info("[gpt2] status=%s global_E=%.4f peak=%sGB", report["status"], report["global_E"],
             tbc_model.manifest["peak_ram_gb"])

    arch = tbc_model.arch
    targets = get_ternarize_targets(model, arch)
    names = [n for n, _ in targets]
    log.info("[gpt2] %d lineales objetivo", len(names))
    mods = dict(model.named_modules())

    ids = tok(EVAL_TEXT, return_tensors="pt")["input_ids"]
    wins = [ids[:, i:i+32] for i in range(0, ids.shape[1], 32) if ids[:, i:i+32].shape[1] == 32]
    eval_ids = torch.cat(wins[:4], dim=0)

    ppl_fp = perplexity(model, eval_ids)
    log.info("[gpt2] PPL_FP32=%.3f", ppl_fp)
    bkp0 = {}
    with torch.no_grad():
        for n, m in targets:
            if hasattr(m, "weight"):
                bkp0[n] = m.weight.detach().clone()
                W = weight_out_in(m)
                Wt, al = initial_ternarization(W)
                from tbc.sensitivity import dequant_alpha
                write_weight_out_in(m, dequant_alpha(Wt, al, 32))
    ppl_rtn = perplexity(model, eval_ids)
    log.info("[gpt2] PPL_RTN=%.3f", ppl_rtn)
    restore(model, bkp0)
    bkp = apply_dequantized(model, tbc_model.dequantized_state(), names)
    ppl_tbc = perplexity(model, eval_ids)
    log.info("[gpt2] PPL_TBC=%.3f", ppl_tbc)
    restore(model, bkp)

    mdir = os.path.join(OUT, "model.tbc")
    os.makedirs(mdir, exist_ok=True)
    torch.save({n: {"W": tbc_model.ternary_weights[n].cpu(), "alpha": tbc_model.scales[n].cpu()}
                for n in tbc_model.ternary_weights}, os.path.join(mdir, "weights.pt"))
    with open(os.path.join(mdir, "manifest.json"), "w") as f:
        json.dump(tbc_model.manifest, f, indent=2, default=str)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(tbc_model.manifest, f, indent=2, default=str)
    summary = {
        "model": "openai-community/gpt2", "arch": arch, "n_params": n_params,
        "n_targets": len(names), "mode": "fast", "status": report["status"],
        "global_E": report["global_E"], "per_layer_E": report["per_layer_E"],
        "per_layer_t": report["per_layer_t"],
        "ppl_fp32": ppl_fp, "ppl_rtn": ppl_rtn, "ppl_tbc": ppl_tbc,
        "peak_ram_gb": tbc_model.manifest["peak_ram_gb"],
        "compile_time_s": tbc_model.manifest["compile_time_s"],
        "total_s": round(time.time() - t_start, 1),
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(f"GPT2 global_E={report['global_E']:.4f} PPL fp={ppl_fp:.2f} rtn={ppl_rtn:.2f} tbc={ppl_tbc:.2f} peak={tbc_model.manifest['peak_ram_gb']}GB")


if __name__ == "__main__":
    main()
