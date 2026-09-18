#!/usr/bin/env python
"""TBC-Standard en NanoDex-1M dentro de 6GB (Fase 2).

TBCConfig(mode=standard, beam=8, calib 1024 samples, eps_target=0.20,
cross-layer + backtracking). Salidas a tbc_output_std/ + GGUF standard.
"""
import json
import logging
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "tbc_output_std")
os.makedirs(OUT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("tbc-std")


def main():
    from tbc.config import TBCConfig
    from tbc.compiler import TBC_COMPILE, get_ternarize_targets
    from tbc.search import initial_ternarization
    from tbc.sensitivity import dequant_alpha
    from tbc.perplexity import apply_dequantized, restore, perplexity
    from tbc.export import export_bitnet_cpp_compatible
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    cfg = TBCConfig(
        vram_budget="6GB", ram_budget="6GB", group_size=32, beam_size=8,
        epsilon_target=0.20, epsilon_schedule=[0.25, 0.20],
        calibration_samples=1024, calibration_seq_len=16,
        packing_format="I2_S_TL2", max_passes_per_layer=2, candidates_per_group=32,
        mode="standard", enable_cross_layer_refinement=True, enable_backtracking=True,
    )
    log.info("[std] config: %s", cfg)

    log.info("[std] cargando NanoDex-1M...")
    model = AutoModelForCausalLM.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
    model.float().eval()
    vocab = model.config.vocab_size
    from tbc.calib_text import text_windows
    calib_ids = text_windows(tok, 1024, 16, stride=2)  # texto real en distribución
    log.info("[std] calib: %s", tuple(calib_ids.shape))

    tbc_model, report = TBC_COMPILE(model, cfg, calib_ids)
    log.info("[std] status=%s global_E=%.4f peak=%sGB", report["status"], report["global_E"],
             tbc_model.manifest["peak_ram_gb"])

    arch = tbc_model.arch
    targets = get_ternarize_targets(model, arch)
    names = [n for n, _ in targets]

    text = ("Hello, I am a small language model. The quick brown fox jumps over the lazy dog. " * 6).strip()
    ids = tok(text, return_tensors="pt")["input_ids"]
    wins = [ids[:, i:i+32] for i in range(0, ids.shape[1], 32) if ids[:, i:i+32].shape[1] == 32]
    eval_ids = torch.cat(wins[:4], dim=0)

    ppl_fp16 = perplexity(model, eval_ids)
    mods = dict(model.named_modules())
    bkp0 = {}
    with torch.no_grad():
        for n, m in targets:
            if hasattr(m, "weight"):
                bkp0[n] = m.weight.detach().clone()
                Wt, al = initial_ternarization(m.weight.detach().float())
                m.weight.copy_(dequant_alpha(Wt, al, 32).to(m.weight.dtype))
    ppl_rtn = perplexity(model, eval_ids)
    restore(model, bkp0)
    dq = tbc_model.dequantized_state()
    bkp = apply_dequantized(model, dq, names)
    ppl_tbc = perplexity(model, eval_ids)
    restore(model, bkp)
    log.info("[std] PPL fp=%.2f rtn=%.2f tbc=%.2f", ppl_fp16, ppl_rtn, ppl_tbc)

    mdir = os.path.join(OUT, "model.tbc")
    os.makedirs(mdir, exist_ok=True)
    torch.save({n: {"W": tbc_model.ternary_weights[n].cpu(), "alpha": tbc_model.scales[n].cpu()}
                for n in tbc_model.ternary_weights}, os.path.join(mdir, "weights.pt"))
    with open(os.path.join(mdir, "manifest.json"), "w") as f:
        json.dump(tbc_model.manifest, f, indent=2, default=str)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(tbc_model.manifest, f, indent=2, default=str)

    r = subprocess.run([sys.executable, os.path.join(BASE, "export_llama_gguf.py"), mdir,
                        os.path.join(OUT, "nanodex-tbc-standard-f32.gguf"),
                        os.path.join(OUT, "nanodex-tbc-standard-i2s.gguf")],
                       capture_output=True, text=True)
    log.info("[std] export rc=%d\n%s", r.returncode, (r.stdout + r.stderr)[-800:])

    summary = {
        "mode": "standard", "status": report["status"], "global_E": report["global_E"],
        "per_layer_E": report["per_layer_E"], "per_layer_t": report["per_layer_t"],
        "ppl_fp16": ppl_fp16, "ppl_rtn": ppl_rtn, "ppl_tbc": ppl_tbc,
        "peak_ram_gb": tbc_model.manifest["peak_ram_gb"],
        "compile_time_s": tbc_model.manifest["compile_time_s"],
        "total_s": round(time.time() - t_start, 1),
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(f"STD global_E={report['global_E']:.4f} PPL fp={ppl_fp16:.2f} rtn={ppl_rtn:.2f} tbc={ppl_tbc:.2f} peak={tbc_model.manifest['peak_ram_gb']}GB")


if __name__ == "__main__":
    main()
