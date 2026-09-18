"""Recomputa baselines RTN con dequant CORRECTO (tras fix de dequant_alpha, 2026-09-16).

Antes: sensitivity.dequant_alpha indexaba alpha por bloques contiguos en vez
de stride (r*ng+g) -> RTN mal dequantizado. TBC usaba dequantized_state
(correcto). Este script mide RTN correcto para NanoDex-1M.
"""
import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tbc.compiler import get_ternarize_targets
from tbc.search import initial_ternarization, dequantize_ternary
from tbc.linalg import write_weight_out_in
from tbc.perplexity import restore, perplexity

model = AutoModelForCausalLM.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True, dtype=torch.float32)
model.eval()
tok = AutoTokenizer.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
targets = get_ternarize_targets(model, "llama")
text = ("Hello, I am a small language model. The quick brown fox jumps over the lazy dog. " * 6).strip()
ids = tok(text, return_tensors="pt")["input_ids"]
wins = [ids[:, i:i+32] for i in range(0, ids.shape[1], 32) if ids[:, i:i+32].shape[1] == 32]
eval_ids = torch.cat(wins[:4], dim=0)

ppl_fp = perplexity(model, eval_ids)
bkp = {}
with torch.no_grad():
    for n, m in targets:
        if hasattr(m, "weight"):
            bkp[n] = m.weight.detach().clone()
            W = m.weight.detach().float()
            Wt, al = initial_ternarization(W)
            write_weight_out_in(m, dequantize_ternary(Wt, al, 32))
ppl_rtn = perplexity(model, eval_ids)
restore(model, bkp)
print(f"RTN_CORRECT fp={ppl_fp:.2f} rtn={ppl_rtn:.2f}", flush=True)
