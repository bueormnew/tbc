"""Exporta GPT-2 FP32 PURO (pesos HF originales) para aislar bugs del exportador."""
import os
import subprocess
import sys
import torch
from transformers import GPT2LMHeadModel

BASE = r"C:\Users\gerso\Desktop\TBC"
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
blob = {}
for n, m in model.named_modules():
    w = getattr(m, "weight", None)
    if isinstance(w, torch.nn.Parameter) and w.dim() == 2 and any(k in n for k in ("c_attn", "c_proj", "c_fc")):
        W = w.detach().float().T.clone()  # [out,in]
        Wt = torch.zeros_like(W, dtype=torch.int8)
        Wt[W > 0] = 1
        Wt[W < 0] = -1
        rows, cols = W.shape
        ng = (cols + 31) // 32
        al = torch.zeros(rows * ng)
        for r in range(rows):
            for gg in range(ng):
                s2, e2 = gg * 32, min(cols, (gg + 1) * 32)
                seg = W[r, s2:e2].abs()
                al[r * ng + gg] = float(seg[seg > 0].mean().item()) if (seg > 0).any() else 0.0
        blob[n] = {"W": Wt, "alpha": al}
d = os.path.join(BASE, "tbc_output_gpt2", "pure.tbc")
os.makedirs(d, exist_ok=True)
torch.save(blob, os.path.join(d, "weights.pt"))
print("fake weights.pt listo:", len(blob))
r = subprocess.run([sys.executable, os.path.join(BASE, "export_gpt2_gguf.py"), d,
                    os.path.join(BASE, "tbc_output_gpt2", "gpt2-pure-f32.gguf"),
                    os.path.join(BASE, "tbc_output_gpt2", "gpt2-pure-i2s.gguf")],
                   capture_output=True, text=True)
print(r.stdout[-600:])
print(r.stderr[-500:])
