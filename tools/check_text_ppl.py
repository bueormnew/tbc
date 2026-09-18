import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tbc.perplexity import perplexity, apply_dequantized, restore
from tbc.compiler import get_ternarize_targets

tok = AutoTokenizer.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
print("vocab", tok.vocab_size)
text = ("Hello, I am a small language model. The quick brown fox jumps over the lazy dog. " * 6).strip()
ids = tok(text, return_tensors="pt")["input_ids"]
print("ids shape", tuple(ids.shape))
# parte en 4 ventanas de 32 para PPL por ventanas
windows = [ids[:, i:i+32] for i in range(0, ids.shape[1], 32) if ids[:, i:i+32].shape[1] == 32]
eval_ids = torch.cat(windows[:4], dim=0)
print("eval", tuple(eval_ids.shape))

model = AutoModelForCausalLM.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True, dtype=torch.float32)
model.eval()
ppl_fp = perplexity(model, eval_ids)
print("PPL_FP16_text=%.4f" % ppl_fp)

# RTN
from tbc.search import initial_ternarization
from tbc.sensitivity import dequant_alpha
arch_targets = get_ternarize_targets(model, "llama")
backup = {}
with torch.no_grad():
    for n, m in arch_targets:
        if hasattr(m, "weight"):
            backup[n] = m.weight.detach().clone()
            Wt, al = initial_ternarization(m.weight.detach().float())
            m.weight.copy_(dequant_alpha(Wt, al, 32).to(m.weight.dtype))
ppl_rtn = perplexity(model, eval_ids)
print("PPL_RTN_text=%.4f" % ppl_rtn)
restore(model, backup)

# TBC
blob = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output\model.tbc\weights.pt", map_location="cpu")
dq = {}
for n, e in blob.items():
    Wt = e["W"]
    al = e["alpha"].float()
    rows, cols = Wt.shape
    gs, ng = 32, (Wt.shape[-1] + 31) // 32
    out = torch.empty(rows, cols)
    for r in range(rows):
        for g in range(ng):
            s2, e2 = g * gs, min(cols, (g + 1) * gs)
            out[r, s2:e2] = Wt[r, s2:e2].float() * float(al[r * ng + g].item())
    dq[n] = out
bkp = apply_dequantized(model, dq, [n for n, _ in arch_targets])
ppl_tbc = perplexity(model, eval_ids)
print("PPL_TBC_text=%.4f" % ppl_tbc)
restore(model, bkp)
print("DONE text_ppl fp=%.2f rtn=%.2f tbc=%.2f" % (ppl_fp, ppl_rtn, ppl_tbc))
