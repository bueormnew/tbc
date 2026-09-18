import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tbc.compiler import get_ternarize_targets
from tbc.search import initial_ternarization, dequantize_ternary
from tbc.linalg import write_weight_out_in
from tbc.perplexity import apply_dequantized, restore, perplexity

model = AutoModelForCausalLM.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True, dtype=torch.float32)
model.eval()
tok = AutoTokenizer.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
txt = open(r"C:\Users\gerso\Desktop\TBC\eval_corpus.txt").read()
ids = tok(txt, return_tensors="pt")["input_ids"]
print("tokens:", ids.shape[1])
wins = [ids[:, i:i+32] for i in range(0, ids.shape[1], 32) if ids[:, i:i+32].shape[1] == 32]
ev = torch.cat(wins, dim=0)
print("windows:", ev.shape[0])
pf = perplexity(model, ev)
print("FP32:", round(pf, 2), flush=True)
targets = get_ternarize_targets(model, "llama")
bk = {}
with torch.no_grad():
    for n, m in targets:
        if hasattr(m, "weight"):
            bk[n] = m.weight.detach().clone()
            W = m.weight.detach().float()
            Wt, al = initial_ternarization(W)
            write_weight_out_in(m, dequantize_ternary(Wt, al, 32))
print("RTN:", round(perplexity(model, ev), 2), flush=True)
restore(model, bk)
blob = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output\model.tbc\weights.pt", map_location="cpu")
from tbc.search import dequantize_ternary as dq
dqs = {n: dq(e["W"].to(torch.int8), e["alpha"].float(), 32) for n, e in blob.items()}
b2 = apply_dequantized(model, dqs, [n for n, _ in targets])
print("TBC:", round(perplexity(model, ev), 2), flush=True)
restore(model, b2)
print("DONE")
