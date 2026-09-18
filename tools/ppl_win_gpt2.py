import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tbc.compiler import get_ternarize_targets
from tbc.perplexity import apply_dequantized, restore, perplexity
from tbc.search import dequantize_ternary

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
model.eval()
tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
txt = open(r"C:\Users\gerso\Desktop\TBC\data\eval_corpus.txt").read()
ids = tok(txt, return_tensors="pt")["input_ids"]
print("tokens:", ids.shape[1])
for ws in (32, 256):
    wins = [ids[:, i:i+ws] for i in range(0, ids.shape[1], ws) if ids[:, i:i+ws].shape[1] == ws]
    ev = torch.cat(wins, dim=0)
    pf = perplexity(model, ev)
    print(f"win={ws} n={ev.shape[0]} FP32={pf:.2f}", flush=True)

blob = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2\model.tbc\weights.pt", map_location="cpu")
targets = get_ternarize_targets(model, "gpt2")
names = [n for n, _ in targets]
dqs = {n: dequantize_ternary(e["W"].to(torch.int8), e["alpha"].float(), 32) for n, e in blob.items()}
bk = apply_dequantized(model, dqs, names)
for ws in (32, 256):
    wins = [ids[:, i:i+ws] for i in range(0, ids.shape[1], ws) if ids[:, i:i+ws].shape[1] == ws]
    ev = torch.cat(wins, dim=0)
    print(f"win={ws} n={ev.shape[0]} TBC={perplexity(model, ev):.2f}", flush=True)
restore(model, bk)
print("DONE")
