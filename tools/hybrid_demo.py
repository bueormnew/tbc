"""¿Puede GPT-2-1.58 generar algo decente? Experimento TBC-A (Sec 35):
misma TBC, pero las capas cuello (c_proj E~0.71) se quedan en FP32.
Mide PPL + genera muestras comparables (greedy, mismo prompt).
"""
import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tbc.compiler import get_ternarize_targets
from tbc.perplexity import apply_dequantized, restore, perplexity
from tbc.search import dequantize_ternary

PROMPT = "Hello, I am a small language model"
EVAL_TEXT = ("Hello, I am a small language model. The quick brown fox jumps over the lazy dog. " * 6).strip()

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
model.eval()
tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
targets = get_ternarize_targets(model, "gpt2")
names = [n for n, _ in targets]
blob = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2\model.tbc\weights.pt", map_location="cpu")
dqs = {n: dequantize_ternary(e["W"].to(torch.int8), e["alpha"].float(), 32) for n, e in blob.items()}

ids = tok(EVAL_TEXT, return_tensors="pt")["input_ids"]
wins = [ids[:, i:i+32] for i in range(0, ids.shape[1], 32) if ids[:, i:i+32].shape[1] == 32]
ev = torch.cat(wins[:4], dim=0)

configs = {
    "FP32-original": [],
    "TBC-puro(48/48)": names,
    "TBC-A sin-c_proj(24/48)": [n for n in names if "c_proj" not in n],
    "TBC-A solo-c_fc(12/48)": [n for n in names if ".mlp.c_fc" in n],
}
for label, tern in configs.items():
    bkp = apply_dequantized(model, dqs, tern)
    ppl = perplexity(model, ev)
    restore(model, bkp)
    print(f"PPL {label} = {ppl:.2f}", flush=True)

# muestras greedy comparables
def gen(tern, n_new=40):
    bkp = apply_dequantized(model, dqs, tern)
    with torch.no_grad():
        out = model.generate(tok(PROMPT, return_tensors="pt")["input_ids"],
                             max_new_tokens=n_new, do_sample=False, pad_token_id=tok.eos_token_id)
    restore(model, bkp)
    return tok.decode(out[0].tolist())

for label, tern in configs.items():
    print(f"--- {label} ---")
    print(gen(tern), flush=True)
print("DONE")
