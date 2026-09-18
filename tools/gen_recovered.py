"""Muestra de generación del GPT-2 recuperado (R3) vs FP32 vs R0."""
import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tbc.compiler import get_ternarize_targets
from tbc.perplexity import apply_dequantized, restore
from tbc.search import dequantize_ternary

PROMPT = "The challenge is to preserve the behavior of the original model despite"
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
model.eval()
model.requires_grad_(False)
tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
targets = get_ternarize_targets(model, "gpt2")
names = [n for n, _ in targets]
rec = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2_r\recovered.pt", map_location="cpu")
nb = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2_r\norms_bias.pt", map_location="cpu")
promoted = set(rec["promoted"])
keep = [n for n in names if n not in promoted]
dqs = {n: dequantize_ternary(rec["patterns"][n].to(torch.int8), rec["alphas_r3"][n].float(), 32) for n in keep}

def gen(label, tern):
    bkp_params = {}
    mods = dict(model.named_modules())
    bkp = apply_dequantized(model, dqs, tern)
    for n, p in nb.items():
        mod = model
        *pre, last = n.split(".")
        for q in pre:
            mod = getattr(mod, q)
        tgt = getattr(mod, last)
        bkp_params[n] = tgt.detach().clone()
        tgt.copy_(p.to(tgt.dtype))
    with torch.no_grad():
        out = model.generate(tok(PROMPT, return_tensors="pt")["input_ids"],
                             max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    restore(model, bkp)
    for n, v in bkp_params.items():
        mod = model
        *pre, last = n.split(".")
        for q in pre:
            mod = getattr(mod, q)
        getattr(mod, last).copy_(v)
    print(f"--- {label} ---")
    print(tok.decode(out[0].tolist()), flush=True)

gen("FP32-original", [])
gen("R0-TBC-puro", names)
gen("R3-recuperado", keep)
print("DONE")
