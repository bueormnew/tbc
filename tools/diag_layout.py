"""Diagnóstico: ¿lee el binario I2_S secuencial o intercalado? Compara PPL torch."""
import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tbc.perplexity import perplexity, apply_dequantized, restore
from tbc.compiler import get_ternarize_targets
from tbc.export import pack_i2_s_native, unpack_i2_s_native, optimal_global_scale

model = AutoModelForCausalLM.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True, dtype=torch.float32)
model.eval()
tok = AutoTokenizer.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
text = ("Hello, I am a small language model. The quick brown fox jumps over the lazy dog. " * 6).strip()
ids = tok(text, return_tensors="pt")["input_ids"]
windows = [ids[:, i:i+32] for i in range(0, ids.shape[1], 32) if ids[:, i:i+32].shape[1] == 32]
eval_ids = torch.cat(windows[:4], dim=0)

blob = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output\model.tbc\weights.pt", map_location="cpu")
targets = get_ternarize_targets(model, "llama")
names = [n for n, _ in targets]
mods = dict(model.named_modules())

def deq_seq(Wt, scale, shape):
    numel = Wt.numel()
    b = pack_i2_s_native(Wt, scale)
    back, s = unpack_i2_s_native(b, numel)
    assert abs(s - scale) < 1e-6 and bool((back.flatten() == Wt.flatten()).all())
    return (back.float().reshape(shape) * s)

def deq_interleaved_read(Wt_seq, scale, shape):
    """Lee el blob SECUENCIAL como si fuera intercalado-128 (mirror dequantize_row_i2_s)."""
    rows, cols = shape
    flat = Wt_seq.flatten()
    n = flat.numel()
    assert n % 128 == 0
    out = torch.empty(n, dtype=torch.float32)
    # blob secuencial: byte b=i//4 bits 6-2*(i%4), códigos 0/1/2 -> -1/0/+1
    codes = []
    for i in range(n):
        v = int(flat[i].item())
        codes.append({-1: 0, 0: 1, 1: 2}[v])
    packed = bytearray(n // 4)
    for i, c in enumerate(codes):
        packed[i // 4] |= (c & 3) << (6 - 2 * (i % 4))
    mp = [-1.0, 0.0, 1.0, 0.0]
    done = 0
    while done < n:
        for gp in range(32):
            byte = packed[(done // 4) + gp]
            c0, c1, c2, c3 = (byte >> 6) & 3, (byte >> 4) & 3, (byte >> 2) & 3, byte & 3
            if done + gp < n: out[done + gp] = scale * mp[c0]
            if done + 32 + gp < n: out[done + 32 + gp] = scale * mp[c1]
            if done + 64 + gp < n: out[done + 64 + gp] = scale * mp[c2]
            if done + 96 + gp < n: out[done + 96 + gp] = scale * mp[c3]
        done += 128
    return out.reshape(shape)

for mode in ["seq", "interleaved-misread"]:
    dq = {}
    for hn in names:
        base = hn
        Wt = blob[base]["W"].to(torch.int8)
        s = optimal_global_scale(mods[hn].weight.detach().float(), Wt)
        dq[hn] = deq_seq(Wt, s, Wt.shape) if mode == "seq" else deq_interleaved_read(Wt, s, Wt.shape)
    bkp = apply_dequantized(model, dq, names)
    ppl = perplexity(model, eval_ids)
    restore(model, bkp)
    print(f"PPL_torch_{mode} = {ppl:.2f}", flush=True)
print("DONE")
