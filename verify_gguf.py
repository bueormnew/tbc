"""Verifica blob por blob un GGUF F32 contra el state_dict HF (detecta offsets/layout)."""
import struct
import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2\gpt2-pure2-f32.gguf"

with open(PATH, "rb") as f:
    data = f.read()
assert data[:4] == b"GGUF"
pos = 24
nt = struct.unpack("<Q", data[8:16])[0]
nkv = struct.unpack("<Q", data[16:24])[0]
def rs():
    global pos
    (ln,) = struct.unpack("<Q", data[pos:pos+8]); pos += 8
    s = data[pos:pos+ln]; pos += ln
    return s
def sk(t):
    global pos
    S = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t == 8:
        (ln,) = struct.unpack("<Q", data[pos:pos+8]); pos += 8 + ln
    elif t == 9:
        et = struct.unpack("<I", data[pos:pos+4])[0]; pos += 4
        (ln,) = struct.unpack("<Q", data[pos:pos+8]); pos += 8
        for _ in range(ln):
            sk(et)
    else:
        pos += S[t]
for _ in range(nkv):
    rs(); t = struct.unpack("<I", data[pos:pos+4])[0]; pos += 4; sk(t)
infos = []
for _ in range(nt):
    name = rs().decode()
    (nd,) = struct.unpack("<I", data[pos:pos+4]); pos += 4
    dims = [struct.unpack("<Q", data[pos+8*i:pos+8*i+8])[0] for i in range(nd)]; pos += 8*nd
    (tt,) = struct.unpack("<I", data[pos:pos+4]); pos += 4
    (off,) = struct.unpack("<Q", data[pos:pos+8]); pos += 8
    infos.append((name, dims, tt, off))
dstart = pos + (32 - (pos % 32)) % 32
print("tensores:", len(infos), "data_start:", dstart)

from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
state = model.state_dict()

def hf_tensors():
    out = {}
    for i in range(12):
        p = f"transformer.h.{i}."
        W = state[p+"attn.c_attn.weight"].detach().float().T  # [out,in]
        out[f"blk.{i}.attn_qkv.weight"] = W
        out[f"blk.{i}.attn_qkv.bias"] = state[p+"attn.c_attn.bias"].detach().float()
        out[f"blk.{i}.attn_output.weight"] = state[p+"attn.c_proj.weight"].detach().float().T
        out[f"blk.{i}.attn_output.bias"] = state[p+"attn.c_proj.bias"].detach().float()
        out[f"blk.{i}.ffn_up.weight"] = state[p+"mlp.c_fc.weight"].detach().float().T
        out[f"blk.{i}.ffn_up.bias"] = state[p+"mlp.c_fc.bias"].detach().float()
        out[f"blk.{i}.ffn_down.weight"] = state[p+"mlp.c_proj.weight"].detach().float().T
        out[f"blk.{i}.ffn_down.bias"] = state[p+"mlp.c_proj.bias"].detach().float()
        out[f"blk.{i}.attn_norm.weight"] = state[p+"ln_1.weight"].detach().float()
        out[f"blk.{i}.attn_norm.bias"] = state[p+"ln_1.bias"].detach().float()
        out[f"blk.{i}.ffn_norm.weight"] = state[p+"ln_2.weight"].detach().float()
        out[f"blk.{i}.ffn_norm.bias"] = state[p+"ln_2.bias"].detach().float()
    out["token_embd.weight"] = state["transformer.wte.weight"].detach().float()
    out["position_embd.weight"] = state["transformer.wpe.weight"].detach().float()
    out["output_norm.weight"] = state["transformer.ln_f.weight"].detach().float()
    out["output_norm.bias"] = state["transformer.ln_f.bias"].detach().float()
    out["output.weight"] = state["transformer.wte.weight"].detach().float()
    return out

ref = hf_tensors()
bad = 0
for name, dims, tt, off in infos:
    if name not in ref:
        print("EXTRA", name); bad += 1; continue
    w = ref[name]
    n = w.numel()
    blob = data[dstart+off:dstart+off+n*4]
    if len(blob) != n*4:
        print("SIZE", name, len(blob), n*4); bad += 1; continue
    arr = np.frombuffer(blob, dtype="<f4").reshape(w.shape)
    d = float(np.abs(arr - w.numpy()).max())
    flag = "OK " if d == 0.0 else "DIFF"
    if d != 0.0:
        bad += 1
        print(flag, name, dims, "maxdiff=", d)
print("revisados:", len(infos), "mal:", bad)
