import struct
import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
import numpy as np
from transformers import GPT2LMHeadModel

PATH = r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2\gpt2-pure2-f32.gguf"
with open(PATH, "rb") as f:
    data = f.read()
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
infos = {}
for _ in range(nt):
    name = rs().decode()
    (nd,) = struct.unpack("<I", data[pos:pos+4]); pos += 4
    dims = [struct.unpack("<Q", data[pos+8*i:pos+8*i+8])[0] for i in range(nd)]; pos += 8*nd
    (tt,) = struct.unpack("<I", data[pos:pos+4]); pos += 4
    (off,) = struct.unpack("<Q", data[pos:pos+8]); pos += 8
    infos[name] = (dims, tt, off)
dstart = pos + (32 - (pos % 32)) % 32

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
W = model.transformer.h[0].attn.c_attn.weight.detach().float()  # [in,out]=[768,2304]
name, (dims, tt, off) = "blk.0.attn_qkv.weight", infos["blk.0.attn_qkv.weight"]
print("dims en GGUF:", dims, "tipo:", tt)
n = W.numel()
blob = data[dstart+off:dstart+off+n*4]
a1 = np.frombuffer(blob, dtype="<f4").reshape(2304, 768)  # hipótesis [out,in]
a2 = np.frombuffer(blob, dtype="<f4").reshape(768, 2304)  # hipótesis [in,out]
Wo = W.numpy()
print("diff vs [out,in] :", float(np.abs(a1 - Wo.T).max()))
print("diff vs [in,out] :", float(np.abs(a2 - Wo).max()))
