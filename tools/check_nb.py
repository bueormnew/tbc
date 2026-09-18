import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
nb = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2_r\norms_bias.pt", map_location="cpu")
ks = list(nb.keys())
print(len(ks))
for k in ks[:12]:
    print(k, tuple(nb[k].shape))
