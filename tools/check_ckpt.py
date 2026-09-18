import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
ck = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2\ckpt.pt", map_location="cpu", weights_only=False)
print("ckpt OK capas:", len(ck["per_layer_E"]))
es = sorted(ck["per_layer_E"].values())
print("E min/max/mean:", round(es[0], 4), round(es[-1], 4), round(sum(es)/len(es), 4))
