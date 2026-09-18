import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import tbc
print("import ok", tbc.__version__)
from tbc.linalg import weight_out_in, write_weight_out_in
import torch, torch.nn as nn
m = nn.Linear(8, 4)
assert weight_out_in(m).shape == (4, 8)
print("linalg linear ok")
