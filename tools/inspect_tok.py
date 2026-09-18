import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
from transformers import AutoTokenizer
import json
tok = AutoTokenizer.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
bt = tok.backend_tokenizer
print("backend model type:", type(bt.model).__name__)
m = bt.model
d = m.to_dict() if hasattr(m, "to_dict") else {}
print("model keys:", list(d.keys()) if isinstance(d, dict) else type(d))
if isinstance(d, dict) and "merges" in d:
    print("n_merges:", len(d["merges"]), "first3:", d["merges"][:3])
v = tok.get_vocab()
print("vocab size:", len(v))
inv = sorted(v.items(), key=lambda kv: kv[1])
print("tok0:", repr(inv[0][0]), "tok1:", repr(inv[1][0]), "tok2:", repr(inv[2][0]))
print("pre_tokenizer:", type(bt.pre_tokenizer).__name__ if bt.pre_tokenizer else None)
print("decoder:", type(bt.decoder).__name__ if bt.decoder else None)
print("byte_level:", getattr(bt.pre_tokenizer, "to_dict", lambda: {})() if bt.pre_tokenizer else None)
print("encode Hello:", tok.encode("Hello, I am")[:12])
