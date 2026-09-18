import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import json
p = r"C:\Users\gerso\Desktop\TBC\logs\bench_gpt2.json"
b = json.load(open(p))
b["base-FP32"]["ppl_bin"] = 96.4896
b["TBC-R-F32"]["ppl_bin"] = 285.3679
b["TBC-R-I2_S"]["ppl_bin"] = 3769.1058
json.dump(b, open(p, "w"), indent=2)
print("bench_gpt2.json + PPL")
