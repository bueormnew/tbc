import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import json
from collections import defaultdict
s = json.load(open(r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2\summary.json"))
print("keys:", list(s.keys()))
print("global_E:", s["global_E"], "status:", s["status"])
print("ppl:", s["ppl_fp32"], s["ppl_rtn"], s["ppl_tbc"])
print("peak:", s["peak_ram_gb"], "compile_s:", s["compile_time_s"], "total_s:", s["total_s"])
by = defaultdict(list)
for n, e in s["per_layer_E"].items():
    by[n.split(".")[-1]].append(e)
for k, v in by.items():
    print(k, "n=%d mean=%.4f min=%.4f max=%.4f" % (len(v), sum(v)/len(v), min(v), max(v)))
