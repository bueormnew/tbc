import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import json
import re
from collections import defaultdict
from datetime import datetime
per, tim = {}, {}
for fn in ("logs/gpt2_test.log", "logs/gpt2_test2.log"):
    for ln in open(r"C:\Users\gerso\Desktop\TBC\\" + fn, encoding="utf-8", errors="replace"):
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*capa=(\S+) S=[\d.]+ E=([\d.]+) \(init [\d.]+\) evals=\d+ pruned=\d+ t=([\d.]+)s", ln)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            key = m.group(2)
            if key not in per or ts >= per[key][0]:
                per[key] = (ts, float(m.group(3)))
                tim[key] = float(m.group(4))
E = {k: v[1] for k, v in per.items()}
print("capas totales:", len(E))
by = defaultdict(list)
for n, e in E.items():
    by[n.split(".")[-1]].append(e)
for k, v in sorted(by.items()):
    print(k, "n=%d mean=%.4f min=%.4f max=%.4f" % (len(v), sum(v)/len(v), min(v), max(v)))
g = sum(E.values())/len(E)
print("global_E:", round(g, 4))
s = json.load(open(r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2\summary.json"))
s["per_layer_E"] = E
s["per_layer_t"] = tim
s["global_E"] = g
s["status"] = "COMPILED_WITH_WARNINGS"
s["ppl_fp32"] = 22.61526597888981
s["ppl_rtn"] = 12223.05362326909
s["ppl_tbc"] = 11678.519
s["peak_ram_gb"] = 1.835
json.dump(s, open(r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2\summary.json", "w"), indent=2)
print("summary.json OK")
