import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import json
p = r"C:\Users\gerso\Desktop\TBC\tbc_output_gpt2\summary.json"
s = json.load(open(p))
s["ppl_tbc"] = 12133.87
s["ppl_note"] = ("12133.87 medido 2x independiente (summary original + hybrid_demo); "
                 "11678.52 visto una vez en log de run con workers duplicados: outlier descartado")
s["hybrid_tbca"] = {"sin_c_proj_24_48": 625.64, "solo_c_fc_12_48": 477.05}
json.dump(s, open(p, "w"), indent=2)
print("summary.json corregido: ppl_tbc=12133.87 + hybrid")
