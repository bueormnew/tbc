#!/usr/bin/env python
"""make_tables.py — tablas comparativas markdown desde los JSONs de resultados.

Lee (si existen): tbc_output/manifest.json, tbc_output_std/summary.json,
tbc_output_gpt2/summary.json, tbc_output_gpt2_r/recovery.json,
tbc_output_r/recovery.json, logs/bench.json, logs/bench_gpt2.json.
Imprime las tablas del README para verificarlas contra los datos.
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(*parts):
    p = os.path.join(BASE, *parts)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    nano = load("tbc_output", "manifest.json")
    std = load("tbc_output_std", "summary.json")
    gpt2 = load("tbc_output_gpt2", "summary.json")
    rec_n = load("tbc_output_r", "recovery.json")
    rec_g = load("tbc_output_gpt2_r", "recovery.json")
    bench = load("logs", "bench.json")
    bench_g = load("logs", "bench_gpt2.json")

    print("## NanoDex-1M: compilación (PPL torch 4x32)")
    if nano:
        lt = nano.get("layer_tolerances", {})
        es = list(lt.values()) if isinstance(lt, dict) else []
        g = sum(es) / len(es) if es else float("nan")
        print(f"- E medio por capa: {g:.4f} | pico RAM: {nano.get('peak_ram_gb')}GB | "
              f"tiempo: {nano.get('compile_time_s')}s | status: {nano.get('status')}")
    if std:
        print(f"- Standard: E={std['global_E']:.4f} PPL fp/rtn/tbc="
              f"{std['ppl_fp16']:.2f}/{std['ppl_rtn']:.2f}/{std['ppl_tbc']:.2f} "
              f"pico={std['peak_ram_gb']}GB")
    print("## TBC-R (PPL torch + bits efectivos)")
    for label, r in [("NanoDex", rec_n), ("GPT-2", rec_g)]:
        if r:
            print(f"- {label}: " + " | ".join(f"{k}={v:.2f}" for k, v in r["stages"].items())
                  + f" | eff={r['eff_bits']} b/p")
    print("## bitnet.cpp (llama-bench pp/tg + pico RSS)")
    for label, b in [("NanoDex", bench), ("GPT-2", bench_g)]:
        if b:
            for m, d in b.items():
                print(f"- {label} {m}: {d['file_mb']}MB pp={d['pp_ts']} tg={d['tg_ts']} "
                      f"rss={d['peak_rss_mb']}MB ppl={d.get('ppl_bin')}")
    print("## GPT-2 small: E por tipo de capa")
    if gpt2:
        from collections import defaultdict
        by = defaultdict(list)
        for n, e in gpt2["per_layer_E"].items():
            by[n.split(".")[-1]].append(e)
        for k, v in sorted(by.items()):
            print(f"- {k}: n={len(v)} mean={sum(v)/len(v):.4f}")


if __name__ == "__main__":
    main()
