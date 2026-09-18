"""Benchmark GPT-2 small: llama-bench (pp/tg) + pico RSS.
Modelos: base FP32 pura, R3-F32, R3-I2_S. PPL ya medida (ver logs).
"""
import json
import os
import re
import subprocess
import time

import psutil

BASE = r"C:\Users\gerso\Desktop\TBC"
BIN = os.path.join(BASE, "third_party", "BitNet", "build", "bin")
MODELS = [
    ("base-FP32", os.path.join(BASE, "tbc_output_gpt2", "gpt2-pure3-f32.gguf")),
    ("TBC-R-F32", os.path.join(BASE, "tbc_output_gpt2_r", "gpt2-r-f32.gguf")),
    ("TBC-R-I2_S", os.path.join(BASE, "tbc_output_gpt2_r", "gpt2-r-i2s.gguf")),
]


def peak_run(cmd, timeout=900):
    peak = 0
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                tot = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
                peak = max(peak, tot)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.05)
    finally:
        out = p.communicate()[0]
    return peak, out


def main():
    res = {}
    for label, path in MODELS:
        print(f"[bench-gpt2] {label} ...", flush=True)
        cmd = [os.path.join(BIN, "llama-bench.exe"), "-m", path, "-p", "128", "-n", "128", "-t", "4"]
        peak, out = peak_run(cmd)
        pp = tg = None
        for line in out.splitlines():
            m = re.search(r"\|\s*pp\s*\d+\s*\|\s*([\d.]+)", line)
            if m:
                pp = float(m.group(1))
            m = re.search(r"\|\s*tg\s*\d+\s*\|\s*([\d.]+)", line)
            if m:
                tg = float(m.group(1))
        r = {"peak_rss_mb": round(peak / 1e6, 1), "pp_ts": pp, "tg_ts": tg,
             "file_mb": round(os.path.getsize(path) / 1e6, 1)}
        print(f"[bench-gpt2] {label}: pp={pp} tg={tg} rss={r['peak_rss_mb']}MB", flush=True)
        res[label] = r
    json.dump(res, open(os.path.join(BASE, "logs", "bench_gpt2.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
