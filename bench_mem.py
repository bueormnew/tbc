"""Benchmark bitnet.cpp: llama-bench (pp/tg tok/s) + pico RSS + PPL binaria.

Modelos NanoDex: base FP32 pura, R3-F32, R3-I2_S (deliverable).
Salida: logs/bench.json + tabla.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

import psutil

BASE = r"C:\Users\gerso\Desktop\TBC"
BIN = os.path.join(BASE, "third_party", "BitNet", "build", "bin")
MODELS = [
    ("base-FP32", os.path.join(BASE, "tbc_output", "nano-pure-f32.gguf")),
    ("TBC-R-F32", os.path.join(BASE, "tbc_output_r", "nano-r-f32.gguf")),
    ("TBC-R-I2_S", os.path.join(BASE, "tbc_output", "model-i2_s.gguf")),
]
CORPUS = os.path.join(BASE, "data", "eval_corpus.txt")


def peak_run(cmd, timeout=300):
    """Ejecuta cmd midiendo pico RSS del proceso y sus hijos."""
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


def bench(model_path):
    cmd = [os.path.join(BIN, "llama-bench.exe"), "-m", model_path, "-p", "128", "-n", "256", "-t", "4"]
    peak, out = peak_run(cmd)
    pp = tg = None
    for line in out.splitlines():
        m = re.search(r"\|\s*pp\s*\d+\s*\|\s*([\d.]+)", line)
        if m:
            pp = float(m.group(1))
        m = re.search(r"\|\s*tg\s*\d+\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))
    return {"peak_rss_mb": round(peak / 1e6, 2), "pp_ts": pp, "tg_ts": tg, "raw_tail": out[-600:]}


def ppl(model_path):
    cmd = [os.path.join(BIN, "llama-perplexity.exe"), "-m", model_path, "-f", CORPUS,
           "-c", "256", "--threads", "4"]
    if sys.platform == "win32":
        import io
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
        out = p.stdout + p.stderr
    else:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        out = p.stdout + p.stderr
    m = re.search(r"Final estimate: PPL = ([\d.]+)", out)
    return float(m.group(1)) if m else None


def main():
    res = {}
    for label, path in MODELS:
        print(f"[bench] {label} ...", flush=True)
        b = bench(path)
        b["file_mb"] = round(os.path.getsize(path) / 1e6, 3)
        print(f"[bench] pp={b['pp_ts']} tg={b['tg_ts']} rss={b['peak_rss_mb']}MB", flush=True)
        b["ppl_bin"] = ppl(path)
        print(f"[bench] ppl={b['ppl_bin']}", flush=True)
        res[label] = b
    json.dump(res, open(os.path.join(BASE, "logs", "bench.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
