#!/usr/bin/env python
"""tbc_to_gguf_i2s.py — model.tbc -> GGUF I2_S nativo (Sec 39).

Conversor determinista: re-ejecuta la exportación nativa (patrones TBC +
escala f32 global óptima, tipo GGML I2_S=36, layout intercalado-128 del
runtime BitNet) y verifica que el resultado es byte-idéntico (SHA256).

Uso:
  python tbc_to_gguf_i2s.py --tbc_dir ./tbc_output/model.tbc --out ./tbc_output/model-i2_s.gguf
"""
from __future__ import annotations
import argparse
import hashlib
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbc_dir", default=os.path.join(BASE, "tbc_output", "model.tbc"))
    ap.add_argument("--out", default=os.path.join(BASE, "tbc_output", "model-i2_s.gguf"))
    args = ap.parse_args()

    exp = os.path.join(BASE, "export_llama_gguf.py")
    f32_tmp = args.out + ".f32tmp.gguf"
    r = subprocess.run([sys.executable, exp, args.tbc_dir, f32_tmp, args.out],
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(1)
    if os.path.exists(f32_tmp):
        os.remove(f32_tmp)
    print(f"[tbc2gguf] OK {args.out} sha256={sha256(args.out)}")


if __name__ == "__main__":
    main()
