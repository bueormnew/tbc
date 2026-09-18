#!/usr/bin/env python
"""reproduce.py — pipeline TBC completo en NanoDex-1M, de cero a tabla.

Etapas (las mismas de la bitácora del proyecto):
  1. tests sintéticos (5/5)
  2. compilación TBC Fast + PPL FP32/RTN/TBC  (run_real_test.py)
  3. exportación GGUF nativa I2_S          (export_llama_gguf.py)
  4. kernel-test contra el GGUF            (kernel_test.py, necesita tbc_output/)
  5. validación binaria (si existe third_party/BitNet/build, si no se salta)
  6. tabla comparativa                     (tools/make_tables.py)

Uso:  python reproduce.py [--skip-binary]
Tarda ~5-10 min en CPU. Límite 6GB verificado por el propio compilador.
"""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(BASE, "third_party", "BitNet", "build", "bin", "llama-cli.exe")
SKIP_BIN = "--skip-binary" in sys.argv


def run(cmd, log):
    print(f"\n===== {' '.join(cmd)} =====", flush=True)
    env = dict(os.environ, PYTHONPATH=BASE, PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(cmd, cwd=BASE, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"rc={r.returncode} -> {log}", flush=True)
    if r.returncode != 0:
        sys.exit(f"FALLO en {cmd} (ver {log})")


def main():
    py = sys.executable
    run([py, "tests/test_synthetic.py"], "logs/synthetic.log")
    run([py, "run_real_test.py"], "logs/real_test.log")
    run([py, "export_llama_gguf.py"], "logs/export_gguf.log")
    run([py, "kernel_test.py"], "logs/kernel_test.log")
    if os.path.exists(BIN) and not SKIP_BIN:
        gguf = os.path.join(BASE, "tbc_output", "nanodex-tbc-i2s.gguf")
        run([BIN, "-m", gguf, "-f", "prompt.txt", "-n", "50", "--threads", "4"],
            "logs/bitnet_cpp_run.log")
    else:
        print("binario bitnet.cpp ausente: ver third_party/README.md para compilarlo")
    run([py, "tools/make_tables.py"], "logs/tables.log")
    print("\nPIPELINE OK. Ver TBC_REAL_TEST_REPORT.md y README.md")


if __name__ == "__main__":
    main()
