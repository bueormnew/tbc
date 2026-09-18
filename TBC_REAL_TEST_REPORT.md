# TBC_REAL_TEST_REPORT (Fase 1 + Fase 2 + TBC-R)

Modelo: `DedeProGames/NanoDex-1M` (descarga real HF, arch=llama, params=1062272)
Límite: 6GB RAM/VRAM en todas las fases (picos medidos abajo, todos <6GB).

## 1. Compilación y PPL torch (texto inglés real, 4×32, misma eval)

| Config | FP16 | RTN-1.58 | TBC-1.58 | E medio | Tiempo | Pico RAM | Status |
|---|---|---|---|---|---|---|---|
| Fast (K=4, calib texto 16×64) | 45.95 | 377.30 | **361.88** (−4.1%) | 0.408 | ~23 s | 0.51 GB | COMPILED_WITH_WARNINGS |
| Standard (K=8, calib 1024×16, cross-layer) | 45.95 | 377.30 | 385.15 (+2.1%) | 0.413 | ~13 min | 1.32 GB | COMPILED_WITH_WARNINGS |

Representación idéntica (grupo 32, ternario puro). Standard no mejora:
sobreajuste a calibración con búsqueda agresiva (documentado en README §5).
Detalle por capa y mapa de imposibilidad (Sec 34): `tbc_output/manifest.json`
(`layer_tolerances`, `impossibility_map`, 35/35 capas sobre ε=0.05).

## 2. Ejecución nativa bitnet.cpp (binarios compilados 406/406, Sec 39)

Verificación: binario + kernel-test (`bitnet_cpp_compatible: true` en manifest).

| Artefacto | PPL binaria (`llama-perplexity`, corpus ctx-256) | Tamaño |
|---|---|---|
| `tbc_output_r/nano-r-f32.gguf` (R3, α+normas entrenadas) | **111.9** (4.9× mejor que R0) | 5.37 MB |
| `model-i2_s.gguf` = **NanoDex-R** (30×I2_S R2/R3 + 5×F32) | **354.8** (3.7× mejor que R0) | 3.07 MB |
| FP32 torch (referencia, ventanas 32) | 51.6 | 4.25 MB torch |

- `model-i2_s.gguf` (**recuperado**, sha256 `e273978f…8515fd`) carga en
  `llama-cli.exe` y genera 50 tokens (1665 tok/s): `logs/nano_r_gen.log`.
- Kernel-test: decodificador independiente del GGUF → patrones 100% TBC,
  max|Y_kernel−Y_torch| = 0.00 < 1e-3: `logs/kernel_test.log`.
- Formato verificado contra fuentes: I2_S=36 (no 30), intercalado-128,
  escala f32/tensor, permute RoPE q/k, tokenización binaria == HF.
  Detalle completo: README §3. Logs: `logs/bitnet_compile.log`,
  `logs/ppl_*_bin*.log`, `logs/verify_*.log`.

## 3. TBC-R: recuperación por etapas (algoritmo nuevo, `tbc/recovery.py`)

| Etapa | PPL torch (mismo texto) | PPL binaria (corpus ctx-256) |
|---|---|---|
| FP32 original | 45.95 | F32-GGUF: 552.8 |
| R0 TBC puro | 361.88 | I2_S: ~1300 |
| R1 adaptativo (0 promovidas, E<0.65) | 361.88 | — |
| R2 aceptación global | 353.45 | — |
| **R3 destilación** (26k params, 37 s) | **54.57** (1.19× FP32, 1.19 b/p) | **F32: 111.9 / I2_S: 354.8** |

R3 recupera 6.6× con 26k parámetros entrenables (α + normas + biases) en
37 segundos. Detalle y caso GPT-2: README §9. Artefactos:
`tbc_output_r/recovered.pt`, `tbc_output_r/recovery.json`. Logs:
`logs/recovery_nano.log`, `logs/recovery_gpt2.log`.
