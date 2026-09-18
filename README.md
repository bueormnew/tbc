# TBC — Ternary Behavioral Compilation (+ TBC-R Recovery)

> **Compila cualquier LLM a pesos ternarios `{-1, 0, +1}` y recupéralo después:
> PPL de 12133 → 60 en GPT-2 small, ejecutable nativo en bitnet.cpp.**

TBC no redondea pesos: **compila el comportamiento** del modelo original en una
representación ternaria mediante búsqueda discreta con verificación conductual.
Y cuando el ternario puro no basta, **TBC-R** lo recupera por etapas
(adaptativo + aceptación global + destilación de escalas) sin reentrenar pesos
densos. Todo con presupuestos de memoria configurables y exportación a
**GGUF `I2_S` que carga y corre en [bitnet.cpp](https://github.com/microsoft/BitNet)**.

Leer en otro idioma: **Español** (este archivo) · [English](README_EN.md)

## Inicio rápido

```bash
pip install -r requirements.txt

# Cuantizar CUALQUIER modelo causal de HuggingFace (arquitectura autodetectada)
python -m tbc quantize --model Qwen/Qwen2.5-7B --out ./qwen25-7b-tbc --beam 8
python -m tbc quantize --model gpt2 --out ./gpt2-tbc --ram auto

# Recuperarlo (R1 adaptativo + R2 global + R3 destilación)
python -m tbc recover --model gpt2 --tbc ./gpt2-tbc/model.tbc --out ./gpt2-r

# Ver qué arquitecturas soporta este checkout
python -m tbc arch-info            # lista (23 arquitecturas)
python -m tbc arch-info --arch qwen3moe
```

```bash
# Exportar a GGUF y correr nativo en bitnet.cpp
python export_llama_gguf.py  <model.tbc>  out-f32.gguf  out-i2s.gguf   # familia llama/qwen/...
python export_gpt2_gguf.py   <model.tbc>  out-f32.gguf  out-i2s.gguf   # GPT-2
third_party/BitNet/build/bin/llama-cli -m out-i2s.gguf -p "Hello" -n 50 --threads 4
```

Presupuestos (`--ram/--vram`, o `TBC_RAM_BUDGET`): `'auto'` (80% disponible,
por defecto), `'6GB'`, `'24GB'`… El compilador monitoriza el pico y falla si
lo excedes; sin límite duro salvo el que tú fijes.

## Resultados principales

### Recuperación TBC-R (PPL torch, misma eval 4×32 en cada modelo)

| Etapa | NanoDex-1M (llama, 1M) | GPT-2 small (124M) |
|---|---|---|
| FP32 original | 45.95 | 22.62 |
| R0 ternario puro | 361.88 | 12133.87 |
| R1 adaptativo (capas imposibles → FP) | 361.88 (0 promovidas) | 810.74 (19 promovidas) |
| R2 aceptación global RTN-vs-TBC | 353.45 | 785.49 |
| **R3 destilación de escalas** | **54.57** (26k params, 37 s, **1.19 b/p**) | **60.57** (1.69M params, ~12 min) |

### Ejecución nativa en bitnet.cpp (PPL binaria, corpus ctx-256)

| Modelo | FP32 | R3-F32 | R3-I2_S | Genera tokens |
|---|---|---|---|---|
| NanoDex-1M | 552.8 (5.37 MB) | **111.9** | **354.8** (3.07 MB) | ✅ 50 toks, 1665 tok/s |
| GPT-2 small | 96.5 (654 MB) | **285.4** | **3769** (466 MB) | ✅ 40 toks |

### Velocidad y memoria en bitnet.cpp (`llama-bench`, 4 hilos CPU)

| Modelo 124M | Archivo | Prompt | Generación | Pico RAM |
|---|---|---|---|---|
| Base FP32 | 654 MB | 847 tok/s | 60.1 tok/s | 541 MB |
| TBC-R I2_S | **466 MB (−29%)** | **992 tok/s (+17%)** | **82.3 tok/s (+37%)** | **354 MB (−35%)** |

En 1M el I2_S ahorra tamaño/RAM sin cambiar velocidad (régimen de overhead);
a 124M ya acelera. Detalle y metodología: §7.

## Características

- **Compilador conductual**: TraceGenerator, BehaviorCache de memoria fija,
  SensitivityAnalyzer, búsqueda discreta (coordinate + beam K≤16 +
  branch-and-bound + evaluación incremental `Y+XΔW` + rank-1 exacto),
  EquivalenceChecker en cascada, tolerancia progresiva, backtracking y mapa de
  imposibilidad cuando ε es inalcanzable.
- **TBC-R**: pipeline R0→R3 con early-stop y reversión por etapa.
- **23 arquitecturas** (llama, qwen/2/3(+MoE), mistral, mixtral, gemma/2/3,
  phi/3, falcon, gpt2, gpt_neox, gpt_bigcode, bloom, mpt, deepseek_v2,
  olmo/2, glm) con niveles `stable/compatible/experimental` y tests sin
  descargas (`tests/test_arch.py`: 13/13 + 2 smokes + MoE).
- **bitnet.cpp nativo**: GGUF `I2_S` (tipo 36) intercalado-128 verificado
  contra fuentes; kernel-test con diff 0.00; exportador auditado 149/149
  byte-exacto vs HF.
- **Reproducible**: semillas fijas, `scripts/reproduce.py` (NanoDex de cero a
  tabla en ~10 min), `tools/make_tables.py` reimprime estas tablas desde los
  JSON, todos los logs versionados.

## Cómo funciona (resumen)

1. **Compila**: ejecuta el modelo FP, registra trazas con checkpoints
   dispersos, estima sensibilidad por capa y busca configuraciones ternarias
   que reproduzcan activaciones y logits dentro de ε. Escalas α analíticas.
2. **Verifica**: cascada norma→coseno→top-k→KL; si una capa no llega, queda
   marcada en el mapa de imposibilidad en vez de fingir éxito.
3. **Recupera** (TBC-R): promueve capas imposibles, elige patrones por NLL
   global y destila escalas/normas/biases contra el teacher.
4. **Exporta**: GGUF F32 o I2_S nativo + manifiesto reproducible.

## Arquitecturas

| Arquitectura | Estado | Notas |
|---|---|---|
| llama | stable | Validado end-to-end (NanoDex-1M). |
| gpt2 | stable | Conv1D, QKV fusionada, wpe. Validado end-to-end (GPT-2 small). |
| qwen2, qwen3, mistral, gemma2, mixtral, phi3, falcon, gpt_neox, bloom, olmo2 | compatible | Detección + compilación probadas en modelos sintéticos; nombres GGUF según conversor oficial; validar con pesos reales. MoE: expertos sí, router FP. |
| qwen, qwen2moe, qwen3moe, gemma, gemma3, phi, gpt_bigcode, mpt, deepseek_v2, olmo, glm | experimental | Particularidades (MLA, fusiones no estándar, layernorms) pendientes de validación con pesos reales. |

`python -m tbc arch-info --arch <nombre>` muestra lineales objetivo,
exclusiones, formato GGUF y notas de cada una.

## Conexión con bitnet.cpp

1. Clona y compila (una vez): `third_party/README.md`
   (`clone --recursive` + patch de 1 línea + `cmake`; binarios en
   `third_party/BitNet/build/bin/`). Ya verificado: 406/406 targets.
2. Exporta (§4 guías en versión extendida: `TBC_REAL_TEST_REPORT.md`).
3. Carga y genera: `llama-cli -m modelo-i2s.gguf -p "..." -n 50`.

Formato exigido por el runtime (verificado en fuentes, detalle en
`docs/HISTORIA-Y-DETALLE.md` §3): I2_S=36, intercalado-128, 1 escala
f32 por tensor, permute RoPE en q/k (llama), QKV fusionada (gpt2), biases F32,
`ne0 % 128 == 0` o F32 mixto, tokenizador embebido con `pre` correcto.

## Guías de uso

### Requisitos

```text
Python ≥ 3.10, torch, transformers, safetensors, datasets, psutil, numpy
CMake ≥ 3.22 + Clang/GCC + Ninja (solo para compilar bitnet.cpp)
```

### CLI

```bash
python -m tbc quantize --model <id-HF> --out <dir> [--beam 8] [--mode standard]
                       [--ram auto|6GB|24GB] [--vram auto] [--group 32]
                       [--calib 256] [--seqlen 32] [--eps 0.05]
python -m tbc recover  --model <id-HF> --tbc <model.tbc> --out <dir-r>
                       [--adapt 0.65] [--steps 300]
python -m tbc arch-info [--arch <nombre>]
```

Variables: `TBC_RAM_BUDGET`, `TBC_DTYPE=float32|float16|bfloat16`
(para modelos grandes usa `TBC_DTYPE=float16`).

### Pipeline completo y utilidades

```bash
python scripts/reproduce.py              # NanoDex Fast de cero a tabla (~10 min)
python run_standard_test.py              # TBC-Standard (beam 8, cross-layer)
python run_gpt2_test.py                  # GPT-2 124M (ckpt/resume automático)
python run_recovery.py [nanodex|gpt2]    # pipeline TBC-R clásico con reportes
python tools/make_tables.py              # tablas desde los JSON de resultados
python tests/test_synthetic.py && python tests/test_arch.py
```

## Resultados detallados y bitácora

- `TBC_REAL_TEST_REPORT.md`: NanoDex Fase 1+2+TBC-R con per-layer E.
- `README` §8/§9 anteriores (casos detallados por capa, incidentes de
  ingeniería, metodología fina) se conservan íntegros en
  `docs/HISTORIA-Y-DETALLE.md`; la evidencia por corrida vive en `logs/`
  (sintéticos, compilaciones, PPL torch y binaria, generaciones, benchmarks,
  kernel-test, verifies).
- `tbc_output*/`: manifests, `summary.json`, `recovery.json`
  (los pesos/GGUF se regeneran; no se versionan por tamaño).

## Limitaciones conocidas

- El ternario puro post-training no alcanza ε=0.05 en los modelos probados;
  TBC-R existe precisamente para eso. Ver mapa de imposibilidad.
- El formato I2_S admite 1 escala por tensor: colapsar las α por grupo cuesta
  ×3–13 en PPL binaria frente a torch. Un formato con α por grupo lo eliminaría.
- El speedup I2_S aparece cuando el matmul domina (124M: +37%); en modelos
  diminutos solo hay ahorro de tamaño/RAM.
- Arquitecturas `experimental`: compilan, pero valida el GGUF con
  `verify_gguf.py` + una PPL binaria antes de confiar.

## Licencia

MIT — ver [LICENSE](LICENSE). bitnet.cpp es de Microsoft (MIT); no se incluye
aquí, ver `third_party/README.md`.
