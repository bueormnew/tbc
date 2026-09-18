# TBC — Ternary Behavioral Compilation

> **"TBC no cuantiza pesos; compila una función neuronal en pesos ternarios."**

Compilador post-training que transforma `M_FP → M_T` con `W_T ∈ {-1, 0, +1}`
mediante búsqueda discreta y verificación conductual, más **TBC-R**, algoritmo
de recuperación por etapas desde cuantización extrema. Todo exportable a
**GGUF `I2_S` ejecutable de forma nativa en [bitnet.cpp](https://github.com/microsoft/BitNet)**.

Leer en otro idioma: **Español** (este archivo) · [English](README_EN.md)

---

## 1. Estado del proyecto

| Hito | Estado | Evidencia |
|---|---|---|
| Compilador completo (6 componentes, Sec 3/24) | ✅ | `tbc/` (14 módulos), sin stubs |
| Memoria fija < 6GB verificada | ✅ | pico Fast 0.5GB, Standard 1.3GB, GPT-2 1.8GB |
| 5 tests sintéticos | ✅ 5/5 | `logs/synthetic*.log` |
| NanoDex-1M (llama, 1M) TBC-Fast + Standard | ✅ | `tbc_output/`, `tbc_output_std/` |
| GPT-2 small (124M) TBC completo | ✅ | `tbc_output_gpt2/` + §8 |
| bitnet.cpp compilado (406/406 targets) | ✅ | `logs/bitnet_compile.log`, 48 binarios |
| GGUF I2_S carga y genera tokens reales | ✅ | `logs/bitnet_cpp_run.log`, `logs/gpt2_gen_i2s.log` |
| Kernel-test I2_S (diff 0.00 < 1e-3) | ✅ | `logs/kernel_test.log` |
| Benchmark bitnet (memoria + tok/s) | ✅ | §7, `logs/bench.json` |
| Exportador byte-exacto (149/149 tensores) | ✅ | `logs/verify_pure3.log` |

Arquitecturas soportadas end-to-end: **llama** (NanoDex-1M), **GPT-2**
(GPT2-small, vía Conv1D), **qwen2/qwen3** (mismo pipeline que llama).
Ver `tbc/arch.py`.

---

## 2. Resultados con NanoDex-1M (`DedeProGames/NanoDex-1M`, llama 5 capas, 1.06M params)

### 2.1 Perplejidad torch (texto inglés real, 4×32, idéntica eval)

| Config | FP16 | RTN-1.58 | TBC-1.58 | Mejora vs RTN |
|---|---|---|---|---|
| Fast (K=4, calib texto 16×64) | 45.95 | 377.30 | **361.88** | −4.1% |
| Standard (K=8, calib 1024×16, cross-layer) | 45.95 | 377.30 | 385.15 | +2.1% (peor) |

Misma representación (grupo 32, ternario puro): TBC-Fast supera al baseline
RTN. Standard no mejora (ver §5, hallazgo de sobreajuste a calibración).

### 2.2 Compilación

| Métrica | Fast | Standard |
|---|---|---|
| E medio por capa | 0.408 | 0.413 |
| Capas sobre ε=0.05 | 35/35 (COMPILED_WITH_WARNINGS + mapa Sec 34) | 35/35 |
| Tiempo | ~23 s | ~13 min |
| Pico RAM | 0.51 GB | 1.32 GB (límite 6 GB ✓) |

### 2.3 Ejecución nativa bitnet.cpp (binarios compilados de `third_party/BitNet`)

| Artefacto | PPL binaria (`llama-perplexity`, corpus, ctx 256) | Tamaño |
|---|---|---|
| `nanodex-tbc-f32.gguf` (R0, 48 tensores) | 552.8 | 5.37 MB |
| `model-i2_s.gguf` = **NanoDex-R** (30×I2_S R2/R3 + 5×F32, 48 tensores) | **354.8** (antes 1300 en R0: 3.7×) | 3.07 MB |
| `tbc_output_r/nano-r-f32.gguf` (R3, α+nomas entrenadas) | **111.9** (antes 552.8: 4.9×) | 5.37 MB |
| FP32 torch (referencia, ventanas 32) | 51.6 | 4.25 MB torch |

`model-i2_s.gguf` (**NanoDex-R recuperado**, sha256 `e273978f…8515fd`)
carga en `llama-cli.exe` y **genera 50 tokens** (1665 tok/s,
`logs/nano_r_gen.log`). Re-exportación determinista byte-idéntica
(`export_llama_gguf.py --recovered tbc_output_r`). El GGUF R0 previo
(`0cfff4fd…`, PPL ~1300) queda archivado como referencia en §9.1.

> Nota metodológica: la PPL binaria usa chunks ctx-256 y la torch ventanas
> de 32; los valores absolutos no son comparables entre metodologías, solo
> los órdenes relativos dentro de cada una. Ver §5 (efecto longitud).

---

## 3. Compatibilidad nativa bitnet.cpp (Sec 39) — lo verificado contra fuentes

Todo lo siguiente se verificó leyendo el código de `third_party/BitNet`, no el texto del prompt:

- `GGML_TYPE_I2_S = 36` (`3rdparty/llama.cpp/ggml/include/ggml.h:426`). El prompt decía 30: **corregido**.
- `quantize_i2_s` (`ggml/src/ggml-cpu/quants.c:1358`): packing **secuencial** en GGUF
  (peso `i` → byte `i/4`, bits `6-2*(i%4)`), códigos `0→−1, 1→0, 2→+1`,
  **una escala f32 por tensor** en offset `numel/4`, tamaño `numel/4+32` (`ggml.c:1316`).
- El runtime lee layout **intercalado-128** (`dequantize_row_i2_s`, `quants.c:1335`;
  `byte gp = c(gp)<<6|c(32+gp)<<4|c(64+gp)<<2|c(96+gp)`). Prueba decisiva:
  GGUF secuencial → PPL binaria 101828 (pesos permutados); GGUF intercalado →
  ~1240. El exportador emite intercalado-128.
- Restricción del fork: filas con `ne0` no múltiplo de 128 se leen fuera de fila;
  tensores así quedan F32 (cuantización mixta). NanoDex: 5 `ffn_down` (ne0=288).
  GPT-2 small: todos múltiplos → I2_S completo.
- Conversión q/k RoPE (`conversion/llama.py:163`, `permute`): aplicada a q/k en
  exportación llama (sin ella la PPL binaria se duplica: 997→553).
- Tokenizadores embebidos (tokens+scores+tipos+merges). Tokenización binaria
  verificada idéntica a HF (NanoDex: mismos ids; GPT-2: 577=577 tokens).
- Build Windows: VS2026 + LLVM-MinGW clang 22 + Ninja, flags oficiales de
  `setup_env.py` (`BITNET_X86_TL2=OFF`; con `=ON` main no linka:
  `ggml_bitnet_mul_mat` solo existe bajo `ARM_TL1`), fix de 1 línea
  (`mad.cpp` al CMake de `ggml-cpu`, ver `third_party/`), `_WIN32_WINNT=0x0A00`.

### Cómo conectar un modelo TBC a bitnet.cpp (checklist del loader)

1. `general.architecture` = `llama` / `gpt2` / `qwen2` + hiperparámetros del
   bloque (`block_count`, `embedding_length`, …) y `general.quantization_version = 2`.
2. Nombres de tensor del fork (`blk.i.attn_q.weight`, `attn_qkv` fusionada en
   GPT-2, `token_embd`, `position_embd` en GPT-2) **más biases** donde la
   arquitectura los tiene (al loader le faltan y lo dice: `missing tensor`).
3. Lineales en `I2_S` (36) intercalado-128 con trailer de 32 B; resto F32.
4. Tokenizador embebido con `model` + `pre` correctos (`gpt-2` para BPE
   byte-level) + tokens/scores/tipos/merges.
5. Si el loader rechaza, su mensaje (`missing tensor`, `bad magic`, conteos)
   dice exactamente qué falta: itera con él (así se depuró este proyecto).

---

## 4. Guías de uso

### 4.1 Requisitos

```text
Python ≥ 3.10, torch, transformers, safetensors, datasets, psutil, numpy
CMake ≥ 3.22 + Clang/GCC + Ninja (solo para compilar bitnet.cpp)
```

### 4.2 Cuantizar un modelo con TBC

```python
from tbc.config import TBCConfig
from tbc.compiler import TBC_COMPILE
from tbc.calib_text import text_windows  # calibración en distribución

cfg = TBCConfig(group_size=32, beam_size=4, epsilon_target=0.05,
                mode="fast", enable_backtracking=True)  # límite 6GB por defecto
calib = text_windows(tokenizer, n_seq=64, seq_len=64, stride=16)
tbc_model, report = TBC_COMPILE(model, cfg, calib_ids=calib,
                                ckpt_path="ckpt.pt")  # resume ante cortes
```

- `beam_size` 4 (Fast) / 8 (Standard) / 16 (Max, vía `mode="max"`).
- `enable_cross_layer_refinement=True` añade pasada Max sobre las 4 peores capas.
- Conv1D (GPT-2) se maneja solo (`tbc/linalg.py`, convención `[out,in]` interna).
- Si `min_D > ε`: estado `COMPILED_WITH_WARNINGS` + `impossibility_map` (Sec 34).

Scripts listos: `run_real_test.py` (NanoDex Fast), `run_standard_test.py`,
`run_gpt2_test.py` (cada uno: compila + PPL FP/RTN/TBC + artefactos + reporte).

### 4.3 Exportar a GGUF y cargar en bitnet.cpp

```bash
# 1. Compilar bitnet.cpp una vez (Windows: scripts\build_bitnet.bat;
#    Linux: mismo cmake sin vcvars — ver third_party/README.md)

# 2. Exportar (llama / gpt2; --pure vuelca pesos HF para depurar el exportador)
python export_llama_gguf.py  tbc_output/model.tbc  out-f32.gguf  out-i2s.gguf
python export_gpt2_gguf.py   tbc_output_gpt2/model.tbc out-f32.gguf out-i2s.gguf
python tbc_to_gguf_i2s.py --tbc_dir tbc_output/model.tbc --out tbc_output/model-i2_s.gguf

# 3. Ejecutar (I2_S nativo)
third_party\BitNet\build\bin\llama-cli.exe -m out-i2s.gguf -f prompt.txt -n 50 --threads 4
third_party\BitNet\build\bin\llama-perplexity.exe -m out-i2s.gguf -f texto.txt -c 256 --threads 4
```

Reglas del formato que el exportador ya aplica: I2_S=36 intercalado-128 con
escala f32 global óptima (`optimal_global_scale`, Sec 22 colapsada a 1 escala
por tensor), q/k con permute RoPE (llama), QKV fusionada (gpt2), biases F32
intactos, `ne0 % 128 != 0` → F32 mixto.

### 4.4 Verificar una exportación (recomendado antes de publicar un GGUF)

```bash
python verify_gguf.py <f32.gguf>   # 149/149 byte-exactos vs HF = OK
python kernel_test.py              # patrones 100% + diff Y < 1e-3
```

### 4.5 Pipeline completo por fases (cómo se obtuvo cada resultado)

```bash
python scripts/reproduce.py              # NanoDex Fast de cero a tabla (~10 min)
python run_standard_test.py              # TBC-Standard (beam 8, cross-layer)
python run_gpt2_test.py                  # GPT-2 small 124M (~80 min, con ckpt/resume)
python run_recovery.py nanodex           # TBC-R NanoDex (~1 min)
python run_recovery.py gpt2              # TBC-R GPT-2 (~12 min)
python tools/make_tables.py              # reimprime las tablas desde los JSON
```

Fases de la bitácora: 0) especificación → 1) compilador + tests + NanoDex →
2) bitnet.cpp nativo + Standard → 3) escala GPT-2 → 4) algoritmo TBC-R →
5) benchmarks → 6) publicación. Cada fase dejó sus logs en `logs/`.

---

## 5. Hallazgos honestos (límites, no solo éxitos)

1. **Suelo representacional (Sec 33).** Ternario puro no alcanza ε=0.05 en
   ningún modelo probado (NanoDex E≈0.41, GPT-2 E≈0.57). Más búsqueda no lo
   rompe: Standard (8× candidatos, beam 8, cross-layer) deja E igual
   (0.4130 vs 0.4082) y el refinamiento mueve 0.6854→0.6819. El mapa de
   imposibilidad es el entregable correcto en ese régimen.
2. **Sobreajuste a calibración.** Standard empeora PPL de texto vs Fast
   (385.15 vs 361.88) pese a igual E: optimizar agresivamente por capas sobre
   calibración corta (16 tokens) transfiere peor. La calibración debe seguir a
   despliegue (`tbc/calib_text.py`, texto real; los ids aleatorios dan peores
   resultados).
3. **El error ternario se amplifica con la longitud.** GPT-2 TBC torch:
   PPL 17113 (ventanas 32) → 467290 (ventanas 256); FP32 mejora con contexto
   (265→108). La calibración usó ventanas de 32: desajuste de longitud
   documentado. Ciencia pendiente: calibración multi-longitud.
4. **Umbral del Test 1.** Ruido iid `N(0,1)` da L2≈0.45 con α óptimo
   (imposible <20% a 1.58 bits); el <20% se verifica en pesos estructurados
   (0.086). Documentado en `tests/test_synthetic.py`.
5. **Main de BitNet a fecha del snapshot**: `BITNET_X86_TL2=ON` no linka y
   `llama-quantize` no lista `I2_S` como ftype destino; el path oficial es
   OFF + conversores Python. Detalles en `TBC_REAL_TEST_REPORT.md §2`.

---

## 6. Estructura del repo

```text
tbc/                  compilador (config, memory, cache, trace, sensitivity,
                      search, equivalence, compiler, linalg, arch, calib_text,
                      pack, export, perplexity, recovery)
run_*_test.py         compilaciones reproducibles (NanoDex Fast/Standard, GPT-2)
run_recovery.py       TBC-R por etapas (nanodex|gpt2) -> tbc_output_r/
export_*_gguf.py      exportación GGUF arch llama/gpt2 (F32 + I2_S nativo)
tbc_to_gguf_i2s.py    conversor determinista model.tbc -> model-i2_s.gguf
kernel_test.py        test numérico del kernel I2_S contra el GGUF
verify_gguf.py        auditoría blob-a-blob vs pesos HF
bench_mem.py          benchmark bitnet (pp/tg + pico RSS) + PPL binaria
scripts/reproduce.py  pipeline completo NanoDex de cero a tabla (~10 min)
scripts/build_bitnet.bat  compilación bitnet.cpp en Windows
scripts/run_cli.bat   atajo para generar con llama-cli
tools/                diagnósticos y utilidades (make_tables.py, híbridos,
                      verificaciones puntuales; ver tools/ en repo)
tests/test_synthetic.py  5 tests sintéticos (2-4 s)
tbc_output*/          datos: manifests, summaries, recovery.json (sin pesos*)
third_party/          README + patch de build (el upstream se clona aparte)
logs/                 todos los logs de ejecución (evidencia)
prompt.txt, eval_*.txt  prompts y textos de evaluación
TBC_REAL_TEST_REPORT.md  reporte NanoDex (Fase 1+2+TBC-R)
```

\* Los pesos/GGUF (GB) no se versionan: se regeneran con los scripts (§4).
`tools/make_tables.py` reimprime todas las tablas del README desde los JSON.

## 7. Rendimiento en bitnet.cpp: memoria y velocidad (NanoDex-1M, CPU, 4 hilos)

Medido con `llama-bench.exe` (`-p 128 -n 256`) + monitor de pico RSS
(`bench_mem.py`, `logs/bench.json` + `logs/bench.log`):

| Modelo | Archivo | PPL binaria | Prompt (pp) | Generación (tg) | Pico RSS |
|---|---|---|---|---|---|
| Base FP32 (original) | 5.37 MB | 269.2 | 22990 tok/s | 2601 tok/s | 29.0 MB |
| TBC-R F32 | 5.37 MB | 111.9 | 29849 tok/s | 2214 tok/s | 22.0 MB |
| **TBC-R I2_S** (`model-i2_s.gguf`) | **3.07 MB (−43%)** | 354.8 | 20803 tok/s | 2519 tok/s | **20.4 MB** |

Lectura honesta:

- **Tamaño**: 5.37 → 3.07 MB (−43%, 1.75×). En 1M los embeddings/normas F32
  dominan; en modelos grandes la razón tiende a ~8–10× en lineales.
- **Velocidad: sin diferencia real aquí** (pp 21–30k, tg 2.2–2.6k, ±15% ruido).
  Con 1M params los GEMM son tan pequeños que manda el overhead, no el ancho
  de banda: I2_S no puede acelerar lo que ya va al límite del overhead.
  El speedup 2–6× de BitNet aparece en modelos de escala B, donde el matmul
  domina. Afirmar speedup en NanoDex sería falso.
- **RAM residente**: 29.0 → 20.4 MB (−30%).
- **Calidad**: R-F32 binaria (111.9) supera incluso a la base (269.2) en este
  corpus — la destilación R3 se adaptó a su distribución (en torch 32-win la
  base sigue delante 51.6 vs 54.6; son metodologías distintas, ambas
  reportadas).

## 8. Segundo caso: GPT-2 small (`openai-community/gpt2`, 124M, 12 capas)

Escalado a una arquitectura distinta (Conv1D, QKV fusionada, sin RoPE, vocab
50257) con el mismo pipeline, para probar que TBC no está atado a LLaMA.

### 8.1 Cuantización (`run_gpt2_test.py`, 48 Conv1D, beam 4, calib texto 256×32)

| Métrica | FP32 | RTN-1.58 | TBC-1.58 | TBC-A mixto* |
|---|---|---|---|---|
| PPL torch (texto simple, 4×32) | 22.62 | 12223.05 | **12133.87** (−0.7% vs RTN) | 625.6 / 477.1 |
| PPL torch (corpus, ventanas 32) | 265.6 | — | 17113 | — |
| PPL torch (corpus, ventanas 256) | 108.1 | — | 467290 | — |
| E medio por capa | — | — | 0.570 (COMPILED_WITH_WARNINGS) | — |
| Pico RAM / tiempo | — | — | 1.84 GB (<6 GB ✓) / ~78 min pared** | — |

\* TBC-A (extensión Sec 35, `logs/hybrid_demo.log`): mismos patrones TBC pero
capas cuello en FP32. `sin-c_proj` = 24/48 lineales ternarios (PPL 625.6);
`solo-c_fc` = 12/48 (PPL 477.1). 25× mejor que ternario puro.
\** Incluye reanudación desde checkpoint (`ckpt_path`, ver §8.4).

Muestras greedy (`"Hello, I am a small language model"` + 40 tokens):

| Modelo | Salida |
|---|---|
| FP32 | `...modeler. I am a small language modeler. I am a small...` (repetitivo pero gramatical) |
| TBC puro 1.58 | `...model was was was was was ...` (colapso) |
| TBC-A sin-c_proj | `...model.` + EOS (una frase limpia y se detiene) |
| TBC-A solo-c_fc | `...model model model ...` (repetitivo) |

### 8.2 E por tipo de capa: dónde duele el ternario

| Tipo (n) | E medio | min–max | Lectura |
|---|---|---|---|
| `mlp.c_fc` (12) | 0.281 | 0.25–0.35 | La expansión MLP ternariza mejor |
| `attn.c_attn` (12) | 0.583 | 0.48–0.72 | QKV fusionada, dura |
| `attn/mlp.c_proj` (24) | 0.708 | 0.22–1.09 | Las proyecciones de vuelta son el cuello (ver mapa Sec 34 en `tbc_output_gpt2/manifest.json`) |

Mismo patrón de cuellos que NanoDex (proyecciones/atención) a distinta escala.
En GPT-2 la ventaja TBC vs RTN en puro es marginal (−0.7%, verificar en
`s["ppl_note"]` de `summary.json`: un 11678 visto una vez se descartó como
outlier frente a 12133 medido dos veces). El hallazgo práctico es el TBC-A:
el mapa de imposibilidad dice exactamente qué capas no pueden ser ternarias,
y mantenerlas en FP32 recupera 25× de PPL. Ternario puro en 124M post-training
no genera texto decente; adaptativo sí produce frases limpias.

### 8.3 Ejecución nativa (`export_gpt2_gguf.py`, arch `gpt2`, QKV fusionada)

| Artefacto (149 tensores) | PPL binaria (corpus, ctx 256) | Tamaño |
|---|---|---|
| `gpt2-pure3-f32.gguf` (pesos HF, control) | 96.5 | 654 MB |
| `gpt2-tbc-f32.gguf` (lineales TBC) | 980656 | 654 MB |
| `gpt2-tbc-i2s.gguf` (**I2_S completo**, todos los `ne0` múltiplos de 128) | 216998 | 336 MB |

- El control puro valida el exportador: **149/149 tensores byte-exactos vs HF**
  (`logs/verify_pure3.log`) y PPL binaria en régimen del torch (96.5 vs 108.1).
- `gpt2-tbc-i2s.gguf` carga en `llama-cli.exe` y **genera 40 tokens** por el
  path I2_S (`logs/gpt2_gen_i2s.log`).
- Brecha torch↔binario en TBC (467k vs 217k, mismo orden) frente a pure
  (108 vs 96): el formato es correcto; la calidad la limita el ternario.

### Rendimiento GPT-2 en bitnet.cpp (`llama-bench -p 128 -n 128`, 4 hilos CPU)

`bench_gpt2.py` → `logs/bench_gpt2.json`:

| Modelo (124M) | Archivo | PPL binaria | Prompt (pp) | Generación (tg) | Pico RSS |
|---|---|---|---|---|---|
| Base FP32 | 654 MB | 96.5 | 847 tok/s | 60.1 tok/s | 541 MB |
| TBC-R F32 | 654 MB | 285.4 | 883 tok/s | 55.2 tok/s | 541 MB |
| **TBC-R I2_S** | **466 MB (−29%)** | 3769 | **992 tok/s (+17%)** | **82.3 tok/s (+37%)** | **354 MB (−35%)** |

Lectura: a 124M el matmul ya pesa y el I2_S acelera de verdad (+37% en
generación, +17% en prompt) además de −29% disco y −35% RAM. Contrasta con
NanoDex-1M (§7), donde todo era ruido de overhead: el speedup I2_S emerge con
el tamaño, exactamente como predice la teoría (régimen memory-bound). Precio:
PPL 96.5→3769 por el colapso a 1 escala/tensor (§9.2).

### 8.4 Lecciones de ingeniería de esta escala

1. **Conv1D sin bifurcar el código** (`tbc/linalg.py`): todo el compilador
   trabaja en convención `[out,in]`; `weight_out_in`/`write_weight_out_in`
   aíslan la transposición de GPT-2. `tbc/arch.py` registra specs por
   arquitectura (llama, qwen2/qwen3, gpt2 + fallback genérico).
2. **Rendimiento del search**: vectorización de α/dequant (bucles
   Python → reshape), `Y_base` hoisted + updates rank-1 exactos para
   Hamming-1, norma de referencia hoisted, evaluación exacta en submuestra
   de 256 tokens (E final siempre en X completa), presupuesto adaptativo por
   tamaño de capa y salto de capas convergidas. Tests sintéticos: 7.4 s → 2.8 s.
3. **Checkpoint/resume** (`TBC_COMPILE(..., ckpt_path)`): guarda por capa y
   reanuda sin repetir. Imprescindible a esta escala (un corte eléctrico no
   debe costar 78 min).
4. **Efecto longitud (§5.3 confirmado)**: con la misma TBC, ventanas 256 dan
   27× peor PPL que ventanas 32 en torch (17113→467290) mientras FP32 mejora
   (265→108). La calibración (ventanas 32) manda: próxima mejora, calibración
   multi-longitud.
5. **Incidentes documentados**: dos procesos duplicados por timeouts que no
   matan el worker (misma semilla → mismos valores; `summary.json`
   reconstruido de logs con E=0.5702 exacto), `n_kv` mal contado en GGUF
   (loader lo delata al instante), tensores con bias que el loader exige
   (`attn_qkv.bias`, normas), y el fork esperando `attn_qkv` fusionada en
   GPT-2 (no q/k/v separadas).

## 9. TBC-R: recuperación por etapas desde cuantización extrema (algoritmo nuevo)

El ternario puro post-training deja PPL inservibles (GPT-2: 22.6→12133;
NanoDex: 45.9→361.9). TBC-R (`tbc/recovery.py`, `run_recovery.py`) es un
algoritmo de compresión-recuperación por pasos que no reentrena pesos densos:

- **R0**: compilado TBC existente (patrones Ŵ + α por grupo + mapa E).
- **R1 adaptativo** (extensión Sec 35): capas con E > 0.65 vuelven a FP
  original. Decide el mapa de imposibilidad, no un humano.
- **R2 aceptación global**: por capa (orden de sensibilidad), se queda el
  patrón RTN o TBC según NLL de modelo completo en calibración. Corrige el
  desajuste entre error local y transferencia global.
- **R3 destilación de escalas**: patrones congelados; solo se entrenan α por
  grupo + normas + biases (<1.4% de parámetros) con Adam contra
  `β·KL(teacher/T) + (1−β)·CE` (T=2, β=0.7), early-stop en held-out.
  Implementado con hooks forward (sin cambiar la arquitectura).
- **R4 exportación dual**: torch (α por grupo) + GGUF (colapso honesto a una
  escala f32 por tensor). Se reportan ambas; el binario refleja el colapso.

Cada etapa debe mejorar held-out o se revierte. Todo bajo 6 GB.

### 9.1 Resultados (PPL torch, mismo texto 4×32; binaria en corpus ctx-256)

| Etapa | NanoDex-1M torch | NanoDex binario | GPT-2 small torch | GPT-2 binario |
|---|---|---|---|---|
| FP32 original | 45.95 | 552.8 (F32) | 22.62 | 96.5 |
| R0 TBC puro | 361.88 | ~1300 (I2_S) | 12133.87 | 216998 (I2_S) / 980656 (F32) |
| R1 adaptativo | 361.88 (0 promovidas) | — | 810.74 (19 promovidas) | — |
| R2 aceptación | 353.45 | — | 785.49 | — |
| **R3 destilación** | **54.57** (26k params, 37 s) | **111.9** (F32) / **354.8** (I2_S) | **60.57** (1.69M params, ~12 min) | **285.4** (F32) / **3769** (I2_S) |
| Bits efectivos | 1.19 b/p | — | 9.59 b/p | — |

Lectura: R3 recupera **200×** en GPT-2 (12133→60.6, a 2.7× del FP32) y **6.6×**
en NanoDex (361.9→54.6, a 1.19× del FP32 con 1.19 bits/peso). Los patrones
ternarios estaban bien; lo que faltaba era sintonizar escalas con objetivo de
salida, no de pesos. En binario (mismo corpus/ctx para todos): NanoDex-R
F32 111.9 (4.9× mejor que R0) e I2_S 354.8 (3.7×); GPT-2 R-I2_S 3769 vs
216998 antes (57×), con generación nativa (`logs/r_i2s_gen.log` y
`logs/nano_r_gen.log`).

Muestras greedy, prompt difícil (`The challenge is to preserve...`):

| Modelo | Salida |
|---|---|
| FP32 | `...not a good fit for the current model.` |
| R0 puro | `...not a model that is a model of the same...` |
| R3 | idéntica a R0 en argmax |

Caveat honesto: la recuperación vive en la **distribución** (PPL), no siempre
en el argmax greedy: R0 y R3 generan parecido con greedy, pero R3 asigna
200× más masa al texto real. Con muestreo por temperatura la diferencia sí
sería visible; con greedy, ambos repiten. El TBC-A (§8) da el punto medio
práctico si se exige generación greedy aceptable hoy.

### 9.2 Cuándo usar cada modo

- Máxima compresión pura (1.58 b/p): R0 (+R2). Calidad limitada (Sec 33).
- Calidad con presupuesto mixto: R1 (τ=0.65) + R3. Recomendado general.
- Límite del método: el colapso a 1 escala/tensor del formato I2_S penaliza
  ×13 en binario (285→3769); un formato con α por grupo lo eliminaría.

## 10. Reproducibilidad

Semillas fijas (`seed=1234`, calibración determinista), `SHA256` del GGUF
principal (NanoDex-R) `e273978f…8515fd`, re-exportación byte-idéntica,
manifiestos con `calibration_signature`, `sensitivity_map` e
`impossibility_map`. Benchmarks: `bench_mem.py`/`bench_gpt2.py` → `logs/bench*.json`.

## Licencia

MIT — ver [LICENSE](LICENSE).
