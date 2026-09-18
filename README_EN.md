# TBC — Ternary Behavioral Compilation

> **"TBC doesn't quantize weights; it compiles a neural function into ternary weights."**

A post-training compiler that transforms `M_FP → M_T` with `W_T ∈ {-1, 0, +1}`
through discrete search and behavioral verification — not weight rounding —
plus **TBC-R**, a staged recovery algorithm from extreme quantization.
Everything exports to **`I2_S` GGUF runnable natively in [bitnet.cpp](https://github.com/microsoft/BitNet)**.

Read in another language: [Español](README.md) · **English** (this file)

## 1. Project status

| Milestone | State | Evidence |
|---|---|---|
| Full compiler (6 components, Sec 3/24) | ✅ | `tbc/` (14 modules), no stubs |
| Fixed memory < 6GB, verified | ✅ | Fast peak 0.5GB, Standard 1.3GB, GPT-2 1.8GB |
| 5 synthetic tests | ✅ 5/5 | `logs/synthetic*.log` |
| NanoDex-1M (llama, 1M) TBC-Fast + Standard | ✅ | `tbc_output/`, `tbc_output_std/` |
| GPT-2 small (124M) full TBC | ✅ | `tbc_output_gpt2/` + §8 |
| bitnet.cpp built (406/406 targets) | ✅ | `logs/bitnet_compile.log`, 48 binaries |
| I2_S GGUF loads and generates real tokens | ✅ | `logs/nano_r_gen.log`, `logs/gpt2_gen_i2s.log` |
| I2_S kernel test (diff 0.00 < 1e-3) | ✅ | `logs/kernel_test.log` |
| Byte-exact exporter (149/149 tensors) | ✅ | `logs/verify_pure3.log` |
| bitnet.cpp benchmark (memory + tok/s) | ✅ | §7, `logs/bench*.json` |
| Staged recovery algorithm TBC-R | ✅ | §9, `tbc/recovery.py` |

End-to-end architectures: **llama** (NanoDex-1M), **GPT-2** (GPT2-small, via
Conv1D), **qwen2/qwen3** (same pipeline as llama). See `tbc/arch.py`.

## 2. Results on NanoDex-1M (`DedeProGames/NanoDex-1M`, 5-layer llama, 1.06M params)

### 2.1 Torch perplexity (real English text, 4×32, identical eval)

| Config | FP16 | RTN-1.58 | TBC-1.58 | Gain vs RTN |
|---|---|---|---|---|
| Fast (K=4, text calib 16×64) | 45.95 | 377.30 | **361.88** | −4.1% |
| Standard (K=8, 1024×16 calib, cross-layer) | 45.95 | 377.30 | 385.15 (+2.1%) | worse |

Same representation (group 32, pure ternary): TBC-Fast beats the RTN baseline.
Standard doesn't improve (see §5, calibration-overfitting finding).

### 2.2 Compilation

| Metric | Fast | Standard |
|---|---|---|
| Mean per-layer E | 0.408 | 0.413 |
| Layers above ε=0.05 | 35/35 (COMPILED_WITH_WARNINGS + Sec 34 map) | 35/35 |
| Time | ~23 s | ~13 min |
| Peak RAM | 0.51 GB | 1.32 GB (6GB limit ✓) |

### 2.3 Native bitnet.cpp execution (binaries built from `third_party/BitNet`)

| Artifact | Binary PPL (`llama-perplexity`, corpus ctx-256) | Size |
|---|---|---|
| `tbc_output_r/nano-r-f32.gguf` (R3, trained α+norms) | **111.9** (was 552.8: 4.9×) | 5.37 MB |
| `model-i2_s.gguf` = **NanoDex-R** (30×I2_S R2/R3 + 5×F32) | **354.8** (was ~1300: 3.7×) | 3.07 MB |
| Torch FP32 reference (32-windows) | 51.6 | 4.25 MB torch |

`model-i2_s.gguf` (**recovered**, sha256 `e273978f…8515fd`) loads in
`llama-cli.exe` and **generates 50 tokens** (1665 tok/s, `logs/nano_r_gen.log`).
Deterministic byte-identical re-export
(`export_llama_gguf.py --recovered tbc_output_r`).

> Methodology note: binary PPL uses ctx-256 chunks, torch uses 32-windows;
> absolute values aren't comparable across methodologies, only relative order
> within each. See §5 (length effect).

## 3. Native bitnet.cpp compatibility (Sec 39) — verified against sources

Everything below was verified by reading `third_party/BitNet` code, not the prompt text:

- `GGML_TYPE_I2_S = 36` (`3rdparty/llama.cpp/ggml/include/ggml.h:426`). The prompt said 30: **fixed**.
- `quantize_i2_s` (`ggml/src/ggml-cpu/quants.c:1358`): **sequential** packing in GGUF
  (weight `i` → byte `i/4`, bits `6-2*(i%4)`), codes `0→−1, 1→0, 2→+1`,
  **one f32 scale per tensor** at offset `numel/4`, size `numel/4+32` (`ggml.c:1316`).
- The runtime reads **interleaved-128** layout (`dequantize_row_i2_s`, `quants.c:1335`;
  `byte gp = c(gp)<<6|c(32+gp)<<4|c(64+gp)<<2|c(96+gp)`). Decisive test:
  sequential GGUF → binary PPL 101828 (permuted weights); interleaved GGUF →
  ~1240. The exporter emits interleaved-128.
- Fork restriction: rows with `ne0` not a multiple of 128 read out-of-row;
  such tensors stay F32 (mixed quantization). NanoDex: 5 `ffn_down` (ne0=288).
  GPT-2 small: all multiples → full I2_S.
- q/k RoPE permutation (`conversion/llama.py:163`, `permute`): applied on llama
  export (without it binary PPL doubles: 997→553).
- Embedded tokenizers (tokens+scores+types+merges). Binary tokenization
  verified identical to HF (NanoDex: same ids; GPT-2: 577=577 tokens).
- Windows build: VS2026 + LLVM-MinGW clang 22 + Ninja, official
  `setup_env.py` flags (`BITNET_X86_TL2=OFF`; with `=ON` main doesn't link:
  `ggml_bitnet_mul_mat` only exists under `ARM_TL1`), 1-line fix
  (`mad.cpp` into `ggml-cpu` CMake, see `third_party/`), `_WIN32_WINNT=0x0A00`.

### Connecting a TBC model to bitnet.cpp (loader checklist)

1. `general.architecture` = `llama` / `gpt2` / `qwen2` + block hyperparams
   (`block_count`, `embedding_length`, …) and `general.quantization_version = 2`.
2. Fork tensor names (`blk.i.attn_q.weight`, fused `attn_qkv` on GPT-2,
   `token_embd`, `position_embd` on GPT-2) **plus biases** where the arch has
   them (the loader tells you what's missing: `missing tensor`).
3. Linears in `I2_S` (36) interleaved-128 with 32 B trailer; rest F32.
4. Embedded tokenizer with correct `model` + `pre` (`gpt-2` for byte-level BPE)
   + tokens/scores/types/merges.
5. If the loader rejects, its message (`missing tensor`, `bad magic`, counts)
   says exactly what's missing: iterate with it (that's how this was debugged).

## 4. Usage guides

### 4.1 Requirements

```text
Python ≥ 3.10, torch, transformers, safetensors, datasets, psutil, numpy
CMake ≥ 3.22 + Clang/GCC + Ninja (only to build bitnet.cpp)
```

### 4.2 Quantize a model with TBC

```python
from tbc.config import TBCConfig
from tbc.compiler import TBC_COMPILE
from tbc.calib_text import text_windows  # in-distribution calibration

cfg = TBCConfig(group_size=32, beam_size=4, epsilon_target=0.05,
                mode="fast", enable_backtracking=True)  # 6GB limit by default
calib = text_windows(tokenizer, n_seq=64, seq_len=64, stride=16)
tbc_model, report = TBC_COMPILE(model, cfg, calib_ids=calib,
                                ckpt_path="ckpt.pt")  # resume after interruptions
```

- `beam_size` 4 (Fast) / 8 (Standard) / 16 (Max via `mode="max"`).
- `enable_cross_layer_refinement=True` adds a Max pass over the 4 worst layers.
- Conv1D (GPT-2) handled automatically (`tbc/linalg.py`, internal `[out,in]`).
- If `min_D > ε`: `COMPILED_WITH_WARNINGS` status + `impossibility_map` (Sec 34).

Ready scripts: `run_real_test.py` (NanoDex Fast), `run_standard_test.py`,
`run_gpt2_test.py` (each: compile + FP/RTN/TBC PPL + artifacts + report).

### 4.3 Export to GGUF and load in bitnet.cpp

```bash
# 1. Build bitnet.cpp once (Windows: scripts\build_bitnet.bat;
#    Linux: same cmake without vcvars — see third_party/README.md)
python export_llama_gguf.py  tbc_output/model.tbc  out-f32.gguf  out-i2s.gguf
python export_gpt2_gguf.py   tbc_output_gpt2/model.tbc out-f32.gguf out-i2s.gguf
python tbc_to_gguf_i2s.py --tbc_dir tbc_output/model.tbc --out tbc_output/model-i2_s.gguf

# 3. Run (native I2_S)
third_party\BitNet\build\bin\llama-cli.exe -m out-i2s.gguf -f prompt.txt -n 50 --threads 4
third_party\BitNet\build\bin\llama-perplexity.exe -m out-i2s.gguf -f texto.txt -c 256 --threads 4
```

Format rules the exporter already applies: I2_S=36 interleaved-128 with
globally optimal f32 scale (`optimal_global_scale`, Sec 22 collapsed to 1 scale
per tensor), q/k RoPE permute (llama), fused QKV (gpt2), intact F32 biases,
`ne0 % 128 != 0` → mixed F32.

### 4.4 Verify an export (recommended before publishing a GGUF)

```bash
python verify_gguf.py <f32.gguf>   # 149/149 byte-exact vs HF = OK
python kernel_test.py              # 100% patterns + Y diff < 1e-3
```

### 4.5 Full phased pipeline (how each result was obtained)

```bash
python scripts/reproduce.py              # NanoDex Fast from zero to table (~10 min)
python run_standard_test.py              # TBC-Standard (beam 8, cross-layer)
python run_gpt2_test.py                  # GPT-2 small 124M (~80 min, ckpt/resume)
python run_recovery.py nanodex           # TBC-R NanoDex (~1 min)
python run_recovery.py gpt2              # TBC-R GPT-2 (~12 min)
python tools/make_tables.py              # reprint README tables from the JSONs
```

Log phases: 0) spec → 1) compiler + tests + NanoDex → 2) native bitnet.cpp +
Standard → 3) GPT-2 scale → 4) TBC-R algorithm → 5) benchmarks → 6) release.
Each phase left its logs in `logs/`.

## 5. Honest findings (limits, not just wins)

1. **Representational floor (Sec 33).** Pure ternary never reaches ε=0.05 on
   any tested model (NanoDex E≈0.41, GPT-2 E≈0.57). More search doesn't break
   it: Standard (8× candidates, beam 8, cross-layer) leaves E flat
   (0.4130 vs 0.4082). The impossibility map is the right deliverable there.
2. **Calibration overfitting.** Standard worsens text PPL vs Fast
   (385.15 vs 361.88) at equal E: aggressive per-layer optimization on short
   (16-token) calibration transfers worse.
3. **Ternary error amplifies with length.** Same GPT-2 TBC: PPL 17113
   (32-windows) → 467290 (256-windows), while FP32 improves (265→108).
   Calibration used 32-token windows: documented length mismatch. Next:
   multi-length calibration.
4. **Test-1 threshold.** iid `N(0,1)` noise gives L2≈0.45 with optimal α
   (impossible <20% at 1.58 bits); <20% verified on structured weights
   (0.086). Documented in `tests/test_synthetic.py`.
5. **BitNet main snapshot quirks**: `BITNET_X86_TL2=ON` doesn't link and
   `llama-quantize` offers no `I2_S` target ftype; the official path is OFF +
   Python converters. See `TBC_REAL_TEST_REPORT.md §2`.

## 6. Repo layout

```text
tbc/                  compiler (config, memory, cache, trace, sensitivity,
                      search, equivalence, compiler, linalg, arch, calib_text,
                      pack, export, perplexity, recovery)
run_*_test.py         reproducible compilations (NanoDex Fast/Standard, GPT-2)
run_recovery.py       staged TBC-R (nanodex|gpt2) -> tbc_output_r/
export_*_gguf.py      arch llama/gpt2 GGUF export (F32 + native I2_S)
tbc_to_gguf_i2s.py    deterministic model.tbc -> model-i2_s.gguf converter
kernel_test.py        I2_S kernel numerics test against the GGUF
verify_gguf.py        blob-by-blob audit vs HF weights
bench_mem.py          bitnet bench (pp/tg + peak RSS) + binary PPL
scripts/reproduce.py  full NanoDex pipeline from zero to table (~10 min)
scripts/build_bitnet.bat  bitnet.cpp build on Windows
scripts/run_cli.bat   shortcut to generate with llama-cli
tools/                diagnostics and utilities (make_tables.py, hybrids,
                      spot checks; see tools/ in repo)
tests/test_synthetic.py  5 synthetic tests (2-4 s)
tbc_output*/          data: manifests, summaries, recovery.json (no weights*)
third_party/          README + build patch (upstream cloned separately)
logs/                 all execution logs (evidence)
prompt.txt, eval_*.txt  prompts and evaluation texts
TBC_REAL_TEST_REPORT.md  NanoDex report (Phase 1+2+TBC-R)
```

\* Weight/GGUF binaries (GB) aren't versioned: regenerate with the scripts (§4).
`tools/make_tables.py` reprints every README table from the JSONs.

## 7. bitnet.cpp performance: memory and speed (NanoDex-1M, CPU, 4 threads)

Measured with `llama-bench.exe` (`-p 128 -n 256`) + peak-RSS monitor
(`bench_mem.py`, `logs/bench.json` + `logs/bench.log`):

| Model | File | Binary PPL | Prompt (pp) | Generation (tg) | Peak RSS |
|---|---|---|---|---|---|
| Base FP32 (original) | 5.37 MB | 269.2 | 22990 tok/s | 2601 tok/s | 29.0 MB |
| TBC-R F32 | 5.37 MB | 111.9 | 29849 tok/s | 2214 tok/s | 22.0 MB |
| **TBC-R I2_S** (`model-i2_s.gguf`) | **3.07 MB (−43%)** | 354.8 | 20803 tok/s | 2519 tok/s | **20.4 MB** |

Honest reading:

- **Size**: 5.37 → 3.07 MB (−43%, 1.75×). On 1M models F32 embeddings/norms
  dominate; on large models the ratio approaches ~8–10× on linears.
- **Speed: no real difference here** (pp 21–30k, tg 2.2–2.6k, ±15% noise).
  At 1M params GEMMs are so small that overhead rules, not bandwidth: I2_S
  can't accelerate what's already overhead-bound. BitNet's 2–6× speedup shows
  at B-scale, where matmul dominates. Claiming speedup on NanoDex would be false.
- **Resident RAM**: 29.0 → 20.4 MB (−30%).
- **Quality**: binary R-F32 (111.9) even beats base (269.2) on this corpus —
  R3 distillation adapted to its distribution (on torch 32-win base still leads
  51.6 vs 54.6; different methodologies, both reported).

## 8. Second case: GPT-2 small (`openai-community/gpt2`, 124M, 12 layers)

Scaling to a different architecture (Conv1D, fused QKV, no RoPE, vocab 50257)
with the same pipeline, to prove TBC isn't tied to LLaMA.

### 8.1 Quantization (`run_gpt2_test.py`, 48 Conv1D, beam 4, 256×32 text calib)

| Metric | FP32 | RTN-1.58 | TBC-1.58 | TBC-A mixed* |
|---|---|---|---|---|
| Torch PPL (simple text, 4×32) | 22.62 | 12223.05 | **12133.87** (−0.7% vs RTN) | 625.6 / 477.1 |
| Torch PPL (corpus, 32-windows) | 265.6 | — | 17113 | — |
| Torch PPL (corpus, 256-windows) | 108.1 | — | 467290 | — |
| Mean per-layer E | — | — | 0.570 (COMPILED_WITH_WARNINGS) | — |
| Peak RAM / time | — | — | 1.84 GB (<6 GB ✓) / ~78 min wall** | — |

\* TBC-A (Sec 35 extension, `logs/hybrid_demo.log`): same TBC patterns but
bottleneck layers kept in FP32. `no-c_proj` = 24/48 ternary linears (PPL 625.6);
`c_fc-only` = 12/48 (PPL 477.1). 25× better than pure ternary.
\** Includes checkpoint resume (`ckpt_path`, see §8.4).

Greedy samples (`"Hello, I am a small language model"` + 40 tokens):

| Model | Output |
|---|---|
| FP32 | `...modeler. I am a small language modeler...` (repetitive but grammatical) |
| Pure 1.58 TBC | `...model was was was was was ...` (collapsed) |
| TBC-A no-c_proj | `...model.` + EOS (one clean sentence, then stops) |
| TBC-A c_fc-only | `...model model model ...` (repetitive) |

Pure-ternary 124M post-training can't generate decent text; adaptive can
produce clean sentences. The impossibility map says exactly which layers can't
be ternary — keep those in FP32 and PPL recovers 25×. "Truly decent" would
still need a bigger model + recovery training (that's §9).

Marginal TBC-vs-RTN note on GPT-2 (−0.7%): 12133.87 measured twice
independently (original summary + `hybrid_demo`); a one-off 11678 seen in a log
from a run with duplicated workers was discarded as an outlier. See
`s["ppl_note"]` in `tbc_output_gpt2/summary.json`.

### 8.2 Per-layer-type E: where ternary hurts

| Type (n) | Mean E | min–max | Reading |
|---|---|---|---|
| `mlp.c_fc` (12) | 0.281 | 0.25–0.35 | MLP expansion ternarizes best |
| `attn.c_attn` (12) | 0.583 | 0.48–0.72 | Fused QKV, hard |
| `attn/mlp.c_proj` (24) | 0.708 | 0.22–1.09 | Back-projections are the bottleneck (Sec 34 map in `tbc_output_gpt2/manifest.json`) |

Same bottleneck pattern as NanoDex (projections/attention) at a different
scale. On GPT-2 the pure TBC-vs-RTN edge is marginal, so the practical finding
is the hybrid one above.

### 8.3 Native execution (`export_gpt2_gguf.py`, `gpt2` arch, fused QKV)

| Artifact (149 tensors) | Binary PPL (corpus, ctx 256) | Size |
|---|---|---|
| `gpt2-pure3-f32.gguf` (HF weights, control) | 96.5 | 654 MB |
| `gpt2-tbc-f32.gguf` (TBC linears) | 980656 | 654 MB |
| `gpt2-tbc-i2s.gguf` (**full I2_S**, all `ne0` multiples of 128) | 216998 | 336 MB |

- The pure control validates the exporter: **149/149 byte-exact tensors vs HF**
  (`logs/verify_pure3.log`) and binary PPL in the torch regime (96.5 vs 108.1).
- `gpt2-tbc-i2s.gguf` loads in `llama-cli.exe` and **generates 40 tokens**
  through the I2_S path (`logs/gpt2_gen_i2s.log`).
- Torch↔binary gap on TBC (467k vs 217k, same order) vs pure (108 vs 96):
  the format is right; quality is limited by ternary itself.

### GPT-2 performance on bitnet.cpp (`llama-bench -p 128 -n 128`, 4 CPU threads)

`bench_gpt2.py` → `logs/bench_gpt2.json`:

| 124M model | File | Binary PPL | Prompt (pp) | Generation (tg) | Peak RSS |
|---|---|---|---|---|---|
| Base FP32 | 654 MB | 96.5 | 847 tok/s | 60.1 tok/s | 541 MB |
| TBC-R F32 | 654 MB | 285.4 | 883 tok/s | 55.2 tok/s | 541 MB |
| **TBC-R I2_S** | **466 MB (−29%)** | 3769 | **992 tok/s (+17%)** | **82.3 tok/s (+37%)** | **354 MB (−35%)** |

Reading: at 124M the matmul already matters and I2_S genuinely accelerates
(+37% generation, +17% prompt) on top of −29% disk and −35% RAM. Contrast
NanoDex-1M (§7), where everything was overhead noise: the I2_S speedup emerges
with size, exactly as theory predicts (memory-bound regime). Cost: PPL
96.5→3769 from the single-scale-per-tensor collapse (§9.2).

### 8.4 Engineering lessons from this scale

1. **Conv1D without forking the code** (`tbc/linalg.py`): the whole compiler
   works in `[out,in]` convention; `weight_out_in`/`write_weight_out_in`
   isolate GPT-2's transposition. `tbc/arch.py` registers per-architecture
   specs (llama, qwen2/qwen3, gpt2 + generic fallback).
2. **Search performance**: vectorized α/dequant (Python loops → reshape),
   hoisted `Y_base` + exact rank-1 updates for Hamming-1, hoisted reference
   norm, exact eval on a 256-token subsample (final E always on full X),
   size-adaptive budget and converged-layer skipping. Synthetic tests: 7.4 s → 2.8 s.
3. **Checkpoint/resume** (`TBC_COMPILE(..., ckpt_path)`): saves per layer and
   resumes without repeating. Mandatory at this scale (a power cut must not
   cost 78 min).
4. **Length effect (§5.3 confirmed)**: same TBC scores 27× worse PPL on
   256-windows than 32-windows in torch (17113→467290) while FP32 improves
   (265→108). Calibration used 32-token windows: documented mismatch. Next:
   multi-length calibration.
5. **Documented incidents**: duplicated workers from timeouts that don't kill
   the worker (same seed → same values; `summary.json` rebuilt from logs with
   exact E=0.5702), miscounted `n_kv` in GGUF (the loader flags it instantly),
   bias tensors the loader requires (`attn_qkv.bias`, norms), and the fork
   expecting fused `attn_qkv` on GPT-2 (not split q/k/v).

## 9. TBC-R: staged recovery from extreme quantization (new algorithm)

Pure post-training ternary leaves unusable PPL (GPT-2: 22.6→12133;
NanoDex: 45.9→361.9). TBC-R (`tbc/recovery.py`, `run_recovery.py`) is a
step-by-step compression-recovery algorithm that never retrains dense weights:

- **R0**: existing TBC build (Ŵ patterns + per-group α + E map).
- **R1 adaptive** (Sec 35 extension): layers with E > 0.65 return to original
  FP. The impossibility map decides, not a human.
- **R2 global acceptance**: per layer (sensitivity order), keep the RTN or TBC
  pattern, whichever gives better full-model NLL on calibration. Fixes the
  local-error vs global-transfer mismatch.
- **R3 scale distillation**: frozen patterns; only per-group α + norms + biases
  (<1.4% of params) trained with Adam against `β·KL(teacher/T) + (1−β)·CE`
  (T=2, β=0.7), held-out early stop. Implemented with forward hooks (no arch change).
- **R4 dual export**: torch (per-group α) + GGUF (honest collapse to one f32
  scale per tensor). Both reported; the binary reflects the collapse.

Each stage must improve held-out NLL or it is reverted. All under 6 GB.

### 9.1 Results (torch PPL, same 4×32 text; binary on ctx-256 corpus)

| Stage | NanoDex-1M torch | NanoDex binary | GPT-2 small torch | GPT-2 binary |
|---|---|---|---|---|
| Original FP32 | 45.95 | 552.8 (F32) | 22.62 | 96.5 |
| R0 pure TBC | 361.88 | ~1300 (I2_S) | 12133.87 | 216998 (I2_S) / 980656 (F32) |
| R1 adaptive | 361.88 (0 promoted) | — | 810.74 (19 promoted) | — |
| R2 acceptance | 353.45 | — | 785.49 | — |
| **R3 distillation** | **54.57** (26k params, 37 s) | **111.9** (F32) / **354.8** (I2_S) | **60.57** (1.69M params, ~12 min) | **285.4** (F32) / **3769** (I2_S) |
| Effective bits | 1.19 b/p | — | 9.59 b/p | — |

Reading: R3 recovers **200×** on GPT-2 (12133→60.6, within 2.7× of FP32) and
**6.6×** on NanoDex (361.9→54.6, within 1.19× of FP32 at 1.19 bits/weight).
The ternary patterns were fine all along; what was missing was tuning scales
against output, not weights. In binary (same corpus/ctx for all): NanoDex-R
F32 111.9 (4.9× better than R0) and I2_S 354.8 (3.7×); GPT-2 R-I2_S 3769 vs
216998 before (57×), with native generation (`logs/r_i2s_gen.log` and
`logs/nano_r_gen.log`).

Greedy samples, hard prompt (`The challenge is to preserve...`):

| Model | Output |
|---|---|
| FP32 | `...not a good fit for the current model.` |
| R0 pure | `...not a model that is a model of the same...` |
| R3 | same as R0 at argmax |

Honest caveat: the recovery lives in the **distribution** (PPL), not always in
greedy argmax: R0 and R3 generate alike greedily, but R3 assigns 200× more mass
to real text. With temperature sampling the difference would show; with greedy,
both loop. TBC-A (§8) is the practical middle ground if acceptable greedy
generation is needed today.

### 9.2 When to use each mode

- Max pure compression (1.58 b/p): R0 (+R2). Limited quality (Sec 33).
- Quality on a mixed budget: R1 (τ=0.65) + R3. General recommendation.
- Method limit: collapsing to 1 scale/tensor in I2_S costs ×13 in binary
  (285→3769); a per-group-α format would remove it.

## 10. Reproducibility

Fixed seeds (`seed=1234`, deterministic calibration), `SHA256` of the main
(NanoDex-R) GGUF `e273978f…8515fd`, byte-identical re-export, manifests with
`calibration_signature`, `sensitivity_map` and `impossibility_map`.
Benchmarks: `bench_mem.py`/`bench_gpt2.py` → `logs/bench*.json`.

## License

MIT — see [LICENSE](LICENSE).
