# TBC — Ternary Behavioral Compilation (+ TBC-R Recovery)

> **Compile any LLM into ternary weights `{-1, 0, +1}`, then recover it:
> PPL from 12133 → 60 on GPT-2 small, running natively in bitnet.cpp.**

TBC doesn't round weights: it **compiles behavior** of the original model into
a ternary representation via discrete search with behavioral verification. And
when pure ternary isn't enough, **TBC-R** recovers it in stages (adaptive +
global acceptance + scale distillation) without retraining dense weights — all
with configurable memory budgets and export to **`I2_S` GGUF that loads and
runs in [bitnet.cpp](https://github.com/microsoft/BitNet)**.

Read in another language: [Español](README.md) · **English** (this file)

## Quickstart

```bash
pip install -r requirements.txt

# Quantize ANY HuggingFace causal model (architecture auto-detected)
python -m tbc quantize --model Qwen/Qwen2.5-7B --out ./qwen25-7b-tbc --beam 8
python -m tbc quantize --model gpt2 --out ./gpt2-tbc --ram auto

# Recover it (R1 adaptive + R2 global + R3 distillation)
python -m tbc recover --model gpt2 --tbc ./gpt2-tbc/model.tbc --out ./gpt2-r

# See which architectures this checkout supports
python -m tbc arch-info            # list (23 architectures)
python -m tbc arch-info --arch qwen3moe
```

```bash
# Export to GGUF and run natively in bitnet.cpp
python export_llama_gguf.py  <model.tbc>  out-f32.gguf  out-i2s.gguf   # llama/qwen/... family
python export_gpt2_gguf.py   <model.tbc>  out-f32.gguf  out-i2s.gguf   # GPT-2
third_party/BitNet/build/bin/llama-cli -m out-i2s.gguf -p "Hello" -n 50 --threads 4
```

Budgets (`--ram/--vram`, or `TBC_RAM_BUDGET`): `'auto'` (80% available,
default), `'6GB'`, `'24GB'`… The compiler tracks peak RSS and fails if you
exceed your budget; no hard limit other than the one you set.

## Headline results

### TBC-R recovery (torch PPL, same 4×32 eval per model)

| Stage | NanoDex-1M (llama, 1M) | GPT-2 small (124M) |
|---|---|---|
| Original FP32 | 45.95 | 22.62 |
| R0 pure ternary | 361.88 | 12133.87 |
| R1 adaptive (impossible layers → FP) | 361.88 (0 promoted) | 810.74 (19 promoted) |
| R2 global acceptance | 353.45 | 785.49 |
| **R3 scale distillation** | **54.57** (26k params, 37 s, **1.19 b/p**) | **60.57** (1.69M params, ~12 min) |

### Native bitnet.cpp execution (binary PPL, ctx-256 corpus)

| Model | FP32 | R3-F32 | R3-I2_S | Generates tokens |
|---|---|---|---|---|
| NanoDex-1M | 552.8 (5.37 MB) | **111.9** | **354.8** (3.07 MB) | ✅ 50 toks, 1665 tok/s |
| GPT-2 small | 96.5 (654 MB) | **285.4** | **3769** (466 MB) | ✅ 40 toks |

### bitnet.cpp speed and memory (`llama-bench`, 4 CPU threads)

| 124M model | File | Binary PPL | Prompt | Generation | Peak RSS |
|---|---|---|---|---|---|
| Base FP32 | 654 MB | 96.5 | 847 tok/s | 60.1 tok/s | 541 MB |
| TBC-R I2_S | **466 MB (−29%)** | 3769 | **992 tok/s (+17%)** | **82.3 tok/s (+37%)** | **354 MB (−35%)** |

(On 1M models I2_S saves size/RAM with no speed change — overhead-bound;
at 124M the matmul dominates and I2_S genuinely accelerates.)

## Features

- **Behavioral compiler**: TraceGenerator, fixed-memory BehaviorCache,
  SensitivityAnalyzer, discrete search (coordinate + beam K≤16 +
  branch-and-bound + incremental `Y+XΔW` eval + exact rank-1 updates),
  cascaded EquivalenceChecker, progressive tolerance, backtracking, and an
  impossibility map when ε is unreachable.
- **TBC-R**: R0→R3 pipeline with early-stop and per-stage revert.
- **23 architectures** (llama, qwen/2/3(+MoE), mistral, mixtral, gemma/2/3,
  phi/3, falcon, gpt2, gpt_neox, gpt_bigcode, bloom, mpt, deepseek_v2,
  olmo/2, glm) with `stable/compatible/experimental` levels and
  download-free tests (`tests/test_arch.py`: 13/13 + 2 smokes + MoE).
- **Native bitnet.cpp**: `I2_S` (type 36) interleaved-128 GGUF verified
  against sources; kernel test diff 0.00; exporter audited 149/149
  byte-exact vs HF.
- **Reproducible**: fixed seeds, `scripts/reproduce.py` (NanoDex zero-to-table
  in ~10 min), `tools/make_tables.py` reprints these tables from the JSONs,
  every run logged.

## How it works (summary)

1. **Compile**: run the FP model, record traces with sparse checkpoints,
   estimate per-layer sensitivity, and search ternary configurations that
   reproduce activations and logits within ε. Analytic α scales.
2. **Verify**: norm→cosine→top-k→KL cascade; layers that can't comply are
   recorded in the impossibility map instead of faking success.
3. **Recover** (TBC-R): promote impossible layers, pick patterns by global NLL,
   distill scales/norms/biases against the teacher.
4. **Export**: F32 or native I2_S GGUF + reproducible manifest.

## Architectures

| Architecture | State | Notes |
|---|---|---|
| llama | stable | Validated end-to-end (NanoDex-1M). |
| gpt2 | stable | Conv1D, fused QKV, learned pos-emb. Validated end-to-end (GPT-2 small). |
| qwen2, qwen3, mistral, gemma2, mixtral, phi3, falcon, gpt_neox, bloom, olmo2 | compatible | Detection + compilation proven on synthetic models; GGUF names follow the official converter; validate on real weights. MoE: experts yes, router stays FP. |
| qwen, qwen2moe, qwen3moe, gemma, gemma3, phi, gpt_bigcode, mpt, deepseek_v2, olmo, glm | experimental | Quirks (MLA, non-standard fusions, norms) pending validation on real weights. |

`python -m tbc arch-info --arch <name>` shows target linears, exclusions,
GGUF format and notes for each.

## Connecting to bitnet.cpp

1. Clone and build (once): `third_party/README.md`
   (`clone --recursive` + 1-line patch + `cmake`; binaries land in
   `third_party/BitNet/build/bin/`). Verified: 406/406 targets.
2. Export (§4 guides, full detail in `docs/HISTORIA-Y-DETALLE.md` §3).
3. Load and generate: `llama-cli -m model-i2s.gguf -p "..." -n 50`.

Runtime format requirements (source-verified): I2_S=36 interleaved-128, 1 f32
scale per tensor, RoPE permute on q/k (llama), fused QKV (gpt2), intact F32
biases, `ne0 % 128 == 0` or mixed F32, embedded tokenizer with correct `pre`.

## Usage guides

### Requirements

```text
Python ≥ 3.10, torch, transformers, safetensors, datasets, psutil, numpy
CMake ≥ 3.22 + Clang/GCC + Ninja (only to build bitnet.cpp)
```

### CLI

```bash
python -m tbc quantize --model <HF-id> --out <dir> [--beam 8] [--mode standard]
                       [--ram auto|6GB|24GB] [--vram auto] [--group 32]
                       [--calib 256] [--seqlen 32] [--eps 0.05]
python -m tbc recover  --model <HF-id> --tbc <model.tbc> --out <dir-r>
                       [--adapt 0.65] [--steps 300]
python -m tbc arch-info [--arch <name>]
```

Env: `TBC_RAM_BUDGET`, `TBC_DTYPE=float32|float16|bfloat16`
(use `TBC_DTYPE=float16` for large models).

### Full pipeline and utilities

```bash
python scripts/reproduce.py              # NanoDex Fast zero-to-table (~10 min)
python run_standard_test.py              # TBC-Standard (beam 8, cross-layer)
python run_gpt2_test.py                  # GPT-2 124M (automatic ckpt/resume)
python run_recovery.py [nanodex|gpt2]    # classic TBC-R pipeline with reports
python tools/make_tables.py              # tables from the result JSONs
python tests/test_synthetic.py && python tests/test_arch.py
```

## Detailed results and logbook

- `TBC_REAL_TEST_REPORT.md`: NanoDex Phase 1+2+TBC-R with per-layer E.
- `docs/HISTORIA-Y-DETALLE.md`: full prior version (per-layer tables,
  engineering incidents, fine methodology).
- `tbc_output*/`: manifests, `summary.json`, `recovery.json`
  (weights/GGUFs regenerate; not versioned by size).
- `logs/`: every execution log.

## Known limitations

- Pure post-training ternary doesn't reach ε=0.05 on tested models;
  that's exactly what TBC-R is for.
- I2_S carries 1 scale per tensor: collapsing per-group α costs ×3–13 in
  binary PPL vs torch. A per-group-α format would remove it.
- I2_S speedup shows when matmul dominates (124M: +37%); on tiny models
  only size/RAM savings apply.
- `experimental` architectures compile, but validate the GGUF with
  `verify_gguf.py` + one binary PPL before trusting it.

## License

MIT — see [LICENSE](LICENSE). bitnet.cpp belongs to Microsoft (MIT); not
included here, see `third_party/README.md`.
