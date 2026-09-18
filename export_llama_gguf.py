#!/usr/bin/env python
"""export_llama_gguf.py — NanoDex-TBC -> GGUF estándar arch=llama (F32 e I2_S nativo).

F32: lineales con pesos TBC dequantizados + resto original.
I2_S: patrones ternarios TBC + escala f32 global óptima (formato BitNet, tipo 36).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from tbc.export import (
    GGML_TYPE_F32, GGML_TYPE_I2_S, KV_U32, KV_F32, KV_BOOL,
    kv, kv_arr, f32_blob, write_gguf, pack_i2_s_native, pack_i2_s_interleaved,
    optimal_global_scale,
)

BASE = os.path.dirname(os.path.abspath(__file__))
_POS = [a for a in sys.argv[1:] if not a.startswith("--")]
TBC_DIR = _POS[0] if len(_POS) > 0 else os.path.join(BASE, "tbc_output", "model.tbc")
OUT_F32 = _POS[1] if len(_POS) > 1 else os.path.join(BASE, "tbc_output", "nanodex-tbc-f32.gguf")
OUT_I2S = _POS[2] if len(_POS) > 2 else os.path.join(BASE, "tbc_output", "nanodex-tbc-i2s.gguf")


def hf_to_llama(name: str) -> str | None:
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    if name.startswith("model.layers."):
        parts = name.split(".")
        i = parts[2]
        rest = ".".join(parts[3:])
        m = {
            "self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
            "self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
            "self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
            "self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
            "mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
            "mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
            "mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
            "input_layernorm.weight": f"blk.{i}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
        }
        return m.get(rest)
    return None


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[exp] cargando modelo + tokenizer...", flush=True)
    model = AutoModelForCausalLM.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
    cfg = model.config
    blob = torch.load(os.path.join(TBC_DIR, "weights.pt"), map_location="cpu")
    manifest = json.load(open(os.path.join(TBC_DIR, "manifest.json"))) if os.path.exists(os.path.join(TBC_DIR, "manifest.json")) else {}
    # --recovered <dir>: patrones R2 + alfas R3 + normas/biases entrenados.
    # Capas ausentes en alphas_r3 (promovidas R1) quedan FP originales.
    REC = "--recovered" in sys.argv and sys.argv[sys.argv.index("--recovered") + 1] if "--recovered" in sys.argv else None
    nb_ov: dict[str, torch.Tensor] = {}
    if REC:
        rec = torch.load(os.path.join(REC, "recovered.pt"), map_location="cpu")
        nb_ov = {k: v.float() for k, v in torch.load(os.path.join(REC, "norms_bias.pt"), map_location="cpu").items()}
        have = set(rec.get("alphas_r3", {}))
        blob = {n: {"W": rec["patterns"][n], "alpha": rec["alphas_r3"][n]} for n in rec["patterns"] if n in have}
        print(f"[exp] recovered: {len(blob)} ternarias R3, {len(nb_ov)} normas/biases, resto FP (promovidas)")

    # dequant TBC por lineal (vectorizado, idéntico a dequantized_state)
    from tbc.search import dequantize_ternary as _dq
    dq: dict[str, torch.Tensor] = {}
    scales_used: dict[str, float] = {}
    for n, e in blob.items():
        dq[n] = _dq(e["W"].to(torch.int8), e["alpha"].float(), 32)

    state = model.state_dict()
    if nb_ov:
        state = {k: nb_ov[k].to(v.dtype) if k in nb_ov else v for k, v in state.items()}
        print(f"[exp] override normas/biases entrenados: {len(nb_ov)}")
    n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads

    def llama_permute_qk(t2d: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Mirror de conversion/llama.py:163 (undo_permute RoPE en q/k)."""
        rows, cols = t2d.shape
        return (t2d.reshape(n_heads, 2, rows // n_heads // 2, cols)
                    .swapaxes(1, 2).reshape(rows, cols).contiguous())

    def permute_for(ln: str, t2d: torch.Tensor) -> torch.Tensor:
        if ln.endswith("attn_q.weight"):
            return llama_permute_qk(t2d, n_head)
        if ln.endswith("attn_k.weight"):
            return llama_permute_qk(t2d, n_kv if n_head != n_kv else n_head)
        return t2d

    import sys as _sys
    PURE = "--pure" in _sys.argv
    items_f32: list[tuple[str, list[int], int, bytes]] = []
    i2s_items: list[tuple[str, list[int], int, bytes]] = []
    n_tbc = 0
    for hn, w in state.items():
        ln = hf_to_llama(hn)
        if ln is None:
            print("[exp] SKIP", hn, w.shape)
            continue
        wf = w.detach().float()
        base = hn[:-7] if hn.endswith(".weight") else hn
        if PURE and base in dq:
            # pesos HF originales (modelo base para benchmark justo)
            from tbc.linalg import weight_out_in as _woi
            mods = dict(model.named_modules())
            wf = _woi(mods[base])
            dims = [wf.shape[0]] if wf.dim() == 1 else [wf.shape[1], wf.shape[0]]
            items_f32.append((ln, dims, GGML_TYPE_F32, f32_blob(wf)))
            i2s_items.append((ln, dims, GGML_TYPE_F32, f32_blob(wf)))
            continue
        if base in dq:  # lineal ternarizado
            n_tbc += 1
            Wt = blob[base]["W"].to(torch.int8)
            Wt = permute_for(ln, Wt)
            Worig = permute_for(ln, w.detach().float())
            s = optimal_global_scale(Worig, Wt)
            scales_used[ln] = s
            if wf.shape[1] % 128 == 0 and Wt.numel() % 128 == 0:
                # I2_S nativo intercalado (runtime BitNet)
                wf = permute_for(ln, dq[base])  # F32 dequant para el GGUF F32
                i2s_items.append((ln, [wf.shape[1], wf.shape[0]], GGML_TYPE_I2_S,
                                  pack_i2_s_interleaved(Wt, s)))
            else:
                # ancho no múltiplo de 128: el lector I2_S del fork lo exige;
                # se conserva F32 (cuantización mixta, soportada por el loader)
                print(f"[exp] {ln} ne0={wf.shape[1]} no mult.128 -> F32 (mixto)")
                wf = permute_for(ln, dq[base])
                i2s_items.append((ln, [wf.shape[1], wf.shape[0]], GGML_TYPE_F32, f32_blob(wf)))
        else:
            d1 = [wf.shape[0]] if wf.dim() == 1 else [wf.shape[1], wf.shape[0]]
            i2s_items.append((ln, d1, GGML_TYPE_F32, f32_blob(wf)))
        dims = [wf.shape[0]] if wf.dim() == 1 else [wf.shape[1], wf.shape[0]]
        items_f32.append((ln, dims, GGML_TYPE_F32, f32_blob(wf)))
    print(f"[exp] tensores: {len(items_f32)} (lineales TBC: {n_tbc})")

    # ---- metadatos ----
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kvb = b""
    kvb += kv("general.architecture", 8, "llama")
    kvb += kv("general.name", 8, "NanoDex-1M-TBC-1.58")
    kvb += kv("general.quantization_version", 4, 2)
    kvb += kv("general.file_type", 4, 0)
    A = "llama."
    kvb += kv(A + "block_count", 4, cfg.num_hidden_layers)
    kvb += kv(A + "context_length", 4, cfg.max_position_embeddings)
    kvb += kv(A + "embedding_length", 4, cfg.hidden_size)
    kvb += kv(A + "feed_forward_length", 4, cfg.intermediate_size)
    kvb += kv(A + "attention.head_count", 4, cfg.num_attention_heads)
    kvb += kv(A + "attention.head_count_kv", 4, cfg.num_key_value_heads)
    kvb += kv(A + "attention.layer_norm_rms_epsilon", 6, cfg.rms_norm_eps)
    kvb += kv(A + "rope.dimension_count", 4, head_dim)
    kvb += kv(A + "rope.freq_base", 6, float(getattr(cfg, "rope_theta", 10000.0) or 10000.0))
    n_kv = 13

    # ---- tokenizador BPE ----
    vocab = tok.get_vocab()
    inv = [None] * len(vocab)
    for t, i in vocab.items():
        inv[i] = t
    assert all(x is not None for x in inv), "vocab incompleto"
    try:
        merges = json.loads(tok.backend_tokenizer.to_str())["model"]["merges"]
        merges = [m if isinstance(m, str) else " ".join(m) for m in merges]
    except Exception as e:
        print("[exp] sin merges:", e)
        merges = []
    scores = [0.0] * len(inv)
    ttype = [1] * len(inv)
    if len(inv) > 0:
        ttype[0] = 3
    if len(inv) > 1:
        ttype[1] = 3
    kvb += kv("tokenizer.ggml.model", 8, "gpt2")
    kvb += kv_arr("tokenizer.ggml.tokens", 8, inv)
    kvb += kv_arr("tokenizer.ggml.scores", 6, scores)
    kvb += kv_arr("tokenizer.ggml.token_type", 4, ttype)
    kvb += kv_arr("tokenizer.ggml.merges", 8, merges)
    kvb += kv("tokenizer.ggml.bos_token_id", 4, int(tok.bos_token_id or 0))
    kvb += kv("tokenizer.ggml.eos_token_id", 4, int(tok.eos_token_id or 0))
    kvb += kv("tokenizer.ggml.padding_token_id", 4, int(tok.pad_token_id or 1))
    kvb += kv("tokenizer.ggml.add_bos_token", 7, False)
    kvb += kv("tokenizer.ggml.add_eos_token", 7, False)
    n_kv += 10
    # trazabilidad TBC
    kvb += kv("tbc.compiler_version", 8, "tbc-1.0.0")
    kvb += kv("tbc.mode", 8, str(manifest.get("mode", "fast")))
    kvb += kv("tbc.scales_json", 8, json.dumps({k: round(v, 6) for k, v in scales_used.items()})[:8192])
    n_kv += 3

    r1 = write_gguf(OUT_F32, kvb, n_kv, items_f32)
    print("[exp] F32:", r1, flush=True)
    r2 = write_gguf(OUT_I2S, kvb, n_kv, i2s_items)
    print("[exp] I2_S:", r2, flush=True)


if __name__ == "__main__":
    main()
