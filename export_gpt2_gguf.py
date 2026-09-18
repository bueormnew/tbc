#!/usr/bin/env python
"""export_gpt2_gguf.py — GPT-2-TBC -> GGUF estándar arch=gpt2 (F32 e I2_S nativo).

- Conv1D c_attn fusionada [in,3*in] -> split q/k/v ([in,in] c/u).
- c_fc/c_proj directos. wpe + ln_f + output atado incluidos.
- I2_S intercalado-128 + escala f32 global (tipo 36). GPT-2 small tiene todos
  los ne0 múltiplos de 128 -> I2_S completo, sin excepciones F32.
Uso: python export_gpt2_gguf.py [tbc_dir] [out_f32] [out_i2s]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from tbc.export import (
    GGML_TYPE_F32, GGML_TYPE_I2_S,
    kv, kv_arr, f32_blob, write_gguf, pack_i2_s_interleaved,
    optimal_global_scale,
)
from tbc.search import dequantize_ternary

BASE = os.path.dirname(os.path.abspath(__file__))
TBC_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "tbc_output_gpt2", "model.tbc")
OUT_F32 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "tbc_output_gpt2", "gpt2-tbc-f32.gguf")
OUT_I2S = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE, "tbc_output_gpt2", "gpt2-tbc-i2s.gguf")


def get_merges(tok) -> list:
    bt = getattr(tok, "backend_tokenizer", None)
    if bt is not None:
        try:
            m = json.loads(bt.to_str())["model"]["merges"]
            return [x if isinstance(x, str) else " ".join(x) for x in m]
        except Exception:
            pass
    br = getattr(tok, "bpe_ranks", None)
    if br:
        return ["%s %s" % p for p, _ in sorted(br.items(), key=lambda kv: kv[1])]
    return []


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[exp-gpt2] cargando modelo + tokenizer...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
    tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    cfg = model.config
    blob = torch.load(os.path.join(TBC_DIR, "weights.pt"), map_location="cpu")
    # TBC_RECOVERED=<dir>: usa patrones/alfas R2+R3 + normas/biases entrenados;
    # las capas promovidas (ausentes en recovered) quedan FP originales.
    REC = os.environ.get("TBC_RECOVERED") or ("--recovered" in sys.argv and sys.argv[sys.argv.index("--recovered") + 1] if "--recovered" in sys.argv else None)
    nb_ov: dict[str, torch.Tensor] = {}
    if REC:
        rec = torch.load(os.path.join(REC, "recovered.pt"), map_location="cpu")
        nb_ov = {k: v.float() for k, v in torch.load(os.path.join(REC, "norms_bias.pt"), map_location="cpu").items()}
        blob = {n: {"W": rec["patterns"][n], "alpha": rec["alphas_r3"][n]}
                for n in rec["patterns"] if n in rec.get("alphas_r3", {})}
        print(f"[exp-gpt2] recovered: {len(blob)} ternarias, {len(nb_ov)} normas/biases, resto FP (promovidas)")
        print(f"[exp-gpt2] recovered: {len(blob)} ternarias, {len(nb_ov)} normas/biases, promovidas FP")
    n_layer, n_embd = cfg.n_layer, cfg.n_embd
    n_head = cfg.n_head
    assert cfg.n_inner in (None, 4 * n_embd), "MLP no estándar"
    n_inner = cfg.n_inner or 4 * n_embd

    # W_oi [out,in] dequant por lineal TBC (Conv1D -> transponer).
    # --pure: usa pesos HF originales (aisla bugs del exportador).
    PURE = os.environ.get("PURE_FP32") == "1" or "--pure" in sys.argv

    def tbc_dequant(hname):
        base = hname[:-7] if hname.endswith(".weight") else hname
        if base not in blob:
            return None, None  # promovida: FP original
        if PURE:
            mod = None
            for nn_, mm in model.named_modules():
                if nn_ == base:
                    mod = mm
                    break
            W_oi = mod.weight.detach().float().T.clone()
            return W_oi, None
        e = blob[base]
        Wt = e["W"].to(torch.int8)
        return dequantize_ternary(Wt, e["alpha"].float(), 32), Wt

    state = model.state_dict()
    if nb_ov:
        state = {k: nb_ov[k].to(v.dtype) if k in nb_ov else v for k, v in state.items()}
        print(f"[exp-gpt2] override normas/biases entrenados: {len(nb_ov)}")
    items_f32, items_i2s = [], []
    n_tbc = 0
    scales_used = {}
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        # QKV fusionada (este fork espera blk.i.attn_qkv.weight).
        # Convención oficial (conversion/gpt2.py:32): Conv1D [in,out] -> transpose
        # -> [out,in]; el blob GGUF son las filas [out,in] en C-order con dims [in,out].
        Wqkv_oi, Wt_qkv = tbc_dequant(p + "attn.c_attn")
        E = n_embd
        ln = f"blk.{i}.attn_qkv.weight"
        if Wqkv_oi is None:  # promovida: original
            Wqkv_oi = state[p + "attn.c_attn.weight"].detach().float().T.clone()
            items_f32.append((ln, [E, 3 * E], GGML_TYPE_F32, f32_blob(Wqkv_oi)))
            items_i2s.append((ln, [E, 3 * E], GGML_TYPE_F32, f32_blob(Wqkv_oi)))
            n_tbc += 0
        else:
            items_f32.append((ln, [E, 3 * E], GGML_TYPE_F32, f32_blob(Wqkv_oi)))
            if PURE or Wt_qkv is None:
                items_i2s.append((ln, [E, 3 * E], GGML_TYPE_F32, f32_blob(Wqkv_oi)))
                scales_used[ln] = 1.0
            else:
                s = optimal_global_scale(state[p + "attn.c_attn.weight"].detach().float().T, Wt_qkv)
                scales_used[ln] = s
                items_i2s.append((ln, [E, 3 * E], GGML_TYPE_I2_S, pack_i2_s_interleaved(Wt_qkv, s)))
            n_tbc += 1
        bqkv = state[p + "attn.c_attn.bias"].detach().float()
        items_f32.append((f"blk.{i}.attn_qkv.bias", [3 * E], GGML_TYPE_F32, f32_blob(bqkv)))
        items_i2s.append((f"blk.{i}.attn_qkv.bias", [3 * E], GGML_TYPE_F32, f32_blob(bqkv)))
        for hn, ln, transpose in [
            (p + "attn.c_proj.weight", f"blk.{i}.attn_output.weight", True),
            (p + "mlp.c_fc.weight", f"blk.{i}.ffn_up.weight", True),
            (p + "mlp.c_proj.weight", f"blk.{i}.ffn_down.weight", True),
        ]:
            W_oi, Wtb = tbc_dequant(hn)
            if W_oi is None:  # promovida: FP original (con normas/biases entrenados ya en state)
                Wp = state[hn].detach().float().T.clone()
                in_f, out_f = Wp.shape[1], Wp.shape[0]
                items_f32.append((ln, [in_f, out_f], GGML_TYPE_F32, f32_blob(Wp)))
                items_i2s.append((ln, [in_f, out_f], GGML_TYPE_F32, f32_blob(Wp)))
                scales_used[ln] = 1.0
            else:
                in_f, out_f = W_oi.shape[1], W_oi.shape[0]
                items_f32.append((ln, [in_f, out_f], GGML_TYPE_F32, f32_blob(W_oi)))
                if PURE:
                    items_i2s.append((ln, [in_f, out_f], GGML_TYPE_F32, f32_blob(W_oi)))
                    scales_used[ln] = 1.0
                else:
                    s = optimal_global_scale(state[hn].detach().float().T, Wtb)
                    scales_used[ln] = s
                    items_i2s.append((ln, [in_f, out_f], GGML_TYPE_I2_S, pack_i2_s_interleaved(Wtb, s)))
                n_tbc += 1
            bh = hn[:-7] + ".bias"  # bias F32 original (no se ternariza)
            if bh in state:
                b = state[bh].detach().float()
                items_f32.append((ln[:-7] + ".bias", [b.numel()], GGML_TYPE_F32, f32_blob(b)))
                items_i2s.append((ln[:-7] + ".bias", [b.numel()], GGML_TYPE_F32, f32_blob(b)))
        # biases de normas (qkv ya lleva su bias fusionado arriba)
        for hn, ln in [(p + "ln_1.weight", f"blk.{i}.attn_norm.weight"),
                       (p + "ln_2.weight", f"blk.{i}.ffn_norm.weight")]:
            w = state[hn].detach().float()
            items_f32.append((ln, [w.numel()], GGML_TYPE_F32, f32_blob(w)))
            items_i2s.append((ln, [w.numel()], GGML_TYPE_F32, f32_blob(w)))
            bh = hn[:-7] + ".bias"
            b = state[bh].detach().float()
            items_f32.append((ln[:-7] + ".bias", [b.numel()], GGML_TYPE_F32, f32_blob(b)))
            items_i2s.append((ln[:-7] + ".bias", [b.numel()], GGML_TYPE_F32, f32_blob(b)))
    for hn, ln in [("transformer.wte.weight", "token_embd.weight"),
                   ("transformer.wpe.weight", "position_embd.weight"),
                   ("transformer.ln_f.weight", "output_norm.weight")]:
        w = state[hn].detach().float()
        dims = [w.shape[1], w.shape[0]] if w.dim() == 2 else [w.numel()]
        items_f32.append((ln, dims, GGML_TYPE_F32, f32_blob(w)))
        items_i2s.append((ln, dims, GGML_TYPE_F32, f32_blob(w)))
    blf = state["transformer.ln_f.bias"].detach().float()
    items_f32.append(("output_norm.bias", [blf.numel()], GGML_TYPE_F32, f32_blob(blf)))
    items_i2s.append(("output_norm.bias", [blf.numel()], GGML_TYPE_F32, f32_blob(blf)))
    wte = state["transformer.wte.weight"].detach().float()
    items_f32.append(("output.weight", [wte.shape[1], wte.shape[0]], GGML_TYPE_F32, f32_blob(wte)))
    items_i2s.append(("output.weight", [wte.shape[1], wte.shape[0]], GGML_TYPE_F32, f32_blob(wte)))
    print(f"[exp-gpt2] lineales TBC: {n_tbc}, tensores: {len(items_f32)}")

    kvb = b""
    kvb += kv("general.architecture", 8, "gpt2")
    kvb += kv("general.name", 8, "GPT2-small-TBC-1.58")
    kvb += kv("general.quantization_version", 4, 2)
    kvb += kv("general.file_type", 4, 0)
    G = "gpt2."
    kvb += kv(G + "block_count", 4, n_layer)
    kvb += kv(G + "context_length", 4, cfg.n_ctx)
    kvb += kv(G + "embedding_length", 4, n_embd)
    kvb += kv(G + "feed_forward_length", 4, n_inner)
    kvb += kv(G + "attention.head_count", 4, n_head)
    kvb += kv(G + "attention.layer_norm_epsilon", 6, cfg.layer_norm_epsilon)
    n_kv = 10
    vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else tok.encoder
    inv = [None] * len(vocab)
    for t, ii in vocab.items():
        inv[ii] = t
    assert all(x is not None for x in inv)
    merges = get_merges(tok)
    print(f"[exp-gpt2] vocab={len(inv)} merges={len(merges)}")
    scores = [0.0] * len(inv)
    ttype = [1] * len(inv)
    kvb += kv("tokenizer.ggml.model", 8, "gpt2")
    kvb += kv("tokenizer.ggml.pre", 8, "gpt-2")
    kvb += kv_arr("tokenizer.ggml.tokens", 8, inv)
    kvb += kv_arr("tokenizer.ggml.scores", 6, scores)
    kvb += kv_arr("tokenizer.ggml.token_type", 4, ttype)
    kvb += kv_arr("tokenizer.ggml.merges", 8, merges)
    kvb += kv("tokenizer.ggml.bos_token_id", 4, int(tok.bos_token_id if tok.bos_token_id is not None else 50256))
    kvb += kv("tokenizer.ggml.eos_token_id", 4, int(tok.eos_token_id if tok.eos_token_id is not None else 50256))
    kvb += kv("tokenizer.ggml.add_bos_token", 7, False)
    kvb += kv("tokenizer.ggml.add_eos_token", 7, False)
    n_kv += 10
    kvb += kv("tbc.compiler_version", 8, "tbc-1.0.0")
    kvb += kv("tbc.mode", 8, "fast-gpt2")
    n_kv += 2

    r1 = write_gguf(OUT_F32, kvb, n_kv, items_f32)
    print("[exp-gpt2] F32:", r1, flush=True)
    r2 = write_gguf(OUT_I2S, kvb, n_kv, items_i2s)
    print("[exp-gpt2] I2_S:", r2, flush=True)


if __name__ == "__main__":
    main()
