"""Matriz de arquitecturas TBC: detección, lineales objetivo, GGUF y notas.

Niveles de soporte (honestos):
- stable:       compilado + exportado + ejecutado en binario con pesos reales.
- compatible:   detección y compilación verificadas en modelos sintéticos
                (sin descargas) + nombres GGUF según el conversor oficial;
                pendiente validación comunitaria con pesos reales.
- experimental: particularidades sin validar (MLA, fused no estándar...).

MoE: se compilan los expertos (lineales) y se EXCLUYE el router/gate,
que debe quedar en precisión alta (es diminuto y crítico).
"""
from __future__ import annotations

_LLAMA_LIKE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

ARCH_SPECS: dict[str, dict] = {
    "llama": {
        "hf_types": ["llama"], "linears": _LLAMA_LIKE, "exclude": (),
        "moe": None, "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "llama", "meta_prefix": "llama",
        "support": "stable", "notes": "Familia principal. I2_S ne0 múltiplo de 128.",
    },
    "qwen2": {
        "hf_types": ["qwen2"], "linears": _LLAMA_LIKE, "exclude": (),
        "moe": None, "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "qwen2", "meta_prefix": "qwen2",
        "support": "compatible", "notes": "Mismo pipeline que llama; metadatos qwen2.*.",
    },
    "qwen3": {
        "hf_types": ["qwen3"], "linears": _LLAMA_LIKE, "exclude": (),
        "moe": None, "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "qwen3", "meta_prefix": "qwen3",
        "support": "compatible", "notes": "Qwen3 denso. q/k norm extra quedan FP.",
    },
    "qwen3moe": {
        "hf_types": ["qwen3_moe"],
        "linears": _LLAMA_LIKE + ("gate_proj", "up_proj", "down_proj", "experts"),
        "exclude": ("mlp.gate", "mlp.shared_expert_gate"),
        "moe": {"experts_sub": "experts", "router_sub": ("mlp.gate",)},
        "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "qwen3moe", "meta_prefix": "qwen3moe",
        "support": "experimental", "notes": "MoE: expertos ternarizables, router FP. GGUF apilado por validar.",
    },
    "qwen2moe": {
        "hf_types": ["qwen2_moe"],
        "linears": _LLAMA_LIKE + ("gate_proj", "up_proj", "down_proj", "experts"),
        "exclude": ("mlp.gate", "mlp.shared_expert_gate"),
        "moe": {"experts_sub": "experts", "router_sub": ("mlp.gate",)},
        "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "qwen2moe", "meta_prefix": "qwen2moe",
        "support": "experimental", "notes": "Igual que qwen3moe.",
    },
    "mistral": {
        "hf_types": ["mistral"], "linears": _LLAMA_LIKE, "exclude": (),
        "moe": None, "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "mistral", "meta_prefix": "mistral",
        "support": "compatible", "notes": " Denominators GQA; mismo pipeline llama.",
    },
    "mixtral": {
        "hf_types": ["mixtral"],
        "linears": ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3", "experts"),
        "exclude": ("mlp.gate", "block_sparse_moe.gate"),
        "moe": {"experts_sub": "experts", "router_sub": ("mlp.gate", "block_sparse_moe.gate")},
        "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "mixtral", "meta_prefix": "mixtral",
        "support": "compatible",
        "notes": "Expertos por separado (w1/w2/w3) o fusionados (experts.*) se compilan "
                 "como matrices; el router queda FP. En transformers nuevos los expertos "
                 "son un tensor apilado único bajo mlp.experts.",
    },
    "gemma": {
        "hf_types": ["gemma"], "linears": _LLAMA_LIKE, "exclude": (),
        "moe": None, "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "gemma", "meta_prefix": "gemma",
        "support": "compatible", "notes": "Normas especiales quedan FP.",
    },
    "gemma2": {
        "hf_types": ["gemma2"], "linears": _LLAMA_LIKE, "exclude": (),
        "moe": None, "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "gemma2", "meta_prefix": "gemma2",
        "support": "compatible", "notes": "Sliding-window no afecta a pesos.",
    },
    "gemma3": {
        "hf_types": ["gemma3", "gemma3_text"],
        "linears": _LLAMA_LIKE + ("q_norm", "k_norm"),
        "exclude": (),
        "moe": None, "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "gemma3", "meta_prefix": "gemma3",
        "support": "experimental", "notes": "q/k_norm son lineales 1D: revisar granularidad.",
    },
    "phi": {
        "hf_types": ["phi"],
        "linears": ("qkv_proj", "out_proj", "fc1", "fc2"),
        "exclude": (),
        "moe": None,
        "fused": {"qkv_proj": {"parts": ["q", "k", "v"], "kind": "qkv"}},
        "conv1d": False, "rope": False,
        "gguf_arch": "phi2", "meta_prefix": "phi2",
        "support": "experimental", "notes": "qkv fusionada (rotary parcial): validar split.",
    },
    "phi3": {
        "hf_types": ["phi3"],
        "linears": ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
        "exclude": (),
        "moe": None,
        "fused": {"qkv_proj": {"parts": ["q", "k", "v"], "kind": "qkv"},
                  "gate_up_proj": {"parts": ["gate", "up"], "kind": "gu"}},
        "conv1d": False, "rope": True,
        "gguf_arch": "phi3", "meta_prefix": "phi3",
        "support": "compatible", "notes": "Doble fusión; el split por filas es exacto.",
    },
    "falcon": {
        "hf_types": ["falcon", "rwkv"],
        "linears": ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
        "exclude": (),
        "moe": None,
        "fused": {"query_key_value": {"parts": ["q", "k", "v"], "kind": "qkv"}},
        "conv1d": False, "rope": False,
        "gguf_arch": "falcon", "meta_prefix": "falcon",
        "support": "compatible", "notes": "Falcon-7B QKV fusionada; 40B/180B verificar layout.",
    },
    "gpt2": {
        "hf_types": ["gpt2"],
        "linears": ("c_attn", "c_proj", "c_fc"),
        "exclude": (),
        "moe": None,
        "fused": {"c_attn": {"parts": ["q", "k", "v"], "kind": "qkv"}},
        "conv1d": True, "rope": False,
        "gguf_arch": "gpt2", "meta_prefix": "gpt2",
        "support": "stable", "notes": "Conv1D [in,out]; QKV fusionada; wpe aprendido.",
    },
    "gpt_neox": {
        "hf_types": ["gpt_neox"],
        "linears": ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
        "exclude": (),
        "moe": None,
        "fused": {"query_key_value": {"parts": ["q", "k", "v"], "kind": "qkv"}},
        "conv1d": False, "rope": False,
        "gguf_arch": "gptneox", "meta_prefix": "gptneox",
        "support": "compatible", "notes": "Rotary parcial; QKV fusionada estándar.",
    },
    "gpt_bigcode": {
        "hf_types": ["gpt_bigcode"],
        "linears": ("c_attn", "c_proj", "c_fc"),
        "exclude": (),
        "moe": None,
        "fused": {"c_attn": {"parts": ["q", "k", "v"], "kind": "qkv"}},
        "conv1d": True, "rope": False,
        "gguf_arch": "starcoder", "meta_prefix": "starcoder",
        "support": "experimental", "notes": "Multi-query attention: validar split por cabezas.",
    },
    "bloom": {
        "hf_types": ["bloom"],
        "linears": ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
        "exclude": (),
        "moe": None,
        "fused": {"query_key_value": {"parts": ["q", "k", "v"], "kind": "qkv"}},
        "conv1d": False, "rope": False,
        "gguf_arch": "bloom", "meta_prefix": "bloom",
        "support": "compatible", "notes": "Alibi (sin RoPE); QKV fusionada estándar.",
    },
    "mpt": {
        "hf_types": ["mpt"],
        "linears": ("Wqkv", "out_proj", "up_proj", "down_proj"),
        "exclude": (),
        "moe": None,
        "fused": {"Wqkv": {"parts": ["q", "k", "v"], "kind": "qkv"}},
        "conv1d": False, "rope": False,
        "gguf_arch": "mpt", "meta_prefix": "mpt",
        "support": "experimental", "notes": "Verificar nombres exactos según versión MPT.",
    },
    "deepseek_v2": {
        "hf_types": ["deepseek_v2"],
        "linears": ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj", "experts"),
        "exclude": ("mlp.gate",),
        "moe": {"experts_sub": "mlp.experts.", "router_sub": ("mlp.gate",)},
        "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "deepseek2", "meta_prefix": "deepseek2",
        "support": "experimental",
        "notes": "MLA: la absorción KV cambia el layout GGUF; validar con pesos reales.",
    },
    "olmo": {
        "hf_types": ["olmo"], "linears": _LLAMA_LIKE, "exclude": (),
        "moe": None, "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "olmo", "meta_prefix": "olmo",
        "support": "experimental", "notes": "Sin RoPE con theta (rotary parcial).",
    },
    "olmo2": {
        "hf_types": ["olmo2"], "linears": _LLAMA_LIKE, "exclude": (),
        "moe": None, "fused": None, "conv1d": False, "rope": True,
        "gguf_arch": "olmo2", "meta_prefix": "olmo2",
        "support": "compatible", "notes": "QK-norm y post-norm quedan FP.",
    },
    "glm": {
        "hf_types": ["glm", "glm4", "chatglm"],
        "linears": ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
                    "self_attention.query_key_value"),
        "exclude": (),
        "moe": None,
        "fused": {"query_key_value": {"parts": ["q", "k", "v"], "kind": "qkv"}},
        "conv1d": False, "rope": True,
        "gguf_arch": "glm4", "meta_prefix": "glm4",
        "support": "experimental", "notes": "RMSNorm parcial + DeepNorm residual: validar.",
    },
    "qwen": {
        "hf_types": ["qwen"],
        "linears": ("c_attn", "c_proj", "w1", "w2", "c_proj"),
        "exclude": (),
        "moe": None,
        "fused": {"c_attn": {"parts": ["q", "k", "v"], "kind": "qkv"}},
        "conv1d": False, "rope": True,
        "gguf_arch": "qwen", "meta_prefix": "qwen",
        "support": "experimental", "notes": "Qwen1 (sin clase local): validar nombres con pesos reales.",
    },
}

# model_type (HF config) -> arch TBC. Orden: específicos antes que genéricos.
MODEL_TYPE_MAP: list[tuple[str, str]] = [
    ("qwen3_moe", "qwen3moe"),
    ("qwen2_moe", "qwen2moe"),
    ("qwen3", "qwen3"),
    ("qwen2", "qwen2"),
    ("qwen", "qwen"),
    ("llama", "llama"),
    ("mistral", "mistral"),
    ("mixtral", "mixtral"),
    ("gemma3", "gemma3"),
    ("gemma2", "gemma2"),
    ("gemma", "gemma"),
    ("phi3", "phi3"),
    ("phi", "phi"),
    ("falcon", "falcon"),
    ("gpt_bigcode", "gpt_bigcode"),
    ("gpt_neox", "gpt_neox"),
    ("gpt2", "gpt2"),
    ("deepseek_v2", "deepseek_v2"),
    ("bloom", "bloom"),
    ("mpt", "mpt"),
    ("olmo2", "olmo2"),
    ("olmo", "olmo"),
    ("glm4", "glm"),
    ("glm", "glm"),
    ("chatglm", "glm"),
]


def detect_arch_from_type(model_type: str) -> str:
    mt = (model_type or "").lower().replace("_", "").replace("-", "")
    for key, arch in MODEL_TYPE_MAP:
        if key.replace("_", "") in mt:
            return arch
    return "unknown"


def arch_spec(arch: str) -> dict:
    return ARCH_SPECS.get(arch, {"linears": (), "exclude": (), "moe": None,
                                 "fused": None, "conv1d": False, "rope": True,
                                 "gguf_arch": arch or "unknown", "meta_prefix": arch or "unknown",
                                 "support": "experimental", "hf_types": [],
                                 "notes": "fallback genérico: todos los lineales 2D."})


def supported_arches() -> list[str]:
    return sorted(ARCH_SPECS)
