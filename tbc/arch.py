"""Specs por arquitectura: lineales objetivo, GGUF y particularidades (Sec 2 del prompt).

Arquitecturas soportadas end-to-end por el compilador:
- llama (LlamaForCausalLM, NanoDex-1M, ...): nn.Linear q/k/v/o/gate/up/down
- qwen2/qwen3 (Qwen2ForCausalLM): mismos lineales que llama + RoPE
- gpt2 (GPT2LMHeadModel): Conv1D c_attn/c_proj/c_fc, QKV fusionada, sin RoPE
- fallback genérico: todo nn.Linear (+ módulos con weight 2D)
"""
from __future__ import annotations

ARCH_SPECS: dict[str, dict] = {
    "llama": {
        "linears": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        "gguf_arch": "llama",
        "rope": True,
        "fused_qkv": None,
        "notes": " Familia principal. Escalas por grupo, I2_S ne0 múltiplo de 128.",
    },
    "qwen2": {
        "linears": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        "gguf_arch": "qwen2",
        "rope": True,
        "fused_qkv": None,
        "notes": "Mismo pipeline que llama; GGUF arch=qwen2, mismos nombres blk.*.",
    },
    "qwen3": {
        "linears": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        "gguf_arch": "qwen3",
        "rope": True,
        "fused_qkv": None,
        "notes": "Igual que qwen2; verificar head_dim/GQA del config al exportar.",
    },
    "gpt2": {
        "lineales": ("c_attn", "c_proj", "c_fc"),
        "linears": ("c_attn", "c_proj", "c_fc"),
        "gguf_arch": "gpt2",
        "rope": False,
        "fused_qkv": "c_attn",  # [in, 3*in] -> split q/k/v al exportar
        "pos_emb": "wpe",
        "notes": "Conv1D [in,out] (ver tbc/linalg.py). Sin RoPE: usa wpe aprendido.",
    },
}


def arch_spec(arch: str) -> dict:
    return ARCH_SPECS.get(arch, {"linears": (), "gguf_arch": arch or "unknown",
                                 "rope": True, "fused_qkv": None, "notes": "fallback genérico"})


def supported_arches() -> list[str]:
    return sorted(ARCH_SPECS)
