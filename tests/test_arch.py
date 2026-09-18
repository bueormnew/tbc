"""Compatibilidad de arquitecturas SIN descargas (modelos sintéticos diminutos).

Verifica por arquitectura: detección, objetivos (incluye expertos MoE,
excluye routers) y, en llama/mixtral, una compilación TBC smoke.
"""
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("tbc-arch")


def tiny(cls_name, **kw):
    import transformers as T
    cls = getattr(T, cls_name, None)
    if cls is None:
        return None
    cfg_name = cls_name.replace("ForCausalLM", "Config").replace("LMHeadModel", "Config")
    cfg_cls = getattr(T, cfg_name, None)
    if cfg_cls is None:
        return None
    return cls(cfg_cls(**kw))


# (model_class, config_kwargs, arch_esperado, checks)
CASES = [
    ("LlamaForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=4,
                              num_key_value_heads=2), "llama", {}),
    ("Qwen2ForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=4,
                              num_key_value_heads=2), "qwen2", {}),
    ("Qwen3ForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=4,
                              num_key_value_heads=2), "qwen3", {}),
    ("MistralForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=4,
                                num_key_value_heads=2), "mistral", {}),
    ("MixtralForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=4,
                                num_key_value_heads=2, num_local_experts=2,
                                num_experts_per_tok=1), "mixtral",
     {"must_exclude": "mlp.gate"}),
    ("Gemma2ForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=1, num_attention_heads=4,
                               num_key_value_heads=2), "gemma2", {}),
    ("Phi3ForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2), "phi3", {"must_include": "qkv_proj"}),
    ("FalconForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=1, num_attention_heads=4), "falcon",
     {"must_include": "query_key_value"}),
    ("GPT2LMHeadModel", dict(vocab_size=64, n_embd=32, n_layer=1, n_head=4,
                             n_inner=64), "gpt2", {"must_include": "c_attn"}),
    ("GPTNeoXForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=4), "gpt_neox",
     {"must_include": "query_key_value"}),
    ("BloomForCausalLM", dict(vocab_size=64, hidden_size=32, n_inner=64,
                              n_layer=1, n_head=4), "bloom", {"must_include": "query_key_value"}),
    ("DeepseekV2ForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=1, num_attention_heads=4,
                                   num_key_value_heads=2, q_lora_rank=None,
                                   kv_lora_rank=8, qk_rope_head_dim=8, v_head_dim=8,
                                   qk_nope_head_dim=8), "deepseek_v2", {}),
    ("Olmo2ForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=4,
                              num_key_value_heads=2), "olmo2", {}),
    ("Qwen3MoeForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=1, num_attention_heads=4,
                                 num_key_value_heads=2, num_experts=2,
                                 num_experts_per_tok=1, shared_expert_intermediate_size=32),
     "qwen3moe", {"must_exclude": "mlp.gate"}),
]


def fake_moe_checks():
    """Estructura MoE sintética (estilo antiguo: expertos por separado).
    Prueba inclusión de expertos + exclusión del router sin depender de
    los internos de transformers.
    """
    import torch.nn as nn
    from tbc.compiler import get_ternarize_targets

    class FakeMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.ModuleDict({
                "self_attn": nn.ModuleDict({"q_proj": nn.Linear(8, 8)}),
                "mlp": nn.ModuleDict({
                    "experts": nn.ModuleList([nn.ModuleDict({"w1": nn.Linear(8, 8),
                                                             "w3": nn.Linear(8, 8)})]),
                    "gate": nn.Linear(8, 2)})})])

    for arch in ("mixtral", "qwen3moe"):
        names = [n for n, _ in get_ternarize_targets(FakeMoE(), arch)]
        assert any("experts" in n for n in names), f"{arch}: expertos no incluidos"
        assert not any("mlp.gate" in n for n in names), f"{arch}: router incluido"
        log.info("OK fake-MoE %-10s objetivos=%d (router excluido)", arch, len(names))


def main():
    from tbc.compiler import detect_architecture, get_ternarize_targets
    ok, skipped, failed = 0, 0, []
    for cls_name, kw, expected, checks in CASES:
        try:
            m = tiny(cls_name, **kw)
        except Exception as e:
            log.info("SKIP %s (kwargs): %s", cls_name, str(e)[:100])
            skipped += 1
            continue
        if m is None:
            log.info("SKIP %s (sin clase)", cls_name)
            skipped += 1
            continue
        m.eval()
        try:
            arch = detect_architecture(m)
            tgts = get_ternarize_targets(m, arch)
            names = [n for n, _ in tgts]
            assert arch == expected, f"arch {arch} != {expected}"
            assert names, "sin objetivos"
            if checks.get("must_include"):
                assert any(checks["must_include"] in n for n in names), f"falta {checks['must_include']}"
            if checks.get("must_exclude"):
                assert not any(checks["must_exclude"] in n for n in names), f"router incluido: {checks['must_exclude']}"
            log.info("OK %-22s arch=%-10s objetivos=%d", cls_name, arch, len(names))
            ok += 1
        except AssertionError as e:
            log.info("FAIL %s: %s", cls_name, e)
            failed.append(cls_name)
    # smoke de compilación en llama-tiny y mixtral-tiny (MoE)
    for cls_name, kw in [("LlamaForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                                                   num_hidden_layers=1, num_attention_heads=4,
                                                   num_key_value_heads=2)),
                         ("MixtralForCausalLM", dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                                                     num_hidden_layers=1, num_attention_heads=4,
                                                     num_key_value_heads=2, num_local_experts=2,
                                                     num_experts_per_tok=1))]:
        m = tiny(cls_name, **kw)
        if m is None:
            continue
        m.eval()
        from tbc.config import TBCConfig
        from tbc.compiler import TBC_COMPILE
        cfg = TBCConfig(beam_size=2, epsilon_target=0.9, epsilon_schedule=[0.9],
                        max_passes_per_layer=1, candidates_per_group=4,
                        max_tokens_per_layer=256)
        g = torch.Generator().manual_seed(0)
        cal = torch.randint(0, 64, (4, 16), generator=g)
        _, rep = TBC_COMPILE(m, cfg, cal)
        log.info("SMOKE %-20s E=%.3f status=%s", cls_name, rep["global_E"], rep["status"])
    log.info("arch: ok=%d skipped=%d failed=%s", ok, skipped, failed)
    assert not failed, f"fallos: {failed}"
    fake_moe_checks()


if __name__ == "__main__":
    main()
