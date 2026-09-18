import sys, os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
import transformers as T
m = T.MixtralForCausalLM(T.MixtralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=1, num_attention_heads=4,
                                        num_key_value_heads=2, num_local_experts=2,
                                        num_experts_per_tok=1))
for n, mod in m.named_modules():
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.nn.Parameter):
        print("mixtral:", n, tuple(w.shape))
m2 = T.Qwen3MoeForCausalLM(T.Qwen3MoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                           num_hidden_layers=1, num_attention_heads=4,
                                           num_key_value_heads=2, num_experts=2,
                                           num_experts_per_tok=1, shared_expert_intermediate_size=32))
for n, mod in m2.named_modules():
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.nn.Parameter):
        print("qwen3moe:", n, tuple(w.shape))
