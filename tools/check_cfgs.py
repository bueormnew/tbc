import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import transformers as T

cands = ["LlamaConfig", "Qwen2Config", "Qwen3Config", "Qwen3MoeConfig",
         "Qwen2MoeConfig", "MistralConfig", "MixtralConfig", "GemmaConfig",
         "Gemma2Config", "PhiConfig", "Phi3Config", "FalconConfig",
         "GPT2Config", "GPTNeoXConfig", "DeepseekV2Config", "DeepSeekV2Config",
         "MptConfig", "BloomConfig", "GPTBigCodeConfig", "OlmoConfig",
         "Olmo2Config", "GlmConfig", "Gemma3Config", "Qwen1Config"]
have = {}
for c in cands:
    have[c] = hasattr(T, c)
    print(c, "OK" if have[c] else "--")
