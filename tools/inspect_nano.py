import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import json
from transformers import AutoConfig, AutoTokenizer
c = AutoConfig.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
print("model_type:", c.model_type)
for k in ["vocab_size","hidden_size","intermediate_size","num_hidden_layers","num_attention_heads","num_key_value_heads","max_position_embeddings","rms_norm_eps","rope_theta","tie_word_embeddings","hidden_act","head_dim"]:
    print(k, "=", getattr(c, k, "N/A"))
tok = AutoTokenizer.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
print("tok class:", type(tok).__name__, "len:", len(tok))
print("bos:", tok.bos_token_id, "eos:", tok.eos_token_id, "pad:", tok.pad_token_id, "unk:", tok.unk_token_id)
print("sample decode:", tok.decode([1,2,3,4,5])[:80])
