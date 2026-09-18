import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
from transformers import AutoTokenizer, GPT2Tokenizer
from tbc.calib_text import CORPUS
for name, cls, kw in [("nano", AutoTokenizer, {"pretrained_model_name_or_path": "DedeProGames/NanoDex-1M"}),
                      ("gpt2", GPT2Tokenizer, {"pretrained_model_name_or_path": "openai-community/gpt2"})]:
    tok = cls.from_pretrained(trust_remote_code=True, **kw) if name == "nano" else cls.from_pretrained(**kw)
    print(name, "corpus tokens:", len(tok(CORPUS)["input_ids"]))
