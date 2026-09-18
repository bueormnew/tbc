import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
txt = open(r"C:\Users\gerso\Desktop\TBC\eval_text.txt").read()
ids = tok(txt)["input_ids"]
print("HF tokens:", len(ids))
txt2 = open(r"C:\Users\gerso\Desktop\TBC\eval_text_big.txt").read()
print("HF tokens big:", len(tok(txt2)["input_ids"]))
