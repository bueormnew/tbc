import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tbc.perplexity import perplexity

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
model.eval()
tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
txt = open(r"C:\Users\gerso\Desktop\TBC\data\eval_corpus.txt").read()
ids = tok(txt, return_tensors="pt")["input_ids"]
print("tokens:", ids.shape[1])
wins = [ids[:, i:i+32] for i in range(0, ids.shape[1], 32) if ids[:, i:i+32].shape[1] == 32]
eval_ids = torch.cat(wins, dim=0)
print("windows:", eval_ids.shape[0])
print("PPL torch-fp32 corpus =", round(perplexity(model, eval_ids), 3), flush=True)
