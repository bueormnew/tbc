import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("DedeProGames/NanoDex-1M", trust_remote_code=True)
txt = open(r"C:\Users\gerso\Desktop\TBC\data\prompt.txt").read()
print("HF:", tok(txt)["input_ids"])
