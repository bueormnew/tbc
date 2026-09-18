import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
from tbc.calib_text import CORPUS
open(r"C:\Users\gerso\Desktop\TBC\eval_corpus.txt", "w").write(CORPUS.strip() + "\n")
print("corpus chars:", len(CORPUS))
