import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
# Prepara texto de evaluación para llama-perplexity (inglés real, ~2KB)
text = (
    "The quick brown fox jumps over the lazy dog. "
    "Language models predict the next token given previous context. "
    "Ternary quantization constrains weights to minus one, zero, or plus one. "
    "Behavioral compilation searches discrete configurations that preserve function. "
) * 12
open(r"C:\Users\gerso\Desktop\TBC\eval_text.txt", "w").write(text)
print("bytes:", len(text))
