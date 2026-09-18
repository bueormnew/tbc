import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import json
p = r"C:\Users\gerso\Desktop\TBC\tbc_output\manifest.json"
m = json.load(open(p))
m["gguf_native"] = {
  "file": "tbc_output/model-i2_s.gguf",
  "sha256": "e273978f7d5d72e8b5b9f6ffc311016fbd7c2c088f7f96fdb940bbb0858515fd",
  "bytes": 3066560,
  "ggml_type_i2_s": 36,
  "layout": "interleaved-128 + f32 per-tensor scale, numel/4+32B",
  "quant_mix": "30 lineares I2_S (patrones R2 + alfas R3 colapsadas) + 5 ffn_down F32",
  "tensors": 48,
  "model": "NanoDex-1M TBC-R (R3: PPL torch 54.57)",
  "reconversion": "byte-identical via export_llama_gguf.py --recovered tbc_output_r (deterministic)",
}
m["bitnet_verification"] = {
  "method": "binary + kernel-test",
  "build": "third_party/BitNet/build/bin/ (cmake, GGML_BITNET_X86_TL2=OFF flags oficiales)",
  "binary_ppl_f32_r_gguf": 111.9,
  "binary_ppl_i2s_r_gguf": 354.8,
  "binary_ppl_f32_r0_gguf": 552.8,
  "generation_50_tokens": "logs/nano_r_gen.log (1665 tok/s, I2_S kernels)",
  "kernel_test": "logs/kernel_test.log (patrones 100%, diff 0.0 < 1e-3)",
  "compile_log": "logs/bitnet_compile.log",
}
json.dump(m, open(p, "w"), indent=2, sort_keys=True, default=str)
print("manifest OK")
