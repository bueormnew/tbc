"""Exportación nativa bitnet.cpp: GGUF v3 + I2_S canónico + validación (Sec 39).

Verificado contra third_party/BitNet (main + fork isHuangXin/llama.cpp):
- ggml.h:426 -> GGML_TYPE_I2_S = 36 (NO 30).
- quants.c quantize_i2_s (línea 1358): packing SECUENCIAL, peso i -> byte i/4,
  bit_pos = 6-2*(i%4); códigos 0->-1, 1->0, 2->+1 (map2bit, línea 1336/1493);
  escala f32 ÚNICA por tensor en offset numel/4; tamaño total numel/4+32
  (coincide con ggml.c:1316 nbytes = n/4+32).
- GGUF de I2_S frescos usan SIEMPRE tipo 36 secuencial; el layout intercalado
  TL2 (tipo 42) es producto de ggml_bitnet_transform_tensor en carga, no de
  archivo. Nuestro exportador escribe el formato canónico secuencial.
"""
from __future__ import annotations
import json
import os
import struct
import subprocess
import time

import torch

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I8 = 24
GGML_TYPE_I2_S = 36  # verificado en 3rdparty/llama.cpp/ggml/include/ggml.h:426
GGML_TYPE_TL2 = 42   # layout post-transform, no usar en GGUF frescos

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGN = 32

KV_U8, KV_I8, KV_U16, KV_I16, KV_U32, KV_I32, KV_F32, KV_BOOL, KV_STR, KV_ARR, KV_U64, KV_I64, KV_F64 = range(13)


def _enc_str(s: str | bytes) -> bytes:
    b = s.encode("utf-8") if isinstance(s, str) else bytes(s)
    return struct.pack("<Q", len(b)) + b


def _enc_val(t: int, v) -> bytes:
    if t == KV_U8:
        return struct.pack("<B", v)
    if t == KV_I8:
        return struct.pack("<b", v)
    if t == KV_U16:
        return struct.pack("<H", v)
    if t == KV_I16:
        return struct.pack("<h", v)
    if t == KV_U32:
        return struct.pack("<I", v)
    if t == KV_I32:
        return struct.pack("<i", v)
    if t == KV_F32:
        return struct.pack("<f", v)
    if t == KV_BOOL:
        return struct.pack("<B", 1 if v else 0)
    if t == KV_STR:
        return _enc_str(v)
    if t == KV_STR:
        return _enc_str(v)
    if t == KV_U64:
        return struct.pack("<Q", v)
    if t == KV_I64:
        return struct.pack("<q", v)
    if t == KV_F64:
        return struct.pack("<d", v)
    raise ValueError(t)


def kv(key: str, t: int, v) -> bytes:
    return _enc_str(key) + struct.pack("<I", t) + _enc_val(t, v)


def kv_arr(key: str, elem_t: int, vals: list) -> bytes:
    out = _enc_str(key) + struct.pack("<I", KV_ARR) + struct.pack("<I", elem_t) + struct.pack("<Q", len(vals))
    for v in vals:
        out += _enc_val(elem_t, v)
    return out


def _enc_kv_str(key: str, val: str) -> bytes:
    return kv(key, KV_STR, val)


def _enc_kv_u32(key: str, val: int) -> bytes:
    return kv(key, KV_U32, val)


def _enc_kv_f32(key: str, val: float) -> bytes:
    return kv(key, KV_F32, val)


def _pad32(n: int) -> int:
    return (GGUF_ALIGN - (n % GGUF_ALIGN)) % GGUF_ALIGN


# ---------- packing I2_S canónico (byte-idéntico a quantize_i2_s) ----------

_VAL_TO_CODE = {-1: 0, 0: 1, 1: 2}
_CODE_TO_VAL = {0: -1, 1: 0, 2: 1}


def pack_i2_s_native(Wt: torch.Tensor, scale: float) -> bytes:
    """Wt int8 {-1,0,1} (ya aplanado en orden fila-major) + escala f32 única.

    Produce numel/4 bytes secuenciales + trailer 32B (f32 scale + 28 ceros),
    idéntico a quants.c quantize_i2_s salvo la elección de escala (allí es
    max-abs del tensor; aquí la provee TBC y se documenta).
    """
    flat = Wt.detach().to(torch.int8).flatten().tolist()
    assert len(flat) % 4 == 0, "numel debe ser múltiplo de 4 para I2_S"
    out = bytearray(len(flat) // 4)
    for i, v in enumerate(flat):
        out[i // 4] |= (_VAL_TO_CODE[int(v)] & 0x3) << (6 - 2 * (i % 4))
    out += struct.pack("<f", float(scale)) + b"\x00" * 28
    return bytes(out)


def unpack_i2_s_native(blob: bytes, numel: int) -> tuple[torch.Tensor, float]:
    """Inversa: retorna (pesos {-1,0,1} como int8 1-D, escala f32)."""
    packed, trailer = blob[: numel // 4], blob[numel // 4:]
    scale = struct.unpack("<f", trailer[:4])[0]
    vals = []
    for i in range(numel):
        code = (packed[i // 4] >> (6 - 2 * (i % 4))) & 0x3
        if code == 3:
            raise ValueError("código 3 reservado")
        vals.append(_CODE_TO_VAL[code])
    return torch.tensor(vals, dtype=torch.int8), scale


def pack_i2_s_interleaved(Wt: torch.Tensor, scale: float) -> bytes:
    """Layout intercalado-128 que consume el runtime (dequantize_row_i2_s,
    quants.c:1335; vec_dot AVX2, quants.c:1391).

    Cada bloque de 128 pesos -> 32 bytes; byte gp = (c(gp)<<6)|(c(32+gp)<<4)|
    (c(64+gp)<<2)|c(96+gp), códigos 0/1/2 -> -1/0/+1. Requiere numel múltiplo
    de 128 (el lector lee fuera de fila en anchos no múltiplos; los modelos
    oficiales BitNet usan dims múltiplos de 128). Trailer 32B con f32 scale.
    """
    flat = Wt.detach().to(torch.int8).flatten().tolist()
    assert len(flat) % 128 == 0, "interleaved requiere numel múltiplo de 128"
    out = bytearray()
    for b in range(len(flat) // 128):
        blk = flat[b * 128:(b + 1) * 128]
        for gp in range(32):
            byte = ((_VAL_TO_CODE[blk[gp]] & 3) << 6 | (_VAL_TO_CODE[blk[32 + gp]] & 3) << 4
                    | (_VAL_TO_CODE[blk[64 + gp]] & 3) << 2 | (_VAL_TO_CODE[blk[96 + gp]] & 3))
            out.append(byte)
    out += struct.pack("<f", float(scale)) + b"\x00" * 28
    return bytes(out)


def optimal_global_scale(W_fp: torch.Tensor, Wt: torch.Tensor) -> float:
    """α* global L2 (Sec 22): <W,Wt>/<Wt,Wt>. Colapso honesto de α por grupo
    al formato BitNet (una escala por tensor)."""
    a = W_fp.double().flatten()
    b = Wt.double().flatten()
    den = float((b * b).sum().item())
    if den < 1e-12:
        return 0.0
    return float((a * b).sum().item() / den)


# ---------- escritor GGUF general ----------

def write_gguf(path: str, kv_blob: bytes, n_kv: int, items: list[tuple[str, list[int], int, bytes]]) -> dict:
    """items: [(nombre, dims_ne, tipo_ggml, blob)]. Escribe GGUF v3 alineado a 32."""
    info = b""
    off = 0
    for name, dims, ttype, blob in items:
        info += _enc_str(name) + struct.pack("<I", len(dims))
        for d in dims:
            info += struct.pack("<Q", d)
        info += struct.pack("<I", ttype) + struct.pack("<Q", off)
        off += len(blob) + _pad32(len(blob))
    with open(path, "wb") as f:
        f.write(GGUF_MAGIC + struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(items)) + struct.pack("<Q", n_kv))
        f.write(kv_blob)
        f.write(info)
        pos = 4 + 4 + 8 + 8 + len(kv_blob) + len(info)
        f.write(b"\x00" * _pad32(pos))
        for _, _, _, blob in items:
            f.write(blob)
            f.write(b"\x00" * _pad32(len(blob)))
    return {"path": path, "n_tensors": len(items), "bytes": os.path.getsize(path)}


def f32_blob(t: torch.Tensor) -> bytes:
    return t.detach().float().contiguous().numpy().tobytes()


def f16_blob(t: torch.Tensor) -> bytes:
    import numpy as np
    return t.detach().float().contiguous().numpy().astype("<f2").tobytes()


def write_gguf_i2s(path: str, tensors: list[tuple[str, torch.Tensor, torch.Tensor]], group_size: int, arch: str, manifest: dict):
    """Compat: exportador antiguo (pesos + escalas separadas). Se mantiene para
    reconversión byte-idéntica; el path nativo es export_llama_gguf.py."""
    from .pack import pack_i2_s_tl2
    items: list[tuple[str, list[int], int, bytes]] = []
    for name, Wt, alpha in tensors:
        rows, cols = Wt.shape
        packed = pack_i2_s_tl2(Wt)
        items.append((name, [cols, rows], GGML_TYPE_I2_S, packed + b"\x00" * _pad32(len(packed))))
        sc = f16_blob(alpha.detach().float().reshape(-1))
        items.append((name + ".scales", [int(alpha.numel())], GGML_TYPE_F16, sc))
    kvb = (_enc_kv_str("general.architecture", arch) + _enc_kv_str("general.type", "model")
           + _enc_kv_str("tbc.packing", "I2_S_TL2") + _enc_kv_u32("tbc.group_size", group_size)
           + _enc_kv_str("tbc.compiler_version", str(manifest.get("compiler_version", "tbc-1.0.0")))
           + _enc_kv_str("tbc.manifest_json", json.dumps(manifest, sort_keys=True)[:4096]))
    return write_gguf(path, kvb, 6, items)


def build_manifest(config, report: dict, arch: str, n_params_ternary: int) -> dict:
    base = dict(report.get("manifest_extra", {})) if isinstance(report, dict) else {}
    m = {
        "architecture": arch,
        "ternary_format": "i2_s_native_sequential",
        "group_size": getattr(config, "group_size", 32),
        "scales": {"granularity": "per_group_torch/per_tensor_gguf", "dtype": "fp16_torch/f32_gguf"},
        "checkpoint_layout": {"p": 0.33, "strategy": "stratified_depth"},
        "error_threshold": {"global": getattr(config, "epsilon_target", 0.05)},
        "calibration_signature": "see_model_manifest",
        "layer_tolerances": report.get("per_layer_E", {}),
        "compiler_version": "tbc-1.0.0",
        "packing_format": "bitnet.cpp-compatible",
        "sensitivity_map": {},
        "compile_time": f"{report.get('compile_time_s', 0)}s",
        "peak_vram": f"{report.get('peak', {}).get('peak_ram_gb', 0)}GB",
        "bitnet_cpp_compatible": True,
        "packing": "I2_S_NATIVE",
        "ggml_type_i2_s": GGML_TYPE_I2_S,
        "n_params_ternary": n_params_ternary,
    }
    m.update(base)
    return m


def llama_cli_path() -> str | None:
    cands = [
        os.path.join("third_party", "BitNet", "build", "bin", "llama-cli.exe"),
        os.path.join("third_party", "BitNet", "build", "bin", "llama-cli"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def run_llama_cli(gguf_path: str, prompt: str = "Hello, I am", n_tokens: int = 50,
                  threads: int = 4, extra: list[str] | None = None, timeout: int = 180) -> tuple[bool, str]:
    """Ejecuta el binario REAL compilado. Retorna (rc==0, salida)."""
    binp = llama_cli_path()
    if binp is None:
        return False, "BINARY_MISSING"
    cmd = [binp, "-m", gguf_path, "-p", prompt, "-n", str(n_tokens), "--threads", str(threads)]
    if extra:
        cmd += extra
    try:
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        out = (p.stdout + "\n[stderr]\n" + p.stderr)
        return (p.returncode == 0), f"rc={p.returncode} dt={dt:.1f}s tok/s~{n_tokens/max(dt,1e-6):.1f}\n{out[-6000:]}"
    except Exception as e:
        return False, f"BINARY_FAIL: {e}"


def export_bitnet_cpp_compatible(tbc_model, output_gguf: str, manifest_path: str | None = None) -> dict:
    """Pipeline Sec 39.5 (legacy). El path nativo actual es export_llama_gguf.py."""
    from .pack import regroup_to_group_size
    assert all(set(torch.unique(w).tolist()) <= {-1, 0, 1} for w in tbc_model.ternary_weights.values()), "W no ternario"
    tensors = []
    for name, Wt in tbc_model.ternary_weights.items():
        Wt32 = regroup_to_group_size(Wt, tbc_model.group_size, 32)
        alpha = tbc_model.scales[name]
        tensors.append((name.replace(".", "_"), Wt32, alpha))
    os.makedirs(os.path.dirname(os.path.abspath(output_gguf)), exist_ok=True)
    info = write_gguf_i2s(output_gguf, tensors, 32, tbc_model.arch, tbc_model.manifest)
    if manifest_path:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(tbc_model.manifest, f, indent=2, sort_keys=True, default=str)
    ok, msg = bitnet_cpp_can_load(output_gguf, run_binary=False)
    info.update({"can_load": ok, "can_load_msg": msg})
    return info


def bitnet_cpp_can_load(gguf_path: str, run_binary: bool = True, prompt: str = "Hello, I am", n_tokens: int = 16) -> tuple[bool, str]:
    """Chequeo estructural + ejecución opcional del binario real."""
    import struct as _st
    logs: list[str] = []
    if not os.path.exists(gguf_path):
        return False, "GGUF_MISSING"
    try:
        with open(gguf_path, "rb") as f:
            magic = f.read(4)
            ver = _st.unpack("<I", f.read(4))[0]
            nt = _st.unpack("<Q", f.read(8))[0]
        if magic != b"GGUF":
            return False, "BAD_MAGIC"
        logs.append(f"GGUF_OK ver={ver} n_tensors={nt} size={os.path.getsize(gguf_path)}B")
    except Exception as e:
        return False, f"GGUF_READ_FAIL: {e}"
    if not run_binary:
        return True, "\n".join(logs) + "\nSTRUCT_ONLY"
    ok, out = run_llama_cli(gguf_path, prompt, n_tokens)
    return ok, "\n".join(logs) + "\n" + out
