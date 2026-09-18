"""Kernel test I2_S (Opción B, Sec 39): verifica el GGUF contra la semántica exacta
del runtime BitNet con un decodificador INDEPENDIENTE (mirror de
dequantize_row_i2_s, quants.c:1335 + map2bit quants.c:1336/1493).

1. Parsea nanodex-tbc-i2s.gguf, extrae un tensor I2_S + su trailer de escala.
2. Decodifica con el mirror-C (bucle done/gp/cols0..3 idéntico al C).
3. Compara patrones ternarios vs TBC (deben coincidir 100% en signo/cero).
4. Compara Y_kernel = X@dequant vs Y_ref(torch, misma escala): diff < 1e-3.
5. Verifica nbytes = numel/4 + 32 (ggml.c:1316).
"""
import struct
import sys
sys.path.insert(0, r"C:\Users\gerso\Desktop\TBC")
import torch

GGUF = r"C:\Users\gerso\Desktop\TBC\tbc_output\nanodex-tbc-i2s.gguf"
MAP2BIT = [-1.0, 0.0, 1.0, 0.0]  # quants.c:1336


def parse_gguf(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"GGUF"
    ver = struct.unpack("<I", data[4:8])[0]
    nt = struct.unpack("<Q", data[8:16])[0]
    nkv = struct.unpack("<Q", data[16:24])[0]
    pos = 24
    def read_str():
        nonlocal pos
        (ln,) = struct.unpack("<Q", data[pos:pos+8]); pos += 8
        s = data[pos:pos+ln]; pos += ln
        return s
    def skip_val(t):
        nonlocal pos
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8:
            (ln,) = struct.unpack("<Q", data[pos:pos+8]); pos += 8 + ln
        elif t == 9:
            et = struct.unpack("<I", data[pos:pos+4])[0]; pos += 4
            (ln,) = struct.unpack("<Q", data[pos:pos+8]); pos += 8
            for _ in range(ln):
                skip_val(et)
        else:
            pos += sizes[t]
    for _ in range(nkv):
        read_str(); t = struct.unpack("<I", data[pos:pos+4])[0]; pos += 4
        skip_val(t)
    infos = []
    for _ in range(nt):
        name = read_str().decode()
        (nd,) = struct.unpack("<I", data[pos:pos+4]); pos += 4
        dims = [struct.unpack("<Q", data[pos+8*i:pos+8*i+8])[0] for i in range(nd)]; pos += 8*nd
        (tt,) = struct.unpack("<I", data[pos:pos+4]); pos += 4
        (off,) = struct.unpack("<Q", data[pos:pos+8]); pos += 8
        infos.append((name, dims, tt, off))
    data_start = pos + (32 - (pos % 32)) % 32
    return ver, infos, data, data_start


def mirror_dequantize_row_i2_s(x: bytes, n: int, scale: float) -> list:
    """Mirror exacto de quants.c:1335-1356."""
    y = [0.0] * n
    done = 0
    while done < n:
        cols0 = min(32, n - done)
        cols1 = min(32, n - done - 32)
        cols2 = min(32, n - done - 64)
        cols3 = min(32, n - done - 96)
        for gp in range(32):
            byte = x[(done // 4) + gp]
            c0, c1, c2, c3 = (byte >> 6) & 3, (byte >> 4) & 3, (byte >> 2) & 3, byte & 3
            if gp < cols0: y[done + gp] = scale * MAP2BIT[c0]
            if gp < cols1: y[done + 32 + gp] = scale * MAP2BIT[c1]
            if gp < cols2: y[done + 64 + gp] = scale * MAP2BIT[c2]
            if gp < cols3: y[done + 96 + gp] = scale * MAP2BIT[c3]
        done += 128
    return y


def main():
    ver, infos, data, dstart = parse_gguf(GGUF)
    print(f"GGUF ver={ver} tensores={len(infos)}")
    # elige blk.0.attn_q (I2_S, ne0=128 conformante)
    tgt = next(t for t in infos if t[0] == "blk.0.attn_q.weight")
    name, dims, tt, off = tgt
    assert tt == 36, f"tipo esperado 36, hallado {tt}"
    ne0, ne1 = dims[0], dims[1]
    numel = ne0 * ne1
    nbytes_expected = numel // 4 + 32
    blob = data[dstart+off:dstart+off+nbytes_expected]
    assert len(blob) == nbytes_expected, "nbytes != numel/4+32"
    print(f"{name}: dims={dims} numel={numel} blob={len(blob)}B (=numel/4+32 OK)")
    packed, trailer = blob[:numel//4], blob[numel//4:]
    scale = struct.unpack("<f", trailer[:4])[0]
    assert trailer[4:] == b"\x00" * 28, "padding trailer no cero"
    print(f"scale f32 = {scale:.6f}")

    # decodifica fila a fila como el runtime (n = ne0 por fila)
    assert numel % ne0 == 0
    dec = []
    for r in range(numel // ne0):
        row_packed = packed[r*(ne0//4):(r+1)*(ne0//4)]
        # el runtime indexa el bloque empaquetado global con done global; por fila
        # contigua equivale a decodificar el stream completo de una vez:
        dec = None
    full = mirror_dequantize_row_i2_s(packed, numel, scale)
    Wd = torch.tensor(full).reshape(ne1, ne0)
    print(f"dequant ok: min={Wd.min():.4f} max={Wd.max():.4f} (esperado ±{scale:.4f} y 0)")

    # compara patrones vs TBC
    tbc = torch.load(r"C:\Users\gerso\Desktop\TBC\tbc_output\model.tbc\weights.pt", map_location="cpu")
    Wt = tbc["model.layers.0.self_attn.q_proj"]["W"].float()
    agree = ((Wd.sign() == Wt.reshape(ne1, ne0).sign())).float().mean().item()
    print(f"acuerdo de patrones ternarios kernel-vs-TBC: {agree*100:.2f}%")
    assert agree == 1.0, "patrones difieren"

    # Y_kernel vs Y_torch misma escala
    torch.manual_seed(0)
    X = torch.randn(64, ne0)
    Yk = X @ Wd.T
    Yt = X @ (Wt.reshape(ne1, ne0).float() * scale).T
    diff = float(((Yk - Yt).abs().max()).item())
    print(f"max|Y_kernel - Y_torch| = {diff:.2e} (umbral 1e-3)")
    assert diff < 1e-3, "KERNEL FAIL"
    print("KERNEL TEST PASS")


if __name__ == "__main__":
    main()
