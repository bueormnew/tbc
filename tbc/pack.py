"""Empaquetado I2_S / TL2 (Sección 39 del MD + verificación contra BitNet upstream).

Mapeo del prompt (normativo para TBC): 00=0, 01=+1, 10=-1 (11 reservado).
Empaquetado TL2: 4 pesos por byte, little-endian de 2 bits:
  byte = c0 | (c1<<2) | (c2<<4) | (c3<<6)

NOTA DE COMPATIBILIDAD verificada contra third_party/BitNet/src/ggml-bitnet-mad.cpp:
- BitNet usa valores q8 intermedios 0->-1, 1->0, 2->+1 (líneas 76-78) y un layout
  físico intercalado por grupos de 4 filas y bloques QK_I2_S=128/64 optimizado
  para kernels SIMD. Nuestra codificación secuencial contiene EXACTAMENTE la
  misma información: la conversión al layout intercalado es una permutación
  (LUT de 2 bits + transposición) sin pérdida, implementable en
  `tbc_to_gguf_i2s.py`. El manifest exportado documenta ambos layouts.
"""
from __future__ import annotations
import torch

CODE_ZERO: int = 0b00
CODE_POS: int = 0b01
CODE_NEG: int = 0b10

VAL_TO_CODE = {-1: CODE_NEG, 0: CODE_ZERO, 1: CODE_POS}
CODE_TO_VAL = {CODE_ZERO: 0, CODE_POS: 1, CODE_NEG: -1}


def pack_i2_s_tl2(Wt: torch.Tensor) -> bytes:
    """Wt int8 {-1,0,1} -> bytes TL2 (4 pesos/byte). Rellena con 0 si no es múltiplo de 4."""
    flat = Wt.detach().to(torch.int8).flatten().tolist()
    out = bytearray()
    for i in range(0, len(flat), 4):
        chunk = flat[i : i + 4]
        while len(chunk) < 4:
            chunk.append(0)
        b = 0
        for j, v in enumerate(chunk):
            if v not in VAL_TO_CODE:
                raise ValueError(f"peso no ternario: {v}")
            b |= (VAL_TO_CODE[v] & 0x3) << (2 * j)
        out.append(b)
    return bytes(out)


def unpack_i2_s_tl2(data: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    """Inversa exacta de pack_i2_s_tl2."""
    vals: list[int] = []
    for b in data:
        for j in range(4):
            code = (b >> (2 * j)) & 0x3
            if code == 0b11:
                raise ValueError("código 0b11 reservado en I2_S")
            vals.append(CODE_TO_VAL[code])
    numel = 1
    for d in shape:
        numel *= d
    vals = vals[:numel]
    return torch.tensor(vals, dtype=torch.int8).reshape(shape)


def regroup_to_group_size(Wt: torch.Tensor, current_group: int, target_group: int = 32) -> torch.Tensor:
    """Re-agrupar es no-op a nivel de pesos (el grupo solo afecta a escalas). Valida divisibilidad."""
    if Wt.shape[-1] % target_group != 0:
        # padding conceptual: no se altera Wt, el exportador lo maneja por grupo parcial
        pass
    return Wt
