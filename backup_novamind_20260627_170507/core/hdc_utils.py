"""HDC bit-packing utilities: compress bipolar ±1 vectors from 40KB → 1.25KB."""

import torch
import numpy as np

DIM = 10000
PACKED_BYTES = DIM // 8  # 1250

def pack_bipolar(vec: torch.Tensor) -> bytes:
    """Pack a 10000D bipolar float32 (±1) vector into 1250 bytes (1 bit/dim)."""
    bits = (vec.view(-1) > 0).cpu().numpy().astype(np.uint8)[:DIM]
    return np.packbits(bits).tobytes()

def unpack_bipolar(data: bytes, dim: int = DIM) -> torch.Tensor:
    """Unpack 1250 bytes back to 10000D bipolar float32 {+1, -1} tensor."""
    packed = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(packed)[:dim]
    return torch.from_numpy(bits.astype(np.float32) * 2.0 - 1.0)
