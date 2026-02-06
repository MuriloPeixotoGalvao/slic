from __future__ import annotations
import numpy as np

def iter_tiles(H: int, W: int, tile: int, overlap: int):
    stride = max(1, int(tile - overlap))
    for y0 in range(0, H, stride):
        y1 = min(H, y0 + tile)
        y0 = max(0, y1 - tile)
        for x0 in range(0, W, stride):
            x1 = min(W, x0 + tile)
            x0 = max(0, x1 - tile)
            yield x0, y0, x1, y1

def paste_inner_region(
    full_u16: np.ndarray,
    tile_u16: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
    overlap: int,
    W: int, H: int
):
    cut = overlap // 2
    ix0 = x0 + (cut if x0 > 0 else 0)
    iy0 = y0 + (cut if y0 > 0 else 0)
    ix1 = x1 - (cut if x1 < W else 0)
    iy1 = y1 - (cut if y1 < H else 0)

    tx0 = ix0 - x0
    ty0 = iy0 - y0
    tx1 = tx0 + (ix1 - ix0)
    ty1 = ty0 + (iy1 - iy0)

    if ix1 > ix0 and iy1 > iy0:
        full_u16[iy0:iy1, ix0:ix1] = tile_u16[ty0:ty1, tx0:tx1]
