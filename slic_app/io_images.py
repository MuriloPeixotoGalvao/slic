from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import streamlit as st

def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Garante RGB uint8 para overlay/visual."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img[:, :, :3]

    if img.dtype == np.uint8:
        return img

    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)

    if np.issubdtype(img.dtype, np.floating):
        x = np.clip(img, 0.0, 1.0) if img.max() <= 1.5 else np.clip(img, 0.0, 255.0)
        if x.max() <= 1.5:
            x = (x * 255.0)
        return x.astype(np.uint8)

    x = img.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        x = (x - mn) / (mx - mn) * 255.0
    else:
        x = x * 0.0
    return x.astype(np.uint8)

@st.cache_data(show_spinner=False)
def _file_mtime(p: str) -> float:
    try:
        return Path(p).stat().st_mtime
    except Exception:
        return -1.0

@st.cache_data(show_spinner=False)
def imread_rgb_cached(path_str: str, mtime: float) -> np.ndarray:
    img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Falha ao ler: {path_str}")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img[:, :, :3]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _to_rgb_uint8(rgb)

def imread_rgb(path: Path) -> np.ndarray:
    return imread_rgb_cached(str(path), _file_mtime(str(path)))

def imread_rgb_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Falha ao decodificar imagem enviada.")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img[:, :, :3]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _to_rgb_uint8(rgb)
