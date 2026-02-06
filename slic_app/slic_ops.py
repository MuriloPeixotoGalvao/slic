from __future__ import annotations
from typing import Tuple, Optional
from collections import deque
import numpy as np
import cv2
import streamlit as st

from skimage.segmentation import slic, find_boundaries
from skimage.util import img_as_float

from .knn import knn_predict_numpy_balanced
from .supcon import TORCH_OK, embed_all_feats, ContrastiveHead

@st.cache_data(show_spinner=False)
def compute_slic_labels_cached(
    rgb: np.ndarray,
    n_segments: int,
    compactness: float,
    sigma: float,
    enforce_connectivity: bool,
    key: str,
) -> np.ndarray:
    labels = slic(
        img_as_float(rgb),
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        enforce_connectivity=bool(enforce_connectivity),
        channel_axis=-1,
    )
    return labels.astype(np.int32)

@st.cache_data(show_spinner=False)
def compute_superpixel_features_cached(rgb: np.ndarray, labels: np.ndarray, key: str) -> np.ndarray:
    """12 feats: mean/var Lab (6), mean RGB (3), gradiente (1), centro (2)."""
    H, W, _ = rgb.shape
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    K = int(labels.max()) + 1
    feats = np.zeros((K, 12), np.float32)

    yy, xx = np.indices((H, W))
    flat_lab = lab.reshape(-1, 3).astype(np.float32)
    flat_rgb = rgb.reshape(-1, 3).astype(np.float32)
    flat_lbl = labels.reshape(-1)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    gmag = cv2.magnitude(gx, gy).reshape(-1)

    xx_f = (xx.reshape(-1).astype(np.float32) / max(1, W - 1))
    yy_f = (yy.reshape(-1).astype(np.float32) / max(1, H - 1))

    for k in range(K):
        sel = (flat_lbl == k)
        if not np.any(sel):
            continue
        vlab = flat_lab[sel]
        vrgb = flat_rgb[sel]
        feats[k, 0:3] = vlab.mean(axis=0)
        feats[k, 3:6] = vlab.var(axis=0)
        feats[k, 6:9] = vrgb.mean(axis=0)
        feats[k, 9] = gmag[sel].mean()
        feats[k, 10] = xx_f[sel].mean()
        feats[k, 11] = yy_f[sel].mean()

    return feats

def zscore(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)

@st.cache_data(show_spinner=False)
def build_adjacency_csr_cached(labels: np.ndarray, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Grafo de adjacência em CSR: indptr (K+1), indices (E)."""
    lbl = labels.astype(np.int32, copy=False)
    K = int(lbl.max()) + 1

    a = lbl[:, :-1].ravel()
    b = lbl[:, 1:].ravel()
    m = (a != b)
    pairs_r = np.stack([a[m], b[m]], axis=1) if np.any(m) else np.empty((0, 2), np.int32)

    a2 = lbl[:-1, :].ravel()
    b2 = lbl[1:, :].ravel()
    m2 = (a2 != b2)
    pairs_d = np.stack([a2[m2], b2[m2]], axis=1) if np.any(m2) else np.empty((0, 2), np.int32)

    pairs = np.vstack([pairs_r, pairs_d]).astype(np.int32, copy=False)
    if pairs.size == 0:
        indptr = np.zeros(K + 1, np.int32)
        indices = np.empty((0,), np.int32)
        return indptr, indices

    pairs = np.vstack([pairs, pairs[:, ::-1]])
    pairs = np.unique(pairs, axis=0)

    src = pairs[:, 0]
    dst = pairs[:, 1]

    order = np.argsort(src, kind="mergesort")
    src = src[order]
    dst = dst[order]

    counts = np.bincount(src, minlength=K).astype(np.int32)
    indptr = np.zeros(K + 1, np.int32)
    indptr[1:] = np.cumsum(counts)

    indices = dst.astype(np.int32, copy=False)
    return indptr, indices

def expand_region_by_similarity(
    start_sid: int,
    feats_space: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    dist_thr: float,
    max_nodes: int = 500,
    mode: str = "contiguo",
) -> np.ndarray:
    """
    mode:
      - "contiguo": BFS só por vizinhos e similares ao SPX inicial
      - "global": todos os SPX similares (não precisa contiguidade)
    """
    K = feats_space.shape[0]
    start_sid = int(start_sid)
    if start_sid < 0 or start_sid >= K:
        return np.array([start_sid], np.int32)

    v0 = feats_space[start_sid].astype(np.float32, copy=False)
    thr2 = float(dist_thr * dist_thr)

    if mode == "global":
        d2 = ((feats_space.astype(np.float32) - v0[None, :]) ** 2).sum(axis=1)
        sel = np.where(d2 <= thr2)[0].astype(np.int32)
        if sel.size > max_nodes:
            idx = np.argpartition(d2[sel], kth=max_nodes - 1)[:max_nodes]
            sel = sel[idx]
        if start_sid not in sel:
            sel = np.concatenate([np.array([start_sid], np.int32), sel])
        return np.unique(sel)

    visited = np.zeros(K, dtype=np.uint8)
    out = []
    q = deque([start_sid])
    visited[start_sid] = 1
    out.append(start_sid)

    while q and len(out) < int(max_nodes):
        u = q.popleft()
        neigh = indices[indptr[u]:indptr[u + 1]]
        for v in neigh:
            v = int(v)
            if visited[v]:
                continue
            visited[v] = 1
            dv2 = float(((feats_space[v].astype(np.float32, copy=False) - v0) ** 2).sum())
            if dv2 <= thr2:
                out.append(v)
                q.append(v)
                if len(out) >= int(max_nodes):
                    break

    return np.array(out, np.int32)

def majority_label_per_superpixel(
    slic_labels: np.ndarray,
    labelmap_u16: np.ndarray,
    n_classes: int,
    min_frac: float,
    UNLAB: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lbl = slic_labels.reshape(-1).astype(np.int64)
    lm  = labelmap_u16.reshape(-1).astype(np.int64)
    K = int(slic_labels.max()) + 1
    sizes = np.bincount(lbl, minlength=K).astype(np.int32)

    m = (lm != int(UNLAB)) & (lm >= 0) & (lm < int(n_classes))
    if not np.any(m):
        return np.full(K, -1, np.int32), np.zeros(K, np.float32)

    sid = lbl[m]
    cid = lm[m]
    C = int(n_classes)

    comb = sid * C + cid
    counts = np.bincount(comb, minlength=K * C).reshape(K, C)
    maj_c = counts.argmax(axis=1).astype(np.int32)
    maj_n = counts.max(axis=1).astype(np.int32)

    frac = (maj_n / np.maximum(1, sizes)).astype(np.float32)
    ok = (maj_n > 0) & (frac >= float(min_frac))
    y = np.where(ok, maj_c, -1).astype(np.int32)
    return y, frac

def predict_superpixels_for_image(
    rgb: np.ndarray,
    slic_params: Tuple[int, float, float, bool],
    *,
    key_prefix: str,
    model_is_supcon: bool,
    model_head: Optional["ContrastiveHead"],
    Z_lab: np.ndarray,
    X_lab: np.ndarray,
    y_lab: np.ndarray,
    k_neighbors: int,
    balance_knn: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_segments, compactness, sigma, enforce_conn = slic_params

    slic_labels = compute_slic_labels_cached(
        rgb,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        enforce_connectivity=bool(enforce_conn),
        key=key_prefix + ":slic",
    )
    feats = compute_superpixel_features_cached(rgb, slic_labels, key=key_prefix + ":feats")
    feats_z = zscore(feats)

    if model_is_supcon and TORCH_OK and (model_head is not None):
        Z_all = embed_all_feats(model_head, feats_z)
        Z_ref = Z_lab
    else:
        Z_all = feats_z
        Z_ref = X_lab

    k = int(min(max(1, int(k_neighbors)), len(y_lab)))
    pred_classes = knn_predict_numpy_balanced(Z_ref, y_lab, Z_all, k=k, balance=bool(balance_knn))
    return slic_labels, feats_z, pred_classes

def boundaries_mask(slic_labels: np.ndarray) -> np.ndarray:
    return find_boundaries(slic_labels, connectivity=1, mode="outer").astype(np.uint8)
