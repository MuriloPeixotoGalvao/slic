# slic_app/undo.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import streamlit as st

from .persistence import save_label_outputs, update_meta_index
from .io_images import imread_rgb
from .slic_ops import compute_slic_labels_cached
from .vis import overlay_from_labelmap_and_pred
from .constants import UNLAB


# ---------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------
def _get_max_undo() -> int:
    """Máximo de ações guardadas POR IMAGEM (uid)."""
    try:
        v = int(st.session_state.get("max_undo", 500))
        return max(1, v)
    except Exception:
        return 500


def _bump_click_nonce_if_exists(uid: str) -> None:
    """
    Evita o efeito 'desfaz e volta' quando algum clique antigo/canvas é
    reprocessado no rerun. Só funciona se seu pages usa click_nonce_{uid}.
    """
    k = f"click_nonce_{uid}"
    if k in st.session_state:
        try:
            st.session_state[k] = int(st.session_state.get(k, 0)) + 1
        except Exception:
            st.session_state[k] = 0

    # também costuma ajudar:
    if "last_processed_click" in st.session_state:
        st.session_state["last_processed_click"] = None


def _truncate_stack(stack: list, max_len: int) -> None:
    """Mantém só as últimas max_len ações (remove as mais antigas)."""
    extra = len(stack) - max_len
    if extra > 0:
        del stack[:extra]


# ---------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------
def push_undo_action(
    uid: str,
    img_path: Path,
    mask: np.ndarray,
    labelmap_u16: np.ndarray,
    new_label: np.uint16,
) -> None:
    """
    Empilha uma ação no histórico de UNDO da imagem (uid).

    Guarda:
    - índices (flat) onde houve alteração
    - valores anteriores nesses índices
    - novo rótulo aplicado
    """
    if mask is None:
        return

    # garante bool 2D
    if mask.dtype != np.bool_:
        mask = mask.astype(bool, copy=False)

    if mask.ndim != 2:
        return

    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return

    undo = st.session_state.undo.setdefault(uid, [])
    redo = st.session_state.redo.setdefault(uid, [])

    # nova ação invalida redo
    redo.clear()

    # valores anteriores
    prev = labelmap_u16.ravel()[idx].copy()

    undo.append(
        {
            "img_path": str(Path(img_path).resolve()),
            "idx": idx.astype(np.int32, copy=False),
            "prev": prev.astype(np.uint16, copy=False),
            "new": np.uint16(new_label),
        }
    )

    # limita histórico
    _truncate_stack(undo, _get_max_undo())


def push_undo_sids(
    uid: str,
    img_path: Path,
    slic: np.ndarray,
    labelmap_u16: np.ndarray,
    sids: np.ndarray,
    new_label: np.uint16,
) -> None:
    """
    Compat: alguns trechos do app chamam push_undo_sids.
    Converte sids -> máscara e delega para push_undo_action.
    """
    if sids is None:
        return
    sids = np.asarray(sids)
    if sids.size == 0:
        return

    mask = np.isin(slic, sids)
    push_undo_action(uid, img_path, mask, labelmap_u16, new_label)


def undo_last_click(
    uid: str,
    base_dir: Path,
    slic_params,
    overlay_alpha: float,
    show_boundaries: bool,
    boundary_thick_px: int,
) -> bool:
    """
    Desfaz a última ação do uid (imagem atual).
    Retorna True se desfez algo.
    """
    stack = st.session_state.undo.get(uid, [])
    if not stack:
        return False

    act: Dict[str, Any] = stack.pop()
    st.session_state.redo.setdefault(uid, []).append(act)

    img_path = Path(act["img_path"])
    rgb = imread_rgb(img_path)

    # labelmap precisa estar no cache
    if uid not in st.session_state.labelmaps_cache:
        # fallback: cria vazio (evita crash). Ideal é o pages garantir cache.
        H, W = rgb.shape[:2]
        st.session_state.labelmaps_cache[uid] = np.full((H, W), UNLAB, dtype=np.uint16)

    labelmap = st.session_state.labelmaps_cache[uid]

    idx = act["idx"]
    prev = act["prev"]
    # aplica valores anteriores
    labelmap.ravel()[idx] = prev

    # refaz overlay + salva
    key = f"{img_path.name}:{int(slic_params[0])}:{float(slic_params[1])}:{float(slic_params[2])}:{bool(slic_params[3])}"
    slic = compute_slic_labels_cached(
        rgb,
        n_segments=int(slic_params[0]),
        compactness=float(slic_params[1]),
        sigma=float(slic_params[2]),
        enforce_connectivity=bool(slic_params[3]),
        key=key,
    )

    ov = overlay_from_labelmap_and_pred(
        rgb,
        slic,
        labelmap,
        pred_classes=None,
        n_classes=len(st.session_state.classes),
        alpha=float(overlay_alpha),
        show_boundaries=bool(show_boundaries),
        boundary_thick_px=int(boundary_thick_px),
        apply_preds_only_unlabeled=True,
    )
    save_label_outputs(base_dir, img_path, labelmap, ov)
    update_meta_index(base_dir, img_path)

    _bump_click_nonce_if_exists(uid)
    return True


def redo_last_click(
    uid: str,
    base_dir: Path,
    slic_params,
    overlay_alpha: float,
    show_boundaries: bool,
    boundary_thick_px: int,
) -> bool:
    """
    Refaz a última ação desfeita (redo).
    Retorna True se refez algo.
    """
    stack = st.session_state.redo.get(uid, [])
    if not stack:
        return False

    act: Dict[str, Any] = stack.pop()
    st.session_state.undo.setdefault(uid, []).append(act)

    img_path = Path(act["img_path"])
    rgb = imread_rgb(img_path)

    # labelmap precisa estar no cache
    if uid not in st.session_state.labelmaps_cache:
        H, W = rgb.shape[:2]
        st.session_state.labelmaps_cache[uid] = np.full((H, W), UNLAB, dtype=np.uint16)

    labelmap = st.session_state.labelmaps_cache[uid]

    idx = act["idx"]
    newv = np.uint16(act["new"])
    labelmap.ravel()[idx] = newv

    # refaz overlay + salva
    key = f"{img_path.name}:{int(slic_params[0])}:{float(slic_params[1])}:{float(slic_params[2])}:{bool(slic_params[3])}"
    slic = compute_slic_labels_cached(
        rgb,
        n_segments=int(slic_params[0]),
        compactness=float(slic_params[1]),
        sigma=float(slic_params[2]),
        enforce_connectivity=bool(slic_params[3]),
        key=key,
    )

    ov = overlay_from_labelmap_and_pred(
        rgb,
        slic,
        labelmap,
        pred_classes=None,
        n_classes=len(st.session_state.classes),
        alpha=float(overlay_alpha),
        show_boundaries=bool(show_boundaries),
        boundary_thick_px=int(boundary_thick_px),
        apply_preds_only_unlabeled=True,
    )
    save_label_outputs(base_dir, img_path, labelmap, ov)
    update_meta_index(base_dir, img_path)

    _bump_click_nonce_if_exists(uid)
    return True
