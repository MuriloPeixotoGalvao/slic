# slic_app/pages_label_train.py
from __future__ import annotations

from typing import Tuple, List, Dict, Any
from pathlib import Path
import re

import numpy as np
import streamlit as st
import cv2
from PIL import Image

from .constants import EXTS, UNLAB
from .io_images import imread_rgb
from .persistence import (
    image_uid,
    ensure_out_dirs,
    load_or_create_labelmap,
    save_label_outputs,
    update_meta_index,
    save_classes,
)
from .slic_ops import (
    compute_slic_labels_cached,
    compute_superpixel_features_cached,
    zscore,
    build_adjacency_csr_cached,
    expand_region_by_similarity,
    majority_label_per_superpixel,
)
from .vis import overlay_from_labelmap_and_pred, resize_for_view, curtain_merge
from .undo import push_undo_action, undo_last_click, redo_last_click
from .knn import knn_predict_numpy_balanced
from .supcon import TORCH_OK, train_supcon_on_feats, embed_all_feats
from .exporter import build_labeled_dataset_zip


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _img_key(img_name: str, slic_params: Tuple[int, float, float, bool]) -> str:
    n_segments, compactness, sigma, enforce_conn = slic_params
    return f"{img_name}:{n_segments}:{compactness}:{sigma}:{enforce_conn}"


def _bump_click_nonce(uid: str) -> None:
    k = f"click_nonce_{uid}"
    st.session_state[k] = int(st.session_state.get(k, 0)) + 1


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "classe"


def _make_palette_rgb(n: int) -> np.ndarray:
    """Paleta RGB determinística (n cores), retorna (n,3) uint8 em RGB."""
    if n <= 0:
        return np.zeros((0, 3), np.uint8)
    hsv = np.zeros((n, 1, 3), np.uint8)
    for i in range(n):
        h = int(round(179 * i / max(1, n)))  # 0..179 (OpenCV HSV)
        hsv[i, 0] = (h, 200, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(n, 3)
    return rgb.astype(np.uint8, copy=False)


def _get_feats_and_adj(img_key: str, rgb: np.ndarray, slic: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats_cache = st.session_state.setdefault("feats_cache", {})
    adj_cache = st.session_state.setdefault("adj_cache", {})

    if img_key not in feats_cache:
        feats = compute_superpixel_features_cached(rgb, slic, key=img_key + ":feats")
        feats_cache[img_key] = zscore(feats)

    if img_key not in adj_cache:
        adj_cache[img_key] = build_adjacency_csr_cached(slic, key=img_key + ":adj")

    feats_z = feats_cache[img_key]
    indptr, indices = adj_cache[img_key]
    return feats_z, indptr, indices


def _save_sidecars(
    *,
    base_dir: Path,
    img_path: Path,
    labelmap: np.ndarray,
    overlay_rgb: np.ndarray,
    classes: List[str],
    clean_legacy_in_image_dir: bool = True,
) -> None:
    """
    Cria/atualiza arquivos em base_dir/masks:
    - *_overlay.png
    - *_mask_color.png
    - *_mask_id.png (1 canal, UNLAB=255 quando <=255 classes)
    - *_bin_<classe>.png (uma por classe)

    NÃO altera o .npy (isso continua via save_label_outputs).
    """
    out_dir = Path(base_dir) / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    # limpa versões antigas para evitar lixo quando renomeia classes
    for p in out_dir.glob(f"{stem}__bin_*.png"):
        try:
            p.unlink()
        except Exception:
            pass

    # remove sidecars antigos na pasta da imagem (legacy)
    if clean_legacy_in_image_dir:
        legacy_dir = img_path.parent
        legacy_patterns = [
            f"{stem}__overlay.png",
            f"{stem}__mask_color.png",
            f"{stem}__mask_id.png",
            f"{stem}__mask_id_u16.png",
            f"{stem}__bin_*.png",
        ]
        for pat in legacy_patterns:
            for p in legacy_dir.glob(pat):
                try:
                    p.unlink()
                except Exception:
                    pass

    try:
        n = int(len(classes))
        H, W = labelmap.shape[:2]

        # 1) overlay
        ov_path = out_dir / f"{stem}__overlay.png"
        cv2.imwrite(str(ov_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        # 2) máscara colorida (UNLAB preto)
        pal = _make_palette_rgb(n)  # (n,3) RGB
        mask_color = np.zeros((H, W, 3), np.uint8)
        for cid in range(n):
            mask_color[labelmap == np.uint16(cid)] = pal[cid]
        color_path = out_dir / f"{stem}__mask_color.png"
        cv2.imwrite(str(color_path), cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR))

        # 3) máscara id (1 canal)
        if n <= 255:
            mask_id_u8 = np.where(
                labelmap == np.uint16(UNLAB),
                np.uint16(255),
                labelmap.astype(np.uint16, copy=False),
            ).astype(np.uint8, copy=False)
            id_path = out_dir / f"{stem}__mask_id.png"
            cv2.imwrite(str(id_path), mask_id_u8)
        else:
            mask_id_u16 = labelmap.astype(np.uint16, copy=False)
            id_path = out_dir / f"{stem}__mask_id_u16.png"
            cv2.imwrite(str(id_path), mask_id_u16)

        # 4) binárias por classe
        for cid, name in enumerate(classes):
            safe = _slugify(name)
            bin_path = out_dir / f"{stem}__bin_{cid:02d}_{safe}.png"
            bin_u8 = (labelmap == np.uint16(cid)).astype(np.uint8) * 255
            cv2.imwrite(str(bin_path), bin_u8)

    except Exception:
        # não quebra o app por falhas de IO
        pass


def _save_everything(
    *,
    base_dir: Path,
    img_path: Path,
    rgb: np.ndarray,
    slic: np.ndarray,
    labelmap: np.ndarray,
    cfg,
) -> None:
    """
    1) mantém o comportamento atual: salva .npy + overlay/meta (na estrutura do app)
    2) adicional: salva sidecars ao lado da imagem original (overlay/color/id/bin)
    """
    overlay_rgb = overlay_from_labelmap_and_pred(
        rgb,
        slic,
        labelmap,
        pred_classes=None,
        n_classes=len(st.session_state.classes),
        alpha=float(cfg.overlay_alpha),
        show_boundaries=bool(cfg.show_boundaries),
        boundary_thick_px=int(cfg.boundary_thick_px),
        apply_preds_only_unlabeled=True,
    )

    # mantém o que já existia (inclui .npy)
    save_label_outputs(base_dir, img_path, labelmap, overlay_rgb)
    update_meta_index(base_dir, img_path)

    # novo: sidecars em base_dir/masks
    _save_sidecars(
        base_dir=base_dir,
        img_path=img_path,
        labelmap=labelmap,
        overlay_rgb=overlay_rgb,
        classes=list(st.session_state.classes),
        clean_legacy_in_image_dir=True,
    )

    # marca que a renderizaÃ§Ã£o precisa ser refeita
    try:
        uid = image_uid(img_path)
        k = f"render_nonce_{uid}"
        st.session_state[k] = int(st.session_state.get(k, 0)) + 1
    except Exception:
        pass


def _render_classes_panel(base_dir: Path, *, key_prefix: str, title: str = "### 🎨 Classes") -> None:
    st.markdown(title)
    id2name = {i: nm for i, nm in enumerate(st.session_state.classes)}

    c1, c2 = st.columns([2.2, 1])
    new_cls_name = c1.text_input("Nova classe", "", placeholder="ex: folha, solo, tronco…", key=f"{key_prefix}_new_cls")
    if c2.button("➕", key=f"{key_prefix}_add_cls", use_container_width=True):
        name = new_cls_name.strip() if new_cls_name.strip() else f"classe_{len(st.session_state.classes)}"
        st.session_state.classes.append(name)
        st.session_state.active_class = len(st.session_state.classes) - 1
        save_classes(base_dir, st.session_state.classes)
        st.rerun()

    cls_options = ["🧽 Borracha"] + [f"{i}: {nm}" for i, nm in enumerate(st.session_state.classes)]
    cur_idx = 0 if st.session_state.active_class == -1 else (st.session_state.active_class + 1)
    picked = st.radio(
        "Classe ativa",
        cls_options,
        index=int(np.clip(cur_idx, 0, len(cls_options) - 1)),
        key=f"{key_prefix}_radio_cls",
    )

    if picked == "🧽 Borracha":
        st.session_state.active_class = -1
    else:
        st.session_state.active_class = int(picked.split(":")[0].strip())

    st.caption(
        "Ativa: "
        + (
            "**Borracha**"
            if st.session_state.active_class == -1
            else f"**{st.session_state.active_class} — {id2name.get(st.session_state.active_class)}**"
        )
    )

    ac = st.session_state.active_class
    if 0 <= ac < len(st.session_state.classes):
        r1, r2 = st.columns([3, 1])
        new_name = r1.text_input("Renomear ativa", value=st.session_state.classes[ac], key=f"{key_prefix}_rename_{ac}")
        if r2.button("💾", key=f"{key_prefix}_save_rename_{ac}", use_container_width=True):
            nnm = (new_name or "").strip()
            if nnm:
                st.session_state.classes[ac] = nnm
                save_classes(base_dir, st.session_state.classes)
                st.rerun()

    if st.button(
        "🗑️ Remover classe ativa",
        key=f"{key_prefix}_rm_cls",
        use_container_width=True,
        disabled=(len(st.session_state.classes) <= 1 or st.session_state.active_class == -1),
    ):
        idx = st.session_state.active_class
        del st.session_state.classes[idx]
        st.session_state.active_class = min(idx, len(st.session_state.classes) - 1)
        save_classes(base_dir, st.session_state.classes)
        st.rerun()


def _maybe_embed_feats(uid: str, img_key: str, feats_z: np.ndarray, cfg) -> np.ndarray:
    feats_space = feats_z
    if (
        getattr(cfg, "use_model_embedding_for_click", False)
        and st.session_state.get("MODEL_IS_SUPCON", False)
        and (st.session_state.get("MODEL_HEAD") is not None)
        and TORCH_OK
    ):
        emb_k = (uid, img_key)
        if emb_k not in st.session_state.emb_cache:
            try:
                st.session_state.emb_cache[emb_k] = embed_all_feats(st.session_state["MODEL_HEAD"], feats_z)
            except Exception:
                st.session_state.emb_cache[emb_k] = feats_z
        feats_space = st.session_state.emb_cache[emb_k]
    return feats_space


def _predict_superpixel_labels(feats_z: np.ndarray, cfg) -> Tuple[np.ndarray | None, str | None]:
    """
    Prediz rótulo por superpixel usando o modelo em memória (kNN ou SupCon+kNN).
    Retorna (y_pred, erro). Se erro não for None, y_pred vem None.
    """
    y_lab = st.session_state.get("MODEL_Y_LAB", None)
    X_lab = st.session_state.get("MODEL_X_LAB", None)
    Z_lab = st.session_state.get("MODEL_Z_LAB", None)
    is_supcon = bool(st.session_state.get("MODEL_IS_SUPCON", False))
    head = st.session_state.get("MODEL_HEAD", None)

    if y_lab is None or X_lab is None:
        return None, "Treine o modelo primeiro."
    if len(y_lab) == 0:
        return None, "Modelo sem amostras válidas."

    use_supcon = bool(is_supcon and TORCH_OK and (head is not None))
    query_space = feats_z
    ref_space = X_lab

    if use_supcon:
        try:
            query_space = embed_all_feats(head, feats_z)
            ref_space = Z_lab if (Z_lab is not None) else embed_all_feats(head, X_lab)
        except Exception:
            # Fallback para espaço original caso embedding falhe.
            query_space = feats_z
            ref_space = X_lab

    k = int(min(max(1, int(cfg.k_neighbors)), len(y_lab)))
    y_pred = knn_predict_numpy_balanced(
        ref_space,
        y_lab,
        query_space,
        k=k,
        balance=bool(cfg.balance_knn),
    )
    return y_pred.astype(np.int32, copy=False), None


def _apply_sids_with_undo(
    *,
    uid: str,
    img_path: Path,
    slic: np.ndarray,
    labelmap: np.ndarray,
    sids: np.ndarray,
    only_unlabeled: bool,
) -> int:
    """
    Aplica o rótulo atual aos superpixels 'sids'.
    - only_unlabeled=True: pinta só onde labelmap == UNLAB (não sobrescreve).
    - only_unlabeled=False: sobrescreve (edição local).
    Borracha sempre apaga (independe de only_unlabeled).
    Retorna quantos pixels foram alterados.
    """
    if sids is None or sids.size == 0:
        return 0

    spx_mask = np.isin(slic, sids)
    new_label = np.uint16(UNLAB) if (st.session_state.active_class == -1) else np.uint16(int(st.session_state.active_class))

    if new_label != np.uint16(UNLAB) and only_unlabeled:
        apply_mask = spx_mask & (labelmap == np.uint16(UNLAB))
    else:
        apply_mask = spx_mask

    if not apply_mask.any():
        return 0

    push_undo_action(uid, img_path, apply_mask, labelmap, new_label)
    labelmap[apply_mask] = new_label
    return int(apply_mask.sum())


def _apply_mask_to_superpixels(
    *,
    mask_full: np.ndarray,
    slic: np.ndarray,
    labelmap: np.ndarray,
    feats_z: np.ndarray,
    indptr,
    indices,
    uid: str,
    img_key: str,
    img_path: Path,
    cfg,
    protect_expand: bool,
) -> np.ndarray:
    """
    Converte uma máscara (pixel) em superpixels seed e aplica:
    - SEED (local): sempre pode sobrescrever (edição local).
    - EXPANSÃO (global/contíguo): respeita 'protect_expand' (se True, só preenche UNLAB).
    Retorna os sids finais (seed + expansão).
    """
    seed_sids = np.unique(slic[mask_full].astype(np.int32))
    if seed_sids.size == 0:
        return np.array([], np.int32)

    # 1) sempre aplica a parte "local" (seed) permitindo sobrescrever
    _apply_sids_with_undo(
        uid=uid,
        img_path=img_path,
        slic=slic,
        labelmap=labelmap,
        sids=seed_sids,
        only_unlabeled=False,  # edição local sempre sobrescreve
    )

    # 2) expansão (se habilitada): só a parte expandida respeita o lock
    if getattr(cfg, "expand_on_click", False):
        feats_space = _maybe_embed_feats(uid, img_key, feats_z, cfg)
        mode = "contiguo" if getattr(cfg, "expand_mode", "contíguo") == "contíguo" else "global"

        sset: set[int] = set(seed_sids.tolist())
        for sid0 in seed_sids.tolist():
            sids_i = expand_region_by_similarity(
                start_sid=int(sid0),
                feats_space=feats_space,
                indptr=indptr,
                indices=indices,
                dist_thr=float(getattr(cfg, "dist_thr", 0.6)),
                max_nodes=int(getattr(cfg, "max_region", 400)),
                mode=mode,
            )
            for s in sids_i.tolist():
                sset.add(int(s))

        all_sids = np.array(sorted(sset), np.int32)

        # somente os sids além do seed
        extra = np.setdiff1d(all_sids, seed_sids, assume_unique=False)
        if extra.size:
            _apply_sids_with_undo(
                uid=uid,
                img_path=img_path,
                slic=slic,
                labelmap=labelmap,
                sids=extra,
                only_unlabeled=bool(protect_expand),  # lock só vale pra expansão
            )

        return all_sids

    return seed_sids


def _sync_jump_key(jump_key: str, sync_key: str, idx0: int) -> None:
    if st.session_state.get(sync_key, None) != idx0:
        st.session_state[jump_key] = int(idx0 + 1)
        st.session_state[sync_key] = int(idx0)


def _jump_cb(index_key: str, jump_key: str, max_idx: int) -> None:
    v = int(st.session_state.get(jump_key, 1))
    st.session_state[index_key] = int(np.clip(v - 1, 0, max_idx))
    st.session_state.last_processed_click = None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def render_label_train(cfg):
    deps = cfg.deps
    if not deps["HAS_IMG_COORD"]:
        st.warning("Para clique interativo: `pip install streamlit-image-coordinates`")
    if not deps["HAS_CANVAS"]:
        st.info("Para polígono/freedraw: `pip install streamlit-drawable-canvas`")

    base_dir: Path = cfg.base_dir
    slic_params = cfg.slic_params

    glob_dir = (base_dir / "previews") if (base_dir / "previews").exists() else base_dir
    img_paths = sorted([p for p in glob_dir.glob("*") if p.suffix.lower() in EXTS])
    if not img_paths:
        st.info("Coloque imagens em previews/ ou na pasta escolhida.")
        st.stop()

    max_idx = len(img_paths) - 1
    st.session_state.train_img_idx = int(np.clip(st.session_state.train_img_idx, 0, max_idx))

    train_img_path = img_paths[st.session_state.train_img_idx]
    train_rgb = imread_rgb(train_img_path)
    Htr, Wtr = train_rgb.shape[:2]

    uid_tr = image_uid(train_img_path)

    if uid_tr not in st.session_state.labelmaps_cache:
        st.session_state.labelmaps_cache[uid_tr] = load_or_create_labelmap(base_dir, train_img_path, (Htr, Wtr))
    train_labelmap = st.session_state.labelmaps_cache[uid_tr]

    train_key = _img_key(train_img_path.name, slic_params)

    train_slic = compute_slic_labels_cached(
        train_rgb,
        n_segments=int(slic_params[0]),
        compactness=float(slic_params[1]),
        sigma=float(slic_params[2]),
        enforce_connectivity=bool(slic_params[3]),
        key=train_key,
    )
    # feats/adj são calculados sob demanda (evita custo ao navegar entre imagens)

    # =========================================================================
    # ① Rotular — Treino
    # =========================================================================
    st.markdown("<div class='step-title'>① Rotular — Imagem de treinamento</div>", unsafe_allow_html=True)
    col_img, col_cls = st.columns([3.4, 1.3], gap="large")

    with col_cls:
        _render_classes_panel(base_dir, key_prefix=f"train_{uid_tr}", title="### 🎨 Classes (treino)")
        
        st.caption(f"Undo: **{len(st.session_state.undo.get(uid_tr, []))}**  |  Redo: **{len(st.session_state.redo.get(uid_tr, []))}**")
        st.caption(f"Histórico máx.: **{int(st.session_state.get('max_undo', 500))}** (por imagem)")

        # Preencher vazios: pixel-a-pixel (não depende de superpixel)
        st.markdown("---")
        st.markdown("### 🪣 Preencher vazios (UNLAB)")

        fill_opt = st.selectbox(
            "Classe para preencher",
            options=[f"{i}: {nm}" for i, nm in enumerate(st.session_state.classes)],
            key=f"fill_cls_{uid_tr}",
        )

        if st.button("🪣 Preencher", key=f"fill_btn_{uid_tr}", use_container_width=True):
            cls_id = int(fill_opt.split(":")[0].strip())
            new_label = np.uint16(cls_id)

            mask = (train_labelmap == np.uint16(UNLAB))
            if not mask.any():
                st.toast("Não há pixels vazios (UNLAB) nesta imagem.", icon="ℹ️")
            else:
                push_undo_action(uid_tr, train_img_path, mask, train_labelmap, new_label)
                train_labelmap[mask] = new_label

                _save_everything(
                    base_dir=base_dir,
                    img_path=train_img_path,
                    rgb=train_rgb,
                    slic=train_slic,
                    labelmap=train_labelmap,
                    cfg=cfg,
                )

                st.session_state.last_processed_click = None
                _bump_click_nonce(uid_tr)
                st.rerun()

    with col_img:
        st.markdown(
            f"<div class='img-title'>🖼️ {train_img_path.name} ({st.session_state.train_img_idx+1}/{len(img_paths)})</div>",
            unsafe_allow_html=True,
        )

        cur_train = st.slider("", 0, 100, 0, key=f"curtain_train_{uid_tr}")

        # cache base/overlay (rápido para slider em tempo real)
        render_nonce = int(st.session_state.get(f"render_nonce_{uid_tr}", 0))
        base_meta = (
            uid_tr,
            int(cfg.img_width),
            float(cfg.overlay_alpha),
            bool(cfg.show_boundaries),
            int(cfg.boundary_thick_px),
            tuple(cfg.slic_params),
            render_nonce,
        )
        base_meta_key = f"disp_base_meta_{uid_tr}"
        base_img_key = f"disp_base_img_{uid_tr}"
        base_ov_key = f"disp_ov_img_{uid_tr}"
        base_sx1_key = f"disp_sx1_{uid_tr}"
        base_sy1_key = f"disp_sy1_{uid_tr}"

        if st.session_state.get(base_meta_key) != base_meta:
            train_overlay = overlay_from_labelmap_and_pred(
                train_rgb,
                train_slic,
                train_labelmap,
                pred_classes=None,
                n_classes=len(st.session_state.classes),
                alpha=float(cfg.overlay_alpha),
                show_boundaries=bool(cfg.show_boundaries),
                boundary_thick_px=int(cfg.boundary_thick_px),
                apply_preds_only_unlabeled=True,
            )
            disp_base, sx1, sy1 = resize_for_view(train_rgb, int(cfg.img_width))
            disp_overlay = cv2.resize(train_overlay, (disp_base.shape[1], disp_base.shape[0]), interpolation=cv2.INTER_NEAREST)

            st.session_state[base_meta_key] = base_meta
            st.session_state[base_img_key] = disp_base
            st.session_state[base_ov_key] = disp_overlay
            st.session_state[base_sx1_key] = float(sx1)
            st.session_state[base_sy1_key] = float(sy1)

        disp_base = st.session_state.get(base_img_key)
        disp_overlay = st.session_state.get(base_ov_key)
        sx1 = st.session_state.get(base_sx1_key)
        sy1 = st.session_state.get(base_sy1_key)
        if disp_base is None or disp_overlay is None or sx1 is None or sy1 is None:
            # fallback (deve ser raro)
            disp_base, sx1, sy1 = resize_for_view(train_rgb, int(cfg.img_width))
            disp_overlay = cv2.resize(
                overlay_from_labelmap_and_pred(
                    train_rgb,
                    train_slic,
                    train_labelmap,
                    pred_classes=None,
                    n_classes=len(st.session_state.classes),
                    alpha=float(cfg.overlay_alpha),
                    show_boundaries=bool(cfg.show_boundaries),
                    boundary_thick_px=int(cfg.boundary_thick_px),
                    apply_preds_only_unlabeled=True,
                ),
                (disp_base.shape[1], disp_base.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        disp_show = curtain_merge(
            base_rgb=disp_base,
            overlay_rgb=disp_overlay,
            pos01=cur_train / 100.0,
            show_line=True,
            line_thick=2,
            left_is="base",
        )
        disp_show = disp_show.astype(np.uint8, copy=False)
        disp_show_img = Image.fromarray(disp_show)

        tool_key = f"tool_{uid_tr}"

        tool_col, pred_col = st.columns([4.9, 1.7], gap="small")
        with tool_col:
            tool = st.radio(
                "Ferramenta",
                ["Clique", "Polígono", "Freedraw"],
                horizontal=True,
                index=0,
                key=tool_key,
            )
        with pred_col:
            st.markdown("<div style='height: 1.0rem;'></div>", unsafe_allow_html=True)
            if st.button("Aplicar predição (UNLAB)", key=f"apply_pred_unlab_{uid_tr}", use_container_width=True):
                unl_mask = (train_labelmap == np.uint16(UNLAB))
                if not unl_mask.any():
                    st.toast("Não há pixels UNLAB nesta imagem.", icon="ℹ️")
                else:
                    feats_z, _, _ = _get_feats_and_adj(train_key, train_rgb, train_slic)
                    y_pred_spx, err = _predict_superpixel_labels(feats_z, cfg)

                    if err is not None or y_pred_spx is None:
                        st.warning(err or "Falha ao predizer.")
                    else:
                        pred_pix = y_pred_spx[train_slic]
                        apply_mask = unl_mask & (pred_pix >= 0)

                        if not apply_mask.any():
                            st.toast("Nenhuma região UNLAB elegível para aplicar predição.", icon="ℹ️")
                        else:
                            changed = 0
                            classes_n = len(st.session_state.classes)
                            cls_ids = np.unique(pred_pix[apply_mask].astype(np.int32))
                            for cid in cls_ids.tolist():
                                if cid < 0 or cid >= classes_n:
                                    continue
                                m = apply_mask & (pred_pix == cid)
                                if not m.any():
                                    continue
                                push_undo_action(uid_tr, train_img_path, m, train_labelmap, np.uint16(cid))
                                train_labelmap[m] = np.uint16(cid)
                                changed += int(m.sum())

                            if changed <= 0:
                                st.toast("Nenhum pixel atualizado.", icon="ℹ️")
                            else:
                                _save_everything(
                                    base_dir=base_dir,
                                    img_path=train_img_path,
                                    rgb=train_rgb,
                                    slic=train_slic,
                                    labelmap=train_labelmap,
                                    cfg=cfg,
                                )
                                st.session_state.last_processed_click = None
                                _bump_click_nonce(uid_tr)
                                st.toast(f"Predição aplicada em {changed} pixels UNLAB.", icon="✅")
                                st.rerun()

        # (slider e disp_show já definidos acima)

        # canvas estável: só reseta quando pedir (aplicar/limpar)
        draw_nonce_key = f"draw_nonce_{uid_tr}"
        st.session_state.setdefault(draw_nonce_key, 0)

        protect_expand = bool(st.session_state.get("protect_expand", True))

        # ============================
        # Clique
        # ============================
        if tool == "Clique":
            click = None
            if deps["HAS_CANVAS"]:
                from streamlit_drawable_canvas import st_canvas

                canvas_key = f"draw_canvas_{uid_tr}_{st.session_state[draw_nonce_key]}"
                canvas_res = st_canvas(
                    background_image=disp_show_img,
                    height=int(disp_show.shape[0]),
                    width=int(disp_show.shape[1]),
                    drawing_mode="point",
                    stroke_width=4,
                    update_streamlit=True,
                    key=canvas_key,
                    display_toolbar=False,
                )

                if canvas_res is not None and canvas_res.json_data and canvas_res.json_data.get("objects"):
                    obj = canvas_res.json_data["objects"][-1]
                    if isinstance(obj, dict) and "left" in obj and "top" in obj:
                        scale_x = float(obj.get("scaleX", 1.0))
                        scale_y = float(obj.get("scaleY", 1.0))
                        r = obj.get("radius", obj.get("rx", obj.get("ry", 0)))
                        try:
                            r = float(r)
                        except Exception:
                            r = 0.0
                        cx = float(obj["left"]) + r * scale_x
                        cy = float(obj["top"]) + r * scale_y
                        click = {"x": cx, "y": cy}
            elif deps["HAS_IMG_COORD"]:
                from streamlit_image_coordinates import streamlit_image_coordinates

                nonce_k = f"click_nonce_{uid_tr}"
                st.session_state.setdefault(nonce_k, 0)

                click = streamlit_image_coordinates(
                    disp_show_img,
                    key=f"click_train_{uid_tr}_{st.session_state[nonce_k]}",
                    width=disp_show.shape[1],
                )
            else:
                st.image(disp_show, width=disp_show.shape[1])

            if click and "x" in click and "y" in click:
                click_sig = (uid_tr, int(click["x"]), int(click["y"]))
                if st.session_state.last_processed_click != click_sig:
                    st.session_state.last_processed_click = click_sig

                    cx = int(round(click["x"] * sx1))
                    cy = int(round(click["y"] * sy1))
                    if 0 <= cx < Wtr and 0 <= cy < Htr:
                        sid0 = int(train_slic[cy, cx])

                        # seed local sempre sobrescreve
                        _apply_sids_with_undo(
                            uid=uid_tr,
                            img_path=train_img_path,
                            slic=train_slic,
                            labelmap=train_labelmap,
                            sids=np.array([sid0], np.int32),
                            only_unlabeled=False,
                        )

                        # expansão respeita o lock
                        if getattr(cfg, "expand_on_click", False):
                            train_feats_z, train_indptr, train_indices = _get_feats_and_adj(train_key, train_rgb, train_slic)
                            feats_space = _maybe_embed_feats(uid_tr, train_key, train_feats_z, cfg)
                            sids_all = expand_region_by_similarity(
                                start_sid=sid0,
                                feats_space=feats_space,
                                indptr=train_indptr,
                                indices=train_indices,
                                dist_thr=float(cfg.dist_thr),
                                max_nodes=int(cfg.max_region),
                                mode="contiguo" if cfg.expand_mode == "contíguo" else "global",
                            ).astype(np.int32, copy=False)

                            extra = np.setdiff1d(sids_all, np.array([sid0], np.int32), assume_unique=False)
                            if extra.size:
                                _apply_sids_with_undo(
                                    uid=uid_tr,
                                    img_path=train_img_path,
                                    slic=train_slic,
                                    labelmap=train_labelmap,
                                    sids=extra,
                                    only_unlabeled=protect_expand,
                                )

                        _save_everything(
                            base_dir=base_dir,
                            img_path=train_img_path,
                            rgb=train_rgb,
                            slic=train_slic,
                            labelmap=train_labelmap,
                            cfg=cfg,
                        )

                        _bump_click_nonce(uid_tr)
                        st.session_state[draw_nonce_key] += 1
                        st.rerun()

        # ============================
        # Canvas: Polígono / Freedraw
        # ============================
        else:
            if not deps["HAS_CANVAS"]:
                st.info("Instale `streamlit-drawable-canvas` para usar Polígono/Freedraw")
                st.image(disp_show, width=disp_show.shape[1])
            else:
                from streamlit_drawable_canvas import st_canvas

                if tool == "Polígono":
                    draw_mode = "polygon"
                    fill = "rgba(0, 0, 0, 0.25)"
                else:
                    draw_mode = "freedraw"
                    fill = "rgba(0, 0, 0, 0.0)"

                canvas_key = f"draw_canvas_{uid_tr}_{st.session_state[draw_nonce_key]}"

                canvas_res = st_canvas(
                    background_image=disp_show_img,
                    height=int(disp_show.shape[0]),
                    width=int(disp_show.shape[1]),
                    drawing_mode=draw_mode,
                    fill_color=fill,
                    stroke_color=str(getattr(cfg, "stroke_color", "#FF0000")),
                    stroke_width=2,
                    update_streamlit=(tool == "Freedraw"),
                    key=canvas_key,
                    display_toolbar=True,
                )

                cA, cB = st.columns(2)
                if cA.button("✅ Aplicar", key=f"apply_draw_{uid_tr}_{tool}", use_container_width=True):
                    if canvas_res is None or canvas_res.image_data is None:
                        st.toast("Desenhe primeiro.", icon="✏️")
                    else:
                        img_rgba = canvas_res.image_data
                        alpha = img_rgba[:, :, 3]
                        if alpha.dtype != np.uint8:
                            amax = float(alpha.max()) if alpha.size else 0.0
                            alpha = (alpha * 255.0).astype(np.uint8) if amax <= 1.0 else alpha.astype(np.uint8)

                        mask_disp = (alpha > 0)
                        if mask_disp.any():
                            mask_u8 = (mask_disp.astype(np.uint8) * 255)
                            mask_full = cv2.resize(mask_u8, (Wtr, Htr), interpolation=cv2.INTER_NEAREST) > 0

                            train_feats_z, train_indptr, train_indices = _get_feats_and_adj(train_key, train_rgb, train_slic)
                            _apply_mask_to_superpixels(
                                mask_full=mask_full,
                                slic=train_slic,
                                labelmap=train_labelmap,
                                feats_z=train_feats_z,
                                indptr=train_indptr,
                                indices=train_indices,
                                uid=uid_tr,
                                img_key=train_key,
                                img_path=train_img_path,
                                cfg=cfg,
                                protect_expand=protect_expand,
                            )

                            _save_everything(
                                base_dir=base_dir,
                                img_path=train_img_path,
                                rgb=train_rgb,
                                slic=train_slic,
                                labelmap=train_labelmap,
                                cfg=cfg,
                            )

                            st.session_state[draw_nonce_key] += 1
                            st.session_state.last_processed_click = None
                            _bump_click_nonce(uid_tr)
                            st.rerun()
                        else:
                            st.toast("Nada desenhado.", icon="ℹ️")

                if cB.button("🧽 Limpar desenho", key=f"clear_draw_{uid_tr}_{tool}", use_container_width=True):
                    st.session_state[draw_nonce_key] += 1
                    st.rerun()

        # =========================================================
        # MENU: navegação + jump + undo/redo/clear
        # =========================================================
        jump_key = f"jump_train_{uid_tr}"
        sync_key = f"jump_train_sync_{uid_tr}"
        _sync_jump_key(jump_key, sync_key, int(st.session_state.train_img_idx))

        u = st.session_state.undo.get(uid_tr, [])
        r = st.session_state.redo.get(uid_tr, [])

        # callbacks (evita “pular errado”)
        def _nav_delta(delta: int):
            st.session_state.train_img_idx = int(np.clip(st.session_state.train_img_idx + int(delta), 0, max_idx))
            st.session_state.last_processed_click = None

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([0.8, 0.8, 0.8, 0.8, 1.2, 0.9, 0.9, 0.9], gap="small")

        c1.button("-10 ⏪", key="train_bk10", use_container_width=True, disabled=(st.session_state.train_img_idx <= 0), on_click=_nav_delta, args=(-10,))
        c2.button("-1 ◀" , key="train_prev", use_container_width=True, disabled=(st.session_state.train_img_idx <= 0), on_click=_nav_delta, args=(-1,))
        c3.button("▶ +1" , key="train_next", use_container_width=True, disabled=(st.session_state.train_img_idx >= max_idx), on_click=_nav_delta, args=(+1,))
        c4.button("⏩ +10", key="train_fw10", use_container_width=True, disabled=(st.session_state.train_img_idx >= max_idx), on_click=_nav_delta, args=(+10,))

        c5.number_input(
            "Ir",
            min_value=1,
            max_value=len(img_paths),
            key=jump_key,
            label_visibility="collapsed",
            on_change=_jump_cb,
            args=("train_img_idx", jump_key, max_idx),
        )

        if c6.button("↩️", key="train_undo", use_container_width=True, disabled=(len(u) == 0)):
            if undo_last_click(uid_tr, base_dir, slic_params, cfg.overlay_alpha, cfg.show_boundaries, cfg.boundary_thick_px):
                # garante sidecars atualizados também no undo
                _save_everything(
                    base_dir=base_dir,
                    img_path=train_img_path,
                    rgb=train_rgb,
                    slic=train_slic,
                    labelmap=train_labelmap,
                    cfg=cfg,
                )
                _bump_click_nonce(uid_tr)
                st.rerun()

        if c7.button("↪️", key="train_redo", use_container_width=True, disabled=(len(r) == 0)):
            if redo_last_click(uid_tr, base_dir, slic_params, cfg.overlay_alpha, cfg.show_boundaries, cfg.boundary_thick_px):
                # garante sidecars atualizados também no redo
                _save_everything(
                    base_dir=base_dir,
                    img_path=train_img_path,
                    rgb=train_rgb,
                    slic=train_slic,
                    labelmap=train_labelmap,
                    cfg=cfg,
                )
                _bump_click_nonce(uid_tr)
                st.rerun()

        if c8.button("🧹", key="train_clear", use_container_width=True):
            train_labelmap[:] = UNLAB
            _save_everything(
                base_dir=base_dir,
                img_path=train_img_path,
                rgb=train_rgb,
                slic=train_slic,
                labelmap=train_labelmap,
                cfg=cfg,
            )
            _bump_click_nonce(uid_tr)
            st.rerun()

    # =========================================================================
    # ② Treinamento / Avaliação
    # =========================================================================
    st.markdown("---")
    st.markdown("<div class='step-title'>② Treinamento / Avaliação</div>", unsafe_allow_html=True)

    def collect_training_sets() -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        n_classes = len(st.session_state.classes)

        for p in img_paths:
            rgb = imread_rgb(p)
            H, W = rgb.shape[:2]
            uid = image_uid(p)

            if uid not in st.session_state.labelmaps_cache:
                st.session_state.labelmaps_cache[uid] = load_or_create_labelmap(base_dir, p, (H, W))
            lm = st.session_state.labelmaps_cache[uid]

            labels = compute_slic_labels_cached(
                rgb,
                n_segments=int(slic_params[0]),
                compactness=float(slic_params[1]),
                sigma=float(slic_params[2]),
                enforce_connectivity=bool(slic_params[3]),
                key=_img_key(p.name, slic_params),
            )
            feats = compute_superpixel_features_cached(rgb, labels, key=_img_key(p.name, slic_params) + ":feats")
            feats_z = zscore(feats)

            y_spx, _ = majority_label_per_superpixel(
                labels,
                lm,
                n_classes=n_classes,
                min_frac=float(cfg.min_frac),
                UNLAB=int(UNLAB),
            )
            sel = np.where(y_spx >= 0)[0]
            if sel.size == 0:
                continue

            X_list.append(feats_z[sel, :])
            y_list.append(y_spx[sel])

        if not X_list:
            return np.empty((0, 12), np.float32), np.empty((0,), np.int32)

        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list).astype(np.int32)
        return X, y

    colT1, colT2, colT3 = st.columns([1.2, 1, 1])

    with colT1:
        if st.button("🏋️ Treinar (todas as imagens rotuladas)", key="btn_train"):
            X_lab, y_lab = collect_training_sets()
            if X_lab.shape[0] < 4:
                st.error("Rotule mais regiões (≥4 amostras no total).")
            else:
                cls, cnt = np.unique(y_lab, return_counts=True)
                if len(cls) < 2:
                    st.error("Precisa de pelo menos **2 classes**.")
                elif (cnt < 2).any():
                    st.error("Cada classe precisa de **≥2 amostras**.")
                else:
                    from .persistence import save_knn_bundle, save_supcon_head

                    save_classes(base_dir, st.session_state.classes)

                    if cfg.train_mode.startswith("SupCon"):
                        if not TORCH_OK:
                            st.warning("PyTorch indisponível; usando k-NN (NumPy).")
                            st.session_state["MODEL_MODE"] = "kNN"
                            st.session_state["MODEL_X_LAB"] = X_lab
                            st.session_state["MODEL_Y_LAB"] = y_lab
                            st.session_state["MODEL_IS_SUPCON"] = False
                            st.session_state["MODEL_Z_LAB"] = None
                            st.session_state["MODEL_HEAD"] = None
                            st.session_state["MODEL_LOG"] = []
                            save_knn_bundle(base_dir, X_lab=X_lab, y_lab=y_lab, is_supcon=False, Z_lab=None, slic_params=slic_params)
                            st.success("Modelo k-NN salvo em disco!")
                        else:
                            prog = st.progress(0, text="Iniciando SupCon…")
                            msg = st.empty()

                            def _cb(ep, total, loss):
                                fracp = ep / float(total)
                                txt = f"Treinando SupCon… época {ep}/{total}"
                                if loss is not None:
                                    txt += f" • loss {loss:.4f}"
                                prog.progress(min(1.0, fracp), text=txt)
                                msg.write(txt)

                            with st.spinner("Treinando SupCon…"):
                                cfg_head = dict(in_dim=12, hidden=64, out_dim=32, p_drop=0.1)
                                head, loss_log = train_supcon_on_feats(
                                    X_lab,
                                    y_lab,
                                    epochs=int(cfg.supcon_epochs),
                                    batch=128,
                                    lr=float(cfg.supcon_lr),
                                    temperature=float(cfg.supcon_temp),
                                    device=None,
                                    log_every=25,
                                    progress_cb=_cb,
                                    hidden=cfg_head["hidden"],
                                    out_dim=cfg_head["out_dim"],
                                    p_drop=cfg_head["p_drop"],
                                )

                            prog.progress(1.0, text="SupCon concluído!")
                            msg.write("✅ SupCon concluído.")

                            Z_lab = embed_all_feats(head, X_lab)

                            st.session_state["MODEL_MODE"] = "SupCon+kNN"
                            st.session_state["MODEL_HEAD"] = head
                            st.session_state["MODEL_X_LAB"] = X_lab
                            st.session_state["MODEL_Z_LAB"] = Z_lab
                            st.session_state["MODEL_Y_LAB"] = y_lab
                            st.session_state["MODEL_IS_SUPCON"] = True
                            st.session_state["MODEL_LOG"] = loss_log
                            st.session_state["MODEL_SUPCON_CFG"] = cfg_head

                            save_supcon_head(base_dir, head, cfg_head)
                            save_knn_bundle(base_dir, X_lab=X_lab, y_lab=y_lab, is_supcon=True, Z_lab=Z_lab, slic_params=slic_params, supcon_cfg=cfg_head)
                            st.session_state.emb_cache.clear()
                            st.success("SupCon+kNN treinado e salvo em disco!")
                    else:
                        from .persistence import save_knn_bundle

                        st.session_state["MODEL_MODE"] = "kNN"
                        st.session_state["MODEL_X_LAB"] = X_lab
                        st.session_state["MODEL_Y_LAB"] = y_lab
                        st.session_state["MODEL_IS_SUPCON"] = False
                        st.session_state["MODEL_Z_LAB"] = None
                        st.session_state["MODEL_HEAD"] = None
                        st.session_state["MODEL_LOG"] = []
                        save_knn_bundle(base_dir, X_lab=X_lab, y_lab=y_lab, is_supcon=False, Z_lab=None, slic_params=slic_params)
                        st.success("k-NN preparado e salvo em disco!")

    with colT2:
        if st.button("🧪 Avaliar (leave-one-out)", key="btn_eval"):
            if st.session_state.get("MODEL_Y_LAB") is None:
                st.error("Treine o modelo primeiro.")
            else:
                y_lab = st.session_state["MODEL_Y_LAB"]
                X_lab = st.session_state.get("MODEL_X_LAB", None)
                Z_lab = st.session_state.get("MODEL_Z_LAB", None)
                is_supcon = bool(st.session_state.get("MODEL_IS_SUPCON", False))
                head = st.session_state.get("MODEL_HEAD", None)

                if X_lab is None or y_lab is None:
                    st.error("Modelo inválido. Treine novamente.")
                else:
                    Z_ref = Z_lab if (is_supcon and TORCH_OK and (head is not None) and (Z_lab is not None)) else X_lab
                    k = int(min(max(1, int(cfg.k_neighbors)), len(y_lab)))
                    loo_idx = np.arange(len(y_lab))
                    y_pred = knn_predict_numpy_balanced(
                        Z_ref,
                        y_lab,
                        Z_ref,
                        k=k,
                        balance=bool(cfg.balance_knn),
                        leave_one_out_idx=loo_idx,
                    )
                    acc = float((y_pred == y_lab).mean())
                    st.info(f"Acurácia LOO: **{acc * 100:.1f}%**  |  k={k}  |  balanceado={bool(cfg.balance_knn)}")

    with colT3:
        if st.button("🧽 Limpar modelo/predição (memória)", key="btn_clear_all"):
            for k_ in [
                "MODEL_MODE",
                "MODEL_HEAD",
                "MODEL_Z_LAB",
                "MODEL_X_LAB",
                "MODEL_Y_LAB",
                "MODEL_IS_SUPCON",
                "MODEL_LOG",
                "MODEL_SUPCON_CFG",
            ]:
                st.session_state.pop(k_, None)
            st.session_state.pred_cache.clear()
            st.session_state.emb_cache.clear()
            st.success("Modelo e predições limpos (memória).")

    # =========================================================================
    # ④ Exportar dataset rotulado
    # =========================================================================
    st.markdown("---")
    st.markdown("<div class='step-title'>④ Exportar / Enviar base rotulada</div>", unsafe_allow_html=True)

    inc_img = st.checkbox("Incluir imagens no ZIP", value=True, key="zip_inc_img")
    inc_ov = st.checkbox("Incluir overlays no ZIP", value=False, key="zip_inc_ov")

    if st.button("📦 Gerar ZIP do dataset rotulado", use_container_width=True, key="zip_build_btn"):
        data = build_labeled_dataset_zip(base_dir, include_images=inc_img, include_overlays=inc_ov)
        st.download_button(
            "⬇️ Baixar dataset rotulado (ZIP)",
            data=data,
            file_name="slic_dataset_rotulado.zip",
            mime="application/zip",
            use_container_width=True,
            key="zip_download_btn",
        )

    dirs = ensure_out_dirs(base_dir)
    st.markdown("### 💾 Saída (salvo em tempo real)")
    st.write(
        f"- Labelmaps: `{dirs['labelmaps']}`  \n"
        f"- Overlays: `{dirs['overlays']}`  \n"
        f"- Masks/sidecars: `{(Path(base_dir) / 'masks')}`  \n"
        f"- Meta/index: `{(dirs['meta']/'index.json')}`  \n"
        f"- classes.json: `{(dirs['meta']/'classes.json')}`"
    )
