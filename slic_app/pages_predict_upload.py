from __future__ import annotations

from typing import Optional
import io
from pathlib import Path

import numpy as np
import streamlit as st
import cv2
from PIL import Image

from slic_app.ui_pick import pick_folder_button  # ✅ NOVO

from .constants import UNLAB
from .slic_ops import (
    compute_slic_labels_cached,
    compute_superpixel_features_cached,
    zscore,
)
from .knn import knn_predict_numpy_balanced
from .supcon import TORCH_OK, embed_all_feats
from .vis import resize_for_view


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _bytes_to_rgb(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def _path_to_rgb(path: str | Path) -> np.ndarray:
    p = Path(path)
    img = Image.open(p).convert("RGB")
    return np.array(img)


def _make_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    pal = rng.integers(0, 255, size=(max(1, n), 3), dtype=np.uint8)
    if n > 0:
        pal[0] = np.array([0, 0, 0], np.uint8)
    return pal


def _overlay_from_predmap(rgb: np.ndarray, pred_u16: np.ndarray, n_classes: int, alpha: float) -> np.ndarray:
    H, W = rgb.shape[:2]
    pal = _make_palette(n_classes)
    color = np.zeros((H, W, 3), np.uint8)

    valid = (pred_u16 != np.uint16(UNLAB)) & (pred_u16 < np.uint16(n_classes))
    if valid.any():
        color[valid] = pal[pred_u16[valid].astype(np.int32)]

    out = (rgb.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return out


def _iter_tiles(H: int, W: int, tile: int, overlap: int):
    step = max(1, tile - overlap)
    y0 = 0
    while y0 < H:
        x0 = 0
        y1 = min(H, y0 + tile)
        while x0 < W:
            x1 = min(W, x0 + tile)
            yield y0, y1, x0, x1
            x0 += step
        y0 += step


@st.cache_data(show_spinner=False)
def _list_images_cached(folder: str, max_files: int = 2000, nonce: int = 0) -> list[str]:
    _ = nonce  # só para quebrar cache quando clicar "Atualizar"
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    p = Path(folder)
    if not p.exists():
        return []
    out: list[str] = []
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            try:
                rel = fp.relative_to(p)
                if rel.parts and rel.parts[0].lower() == "masks":
                    continue
            except Exception:
                pass
            out.append(str(fp))
            if len(out) >= max_files:
                break
    out.sort()
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def render_predict_upload(cfg) -> None:
    st.markdown(
        "<div class='step-title'>Predição — Upload / Diretório (tiling/janela deslizante)</div>",
        unsafe_allow_html=True,
    )

    # Checar modelo
    if st.session_state.get("MODEL_Y_LAB") is None:
        st.info("Treine o modelo na página de rotulagem antes de usar esta página.")
        return

    y_lab = st.session_state["MODEL_Y_LAB"]
    X_lab = st.session_state.get("MODEL_X_LAB", None)
    Z_lab = st.session_state.get("MODEL_Z_LAB", None)
    is_supcon = bool(st.session_state.get("MODEL_IS_SUPCON", False))
    head = st.session_state.get("MODEL_HEAD", None)
    model_mode = str(st.session_state.get("MODEL_MODE", "kNN"))
    rf = st.session_state.get("MODEL_RF", None)

    if model_mode == "RandomForest":
        if rf is None or y_lab is None:
            st.error("Modelo RandomForest inválido em memória. Treine novamente.")
            return
    elif X_lab is None or y_lab is None:
        st.error("Modelo inválido em memória. Treine novamente.")
        return

    n_classes = len(st.session_state.classes)

    # -------------------------------------------------------------------------
    # Fonte da imagem
    # -------------------------------------------------------------------------
    src = st.radio("Fonte da imagem", ["Upload", "Diretório"], horizontal=True, index=0, key="pred_src_mode")

    rgb: Optional[np.ndarray] = None
    src_name: Optional[str] = None
    uid: Optional[str] = None

    if src == "Upload":
        up = st.file_uploader(
            "Envie uma imagem (JPG/PNG/TIF). Para ortos grandes, use tiles menores.",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            accept_multiple_files=False,
            key="pred_upload_file",
        )
        if not up:
            return

        file_bytes = up.getvalue()
        rgb = _bytes_to_rgb(file_bytes)
        H, W = rgb.shape[:2]
        src_name = up.name
        uid = f"UPLOAD:{up.name}:{H}x{W}:{len(file_bytes)}"

    else:
        # ✅ Campo + botão que preenche automaticamente (SEM dar StreamlitAPIException)
        st.text_input("Diretório (para buscar imagens)", key="pred_dir", placeholder=r"ex: E:\dataset\tiles")

        pick_folder_button(
            "pred_dir",
            label="📂 Escolher diretório de imagens…",
            initialdir="E:/",
        )

        folder = (st.session_state.get("pred_dir") or "").strip()
        if not folder:
            st.info("Selecione um diretório acima.")
            return

        if not Path(folder).exists():
            st.error("Diretório não existe.")
            return

        # refresh nonce
        if "pred_dir_nonce" not in st.session_state:
            st.session_state.pred_dir_nonce = 0

        r1, r2 = st.columns([1, 2])
        if r1.button("🔄 Atualizar lista", use_container_width=True, key="pred_dir_refresh_btn"):
            st.session_state.pred_dir_nonce += 1
            st.rerun()
        r2.caption("Use quando adicionar/remover arquivos (quebra o cache).")

        max_files = int(st.slider("Limite de arquivos listados", 100, 5000, 2000, 100, key="pred_dir_maxfiles"))
        files = _list_images_cached(folder, max_files=max_files, nonce=int(st.session_state.pred_dir_nonce))
        if not files:
            st.warning("Nenhuma imagem encontrada nessa pasta (recursivo).")
            return

        only_name = st.text_input(
            "Filtrar nome (opcional)",
            value="",
            placeholder="ex: tile_001, 2024, areaA",
            key="pred_dir_filter",
        )
        if only_name.strip():
            f = only_name.strip().lower()
            files2 = [p for p in files if f in Path(p).name.lower()]
            if files2:
                files = files2

        pick = st.selectbox("Escolha a imagem da pasta", files, index=0, key="pred_dir_pick")
        if not pick:
            return

        try:
            rgb = _path_to_rgb(pick)
        except Exception as e:
            st.error(f"Falha ao abrir imagem: {e}")
            return

        H, W = rgb.shape[:2]
        src_name = Path(pick).name
        try:
            st_mtime = int(Path(pick).stat().st_mtime)
        except Exception:
            st_mtime = 0
        uid = f"DIR:{pick}:{H}x{W}:{st_mtime}"

    # -------------------------------------------------------------------------
    # Controles
    # -------------------------------------------------------------------------
    H, W = rgb.shape[:2]

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        tile = st.selectbox("Tile (px)", [512, 640, 768, 1024, 1280, 1536, 2048], index=3, key="upl_tile")
    with c2:
        overlap = st.selectbox("Overlap (px)", [0, 64, 128, 192, 256, 320, 384], index=4, key="upl_overlap")
    with c3:
        k = int(st.number_input("k vizinhos", min_value=1, max_value=99, value=int(getattr(cfg, "k_neighbors", 5)), step=1, key="upl_k"))
    with c4:
        alpha = float(st.slider("Overlay α", 0.0, 1.0, float(getattr(cfg, "overlay_alpha", 0.35)), 0.01, key="upl_alpha"))

    balance = bool(getattr(cfg, "balance_knn", True))
    st.caption(f"Imagem: **{src_name}** | {W}×{H} | tiles={tile} | overlap={overlap} | k={k} | SupCon={is_supcon}")

    run = st.button("🔮 Rodar predição", use_container_width=True, key="upl_run")
    if not run:
        disp, _, _ = resize_for_view(rgb, int(min(1200, getattr(cfg, "img_width", 1200))))
        st.image(disp, caption="Preview", use_container_width=True)
        return

    # -------------------------------------------------------------------------
    # Predição por tiling
    # -------------------------------------------------------------------------
    pred_u16 = np.full((H, W), np.uint16(UNLAB), dtype=np.uint16)

    prog = st.progress(0, text="Iniciando…")
    total_tiles = sum(1 for _ in _iter_tiles(H, W, int(tile), int(overlap)))

    # referência (SupCon ou raw) para kNN
    Z_ref = Z_lab if (is_supcon and TORCH_OK and (head is not None) and (Z_lab is not None)) else X_lab
    slic_params = cfg.slic_params

    for tile_i, (y0, y1, x0, x1) in enumerate(_iter_tiles(H, W, int(tile), int(overlap)), start=1):
        prog.progress(min(1.0, tile_i / max(1, total_tiles)), text=f"Processando tile {tile_i}/{total_tiles}…")

        rgb_t = rgb[y0:y1, x0:x1, :]
        Ht, Wt = rgb_t.shape[:2]
        if Ht < 32 or Wt < 32:
            continue

        key = f"{uid}:{y0}:{x0}:{slic_params[0]}:{slic_params[1]}:{slic_params[2]}:{slic_params[3]}"

        labels_t = compute_slic_labels_cached(
            rgb_t,
            n_segments=int(slic_params[0]),
            compactness=float(slic_params[1]),
            sigma=float(slic_params[2]),
            enforce_connectivity=bool(slic_params[3]),
            key=key,
        )
        feats_t = compute_superpixel_features_cached(rgb_t, labels_t, key=key + ":feats")
        feats_tz = zscore(feats_t).astype(np.float32)

        if model_mode == "RandomForest":
            y_pred_spx = rf.predict(feats_tz).astype(np.int32, copy=False)
        else:
            Z_all = embed_all_feats(head, feats_tz) if (is_supcon and TORCH_OK and (head is not None) and (Z_lab is not None)) else feats_tz
            kk = int(min(max(1, k), len(y_lab)))
            y_pred_spx = knn_predict_numpy_balanced(Z_ref, y_lab, Z_all, k=kk, balance=balance).astype(np.int32)
        pred_pix = y_pred_spx[labels_t].astype(np.int32)

        # colar usando “miolo”
        m = int(overlap) // 2
        iy0 = 0 if y0 == 0 else m
        ix0 = 0 if x0 == 0 else m
        iy1 = Ht if y1 == H else (Ht - m)
        ix1 = Wt if x1 == W else (Wt - m)
        if iy1 <= iy0 or ix1 <= ix0:
            continue

        patch = pred_pix[iy0:iy1, ix0:ix1]
        pred_u16[y0 + iy0 : y0 + iy1, x0 + ix0 : x0 + ix1] = patch.astype(np.uint16)

    prog.progress(1.0, text="✅ Predição concluída!")

    # -------------------------------------------------------------------------
    # Visualização
    # -------------------------------------------------------------------------
    overlay = _overlay_from_predmap(rgb, pred_u16, n_classes=n_classes, alpha=float(alpha))

    disp_rgb, _, _ = resize_for_view(rgb, int(min(1200, getattr(cfg, "img_width", 1200))))
    disp_ov, _, _ = resize_for_view(overlay, int(min(1200, getattr(cfg, "img_width", 1200))))

    v1, v2 = st.columns(2)
    with v1:
        st.image(disp_rgb, caption="Original (preview)", use_container_width=True)
    with v2:
        st.image(disp_ov, caption="Overlay da predição (preview)", use_container_width=True)

    ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if ok:
        stem = Path(src_name).stem if src_name else "img"
        st.download_button(
            "⬇️ Baixar overlay PNG",
            data=buf.tobytes(),
            file_name=f"pred_overlay_{stem}.png",
            mime="image/png",
            use_container_width=True,
            key="upl_dl_png",
        )

    st.caption("Obs.: esta página gera um mapa de classe por pixel via tiling (sem salvar labelmap no projeto).")
