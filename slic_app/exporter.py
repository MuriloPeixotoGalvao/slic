# slic_app/exporter.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io
import re
import zipfile

import numpy as np
import cv2

from .constants import EXTS, UNLAB
from .io_images import imread_rgb
from .persistence import load_or_create_labelmap, save_classes


# -----------------------------
# Helpers
# -----------------------------
def _sanitize(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", name)
    return name or "class"


def _color_palette(n: int) -> np.ndarray:
    """
    Paleta determinística (n x 3) em RGB.
    UNLAB não entra aqui (tratamos como preto).
    """
    # determinístico e com boa separação
    rng = np.random.RandomState(12345)
    cols = rng.randint(0, 255, size=(max(n, 1), 3), dtype=np.uint8)

    # evita cores muito escuras
    mins = cols.min(axis=1)
    dark = mins < 40
    cols[dark] = np.clip(cols[dark] + 60, 0, 255)

    return cols


def _colorize_labelmap(labelmap: np.ndarray, n_classes: int, unl_id: int) -> np.ndarray:
    """
    labelmap: HxW (uint16/uint8) com ids de classe e UNLAB
    retorna RGB uint8 HxWx3
    """
    H, W = labelmap.shape[:2]
    out = np.zeros((H, W, 3), np.uint8)  # UNLAB = preto
    pal = _color_palette(n_classes)      # id -> cor

    # pinta cada classe
    for cid in range(n_classes):
        m = (labelmap == np.uint16(cid))
        if m.any():
            out[m] = pal[cid]
    return out


def _encode_png(arr: np.ndarray) -> bytes:
    """
    Encode para PNG.
    Aceita:
      - HxW uint8 (binária)
      - HxWx3 uint8 (RGB)
      - HxW uint16 (id mask 16-bit)
    """
    if arr.ndim == 3 and arr.shape[2] == 3:
        # cv2 espera BGR
        bgr = arr[:, :, ::-1]
        ok, buf = cv2.imencode(".png", bgr)
    else:
        ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("Falha ao codificar PNG.")
    return buf.tobytes()


def _read_file_bytes(p: Path) -> bytes:
    return p.read_bytes()


# -----------------------------
# Main export
# -----------------------------
def build_labeled_dataset_zip(
    base_dir: Path,
    *,
    include_images: bool = True,
    include_masks_color: bool = True,
    include_masks_id: bool = True,
    include_masks_bin: bool = True,
    include_only_labeled: bool = True,
    source_dir: Optional[Path] = None,
) -> bytes:
    """
    Exporta um ZIP com:
      images/               -> imagem original
      masks_color/          -> máscara RGB (todas as classes)
      masks_id/             -> máscara 1-canal com ids (PNG 16-bit)
      masks_bin/<classe>/   -> máscaras binárias 0/255 (uma por classe)

    include_only_labeled=True:
      só exporta imagens cujo labelmap tenha pelo menos 1 pixel != UNLAB
    """
    base_dir = Path(base_dir)

    # Fonte de imagens: por padrão, usa base_dir (ideal: full-res)
    if source_dir is None:
        source_dir = base_dir
    source_dir = Path(source_dir)

    # lista imagens
    img_paths = sorted([p for p in source_dir.glob("*") if p.suffix.lower() in EXTS])

    # classes (do session_state geralmente já existem, mas aqui export é “agnóstico”)
    # se existir classes em disco, melhor — mas como você já salva classes.json,
    # a UI geralmente chama save_classes(). Ainda assim, garantimos que exista.
    classes: List[str] = []
    # tenta obter classes do st.session_state se estiver em runtime streamlit
    try:
        import streamlit as st  # type: ignore
        classes = list(st.session_state.get("classes", []))
    except Exception:
        classes = []

    if not classes:
        # fallback: ao menos 1 classe
        classes = ["class_0"]

    # salva classes (mantém compatível com seu pipeline)
    save_classes(base_dir, classes)

    n_classes = len(classes)
    class_folders = [_sanitize(f"{i:02d}_{nm}") for i, nm in enumerate(classes)]

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # manifesto simples
        z.writestr("README.txt",
                   "Estrutura:\n"
                   "- images/                 imagem original\n"
                   "- masks_color/            mascara RGB (todas as classes)\n"
                   "- masks_id/               mascara 1-canal com ids (PNG 16-bit)\n"
                   "- masks_bin/<classe>/     mascara binaria 0/255 por classe\n"
                   f"UNLAB={int(UNLAB)}\n")

        # classes
        z.writestr("classes.txt", "\n".join([f"{i}\t{nm}" for i, nm in enumerate(classes)]) + "\n")

        exported = 0

        for img_path in img_paths:
            # lê imagem só para pegar shape e (opcionalmente) exportar o original
            rgb = imread_rgb(img_path)
            H, W = rgb.shape[:2]

            # carrega labelmap “do jeito oficial” do seu app (não precisa saber nome do arquivo)
            labelmap = load_or_create_labelmap(base_dir, img_path, (H, W))

            # garante shape
            if labelmap.shape[:2] != (H, W):
                labelmap = cv2.resize(labelmap, (W, H), interpolation=cv2.INTER_NEAREST)

            # filtra não rotuladas
            if include_only_labeled:
                if not (labelmap != np.uint16(UNLAB)).any():
                    continue

            stem = img_path.stem
            # nome base padronizado (mantém extensão original para images/)
            img_rel = f"images/{img_path.name}"

            # 1) imagem original
            if include_images:
                try:
                    z.writestr(img_rel, _read_file_bytes(img_path))
                except Exception:
                    # fallback: re-encode lossless png
                    z.writestr(f"images/{stem}.png", _encode_png(rgb))

            # 2) máscara colorida (RGB)
            if include_masks_color:
                color = _colorize_labelmap(labelmap, n_classes=n_classes, unl_id=int(UNLAB))
                z.writestr(f"masks_color/{stem}.png", _encode_png(color))

            # 3) máscara de ids (1 canal)
            if include_masks_id:
                # export 16-bit para suportar >255 classes sem perder id
                id_mask = labelmap.astype(np.uint16, copy=False)
                z.writestr(f"masks_id/{stem}.png", _encode_png(id_mask))

            # 4) binárias por classe
            if include_masks_bin:
                for cid, folder in enumerate(class_folders):
                    m = (labelmap == np.uint16(cid))
                    if not m.any():
                        # opcional: se você quiser sempre criar os arquivos vazios, remova esse continue
                        continue
                    bin_mask = (m.astype(np.uint8) * 255)
                    z.writestr(f"masks_bin/{folder}/{stem}.png", _encode_png(bin_mask))

            exported += 1

        z.writestr("export_stats.txt", f"exported_images={exported}\nclasses={n_classes}\n")

    return mem.getvalue()
