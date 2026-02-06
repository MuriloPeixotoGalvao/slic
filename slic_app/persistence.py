# slic_app/persistence.py
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import UNLAB
from .supcon import TORCH_OK, ContrastiveHead


# -----------------------------
# Pastas de saída
# -----------------------------
def ensure_out_dirs(base_dir: Path) -> Dict[str, Path]:
    base_dir = Path(base_dir)
    dirs = {
        "labelmaps": base_dir / "labelmaps",
        "overlays": base_dir / "overlays",
        "meta": base_dir / "meta",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def image_uid(img_path: Path) -> str:
    """
    UID estável (pelo caminho absoluto). Evita colisão em stems iguais.
    """
    s = str(Path(img_path).resolve()).encode("utf-8", errors="ignore")
    return hashlib.md5(s).hexdigest()[:12]


def _labelmap_path(base_dir: Path, img_path: Path) -> Path:
    dirs = ensure_out_dirs(base_dir)
    uid = image_uid(img_path)
    return dirs["labelmaps"] / f"{img_path.stem}__{uid}.npy"


def _overlay_path(base_dir: Path, img_path: Path) -> Path:
    dirs = ensure_out_dirs(base_dir)
    uid = image_uid(img_path)
    return dirs["overlays"] / f"{img_path.stem}__{uid}.png"


def load_or_create_labelmap(base_dir: Path, img_path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    p = _labelmap_path(base_dir, img_path)
    if p.exists():
        lm = np.load(p)
        if lm.shape != tuple(shape_hw):
            # se mudou resolução, recria (evita crash)
            lm = np.full(shape_hw, UNLAB, dtype=np.uint16)
            np.save(p, lm)
        return lm.astype(np.uint16, copy=False)
    lm = np.full(shape_hw, UNLAB, dtype=np.uint16)
    np.save(p, lm)
    return lm


def save_label_outputs(base_dir: Path, img_path: Path, labelmap: np.ndarray, overlay_rgb: np.ndarray) -> None:
    dirs = ensure_out_dirs(base_dir)
    np.save(_labelmap_path(base_dir, img_path), labelmap.astype(np.uint16, copy=False))

    # salvar overlay (RGB) como PNG
    out = _overlay_path(base_dir, img_path)
    try:
        from PIL import Image
        Image.fromarray(overlay_rgb).save(out)
    except Exception:
        # fallback mínimo
        out.write_bytes(b"")


# -----------------------------
# Classes
# -----------------------------
def _classes_path(base_dir: Path) -> Path:
    dirs = ensure_out_dirs(base_dir)
    return dirs["meta"] / "classes.json"


def save_classes(base_dir: Path, classes: List[str]) -> None:
    p = _classes_path(base_dir)
    p.write_text(json.dumps({"classes": classes}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_classes(base_dir: Path, default: Optional[List[str]] = None) -> List[str]:
    p = _classes_path(base_dir)
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            cls = obj.get("classes", None)
            if isinstance(cls, list) and cls:
                return [str(x) for x in cls]
        except Exception:
            pass
    return default if (default is not None and len(default) > 0) else ["classe_0"]


# -----------------------------
# Index/meta
# -----------------------------
def _index_path(base_dir: Path) -> Path:
    dirs = ensure_out_dirs(base_dir)
    return dirs["meta"] / "index.json"


# -----------------------------
# Config (UI)
# -----------------------------
def _config_path(base_dir: Path) -> Path:
    dirs = ensure_out_dirs(base_dir)
    return dirs["meta"] / "config.json"


def save_config(base_dir: Path, cfg: Dict[str, Any]) -> None:
    """
    Salva configuração de UI para restaurar na próxima sessão.
    """
    p = _config_path(base_dir)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(base_dir: Path) -> Dict[str, Any]:
    """
    Carrega configuração de UI. Retorna {} se não existir.
    """
    p = _config_path(base_dir)
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def clear_config(base_dir: Path) -> None:
    """
    Remove config.json (restaura defaults).
    """
    p = _config_path(base_dir)
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def update_meta_index(base_dir: Path, img_path: Path) -> None:
    """
    Mantém um index simples com os arquivos já processados.
    """
    base_dir = Path(base_dir)
    p = _index_path(base_dir)
    uid = image_uid(img_path)

    try:
        idx = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        idx = {}

    idx.setdefault("images", {})
    idx["images"][uid] = {
        "name": img_path.name,
        "stem": img_path.stem,
        "uid": uid,
        "labelmap": str(_labelmap_path(base_dir, img_path).as_posix()),
        "overlay": str(_overlay_path(base_dir, img_path).as_posix()),
    }

    p.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# Modelo kNN / SupCon bundle
# -----------------------------
def save_knn_bundle(
    base_dir: Path,
    X_lab: np.ndarray,
    y_lab: np.ndarray,
    is_supcon: bool,
    Z_lab: Optional[np.ndarray],
    slic_params: Tuple[int, float, float, bool],
    supcon_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    dirs = ensure_out_dirs(base_dir)
    out = dirs["meta"] / "knn_bundle.npz"
    np.savez_compressed(
        out,
        X_lab=X_lab.astype(np.float32),
        y_lab=y_lab.astype(np.int32),
        is_supcon=np.array([1 if is_supcon else 0], np.int8),
        Z_lab=(Z_lab.astype(np.float32) if Z_lab is not None else np.empty((0, 0), np.float32)),
        slic_params=np.array(list(slic_params), dtype=object),
        supcon_cfg=json.dumps(supcon_cfg or {}, ensure_ascii=False),
    )


def save_supcon_head(base_dir: Path, head: "ContrastiveHead", cfg_head: Dict[str, Any]) -> None:
    """
    Salva o state_dict do head + cfg.
    """
    dirs = ensure_out_dirs(base_dir)
    p = dirs["meta"] / "supcon_head.pt"
    p_cfg = dirs["meta"] / "supcon_head_cfg.json"

    p_cfg.write_text(json.dumps(cfg_head, ensure_ascii=False, indent=2), encoding="utf-8")

    if not TORCH_OK:
        raise RuntimeError("PyTorch não disponível. Não é possível salvar SupCon head.")

    import torch
    torch.save(head.state_dict(), p)


def load_supcon_head(base_dir: Path) -> Tuple[Optional["ContrastiveHead"], Optional[Dict[str, Any]]]:
    """
    Carrega head + cfg (se existirem). Retorna (head|None, cfg|None).
    """
    dirs = ensure_out_dirs(base_dir)
    p = dirs["meta"] / "supcon_head.pt"
    p_cfg = dirs["meta"] / "supcon_head_cfg.json"

    if not (p.exists() and p_cfg.exists()):
        return None, None

    cfg = json.loads(p_cfg.read_text(encoding="utf-8"))

    if not TORCH_OK:
        return None, cfg

    import torch
    head = ContrastiveHead(
        in_dim=int(cfg["in_dim"]),
        hidden=int(cfg.get("hidden", 64)),
        out_dim=int(cfg.get("out_dim", 32)),
        p_drop=float(cfg.get("p_drop", 0.1)),
    )
    sd = torch.load(p, map_location="cpu")
    head.load_state_dict(sd)
    head.eval()
    return head, cfg
