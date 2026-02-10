# slic_app/persistence.py
from __future__ import annotations

import json
import hashlib
import re
import shutil
from datetime import datetime
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


def image_uid(img_path: Path, base_dir: Optional[Path] = None) -> str:
    """
    UID estável.
    - Preferencialmente por caminho relativo ao base_dir (portável entre PCs).
    - Fallback: caminho absoluto.
    """
    p = Path(img_path)
    if base_dir is not None:
        try:
            rel = p.resolve().relative_to(Path(base_dir).resolve())
            s = str(rel.as_posix()).encode("utf-8", errors="ignore")
        except Exception:
            s = str(p.resolve()).encode("utf-8", errors="ignore")
    else:
        s = str(p.resolve()).encode("utf-8", errors="ignore")
    return hashlib.md5(s).hexdigest()[:12]


def _rel_image_path(base_dir: Path, img_path: Path) -> Optional[Path]:
    """
    Caminho relativo da imagem dentro do projeto, para persistência portável.
    """
    try:
        return Path(img_path).resolve().relative_to(Path(base_dir).resolve())
    except Exception:
        return None


def _labelmap_path(base_dir: Path, img_path: Path) -> Path:
    dirs = ensure_out_dirs(base_dir)
    rel = _rel_image_path(base_dir, img_path)
    if rel is not None:
        p = dirs["labelmaps"] / rel.with_suffix(".npy")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    uid = image_uid(img_path, base_dir=base_dir)
    return dirs["labelmaps"] / f"{img_path.stem}__{uid}.npy"


def _overlay_path(base_dir: Path, img_path: Path) -> Path:
    dirs = ensure_out_dirs(base_dir)
    rel = _rel_image_path(base_dir, img_path)
    if rel is not None:
        p = dirs["overlays"] / rel.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    uid = image_uid(img_path, base_dir=base_dir)
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
    uid = image_uid(img_path, base_dir=base_dir)

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


def load_knn_bundle(base_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Carrega knn_bundle.npz salvo em disco.
    Retorna dict com campos do modelo ou None se não existir/inválido.
    """
    dirs = ensure_out_dirs(base_dir)
    p = dirs["meta"] / "knn_bundle.npz"
    if not p.exists():
        return None
    try:
        obj = np.load(p, allow_pickle=True)
        X_lab = obj["X_lab"].astype(np.float32)
        y_lab = obj["y_lab"].astype(np.int32)
        is_supcon = bool(int(np.array(obj["is_supcon"]).reshape(-1)[0]))
        Z_raw = obj["Z_lab"]
        Z_lab = None if Z_raw.size == 0 else Z_raw.astype(np.float32)
        slic_params_raw = tuple(obj["slic_params"].tolist())
        if len(slic_params_raw) == 4:
            slic_params = (
                int(slic_params_raw[0]),
                float(slic_params_raw[1]),
                float(slic_params_raw[2]),
                bool(slic_params_raw[3]),
            )
        else:
            slic_params = None
        supcon_cfg_raw = obj["supcon_cfg"]
        if isinstance(supcon_cfg_raw, np.ndarray):
            supcon_cfg_raw = supcon_cfg_raw.item() if supcon_cfg_raw.size else "{}"
        supcon_cfg = json.loads(str(supcon_cfg_raw)) if str(supcon_cfg_raw).strip() else {}
        return {
            "X_lab": X_lab,
            "y_lab": y_lab,
            "is_supcon": is_supcon,
            "Z_lab": Z_lab,
            "slic_params": slic_params,
            "supcon_cfg": supcon_cfg if isinstance(supcon_cfg, dict) else {},
        }
    except Exception:
        return None


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


def save_named_model_snapshot(base_dir: Path, model_name: str, is_supcon: bool) -> Dict[str, str]:
    """
    Salva uma cópia nomeada dos arquivos atuais do modelo (bundle e, se existir, head SupCon).
    Requer que os arquivos padrão já tenham sido salvos em meta/.
    """
    dirs = ensure_out_dirs(base_dir)
    meta_dir = dirs["meta"]
    models_dir = meta_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", (model_name or "").strip()).strip("._-")
    if not safe:
        safe = "modelo"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{safe}__{ts}"

    src_bundle = meta_dir / "knn_bundle.npz"
    if not src_bundle.exists():
        raise FileNotFoundError("knn_bundle.npz não encontrado para snapshot.")
    dst_bundle = models_dir / f"{stem}__knn_bundle.npz"
    shutil.copy2(src_bundle, dst_bundle)

    out: Dict[str, str] = {"bundle": str(dst_bundle)}

    if is_supcon:
        src_head = meta_dir / "supcon_head.pt"
        src_head_cfg = meta_dir / "supcon_head_cfg.json"
        if src_head.exists() and src_head_cfg.exists():
            dst_head = models_dir / f"{stem}__supcon_head.pt"
            dst_head_cfg = models_dir / f"{stem}__supcon_head_cfg.json"
            shutil.copy2(src_head, dst_head)
            shutil.copy2(src_head_cfg, dst_head_cfg)
            out["head"] = str(dst_head)
            out["head_cfg"] = str(dst_head_cfg)

    return out
