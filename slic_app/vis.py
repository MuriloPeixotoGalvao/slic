from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2

from .constants import PALETTE_ARR, UNLAB
from .slic_ops import boundaries_mask

def overlay_from_labelmap_and_pred(
    rgb: np.ndarray,
    slic_labels: np.ndarray,
    labelmap_u16: np.ndarray,
    pred_classes: Optional[np.ndarray],
    n_classes: int,
    alpha: float,
    show_boundaries: bool,
    boundary_thick_px: int,
    apply_preds_only_unlabeled: bool = True,
) -> np.ndarray:
    H, W, _ = rgb.shape
    overlay = rgb.copy()

    lm = labelmap_u16
    m_man = (lm != UNLAB) & (lm < np.uint16(max(1, n_classes)))
    if np.any(m_man):
        cid = (lm[m_man].astype(np.int32) % len(PALETTE_ARR))
        overlay[m_man] = PALETTE_ARR[cid]

    if pred_classes is not None:
        pred_map = pred_classes[slic_labels]
        if apply_preds_only_unlabeled:
            m_pred = (lm == UNLAB)
        else:
            m_pred = np.ones((H, W), dtype=bool)
        if np.any(m_pred):
            cidp = (pred_map[m_pred].astype(np.int32) % len(PALETTE_ARR))
            overlay[m_pred] = PALETTE_ARR[cidp]

    out = cv2.addWeighted(overlay, float(alpha), rgb, 1.0 - float(alpha), 0)

    if show_boundaries and boundary_thick_px > 0:
        bnd = boundaries_mask(slic_labels)
        if boundary_thick_px == 1:
            out[bnd == 1] = (0, 0, 0)
        else:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (boundary_thick_px, boundary_thick_px))
            bnd2 = cv2.dilate(bnd, k, 1)
            out[bnd2 == 1] = (0, 0, 0)
    return out

def resize_for_view(img_rgb: np.ndarray, target_width: int, max_pixels: int = 6_000_000) -> Tuple[np.ndarray, float, float]:
    H, W = img_rgb.shape[:2]
    s = (target_width / float(W)) if target_width else 1.0
    new_w = max(1, int(round(W * s)))
    new_h = max(1, int(round(H * s)))

    if new_w * new_h > max_pixels:
        s2 = (max_pixels / float(new_w * new_h)) ** 0.5
        new_w = max(1, int(round(new_w * s2)))
        new_h = max(1, int(round(new_h * s2)))

    if new_w != W or new_h != H:
        interp = cv2.INTER_NEAREST if (new_w > W or new_h > H) else cv2.INTER_AREA
        disp = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)
    else:
        disp = img_rgb

    sx = W / float(disp.shape[1])
    sy = H / float(disp.shape[0])
    return disp, sx, sy

def curtain_merge(base_rgb: np.ndarray, overlay_rgb: np.ndarray, pos01: float,
                  show_line: bool = True, line_thick: int = 2,
                  left_is: str = "base") -> np.ndarray:
    assert base_rgb.shape == overlay_rgb.shape, "base e overlay precisam ter o mesmo shape"
    H, W, _ = base_rgb.shape
    pos01 = float(np.clip(pos01, 0.0, 1.0))
    cut = int(round(W * pos01))

    out = overlay_rgb.copy()
    if left_is == "base":
        out[:, :cut] = base_rgb[:, :cut]
    else:
        out[:, :cut] = overlay_rgb[:, :cut]
        out[:, cut:] = base_rgb[:, cut:]

    if show_line and 0 < cut < W:
        x0 = max(0, cut - line_thick // 2)
        x1 = min(W, cut + (line_thick - line_thick // 2))
        out[:, x0:x1] = (0, 0, 0)
    return out

def overlay_from_predmap(rgb: np.ndarray, predmap_u16: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = rgb.copy()
    cid = (predmap_u16.astype(np.int32) % len(PALETTE_ARR))
    overlay[:] = PALETTE_ARR[cid]
    out = cv2.addWeighted(overlay, float(alpha), rgb, 1.0 - float(alpha), 0)
    return out
