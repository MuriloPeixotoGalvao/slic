from __future__ import annotations
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

UNLAB = np.uint16(65535)

PALETTE = [
    (66, 135, 245), (76, 175, 80), (255, 193, 7), (156, 39, 176), (244, 67, 54),
    (0, 188, 212), (233, 30, 99), (0, 150, 136), (121, 85, 72), (158, 158, 158),
    (255, 152, 0), (33, 150, 243), (63, 81, 181), (139, 195, 74), (103, 58, 183),
    (205, 220, 57),
]
PALETTE_ARR = np.array(PALETTE, dtype=np.uint8)

CSS = """
<style>
div.block-container { padding-top: 0.8rem; padding-bottom: 1rem; }
.app-title{ font-size: 24px; font-weight: 800; margin: 0 0 0.45rem 0; line-height: 1.1; }
.step-title{ font-size: 18px; font-weight: 800; margin: 0.35rem 0 0.35rem 0; line-height: 1.15; }
.img-title{ font-size: 14px; font-weight: 700; margin: 0.2rem 0 0.35rem 0; opacity: 0.95; }
.subtle{ font-size: 13px; margin: 0.15rem 0 0.25rem 0; opacity: 0.9; }
.stColumn { overflow: hidden; }
.stColumn canvas, .stColumn img { max-width: 100% !important; height: auto !important; }
</style>
"""
