from __future__ import annotations


def check_optional_deps():
    # clique
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates  # noqa
        has_img_coord = True
    except Exception:
        has_img_coord = False

    # canvas polígono
    try:
        from streamlit_drawable_canvas import st_canvas  # noqa
        has_canvas = True
    except Exception:
        has_canvas = False

    # tkinter folder picker
    try:
        import tkinter as _tk  # noqa
        from tkinter import filedialog as _fd  # noqa
        tk_ok = True
    except Exception:
        tk_ok = False

    return dict(HAS_IMG_COORD=has_img_coord, HAS_CANVAS=has_canvas, TK_OK=tk_ok)
