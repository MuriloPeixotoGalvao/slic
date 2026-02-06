# slic_app/ui_pick.py
from __future__ import annotations

from typing import Optional
import streamlit as st


def pick_folder_button(
    target_key: str,
    *,
    label: str = "📂 Escolher diretório…",
    initialdir: str = "C:/",
    help: Optional[str] = None,
    help_text: Optional[str] = None,
    **_ignore,
) -> None:
    """
    Abre um seletor nativo de pastas (Windows) e grava o caminho em st.session_state[target_key].

    - Aceita help OU help_text (compatível com seu código atual).
    - Evita StreamlitAPIException usando callback on_click.
    """
    _help = help if help is not None else help_text

    def _pick() -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass

            folder = filedialog.askdirectory(initialdir=initialdir, mustexist=True)

            try:
                root.destroy()
            except Exception:
                pass

            if folder:
                st.session_state[target_key] = folder  # ✅ OK dentro do callback
        except Exception as e:
            st.session_state[f"{target_key}__pick_error"] = str(e)

    st.button(label, key=f"{target_key}__pickbtn", help=_help, on_click=_pick)

    err = st.session_state.pop(f"{target_key}__pick_error", None)
    if err:
        st.error(f"Falha ao abrir seletor de pasta: {err}")
