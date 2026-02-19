# slic_app/state.py
from __future__ import annotations

from pathlib import Path
import streamlit as st
from .persistence import load_classes


def init_session_state(base_dir: Path | None = None) -> None:
    """
    Inicializa estado sem resetar tudo a cada rerun.
    - Usa setdefault para não sobrescrever.
    - Se base_dir for informado e mudar, reseta apenas o que é do projeto.
    """

    # --- defaults globais (não resetam)
    st.session_state.setdefault("page", "Rotular/Treinar")

    st.session_state.setdefault("base_dir", "")
    st.session_state.setdefault("_base_dir_prev", "")

    st.session_state.setdefault("train_img_idx", 0)
    st.session_state.setdefault("pred_img_idx", 0)

    st.session_state.setdefault("classes", ["classe_0"])
    st.session_state.setdefault("active_class", 0)

    st.session_state.setdefault("labelmaps_cache", {})
    st.session_state.setdefault("pred_cache", {})
    st.session_state.setdefault("emb_cache", {})

    st.session_state.setdefault("undo", {})
    st.session_state.setdefault("redo", {})

    st.session_state.setdefault("last_processed_click", None)

    # --- se não veio base_dir, para aqui
    if base_dir is None:
        return

    cur = str(base_dir.resolve())
    prev = str(st.session_state.get("_base_dir_prev") or "")

    # ✅ Só faz reset quando muda a pasta do projeto
    if cur != prev:
        # carrega classes persistidas no disco (se existirem)
        try:
            st.session_state["classes"] = load_classes(Path(base_dir))
            # mantém active_class válido
            if not (0 <= int(st.session_state.get("active_class", 0)) < len(st.session_state["classes"])):
                st.session_state["active_class"] = 0
        except Exception:
            pass

        # garante cor de linha padrão
        st.session_state.setdefault("cfg_stroke_color", "#ff0000")

        # reseta caches/undo/redo e volta para a primeira imagem
        st.session_state["train_img_idx"] = 0
        st.session_state["pred_img_idx"] = 0
        st.session_state["labelmaps_cache"] = {}
        st.session_state["pred_cache"] = {}
        st.session_state["emb_cache"] = {}
        st.session_state["undo"] = {}
        st.session_state["redo"] = {}
        st.session_state["last_processed_click"] = None

        st.session_state["_base_dir_prev"] = cur
