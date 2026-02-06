# slic_app/app.py
from __future__ import annotations

import os
from pathlib import Path

# threads/seed — ideal setar antes de numpy/pytorch pesado
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
np.random.seed(0)

import streamlit as st


def run_slic_labeler() -> None:
    """
    Entry-point do app SLIC.

    Regras:
    - NÃO chame st.set_page_config aqui (fica no arquivo em pages/)
    - Aplique o patch do canvas ANTES de importar módulos que dependem dele
    """

    # --- 1) Patch do streamlit-drawable-canvas (precisa vir cedo)
    from slic_app.compat import patch_streamlit_drawable_canvas_compat
    patch_streamlit_drawable_canvas_compat()

    # --- 2) Imports do app (depois do patch)
    try:
        from slic_app.constants import CSS
        from slic_app.state import init_session_state
        from slic_app.sidebar import render_sidebar
        from slic_app.pages_label_train import render_label_train
        from slic_app.pages_predict_upload import render_predict_upload
        from slic_app.ui_pick import pick_folder_button
    except Exception as e:
        st.error("❌ Erro ao importar módulos do app. Veja detalhes abaixo:")
        st.exception(e)
        st.stop()

    # --- 3) UI base
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<div class='app-title'>SLIC rotulador + SupCon + kNN</div>", unsafe_allow_html=True)

    # --- 4) Base state mínimo (NÃO reseta índices/undo/redo)
    # (o reset controlado vai acontecer só quando base_dir mudar)
    init_session_state()

    # --- 5) Base_dir: escolher ANTES do resto
    with st.sidebar:
        st.markdown("### 📁 Pasta do projeto (base_dir_str)")

        st.text_input(
            "Caminho da pasta",
            key="base_dir_str",
            placeholder=r"ex: E:\DADOS_SUPERPIXELS",
        )

        # ✅ Não passe help_text se sua função não aceita
        pick_folder_button(
            "base_dir_str",
            label="📂 Escolher pasta do projeto…",
            initialdir="E:/",
        )
        st.caption("Dica: selecione a pasta raiz do projeto (onde ficam previews/ e meta/).")

        cur = (st.session_state.get("base_dir_str") or "").strip()
        if cur:
            st.code(cur)

    base_dir_str = (st.session_state.get("base_dir_str") or "").strip()
    if not base_dir_str:
        st.warning("Selecione uma pasta (base_dir_str) na barra lateral.")
        st.stop()

    base_dir = Path(base_dir_str)
    if not base_dir.exists():
        st.error(f"Diretório não existe: {base_dir_str}")
        st.stop()

    # ✅ 6) Agora sim: “armar” o estado do projeto (reseta caches/undo/redo só se base_dir mudou)
    init_session_state(base_dir=base_dir)

    # Sidebar original (parâmetros, etc.)
    cfg = render_sidebar()

    # Forçar cfg.base_dir vindo do botão/entrada
    cfg.base_dir = base_dir  # ✅ garante que TODAS as páginas usem a pasta escolhida

    # --- 7) Roteamento de páginas
    page = st.session_state.get("page", "Rotular/Treinar")
    if page == "Rotular/Treinar":
        render_label_train(cfg)
    else:
        render_predict_upload(cfg)


def main() -> None:
    st.set_page_config(page_title="SLIC rotulador + SupCon + kNN", layout="wide")
    run_slic_labeler()


if __name__ == "__main__":
    main()
