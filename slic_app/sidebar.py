from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from .persistence import save_config, clear_config, load_config

@dataclass
class Cfg:
    base_dir: Path
    slic_params: tuple[int, float, float, bool]
    overlay_alpha: float
    show_boundaries: bool
    boundary_thick_px: int
    img_width: int

    expand_on_click: bool
    expand_mode: str
    dist_thr: float
    max_region: int
    min_frac: float

    train_mode: str
    k_neighbors: int
    balance_knn: bool
    apply_preds_only_unlabeled: bool
    use_model_embedding_for_click: bool

    supcon_epochs: int
    supcon_lr: float
    supcon_temp: float

    deps: dict

def _has_img_coord() -> bool:
    try:
        import streamlit_image_coordinates  # noqa
        return True
    except Exception:
        return False

def _has_canvas() -> bool:
    try:
        import streamlit_drawable_canvas  # noqa
        return True
    except Exception:
        return False

def render_sidebar() -> Cfg:
    st.sidebar.header("🧩 SLIC Labeler")

    # base dir (widget lives in app.py to avoid duplicate key)
    base_dir_str = st.session_state.get("base_dir_str", "")
    base_dir = Path(base_dir_str).expanduser()
    app_root = Path(__file__).resolve().parents[1]

    # carregar config do disco antes de criar widgets
    cfg_disk = load_config(base_dir) if (base_dir_str and base_dir.exists()) else {}
    if not cfg_disk:
        # fallback: config global do app
        cfg_disk = load_config(app_root)
    cfg_loaded_key = "_cfg_loaded_for"
    cur_cfg_scope = str(base_dir.resolve()) if (base_dir_str and base_dir.exists()) else ""

    # aplica config do disco uma única vez por base_dir (evita sobrescrever em rerun)
    if cur_cfg_scope and st.session_state.get(cfg_loaded_key) != cur_cfg_scope:
        for k, v in cfg_disk.items():
            if k.startswith("cfg_"):
                st.session_state[k] = v
        st.session_state[cfg_loaded_key] = cur_cfg_scope

    # page (carrega do config antes do widget)
    st.session_state.setdefault("page", str(cfg_disk.get("page", "Rotular/Treinar")))
    page = st.sidebar.radio("Página", ["Rotular/Treinar", "Predição (upload)"], key="page")

    # restore defaults (clear config.json)
    if base_dir_str and base_dir.exists():
        if st.sidebar.button("Restaurar defaults", key="btn_reset_cfg", use_container_width=True):
            clear_config(base_dir)
            for k in list(st.session_state.keys()):
                if k.startswith("cfg_") or k == "page":
                    st.session_state.pop(k, None)
            st.rerun()

    # defaults locais (evita warning de value + session_state)
    st.session_state.setdefault("cfg_n_segments", int(cfg_disk.get("cfg_n_segments", 2000)))
    st.session_state.setdefault("cfg_compactness", float(cfg_disk.get("cfg_compactness", 12.0)))
    st.session_state.setdefault("cfg_sigma", float(cfg_disk.get("cfg_sigma", 0.0)))
    st.session_state.setdefault("cfg_enforce_connectivity", bool(cfg_disk.get("cfg_enforce_connectivity", True)))
    st.session_state.setdefault("cfg_img_width", int(cfg_disk.get("cfg_img_width", 900)))
    st.session_state.setdefault("cfg_overlay_alpha", float(cfg_disk.get("cfg_overlay_alpha", 0.50)))
    st.session_state.setdefault("cfg_show_boundaries", bool(cfg_disk.get("cfg_show_boundaries", True)))
    st.session_state.setdefault("cfg_boundary_thick_px", int(cfg_disk.get("cfg_boundary_thick_px", 1)))
    st.session_state.setdefault("cfg_expand_on_click", bool(cfg_disk.get("cfg_expand_on_click", True)))
    st.session_state.setdefault("cfg_expand_mode", str(cfg_disk.get("cfg_expand_mode", "contíguo")))
    st.session_state.setdefault("cfg_dist_thr", float(cfg_disk.get("cfg_dist_thr", 1.5)))
    st.session_state.setdefault("cfg_max_region", int(cfg_disk.get("cfg_max_region", 80)))
    st.session_state.setdefault("cfg_train_mode", str(cfg_disk.get("cfg_train_mode", "kNN")))
    st.session_state.setdefault("cfg_k_neighbors", int(cfg_disk.get("cfg_k_neighbors", 7)))
    st.session_state.setdefault("cfg_balance_knn", bool(cfg_disk.get("cfg_balance_knn", True)))
    st.session_state.setdefault("cfg_min_frac", float(cfg_disk.get("cfg_min_frac", 0.6)))
    st.session_state.setdefault("cfg_apply_preds_only_unlabeled", bool(cfg_disk.get("cfg_apply_preds_only_unlabeled", True)))
    st.session_state.setdefault("cfg_use_model_embedding_for_click", bool(cfg_disk.get("cfg_use_model_embedding_for_click", True)))
    st.session_state.setdefault("cfg_supcon_epochs", int(cfg_disk.get("cfg_supcon_epochs", 200)))
    st.session_state.setdefault("cfg_supcon_lr", float(cfg_disk.get("cfg_supcon_lr", 1e-3)))
    st.session_state.setdefault("cfg_supcon_temp", float(cfg_disk.get("cfg_supcon_temp", 0.1)))
    st.session_state.setdefault("cfg_stroke_color", str(cfg_disk.get("cfg_stroke_color", "#ff2d2d")))

    # slic params
    st.sidebar.subheader("SLIC")
    n_segments = st.sidebar.slider("n_segments", 50, 2000, step=10, key="cfg_n_segments")
    compactness = st.sidebar.slider("compactness", 1.0, 40.0, step=0.5, key="cfg_compactness")
    sigma = st.sidebar.slider("sigma", 0.0, 5.0, step=0.5, key="cfg_sigma")
    enforce_conn = st.sidebar.checkbox("enforce_connectivity", key="cfg_enforce_connectivity")

    # vis
    st.sidebar.subheader("Visual")
    img_width = st.sidebar.slider("Largura (px)", 400, 2000, step=50, key="cfg_img_width")
    overlay_alpha = st.sidebar.slider("Opacidade overlay", 0.0, 1.0, step=0.05, key="cfg_overlay_alpha")
    show_boundaries = st.sidebar.checkbox("Mostrar bordas (superpixel)", key="cfg_show_boundaries")
    boundary_thick_px = st.sidebar.slider("Espessura borda (px)", 1, 6, step=1, key="cfg_boundary_thick_px")

    # click expansion
    st.sidebar.subheader("Clique → expansão")
    expand_on_click = st.sidebar.checkbox("Expandir por similaridade", key="cfg_expand_on_click")
    expand_mode = st.sidebar.selectbox("Modo", ["contíguo", "global"], key="cfg_expand_mode")
    dist_thr = st.sidebar.slider("dist_thr", 0.1, 6.0, step=0.1, key="cfg_dist_thr")
    max_region = st.sidebar.slider("max_region", 1, 500, step=5, key="cfg_max_region")

    # treino
    st.sidebar.subheader("Treino / kNN")
    train_mode = st.sidebar.selectbox("Modo de treino", ["kNN", "SupCon+kNN"], key="cfg_train_mode")
    k_neighbors = st.sidebar.slider("k (vizinhos)", 1, 50, step=1, key="cfg_k_neighbors")
    balance_knn = st.sidebar.checkbox("Balancear voto (anti-desbalanceamento)", key="cfg_balance_knn")
    min_frac = st.sidebar.slider("min_frac (maioria p/ SPX)", 0.1, 1.0, step=0.05, key="cfg_min_frac")
    apply_preds_only_unlabeled = st.sidebar.checkbox("Predição só em UNLAB", key="cfg_apply_preds_only_unlabeled")
    use_model_embedding_for_click = st.sidebar.checkbox("Clique usa embedding SupCon (se existir)", key="cfg_use_model_embedding_for_click")

    # supcon params
    st.sidebar.subheader("SupCon")
    supcon_epochs = st.sidebar.slider("epochs", 50, 1000, step=25, key="cfg_supcon_epochs")
    supcon_lr = st.sidebar.select_slider("lr", options=[1e-4, 3e-4, 1e-3, 3e-3], key="cfg_supcon_lr")
    supcon_temp = st.sidebar.slider("temperature", 0.02, 0.5, step=0.02, key="cfg_supcon_temp")

    deps = {"HAS_IMG_COORD": _has_img_coord(), "HAS_CANVAS": _has_canvas()}

    cfg = Cfg(
        base_dir=base_dir,
        slic_params=(int(n_segments), float(compactness), float(sigma), bool(enforce_conn)),
        overlay_alpha=float(overlay_alpha),
        show_boundaries=bool(show_boundaries),
        boundary_thick_px=int(boundary_thick_px),
        img_width=int(img_width),

        expand_on_click=bool(expand_on_click),
        expand_mode=str(expand_mode),
        dist_thr=float(dist_thr),
        max_region=int(max_region),
        min_frac=float(min_frac),

        train_mode=str(train_mode),
        k_neighbors=int(k_neighbors),
        balance_knn=bool(balance_knn),
        apply_preds_only_unlabeled=bool(apply_preds_only_unlabeled),
        use_model_embedding_for_click=bool(use_model_embedding_for_click),

        supcon_epochs=int(supcon_epochs),
        supcon_lr=float(supcon_lr),
        supcon_temp=float(supcon_temp),

        deps=deps,
    )

    # Persistir configuração (se base_dir válido)
    if base_dir_str and base_dir.exists():
        save_config(base_dir, {
            "cfg_n_segments": int(n_segments),
            "cfg_compactness": float(compactness),
            "cfg_sigma": float(sigma),
            "cfg_enforce_connectivity": bool(enforce_conn),
            "cfg_img_width": int(img_width),
            "cfg_overlay_alpha": float(overlay_alpha),
            "cfg_show_boundaries": bool(show_boundaries),
            "cfg_boundary_thick_px": int(boundary_thick_px),
            "cfg_expand_on_click": bool(expand_on_click),
            "cfg_expand_mode": str(expand_mode),
            "cfg_dist_thr": float(dist_thr),
            "cfg_max_region": int(max_region),
            "cfg_train_mode": str(train_mode),
            "cfg_k_neighbors": int(k_neighbors),
            "cfg_balance_knn": bool(balance_knn),
            "cfg_min_frac": float(min_frac),
            "cfg_apply_preds_only_unlabeled": bool(apply_preds_only_unlabeled),
            "cfg_use_model_embedding_for_click": bool(use_model_embedding_for_click),
            "cfg_supcon_epochs": int(supcon_epochs),
            "cfg_supcon_lr": float(supcon_lr),
            "cfg_supcon_temp": float(supcon_temp),
            "cfg_stroke_color": str(st.session_state.get("cfg_stroke_color", "#ff2d2d")),
            "page": str(page),
        })
    # também salva no config global do app
    save_config(app_root, {
        "cfg_n_segments": int(n_segments),
        "cfg_compactness": float(compactness),
        "cfg_sigma": float(sigma),
        "cfg_enforce_connectivity": bool(enforce_conn),
        "cfg_img_width": int(img_width),
        "cfg_overlay_alpha": float(overlay_alpha),
        "cfg_show_boundaries": bool(show_boundaries),
        "cfg_boundary_thick_px": int(boundary_thick_px),
        "cfg_expand_on_click": bool(expand_on_click),
        "cfg_expand_mode": str(expand_mode),
        "cfg_dist_thr": float(dist_thr),
        "cfg_max_region": int(max_region),
        "cfg_train_mode": str(train_mode),
        "cfg_k_neighbors": int(k_neighbors),
        "cfg_balance_knn": bool(balance_knn),
        "cfg_min_frac": float(min_frac),
        "cfg_apply_preds_only_unlabeled": bool(apply_preds_only_unlabeled),
        "cfg_use_model_embedding_for_click": bool(use_model_embedding_for_click),
        "cfg_supcon_epochs": int(supcon_epochs),
        "cfg_supcon_lr": float(supcon_lr),
        "cfg_supcon_temp": float(supcon_temp),
        "cfg_stroke_color": str(st.session_state.get("cfg_stroke_color", "#ff2d2d")),
        "page": str(page),
    })

    return cfg
