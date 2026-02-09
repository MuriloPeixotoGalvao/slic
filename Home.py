# Home.py
#teste
import base64
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Suite – Geoflora", layout="wide")

def set_bg_image(path: str, opacity: float = 0.12, blur_px: int = 0):
    p = Path(path)
    if not p.exists():
        st.error(f"Imagem de fundo não encontrada: {p}")
        return

    data = p.read_bytes()
    b64 = base64.b64encode(data).decode()
    ext = p.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"

    # fundo (com opcional blur leve)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(255,255,255,{1-opacity}), rgba(255,255,255,{1-opacity})),
                url("data:{mime};base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* melhora a leitura do texto */
        .block-container {{
            background: rgba(255,255,255,0.78);
            border-radius: 18px;
            padding: 24px 28px;
            box-shadow: 0 6px 22px rgba(0,0,0,0.08);
            backdrop-filter: blur({blur_px}px);
            -webkit-backdrop-filter: blur({blur_px}px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ✅ fundo (use o caminho certo)
set_bg_image(r"C:\Users\mkale\Downloads\simids.jpg", opacity=0.9, blur_px=00)

st.title("Laboratório de Agricultura Digital da Amazônia - LADA")
st.subheader("⚙️📦 Suíte de Curadoria e Pipeline de Dados 🌿🌱🍃🖼️🛰️")
st.write(
    "Use o menu no canto esquerdo para abrir cada ferramenta:\n"
    "- **Organizador de Imagens**\n"
    "- **Cadastrar Espécies ao Banco de dados**\n"
    "- **Abrir labelImg**\n"
    "- **Relatórios**\n"
)

# opcional: se quiser manter a imagem também como elemento na página, pode deixar.
# se preferir só como fundo, remova esta linha:
# st.image(r"C:\Users\mkale\Downloads\simids.jpg", caption="SIMIDS", use_container_width=True)
