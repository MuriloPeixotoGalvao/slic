import sys
from pathlib import Path
import streamlit as st

# garante que a raiz do projeto está no sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ✅ tem que ser o primeiro comando streamlit da página
st.set_page_config(page_title="SLIC rotulador + SupCon + kNN", layout="wide")


from slic_app.app import run_slic_labeler

run_slic_labeler()

