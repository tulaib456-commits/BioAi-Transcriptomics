import subprocess
import sys
import importlib
import streamlit as st

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "plotly": "plotly",
    "sklearn": "scikit-learn",
    "openpyxl": "openpyxl",
    "mygene": "mygene",
    "fpdf": "fpdf2",
    "kaleido": "kaleido",
    "gseapy": "gseapy",
}

def ensure_packages():
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pip_name]
                )
            except Exception:
                pass
    try:
        importlib.invalidate_caches()
    except Exception:
        pass

ensure_packages()

from modules.auth import require_login, sign_out

st.set_page_config(page_title="BioAI", page_icon="🧬", layout="wide")

require_login()

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.logged_in_user['full_name']}**")
    if st.button("Log Out"):
        sign_out()
        st.session_state.logged_in_user = None
        st.rerun()

st.title("BioAI")
st.markdown(
    """
    # Welcome to BioAI

    Choose a module from the left sidebar.
    """
)