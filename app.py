import subprocess
import sys
import importlib

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
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name]
            )
    try:
        importlib.invalidate_caches()
    except Exception:
        pass

ensure_packages()

import streamlit as st


st.set_page_config(

    page_title="BioAI",

    page_icon="🧬",

    layout="wide"

)

st.title("BioAI")

st.markdown(

"""

# Welcome to BioAI

Choose a module from the left sidebar.

"""

)