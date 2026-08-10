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

from modules.auth import build_authenticator, register_user

st.set_page_config(
    page_title="BioAI",
    page_icon="🧬",
    layout="wide"
)

authenticator = build_authenticator()

authenticator.login(location="main")

if st.session_state.get("authentication_status") is False:

    st.error("Username or password is incorrect.")

elif st.session_state.get("authentication_status") is None:

    st.title("BioAI")
    st.info("Log in above, or create a new account below.")

    with st.expander("Create a new account"):

        new_username = st.text_input("Username", key="reg_username")
        new_email = st.text_input("Email", key="reg_email")
        new_name = st.text_input("Full Name", key="reg_name")
        new_password = st.text_input("Password", type="password", key="reg_password")

        if st.button("Register"):

            if not (new_username and new_email and new_name and new_password):
                st.warning("Please fill in every field.")
            else:
                success, message = register_user(
                    new_username, new_email, new_name, new_password
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.stop()

elif st.session_state.get("authentication_status"):

    with st.sidebar:
        st.write(f"Logged in as **{st.session_state['name']}**")
        authenticator.logout(location="sidebar")

    st.title("BioAI")
    st.markdown(
        """
        # Welcome to BioAI

        Choose a module from the left sidebar.
        """
    )