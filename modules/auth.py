import os
import streamlit as st
from supabase import create_client


def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key)


def get_supabase_client():
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY")
    return create_client(url, key)


def sign_up(email, password, full_name):

    client = get_supabase_client()

    try:
        response = client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"full_name": full_name}}
        })
    except Exception as error:
        return False, f"Could not create account: {error}"

    if response.user is None:
        return False, "Could not create account. Please try again."

    return True, (
        "Account created — check your email for a confirmation "
        "link before logging in."
    )


def sign_in(email, password):

    client = get_supabase_client()

    try:
        response = client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
    except Exception:
        return False, None, (
            "Incorrect email or password, or your email hasn't "
            "been confirmed yet — check your inbox."
        )

    if response.user is None:
        return False, None, "Incorrect email or password."

    user_info = {
        "id": response.user.id,
        "email": response.user.email,
        "full_name": (response.user.user_metadata or {}).get(
            "full_name", response.user.email
        )
    }

    return True, user_info, None


def sign_out():
    client = get_supabase_client()
    try:
        client.auth.sign_out()
    except Exception:
        pass


def require_login():
    """
    Call this as the VERY FIRST thing in every page file. Blocks
    all page content until the user is logged in.
    """

    if st.session_state.get("logged_in_user") is not None:
        return

    st.title("BioAI")

    tab_login, tab_signup = st.tabs(["Log In", "Create Account"])

    with tab_login:

        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input(
            "Password", type="password", key="login_password"
        )

        if st.button("Log In", key="login_button"):

            success, user_info, error = sign_in(login_email, login_password)

            if success:
                st.session_state.logged_in_user = user_info
                st.rerun()
            else:
                st.error(error)

    with tab_signup:

        signup_name = st.text_input("Full Name", key="signup_name")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input(
            "Password (at least 6 characters)",
            type="password", key="signup_password"
        )

        if st.button("Create Account", key="signup_button"):

            if not (signup_name and signup_email and signup_password):
                st.warning("Please fill in every field.")
            elif len(signup_password) < 6:
                st.warning("Password must be at least 6 characters.")
            else:
                success, message = sign_up(
                    signup_email, signup_password, signup_name
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.stop()