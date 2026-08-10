import streamlit as st
import streamlit_authenticator as stauth
from supabase import create_client


def get_supabase_client():

    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]

    return create_client(url, key)


def fetch_credentials():
    """
    Pulls all users from Supabase and builds the credentials dict
    shape streamlit-authenticator expects.
    """

    client = get_supabase_client()

    response = client.table("users").select("*").execute()

    usernames = {}

    for user in response.data:
        usernames[user["username"]] = {
            "email": user["email"],
            "name": user["name"],
            "password": user["password_hash"]
        }

    return {"usernames": usernames}


def register_user(username, email, name, password):
    """
    Returns (success: bool, message: str).
    """

    client = get_supabase_client()

    existing = client.table("users").select("username").eq(
        "username", username
    ).execute()

    if existing.data:
        return False, "That username is already taken."

    hasher = stauth.Hasher()
    hashed_password = hasher.hash(password)

    try:
        client.table("users").insert({
            "username": username,
            "email": email,
            "name": name,
            "password_hash": hashed_password
        }).execute()
    except Exception as error:
        return False, f"Could not create account: {error}"

    return True, "Account created. You can now log in."


def build_authenticator():

    credentials = fetch_credentials()

    return stauth.Authenticate(
        credentials,
        "bioai_cookie",
        st.secrets["COOKIE_SIGNATURE_KEY"],
        cookie_expiry_days=7,
        auto_hash=False
    )