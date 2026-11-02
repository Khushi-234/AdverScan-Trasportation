from pathlib import Path
import streamlit as st

# Resolve styles directory relative to this file
STYLES_DIR = Path(__file__).resolve().parents[1] / "styles"


def load_theme(theme: str = "Dark"):
    """
    Load the selected theme and shared base styles.
    Theme CSS must load before base.css.
    """

    theme_files = {
        "Dark": "dark.css",
        "Light": "light.css",
    }

    normalized = theme.capitalize() if isinstance(theme, str) else "Dark"
    theme_file = theme_files.get(normalized, "dark.css")

    theme_path = STYLES_DIR / theme_file
    base_path = STYLES_DIR / "base.css"

    if not theme_path.exists():
        raise FileNotFoundError(f"Theme file not found: {theme_path}")

    if not base_path.exists():
        raise FileNotFoundError(f"Base CSS not found: {base_path}")

    theme_css = theme_path.read_text(encoding="utf-8")
    base_css = base_path.read_text(encoding="utf-8")

    st.markdown(
        f"""
        <style>
        {theme_css}

        {base_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_theme():
    """
    Initialize theme selection and loading.
    Checks native Streamlit theme context, session state, and renders the right-side theme selector.
    Removes theme selection from the left sidebar to prevent conflicts.
    """
    # Detect native Streamlit theme from st.context if selected via native right-side menu
    native_theme = None
    if hasattr(st, "context") and hasattr(st.context, "theme"):
        ctx_theme = getattr(st.context, "theme", None)
        if isinstance(ctx_theme, dict) and ctx_theme.get("type"):
            native_theme = "Light" if ctx_theme["type"].lower() == "light" else "Dark"

    if "theme" not in st.session_state:
        st.session_state["theme"] = native_theme or "Dark"
    elif native_theme and native_theme != st.session_state.get("_last_native_theme"):
        st.session_state["theme"] = native_theme

    if native_theme:
        st.session_state["_last_native_theme"] = native_theme

    current_theme = st.session_state.get("theme", "Dark")

    # Render right-aligned theme selector in the top-right header area
    _, col_theme = st.columns([5.8, 1.2])
    with col_theme:
        selected_theme = st.selectbox(
            "Interface Theme",
            options=["Dark", "Light"],
            index=0 if current_theme == "Dark" else 1,
            key="interface_theme",
            label_visibility="collapsed",
            help="Switch Interface Theme (Dark Cybersecurity / Light Modern Dashboard)",
        )

    st.session_state["theme"] = selected_theme
    load_theme(selected_theme)

    return selected_theme
