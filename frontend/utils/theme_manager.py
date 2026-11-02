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


def initialize_theme() -> str:
    """
    Initialize theme loading synchronized with native Streamlit theme selection.
    Respects the theme chosen in the native 3-dots (⋮) Settings menu (Light / Dark).
    Eliminates redundant on-page theme selector dropdowns.
    """
    # 1. Detect native Streamlit theme selected from the 3-dots menu (Settings -> Theme)
    native_theme = None
    if hasattr(st, "context") and hasattr(st.context, "theme"):
        ctx_theme = getattr(st.context, "theme", None)
        if isinstance(ctx_theme, dict) or hasattr(ctx_theme, "get"):
            theme_val = ctx_theme.get("type") or ctx_theme.get("base")
            if theme_val:
                native_theme = "Light" if str(theme_val).lower() == "light" else "Dark"

    # 2. Resolve active theme
    if native_theme:
        theme = native_theme
    else:
        theme = st.session_state.get("theme", "Dark")

    st.session_state["theme"] = theme
    load_theme(theme)
    return theme
