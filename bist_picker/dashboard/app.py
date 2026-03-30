"""BIST Stock Picker — Streamlit Dashboard.

Main entry point for the visual dashboard.
Launch with: streamlit run bist_picker/dashboard/app.py

Uses st.navigation for multi-page layout with sidebar navigation.
"""

import streamlit as st


@st.cache_resource
def _bootstrap_runtime() -> None:
    """Ensure DB tables and runtime indexes exist once per Streamlit process."""
    from bist_picker.db.connection import ensure_runtime_db_ready

    ensure_runtime_db_ready()


def main():
    """Configure and run the Streamlit dashboard."""
    st.set_page_config(
        page_title="BIST Stock Picker",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _bootstrap_runtime()

    # -- Navigation --
    # Backtesting was intentionally removed from the dashboard per user request.
    # Keep the old page disabled unless the user explicitly asks for it again.
    pages = {
        "Ana Sayfa": _page_ana_sayfa,
        "Skorlama": _page_skorlama,
        "Hisse Detay": _page_hisse_detay,
        "Pipeline": _page_pipeline,
    }

    page_list = list(pages.keys())

    # Support programmatic navigation via session_state (e.g. from skorlama)
    nav_index = 0
    if "nav_page" in st.session_state:
        target = st.session_state.pop("nav_page")
        if target in page_list:
            nav_index = page_list.index(target)

    with st.sidebar:
        st.title("📊 BIST Picker")
        st.caption("Buffett-style fundamental analysis")
        st.divider()
        selected = st.radio(
            "Sayfa", page_list, index=nav_index, label_visibility="collapsed"
        )

    # Render selected page
    pages[selected]()


def _page_ana_sayfa():
    from bist_picker.dashboard.pages.ana_sayfa import render
    render()


def _page_skorlama():
    from bist_picker.dashboard.pages.skorlama import render
    render()


def _page_hisse_detay():
    from bist_picker.dashboard.pages.hisse_detay import render
    render()


def _page_pipeline():
    from bist_picker.dashboard.pages.pipeline import render
    render()


# def _page_backtest():
#     from bist_picker.dashboard.pages.backtest import render
#     render()
#
# Backtesting is intentionally disabled. The user no longer wants it to appear
# in the Streamlit software, so the page stays commented out on purpose.


if __name__ == "__main__":
    main()
