"""Ana Sayfa — Portfolio summary, open positions, P&L, macro regime.

Displays:
- KPI metric cards (total return, win rate, active positions)
- Open positions table with P&L coloring
- Portfolio History table (closed trades)
- Macro regime panel (CPI, interest rate, USD/TRY, RISK_ON/OFF)
"""

import streamlit as st
import pandas as pd

from bist_picker.dashboard.data_access import (
    get_all_portfolio_performance,
    get_latest_macro,
    get_open_positions,
    get_portfolio_history,
)
from bist_picker.dashboard.style import (
    fmt_pct,
    fmt_price,
    pnl_color,
    regime_emoji,
    risk_color,
    score_color,
    style_dataframe,
    GREEN,
    RED,
    YELLOW,
    DIM,
    BLUE,
)


def _macro_metric_help(field_date, latest_date):
    """Show when a metric is filled from the latest available prior date."""
    if field_date is None or latest_date is None or field_date == latest_date:
        return None
    return f"Son dolu veri tarihi: {field_date}"


def render():
    """Render the Ana Sayfa (home page)."""
    st.header("Portfolyo Ozeti")

    # -- KPI Cards --
    perf = get_all_portfolio_performance()
    has_perf = bool(perf)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_ret = perf.get("total_return_avg") if has_perf else None
        st.metric(
            label="Toplam Getiri (Ort.)",
            value=fmt_pct(total_ret),
        )
    with col2:
        active_ret = perf.get("active_return_avg") if has_perf else None
        st.metric(
            label="Aktif Getiri (Ort.)",
            value=fmt_pct(active_ret),
        )
    with col3:
        win_rate = perf.get("win_rate") if has_perf else None
        st.metric(
            label="Kazanma Orani",
            value=fmt_pct(win_rate),
            help=(
                "Pozitif getiride olan secimlerin oranidir. "
                "Acik pozisyonlar da guncel fiyata gore hesaba katilir."
            ),
        )
    with col4:
        positions_df = get_open_positions()
        if positions_df is None:
            positions_df = pd.DataFrame()
        st.metric(
            label="Aktif Pozisyon",
            value=str(len(positions_df)),
        )

    if not has_perf:
        st.caption("Performans verisi bulunamadi.")

    st.divider()

    # -- Open Positions Table --
    st.subheader("Aktif Portfolyo (Acik Pozisyonlar)")
    
    if positions_df.empty:
        st.info("Su an acik bir pozisyon bulunmuyor. 'Score' ve 'Pick' islemlerini calistirin.")
    else:
        # Display as a clean table
        display_df = positions_df[[
            "portfolio", "ticker", "name", "entry_price", "current_price",
            "pnl_pct", "selection_date"
        ]].copy()
        
        display_df.columns = [
            "Portfolyo", "Hisse", "Sirket", "Giris", "Guncel", "Getiri %", "Alim Tarihi"
        ]
        
        # Style and format
        styler = style_dataframe(display_df, pnl_columns=["Getiri %"])
        st.dataframe(styler, use_container_width=True, hide_index=True)

    st.divider()

    # -- Portfolio History Table --
    st.subheader("Portfolyo Gecmisi (Kapatilmis Islemler)")
    history_df = get_portfolio_history()
    
    if history_df.empty:
        st.caption("Henuz kapatilmis bir islem bulunmuyor.")
    else:
        hist_display = history_df[[
            "portfolio", "ticker", "selection_date", "exit_date",
            "entry_price", "exit_price", "pnl_pct"
        ]].copy()
        
        hist_display.columns = [
            "Portfolyo", "Hisse", "Alim", "Satim", "Giris", "Cikis", "Kar/Zarar %"
        ]
        
        st.dataframe(
            style_dataframe(hist_display, pnl_columns=["Kar/Zarar %"]),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # -- Macro Regime Panel --
    st.subheader("Ekonomik Gostergeler (Makro)")

    macro = get_latest_macro()
    if macro is None:
        st.info("Makro veri bulunamadi.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Fix decimal formatting for CPI and Policy Rate
            cpi = macro.get("cpi_yoy_pct")
            if cpi and cpi < 1.0: cpi *= 100.0
            st.metric(
                "Enflasyon (YoY)",
                fmt_pct(cpi),
                help=_macro_metric_help(macro.get("cpi_yoy_date"), macro.get("date")),
            )
        with col2:
            policy = macro.get("policy_rate_pct")
            if policy and policy < 1.0: policy *= 100.0
            st.metric(
                "Politika Faizi",
                fmt_pct(policy),
                help=_macro_metric_help(
                    macro.get("policy_rate_date"), macro.get("date")
                ),
            )
        with col3:
            usdtry = macro.get("usdtry_rate")
            st.metric(
                "USD/TRY",
                f"{usdtry:.2f}" if usdtry is not None else "--",
                help=_macro_metric_help(macro.get("usdtry_date"), macro.get("date")),
            )
        with col4:
            regime = macro.get("regime", "--")
            emoji = regime_emoji(regime)
            st.metric(
                "Piyasa Rejimi",
                f"{emoji} {regime}",
                help=_macro_metric_help(macro.get("regime_date"), macro.get("date")),
            )

        st.caption(f"Veri Guncelleme: {macro.get('date', '--')}")
