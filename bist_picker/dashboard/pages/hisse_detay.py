"""Hisse Detay — Single stock deep-dive page.

Displays:
- Ticker selector (selectbox)
- Company info header
- Price chart (Plotly) with entry/target/stop horizontal lines
- 9-factor radar/spider chart (Plotly)
- Financial metrics table (from AdjustedMetric)
- Quality flags
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from bist_picker.dashboard.data_access import (
    get_adjusted_metrics,
    get_all_tickers,
    get_company_info,
    get_factor_scores,
    get_price_history,
    get_stock_position,
)
from bist_picker.dashboard.style import (
    colored_risk_html,
    colored_score_html,
    fmt_float,
    fmt_pct,
    fmt_price,
    score_color,
    risk_color,
    GREEN,
    RED,
    YELLOW,
    DIM,
)


def render():
    """Render the Hisse Detay (stock detail) page."""
    st.header("Hisse Detay")

    # -- Ticker Selector --
    tickers = get_all_tickers()
    if not tickers:
        st.warning("Veritabaninda aktif hisse bulunamadi.")
        return

    # Check if ticker was passed via query params
    params = st.query_params
    default_idx = 0
    if "ticker" in params:
        try:
            default_idx = tickers.index(params["ticker"].upper())
        except ValueError:
            pass

    selected = st.selectbox("Hisse Sec", tickers, index=default_idx)

    if not selected:
        return

    # -- Company Info --
    info = get_company_info(selected)
    if not info:
        st.error(f"{selected} bulunamadi.")
        return

    _render_company_header(info)

    # -- Tabs --
    tab_fiyat, tab_skor, tab_metrik = st.tabs(["Fiyat Grafigi", "Faktor Skorlari", "Finansal Metrikler"])

    with tab_fiyat:
        _render_price_chart(selected, info)

    with tab_skor:
        _render_scores(selected)

    with tab_metrik:
        _render_metrics(selected)


def _render_company_header(info: dict):
    """Render company info as a header panel."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**{info['name'] or info['ticker']}**")
        st.caption(f"Tur: {info['company_type'] or '--'}")
    with col2:
        st.caption(f"Sektor: {info['sector_custom'] or info['sector_bist'] or '--'}")
        st.caption(f"BIST-100: {'Evet' if info['is_bist100'] else 'Hayir'}")
    with col3:
        st.caption(f"IPO: {'Evet' if info['is_ipo'] else 'Hayir'}")
        st.caption(f"Free Float: {fmt_pct(info['free_float_pct'])}")
    with col4:
        st.caption(f"Listeleme: {info['listing_date'] or '--'}")
        status = "Aktif" if info['is_active'] else "Pasif"
        st.caption(f"Durum: {status}")

    st.divider()


def _render_price_chart(ticker: str, info: dict):
    """Render price chart with entry/target/stop lines."""
    period = st.selectbox(
        "Periyot",
        options=[90, 180, 365, 730],
        format_func=lambda x: {90: "3 Ay", 180: "6 Ay", 365: "1 Yil", 730: "2 Yil"}[x],
        index=2,
    )

    df = get_price_history(ticker, days=period)
    if df.empty:
        st.info("Fiyat verisi bulunamadi.")
        return

    price_col = "adjusted_close" if df["adjusted_close"].notna().any() else "close"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df[price_col],
        mode="lines",
        name="Fiyat",
        line=dict(color="#17a2b8", width=2),
    ))

    # Add position lines if exists
    position = get_stock_position(ticker)
    if position:
        entry = position.get("entry_price")
        target = position.get("target_price")
        stop = position.get("stop_loss_price")

        if entry:
            fig.add_hline(y=entry, line_dash="dash", line_color="#ffc107",
                         annotation_text=f"Giris: {entry:.2f}")
        if target:
            fig.add_hline(y=target, line_dash="dash", line_color="#28a745",
                         annotation_text=f"Hedef: {target:.2f}")
        if stop:
            fig.add_hline(y=stop, line_dash="dash", line_color="#dc3545",
                         annotation_text=f"Stop: {stop:.2f}")

    fig.update_layout(
        title=f"{ticker} Fiyat Grafigi",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (TL)",
        height=450,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Volume bar chart
    if df["volume"].notna().any():
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Hacim",
            marker_color="rgba(23, 162, 184, 0.4)",
        ))
        fig_vol.update_layout(
            title="Islem Hacmi",
            height=200,
            xaxis_title="Tarih",
            yaxis_title="Hacim",
        )
        st.plotly_chart(fig_vol, use_container_width=True)


def _render_scores(ticker: str):
    """Render factor scores table and radar chart."""
    scores = get_factor_scores(ticker)
    if not scores:
        st.info("Skorlama sonucu bulunamadi.")
        return

    st.caption(
        f"Skorlama tarihi: {scores['scoring_date']}  |  "
        f"Model: {scores['model_used'] or '--'}  |  "
        f"Veri tamligi: {fmt_pct(scores['data_completeness'])}"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        # Factor scores table
        st.subheader("Faktor Skorlari")
        factors = {
            "Buffett Kalite": scores["buffett"],
            "Graham Deger": scores["graham"],
            "Piotroski F-Skor": scores["piotroski"],
            "Magic Formula": scores["magic_formula"],
            "Lynch PEG": scores["lynch_peg"],
            "DCF Guvenlik Marji": scores["dcf_mos"],
            "Momentum": scores["momentum"],
            "Insider Aktivite": scores["insider"],
            "Teknik": scores["technical"],
        }

        factor_df = pd.DataFrame([
            {"Faktor": k, "Skor": round(v, 1) if v is not None else None}
            for k, v in factors.items()
        ])

        styled = factor_df.style.map(
            lambda val: f"background-color: rgba(40,167,69,0.25)" if isinstance(val, (int, float)) and val >= 70
            else f"background-color: rgba(255,193,7,0.20)" if isinstance(val, (int, float)) and val >= 50
            else f"background-color: rgba(220,53,69,0.25)" if isinstance(val, (int, float)) and val is not None
            else "",
            subset=["Skor"],
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Composite & Risk
        st.subheader("Kompozit Skor")
        alpha = scores.get("alpha")
        risk = scores.get("risk_tier", "--")
        st.metric("ALPHA", f"{alpha:.1f}" if alpha is not None else "--")
        st.markdown(f"Risk: **{risk}**")

    with col2:
        # Radar chart
        st.subheader("Faktor Radar")
        _render_radar_chart(factors)

    # Quality flags
    flags = scores.get("quality_flags")
    if flags:
        st.divider()
        st.subheader("Kalite Bayraklari")
        if isinstance(flags, dict):
            for key, detail in flags.items():
                st.warning(f"**{key}**: {detail}")
        elif isinstance(flags, list):
            for item in flags:
                st.warning(str(item))


def _render_radar_chart(factors: dict):
    """Render a radar/spider chart for factor scores."""
    # Filter out factors with None values to avoid misleading 0s
    filtered = {k: v for k, v in factors.items() if v is not None}
    if not filtered:
        st.info("Gosterilecek skor bulunamadi.")
        return

    labels = list(filtered.keys())
    values = list(filtered.values())

    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(23, 162, 184, 0.2)",
        line=dict(color="#17a2b8", width=2),
        name="Skorlar",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=False,
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_metrics(ticker: str):
    """Render adjusted financial metrics table."""
    metrics = get_adjusted_metrics(ticker)
    if not metrics:
        st.info("Duzeltilmis metrik bulunamadi.")
        return

    st.caption(f"Donem: {metrics['period_end']}")

    def _fmt_money(val):
        if val is None:
            return "--"
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:,.1f}M TL"
        return f"{val:,.0f} TL"

    data = [
        ("Raporlanan Net Gelir", _fmt_money(metrics["reported_net_income"])),
        ("Parasal Kazanc/Kayip", _fmt_money(metrics["monetary_gain_loss"])),
        ("Duzeltilmis Net Gelir", _fmt_money(metrics["adjusted_net_income"])),
        ("Owner Earnings", _fmt_money(metrics["owner_earnings"])),
        ("Serbest Nakit Akisi", _fmt_money(metrics["free_cash_flow"])),
        ("ROE (duzeltilmis)", fmt_pct(metrics["roe_adjusted"] * 100 if metrics["roe_adjusted"] else None)),
        ("ROA (duzeltilmis)", fmt_pct(metrics["roa_adjusted"] * 100 if metrics["roa_adjusted"] else None)),
        ("EPS (duzeltilmis)", fmt_float(metrics["eps_adjusted"])),
        ("Reel EPS Buyumesi", fmt_pct(metrics["real_eps_growth_pct"] * 100 if metrics["real_eps_growth_pct"] else None)),
        ("Iliskili Taraf Gelir %", fmt_pct(metrics["related_party_revenue_pct"] * 100 if metrics["related_party_revenue_pct"] else None)),
    ]

    if metrics.get("maintenance_capex") is not None:
        data.append(("Bakim CapEx", _fmt_money(metrics["maintenance_capex"])))
    if metrics.get("growth_capex") is not None:
        data.append(("Buyume CapEx", _fmt_money(metrics["growth_capex"])))

    metric_df = pd.DataFrame(data, columns=["Metrik", "Deger"])
    st.dataframe(metric_df, use_container_width=True, hide_index=True)
