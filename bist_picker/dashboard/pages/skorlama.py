"""Skorlama page for the Streamlit dashboard.

Displays the ALPHA-focused scoring table together with research-oriented
shadow buckets, model-specific rankings for non-operating company types,
and a comparison-only Piotroski-4 scenario.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from bist_picker.dashboard.data_access import (
    get_company_types,
    get_scoring_dates,
    get_scoring_results,
    get_sectors,
)
from bist_picker.dashboard.style import style_dataframe


def render() -> None:
    """Render the scoring page."""
    st.header("Skor Tablosu")

    with st.sidebar:
        st.subheader("Filtreler")

        dates = get_scoring_dates()
        selected_date = None
        if dates:
            selected_date = st.selectbox(
                "Skorlama Tarihi",
                options=dates,
                index=0,
                format_func=lambda value: value.strftime("%Y-%m-%d"),
            )

        view_mode = st.selectbox(
            "Gorunum Modu",
            options=["ALPHA Core", "ALPHA X", "Tum Sirketler", "Research Buckets", "Model Bazli"],
            index=0,
            help=(
                "ALPHA Core sadece uretim evrenini gosterir. "
                "ALPHA X ise scoring-only ikinci evrendir. "
                "Tum Sirketler blocker kolonlarini acar. "
                "Research Buckets ise kacan adaylari ayirir. "
                "Model Bazli banka/holding/reit gibi tipleri kendi skoruyla listeler."
            ),
        )

        types = get_company_types()
        selected_type = st.selectbox(
            "Sirket Turu",
            options=["Tumu"] + types,
            index=0,
        )
        selected_type = None if selected_type == "Tumu" else selected_type

        bist100_option = st.selectbox(
            "BIST-100",
            options=["Tumu", "Sadece BIST-100", "BIST-100 Disinda"],
            index=0,
        )
        is_bist100 = None
        if bist100_option == "Sadece BIST-100":
            is_bist100 = True
        elif bist100_option == "BIST-100 Disinda":
            is_bist100 = False

        sectors = get_sectors()
        selected_sector = st.selectbox(
            "Sektor",
            options=["Tumu"] + sectors,
            index=0,
        )
        selected_sector = None if selected_sector == "Tumu" else selected_sector

        selected_risk = st.selectbox(
            "Risk",
            options=["Tumu", "LOW", "MEDIUM", "HIGH"],
            index=0,
        )
        selected_risk = None if selected_risk == "Tumu" else selected_risk

        min_score_label = "Min X Skor" if view_mode == "ALPHA X" else "Min Alpha Skor"
        min_score = st.slider(min_score_label, 0, 100, 0)
        min_score = min_score if min_score > 0 else None

    analysis_df = get_scoring_results(
        scoring_date=selected_date,
        company_type=selected_type,
        is_bist100=is_bist100,
        sector_custom=selected_sector,
        risk_tier=selected_risk,
        min_score=None,
        alpha_eligible_only=False,
    )

    if analysis_df.empty:
        st.info("Secilen filtrelere uygun sonuc bulunamadi.")
        return

    analysis_df = analysis_df.copy()
    if "model_score" not in analysis_df.columns:
        analysis_df["model_score"] = analysis_df.apply(_get_model_score, axis=1)
    if view_mode == "ALPHA X":
        if min_score is not None:
            analysis_df = analysis_df[
                analysis_df["alpha_x_score"].notna()
                & (analysis_df["alpha_x_score"] >= float(min_score))
            ].copy()
    elif min_score is not None:
        analysis_df = analysis_df[
            analysis_df["alpha"].notna()
            & (analysis_df["alpha"] >= float(min_score))
        ].copy()

    if analysis_df.empty:
        st.info("Secilen filtrelerde skor esigini gecen sonuc bulunamadi.")
        return

    if view_mode == "ALPHA Core":
        df = analysis_df[analysis_df["alpha_core_eligible"]].copy()
    elif view_mode == "ALPHA X":
        df = analysis_df[analysis_df["alpha_x_eligible"]].copy()
    elif view_mode == "Research Buckets":
        df = analysis_df[
            analysis_df["alpha_research_bucket"].isin(
                [
                    "Quality Shadow",
                    "Free-Float Shadow",
                    "Non-Core Research",
                    "Data-Unscorable",
                ]
            )
        ].copy()
    elif view_mode == "Model Bazli":
        df = analysis_df.copy()
    else:
        df = analysis_df.copy()

    if view_mode == "Model Bazli":
        st.subheader("Model Coverage Ozet")
        _render_model_summary(analysis_df)

        st.divider()
        _render_model_rankings(df, selected_type)

        ranked_df = df[df["ranking_score"].notna()].copy()
        if not ranked_df.empty:
            st.divider()
            _render_detail_picker(ranked_df)

        st.divider()
        _render_explainers()
        return

    if view_mode == "ALPHA X":
        st.subheader("ALPHA X Ozet")
        _render_alpha_x_summary(analysis_df)

        st.divider()
        st.caption(
            "ALPHA X scoring-only deneysel evrendir. Portfoy seciciyi degistirmez; "
            "operating disi tipleri native model + guven + investability ile ortak listede dener."
        )
        if selected_type:
            st.caption(
                f"`Sirket Turu` filtresi aktif: ALPHA X su anda sadece `{selected_type}` "
                "tipindeki adaylari gosteriyor."
            )
        if df.empty:
            st.info(
                "Bu filtrelerde ALPHA X uygun aday yok. Asagidaki blocker ozetleri "
                "hangi tiplerin native model veya investability tarafinda takildigini gosterir."
            )
        else:
            _render_alpha_x_table(df)
            ranked_df = df.sort_values(
                ["alpha_x_score", "alpha_x_confidence", "ticker"],
                ascending=[False, False, True],
                na_position="last",
            ).reset_index(drop=True)
            _render_detail_picker(ranked_df)

        st.divider()
        _render_explainers()
        return

    st.subheader("ALPHA Core Ozet")
    _render_alpha_research_summary(analysis_df)

    st.divider()

    if view_mode == "ALPHA Core":
        st.caption(f"{len(df)} hisse listeleniyor (ALPHA Core uygun evren)")
        if len(df) != len(analysis_df):
            st.caption(
                f"{len(analysis_df) - len(df)} isim arastirma veya disari havuzunda kaliyor."
            )
        st.info(
            "Yeni blocker ve research kolonlarini tablo icinde gormek icin "
            "`Gorunum Modu`nu `Tum Sirketler` veya `Research Buckets` yap."
        )
    elif view_mode == "Research Buckets":
        st.caption(
            f"{len(df)} hisse listeleniyor "
            "(Quality Shadow, Free-Float Shadow, Non-Core Research, Data-Unscorable)"
        )
    else:
        st.caption(
            f"{len(df)} hisse listeleniyor "
            "(tum sirketler; pasif ve skorsuz olanlar dahil)"
        )
        st.caption(
            "`ALPHA Notu`, `Birincil Engel`, `Arastirma Havuzu` ve "
            "`Snapshot Streak` alanlari ALPHA Core karar mantigini aciklar."
        )

    if df.empty:
        st.info(
            "Secilen filtrelerde uygun sonuc yok. Yukaridaki ALPHA Core ozet "
            "ve arastirma havuzlari hangi isimlerin sinira takildigini gosteriyor."
        )
    else:
        _render_scoring_table(df, show_diagnostics=view_mode != "ALPHA Core")
        _render_detail_picker(df)

    st.divider()
    st.subheader("Model Bazli Ozet")
    _render_model_summary(analysis_df)

    st.divider()
    _render_explainers()

    st.divider()
    _render_charts(df if not df.empty else analysis_df)


def _render_scoring_table(df: pd.DataFrame, show_diagnostics: bool) -> None:
    """Render the main scoring dataframe."""
    score_cols = [
        "buffett",
        "graham",
        "piotroski",
        "magic_formula",
        "lynch_peg",
        "dcf_mos",
        "momentum",
        "insider",
        "technical",
    ]

    display_cols = [
        "ticker",
        "name",
        "type",
        "sector",
        *score_cols,
        "model_score",
        "alpha",
        "data_completeness",
        "risk",
        "alpha_snapshot_streak",
    ]
    display_labels = [
        "Hisse",
        "Isim",
        "Tur",
        "Sektor",
        "Buffett",
        "Graham",
        "Piotroski",
        "Magic F.",
        "Lynch",
        "DCF MoS",
        "Momentum",
        "Insider",
        "Teknik",
        "Model",
        "Alpha",
        "Veri %",
        "Risk",
        "Snapshot Streak",
    ]

    if show_diagnostics:
        extra_cols = [
            "alpha_core_eligible",
            "alpha_core_status",
            "alpha_reason",
            "alpha_primary_blocker",
            "alpha_all_blockers",
            "alpha_research_bucket",
            "alpha_value_group",
            "alpha_growth_group",
            "alpha_missing_groups",
            "alpha_x_delta",
            "alpha_relaxed_p4_eligible",
            "is_active",
        ]
        extra_labels = [
            "ALPHA Core Uygun",
            "Alpha Core Durumu",
            "ALPHA Notu",
            "Birincil Engel",
            "Tum Engeller",
            "Arastirma Havuzu",
            "Value Grup",
            "Growth Grup",
            "Eksik Grup",
            "X-Alpha",
            "P4 Senaryo",
            "Durum",
        ]
        display_cols = display_cols[:4] + extra_cols + display_cols[4:]
        display_labels = display_labels[:4] + extra_labels + display_labels[4:]

    display_df = df[display_cols].copy()
    display_df.columns = display_labels

    if "ALPHA Core Uygun" in display_df.columns:
        display_df["ALPHA Core Uygun"] = display_df["ALPHA Core Uygun"].map(
            lambda value: "Evet" if value else "Hayir"
        )
    if "P4 Senaryo" in display_df.columns:
        display_df["P4 Senaryo"] = display_df["P4 Senaryo"].map(
            lambda value: "Evet" if value else "Hayir"
        )
    if "Durum" in display_df.columns:
        display_df["Durum"] = display_df["Durum"].map(
            lambda value: "Aktif" if value else "Pasif"
        )

    for text_col in [
        "ALPHA Notu",
        "Birincil Engel",
        "Tum Engeller",
        "Arastirma Havuzu",
        "Alpha Core Durumu",
        "Eksik Grup",
    ]:
        if text_col in display_df.columns:
            display_df[text_col] = display_df[text_col].fillna("-")
            display_df[text_col] = display_df[text_col].replace("", "-")

    numeric_cols = [
        "Buffett",
        "Graham",
        "Piotroski",
        "Magic F.",
        "Lynch",
        "DCF MoS",
        "Momentum",
        "Insider",
        "Teknik",
        "Model",
        "Alpha",
        "Value Grup",
        "Growth Grup",
        "X-Alpha",
        "Veri %",
    ]
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda value: round(value, 1) if pd.notna(value) else None
            )

    if "Snapshot Streak" in display_df.columns:
        display_df["Snapshot Streak"] = display_df["Snapshot Streak"].apply(
            lambda value: int(value) if pd.notna(value) else 0
        )

    row_types = display_df["Tur"].fillna("").str.upper()

    def missing_css(column_name: str, row_idx: int) -> str:
        """Return CSS for a missing factor cell."""
        company_type = row_types.iloc[row_idx]
        classic_factors = {
            "Buffett",
            "Graham",
            "Piotroski",
            "Magic F.",
            "Lynch",
            "DCF MoS",
        }
        if column_name not in classic_factors:
            return ""
        if company_type in ("BANK", "FINANCIAL", "INSURANCE", "HOLDING", "REIT", "SPORT"):
            return "background-color: rgba(108, 117, 125, 0.4)"
        if column_name == "Graham":
            return "background-color: rgba(111, 66, 193, 0.4)"
        if column_name == "DCF MoS":
            return "background-color: rgba(253, 126, 20, 0.4)"
        return "background-color: rgba(13, 202, 240, 0.4)"

    def highlight_missing(frame: pd.DataFrame) -> pd.DataFrame:
        """Apply contextual color styling to missing cells."""
        css_df = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for col in ["Buffett", "Graham", "Piotroski", "Magic F.", "Lynch", "DCF MoS"]:
            if col not in frame.columns:
                continue
            for idx in range(len(frame)):
                value = frame[col].iloc[idx]
                if pd.isna(value) or value is None:
                    css_df.iloc[idx, css_df.columns.get_loc(col)] = missing_css(col, idx)
        return css_df

    styled = style_dataframe(
        display_df,
        score_columns=numeric_cols,
        pnl_columns=None,
    ).apply(highlight_missing, axis=None)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=620,
    )


def _render_detail_picker(df: pd.DataFrame) -> None:
    """Render the quick link into the stock detail page."""
    if df.empty:
        return

    detail_col1, detail_col2 = st.columns([3, 1])
    with detail_col1:
        selected_ticker = st.selectbox(
            "Detay icin hisse sec",
            options=df["ticker"].tolist(),
            key="score_detail_ticker",
        )
    with detail_col2:
        open_detail = st.button(
            "Hisse Detay Ac",
            key="open_score_detail",
            use_container_width=True,
        )

    if open_detail:
        st.query_params["ticker"] = selected_ticker
        st.session_state["nav_page"] = "Hisse Detay"
        st.rerun()


def _render_model_rankings(df: pd.DataFrame, selected_type: str | None) -> None:
    """Render per-company-type ranking tables for non-operating model families."""
    if df.empty:
        st.info("Model bazli listeleme icin veri bulunamadi.")
        return

    available_types = [t for t in df["type"].dropna().unique().tolist() if t]
    default_types = ["BANK", "FINANCIAL", "INSURANCE", "HOLDING", "REIT"]

    if selected_type:
        types_to_render = [selected_type] if selected_type in available_types else []
    else:
        types_to_render = [t for t in default_types if t in available_types]

    if not types_to_render:
        st.info(
            "Bu filtrelerde model bazli gosterilecek banka/financial/insurance/holding/reit "
            "sonucu bulunamadi."
        )
        return

    st.caption(
        "Bu gorunum ALPHA Core disindaki sirketleri kendi tipine gore ranklar. "
        "`Model Skoru` varsa ana kaynak odur; yoksa `Alpha Fallback` veya "
        "`Market Fallback` ile siralama devam eder."
    )

    for company_type in types_to_render:
        subset = df[df["type"] == company_type].copy()
        if subset.empty:
            continue

        subset = subset.sort_values(
            ["ranking_score", "model_score", "alpha", "ticker"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)

        native_count = int(subset["has_native_model_score"].fillna(False).sum())
        fallback_count = int(subset["ranking_uses_fallback"].fillna(False).sum())
        ranked_count = int(subset["ranking_score"].notna().sum())

        st.markdown(f"**{_type_label(company_type)}**")
        st.caption(
            f"{len(subset)} isim | ranklanan {ranked_count} | "
            f"native model {native_count} | fallback {fallback_count}"
        )

        _render_model_ranking_table(subset)


def _render_model_ranking_table(df: pd.DataFrame) -> None:
    """Render a compact ranking table for a specific company type."""
    display_cols = [
        "type_rank",
        "ticker",
        "name",
        "sector",
        "ranking_source",
        "ranking_score",
        "model_score",
        "alpha",
        "data_completeness",
        "risk",
        "free_float_pct",
        "alpha_reason",
    ]
    display_labels = [
        "Tip Rank",
        "Hisse",
        "Isim",
        "Sektor",
        "Kaynak",
        "Rank Skoru",
        "Model Skoru",
        "Alpha",
        "Veri %",
        "Risk",
        "Free Float %",
        "ALPHA Notu",
    ]

    display_df = df[display_cols].copy()
    display_df.columns = display_labels

    if "Tip Rank" in display_df.columns:
        display_df["Tip Rank"] = display_df["Tip Rank"].apply(
            lambda value: int(value) if pd.notna(value) else None
        )

    numeric_cols = [
        "Rank Skoru",
        "Model Skoru",
        "Alpha",
        "Veri %",
        "Free Float %",
    ]
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda value: round(value, 1) if pd.notna(value) else None
        )

    display_df["ALPHA Notu"] = display_df["ALPHA Notu"].fillna("-").replace("", "-")
    display_df["Kaynak"] = display_df["Kaynak"].fillna("Unscored")

    styled = style_dataframe(
        display_df,
        score_columns=numeric_cols,
        pnl_columns=None,
    )
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(560, 80 + len(display_df) * 35),
    )


def _render_alpha_x_summary(df: pd.DataFrame) -> None:
    """Render summary diagnostics for the scoring-only ALPHA X universe."""
    if df.empty:
        st.info("ALPHA X ozetini gostermek icin veri bulunamadi.")
        return

    candidate_df = df[df["type"].isin(["OPERATING", "REIT", "HOLDING", "INSURANCE", "FINANCIAL"])].copy()
    if candidate_df.empty:
        st.info("ALPHA X kapsaminda aday type bulunamadi.")
        return

    eligible_mask = candidate_df["alpha_x_eligible"].fillna(False)
    bucket_series = candidate_df["alpha_x_bucket"].fillna("Excluded")

    metric_cols = st.columns(5)
    metric_cols[0].metric("ALPHA X", int(eligible_mask.sum()))
    metric_cols[1].metric(
        "Operating",
        int((eligible_mask & (candidate_df["type"] == "OPERATING")).sum()),
    )
    metric_cols[2].metric(
        "REIT",
        int((eligible_mask & (candidate_df["type"] == "REIT")).sum()),
    )
    metric_cols[3].metric(
        "Holding",
        int((eligible_mask & (candidate_df["type"] == "HOLDING")).sum()),
    )
    metric_cols[4].metric(
        "Shadow",
        int(bucket_series.isin(["Native Shadow", "Confidence Shadow"]).sum()),
    )

    breakdown_rows = []
    for company_type in ["OPERATING", "REIT", "HOLDING", "INSURANCE", "FINANCIAL"]:
        subset = candidate_df[candidate_df["type"] == company_type]
        if subset.empty:
            continue
        eligible_subset = subset[subset["alpha_x_eligible"].fillna(False)]
        breakdown_rows.append(
            {
                "Tur": company_type,
                "Adet": len(subset),
                "Uygun": int(len(eligible_subset)),
                "Native": int(subset["has_native_model_score"].fillna(False).sum()),
                "Ort X": round(eligible_subset["alpha_x_score"].mean(), 1)
                if not eligible_subset.empty
                else None,
                "Ort Guven %": round(eligible_subset["alpha_x_confidence"].mean() * 100.0, 1)
                if not eligible_subset.empty
                else None,
            }
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Type Breakdown**")
        st.dataframe(
            pd.DataFrame(breakdown_rows).fillna("-"),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("**Top Blockers**")
        blockers = (
            candidate_df.loc[~eligible_mask, "alpha_x_reason"]
            .fillna("Bilinmiyor")
            .value_counts()
            .reset_index()
        )
        blockers.columns = ["Birincil Engel", "Adet"]
        if blockers.empty:
            st.caption("ALPHA X blocker yok.")
        else:
            st.dataframe(blockers.head(10), hide_index=True, use_container_width=True)

    native_shadow = candidate_df[candidate_df["alpha_x_bucket"] == "Native Shadow"].copy()
    native_shadow = native_shadow.sort_values(
        ["ranking_score", "alpha", "ticker"],
        ascending=[False, False, True],
        na_position="last",
    )
    native_shadow_table = _prepare_summary_table(
        native_shadow,
        columns=["ticker", "name", "type", "ranking_source", "ranking_score", "alpha"],
        labels=["Hisse", "Isim", "Tur", "Kaynak", "Rank", "Alpha"],
        sort_by=["ranking_score", "alpha"],
        limit=8,
    )
    st.markdown("**Native Shadow Leaders**")
    _render_summary_table(
        native_shadow_table,
        empty_text="Native model eksigi olan shadow aday yok.",
    )


def _render_alpha_x_table(df: pd.DataFrame) -> None:
    """Render the combined ALPHA X eligible ranking table."""
    if df.empty:
        st.info("ALPHA X icin gosterilecek aday yok.")
        return

    ranked = df.sort_values(
        ["alpha_x_score", "alpha_x_confidence", "ticker"],
        ascending=[False, False, True],
        na_position="last",
    ).copy()

    display_df = ranked[
        [
            "alpha_x_rank",
            "ticker",
            "name",
            "type",
            "sector",
            "alpha_x_score",
            "alpha_x_base_score",
            "alpha_x_confidence",
            "alpha_x_investability",
            "alpha_x_delta",
            "alpha_value_group",
            "alpha_growth_group",
            "alpha_missing_groups",
            "ranking_source",
            "model_score",
            "alpha",
            "data_completeness",
            "risk",
            "free_float_pct",
            "avg_volume_try",
        ]
    ].copy()
    display_df.columns = [
        "X Rank",
        "Hisse",
        "Isim",
        "Tur",
        "Sektor",
        "X Skor",
        "Baz Skor",
        "Guven %",
        "Investability",
        "X-Alpha",
        "Value Grup",
        "Growth Grup",
        "Eksik Grup",
        "Kaynak",
        "Model",
        "Alpha",
        "Veri %",
        "Risk",
        "Free Float %",
        "Likidite (M TRY)",
    ]

    display_df["X Rank"] = display_df["X Rank"].apply(
        lambda value: int(value) if pd.notna(value) else None
    )
    display_df["Guven %"] = display_df["Guven %"].apply(
        lambda value: value * 100.0 if pd.notna(value) else None
    )
    display_df["Likidite (M TRY)"] = display_df["Likidite (M TRY)"].apply(
        lambda value: value / 1_000_000 if pd.notna(value) else None
    )

    numeric_cols = [
        "X Skor",
        "Baz Skor",
        "Guven %",
        "Investability",
        "X-Alpha",
        "Value Grup",
        "Growth Grup",
        "Model",
        "Alpha",
        "Veri %",
        "Free Float %",
        "Likidite (M TRY)",
    ]
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda value: round(value, 1) if pd.notna(value) else None
        )
    if "Eksik Grup" in display_df.columns:
        display_df["Eksik Grup"] = display_df["Eksik Grup"].fillna("-").replace("", "-")

    styled = style_dataframe(
        display_df,
        score_columns=numeric_cols,
        pnl_columns=None,
    )
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(620, 80 + len(display_df) * 35),
    )


def _render_alpha_research_summary(df: pd.DataFrame) -> None:
    """Render ALPHA Core blockers, research buckets, and P4 scenario summaries."""
    if df.empty:
        st.info("ALPHA ozetini gostermek icin veri bulunamadi.")
        return

    core_mask = df["alpha_core_eligible"].fillna(False)
    bucket_series = df["alpha_research_bucket"].fillna("Excluded")
    scenario_mask = df["alpha_relaxed_p4_eligible"].fillna(False)

    metric_cols = st.columns(5)
    metric_cols[0].metric("ALPHA Core", int(core_mask.sum()))
    metric_cols[1].metric(
        "Quality Shadow",
        int((bucket_series == "Quality Shadow").sum()),
    )
    metric_cols[2].metric(
        "Free-Float Shadow",
        int((bucket_series == "Free-Float Shadow").sum()),
    )
    metric_cols[3].metric(
        "Non-Core Research",
        int((bucket_series == "Non-Core Research").sum()),
    )
    metric_cols[4].metric(
        "Data-Unscorable",
        int((bucket_series == "Data-Unscorable").sum()),
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Core Blockers**")
        blockers = (
            df.loc[
                (~core_mask) & (bucket_series != "Data-Unscorable"),
                "alpha_primary_blocker",
            ]
            .fillna("Bilinmiyor")
            .value_counts()
            .reset_index()
        )
        blockers.columns = ["Birincil Engel", "Adet"]
        if blockers.empty:
            st.caption("Baglayici blocker yok.")
        else:
            st.dataframe(blockers.head(10), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Data-Unscorable Segmenti**")
        unscorable = (
            df.loc[bucket_series == "Data-Unscorable", "type"]
            .fillna("Bilinmiyor")
            .value_counts()
            .reset_index()
        )
        unscorable.columns = ["Tur", "Adet"]
        if unscorable.empty:
            st.caption("Data-Unscorable isim yok.")
        else:
            st.dataframe(unscorable, hide_index=True, use_container_width=True)

    st.markdown("**Quality Shadow Leaders**")
    quality = _prepare_summary_table(
        df[df["alpha_research_bucket"] == "Quality Shadow"],
        columns=["ticker", "name", "alpha", "piotroski_raw", "alpha_snapshot_streak"],
        labels=["Hisse", "Isim", "Alpha", "Raw F", "Streak"],
        sort_by=["alpha", "alpha_snapshot_streak"],
    )
    _render_summary_table(quality, empty_text="Quality Shadow adayi bulunmadi.")

    st.markdown("**Free-Float Shadow Leaders**")
    free_float = _prepare_summary_table(
        df[df["alpha_research_bucket"] == "Free-Float Shadow"],
        columns=["ticker", "name", "alpha", "free_float_pct", "alpha_snapshot_streak"],
        labels=["Hisse", "Isim", "Alpha", "Free Float %", "Streak"],
        sort_by=["alpha", "alpha_snapshot_streak"],
    )
    _render_summary_table(free_float, empty_text="Free-Float Shadow adayi bulunmadi.")

    st.markdown("**Non-Core Research Leaders**")
    non_core = _prepare_summary_table(
        df[df["alpha_research_bucket"] == "Non-Core Research"],
        columns=["ticker", "name", "type", "alpha", "alpha_all_blockers", "alpha_snapshot_streak"],
        labels=["Hisse", "Isim", "Tur", "Alpha", "Tum Engeller", "Streak"],
        sort_by=["alpha", "alpha_snapshot_streak"],
    )
    _render_summary_table(non_core, empty_text="Non-Core Research ismi bulunmadi.")

    st.markdown("**Piotroski 5 -> 4 Senaryosu**")
    scenario_df = df[df["alpha"].notna()].copy()
    core_ranked = (
        scenario_df[scenario_df["alpha_core_eligible"]]
        .sort_values(["alpha", "alpha_snapshot_streak"], ascending=[False, False])
        .reset_index(drop=True)
    )
    relaxed_ranked = (
        scenario_df[scenario_df["alpha_relaxed_p4_eligible"]]
        .sort_values(["alpha", "alpha_snapshot_streak"], ascending=[False, False])
        .reset_index(drop=True)
    )

    scenario_additions = relaxed_ranked[
        ~relaxed_ranked["ticker"].isin(core_ranked["ticker"])
    ].copy()
    additions_table = _prepare_summary_table(
        scenario_additions,
        columns=["ticker", "name", "alpha", "piotroski_raw", "alpha_snapshot_streak"],
        labels=["Hisse", "Isim", "Alpha", "Raw F", "Streak"],
        sort_by=["alpha", "alpha_snapshot_streak"],
    )
    _render_summary_table(
        additions_table,
        empty_text="Piotroski 4 senaryosunda yeni aday gelmiyor.",
    )

    rank_diff = _build_scenario_rank_diff(core_ranked, relaxed_ranked)
    if rank_diff.empty:
        st.caption("Top-10 siralama farki olusmadi.")
    else:
        st.dataframe(rank_diff, hide_index=True, use_container_width=True)

    core_top3 = core_ranked["ticker"].head(3).tolist()
    relaxed_top3 = relaxed_ranked["ticker"].head(3).tolist()
    changed = core_top3 != relaxed_top3
    status = "degisirdi" if changed else "degismezdi"
    st.caption(
        "Skor liderlerine gore top-3 "
        f"{status}. Core: {', '.join(core_top3) or '-'} | "
        f"P4 Senaryo: {', '.join(relaxed_top3) or '-'} | "
        f"Ek aday: {int((scenario_mask & ~core_mask).sum())}"
    )


def _prepare_summary_table(
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    sort_by: list[str],
    limit: int = 12,
) -> pd.DataFrame:
    """Prepare a compact summary table with rounded numeric values."""
    if df.empty:
        return pd.DataFrame(columns=labels)

    prepared = df.copy()
    prepared = prepared.sort_values(sort_by, ascending=[False] * len(sort_by)).head(limit)
    prepared = prepared[columns].copy()
    prepared.columns = labels

    for col in prepared.columns:
        if prepared[col].dtype.kind in {"f", "i"}:
            prepared[col] = prepared[col].apply(
                lambda value: round(value, 1) if pd.notna(value) else None
            )

    return prepared.fillna("-")


def _render_summary_table(df: pd.DataFrame, empty_text: str) -> None:
    """Render a summary table or a short empty-state caption."""
    if df.empty:
        st.caption(empty_text)
    else:
        st.dataframe(df, hide_index=True, use_container_width=True)


def _build_scenario_rank_diff(
    core_ranked: pd.DataFrame,
    relaxed_ranked: pd.DataFrame,
) -> pd.DataFrame:
    """Return a compact top-10 rank comparison for the P4 scenario."""
    if core_ranked.empty and relaxed_ranked.empty:
        return pd.DataFrame()

    core_top = core_ranked[["ticker", "name"]].head(10).copy()
    core_top["Core Rank"] = range(1, len(core_top) + 1)
    relaxed_top = relaxed_ranked[["ticker", "name"]].head(10).copy()
    relaxed_top["P4 Rank"] = range(1, len(relaxed_top) + 1)

    merged = core_top.merge(
        relaxed_top,
        on=["ticker", "name"],
        how="outer",
    )

    def _status(row: pd.Series) -> str:
        if pd.isna(row["Core Rank"]):
            return "Yeni"
        if pd.isna(row["P4 Rank"]):
            return "Disari"
        if row["P4 Rank"] < row["Core Rank"]:
            return "Yukari"
        if row["P4 Rank"] > row["Core Rank"]:
            return "Asagi"
        return "Ayni"

    merged["Durum"] = merged.apply(_status, axis=1)
    merged["Sort Rank"] = merged["P4 Rank"].fillna(99).astype(float)
    merged = merged.sort_values(["Sort Rank", "Core Rank", "ticker"]).drop(columns="Sort Rank")
    merged.columns = ["Hisse", "Isim", "Core Rank", "P4 Rank", "Durum"]
    return merged.fillna("-")


def _render_explainers() -> None:
    """Render help text for missing data and ALPHA Core logic."""
    with st.expander("Bos hucreler neden var?"):
        st.markdown(
            """
            - `BANK`, `HOLDING`, `REIT` ve `FINANCIAL` sirketlerde klasik faktorlerin bir kismi bilerek bos olur; bunlar kendi model kompozitleriyle degerlendirilir.
            - `SPORT` sirketleri ALPHA Core portfoyu icin yatirilabilir evrende degildir.
            - `Data-Unscorable`, skor snapshot'i olmayan veya kullanilabilir veri kapsami tasimayan isimleri ayri tutar.
            - `Model Bazli` gorunumde `Kaynak`, native model / alpha fallback / market fallback ayrimini gosterir.
            - `ALPHA X`, operating disi tipleri ortak skala ustunde denemek icin `guven` ve `investability` kalibrasyonu uygular.
            - `Value Grup` = `Graham` ve `DCF MoS` ortalamasi; `Growth Grup` = `Magic Formula` ve `Lynch` ortalamasi.
            - Bir grubun iki parcasi da bossa o grubun agirligi kalan mevcut faktorler arasinda dagitilir; bu yuzden tek tek dusuk kutular son skoru oldugu gibi temsil etmez.
            - `Graham` boslugu genelde negatif veya eksik `EPS` kaynaklidir.
            - `DCF MoS` boslugu genelde negatif `owner earnings`, negatif `adjusted net income` veya eksik pay-basi veri yuzundendir.
            - `Buffett` boslugu genelde yeterli yillik gecmis olmamasindan gelir.
            """
        )

    with st.expander("ALPHA Core neye gore belirleniyor?"):
        st.markdown(
            """
            - `ALPHA Core`, orta vadeli ve uygulanabilirlik odakli uretim evrenidir.
            - Kural seti acik sekilde sabittir: `OPERATING`, `risk != HIGH`, `free float >= %25`, `likidite >= 10M TRY`, `veri kapsami >= %70`, `raw Piotroski >= 5`.
            - Ek koruma olarak son donemde hem `adjusted net income` hem `owner earnings` negatif olan isimler ALPHA Core disinda kalir.
            - `Quality Shadow`, sadece `Piotroski = 4` yuzunden disarida kalan operating isimleri izler.
            - `Free-Float Shadow`, sadece `free float` kapisinda takilan ama Alpha skoru cok guclu operating isimleri izler.
            - `Non-Core Research`, `BANK`, `FINANCIAL`, `HOLDING`, `REIT`, `INSURANCE` isimlerini research-only olarak ayirir.
            - `Model Bazli`, bu non-core tipleri kendi skoruyla ayrica ranklar; model skoru yoksa fallback kaynak acikca yazilir.
            - `ALPHA X`, portfoy seciciyi degistirmeden `OPERATING + secilmis non-operating` tipleri ortak deneysel listede yaristirir.
            - `P4 Senaryo`, uretim secimini degistirmez; sadece `Piotroski 5 -> 4` gevsemesiyle ne degisirdi sorusunu gosterir.
            """
        )

    st.markdown(
        """
        **Eksik Veri (Bos Hucre) Renk Anlamlari:**
        - `Mor`: Zarar aciklamis / negatif EPS
        - `Turuncu`: Nakit yakiyor / DCF tanimsiz
        - `Mavi`: Veri yetersiz veya finansal gecmis zayif
        - `Gri`: Sektor-model uyumsuz; klasik faktor bilincli olarak hesaplanmiyor
        """
    )


def _render_charts(df: pd.DataFrame) -> None:
    """Render the alpha histogram and sector pie chart."""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Alpha Skor Dagilimi")
        if "alpha" in df.columns and df["alpha"].notna().any():
            fig = px.histogram(
                df[df["alpha"].notna()],
                x="alpha",
                nbins=20,
                labels={"alpha": "Alpha Skor"},
                color_discrete_sequence=["#17a2b8"],
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Alpha Skor",
                yaxis_title="Hisse Sayisi",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sektor Dagilimi")
        if "sector" in df.columns and df["sector"].notna().any():
            sector_counts = df["sector"].value_counts().head(15)
            fig = px.pie(
                values=sector_counts.values,
                names=sector_counts.index,
                hole=0.3,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


def _get_model_score(row: pd.Series) -> float | None:
    """Return the company-type-specific model score for display."""
    company_type = (row.get("type") or "").upper()
    if company_type in ("BANK", "FINANCIAL", "INSURANCE"):
        return row.get("banking_composite")
    if company_type == "HOLDING":
        return row.get("holding_composite")
    if company_type == "REIT":
        return row.get("reit_composite")
    return None


def _type_label(company_type: str) -> str:
    """Return a user-facing section label for a company type."""
    labels = {
        "BANK": "Bankalar",
        "FINANCIAL": "Finansallar",
        "INSURANCE": "Sigorta",
        "HOLDING": "Holding",
        "REIT": "GYO / REIT",
        "OPERATING": "Operating",
    }
    return labels.get((company_type or "").upper(), company_type or "Bilinmiyor")


def _render_model_summary(df: pd.DataFrame) -> None:
    """Render summary statistics by company type."""
    if "type" not in df.columns:
        return

    summary_rows = []
    for company_type in sorted(df["type"].dropna().unique()):
        subset = df[df["type"] == company_type]
        model_valid = subset["model_score"].dropna() if "model_score" in subset else pd.Series(dtype=float)
        ranking_valid = subset["ranking_score"].dropna() if "ranking_score" in subset else pd.Series(dtype=float)
        alpha_valid = subset["alpha"].dropna()
        dc_valid = subset["data_completeness"].dropna()
        summary_rows.append(
            {
                "Tur": company_type,
                "Adet": len(subset),
                "Native Model": int(subset["has_native_model_score"].fillna(False).sum()) if "has_native_model_score" in subset else 0,
                "Fallback Rank": int(subset["ranking_uses_fallback"].fillna(False).sum()) if "ranking_uses_fallback" in subset else 0,
                "Ort Rank": round(ranking_valid.mean(), 1) if not ranking_valid.empty else None,
                "Ort Model": round(model_valid.mean(), 1) if not model_valid.empty else None,
                "Ort Alpha": round(alpha_valid.mean(), 1) if not alpha_valid.empty else None,
                "Max Alpha": round(alpha_valid.max(), 1) if not alpha_valid.empty else None,
                "Ort Veri %": round(dc_valid.mean(), 1) if not dc_valid.empty else None,
                "Alpha>=80": int((alpha_valid >= 80).sum()) if not alpha_valid.empty else 0,
                "Veri<%40": int((dc_valid < 40).sum()) if not dc_valid.empty else 0,
            }
        )

    if summary_rows:
        st.dataframe(
            pd.DataFrame(summary_rows),
            hide_index=True,
            use_container_width=True,
        )
