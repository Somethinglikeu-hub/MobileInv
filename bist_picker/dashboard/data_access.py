"""Cached Streamlit wrappers over the shared read-only query service."""

from __future__ import annotations

import streamlit as st

import bist_picker.read_service as read_service
from bist_picker.db.connection import get_engine

CACHE_TTL = read_service.CACHE_TTL
_shared_get_engine = read_service.get_engine
_last_synced_get_engine = _shared_get_engine

_shared_load_latest_macro = read_service._load_latest_macro
_shared_get_open_positions = read_service.get_open_positions
_shared_get_portfolio_performance = read_service.get_portfolio_performance
_shared_get_all_portfolio_performance = read_service.get_all_portfolio_performance
_shared_get_portfolio_history = read_service.get_portfolio_history
_shared_get_scoring_dates = read_service.get_scoring_dates
_shared_get_latest_scoring_date = read_service.get_latest_scoring_date
_shared_get_alpha_universe_diagnostics = read_service.get_alpha_universe_diagnostics
_shared_get_alpha_eligible_company_ids = read_service.get_alpha_eligible_company_ids
_shared_get_alpha_eligibility_reasons = read_service.get_alpha_eligibility_reasons
_shared_get_alpha_snapshot_streaks = read_service.get_alpha_snapshot_streaks
_shared_get_alpha_dashboard_diagnostics = read_service.get_alpha_dashboard_diagnostics
_shared_get_scoring_results = read_service.get_scoring_results
_shared_get_sectors = read_service.get_sectors
_shared_get_company_types = read_service.get_company_types
_shared_get_company_info = read_service.get_company_info
_shared_get_price_history = read_service.get_price_history
_shared_get_factor_scores = read_service.get_factor_scores
_shared_get_adjusted_metrics = read_service.get_adjusted_metrics
_shared_get_stock_position = read_service.get_stock_position
_shared_get_latest_macro = read_service.get_latest_macro
_shared_get_all_tickers = read_service.get_all_tickers


def _sync_shared_dependencies() -> None:
    """Keep the shared module aligned with Streamlit-side monkeypatches."""
    global _last_synced_get_engine
    if read_service.get_engine in {_shared_get_engine, _last_synced_get_engine}:
        read_service.get_engine = get_engine
        _last_synced_get_engine = get_engine
    read_service.get_latest_scoring_date = get_latest_scoring_date
    read_service.get_alpha_eligible_company_ids = get_alpha_eligible_company_ids
    read_service.get_alpha_eligibility_reasons = get_alpha_eligibility_reasons
    read_service.get_alpha_snapshot_streaks = get_alpha_snapshot_streaks
    read_service.get_alpha_dashboard_diagnostics = get_alpha_dashboard_diagnostics


def _load_latest_macro(session):
    """Return the newest macro snapshot, backfilling missing fields safely."""
    _sync_shared_dependencies()
    return _shared_load_latest_macro(session)


@st.cache_data(ttl=CACHE_TTL)
def get_open_positions():
    _sync_shared_dependencies()
    return _shared_get_open_positions()


@st.cache_data(ttl=CACHE_TTL)
def get_portfolio_performance(portfolio_name: str):
    _sync_shared_dependencies()
    return _shared_get_portfolio_performance(portfolio_name)


@st.cache_data(ttl=CACHE_TTL)
def get_all_portfolio_performance():
    _sync_shared_dependencies()
    return _shared_get_all_portfolio_performance()


@st.cache_data(ttl=CACHE_TTL)
def get_portfolio_history():
    _sync_shared_dependencies()
    return _shared_get_portfolio_history()


@st.cache_data(ttl=CACHE_TTL)
def get_scoring_dates():
    _sync_shared_dependencies()
    return _shared_get_scoring_dates()


@st.cache_data(ttl=CACHE_TTL)
def get_latest_scoring_date():
    _sync_shared_dependencies()
    return _shared_get_latest_scoring_date()


@st.cache_data(ttl=CACHE_TTL)
def get_alpha_universe_diagnostics(scoring_date):
    _sync_shared_dependencies()
    return _shared_get_alpha_universe_diagnostics(scoring_date)


@st.cache_data(ttl=CACHE_TTL)
def get_alpha_eligible_company_ids(scoring_date):
    _sync_shared_dependencies()
    return _shared_get_alpha_eligible_company_ids(scoring_date)


@st.cache_data(ttl=CACHE_TTL)
def get_alpha_eligibility_reasons(scoring_date):
    _sync_shared_dependencies()
    return _shared_get_alpha_eligibility_reasons(scoring_date)


@st.cache_data(ttl=CACHE_TTL)
def get_alpha_snapshot_streaks(scoring_date):
    _sync_shared_dependencies()
    return _shared_get_alpha_snapshot_streaks(scoring_date)


@st.cache_data(ttl=CACHE_TTL)
def get_alpha_dashboard_diagnostics(scoring_date):
    _sync_shared_dependencies()
    return _shared_get_alpha_dashboard_diagnostics(scoring_date)


@st.cache_data(ttl=CACHE_TTL)
def get_scoring_results(
    scoring_date=None,
    company_type=None,
    is_bist100=None,
    sector_custom=None,
    risk_tier=None,
    min_score=None,
    alpha_eligible_only=False,
):
    _sync_shared_dependencies()
    return _shared_get_scoring_results(
        scoring_date=scoring_date,
        company_type=company_type,
        is_bist100=is_bist100,
        sector_custom=sector_custom,
        risk_tier=risk_tier,
        min_score=min_score,
        alpha_eligible_only=alpha_eligible_only,
    )


@st.cache_data(ttl=CACHE_TTL)
def get_sectors():
    _sync_shared_dependencies()
    return _shared_get_sectors()


@st.cache_data(ttl=CACHE_TTL)
def get_company_types():
    _sync_shared_dependencies()
    return _shared_get_company_types()


@st.cache_data(ttl=CACHE_TTL)
def get_company_info(ticker: str):
    _sync_shared_dependencies()
    return _shared_get_company_info(ticker)


@st.cache_data(ttl=CACHE_TTL)
def get_price_history(ticker: str, days: int = 365):
    _sync_shared_dependencies()
    return _shared_get_price_history(ticker, days)


@st.cache_data(ttl=CACHE_TTL)
def get_factor_scores(ticker: str):
    _sync_shared_dependencies()
    return _shared_get_factor_scores(ticker)


@st.cache_data(ttl=CACHE_TTL)
def get_adjusted_metrics(ticker: str):
    _sync_shared_dependencies()
    return _shared_get_adjusted_metrics(ticker)


@st.cache_data(ttl=CACHE_TTL)
def get_stock_position(ticker: str):
    _sync_shared_dependencies()
    return _shared_get_stock_position(ticker)


@st.cache_data(ttl=CACHE_TTL)
def get_latest_macro():
    _sync_shared_dependencies()
    return _shared_get_latest_macro()


@st.cache_data(ttl=CACHE_TTL)
def get_all_tickers():
    _sync_shared_dependencies()
    return _shared_get_all_tickers()
