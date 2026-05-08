"""Shared read-only data service for Streamlit and the mobile API.

Contains the uncached SQLAlchemy query helpers used by both the Streamlit
dashboard and the FastAPI layer.
"""

import json
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from bist_picker.db.connection import get_engine
from bist_picker.db.schema import (
    AdjustedMetric,
    CashAllocationState,
    Company,
    DailyPrice,
    MacroRegime,
    PortfolioSelection,
    ScoringResult,
)
from bist_picker.portfolio.selector import get_selection_target_count
from bist_picker.portfolio.universes import UniverseBuilder

logger = logging.getLogger(__name__)

CACHE_TTL = 300  # 5 minutes

_ALPHA_X_INCLUDED_TYPES = {"OPERATING", "REIT", "HOLDING", "INSURANCE", "FINANCIAL"}
_ALPHA_X_EXCLUDED_REASONS = {
    "BANK": "BANK ALPHA X disi",
    "SPORT": "SPORT ALPHA X disi",
    "INDEX": "INDEX ALPHA X disi",
}
_ALPHA_X_TYPE_MIN_DATA = {
    "OPERATING": 70.0,
    "REIT": 70.0,
    "HOLDING": 50.0,
    "INSURANCE": 60.0,
    "FINANCIAL": 60.0,
}
_ALPHA_X_TYPE_MATURITY = {
    "OPERATING": 1.00,
    "REIT": 0.92,
    "HOLDING": 0.82,
    "INSURANCE": 0.65,
    "FINANCIAL": 0.55,
}
_ALPHA_X_MIN_CONFIDENCE = 0.55
_ALPHA_X_FREE_FLOAT_MIN = 25.0
_ALPHA_X_AVG_VOLUME_MIN = 10_000_000.0
_ALPHA_X_PEER_TARGET = 40.0
_ALPHA_X_MARKET_OVERLAY_WEIGHT = 0.25
_ALPHA_X_BASE_WEIGHT = 0.85
_ALPHA_X_INVESTABILITY_WEIGHT = 0.15


def _get_session() -> Session:
    """Create a new session (caller must close)."""
    from sqlalchemy.orm import sessionmaker
    engine = get_engine()
    return sessionmaker(bind=engine)()


def _resolve_macro_value(
    session: Session,
    latest_row: MacroRegime,
    field_name: str,
):
    """Return the newest available value for a macro field and its source date."""
    value = getattr(latest_row, field_name)
    if value is not None:
        return value, latest_row.date

    column = getattr(MacroRegime, field_name)
    row = (
        session.query(MacroRegime.date, column)
        .filter(
            MacroRegime.date <= latest_row.date,
            column.isnot(None),
        )
        .order_by(MacroRegime.date.desc())
        .first()
    )
    if not row:
        return None, None
    return row[1], row[0]


def _load_latest_macro(session: Session) -> Optional[dict]:
    """Load the newest macro snapshot, backfilling missing fields safely."""
    row = (
        session.query(MacroRegime)
        .order_by(MacroRegime.date.desc())
        .first()
    )
    if not row:
        return None

    policy_rate_pct, policy_rate_date = _resolve_macro_value(
        session, row, "policy_rate_pct"
    )
    cpi_yoy_pct, cpi_yoy_date = _resolve_macro_value(
        session, row, "cpi_yoy_pct"
    )
    usdtry_rate, usdtry_date = _resolve_macro_value(
        session, row, "usdtry_rate"
    )
    turkey_cds_5y, turkey_cds_5y_date = _resolve_macro_value(
        session, row, "turkey_cds_5y"
    )
    regime, regime_date = _resolve_macro_value(session, row, "regime")

    return {
        "date": row.date,
        "policy_rate_pct": policy_rate_pct,
        "policy_rate_date": policy_rate_date,
        "cpi_yoy_pct": cpi_yoy_pct,
        "cpi_yoy_date": cpi_yoy_date,
        "usdtry_rate": usdtry_rate,
        "usdtry_date": usdtry_date,
        "turkey_cds_5y": turkey_cds_5y,
        "turkey_cds_5y_date": turkey_cds_5y_date,
        "regime": regime,
        "regime_date": regime_date,
    }


# -- Portfolio & Holdings --


def get_open_positions() -> pd.DataFrame:
    """Fetch the latest open ALPHA portfolio snapshot.

    Phase 5: joins the matching ``ScoringResult`` for each pick's
    ``selection_date`` so the returned frame carries the DCF breakdown and
    red-flag payload needed by the APK pick-detail view. Also computes
    ``stop_pct_from_entry`` — a derived field that answers "how much downside
    does my stop protect against from my entry?" without the APK having to
    do the math.
    """
    session = _get_session()
    try:
        latest_open_date = (
            session.query(PortfolioSelection.selection_date)
            .filter(
                PortfolioSelection.exit_date.is_(None),
                PortfolioSelection.portfolio == "ALPHA",
            )
            .order_by(PortfolioSelection.selection_date.desc())
            .first()
        )
        if not latest_open_date:
            return pd.DataFrame()

        rows = (
            session.query(PortfolioSelection, Company)
            .join(Company, Company.id == PortfolioSelection.company_id)
            .filter(
                PortfolioSelection.exit_date.is_(None),
                PortfolioSelection.portfolio == "ALPHA",
                PortfolioSelection.selection_date == latest_open_date[0],
            )
            .order_by(PortfolioSelection.composite_score.desc())
            .all()
        )

        if not rows:
            return pd.DataFrame()

        # Phase 5: bulk-load the matching scoring rows (same company_ids at the
        # same selection_date) so every pick can surface its DCF breakdown +
        # red flags without an N+1 query.
        company_ids = [company.id for _, company in rows]
        scoring_rows = (
            session.query(ScoringResult)
            .filter(
                ScoringResult.company_id.in_(company_ids),
                ScoringResult.scoring_date == latest_open_date[0],
            )
            .all()
        )
        scoring_by_company: dict[int, ScoringResult] = {
            sr.company_id: sr for sr in scoring_rows
        }

        records = []
        today = date.today()
        price_by_company = _latest_prices(session, company_ids)
        for pos, company in rows:
            current_price = price_by_company.get(company.id)
            entry = pos.entry_price
            pnl_pct = None
            if entry and current_price and entry > 0:
                pnl_pct = (current_price - entry) / entry * 100.0

            days_held = None
            if pos.selection_date:
                days_held = (today - pos.selection_date).days

            # Phase 5 derived: stop distance expressed as percent of entry.
            # Positive values = stop below entry (the normal case).
            stop_pct_from_entry = None
            if entry and entry > 0 and pos.stop_loss_price is not None:
                stop_pct_from_entry = (entry - pos.stop_loss_price) / entry * 100.0

            sr = scoring_by_company.get(company.id)
            records.append({
                "portfolio": pos.portfolio,
                "ticker": company.ticker,
                "name": company.name,
                "company_id": company.id,
                "entry_price": entry,
                "current_price": current_price,
                "pnl_pct": pnl_pct,
                "target_price": pos.target_price,
                "stop_loss_price": pos.stop_loss_price,
                "stop_pct_from_entry": stop_pct_from_entry,
                "composite_score": pos.composite_score,
                "selection_date": pos.selection_date,
                "days_held": days_held,
                # Phase 5 transparency payload. Stored as JSON text so the
                # wire format doesn't change as we evolve the shape; callers
                # that want structure call ``json.loads`` on demand.
                "reason_top_factors_json": pos.reason_top_factors_json,
                "quality_flags_json": sr.quality_flags_json if sr else None,
                "dcf_margin_of_safety_pct": (
                    sr.dcf_margin_of_safety_pct if sr else None
                ),
                "dcf_intrinsic_value": (
                    sr.dcf_intrinsic_value if sr else None
                ),
                "dcf_growth_rate_pct": (
                    sr.dcf_growth_rate_pct if sr else None
                ),
                "dcf_discount_rate_pct": (
                    sr.dcf_discount_rate_pct if sr else None
                ),
                "dcf_terminal_growth_pct": (
                    sr.dcf_terminal_growth_pct if sr else None
                ),
            })

        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.head(get_selection_target_count()).reset_index(drop=True)
    finally:
        session.close()


def get_portfolio_performance(portfolio_name: str) -> dict:
    """Calculate portfolio performance metrics.

    Returns dict with: total_return_avg, active_return_avg, win_rate
    """
    session = _get_session()
    try:
        try:
            from bist_picker.output.performance import PerformanceTracker
        except ModuleNotFoundError:
            logger.warning(
                "Legacy PerformanceTracker module is unavailable; using read_service fallback."
            )
            return _calculate_portfolio_performance_fallback(session, portfolio_name)

        tracker = PerformanceTracker(session)
        return tracker.calculate_portfolio_performance(portfolio_name)
    finally:
        session.close()


def get_all_portfolio_performance() -> dict:
    """Get performance for all portfolios combined."""
    session = _get_session()
    try:
        try:
            from bist_picker.output.performance import PerformanceTracker
        except ModuleNotFoundError:
            result = _calculate_portfolio_performance_fallback(session, "ALPHA")
            result["benchmark_ytd"] = None
            return result

        tracker = PerformanceTracker(session)
        result = tracker.calculate_portfolio_performance("ALPHA")
        result["benchmark_ytd"] = tracker.fetch_benchmark_performance()
        return result
    finally:
        session.close()


def get_portfolio_history() -> pd.DataFrame:
    """Fetch all CLOSED portfolio positions for ALPHA (exit_date IS NOT NULL)."""
    session = _get_session()
    try:
        rows = (
            session.query(PortfolioSelection, Company)
            .join(Company, Company.id == PortfolioSelection.company_id)
            .filter(
                PortfolioSelection.exit_date.isnot(None),
                PortfolioSelection.portfolio == "ALPHA"
            )
            .order_by(PortfolioSelection.exit_date.desc())
            .all()
        )

        if not rows:
            return pd.DataFrame()

        records = []
        for pos, company in rows:
            entry = pos.entry_price
            exit_p = pos.exit_price
            pnl_pct = None
            if entry and exit_p and entry > 0:
                pnl_pct = (exit_p - entry) / entry * 100.0

            records.append({
                "portfolio": pos.portfolio,
                "ticker": company.ticker,
                "name": company.name,
                "selection_date": pos.selection_date,
                "exit_date": pos.exit_date,
                "entry_price": entry,
                "exit_price": exit_p,
                "pnl_pct": pnl_pct,
            })

        return pd.DataFrame(records)
    finally:
        session.close()


# -- Scoring --


def get_scoring_dates() -> list[date]:
    """Return all distinct scoring dates, most recent first."""
    session = _get_session()
    try:
        rows = (
            session.query(ScoringResult.scoring_date)
            .distinct()
            .order_by(ScoringResult.scoring_date.desc())
            .all()
        )
        return [r[0] for r in rows]
    finally:
        session.close()


def get_latest_scoring_date() -> Optional[date]:
    """Return the latest scoring date, or None when no scores exist."""
    session = _get_session()
    try:
        latest = (
            session.query(ScoringResult.scoring_date)
            .order_by(ScoringResult.scoring_date.desc())
            .first()
        )
        return latest[0] if latest else None
    finally:
        session.close()


def _alpha_bucket_from_diag(diag: dict) -> str:
    """Return the normalized ALPHA bucket name for a diagnostic row."""
    if diag.get("alpha_core_eligible") or diag.get("eligible"):
        return "ALPHA Core"
    return diag.get("alpha_research_bucket") or "Excluded"


def _alpha_note_from_diag(diag: dict) -> str:
    """Return the concise ALPHA note shown in the scoring table."""
    if diag.get("alpha_core_eligible"):
        return "ALPHA Core Uygun"

    bucket = diag.get("alpha_research_bucket")
    if bucket == "Quality Shadow":
        return "Quality Shadow adayi"
    if bucket == "Free-Float Shadow":
        return "Free-Float Shadow adayi"
    if bucket == "Non-Core Research":
        return "Research only"
    if bucket == "Data-Unscorable":
        return "Data-Unscorable"
    return diag.get("alpha_primary_blocker") or "ALPHA Core disi"


def _normalized_company_type(company_type: Optional[str]) -> str:
    """Return uppercase company type with a safe fallback."""
    return (company_type or "").upper() or "UNKNOWN"


def _model_family(company_type: Optional[str]) -> str:
    """Group company types into dashboard-friendly model families."""
    normalized = _normalized_company_type(company_type)
    if normalized in {"BANK", "FINANCIAL", "INSURANCE"}:
        return "Financials"
    if normalized == "HOLDING":
        return "Holding"
    if normalized == "REIT":
        return "REIT"
    if normalized == "SPORT":
        return "Sport"
    return "Operating"


def _native_model_score(
    company_type: Optional[str],
    score: Optional[ScoringResult],
) -> Optional[float]:
    """Return the primary model score for the company's own scoring model."""
    if score is None:
        return None

    normalized = _normalized_company_type(company_type)
    if normalized in {"BANK", "FINANCIAL", "INSURANCE"}:
        return score.banking_composite
    if normalized == "HOLDING":
        return score.holding_composite
    if normalized == "REIT":
        return score.reit_composite
    return None


def _weighted_average(parts: list[tuple[Optional[float], float]]) -> Optional[float]:
    """Return a weighted average over the non-null values in *parts*."""
    available = [
        (float(value), weight)
        for value, weight in parts
        if value is not None and not pd.isna(value)
    ]
    if not available:
        return None
    total_weight = sum(weight for _, weight in available)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in available) / total_weight


def _factor_group_average(*values: Optional[float]) -> Optional[float]:
    """Return an equal-weight average for a factor group."""
    return _weighted_average([(value, 1.0) for value in values])


def _market_fallback_score(score: Optional[ScoringResult]) -> Optional[float]:
    """Fallback ranking score when no native model score is available."""
    if score is None:
        return None

    return _weighted_average(
        [
            (score.momentum_score, 0.45),
            (score.technical_score, 0.35),
            (score.dividend_score, 0.20),
        ]
    )


def _ranking_score_bundle(
    company_type: Optional[str],
    score: Optional[ScoringResult],
    alpha_score: Optional[float],
) -> tuple[Optional[float], str]:
    """Return the score used for ranking together with its source label."""
    normalized = _normalized_company_type(company_type)
    native = _native_model_score(normalized, score)

    if normalized == "SPORT":
        return None, "Unscored"
    if normalized == "OPERATING":
        if alpha_score is not None:
            return alpha_score, "ALPHA"
        market_fallback = _market_fallback_score(score)
        if market_fallback is not None:
            return market_fallback, "Market Fallback"
        return None, "Unscored"
    if native is not None:
        if normalized in {"BANK", "FINANCIAL", "INSURANCE"}:
            return native, "Banking Model"
        if normalized == "HOLDING":
            return native, "Holding Model"
        if normalized == "REIT":
            return native, "REIT Model"
    if alpha_score is not None:
        return alpha_score, "Alpha Fallback"

    market_fallback = _market_fallback_score(score)
    if market_fallback is not None:
        return market_fallback, "Market Fallback"
    return None, "Unscored"


def _alpha_x_min_data(company_type: Optional[str]) -> float:
    """Return the minimum data coverage expected for ALPHA X by company type."""
    normalized = _normalized_company_type(company_type)
    return float(_ALPHA_X_TYPE_MIN_DATA.get(normalized, 70.0))


def _bounded_ratio_score(
    value: Optional[float],
    floor: float,
    cap: float,
) -> float:
    """Map a positive metric onto a soft 0-100 score."""
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric <= 0:
        return 0.0
    if numeric <= floor:
        return max(0.0, min(50.0, 50.0 * (numeric / floor)))
    if numeric >= cap:
        return 100.0
    return 50.0 + 50.0 * ((numeric - floor) / (cap - floor))


def _alpha_x_risk_score(risk: Optional[str]) -> float:
    """Return a 0-100 investability score for the risk tier."""
    mapping = {"LOW": 100.0, "MEDIUM": 70.0, "HIGH": 15.0}
    # Guard against NaN/non-string values that pandas coerces in mixed columns;
    # NaN is falsy in `if not risk` but truthy in `risk or ""` (it's a float).
    if not isinstance(risk, str):
        return 50.0
    return mapping.get(risk.upper(), 50.0)


def _alpha_x_signal_raw(row: pd.Series) -> Optional[float]:
    """Return the raw signal used as the first-step ALPHA X anchor."""
    company_type = _normalized_company_type(row.get("type"))
    if company_type == "OPERATING":
        return row.get("alpha")
    if bool(row.get("has_native_model_score")):
        return row.get("model_score")
    if row.get("ranking_source") == "Alpha Fallback":
        return row.get("alpha")
    if row.get("ranking_source") == "Market Fallback":
        return row.get("ranking_score")
    return None


def _alpha_x_market_overlay(row: pd.Series) -> Optional[float]:
    """Return the shared market-behaviour overlay used across company types."""
    return _weighted_average(
        [
            (row.get("momentum"), 0.60),
            (row.get("technical"), 0.40),
        ]
    )


def _alpha_x_investability(row: pd.Series) -> float:
    """Return the soft investability score for ALPHA X ordering."""
    return float(
        _weighted_average(
            [
                (_alpha_x_risk_score(row.get("risk")), 0.50),
                (
                    _bounded_ratio_score(
                        row.get("free_float_pct"),
                        _ALPHA_X_FREE_FLOAT_MIN,
                        50.0,
                    ),
                    0.25,
                ),
                (
                    _bounded_ratio_score(
                        row.get("avg_volume_try"),
                        _ALPHA_X_AVG_VOLUME_MIN,
                        100_000_000.0,
                    ),
                    0.25,
                ),
            ]
        )
        or 0.0
    )


def _alpha_x_confidence(
    row: pd.Series,
    native_peers_by_type: dict[str, int],
) -> float:
    """Return the reliability multiplier for cross-type score calibration."""
    company_type = _normalized_company_type(row.get("type"))
    signal_raw = row.get("alpha_x_signal_raw")
    if signal_raw is None or (isinstance(signal_raw, float) and pd.isna(signal_raw)):
        return 0.0
    if company_type == "OPERATING":
        return 1.0 if pd.notna(row.get("alpha")) else 0.0

    maturity = float(_ALPHA_X_TYPE_MATURITY.get(company_type, 0.0))
    if maturity <= 0:
        return 0.0

    completeness = row.get("data_completeness")
    if completeness is None or pd.isna(completeness):
        coverage_factor = 0.0
    else:
        coverage_ratio = max(0.0, min(1.0, float(completeness) / 100.0))
        coverage_factor = coverage_ratio ** 0.5

    peer_count = float(native_peers_by_type.get(company_type, 0))
    peer_factor = 0.70 + 0.30 * min(1.0, peer_count / _ALPHA_X_PEER_TARGET)

    if bool(row.get("has_native_model_score")):
        source_factor = 1.0
    elif row.get("ranking_source") == "Alpha Fallback":
        source_factor = 0.55
    elif row.get("ranking_source") == "Market Fallback":
        source_factor = 0.35
    else:
        source_factor = 0.0

    confidence = maturity * coverage_factor * peer_factor * source_factor
    return max(0.0, min(1.0, float(confidence)))


def _alpha_x_base_score(row: pd.Series) -> Optional[float]:
    """Return the calibrated signal before the final investability blend."""
    signal_raw = row.get("alpha_x_signal_raw")
    if signal_raw is None or pd.isna(signal_raw):
        return None

    company_type = _normalized_company_type(row.get("type"))
    if company_type == "OPERATING":
        return float(signal_raw)

    confidence = row.get("alpha_x_confidence")
    if confidence is None or pd.isna(confidence):
        confidence = 0.0

    calibrated_signal = 50.0 + (float(signal_raw) - 50.0) * float(confidence)
    market_overlay = row.get("alpha_x_market_overlay")
    if market_overlay is None or pd.isna(market_overlay):
        return float(calibrated_signal)

    return _weighted_average(
        [
            (calibrated_signal, 1.0 - _ALPHA_X_MARKET_OVERLAY_WEIGHT),
            (market_overlay, _ALPHA_X_MARKET_OVERLAY_WEIGHT),
        ]
    )


def _alpha_x_is_eligible(row: pd.Series) -> bool:
    """Return True when the company qualifies for the scoring-only ALPHA X universe."""
    company_type = _normalized_company_type(row.get("type"))
    if company_type not in _ALPHA_X_INCLUDED_TYPES:
        return False

    signal_raw = row.get("alpha_x_signal_raw")
    if signal_raw is None or pd.isna(signal_raw):
        return False

    if company_type == "OPERATING":
        return bool(row.get("alpha_core_eligible"))

    if not bool(row.get("has_native_model_score")):
        return False
    if (row.get("risk") or "").upper() == "HIGH":
        return False

    free_float = row.get("free_float_pct")
    if free_float is None or pd.isna(free_float) or float(free_float) < _ALPHA_X_FREE_FLOAT_MIN:
        return False

    avg_volume_try = row.get("avg_volume_try")
    if avg_volume_try is None or pd.isna(avg_volume_try) or float(avg_volume_try) < _ALPHA_X_AVG_VOLUME_MIN:
        return False

    completeness = row.get("data_completeness")
    min_data = _alpha_x_min_data(company_type)
    if completeness is None or pd.isna(completeness) or float(completeness) < min_data:
        return False

    confidence = row.get("alpha_x_confidence")
    if confidence is None or pd.isna(confidence) or float(confidence) < _ALPHA_X_MIN_CONFIDENCE:
        return False

    return True


def _alpha_x_reason(row: pd.Series) -> str:
    """Return the primary explanation for a company's ALPHA X status."""
    company_type = _normalized_company_type(row.get("type"))
    if company_type in _ALPHA_X_EXCLUDED_REASONS:
        return _ALPHA_X_EXCLUDED_REASONS[company_type]
    if company_type not in _ALPHA_X_INCLUDED_TYPES:
        return "ALPHA X kapsami disi"

    signal_raw = row.get("alpha_x_signal_raw")
    if signal_raw is None or pd.isna(signal_raw):
        return "Kullanilabilir sinyal yok"

    if company_type == "OPERATING" and not bool(row.get("alpha_core_eligible")):
        return row.get("alpha_reason") or "ALPHA Core disi"

    if company_type != "OPERATING" and not bool(row.get("has_native_model_score")):
        return "Native model yok"

    if (row.get("risk") or "").upper() == "HIGH":
        return "Risk tier = HIGH"

    free_float = row.get("free_float_pct")
    if free_float is None or pd.isna(free_float):
        return "Halka aciklik verisi yok"
    if float(free_float) < _ALPHA_X_FREE_FLOAT_MIN:
        return f"Halka aciklik <%{_ALPHA_X_FREE_FLOAT_MIN:.0f}"

    avg_volume_try = row.get("avg_volume_try")
    if avg_volume_try is None or pd.isna(avg_volume_try):
        return "Likidite verisi yok"
    if float(avg_volume_try) < _ALPHA_X_AVG_VOLUME_MIN:
        return f"Likidite < {_ALPHA_X_AVG_VOLUME_MIN / 1_000_000:.0f}M TRY"

    completeness = row.get("data_completeness")
    min_data = _alpha_x_min_data(company_type)
    if completeness is None or pd.isna(completeness):
        return "Veri kapsami yok"
    if float(completeness) < min_data:
        return f"Veri kapsami <%{min_data:.0f}"

    confidence = row.get("alpha_x_confidence")
    if confidence is None or pd.isna(confidence) or float(confidence) < _ALPHA_X_MIN_CONFIDENCE:
        return "Model confidence dusuk"

    return "ALPHA X Uygun"


def _alpha_x_bucket(row: pd.Series) -> str:
    """Return the diagnostic bucket for ALPHA X."""
    if bool(row.get("alpha_x_eligible")):
        return "ALPHA X"

    company_type = _normalized_company_type(row.get("type"))
    if company_type in _ALPHA_X_EXCLUDED_REASONS:
        return "Type Excluded"
    if company_type == "OPERATING":
        return "Operating Shadow"
    if row.get("alpha_x_reason") == "Native model yok":
        return "Native Shadow"
    if row.get("alpha_x_reason") == "Model confidence dusuk":
        return "Confidence Shadow"
    if row.get("alpha_x_reason") == "Kullanilabilir sinyal yok":
        return "Data-Unscorable"
    return "Investability Excluded"


def _apply_alpha_x_fields(
    df: pd.DataFrame,
    native_peers_by_type: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Attach scoring-only ALPHA X fields to the dashboard dataframe."""
    if df.empty:
        return df

    if native_peers_by_type is None:
        native_peers_by_type = (
            df[df["has_native_model_score"].fillna(False)]
            .groupby("type")
            .size()
            .to_dict()
        )

    result = df.copy()
    result["alpha_x_signal_raw"] = result.apply(_alpha_x_signal_raw, axis=1)
    result["alpha_x_market_overlay"] = result.apply(_alpha_x_market_overlay, axis=1)
    result["alpha_x_investability"] = result.apply(_alpha_x_investability, axis=1)
    result["alpha_x_confidence"] = result.apply(
        lambda row: _alpha_x_confidence(row, native_peers_by_type),
        axis=1,
    )
    result["alpha_x_base_score"] = result.apply(_alpha_x_base_score, axis=1)
    result["alpha_x_score"] = result.apply(
        lambda row: (
            _weighted_average(
                [
                    (row.get("alpha_x_base_score"), _ALPHA_X_BASE_WEIGHT),
                    (row.get("alpha_x_investability"), _ALPHA_X_INVESTABILITY_WEIGHT),
                ]
            )
            if pd.notna(row.get("alpha_x_base_score"))
            else None
        ),
        axis=1,
    )
    result["alpha_x_eligible"] = result.apply(_alpha_x_is_eligible, axis=1)
    result["alpha_x_reason"] = result.apply(_alpha_x_reason, axis=1)
    result["alpha_x_status"] = result["alpha_x_eligible"].map(
        lambda value: "ALPHA X Uygun" if value else "ALPHA X Disi"
    )
    result["alpha_x_bucket"] = result.apply(_alpha_x_bucket, axis=1)
    result["alpha_x_rank"] = (
        result["alpha_x_score"]
        .where(result["alpha_x_eligible"])
        .rank(method="dense", ascending=False, na_option="keep")
    )
    return result


def _apply_alpha_debug_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Attach grouped-factor diagnostics used to explain ALPHA and ALPHA X ranks."""
    if df.empty:
        return df

    result = df.copy()
    result["alpha_value_group"] = result.apply(
        lambda row: (
            _factor_group_average(row.get("graham"), row.get("dcf_mos"))
            if _normalized_company_type(row.get("type")) == "OPERATING"
            else None
        ),
        axis=1,
    )
    result["alpha_growth_group"] = result.apply(
        lambda row: (
            _factor_group_average(row.get("magic_formula"), row.get("lynch_peg"))
            if _normalized_company_type(row.get("type")) == "OPERATING"
            else None
        ),
        axis=1,
    )

    def _missing_groups(row: pd.Series) -> Optional[str]:
        if _normalized_company_type(row.get("type")) != "OPERATING":
            return None
        groups = {
            "Quality": row.get("buffett"),
            "Value": row.get("alpha_value_group"),
            "Piotroski": row.get("piotroski"),
            "Growth": row.get("alpha_growth_group"),
            "Momentum": row.get("momentum"),
            "Technical": row.get("technical"),
        }
        missing = [label for label, value in groups.items() if value is None or pd.isna(value)]
        return ", ".join(missing) if missing else "Tam"

    result["alpha_missing_groups"] = result.apply(_missing_groups, axis=1)
    result["alpha_x_delta"] = result.apply(
        lambda row: (
            float(row.get("alpha_x_score")) - float(row.get("alpha"))
            if pd.notna(row.get("alpha_x_score")) and pd.notna(row.get("alpha"))
            else None
        ),
        axis=1,
    )
    return result


def _build_fallback_alpha_diag(
    company: Company,
    score: Optional[ScoringResult],
) -> dict[str, object]:
    """Return an ALPHA diagnostic row for companies outside active exact-date snapshots."""
    if not bool(company.is_active):
        primary = "Pasif sirket"
        bucket = "Excluded"
        status = "Pasif"
        note = primary
    elif score is None:
        primary = "Skor snapshot yok"
        bucket = "Data-Unscorable"
        status = "Skor snapshot yok"
        note = "Data-Unscorable"
    else:
        primary = "Data-Unscorable"
        bucket = "Data-Unscorable"
        status = "ALPHA Core Disi"
        note = primary

    return {
        "alpha_core_eligible": False,
        "alpha_primary_blocker": primary,
        "alpha_all_blockers": primary,
        "alpha_research_bucket": bucket,
        "alpha_snapshot_streak": 0,
        "alpha_relaxed_p4_eligible": False,
        "alpha_core_status": status,
        "alpha_reason": note,
    }


def get_alpha_universe_diagnostics(scoring_date: date) -> dict[int, dict[str, object]]:
    """Return raw ALPHA diagnostics for the exact scoring snapshot."""
    session = _get_session()
    try:
        return UniverseBuilder(scoring_date=scoring_date).get_universe_diagnostics(
            "ALPHA",
            session,
            exact_date=True,
        )
    finally:
        session.close()


def get_alpha_eligible_company_ids(scoring_date: date) -> set[int]:
    """Return the cached ALPHA investable universe for a scoring date."""
    diagnostics = get_alpha_universe_diagnostics(scoring_date)
    return {
        company_id
        for company_id, info in diagnostics.items()
        if bool(info.get("alpha_core_eligible", info.get("eligible")))
    }


def get_alpha_eligibility_reasons(scoring_date: date) -> dict[int, str]:
    """Return per-company ALPHA eligibility explanations for a scoring date."""
    diagnostics = get_alpha_universe_diagnostics(scoring_date)
    return {
        company_id: "; ".join(info.get("alpha_all_blockers") or info.get("reasons") or [])
        for company_id, info in diagnostics.items()
        if not bool(info.get("alpha_core_eligible", info.get("eligible")))
    }


def get_alpha_snapshot_streaks(scoring_date: date) -> dict[int, int]:
    """Return consecutive same-bucket streaks up to *scoring_date*."""
    diagnostics = get_alpha_universe_diagnostics(scoring_date)
    current_buckets = {
        company_id: _alpha_bucket_from_diag(info)
        for company_id, info in diagnostics.items()
    }
    if not current_buckets:
        return {}

    streaks = {company_id: 1 for company_id in current_buckets}
    active = dict(current_buckets)

    prior_dates = [d for d in get_scoring_dates() if d < scoring_date]
    for prior_date in prior_dates:
        prior_diag = get_alpha_universe_diagnostics(prior_date)
        prior_buckets = {
            company_id: _alpha_bucket_from_diag(info)
            for company_id, info in prior_diag.items()
        }
        next_active: dict[int, str] = {}
        for company_id, bucket in active.items():
            if prior_buckets.get(company_id) == bucket:
                streaks[company_id] += 1
                next_active[company_id] = bucket
        if not next_active:
            break
        active = next_active

    return streaks


def get_alpha_dashboard_diagnostics(scoring_date: date) -> dict[int, dict[str, object]]:
    """Return ALPHA dashboard diagnostics enriched with streaks and note text."""
    diagnostics = get_alpha_universe_diagnostics(scoring_date)
    streaks = get_alpha_snapshot_streaks(scoring_date)

    rows: dict[int, dict[str, object]] = {}
    for company_id, info in diagnostics.items():
        blockers = list(info.get("alpha_all_blockers") or info.get("reasons") or [])
        core_eligible = bool(info.get("alpha_core_eligible", info.get("eligible")))
        row = {
            "alpha_core_eligible": core_eligible,
            "alpha_primary_blocker": info.get("alpha_primary_blocker"),
            "alpha_all_blockers": "; ".join(blockers),
            "alpha_research_bucket": _alpha_bucket_from_diag(info),
            "alpha_snapshot_streak": streaks.get(company_id, 1),
            "alpha_relaxed_p4_eligible": bool(
                info.get("alpha_relaxed_p4_eligible", core_eligible)
            ),
            "alpha_core_status": "ALPHA Core Uygun" if core_eligible else "ALPHA Core Disi",
        }
        row["alpha_reason"] = _alpha_note_from_diag({**info, **row})
        rows[company_id] = row
    return rows


def _load_avg_volume_try_map(
    session: Session,
    scoring_date: date,
    lookback_days: int = 30,
) -> dict[int, float]:
    """Return average daily TRY turnover for the selected snapshot window."""
    cutoff = scoring_date - timedelta(days=lookback_days)
    turnover_expr = case(
        (DailyPrice.source.ilike("YAHOO%"), DailyPrice.close * DailyPrice.volume),
        else_=DailyPrice.volume,
    )
    rows = (
        session.query(
            DailyPrice.company_id,
            func.avg(turnover_expr).label("avg_turnover"),
        )
        .filter(DailyPrice.date >= cutoff)
        .filter(DailyPrice.date <= scoring_date)
        .filter(DailyPrice.close.isnot(None))
        .filter(DailyPrice.volume.isnot(None))
        .group_by(DailyPrice.company_id)
        .all()
    )
    return {row.company_id: float(row.avg_turnover or 0.0) for row in rows}


def _load_alpha_x_native_peer_counts(
    session: Session,
    scoring_date: date,
) -> dict[str, int]:
    """Return native-model peer counts per company type for the full snapshot."""
    rows = (
        session.query(
            Company.company_type,
            ScoringResult.banking_composite,
            ScoringResult.holding_composite,
            ScoringResult.reit_composite,
        )
        .join(Company, Company.id == ScoringResult.company_id)
        .filter(
            ScoringResult.scoring_date == scoring_date,
            Company.is_active == True,
        )
        .all()
    )

    counts: dict[str, int] = {}
    for company_type, banking_composite, holding_composite, reit_composite in rows:
        normalized = _normalized_company_type(company_type)
        has_native = (
            (normalized in {"BANK", "FINANCIAL", "INSURANCE"} and banking_composite is not None)
            or (normalized == "HOLDING" and holding_composite is not None)
            or (normalized == "REIT" and reit_composite is not None)
        )
        if has_native:
            counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def get_scoring_results(
    scoring_date: Optional[date] = None,
    company_type: Optional[str] = None,
    is_bist100: Optional[bool] = None,
    sector_custom: Optional[str] = None,
    risk_tier: Optional[str] = None,
    min_score: Optional[float] = None,
    alpha_eligible_only: bool = False,
) -> pd.DataFrame:
    """Fetch scoring results with optional filters.

    Returns DataFrame with ticker, company info, all factor scores,
    composites, risk tier, and company activity/scoring flags.

    When ALPHA-only filtering is off, this includes every company in the
    database for the selected scoring date, even if the company has no score
    snapshot for that date.
    """
    session = _get_session()
    try:
        if scoring_date:
            effective_scoring_date = scoring_date
        else:
            effective_scoring_date = get_latest_scoring_date()
            if effective_scoring_date is None:
                return pd.DataFrame()

        query = (
            session.query(Company, ScoringResult)
            .outerjoin(
                ScoringResult,
                (Company.id == ScoringResult.company_id)
                & (ScoringResult.scoring_date == effective_scoring_date),
            )
        )

        if company_type:
            query = query.filter(Company.company_type == company_type)
        if is_bist100 is not None:
            query = query.filter(Company.is_bist100 == is_bist100)
        if sector_custom:
            query = query.filter(Company.sector_custom == sector_custom)
        if risk_tier:
            query = query.filter(ScoringResult.risk_tier == risk_tier)

        rows = query.all()

        if not rows:
            return pd.DataFrame()

        alpha_diag_map: dict[int, dict[str, object]] = {}
        if effective_scoring_date is not None:
            alpha_diag_map = get_alpha_dashboard_diagnostics(effective_scoring_date)
        avg_volume_try_map = _load_avg_volume_try_map(session, effective_scoring_date)
        alpha_x_native_peer_counts = _load_alpha_x_native_peer_counts(
            session,
            effective_scoring_date,
        )

        records = []
        for company, score in rows:
            alpha = score.composite_alpha if score else None
            if (company.company_type or "").upper() == "SPORT":
                alpha = None
            if min_score is not None and (alpha is None or alpha < min_score):
                continue

            company_type_value = _normalized_company_type(company.company_type)
            model_score = _native_model_score(company_type_value, score)
            ranking_score, ranking_source = _ranking_score_bundle(
                company_type_value,
                score,
                alpha,
            )

            alpha_diag = alpha_diag_map.get(company.id)
            if alpha_diag is None:
                alpha_diag = _build_fallback_alpha_diag(company, score)

            alpha_core_eligible = bool(alpha_diag["alpha_core_eligible"])
            if alpha_eligible_only and not alpha_core_eligible:
                continue

            records.append({
                "ticker": company.ticker,
                "name": company.name,
                "type": company_type_value,
                "model_family": _model_family(company_type_value),
                "sector": company.sector_custom or company.sector_bist,
                "bist100": company.is_bist100,
                "free_float_pct": company.free_float_pct,
                "avg_volume_try": avg_volume_try_map.get(company.id, 0.0),
                "buffett": score.buffett_score if score else None,
                "graham": score.graham_score if score else None,
                "piotroski": score.piotroski_fscore if score else None,
                "piotroski_raw": score.piotroski_fscore_raw if score else None,
                "magic_formula": score.magic_formula_rank if score else None,
                "lynch_peg": score.lynch_peg_score if score else None,
                "dcf_mos": score.dcf_margin_of_safety_pct if score else None,
                "momentum": score.momentum_score if score else None,
                "insider": score.insider_score if score else None,
                "technical": score.technical_score if score else None,
                "dividend": score.dividend_score if score else None,
                "model_score": model_score,
                "ranking_score": ranking_score,
                "ranking_source": ranking_source,
                "has_native_model_score": bool(model_score is not None),
                "ranking_uses_fallback": ranking_source in {"Alpha Fallback", "Market Fallback"},
                "alpha": alpha,
                "risk": score.risk_tier if score else None,
                "model": score.model_used if score else None,
                "data_completeness": score.data_completeness if score else None,
                "banking_composite": score.banking_composite if score else None,
                "holding_composite": score.holding_composite if score else None,
                "reit_composite": score.reit_composite if score else None,
                "scoring_date": score.scoring_date if score else None,
                "alpha_eligible": alpha_core_eligible,
                "alpha_core_eligible": alpha_core_eligible,
                "alpha_core_status": alpha_diag["alpha_core_status"],
                "alpha_primary_blocker": alpha_diag["alpha_primary_blocker"],
                "alpha_all_blockers": alpha_diag["alpha_all_blockers"],
                "alpha_research_bucket": alpha_diag["alpha_research_bucket"],
                "alpha_snapshot_streak": alpha_diag["alpha_snapshot_streak"],
                "alpha_relaxed_p4_eligible": alpha_diag["alpha_relaxed_p4_eligible"],
                "alpha_reason": alpha_diag["alpha_reason"],
                "is_active": bool(company.is_active),
                "has_score": score is not None,
            })

        result = pd.DataFrame(records)
        if result.empty:
            return result
        if "type" in result.columns:
            result["type_rank"] = (
                result.groupby("type")["ranking_score"]
                .rank(method="dense", ascending=False, na_option="bottom")
            )
        if "model_family" in result.columns:
            result["family_rank"] = (
                result.groupby("model_family")["ranking_score"]
                .rank(method="dense", ascending=False, na_option="bottom")
            )
        result = _apply_alpha_x_fields(result, alpha_x_native_peer_counts)
        result = _apply_alpha_debug_fields(result)
        return result.sort_values(
            ["has_score", "alpha_core_eligible", "ranking_score", "alpha", "is_active", "ticker"],
            ascending=[False, False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
    finally:
        session.close()


def get_sectors() -> list[str]:
    """Return all distinct custom sectors."""
    session = _get_session()
    try:
        rows = (
            session.query(Company.sector_custom)
            .filter(Company.sector_custom.isnot(None))
            .distinct()
            .order_by(Company.sector_custom)
            .all()
        )
        return [r[0] for r in rows]
    finally:
        session.close()


def get_company_types() -> list[str]:
    """Return all distinct company types."""
    session = _get_session()
    try:
        rows = (
            session.query(Company.company_type)
            .filter(Company.company_type.isnot(None))
            .distinct()
            .order_by(Company.company_type)
            .all()
        )
        return [r[0] for r in rows]
    finally:
        session.close()


# -- Single Stock Detail --


def get_company_info(ticker: str) -> Optional[dict]:
    """Fetch company info for a single ticker."""
    session = _get_session()
    try:
        company = (
            session.query(Company)
            .filter(Company.ticker == ticker.upper())
            .first()
        )
        if not company:
            return None
        return {
            "id": company.id,
            "ticker": company.ticker,
            "name": company.name,
            "company_type": company.company_type,
            "sector_bist": company.sector_bist,
            "sector_custom": company.sector_custom,
            "is_bist100": company.is_bist100,
            "is_ipo": company.is_ipo,
            "free_float_pct": company.free_float_pct,
            "listing_date": company.listing_date,
            "is_active": company.is_active,
        }
    finally:
        session.close()


def get_price_history(ticker: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily price history for a ticker.

    Returns DataFrame with columns: date, open, high, low, close, volume, adjusted_close
    """
    session = _get_session()
    try:
        company = session.query(Company).filter(Company.ticker == ticker.upper()).first()
        if not company:
            return pd.DataFrame()

        from datetime import timedelta
        cutoff = date.today() - timedelta(days=days)

        rows = (
            session.query(DailyPrice)
            .filter(
                DailyPrice.company_id == company.id,
                DailyPrice.date >= cutoff,
            )
            .order_by(DailyPrice.date)
            .all()
        )

        if not rows:
            return pd.DataFrame()

        records = [{
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "adjusted_close": r.adjusted_close,
        } for r in rows]

        return pd.DataFrame(records)
    finally:
        session.close()


def get_factor_scores(ticker: str) -> Optional[dict]:
    """Fetch latest factor scores for a single ticker."""
    session = _get_session()
    try:
        company = session.query(Company).filter(Company.ticker == ticker.upper()).first()
        if not company:
            return None

        score = (
            session.query(ScoringResult)
            .filter(ScoringResult.company_id == company.id)
            .order_by(ScoringResult.scoring_date.desc())
            .first()
        )
        if not score:
            return None

        quality_flags = None
        if score.quality_flags_json:
            try:
                quality_flags = json.loads(score.quality_flags_json)
            except (json.JSONDecodeError, TypeError):
                pass

        composite_alpha = score.composite_alpha
        composite_beta = score.composite_beta
        composite_delta = score.composite_delta
        if (company.company_type or "").upper() == "SPORT":
            composite_alpha = None
            composite_beta = None
            composite_delta = None

        return {
            "scoring_date": score.scoring_date,
            "model_used": score.model_used,
            "data_completeness": score.data_completeness,
            "buffett": score.buffett_score,
            "graham": score.graham_score,
            "piotroski": score.piotroski_fscore,
            "piotroski_raw": score.piotroski_fscore_raw,
            "magic_formula": score.magic_formula_rank,
            "lynch_peg": score.lynch_peg_score,
            "dcf_mos": score.dcf_margin_of_safety_pct,
            "momentum": score.momentum_score,
            "insider": score.insider_score,
            "technical": score.technical_score,
            "dividend": score.dividend_score,
            "alpha": composite_alpha,
            "beta": composite_beta,
            "delta": composite_delta,
            "risk_tier": score.risk_tier,
            "quality_flags": quality_flags,
        }
    finally:
        session.close()


def get_adjusted_metrics(ticker: str) -> Optional[dict]:
    """Fetch latest adjusted financial metrics for a ticker."""
    session = _get_session()
    try:
        company = session.query(Company).filter(Company.ticker == ticker.upper()).first()
        if not company:
            return None

        metric = (
            session.query(AdjustedMetric)
            .filter(AdjustedMetric.company_id == company.id)
            .order_by(AdjustedMetric.period_end.desc())
            .first()
        )
        if not metric:
            return None

        return {
            "period_end": metric.period_end,
            "reported_net_income": metric.reported_net_income,
            "monetary_gain_loss": metric.monetary_gain_loss,
            "adjusted_net_income": metric.adjusted_net_income,
            "owner_earnings": metric.owner_earnings,
            "free_cash_flow": metric.free_cash_flow,
            "roe_adjusted": metric.roe_adjusted,
            "roa_adjusted": metric.roa_adjusted,
            "eps_adjusted": metric.eps_adjusted,
            "real_eps_growth_pct": metric.real_eps_growth_pct,
            "related_party_revenue_pct": metric.related_party_revenue_pct,
            "maintenance_capex": metric.maintenance_capex,
            "growth_capex": metric.growth_capex,
        }
    finally:
        session.close()


def get_stock_position(ticker: str) -> Optional[dict]:
    """Fetch open portfolio position for a ticker from the latest ALPHA snapshot."""
    session = _get_session()
    try:
        company = session.query(Company).filter(Company.ticker == ticker.upper()).first()
        if not company:
            return None

        latest_open_date = (
            session.query(PortfolioSelection.selection_date)
            .filter(
                PortfolioSelection.exit_date.is_(None),
                PortfolioSelection.portfolio == "ALPHA",
            )
            .order_by(PortfolioSelection.selection_date.desc())
            .first()
        )
        if not latest_open_date:
            return None

        pos = (
            session.query(PortfolioSelection)
            .filter(
                PortfolioSelection.company_id == company.id,
                PortfolioSelection.exit_date.is_(None),
                PortfolioSelection.portfolio == "ALPHA",
                PortfolioSelection.selection_date == latest_open_date[0],
            )
            .first()
        )
        if not pos:
            return None

        return {
            "portfolio": pos.portfolio,
            "entry_price": pos.entry_price,
            "target_price": pos.target_price,
            "stop_loss_price": pos.stop_loss_price,
            "selection_date": pos.selection_date,
            "composite_score": pos.composite_score,
        }
    finally:
        session.close()


# -- Macro --


def get_latest_macro() -> Optional[dict]:
    """Fetch the newest macro snapshot, backfilling missing fields safely."""
    session = _get_session()
    try:
        return _load_latest_macro(session)
    finally:
        session.close()


def get_latest_cash_state() -> Optional[dict]:
    """Phase 4: return the most recent persisted cash-allocation state.

    The row is flattened into a plain dict so both the mobile snapshot writer
    and the API layer can consume it without importing the ORM model.
    """
    session = _get_session()
    try:
        row = (
            session.query(CashAllocationState)
            .order_by(CashAllocationState.date.desc())
            .first()
        )
        if row is None:
            return None
        return {
            "date": row.date,
            "market_regime": row.market_regime,
            "macro_regime": row.macro_regime,
            "raw_signal": row.raw_signal,
            "target_state": row.target_state,
            "state": row.state,
            "cash_pct": row.cash_pct,
            "days_in_state": row.days_in_state,
            "last_transition_date": row.last_transition_date,
            "transitioned_today": row.transitioned_today,
            "notes": row.notes,
        }
    finally:
        session.close()


# -- Ticker List --


def get_all_tickers() -> list[str]:
    """Return all company tickers sorted alphabetically."""
    session = _get_session()
    try:
        rows = (
            session.query(Company.ticker)
            .order_by(Company.ticker)
            .all()
        )
        return [r[0] for r in rows]
    finally:
        session.close()


# -- Factor history (Sprint 2 §5, 2026-05-08) --


def get_factor_history_quarterly(
    company_ids: list[int],
    *,
    quarters: int = 8,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return quarter-end factor snapshots for the given companies.

    For each company × the last ``quarters`` calendar quarter-ends, picks
    the closest ``ScoringResult`` row on or before that quarter-end and
    returns its factor scores. Used to build the v2 mobile sparkline UI:
    the APK detail screen shows 8 dots per factor (Buffett, DCF, Momentum,
    Technical) so the user can see whether the score is improving or
    decaying — which a single point-in-time score can't communicate.

    Output rows are deduped on ``(company_id, scoring_date)`` so a single
    pipeline that scored two days in a row near a quarter-end won't pollute
    the sparkline. Companies with fewer than 1 historical scoring row get
    no rows here (sparkline would be a single dot — useless).
    """
    if not company_ids:
        return pd.DataFrame()

    end_date = end_date or date.today()
    # Walk backwards 1 quarter at a time. Calendar quarter-ends only — we
    # don't try to infer fiscal quarters per company; this is a UI hint,
    # not earnings analysis.
    targets: list[date] = []
    cursor_year = end_date.year
    cursor_q = (end_date.month - 1) // 3 + 1  # 1..4
    for _ in range(quarters):
        last_month = cursor_q * 3
        # Last day of the quarter — easy lookup of next-month-day-1 minus 1.
        if last_month == 12:
            quarter_end = date(cursor_year, 12, 31)
        else:
            from datetime import timedelta as _td
            quarter_end = date(cursor_year, last_month + 1, 1) - _td(days=1)
        targets.append(quarter_end)
        cursor_q -= 1
        if cursor_q == 0:
            cursor_q = 4
            cursor_year -= 1
    targets.sort()  # oldest first

    session = _get_session()
    try:
        # For each (company, target_date), grab the latest ScoringResult
        # on or before target_date. Doing this in one SQL pass keeps it
        # snappy even for ~50 companies × 8 quarters = 400 lookups.
        from sqlalchemy import func as _func
        records: list[dict] = []
        for cid in company_ids:
            for target in targets:
                row = (
                    session.query(ScoringResult)
                    .filter(
                        ScoringResult.company_id == cid,
                        ScoringResult.scoring_date <= target,
                    )
                    .order_by(ScoringResult.scoring_date.desc())
                    .first()
                )
                if row is None:
                    continue
                records.append({
                    "company_id": cid,
                    "quarter_end": target,
                    "scoring_date": row.scoring_date,
                    "buffett": row.buffett_score,
                    "graham": row.graham_score,
                    "piotroski": row.piotroski_fscore,
                    "magic_formula": row.magic_formula_rank,
                    "lynch_peg": row.lynch_peg_score,
                    "dcf_mos": row.dcf_margin_of_safety_pct,
                    "momentum": row.momentum_score,
                    "technical": row.technical_score,
                    "dividend": row.dividend_score,
                    "composite_alpha": row.composite_alpha,
                    "data_completeness": row.data_completeness,
                })
        df = pd.DataFrame(records)
        if df.empty:
            return df
        # Drop dupes when two consecutive quarter-ends both fall back to
        # the same underlying scoring_date (history gap).
        df = df.drop_duplicates(subset=["company_id", "scoring_date"])
        return df
    finally:
        session.close()


# -- Backtest (disabled) --
#
# Backtesting was intentionally removed per user request.
# Keep the old dashboard hook disabled so future AI/developers do not
# reintroduce it unless the user explicitly asks for it again.
#
# def run_backtest_direct(
#     start_date: date,
#     end_date: date,
#     initial_capital: float = 1_000_000.0,
# ) -> dict:
#     ...


# -- Helper --


def _latest_price(session: Session, company_id: int) -> Optional[float]:
    """Return the most recent adjusted_close (or close) for a company."""
    row = (
        session.query(DailyPrice)
        .filter(
            DailyPrice.company_id == company_id,
            DailyPrice.adjusted_close.isnot(None),
        )
        .order_by(DailyPrice.date.desc())
        .first()
    )
    if row:
        return row.adjusted_close

    row = (
        session.query(DailyPrice)
        .filter(
            DailyPrice.company_id == company_id,
            DailyPrice.close.isnot(None),
        )
        .order_by(DailyPrice.date.desc())
        .first()
    )
    return row.close if row else None


def _latest_prices(session: Session, company_ids: list[int]) -> dict[int, float]:
    """Return latest adjusted-close-or-close prices for many companies."""
    if not company_ids:
        return {}

    unique_ids = sorted(set(company_ids))

    adjusted_subq = (
        session.query(
            DailyPrice.company_id.label("company_id"),
            func.max(DailyPrice.date).label("max_date"),
        )
        .filter(
            DailyPrice.company_id.in_(unique_ids),
            DailyPrice.adjusted_close.isnot(None),
        )
        .group_by(DailyPrice.company_id)
        .subquery()
    )
    adjusted_rows = (
        session.query(DailyPrice.company_id, DailyPrice.adjusted_close)
        .join(
            adjusted_subq,
            (DailyPrice.company_id == adjusted_subq.c.company_id)
            & (DailyPrice.date == adjusted_subq.c.max_date),
        )
        .all()
    )
    prices = {
        company_id: float(price)
        for company_id, price in adjusted_rows
        if price is not None
    }

    missing_ids = [company_id for company_id in unique_ids if company_id not in prices]
    if not missing_ids:
        return prices

    close_subq = (
        session.query(
            DailyPrice.company_id.label("company_id"),
            func.max(DailyPrice.date).label("max_date"),
        )
        .filter(
            DailyPrice.company_id.in_(missing_ids),
            DailyPrice.close.isnot(None),
        )
        .group_by(DailyPrice.company_id)
        .subquery()
    )
    close_rows = (
        session.query(DailyPrice.company_id, DailyPrice.close)
        .join(
            close_subq,
            (DailyPrice.company_id == close_subq.c.company_id)
            & (DailyPrice.date == close_subq.c.max_date),
        )
        .all()
    )
    prices.update(
        {
            company_id: float(price)
            for company_id, price in close_rows
            if price is not None
        }
    )
    return prices


def _calculate_portfolio_performance_fallback(
    session: Session,
    portfolio_name: str,
) -> dict:
    """Compute lightweight portfolio metrics when the legacy output module is absent."""
    rows = (
        session.query(PortfolioSelection)
        .filter(PortfolioSelection.portfolio == portfolio_name.upper())
        .all()
    )
    if not rows:
        return {
            "total_return_avg": None,
            "active_return_avg": None,
            "win_rate": None,
        }

    open_company_ids = [
        row.company_id
        for row in rows
        if row.exit_date is None and row.company_id is not None
    ]
    latest_prices = _latest_prices(session, open_company_ids)

    total_returns: list[float] = []
    active_returns: list[float] = []

    for row in rows:
        return_pct = row.return_pct
        if return_pct is None and row.entry_price and row.entry_price > 0:
            if row.exit_price is not None:
                return_pct = ((row.exit_price - row.entry_price) / row.entry_price) * 100.0
            elif row.exit_date is None:
                current_price = latest_prices.get(row.company_id)
                if current_price is not None:
                    return_pct = ((current_price - row.entry_price) / row.entry_price) * 100.0

        if return_pct is None:
            continue

        normalized_return = float(return_pct)
        total_returns.append(normalized_return)
        if row.exit_date is None:
            active_returns.append(normalized_return)

    total_return_avg = (
        sum(total_returns) / len(total_returns)
        if total_returns
        else None
    )
    active_return_avg = (
        sum(active_returns) / len(active_returns)
        if active_returns
        else total_return_avg
    )
    win_rate = (
        (sum(1 for value in total_returns if value > 0) / len(total_returns)) * 100.0
        if total_returns
        else None
    )

    return {
        "total_return_avg": total_return_avg,
        "active_return_avg": active_return_avg,
        "win_rate": win_rate,
    }
