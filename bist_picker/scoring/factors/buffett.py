"""Buffett quality scoring factor for BIST Stock Picker.

Scores OPERATING companies on Warren Buffett's quality criteria using
adjusted financial metrics. Returns None for BANK, HOLDING, INSURANCE,
and REIT companies -- they use separate scoring models.

Sub-factors:
- ROE level (5-year average adjusted ROE)
- ROE consistency (inverse of std dev)
- Gross margin level
- Margin stability (inverse of std dev)
- Debt safety (inverse of debt/equity)
- Earnings quality (years of positive adj EPS)
- FCF quality (years of positive FCF)
- Owner earnings trend (regression slope)

All thresholds pulled from config/thresholds.yaml.
"""

import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from sqlalchemy.orm import Session

from bist_picker.cleaning.inflation import _find_item_by_codes
from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    FinancialStatement,
    MacroRegime,
)

logger = logging.getLogger("bist_picker.scoring.factors.buffett")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"

# Item codes for raw financial statement lookups
_CODES_GROSS_PROFIT = ["3D"]
_CODES_NET_SALES = ["3C"]
_CODES_TOTAL_ASSETS = ["1BL"]
_CODES_TOTAL_EQUITY = ["2N"]
_CODES_CURRENT_LIABILITIES = ["2A"]
_CODES_LT_LIABILITIES = ["2B"]

# Company types that use this scorer
_OPERATING_TYPES = {"OPERATING", None, ""}

# Minimum years of data required for a combined score
_MIN_YEARS = 3

# Sub-factor weights for combined score
# owner_earnings_trend raised from 5% → 15% (the most Buffett-specific concept).
# roe_level 20% → 15%, debt_safety 15% → 10% to compensate.
_WEIGHTS = {
    "roe_level": 0.15,
    "roe_consistency": 0.10,
    "gross_margin": 0.15,
    "margin_stability": 0.10,
    "debt_safety": 0.10,
    "earnings_quality": 0.15,
    "fcf_quality": 0.10,
    "owner_earnings_trend": 0.15,
}


class BuffettScorer:
    """Scores companies on Buffett quality criteria.

    Only works for OPERATING companies. Returns None for BANK, HOLDING,
    INSURANCE, and REIT types.

    Args:
        config_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._thresholds: dict = {}
        self._load_config()
        # Cache the resolved nominal-deflation rate per scoring run.
        # See _resolve_inflation_proxy.
        self._inflation_proxy_cache: Optional[tuple] = None

    def _load_config(self) -> None:
        """Load threshold values from thresholds.yaml."""
        if not self._config_path.exists():
            logger.warning("Thresholds config not found: %s", self._config_path)
            return

        with open(self._config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._thresholds = config.get("buffett", {})

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[_date] = None,
        scoring_context: Optional[Any] = None,
    ) -> Optional[dict]:
        """Score a single company on Buffett quality criteria.

        Args:
            company_id: Database ID of the company.
            session: SQLAlchemy session.
            scoring_date: Date of scoring.
            scoring_context: Optional ScoringContext with pre-loaded data.

        Returns:
            Dict with sub-scores and buffett_combined (0-100 raw),
            or None if company is not OPERATING or has insufficient data.
        """
        if scoring_context:
            ctype = scoring_context.get_company_type(company_id)
            metrics = scoring_context.get_metrics(company_id)
            balance_stmts = scoring_context.get_statements(company_id, "BALANCE")
            income_stmts = scoring_context.get_statements(company_id, "INCOME")
            # Logic below processes these lists
        else:
            # Legacy N+1 path
            company = session.get(Company, company_id)
            if company is None:
                logger.warning("Company ID %d not found", company_id)
                return None
            ctype = (company.company_type or "").upper()
            
            # Load metrics via query
            from datetime import timedelta
            cutoff_date = scoring_date or _date.today()
            lagged_cutoff = cutoff_date - timedelta(days=76)
            query = session.query(AdjustedMetric).filter(
                AdjustedMetric.company_id == company_id,
                AdjustedMetric.period_end <= lagged_cutoff,
            )
            metrics = query.order_by(AdjustedMetric.period_end).all()
            
            # Load statements via query (will be handled by helper fallback logic or we load here)
            # To keep structure similar, we will just use helpers if context is missing.
            balance_stmts = None 
            income_stmts = None

        # 1. Check company type
        if ctype in ("BANK", "HOLDING", "INSURANCE", "REIT", "SPORT", "FINANCIAL"):
            logger.debug("Skipping %s: company_type=%s", company_id, ctype)
            return None

        # 2. Check metrics count
        if len(metrics) < _MIN_YEARS:
            logger.debug(
                "Skipping %s: only %d years of data (need %d)",
                company_id, len(metrics), _MIN_YEARS,
            )
            return None

        # 3. Load financial data if context provided, else use helpers
        if scoring_context:
            balance_data = self._extract_balance_data(balance_stmts or [])
            income_data = self._extract_income_data(income_stmts or [])
        else:
            balance_data = self._load_balance_series(company_id, session, scoring_date)
            income_data = self._load_income_series(company_id, session, scoring_date)

        # Calculate each sub-factor
        result = {}
        result["roe_level"] = self._score_roe_level(metrics)
        result["roe_consistency"] = self._score_roe_consistency(metrics)
        result["gross_margin"] = self._score_gross_margin(income_data)
        result["margin_stability"] = self._score_margin_stability(income_data)
        result["debt_safety"] = self._score_debt_safety(balance_data)
        result["earnings_quality"] = self._score_earnings_quality(metrics)
        result["fcf_quality"] = self._score_fcf_quality(metrics)
        result["owner_earnings_trend"] = self._score_oe_trend(
            metrics, session, scoring_date,
        )

        # Calculate weighted combined score
        result["buffett_combined"] = self._calculate_combined(result)

        return result

    def _parse_stmts(self, stmts: list, *codes_lists) -> list[dict]:
        """Helper to parse statements from context into dicts."""
        if not stmts:
            return []
        
        result = []
        for stmt in stmts:
            if not stmt.data_json:
                continue
            try:
                data = json.loads(stmt.data_json)
            except json.JSONDecodeError:
                continue
            
            # Map codes lists to field names dynamically? 
            # Actually, _load_balance_series maps specific fields.
            # This helper is tricky because field mapping depends on stmt type.
            # Let's just implement inline or specific helpers.
            pass
        return [] # Placeholder, will implement specific parsing inside score or dedicated methods

    # ---- Sub-factor scoring methods ----

    def _score_roe_level(self, metrics: list[AdjustedMetric]) -> Optional[float]:
        """Score based on 5-year (or available) average adjusted ROE.

        0 if ROE < 5%, linear 0-100 for 5%-25%, max 100 at 25%+.
        """
        roe_values = [m.roe_adjusted for m in metrics if m.roe_adjusted is not None]
        if not roe_values:
            return None

        # Use last 5 years or all available
        recent = roe_values[-5:]
        avg_roe = sum(recent) / len(recent)

        roe_min = self._thresholds.get("roe_min_score", 0.05)
        roe_max = self._thresholds.get("roe_full_score", 0.25)

        return _linear_scale(avg_roe, roe_min, roe_max)

    def _score_roe_consistency(self, metrics: list[AdjustedMetric]) -> Optional[float]:
        """Score based on inverse of ROE standard deviation. Stable = good.

        Low std dev (< 0.03) -> 100, high std dev (> 0.15) -> 0.
        """
        roe_values = [m.roe_adjusted for m in metrics if m.roe_adjusted is not None]
        if len(roe_values) < 2:
            return None

        recent = roe_values[-5:]
        std = float(np.std(recent, ddof=1))

        # Inverse scale: low std = high score
        return _linear_scale_inverse(std, low=0.03, high=0.15)

    def _score_gross_margin(
        self, income_data: list[dict],
    ) -> Optional[float]:
        """Score based on most recent gross margin level.

        0 if margin < 10%, linear to 100 at 40%+.
        """
        if not income_data:
            return None

        # Use last available period
        latest = income_data[-1]
        gross_profit = latest.get("gross_profit")
        net_sales = latest.get("net_sales")

        if gross_profit is None or net_sales is None or net_sales == 0:
            return None

        margin = gross_profit / net_sales
        min_margin = self._thresholds.get("min_gross_margin", 0.25)

        # Scale: 0 at 10%, 100 at min_margin*1.6 (i.e. ~40% for default 25%)
        return _linear_scale(margin, 0.10, min_margin * 1.6)

    def _score_margin_stability(
        self, income_data: list[dict],
    ) -> Optional[float]:
        """Score based on inverse of gross margin std dev. Stable = good.

        Low std dev (< 0.02) -> 100, high std dev (> 0.10) -> 0.
        """
        if len(income_data) < 2:
            return None

        margins = []
        for period in income_data:
            gp = period.get("gross_profit")
            ns = period.get("net_sales")
            if gp is not None and ns is not None and ns != 0:
                margins.append(gp / ns)

        if len(margins) < 2:
            return None

        recent = margins[-5:]
        std = float(np.std(recent, ddof=1))
        return _linear_scale_inverse(std, low=0.02, high=0.10)

    def _score_debt_safety(self, balance_data: list[dict]) -> Optional[float]:
        """Score based on debt/equity ratio. Low debt = good.

        100 if D/E < 0.3, linear down to 0 at D/E > 2.0.
        """
        if not balance_data:
            return None

        # Use last available period
        latest = balance_data[-1]
        total_assets = latest.get("total_assets")
        total_equity = latest.get("total_equity")

        if total_assets is None or total_equity is None or total_equity == 0:
            return None

        total_debt = total_assets - total_equity
        de_ratio = total_debt / abs(total_equity)

        # Negative equity -> worst score
        if total_equity < 0:
            return 0.0

        de_ideal = self._thresholds.get("debt_equity_ideal", 0.3)
        de_max = self._thresholds.get("debt_equity_max", 2.0)

        return _linear_scale_inverse(de_ratio, low=de_ideal, high=de_max)

    def _score_earnings_quality(self, metrics: list[AdjustedMetric]) -> Optional[float]:
        """Score based on years of positive adjusted EPS.

        (positive_years / total_years) * 100.
        """
        eps_values = [m.eps_adjusted for m in metrics if m.eps_adjusted is not None]
        if not eps_values:
            return None

        positive = sum(1 for e in eps_values if e > 0)
        return (positive / len(eps_values)) * 100.0

    def _score_fcf_quality(self, metrics: list[AdjustedMetric]) -> Optional[float]:
        """Score based on years of positive free cash flow.

        (positive_years / total_years) * 100.
        """
        fcf_values = [m.free_cash_flow for m in metrics if m.free_cash_flow is not None]
        if not fcf_values:
            return None

        positive = sum(1 for f in fcf_values if f > 0)
        return (positive / len(fcf_values)) * 100.0

    def _score_oe_trend(
        self,
        metrics: list[AdjustedMetric],
        session: Optional[Session] = None,
        scoring_date: Optional[_date] = None,
    ) -> Optional[float]:
        """Score based on owner earnings growth trend (regression slope).

        Positive slope -> higher score. Uses linear regression on OE values.
        Score: 0 if slope <= 0, linear to 100 for strong positive trends.

        2026-04-30 audit: Owner earnings are reported in nominal TRY. With
        Turkey's 50-85% YoY CPI in 2021-2024, a company whose REAL earnings
        are flat shows large nominal upward slope and used to score 100/100.
        We now subtract an inflation pass-through proxy from the relative
        slope so the score reflects real, not nominal, growth.

        Proxy resolution (first non-null wins):
          1. MacroRegime.inflation_expectation_24m_pct on/before scoring_date
          2. MacroRegime.cpi_yoy_pct on/before scoring_date
          3. None -> no deflation (test fixtures that don't populate macro
             keep their historical scoring behavior)
        """
        oe_values = [m.owner_earnings for m in metrics if m.owner_earnings is not None]
        if len(oe_values) < _MIN_YEARS:
            return None

        recent = oe_values[-5:]
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)

        # Normalize y by mean absolute value to get relative slope
        mean_abs = np.mean(np.abs(y))
        if mean_abs == 0:
            return 50.0  # flat at zero

        # Linear regression slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x * x) - np.sum(x) ** 2
        )

        # Normalize slope relative to magnitude
        relative_slope = slope / mean_abs

        # Subtract inflation pass-through so the slope reflects real growth.
        inflation = self._resolve_inflation_proxy(session, scoring_date)
        if inflation is not None:
            relative_slope = relative_slope - inflation

        # Score: -0.2 or worse -> 0, +0.3 or better -> 100
        return _linear_scale(relative_slope, -0.2, 0.3)

    def _resolve_inflation_proxy(
        self,
        session: Optional[Session],
        scoring_date: Optional[_date],
    ) -> Optional[float]:
        """Return a TRY inflation proxy for deflating nominal series.

        Prefers `inflation_expectation_24m_pct` (forward-looking, more stable
        than spot CPI). Falls back to `cpi_yoy_pct`. Returns None if no
        macro row exists, so unit tests without seeded macro data preserve
        the pre-fix nominal scoring behavior.
        """
        if self._inflation_proxy_cache is not None and self._inflation_proxy_cache[0] == scoring_date:
            return self._inflation_proxy_cache[1]

        if session is None:
            self._inflation_proxy_cache = (scoring_date, None)
            return None

        query = session.query(MacroRegime)
        if scoring_date is not None:
            query = query.filter(MacroRegime.date <= scoring_date)
        latest = query.order_by(MacroRegime.date.desc()).first()

        if latest is None:
            self._inflation_proxy_cache = (scoring_date, None)
            return None

        # Prefer forward-looking expectation (24m), fallback to spot YoY.
        proxy = latest.inflation_expectation_24m_pct
        if proxy is None:
            proxy = latest.cpi_yoy_pct
        if proxy is None or proxy <= 0:
            self._inflation_proxy_cache = (scoring_date, None)
            return None

        proxy = float(proxy)
        logger.debug(
            "Buffett OE-trend inflation deflator: %.1f%% (as of %s)",
            proxy * 100, latest.date,
        )
        self._inflation_proxy_cache = (scoring_date, proxy)
        return proxy

    # ---- Combined score ----

    def _calculate_combined(self, scores: dict) -> Optional[float]:
        """Calculate weighted combined Buffett score.

        Args:
            scores: Dict of sub-factor scores (may contain None).

        Returns:
            Combined score 0-100, or None if too few sub-scores available.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for factor, weight in _WEIGHTS.items():
            value = scores.get(factor)
            if value is not None:
                weighted_sum += value * weight
                total_weight += weight

        # Need at least 60% of weight covered
        if total_weight < 0.60:
            return None

        # Scale to full range
        return weighted_sum / total_weight

    # ---- Data loading helpers ----

    # ---- Data loading helpers ----

    def _load_balance_series(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[_date] = None,
    ) -> list[dict]:
        """Load balance sheet key fields for all annual periods.

        Returns:
            List of dicts with total_assets, total_equity, etc., ordered by period.
        """
        from datetime import timedelta
        cutoff_date = scoring_date or _date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        
        from sqlalchemy import or_
        query = session.query(FinancialStatement).filter(
            FinancialStatement.company_id == company_id,
            FinancialStatement.period_type == "ANNUAL",
            FinancialStatement.statement_type == "BALANCE",
            or_(
                FinancialStatement.publication_date <= cutoff_date,
                (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff)),
            )
        )

        statements = query.order_by(FinancialStatement.period_end).all()
        return self._extract_balance_data(statements)

    def _extract_balance_data(self, statements: list[FinancialStatement]) -> list[dict]:
        """Extract needed fields from balance statements."""
        result = []
        for stmt in statements:
            if not stmt.data_json:
                continue
            try:
                data = json.loads(stmt.data_json)
            except json.JSONDecodeError:
                continue

            result.append({
                "period_end": stmt.period_end,
                "total_assets": _find_item_by_codes(data, _CODES_TOTAL_ASSETS),
                "total_equity": _find_item_by_codes(data, _CODES_TOTAL_EQUITY),
            })
        return result

    def _load_income_series(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[_date] = None,
    ) -> list[dict]:
        """Load income statement key fields for all annual periods.

        Returns:
            List of dicts with gross_profit, net_sales, ordered by period.
        """
        from datetime import timedelta
        cutoff_date = scoring_date or _date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        
        from sqlalchemy import or_
        query = session.query(FinancialStatement).filter(
            FinancialStatement.company_id == company_id,
            FinancialStatement.period_type == "ANNUAL",
            FinancialStatement.statement_type == "INCOME",
            or_(
                FinancialStatement.publication_date <= cutoff_date,
                (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff)),
            )
        )

        statements = query.order_by(FinancialStatement.period_end).all()
        return self._extract_income_data(statements)

    def _extract_income_data(self, statements: list[FinancialStatement]) -> list[dict]:
        """Extract needed fields from income statements."""
        result = []
        for stmt in statements:
            if not stmt.data_json:
                continue
            try:
                data = json.loads(stmt.data_json)
            except json.JSONDecodeError:
                continue

            result.append({
                "period_end": stmt.period_end,
                "gross_profit": _find_item_by_codes(data, _CODES_GROSS_PROFIT),
                "net_sales": _find_item_by_codes(data, _CODES_NET_SALES),
            })
        return result


# ---- Utility functions ----


def _linear_scale(value: float, low: float, high: float) -> float:
    """Scale a value linearly from 0 (at low) to 100 (at high).

    Args:
        value: Input value.
        low: Value at which score = 0.
        high: Value at which score = 100.

    Returns:
        Score clamped to 0-100.
    """
    if high == low:
        return 50.0
    score = (value - low) / (high - low) * 100.0
    return max(0.0, min(100.0, score))


def _linear_scale_inverse(value: float, low: float, high: float) -> float:
    """Scale a value inversely: 100 at low, 0 at high.

    Args:
        value: Input value (lower is better).
        low: Value at which score = 100.
        high: Value at which score = 0.

    Returns:
        Score clamped to 0-100.
    """
    if high == low:
        return 50.0
    score = (1.0 - (value - low) / (high - low)) * 100.0
    return max(0.0, min(100.0, score))
