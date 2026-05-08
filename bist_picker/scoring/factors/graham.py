"""Graham value scoring factor for BIST Stock Picker.

Scores companies on Benjamin Graham's value criteria using adjusted
financial metrics and current market price. Returns None for BANK and
INSURANCE companies (financials excluded from Graham analysis).

Sub-factors:
- Graham Number ratio (intrinsic value / price)
- NCAV ratio (net current asset value / price)
- P/E x P/B product (want <= 22.5)
- Graham Growth Formula value vs price

All thresholds pulled from config/thresholds.yaml.
"""

import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml
from sqlalchemy import func
from sqlalchemy.orm import Session

from bist_picker.cleaning.inflation import _find_item_by_codes
from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    DailyPrice,
    FinancialStatement,
    MacroRegime,
)

logger = logging.getLogger("bist_picker.scoring.factors.graham")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"

# Item codes for raw financial statement lookups
_CODES_TOTAL_ASSETS = ["1BL"]
_CODES_CURRENT_ASSETS = ["1A"]
_CODES_TOTAL_EQUITY = ["2N"]
_CODES_CURRENT_LIABILITIES = ["2A"]
_CODES_LT_LIABILITIES = ["2B"]
_CODES_SHARE_CAPITAL = ["2OA"]

# Minimum years of data for growth estimation
_MIN_YEARS_GROWTH = 2

# Sub-factor weights for combined score
_WEIGHTS = {
    "graham_number_ratio": 0.30,
    "ncav_ratio": 0.20,
    "pe_pb_product": 0.25,
    "graham_growth_value": 0.25,
}


class GrahamScorer:
    """Scores companies on Graham value criteria.

    Excludes BANK and INSURANCE companies. Returns None for those types.

    Args:
        config_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._thresholds: dict = {}
        self._load_config()
        # Cache the resolved bond yield per scoring_date so we don't re-query
        # MacroRegime once per company. Mirrors dcf._resolve_discount_rate.
        self._bond_yield_cache: Optional[tuple] = None

    def _load_config(self) -> None:
        """Load threshold values from thresholds.yaml."""
        if not self._config_path.exists():
            logger.warning("Thresholds config not found: %s", self._config_path)
            return

        with open(self._config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._thresholds = config.get("graham", {})

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
        scoring_context: Optional[Any] = None,
    ) -> Optional[dict]:
        """Score a single company on Graham value criteria.

        Args:
            company_id: Database ID of the company.
            session: SQLAlchemy session.
            scoring_date: Date of scoring.
            scoring_context: Optional ScoringContext with pre-loaded data.

        Returns:
            Dict with sub-scores and graham_combined (0-100 raw),
            or None if company is excluded or has insufficient data.
        """
        if scoring_context:
            metrics = scoring_context.get_metrics(company_id)
            ctype = scoring_context.get_company_type(company_id)
        else:
            company = session.get(Company, company_id)
            if company is None:
                logger.warning("Company ID %d not found", company_id)
                return None
            ctype = (company.company_type or "").upper()
            
            # Centralized point-in-time guard (audit CRITICAL #1,
            # 2026-05-07).
            from bist_picker.scoring.context import _adjusted_metric_pit_filter
            cutoff_date = scoring_date or date.today()
            query = session.query(AdjustedMetric).filter(
                AdjustedMetric.company_id == company_id,
                _adjusted_metric_pit_filter(cutoff_date),
            )
            metrics = query.order_by(AdjustedMetric.period_end).all()

        # Exclude financials
        if ctype in ("BANK", "INSURANCE", "SPORT", "FINANCIAL"):
            logger.debug("Skipping %s: company_type=%s", company_id, ctype)
            return None

        if not metrics:
            logger.debug("Skipping %s: no adjusted metrics", company_id)
            return None

        latest_metric = metrics[-1]

        # Get current price
        if scoring_context:
            current_price = scoring_context.get_latest_price(company_id)
        else:
            current_price = self._get_latest_price(company_id, session, scoring_date)
            
        if current_price is None or current_price <= 0:
            logger.debug("Skipping %s: no current price", company_id)
            return None

        # Get latest balance sheet data
        if scoring_context:
            # We need the LATEST annual balance sheet. context returns list sorted by date.
            stmts = scoring_context.get_statements(company_id, "BALANCE")
            balance = self._extract_latest_balance(stmts)
        else:
            balance = self._load_latest_balance(company_id, session, scoring_date)

        # Calculate shares outstanding from share capital (par value = 1 TRY)
        shares = balance.get("share_capital") if balance else None
        if shares is None or shares <= 0:
            logger.debug("Skipping %s: no share capital data", company_id)
            return None

        # Calculate sub-factors
        result = {}
        result["graham_number_ratio"] = self._score_graham_number(
            latest_metric, balance, shares, current_price,
        )
        result["ncav_ratio"] = self._score_ncav(
            balance, shares, current_price,
        )
        result["pe_pb_product"] = self._score_pe_pb(
            latest_metric, balance, shares, current_price,
        )
        result["graham_growth_value"] = self._score_graham_growth(
            metrics, current_price, session, scoring_date,
        )

        # Calculate combined score
        result["graham_combined"] = self._calculate_combined(result)

        return result

    # ---- Sub-factor scoring methods ----

    def _score_graham_number(
        self,
        metric: AdjustedMetric,
        balance: dict,
        shares: float,
        price: float,
    ) -> Optional[float]:
        """Score based on Graham Number / current price ratio.

        Graham Number = sqrt(22.5 * adjusted_EPS * book_value_per_share)
        Ratio > 1.0 means undervalued.

        Score: 0 if ratio < 0.5, linear to 100 at ratio >= 1.5.
        """
        eps = metric.eps_adjusted
        if eps is None or eps <= 0:
            return None

        equity = balance.get("total_equity")
        if equity is None or equity <= 0:
            return None

        bvps = equity / shares
        if bvps <= 0:
            return None

        graham_number = math.sqrt(22.5 * eps * bvps)
        ratio = graham_number / price

        return _linear_scale(ratio, 0.5, 1.5)

    def _score_ncav(
        self,
        balance: dict,
        shares: float,
        price: float,
    ) -> Optional[float]:
        """Score based on NCAV per share / current price ratio.

        NCAV = current_assets - total_liabilities
        Ratio > 1.5 means deep value (classic Graham net-net).

        Score: 0 if ratio < 0, linear to 100 at ratio >= 1.0.
        Most stocks won't qualify for net-net, so scale is generous.
        """
        current_assets = balance.get("current_assets")
        total_equity = balance.get("total_equity")
        total_assets = balance.get("total_assets")

        if current_assets is None or total_assets is None or total_equity is None:
            return None

        total_liabilities = total_assets - total_equity
        ncav = current_assets - total_liabilities
        ncav_per_share = ncav / shares

        if price <= 0:
            return None

        ratio = ncav_per_share / price

        # Most stocks have negative NCAV; generous scale
        return _linear_scale(ratio, -0.5, 1.0)

    def _score_pe_pb(
        self,
        metric: AdjustedMetric,
        balance: dict,
        shares: float,
        price: float,
    ) -> Optional[float]:
        """Score based on P/E x P/B product. Want <= 22.5 (Graham criterion).

        Score: 100 if product <= 10, linear down to 0 at product >= 45.
        """
        eps = metric.eps_adjusted
        if eps is None or eps <= 0:
            return None

        equity = balance.get("total_equity")
        if equity is None or equity <= 0:
            return None

        bvps = equity / shares
        if bvps <= 0:
            return None

        pe = price / eps
        pb = price / bvps
        product = pe * pb

        max_product = self._thresholds.get("max_pe_pb_product", 22.5)

        # Inverse: low product = high score
        return _linear_scale_inverse(product, low=max_product * 0.44, high=max_product * 2.0)

    def _score_graham_growth(
        self,
        metrics: list[AdjustedMetric],
        price: float,
        session: Optional[Session] = None,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Score based on Graham Growth Formula intrinsic value vs price.

        Graham Growth Value = EPS * (8.5 + 2g) * (4.4/Y)
        where g = nominal EPS CAGR (%), Y = TRY bond yield

        g must be NOMINAL to match the nominal bond yield Y. Using real
        (Fisher-deflated) growth against a nominal yield understates
        intrinsic values by ~3x in Turkey's high-inflation environment.
        Nominal CAGR is computed from eps_adjusted values across actual
        calendar years (same approach as dcf.py _estimate_growth_rate).

        Score: 0 if value/price < 0.5, linear to 100 at value/price >= 1.5.
        """
        # Collect ALL (eps, period_end) pairs — do NOT filter out negatives.
        # Filtering negatives hides loss years and overstates growth.
        eps_pairs = [
            (m.eps_adjusted, m.period_end)
            for m in metrics
            if m.eps_adjusted is not None
        ]

        if len(eps_pairs) < _MIN_YEARS_GROWTH:
            return None

        latest_eps = eps_pairs[-1][0]
        if latest_eps is None or latest_eps <= 0:
            return None

        # Compute nominal EPS CAGR using actual calendar year span.
        # CAGR requires both first and last EPS > 0; if first is <= 0,
        # fall back to default growth (sign-change makes CAGR undefined).
        min_g_pct = self._thresholds.get("graham_min_growth_pct", 5.0)
        max_g_pct = self._thresholds.get("graham_max_growth_pct", 35.0)
        default_g_pct = 10.0

        first_eps, first_date = eps_pairs[0]
        last_eps, last_date = eps_pairs[-1]
        n_years = (last_date - first_date).days / 365.25

        if n_years >= 0.5 and first_eps > 0:
            try:
                raw_cagr = (last_eps / first_eps) ** (1.0 / n_years) - 1.0
                if not math.isfinite(raw_cagr):
                    g = default_g_pct
                else:
                    g = max(min_g_pct, min(max_g_pct, raw_cagr * 100))
            except (ZeroDivisionError, ValueError, OverflowError):
                g = default_g_pct
        else:
            g = default_g_pct

        bond_yield = self._resolve_bond_yield(session, scoring_date)

        # Graham Growth Formula: V = EPS * (8.5 + 2g) * (4.4/Y)
        intrinsic_value = latest_eps * (8.5 + 2.0 * g) * (4.4 / (bond_yield * 100))

        if intrinsic_value <= 0:
            return 0.0

        ratio = intrinsic_value / price
        return _linear_scale(ratio, 0.5, 1.5)

    # ---- Dynamic TRY bond yield ----

    def _resolve_bond_yield(
        self,
        session: Optional[Session],
        scoring_date: Optional[date],
    ) -> float:
        """Return the TRY nominal yield used in Graham's V = EPS*(8.5+2g)*(4.4/Y).

        2026-04-30 audit: previously hardcoded to 0.30 in thresholds.yaml.
        That value drifts 15-20% off intrinsic values whenever TCMB moves
        the policy rate. We now use MacroRegime.policy_rate_pct as a
        defensible TRY rate proxy (the curve is typically inverted in
        Turkey, so the policy rate sits close to the long end), falling
        back to the YAML value only when no macro row exists.

        Cached per scoring_date so this only queries once per scoring run.
        """
        if self._bond_yield_cache is not None and self._bond_yield_cache[0] == scoring_date:
            return self._bond_yield_cache[1]

        fallback = float(self._thresholds.get("try_bond_yield", 0.30))
        if fallback <= 0:
            fallback = 0.30

        if session is None:
            self._bond_yield_cache = (scoring_date, fallback)
            return fallback

        query = session.query(MacroRegime).filter(
            MacroRegime.policy_rate_pct.isnot(None),
        )
        if scoring_date is not None:
            query = query.filter(MacroRegime.date <= scoring_date)
        latest = query.order_by(MacroRegime.date.desc()).first()

        if latest is None or latest.policy_rate_pct is None:
            logger.info(
                "Graham: no MacroRegime policy rate; using static %.1f%%",
                fallback * 100,
            )
            self._bond_yield_cache = (scoring_date, fallback)
            return fallback

        y = float(latest.policy_rate_pct)
        if y <= 0:
            self._bond_yield_cache = (scoring_date, fallback)
            return fallback

        logger.info(
            "Graham bond yield: %.2f%% (TCMB policy rate as of %s)",
            y * 100, latest.date,
        )
        self._bond_yield_cache = (scoring_date, y)
        return y

    # ---- Combined score ----

    def _calculate_combined(self, scores: dict) -> Optional[float]:
        """Calculate weighted combined Graham score.

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

        # Need at least 40% weight covered
        if total_weight < 0.40:
            return None

        return weighted_sum / total_weight

    # ---- Data loading helpers ----

    def _get_latest_price(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Get the most recent closing price for a company.

        Args:
            company_id: Company database ID.
            session: SQLAlchemy session.
            scoring_date: If provided, gets latest price on or before this date.

        Returns:
            Latest closing price, or None if not available.
        """
        query = session.query(DailyPrice.close).filter(
            DailyPrice.company_id == company_id,
            DailyPrice.close.isnot(None),
        )

        if scoring_date:
            query = query.filter(DailyPrice.date <= scoring_date)

        latest = query.order_by(DailyPrice.date.desc()).first()

        return latest[0] if latest else None

    def _load_latest_balance(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> dict:
        """Load the most recent balance sheet data.

        Returns:
            Dict with total_assets, current_assets, total_equity, share_capital.
        """
        from datetime import date as _date, timedelta
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

        # Try most recent first; skip records with all-null values (future shells)
        for stmt in query.order_by(FinancialStatement.period_end.desc()).limit(3):
            fields = self._extract_balance_fields(stmt)
            if any(v is not None for v in fields.values()):
                return fields
        return {}

    def _extract_latest_balance(self, statements: list[FinancialStatement]) -> dict:
        """Find and parse the latest balance statement from a list."""
        if not statements:
            return {}
        # Context returns sorted list (ascending). Latest is last.
        stmt = statements[-1]
        return self._extract_balance_fields(stmt)

    def _extract_balance_fields(self, stmt: FinancialStatement) -> dict:
        if not stmt.data_json:
            return {}
        try:
            data = json.loads(stmt.data_json)
        except json.JSONDecodeError:
            return {}

        return {
            "total_assets": _find_item_by_codes(data, _CODES_TOTAL_ASSETS),
            "current_assets": _find_item_by_codes(data, _CODES_CURRENT_ASSETS),
            "total_equity": _find_item_by_codes(data, _CODES_TOTAL_EQUITY),
            "share_capital": _find_item_by_codes(data, _CODES_SHARE_CAPITAL),
        }


# ---- Utility functions (reuse from buffett) ----


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
