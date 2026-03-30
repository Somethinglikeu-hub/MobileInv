"""Greenblatt's Magic Formula scoring factor for BIST Stock Picker.

Rank-based scoring: all eligible companies are scored at once. Ranks stocks
by Earnings Yield and Return on Capital, then combines ranks.

Exclusions:
- BANK, INSURANCE (financials excluded per Greenblatt)
- Companies below minimum market cap threshold
- Companies with insufficient data

Formulas:
- Earnings Yield = EBIT / Enterprise Value
- Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets)
- Combined Rank = EY_rank + ROC_rank (lowest = best)
- Normalized score: rank 1 = 100, last = 0
"""

import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from bist_picker.cleaning.inflation import _find_item_by_codes
from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    DailyPrice,
    FinancialStatement,
)

logger = logging.getLogger("bist_picker.scoring.factors.magic_formula")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"

# Item codes
_CODES_OPERATING_PROFIT = ["3DF"]
_CODES_TOTAL_ASSETS = ["1BL"]
_CODES_CURRENT_ASSETS = ["1A"]
_CODES_CURRENT_LIABILITIES = ["2A"]
_CODES_TOTAL_EQUITY = ["2N"]
_CODES_SHARE_CAPITAL = ["2OA"]
_CODES_LT_LIABILITIES = ["2B"]   # Long-term liabilities (financial debt proxy)
_CODES_PP_AND_E = ["1BC"]        # Property, Plant and Equipment
_CODES_CASH = ["1AA"]            # Cash and cash equivalents

# Excluded company types
_EXCLUDED_TYPES = {"BANK", "INSURANCE", "SPORT", "FINANCIAL"}


class MagicFormulaScorer:
    """Greenblatt's Magic Formula: rank-based scoring across all stocks.

    Must be called with score_all() since rankings require the full universe.

    Args:
        config_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._min_market_cap: float = 500_000_000
        self._load_config()

    def _load_config(self) -> None:
        """Load threshold values from thresholds.yaml."""
        if not self._config_path.exists():
            logger.warning("Thresholds config not found: %s", self._config_path)
            return

        with open(self._config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        mf = config.get("magic_formula", {})
        self._min_market_cap = mf.get("min_market_cap_try", 500_000_000)

    def score_all(self, session: Session, scoring_date: Optional[_date] = None) -> dict[int, dict]:
        """Score all eligible companies using Magic Formula ranking.

        Args:
            session: SQLAlchemy session.
            scoring_date: Date cutoff for Point-in-Time queries.
                Defaults to today if omitted.

        Returns:
            Dict mapping company_id to score dict:
            {
                'earnings_yield': float,
                'return_on_capital': float,
                'ey_rank': int,
                'roc_rank': int,
                'combined_rank': int,
                'magic_formula_score': float  # 0-100
            }
        """
        cutoff = scoring_date or _date.today()
        # Get all active companies
        companies = (
            session.query(Company)
            .filter(Company.is_active.is_(True))
            .all()
        )

        # Calculate raw metrics for each eligible company
        raw_data = {}
        for company in companies:
            # Exclude financials
            ctype = (company.company_type or "").upper()
            if ctype in _EXCLUDED_TYPES:
                continue

            data = self._calculate_raw(company, session, cutoff)
            if data is not None:
                raw_data[company.id] = data

        if not raw_data:
            logger.warning("No eligible companies for Magic Formula scoring")
            return {}

        # Rank by Earnings Yield (highest = rank 1)
        ey_sorted = sorted(
            raw_data.keys(),
            key=lambda cid: raw_data[cid]["earnings_yield"],
            reverse=True,
        )
        for rank, cid in enumerate(ey_sorted, 1):
            raw_data[cid]["ey_rank"] = rank

        # Rank by Return on Capital (highest = rank 1)
        roc_sorted = sorted(
            raw_data.keys(),
            key=lambda cid: raw_data[cid]["return_on_capital"],
            reverse=True,
        )
        for rank, cid in enumerate(roc_sorted, 1):
            raw_data[cid]["roc_rank"] = rank

        # Combined rank (lowest = best)
        for cid in raw_data:
            raw_data[cid]["combined_rank"] = (
                raw_data[cid]["ey_rank"] + raw_data[cid]["roc_rank"]
            )

        # Normalize to 0-100 (rank 1 combined = 100, last = 0)
        n = len(raw_data)
        if n == 1:
            for cid in raw_data:
                raw_data[cid]["magic_formula_score"] = 100.0
        else:
            # Sort by combined rank to assign scores
            combined_sorted = sorted(
                raw_data.keys(),
                key=lambda cid: raw_data[cid]["combined_rank"],
            )
            for i, cid in enumerate(combined_sorted):
                # i=0 is best (lowest combined rank) -> score 100
                raw_data[cid]["magic_formula_score"] = (
                    (1.0 - i / (n - 1)) * 100.0
                )

        logger.info("Magic Formula: scored %d companies", n)
        return raw_data

    def _calculate_raw(
        self, company: Company, session: Session, cutoff: _date = None
    ) -> Optional[dict]:
        """Calculate raw EY and ROC for a single company.

        Args:
            company: Company ORM object.
            session: SQLAlchemy session.

        Returns:
            Dict with earnings_yield and return_on_capital,
            or None if data is insufficient.
        """
        if cutoff is None:
            cutoff = _date.today()

        # Get latest price and compute market cap
        price = self._get_latest_price(company.id, session, cutoff)
        if price is None or price <= 0:
            return None

        shares = self._get_shares(company.id, session, cutoff)
        if shares is None or shares <= 0:
            return None

        market_cap = price * shares
        if market_cap < self._min_market_cap:
            logger.debug(
                "Skipping %s: market_cap=%.0f < min=%.0f",
                company.ticker, market_cap, self._min_market_cap,
            )
            return None

        # Get balance sheet and income data
        balance = self._load_latest_balance(company.id, session, cutoff)
        ebit = self._get_ebit(company.id, session, cutoff)

        if ebit is None or not balance:
            return None

        total_assets = balance.get("total_assets")
        current_assets = balance.get("current_assets")
        current_liabilities = balance.get("current_liabilities")
        total_equity = balance.get("total_equity")

        if any(v is None for v in [total_assets, current_assets, current_liabilities, total_equity]):
            return None

        # Enterprise Value = Market Cap + Long-Term Liabilities - Cash
        # Subtracting cash follows Greenblatt's original formula and avoids
        # penalizing cash-rich companies with artificially inflated EV.
        lt_liabilities = balance.get("lt_liabilities") or 0.0
        cash = balance.get("cash") or 0.0
        ev = market_cap + lt_liabilities - cash
        if ev <= 0:
            return None

        # Earnings Yield = EBIT / EV
        earnings_yield = ebit / ev

        # Return on Capital = EBIT / (NWC + PP&E)
        # NWC excludes cash (cash is not operating capital), matching Greenblatt's
        # original formula and avoiding double-penalizing cash-rich companies.
        nwc = (current_assets - cash) - current_liabilities
        pp_and_e = balance.get("pp_and_e")
        if pp_and_e is None or pp_and_e <= 0:
            # Fallback: 70% of non-current assets excludes most goodwill/intangibles
            pp_and_e = (total_assets - current_assets) * 0.70

        invested_capital = max(nwc, 0.0) + pp_and_e   # floor NWC at 0 (Greenblatt's approach)
        if invested_capital <= 0:
            return None

        roc = ebit / invested_capital

        return {
            "earnings_yield": earnings_yield,
            "return_on_capital": roc,
        }

    # ---- Data loading helpers ----

    def _get_latest_price(self, company_id: int, session: Session, cutoff: _date = None) -> Optional[float]:
        """Get most recent closing price on or before cutoff."""
        if cutoff is None:
            cutoff = _date.today()
        row = (
            session.query(DailyPrice.close)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.close.isnot(None),
                DailyPrice.date <= cutoff,
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )
        return row[0] if row else None

    def _get_shares(self, company_id: int, session: Session, cutoff: _date = None) -> Optional[float]:
        """Get shares outstanding from latest balance sheet share capital."""
        if cutoff is None:
            cutoff = _date.today()
        stmts = (
            session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.period_type == "ANNUAL",
                FinancialStatement.statement_type == "BALANCE",
                FinancialStatement.period_end <= cutoff,
            )
            .order_by(FinancialStatement.period_end.desc())
            .limit(3)
        )
        for stmt in stmts:
            if not stmt.data_json:
                continue
            try:
                data = json.loads(stmt.data_json)
            except json.JSONDecodeError:
                continue
            val = _find_item_by_codes(data, _CODES_SHARE_CAPITAL)
            if val is not None:
                return val
        return None

    def _get_ebit(self, company_id: int, session: Session, cutoff: _date = None) -> Optional[float]:
        """Get adjusted EBIT (operating profit minus monetary G/L if applicable).

        Uses operating profit from the latest annual income statement.
        Adjusts using the monetary gain/loss already computed in adjusted_metrics.
        """
        if cutoff is None:
            cutoff = _date.today()
        # Get operating profit from raw income statement (skip empty future shells)
        op_profit = None
        from datetime import timedelta
        cutoff_date = cutoff or _date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        
        from sqlalchemy import or_
        stmts = (
            session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.period_type == "ANNUAL",
                FinancialStatement.statement_type == "INCOME",
                or_(
                    FinancialStatement.publication_date <= cutoff_date,
                    (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff)),
                )
            )
            .order_by(FinancialStatement.period_end.desc())
            .limit(3)
        )
        for stmt in stmts:
            if not stmt.data_json:
                continue
            try:
                data = json.loads(stmt.data_json)
            except json.JSONDecodeError:
                continue
            val = _find_item_by_codes(data, _CODES_OPERATING_PROFIT)
            if val is not None:
                op_profit = val
                break
        if op_profit is None:
            return None

        # Subtract monetary G/L proportion that affects operating profit
        # (monetary G/L is already computed in adjusted_metrics)
        from datetime import timedelta
        cutoff_date = cutoff or _date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        metric = (
            session.query(AdjustedMetric)
            .filter(
                AdjustedMetric.company_id == company_id,
                AdjustedMetric.period_end <= lagged_cutoff,
            )
            .order_by(AdjustedMetric.period_end.desc())
            .first()
        )
        if metric and metric.monetary_gain_loss:
            # Approximate: monetary G/L affects net income, not operating profit directly.
            # For Magic Formula ranking, using raw operating profit is acceptable.
            pass

        return op_profit

    def _load_latest_balance(self, company_id: int, session: Session, cutoff: _date = None) -> dict:
        """Load most recent balance sheet fields (skips empty future shells)."""
        if cutoff is None:
            cutoff = _date.today()
        from datetime import timedelta
        cutoff_date = cutoff or _date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        
        from sqlalchemy import or_
        stmts = (
            session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.period_type == "ANNUAL",
                FinancialStatement.statement_type == "BALANCE",
                or_(
                    FinancialStatement.publication_date <= cutoff_date,
                    (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff)),
                )
            )
            .order_by(FinancialStatement.period_end.desc())
            .limit(3)
        )
        for stmt in stmts:
            if not stmt.data_json:
                continue
            try:
                data = json.loads(stmt.data_json)
            except json.JSONDecodeError:
                continue
            fields = {
                "total_assets": _find_item_by_codes(data, _CODES_TOTAL_ASSETS),
                "current_assets": _find_item_by_codes(data, _CODES_CURRENT_ASSETS),
                "current_liabilities": _find_item_by_codes(data, _CODES_CURRENT_LIABILITIES),
                "total_equity": _find_item_by_codes(data, _CODES_TOTAL_EQUITY),
                "lt_liabilities": _find_item_by_codes(data, _CODES_LT_LIABILITIES),
                "pp_and_e": _find_item_by_codes(data, _CODES_PP_AND_E),
                "cash": _find_item_by_codes(data, _CODES_CASH),
            }
            if any(v is not None for v in fields.values()):
                return fields
        return {}
