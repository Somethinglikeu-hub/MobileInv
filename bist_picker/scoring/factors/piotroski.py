"""Piotroski F-Score scoring factor for BIST Stock Picker.

Calculates the 9-signal Piotroski F-Score for companies using adjusted
financial metrics and raw financial statements. Critical: signals F3, F5,
F6, F8, F9 use inflation-aware comparisons.

Key nuance for Turkey's IAS 29 environment:
- For RATIO signals (F3, F5, F6, F8, F9): ratios are inherently
  inflation-neutral because both numerator and denominator inflate
  together. We compare the ratios directly (using adjusted values
  where available from adjusted_metrics).
- For ABSOLUTE value signals (F1, F2, F4): we use already-adjusted
  values from the adjusted_metrics table.
- F7 (dilution): compares nominal share counts (inflation irrelevant).

Signals:
  F1: Positive adjusted ROA
  F2: Positive CFO
  F3: Improving adjusted ROA (YoY)
  F4: Accruals quality (CFO > adjusted net income)
  F5: Declining leverage (LT debt / total assets)
  F6: Improving current ratio
  F7: No share dilution
  F8: Improving gross margin
  F9: Improving asset turnover
"""

import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Any, Optional

import yaml
from sqlalchemy.orm import Session

from bist_picker.cleaning.inflation import _find_item_by_codes
from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    FinancialStatement,
)

logger = logging.getLogger("bist_picker.scoring.factors.piotroski")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"

# Item codes for raw financial statement lookups
_CODES_TOTAL_ASSETS = ["1BL"]
_CODES_CURRENT_ASSETS = ["1A"]
_CODES_CURRENT_LIABILITIES = ["2A"]
_CODES_LT_LIABILITIES = ["2B"]
_CODES_TOTAL_EQUITY = ["2N"]
_CODES_SHARE_CAPITAL = ["2OA"]
_CODES_GROSS_PROFIT = ["3D"]
_CODES_NET_SALES = ["3C"]
_CODES_CFO = ["4C"]

# Minimum periods needed
_MIN_PERIODS = 2


class PiotroskiScorer:
    """Calculates Piotroski F-Score (0-9) for BIST companies.

    Uses adjusted metrics for profitability signals and raw financial
    statements for balance sheet and efficiency signals. All YoY
    comparisons on ratios are inherently inflation-neutral.

    Args:
        config_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._thresholds: dict = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load threshold values from thresholds.yaml."""
        if not self._config_path.exists():
            logger.warning("Thresholds config not found: %s", self._config_path)
            return

        with open(self._config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._thresholds = config.get("piotroski", {})

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[_date] = None,
        scoring_context: Optional[Any] = None,
    ) -> Optional[dict]:
        """Calculate Piotroski F-Score for a single company.

        Args:
            company_id: Database ID of the company.
            session: SQLAlchemy session.
            scoring_date: Date of scoring.
            scoring_context: Optional ScoringContext with pre-loaded data.

        Returns:
            Dict with 9 individual signals, fscore_total, and fscore_normalized.
        """
        # Resolve company type — needed before loading metrics
        if scoring_context and hasattr(scoring_context, "get_company_type"):
            ctype = (scoring_context.get_company_type(company_id) or "").upper()
        else:
            _company = session.get(Company, company_id)
            if _company is None:
                logger.warning("Company ID %d not found", company_id)
                return None
            ctype = (_company.company_type or "").upper()

        # Piotroski signals are not applicable to REITs, Holdings, or Sports companies
        if ctype in ("REIT", "HOLDING", "SPORT", "FINANCIAL"):
            logger.debug("Skipping Piotroski for %d: type=%s", company_id, ctype)
            return None

        if scoring_context:
            metrics = scoring_context.get_metrics(company_id)
        else:
            company = session.get(Company, company_id)
            if company is None:
                logger.warning("Company ID %d not found", company_id)
                return None

            # Load metrics
            from datetime import timedelta
            cutoff_date = scoring_date or _date.today()
            lagged_cutoff = cutoff_date - timedelta(days=76)
            query = session.query(AdjustedMetric).filter(
                AdjustedMetric.company_id == company_id,
                AdjustedMetric.period_end <= lagged_cutoff,
            )
            metrics = query.order_by(AdjustedMetric.period_end).all()

        if len(metrics) < _MIN_PERIODS:
            # logger.debug("Skipping %s: insufficient metrics", company_id)
            return None

        current_metric = metrics[-1]
        previous_metric = metrics[-2]

        # Load raw financial data
        if scoring_context:
            bal_stmts = scoring_context.get_statements(company_id, "BALANCE")
            inc_stmts = scoring_context.get_statements(company_id, "INCOME")
            cfo_stmts = scoring_context.get_statements(company_id, "CASHFLOW")
            
            bal_current = self._find_stmt_data(bal_stmts, current_metric.period_end, "BALANCE")
            bal_previous = self._find_stmt_data(bal_stmts, previous_metric.period_end, "BALANCE")
            
            inc_current = self._find_stmt_data(inc_stmts, current_metric.period_end, "INCOME")
            inc_previous = self._find_stmt_data(inc_stmts, previous_metric.period_end, "INCOME")
            
            # CFO is just a single value
            cfo_current = self._find_stmt_item(cfo_stmts, current_metric.period_end, _CODES_CFO)
        else:
            bal_current = self._load_balance(company_id, current_metric.period_end, session, scoring_date)
            bal_previous = self._load_balance(company_id, previous_metric.period_end, session, scoring_date)
            inc_current = self._load_income(company_id, current_metric.period_end, session, scoring_date)
            inc_previous = self._load_income(company_id, previous_metric.period_end, session, scoring_date)
            cfo_current = self._load_cfo(company_id, current_metric.period_end, session, scoring_date)

        # Calculate each signal
        result = {}

        # -- Profitability signals --
        # -- Profitability signals --
        result["f1_positive_roa"] = self._f1_positive_roa(current_metric)
        result["f2_positive_cfo"] = self._f2_positive_cfo(cfo_current)
        result["f3_improving_roa"] = self._f3_improving_roa(current_metric, previous_metric)
        result["f4_accruals"] = self._f4_accruals(current_metric, cfo_current)

        # -- Leverage/Liquidity signals --
        result["f5_declining_leverage"] = self._f5_declining_leverage(
            bal_current, bal_previous,
        )
        result["f6_improving_liquidity"] = self._f6_improving_liquidity(
            bal_current, bal_previous,
        )
        result["f7_no_dilution"] = self._f7_no_dilution(
            bal_current, bal_previous,
        )

        # -- Efficiency signals --
        result["f8_improving_margin"] = self._f8_improving_margin(
            inc_current, inc_previous,
        )
        result["f9_improving_turnover"] = self._f9_improving_turnover(
            inc_current, inc_previous, bal_current, bal_previous,
        )

        # Total and normalized
        signals = [
            result["f1_positive_roa"], result["f2_positive_cfo"],
            result["f3_improving_roa"], result["f4_accruals"],
            result["f5_declining_leverage"], result["f6_improving_liquidity"],
            result["f7_no_dilution"], result["f8_improving_margin"],
            result["f9_improving_turnover"],
        ]

        # Count non-None signals — divide by actual count, not hardcoded 9.
        # A company with 8/9 signals computed should max at 100, not 88.9.
        valid_signals = [s for s in signals if s is not None]
        if not valid_signals:
            return None

        result["fscore_total"] = sum(s for s in valid_signals)
        result["fscore_normalized"] = (result["fscore_total"] / len(valid_signals)) * 100.0

        return result

    # ---- Individual signal methods ----

    def _f1_positive_roa(self, current: AdjustedMetric) -> Optional[int]:
        """F1: 1 if adjusted ROA > 0."""
        if current.roa_adjusted is None:
            return None
        return 1 if current.roa_adjusted > 0 else 0

    def _f2_positive_cfo(self, cfo: Optional[float]) -> Optional[int]:
        """F2: 1 if CFO > 0."""
        if cfo is None:
            return None
        return 1 if cfo > 0 else 0

    def _f3_improving_roa(
        self, current: AdjustedMetric, previous: AdjustedMetric
    ) -> Optional[int]:
        """F3: 1 if adjusted ROA improved YoY.

        Uses adjusted ROA which already accounts for IAS 29.
        ROA is a ratio, so direct comparison is inflation-neutral.
        """
        if current.roa_adjusted is None or previous.roa_adjusted is None:
            return None
        return 1 if current.roa_adjusted > previous.roa_adjusted else 0

    def _f4_accruals(
        self, current: AdjustedMetric, cfo: Optional[float]
    ) -> Optional[int]:
        """F4: 1 if CFO > reported net income.

        Both CFO and reported net income are pre-IAS-29-adjustment.
        Using adjusted_net_income vs raw CFO creates a systematically
        trivial signal in Turkey's IAS 29 environment (IAS 29 gains are
        stripped from adjusted NI but not from CFO, so CFO always wins).
        """
        if cfo is None or current.reported_net_income is None:
            return None
        return 1 if cfo > current.reported_net_income else 0

    def _f5_declining_leverage(
        self, bal_current: dict, bal_previous: dict,
    ) -> Optional[int]:
        """F5: 1 if LT debt / total assets decreased YoY.

        Ratio comparison is inherently inflation-neutral.
        """
        lev_current = _leverage_ratio(bal_current)
        lev_previous = _leverage_ratio(bal_previous)

        if lev_current is None or lev_previous is None:
            return None
        return 1 if lev_current < lev_previous else 0

    def _f6_improving_liquidity(
        self, bal_current: dict, bal_previous: dict,
    ) -> Optional[int]:
        """F6: 1 if current ratio improved YoY.

        Ratio comparison is inherently inflation-neutral.
        """
        cr_current = _current_ratio(bal_current)
        cr_previous = _current_ratio(bal_previous)

        if cr_current is None or cr_previous is None:
            return None
        return 1 if cr_current > cr_previous else 0

    def _f7_no_dilution(
        self, bal_current: dict, bal_previous: dict,
    ) -> Optional[int]:
        """F7: 1 if shares outstanding stayed same or decreased.

        Share count comparison is not affected by inflation.
        """
        shares_current = bal_current.get("share_capital")
        shares_previous = bal_previous.get("share_capital")

        if shares_current is None or shares_previous is None:
            return None
        return 1 if shares_current <= shares_previous else 0

    def _f8_improving_margin(
        self, inc_current: dict, inc_previous: dict,
    ) -> Optional[int]:
        """F8: 1 if gross margin improved YoY.

        Gross margin = gross_profit / net_sales. As a ratio, it is
        inherently inflation-neutral (both numerator and denominator
        inflate together). Direct comparison is correct.
        """
        margin_current = _gross_margin(inc_current)
        margin_previous = _gross_margin(inc_previous)

        if margin_current is None or margin_previous is None:
            return None
        return 1 if margin_current > margin_previous else 0

    def _f9_improving_turnover(
        self,
        inc_current: dict,
        inc_previous: dict,
        bal_current: dict,
        bal_previous: dict,
    ) -> Optional[int]:
        """F9: 1 if asset turnover improved YoY.

        Asset turnover = net_sales / total_assets. As a ratio, it is
        inherently inflation-neutral. Direct comparison is correct.
        """
        turn_current = _asset_turnover(inc_current, bal_current)
        turn_previous = _asset_turnover(inc_previous, bal_previous)

        if turn_current is None or turn_previous is None:
            return None
        return 1 if turn_current > turn_previous else 0

    # ---- Data loading helpers ----

    def _load_balance(
        self,
        company_id: int,
        period_end,
        session: Session,
        scoring_date: Optional[_date] = None,
    ) -> dict:
        """Load balance sheet fields for a specific period.

        Returns:
            Dict with total_assets, current_assets, current_liabilities,
            lt_liabilities, total_equity, share_capital.
        """
        from datetime import timedelta
        cutoff_date = scoring_date or _date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        
        from sqlalchemy import or_
        query = session.query(FinancialStatement).filter(
            FinancialStatement.company_id == company_id,
            FinancialStatement.period_end == period_end,
            FinancialStatement.period_type == "ANNUAL",
            FinancialStatement.statement_type == "BALANCE",
            or_(
                FinancialStatement.publication_date <= cutoff_date,
                (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff)),
            )
        )

        # Try most recent version first; skip records with all-null values
        for stmt in query.order_by(FinancialStatement.version.desc()).limit(3):
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
                "lt_liabilities": _find_item_by_codes(data, _CODES_LT_LIABILITIES),
                "total_equity": _find_item_by_codes(data, _CODES_TOTAL_EQUITY),
                "share_capital": _find_item_by_codes(data, _CODES_SHARE_CAPITAL),
            }
            if any(v is not None for v in fields.values()):
                return fields
        return {}

    def _load_income(
        self,
        company_id: int,
        period_end,
        session: Session,
        scoring_date: Optional[_date] = None,
    ) -> dict:
        """Load income statement fields for a specific period.

        Returns:
            Dict with gross_profit, net_sales.
        """
        from datetime import timedelta
        cutoff_date = scoring_date or _date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        
        from sqlalchemy import or_
        query = session.query(FinancialStatement).filter(
            FinancialStatement.company_id == company_id,
            FinancialStatement.period_end == period_end,
            FinancialStatement.period_type == "ANNUAL",
            FinancialStatement.statement_type == "INCOME",
            or_(
                FinancialStatement.publication_date <= cutoff_date,
                (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff)),
            )
        )

        # Try most recent version first; skip records with all-null values
        for stmt in query.order_by(FinancialStatement.version.desc()).limit(3):
            if not stmt.data_json:
                continue
            try:
                data = json.loads(stmt.data_json)
            except json.JSONDecodeError:
                continue
            fields = {
                "gross_profit": _find_item_by_codes(data, _CODES_GROSS_PROFIT),
                "net_sales": _find_item_by_codes(data, _CODES_NET_SALES),
            }
            if any(v is not None for v in fields.values()):
                return fields
        return {}

    def _load_cfo(
        self,
        company_id: int,
        period_end,
        session: Session,
        scoring_date: Optional[_date] = None,
    ) -> Optional[float]:
        """Load CFO from cashflow statement for a specific period.

        Returns:
            CFO value, or None if not found.
        """
        from datetime import timedelta
        cutoff_date = scoring_date or _date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        
        from sqlalchemy import or_
        query = session.query(FinancialStatement).filter(
            FinancialStatement.company_id == company_id,
            FinancialStatement.period_end == period_end,
            FinancialStatement.period_type == "ANNUAL",
            FinancialStatement.statement_type == "CASHFLOW",
            or_(
                FinancialStatement.publication_date <= cutoff_date,
                (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff)),
            )
        )

        # Try most recent version first; skip records with all-null values
        for stmt in query.order_by(FinancialStatement.version.desc()).limit(3):
            if not stmt.data_json:
                continue
            try:
                data = json.loads(stmt.data_json)
            except json.JSONDecodeError:
                continue
            val = _find_item_by_codes(data, _CODES_CFO)
            if val is not None:
                return val
        return None

    # ---- Context helpers ----

    def _find_stmt_data(self, stmts: list, period_end: date, stmt_type: str) -> dict:
        """Find statement for specific date and extract relevant fields."""
        if not stmts:
            return {}
        
        # Find matching stmt
        stmt = next((s for s in stmts if s.period_end == period_end), None)
        if not stmt or not stmt.data_json:
            return {}
            
        try:
            data = json.loads(stmt.data_json)
        except json.JSONDecodeError:
            return {}
            
        if stmt_type == "BALANCE":
            return {
                "total_assets": _find_item_by_codes(data, _CODES_TOTAL_ASSETS),
                "current_assets": _find_item_by_codes(data, _CODES_CURRENT_ASSETS),
                "current_liabilities": _find_item_by_codes(data, _CODES_CURRENT_LIABILITIES),
                "lt_liabilities": _find_item_by_codes(data, _CODES_LT_LIABILITIES),
                "total_equity": _find_item_by_codes(data, _CODES_TOTAL_EQUITY),
                "share_capital": _find_item_by_codes(data, _CODES_SHARE_CAPITAL),
            }
        elif stmt_type == "INCOME":
            return {
                "gross_profit": _find_item_by_codes(data, _CODES_GROSS_PROFIT),
                "net_sales": _find_item_by_codes(data, _CODES_NET_SALES),
            }
        return {}

    def _find_stmt_item(self, stmts: list, period_end: date, codes: list) -> Optional[float]:
        """Find statement and extract single item."""
        if not stmts:
            return None
            
        stmt = next((s for s in stmts if s.period_end == period_end), None)
        if not stmt or not stmt.data_json:
            return None
            
        try:
            data = json.loads(stmt.data_json)
            return _find_item_by_codes(data, codes)
        except json.JSONDecodeError:
            return None


# ---- Helper functions for ratio calculations ----


def _leverage_ratio(balance: dict) -> Optional[float]:
    """Calculate LT debt / total assets ratio.

    Args:
        balance: Balance sheet dict.

    Returns:
        Leverage ratio, or None if data missing.
    """
    lt_liab = balance.get("lt_liabilities")
    total_assets = balance.get("total_assets")

    if lt_liab is None or total_assets is None or total_assets == 0:
        return None
    return lt_liab / total_assets


def _current_ratio(balance: dict) -> Optional[float]:
    """Calculate current assets / current liabilities ratio.

    Args:
        balance: Balance sheet dict.

    Returns:
        Current ratio, or None if data missing.
    """
    ca = balance.get("current_assets")
    cl = balance.get("current_liabilities")

    if ca is None or cl is None or cl == 0:
        return None
    return ca / cl


def _gross_margin(income: dict) -> Optional[float]:
    """Calculate gross profit / net sales ratio.

    Args:
        income: Income statement dict.

    Returns:
        Gross margin, or None if data missing.
    """
    gp = income.get("gross_profit")
    ns = income.get("net_sales")

    if gp is None or ns is None or ns == 0:
        return None
    return gp / ns


def _asset_turnover(income: dict, balance: dict) -> Optional[float]:
    """Calculate net sales / total assets ratio.

    Args:
        income: Income statement dict.
        balance: Balance sheet dict.

    Returns:
        Asset turnover, or None if data missing.
    """
    ns = income.get("net_sales")
    ta = balance.get("total_assets")

    if ns is None or ta is None or ta == 0:
        return None
    return ns / ta
