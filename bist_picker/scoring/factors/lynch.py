"""Peter Lynch PEG-based scoring factor for BIST Stock Picker.

Scores companies using the PEG ratio (P/E divided by REAL earnings growth)
and classifies them into Lynch categories based on real revenue growth.

Critical: Uses REAL (inflation-adjusted) growth rates. If real growth is
negative or zero, PEG is undefined and gets a low score.

PEG scoring scale:
  < 0.5  -> 100
  0.5-1.0 -> linear 80-100
  1.0-2.0 -> linear 40-80
  > 2.0  -> linear 0-40

Lynch categories (based on real revenue growth):
  > 20%  -> fast_grower
  10-20% -> stalwart
  2-10%  -> slow_grower
  negative with improving trend -> turnaround
  otherwise -> asset_play (catchall)
"""

import json
import math
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
    DailyPrice,
    FinancialStatement,
)

logger = logging.getLogger("bist_picker.scoring.factors.lynch")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"

# Item codes
_CODES_NET_SALES = ["3C"]
_CODES_SHARE_CAPITAL = ["2OA"]


class LynchScorer:
    """Scores companies using Peter Lynch's PEG ratio approach.

    Args:
        config_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._load_config()

    def _load_config(self) -> None:
        """Load threshold values from thresholds.yaml."""
        pass  # No Lynch-specific thresholds currently needed

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[_date] = None,
        scoring_context: Optional[Any] = None,
    ) -> Optional[dict]:
        """Score a single company on Lynch PEG criteria.

        Args:
            company_id: Database ID of the company.
            session: SQLAlchemy session.
            scoring_date: Date of scoring.
            scoring_context: Optional ScoringContext with pre-loaded data.

        Returns:
            Dict with peg_ratio, peg_score, lynch_category.
        """
        if scoring_context:
            metrics = scoring_context.get_metrics(company_id)
        else:
            company = session.get(Company, company_id)
            if company is None:
                logger.warning("Company ID %d not found", company_id)
                return None

            # Load adjusted metrics
            from datetime import timedelta
            cutoff_date = scoring_date or _date.today()
            lagged_cutoff = cutoff_date - timedelta(days=76)
            query = session.query(AdjustedMetric).filter(
                AdjustedMetric.company_id == company_id,
                AdjustedMetric.period_end <= lagged_cutoff,
            )
            metrics = query.order_by(AdjustedMetric.period_end).all()

        if len(metrics) < 2:
            logger.debug("Skipping %s: need >= 2 metric periods", company_id)
            return None

        latest = metrics[-1]

        # Get current price
        if scoring_context:
            price = scoring_context.get_latest_price(company_id)
        else:
            price = self._get_latest_price(company_id, session, scoring_date)
            
        if price is None or price <= 0:
            return None

        # Get adjusted EPS
        eps = latest.eps_adjusted
        if eps is None or eps <= 0:
            # Negative or zero earnings: PEG undefined
            category = self._classify_lynch(metrics, company_id, session, scoring_date)
            return {
                "peg_ratio": None,
                "peg_score": 5.0,  # Low score for negative/zero earnings
                "lynch_category": category,
            }

        pe = price / eps

        # Get real EPS growth rate — geometric mean (CAGR), not arithmetic mean.
        # Arithmetic mean overstates growth for volatile earnings (e.g. +200%, -50%
        # arithmetic avg = +75%, but geometric = ~41%).
        growth_rates = [
            m.real_eps_growth_pct for m in metrics
            if m.real_eps_growth_pct is not None
        ]

        if not growth_rates:
            # No growth data: can't compute PEG
            category = self._classify_lynch(metrics, company_id, session, scoring_date)
            return {
                "peg_ratio": None,
                "peg_score": 20.0,  # Modest score — we have earnings but no growth data
                "lynch_category": category,
            }

        # Geometric mean: product((1+r_i))^(1/n) - 1
        product = 1.0
        for r in growth_rates:
            factor = 1.0 + r
            if factor <= 0:
                # If any period lost >100%, geometric mean is undefined.
                # Treat as catastrophic decline.
                product = 0.0
                break
            product *= factor

        if product <= 0:
            geo_growth = -1.0  # Total wipeout
        else:
            geo_growth = product ** (1.0 / len(growth_rates)) - 1.0

        # Negative/zero growth: assign a graduated floor score instead of flat 0.
        # -5% growth → ~17/100, -50%+ growth → 0/100.
        if geo_growth <= 0:
            if geo_growth <= -0.50:
                neg_score = 0.0
            else:
                # Linear scale: 0% → 20, -50% → 0
                neg_score = max(0.0, 15.0 * (1.0 + geo_growth / 0.50))
            logger.debug(
                "Company %d: real EPS growth(geo) %.1f%% — assigning Lynch score %.1f",
                company_id, geo_growth * 100, neg_score,
            )
            category = self._classify_lynch(metrics, company_id, session, scoring_date)
            return {
                "peg_ratio": None,
                "peg_score": neg_score,
                "lynch_category": category,
            }

        # PEG = P/E / (real growth * 100)
        real_growth_pct = geo_growth * 100.0
        peg = pe / real_growth_pct

        peg_score = _score_peg(peg)
        category = self._classify_lynch_context(
            metrics, company_id, session, scoring_date, scoring_context
        ) if scoring_context else self._classify_lynch(
            metrics, company_id, session, scoring_date
        )

        return {
            "peg_ratio": peg,
            "peg_score": peg_score,
            "lynch_category": category,
        }

    def _classify_lynch_context(
        self,
        metrics: list[AdjustedMetric],
        company_id: int,
        session: Session,
        scoring_date: Optional[_date],
        scoring_context: Any,
    ) -> str:
        """Classify using context data."""
        # Calculate revenue growth from income statements
        stmts = scoring_context.get_statements(company_id, "INCOME")
        
        # We need 2 most recent annual income statements
        # Context returns all, sorted asc.
        if len(stmts) < 2:
            revenue_growth = None
        else:
            latest = stmts[-1]
            prev = stmts[-2]
            revenue_growth = self._calculate_growth_rate(latest, prev)

        return self._determine_category(revenue_growth, metrics)

    def _calculate_growth_rate(self, current_stmt, prev_stmt) -> Optional[float]:
        if not current_stmt.data_json or not prev_stmt.data_json:
            return None
        try:
            curr_data = json.loads(current_stmt.data_json)
            prev_data = json.loads(prev_stmt.data_json)
        except json.JSONDecodeError:
            return None
            
        curr_rev = _find_item_by_codes(curr_data, _CODES_NET_SALES)
        prev_rev = _find_item_by_codes(prev_data, _CODES_NET_SALES)
        
        if curr_rev is None or prev_rev is None or prev_rev <= 0:
            return None
            
        return (curr_rev / prev_rev) - 1.0

    def _determine_category(self, revenue_growth: Optional[float], metrics: list[AdjustedMetric]) -> str:
        if revenue_growth is None:
            return "asset_play"

        if revenue_growth > 0.20:
            return "fast_grower"
        elif revenue_growth > 0.10:
            return "stalwart"
        elif revenue_growth > 0.02:
            return "slow_grower"
        elif revenue_growth < 0:
            # Check if there's an improving trend (latest growth > previous)
            growth_rates = [
                m.real_eps_growth_pct for m in metrics
                if m.real_eps_growth_pct is not None
            ]
            if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[-2]:
                return "turnaround"
            return "turnaround"
        else:
            return "slow_grower"

    def _classify_lynch(
        self,
        metrics: list[AdjustedMetric],
        company_id: int,
        session: Session,
        scoring_date: Optional[_date] = None,
    ) -> str:
        """Classify company into Lynch category based on real revenue growth.

        Categories:
        - fast_grower: real revenue growth > 20%
        - stalwart: 10-20%
        - slow_grower: 2-10%
        - turnaround: negative growth but improving trend
        - asset_play: catchall

        Args:
            metrics: Adjusted metrics ordered by period.
            company_id: Company DB ID.
            session: SQLAlchemy session.
            scoring_date: PiT date.

        Returns:
            Lynch category string.
        """
        # Calculate revenue growth from income statements
        revenue_growth = self._get_real_revenue_growth(company_id, session, scoring_date)
        return self._determine_category(revenue_growth, metrics)

    def _get_real_revenue_growth(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[_date] = None,
    ) -> Optional[float]:
        """Calculate nominal revenue growth rate from income statements.

        Uses the two most recent annual periods. Returns nominal growth
        since revenue ratios are inflation-neutral for categorization.

        Returns:
            Growth rate as decimal (e.g., 0.15 for 15%), or None.
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

        statements = query.order_by(FinancialStatement.period_end.desc()).limit(2).all()

        if len(statements) < 2:
            return None

        revenues = []
        for stmt in statements:
            if not stmt.data_json:
                return None
            try:
                data = json.loads(stmt.data_json)
            except json.JSONDecodeError:
                return None
            rev = _find_item_by_codes(data, _CODES_NET_SALES)
            if rev is None or rev <= 0:
                return None
            revenues.append(rev)

        # statements are desc order: [latest, previous]
        current_rev, prev_rev = revenues[0], revenues[1]
        return (current_rev / prev_rev) - 1.0

    def _get_latest_price(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[_date] = None,
    ) -> Optional[float]:
        """Get most recent closing price."""
        query = session.query(DailyPrice.close).filter(
            DailyPrice.company_id == company_id,
            DailyPrice.close.isnot(None),
        )

        if scoring_date:
            query = query.filter(DailyPrice.date <= scoring_date)
            
        row = query.order_by(DailyPrice.date.desc()).first()
        return row[0] if row else None


def _score_peg(peg: float) -> float:
    """Convert PEG ratio to 0-100 score.

    Scoring scale:
    < 0.5  -> 100
    0.5-1.0 -> linear 80-100
    1.0-2.0 -> linear 40-80
    > 2.0  -> linear 0-40 (capped at 0 for very high PEG)

    Args:
        peg: PEG ratio value.

    Returns:
        Score 0-100.
    """
    if peg < 0.5:
        return 100.0
    elif peg <= 1.0:
        # Linear 100 -> 80 as PEG goes 0.5 -> 1.0
        return 100.0 - (peg - 0.5) / 0.5 * 20.0
    elif peg <= 2.0:
        # Linear 80 -> 40 as PEG goes 1.0 -> 2.0
        return 80.0 - (peg - 1.0) / 1.0 * 40.0
    else:
        # Linear 40 -> 0 as PEG goes 2.0 -> 4.0
        score = 40.0 - (peg - 2.0) / 2.0 * 40.0
        return max(0.0, score)
