"""Dividend yield + consistency factor scorer for BIST Stock Picker.

Computes a dividend score (0-100) for any company type based on:
  - Trailing 12-month dividend yield (60% weight)
  - 5-year dividend consistency — proportion of years with ≥1 dividend (40% weight)

Data source: CorporateAction table (action_type='DIVIDEND', adjustment_factor = DPS).

Adapted from HoldingScorer._calc_dividend_score() as a standalone factor
so operating companies can also benefit from the dividend yield signal
(especially via the macro overlay's RISK_OFF dividend_yield multiplier).
"""

import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from bist_picker.db.schema import Company, CorporateAction, DailyPrice

logger = logging.getLogger(__name__)

# ── Scoring constants ────────────────────────────────────────────────────────
_YIELD_FULL_SCORE = 0.08   # 8% yield → 100
_YIELD_ZERO_SCORE = 0.00   # 0% yield → 0
_CONSISTENCY_YEARS = 5     # lookback window for consistency check
_YIELD_WEIGHT = 0.60       # weight of yield within the blended score
_CONSISTENCY_WEIGHT = 0.40 # weight of consistency within the blended score


class DividendYieldScorer:
    """Batch-scored dividend yield + consistency factor.

    Usage::

        scorer = DividendYieldScorer()
        results = scorer.score_all(session)
        # results[company_id] = {"dividend_score": 0-100 or None}
    """

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Score a single company's dividend history.

        Args:
            company_id: Database ID of the company.
            session: Active SQLAlchemy session.

        Returns:
            Dict with ``dividend_score`` (0-100), or None if no dividend history.
        """
        as_of = scoring_date or date.today()
        cutoff_1y = as_of - timedelta(days=365)
        cutoff_5y = as_of - timedelta(days=_CONSISTENCY_YEARS * 365 + 1)

        dividend_actions = (
            session.query(CorporateAction)
            .filter(
                CorporateAction.company_id == company_id,
                CorporateAction.action_type == "DIVIDEND",
                CorporateAction.action_date >= cutoff_5y,
                CorporateAction.action_date <= as_of,
                CorporateAction.adjustment_factor.isnot(None),
                CorporateAction.adjustment_factor > 0,
            )
            .order_by(CorporateAction.action_date.desc())
            .all()
        )

        if not dividend_actions:
            return None  # no dividend history at all

        # ── Yield (last 12 months) ───────────────────────────────────────
        total_div_per_share = sum(
            a.adjustment_factor
            for a in dividend_actions
            if a.action_date >= cutoff_1y
        )

        # Current share price for yield denominator
        price_row = (
            session.query(DailyPrice.close)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.date <= as_of,
                DailyPrice.close.isnot(None),
                DailyPrice.close > 0,
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )

        if price_row is not None and price_row[0] and total_div_per_share > 0:
            raw_yield = total_div_per_share / price_row[0]
            yield_score = min(
                100.0,
                max(
                    0.0,
                    (raw_yield - _YIELD_ZERO_SCORE)
                    / (_YIELD_FULL_SCORE - _YIELD_ZERO_SCORE)
                    * 100.0,
                ),
            )
        else:
            yield_score = 0.0

        # ── Consistency (last 5 years) ───────────────────────────────────
        years_with_dividends = {a.action_date.year for a in dividend_actions}
        current_year = as_of.year
        relevant_years = {
            y for y in years_with_dividends if y >= (current_year - _CONSISTENCY_YEARS)
        }
        consistency = len(relevant_years) / _CONSISTENCY_YEARS
        consistency_score = min(100.0, consistency * 100.0)

        blended = round(
            _YIELD_WEIGHT * yield_score + _CONSISTENCY_WEIGHT * consistency_score, 2
        )

        return {"dividend_score": blended}

    def score_all(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> dict[int, dict]:
        """Score all active companies for dividend yield + consistency.

        Args:
            session: Active SQLAlchemy session.

        Returns:
            Dict mapping company_id → {"dividend_score": 0-100 or None}.
        """
        companies = (
            session.query(Company.id)
            .filter(Company.is_active == True)
            .all()
        )

        results: dict[int, dict] = {}
        scored = 0
        for (cid,) in companies:
            result = self.score(cid, session, scoring_date=scoring_date)
            if result is not None:
                results[cid] = result
                scored += 1

        logger.info(
            "DividendYieldScorer: scored %d / %d companies with dividend history",
            scored, len(companies),
        )
        return results
