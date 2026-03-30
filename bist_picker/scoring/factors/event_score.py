"""KAP Event scoring factor for BIST Stock Picker Enhanced Pipeline.

Scores companies based on LLM-extracted KAP disclosure events stored in
the kap_events table. Events are weighted by type, monetary significance
(relative to company revenue), and recency (time-decayed).

Higher score = more positive forward-looking catalysts detected.
"""

import json
import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from bist_picker.db.schema import (
    Company,
    FinancialStatement,
    KapEvent,
)

logger = logging.getLogger("bist_picker.scoring.factors.event_score")

# Event type base weights (from llm_config.yaml, hardcoded here as defaults)
_EVENT_WEIGHTS = {
    "NEW_CONTRACT": 1.0,
    "CAPACITY_EXPANSION": 0.9,
    "MERGER_ACQUISITION": 0.8,
    "PARTNERSHIP": 0.7,
    "SHARE_BUYBACK": 0.7,
    "DIVIDEND": 0.6,
    "RATING_CHANGE": 0.5,
    "ASSET_SALE": 0.4,
    "CAPITAL_INCREASE": 0.3,
    "DEBT_ISSUANCE": 0.3,
    "BOARD_CHANGE": 0.2,
    "PENALTY": 0.1,
    "LAWSUIT": 0.1,
    "OTHER": 0.3,
}

# How quickly events lose impact (half-life in days)
_DECAY_HALF_LIFE_DAYS = 30

# Threshold: monetary_value / revenue must exceed this to be "significant"
_SIGNIFICANCE_THRESHOLD = 0.05  # 5% of annual revenue

# Maximum lookback window for events
_LOOKBACK_DAYS = 90


class EventScorer:
    """Scores companies based on recent KAP disclosure events.

    The score combines:
    1. Event type weight (contracts > buybacks > board changes)
    2. Monetary significance (contract_value / annual_revenue)
    3. LLM sentiment score
    4. Time decay (recent events matter more)
    """

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Calculate event score for a single company.

        Args:
            company_id: Database ID of the company.
            session: SQLAlchemy session.
            scoring_date: Date to score from.

        Returns:
            Dict with event_score (0-100), event_count, top_event_type,
            or None if no events found.
        """
        if scoring_date is None:
            scoring_date = date.today()

        cutoff_date = scoring_date - timedelta(days=_LOOKBACK_DAYS)

        # Fetch recent events for this company
        events = (
            session.query(KapEvent)
            .filter(
                KapEvent.company_id == company_id,
                KapEvent.disclosure_date >= cutoff_date,
                KapEvent.disclosure_date <= scoring_date,
            )
            .order_by(KapEvent.disclosure_date.desc())
            .all()
        )

        if not events:
            return None

        # Get trailing 12-month revenue for significance calculation
        trailing_revenue = self._get_trailing_revenue(company_id, session, scoring_date)

        # Calculate weighted event impact
        total_impact = 0.0
        top_event = None
        top_impact = 0.0

        for ev in events:
            impact = self._calculate_event_impact(ev, trailing_revenue, scoring_date)
            total_impact += impact
            if impact > top_impact:
                top_impact = impact
                top_event = ev

        # Normalize to 0-100 scale
        # Typical range: -5 to +5, map to 0-100
        event_score = max(0.0, min(100.0, (total_impact + 5.0) / 10.0 * 100.0))

        return {
            "event_score": round(event_score, 1),
            "event_count": len(events),
            "total_impact": round(total_impact, 3),
            "top_event_type": top_event.event_type if top_event else None,
            "top_event_date": top_event.disclosure_date if top_event else None,
        }

    def score_all(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> dict[int, dict]:
        """Score all companies with recent events.

        Returns:
            Dict mapping company_id to event score dict.
        """
        if scoring_date is None:
            scoring_date = date.today()

        cutoff_date = scoring_date - timedelta(days=_LOOKBACK_DAYS)

        # Find all companies with events in the window
        company_ids = (
            session.query(KapEvent.company_id)
            .filter(
                KapEvent.disclosure_date >= cutoff_date,
                KapEvent.disclosure_date <= scoring_date,
            )
            .distinct()
            .all()
        )

        results = {}
        for (cid,) in company_ids:
            result = self.score(cid, session, scoring_date)
            if result is not None:
                results[cid] = result

        return results

    def _calculate_event_impact(
        self,
        event: KapEvent,
        trailing_revenue: Optional[float],
        scoring_date: date,
    ) -> float:
        """Calculate the impact score for a single event.

        Combines: type_weight × significance × sentiment × decay
        """
        # 1. Event type weight
        type_weight = _EVENT_WEIGHTS.get(event.event_type or "OTHER", 0.3)

        # 2. Monetary significance (if available)
        significance_multiplier = 1.0
        if event.monetary_value and trailing_revenue and trailing_revenue > 0:
            ratio = event.monetary_value / trailing_revenue
            if ratio > 0.20:  # >20% of revenue
                significance_multiplier = 3.0
            elif ratio > _SIGNIFICANCE_THRESHOLD:  # >5%
                significance_multiplier = 1.5
            # Small deals (<5% revenue) keep multiplier at 1.0

        # 3. Sentiment from LLM (-1 to 1)
        sentiment = event.sentiment_score if event.sentiment_score is not None else 0.0

        # 4. Time decay (exponential)
        days_ago = (scoring_date - event.disclosure_date).days
        decay = 0.5 ** (days_ago / _DECAY_HALF_LIFE_DAYS)

        # Combine: positive sentiment × weight × significance × decay
        impact = sentiment * type_weight * significance_multiplier * decay

        return impact

    @staticmethod
    def _get_trailing_revenue(
        company_id: int,
        session: Session,
        scoring_date: date,
    ) -> Optional[float]:
        """Get trailing 12-month revenue for the company.

        Uses the most recent financial statement data available before scoring_date.
        """
        # Look for most recent annual income statement
        from datetime import timedelta
        cutoff_date = scoring_date
        lagged_cutoff = cutoff_date - timedelta(days=76)
        
        from sqlalchemy import or_
        stmt = (
            session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.statement_type == "INCOME",
                FinancialStatement.period_type == "ANNUAL",
                or_(
                    FinancialStatement.publication_date <= cutoff_date,
                    (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff)),
                )
            )
            .order_by(FinancialStatement.period_end.desc())
            .first()
        )

        if stmt is None or not stmt.data_json:
            return None

        try:
            data = json.loads(stmt.data_json)
            # Look for revenue field (Turkish financial statements)
            revenue = data.get("Hasılat") or data.get("Net Satışlar") or data.get("revenue")
            if revenue is not None:
                return float(revenue)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return None
