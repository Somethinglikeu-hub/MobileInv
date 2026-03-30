"""Macro Nowcast scoring factor for BIST Stock Picker Enhanced Pipeline.

Uses BONC (Composite Leading Indicators) and credit card spending data
from the macro_nowcast table to generate forward-looking macro signals.

The score reflects:
  - BONC trend direction (leading indicator for economic cycle)
  - Credit card spending momentum (proxy for consumer demand)
  - LLM-based macro sentiment from financial headlines
"""

import json
import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from bist_picker.db.schema import Company, MacroNowcast

logger = logging.getLogger("bist_picker.scoring.factors.macro_nowcast_score")


# BONC trend scoring
_BONC_SCORES = {
    "RISING": 80.0,
    "FLAT": 50.0,
    "FALLING": 20.0,
}

# LLM macro sentiment scoring
_SENTIMENT_SCORES = {
    "BULLISH": 85.0,
    "NEUTRAL": 50.0,
    "CAUTIOUS": 30.0,
    "BEARISH": 15.0,
}

# Component weights for final macro nowcast score
_WEIGHT_BONC = 0.50
_WEIGHT_CREDIT_CARD = 0.20
_WEIGHT_LLM_SENTIMENT = 0.30


class MacroNowcastScorer:
    """Scores macro environment using forward-looking indicators.

    This is a market-wide signal (same for all companies in a given sector),
    not company-specific like other scoring factors.
    """

    def score_macro(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Calculate the macro nowcast score.

        Returns:
            Dict with macro_nowcast_score (0-100), bonc_component,
            credit_card_component, llm_sentiment_component, details.
            None if no data available.
        """
        if scoring_date is None:
            scoring_date = date.today()

        # Get most recent macro nowcast entry
        nowcast = (
            session.query(MacroNowcast)
            .filter(MacroNowcast.date <= scoring_date)
            .order_by(MacroNowcast.date.desc())
            .first()
        )

        if nowcast is None:
            return None

        # 1. BONC component
        bonc_score = _BONC_SCORES.get(nowcast.bonc_trend or "FLAT", 50.0)

        # Adjust for momentum: strong positive MoM boosts, negative dampens
        if nowcast.bonc_change_mom is not None:
            mom_adjustment = max(-15.0, min(15.0, nowcast.bonc_change_mom * 5.0))
            bonc_score = max(0.0, min(100.0, bonc_score + mom_adjustment))

        # 2. Credit card spending component
        cc_score = 50.0  # Default neutral
        if nowcast.credit_card_total_change_pct is not None:
            # Map MoM spending change to score
            # +10% MoM → 80 score, -10% MoM → 20 score
            cc_score = max(0.0, min(100.0, 50.0 + nowcast.credit_card_total_change_pct * 3.0))

        # 3. LLM sentiment component
        llm_score = _SENTIMENT_SCORES.get(nowcast.llm_macro_sentiment or "NEUTRAL", 50.0)

        # Adjust by confidence
        if nowcast.llm_confidence is not None:
            # Pull toward neutral if confidence is low
            llm_score = 50.0 + (llm_score - 50.0) * nowcast.llm_confidence

        # Weighted combination
        macro_score = (
            bonc_score * _WEIGHT_BONC +
            cc_score * _WEIGHT_CREDIT_CARD +
            llm_score * _WEIGHT_LLM_SENTIMENT
        )

        # Sector-specific adjustments from LLM
        sector_impacts = {}
        if nowcast.sector_impacts_json:
            try:
                sector_impacts = json.loads(nowcast.sector_impacts_json)
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "macro_nowcast_score": round(macro_score, 1),
            "bonc_component": round(bonc_score, 1),
            "credit_card_component": round(cc_score, 1),
            "llm_sentiment_component": round(llm_score, 1),
            "bonc_trend": nowcast.bonc_trend,
            "bonc_change_mom": nowcast.bonc_change_mom,
            "macro_sentiment": nowcast.llm_macro_sentiment,
            "sector_impacts": sector_impacts,
            "data_date": nowcast.date,
        }

    def score_for_company(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Score macro environment adjusted for a company's sector.

        Takes the base macro score and adjusts it using sector-specific
        LLM impact scores if available.
        """
        base = self.score_macro(session, scoring_date)
        if base is None:
            return None

        # Get company sector
        company = session.get(Company, company_id)
        if company is None:
            return base

        sector = company.sector_bist or ""
        sector_impacts = base.get("sector_impacts", {})

        # Apply sector adjustment if available
        adjusted_score = base["macro_nowcast_score"]
        sector_impact = sector_impacts.get(sector, 0.0)
        if sector_impact:
            # Impact is -1.0 to 1.0, scale to ±20 points
            adjustment = sector_impact * 20.0
            adjusted_score = max(0.0, min(100.0, adjusted_score + adjustment))

        result = dict(base)
        result["macro_nowcast_score"] = round(adjusted_score, 1)
        result["sector_adjustment"] = round(sector_impact * 20.0, 1) if sector_impact else 0.0

        return result

    def score_all(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> dict[int, dict]:
        """Score all active companies with macro-adjusted scores.

        Returns:
            Dict mapping company_id to macro nowcast score dict.
        """
        base = self.score_macro(session, scoring_date)
        if base is None:
            return {}

        companies = (
            session.query(Company.id)
            .filter(Company.is_active.is_(True))
            .all()
        )

        results = {}
        for (cid,) in companies:
            result = self.score_for_company(cid, session, scoring_date)
            if result is not None:
                results[cid] = result

        return results
