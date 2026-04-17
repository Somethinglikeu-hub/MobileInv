"""Enhanced Score Composer for BIST Stock Picker V2.5.

Orchestrates the Enhanced Pipeline — runs all enhanced scoring factors,
writes results to the enhanced_signals table, and blends them with
existing classic composite scores.

This module does NOT modify the classic pipeline. It reads classic
composite scores from ScoringResult and enhanced factor outputs
(EventScorer, InsiderScorer.score_enhanced, MacroNowcastScorer) to
produce a blended_alpha/beta/delta score.

Usage:
    composer = EnhancedComposer()
    composer.compose_all(session)  # writes to EnhancedSignal table
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from bist_picker.db.schema import (
    Company,
    EnhancedSignal,
    ScoringResult,
)
from bist_picker.scoring.factors.event_score import EventScorer
from bist_picker.scoring.factors.macro_nowcast_score import MacroNowcastScorer

logger = logging.getLogger("bist_picker.scoring.enhanced_composer")

_DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "scoring_weights.yaml"
)


def _load_weights(path: Path) -> dict:
    """Load scoring weights YAML."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Enhanced weights not found at %s — using defaults", path)
        return {}


_DEFAULT_ENHANCED_WEIGHTS = {
    "classic_weight": 0.70,
    "enhanced_weight": 0.30,
    # insider_cluster removed — no reliable data source for BIST.
    "event_score": 0.50,
    "macro_nowcast": 0.30,
    "analyst_tone": 0.20,
}


class EnhancedComposer:
    """Composes enhanced + classic blended scores.

    The enhanced composite score is:
        enhanced_composite = w1*event + w2*macro + w3*analyst_tone

    The final blended score is:
        blended = classic_weight * classic_composite + enhanced_weight * enhanced_composite

    Note: the former ``insider_cluster`` signal has been dropped because the
    BIST does not expose a reliable insider-transaction data source. Its
    weight was redistributed across the remaining enhanced factors.
    """

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        weights = _load_weights(weights_path or _DEFAULT_WEIGHTS_PATH)

        # Load enhanced alpha weights (fallback to defaults)
        self._alpha_weights = weights.get("enhanced_alpha", dict(_DEFAULT_ENHANCED_WEIGHTS))
        self._beta_weights = weights.get("enhanced_beta", dict(_DEFAULT_ENHANCED_WEIGHTS))
        self._delta_weights = weights.get("enhanced_delta", dict(_DEFAULT_ENHANCED_WEIGHTS))

        # Initialize scorers
        self._event_scorer = EventScorer()
        self._macro_scorer = MacroNowcastScorer()

    def compose(
        self,
        company_id: int,
        session: Session,
        portfolio: str = "alpha",
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Compose enhanced + classic blended score for one company.

        Args:
            company_id: Database ID of the company.
            session: SQLAlchemy session.
            portfolio: 'alpha', 'beta', or 'delta'.
            scoring_date: Scoring date.

        Returns:
            Dict with all enhanced scores, or None if no data.
        """
        if scoring_date is None:
            scoring_date = date.today()

        weights = self._get_weights(portfolio)

        # 1. Get individual enhanced factor scores
        event_result = self._event_scorer.score(company_id, session, scoring_date)
        event_score = event_result["event_score"] if event_result else None

        macro_result = self._macro_scorer.score_for_company(
            company_id, session, scoring_date,
        )
        macro_score = macro_result["macro_nowcast_score"] if macro_result else None

        # Analyst tone is TODO — for now, use None (will be weight-redistributed)
        analyst_tone_score = None

        # 2. Calculate enhanced composite (0-100)
        enhanced_composite = self._weighted_average(weights, {
            "event_score": event_score,
            "macro_nowcast": macro_score,
            "analyst_tone": analyst_tone_score,
        })

        # 3. Get classic composite from ScoringResult
        classic_row = (
            session.query(ScoringResult)
            .filter(
                ScoringResult.company_id == company_id,
                ScoringResult.scoring_date == scoring_date,
            )
            .first()
        )

        classic_composite = None
        if classic_row:
            # Use the matching portfolio composite
            if portfolio == "alpha":
                classic_composite = classic_row.composite_alpha
            elif portfolio == "beta":
                classic_composite = classic_row.composite_beta
            elif portfolio == "delta":
                classic_composite = classic_row.composite_delta

        # 4. Blend classic + enhanced
        classic_w = weights.get("classic_weight", 0.70)
        enhanced_w = weights.get("enhanced_weight", 0.30)

        if classic_composite is not None and enhanced_composite is not None:
            blended = classic_composite * classic_w + enhanced_composite * enhanced_w
        elif classic_composite is not None:
            blended = classic_composite  # No enhanced data — use classic only
        elif enhanced_composite is not None:
            blended = enhanced_composite  # No classic data — use enhanced only
        else:
            blended = None

        return {
            "event_score": event_score,
            "insider_cluster_score": None,  # signal dropped; column retained for schema compat
            "macro_nowcast_score": macro_score,
            "analyst_tone_score": analyst_tone_score,
            "enhanced_composite": round(enhanced_composite, 1) if enhanced_composite else None,
            "classic_composite": classic_composite,
            "blended": round(blended, 1) if blended else None,
            "portfolio": portfolio,
        }

    def compose_all(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
        portfolio: str = "alpha",
    ) -> dict[int, dict]:
        """Compose and persist enhanced scores for all active companies.

        Writes/updates EnhancedSignal rows in the database.

        Returns:
            Dict mapping company_id to enhanced score dict.
        """
        if scoring_date is None:
            scoring_date = date.today()

        company_ids = [
            cid for (cid,) in
            session.query(Company.id)
            .filter(Company.is_active.is_(True))
            .all()
        ]

        results = {}
        for cid in company_ids:
            result = self.compose(cid, session, portfolio, scoring_date)
            if result is None:
                continue

            # Persist to EnhancedSignal table
            existing = (
                session.query(EnhancedSignal)
                .filter(
                    EnhancedSignal.company_id == cid,
                    EnhancedSignal.scoring_date == scoring_date,
                )
                .first()
            )

            if existing:
                existing.event_score = result["event_score"]
                existing.insider_cluster_score = result["insider_cluster_score"]
                existing.macro_nowcast_score = result["macro_nowcast_score"]
                existing.analyst_tone_score = result["analyst_tone_score"]
                existing.enhanced_composite = result["enhanced_composite"]
                existing.classic_composite_alpha = result["classic_composite"]
                existing.blended_alpha = result["blended"]
            else:
                signal = EnhancedSignal(
                    company_id=cid,
                    scoring_date=scoring_date,
                    event_score=result["event_score"],
                    insider_cluster_score=result["insider_cluster_score"],
                    macro_nowcast_score=result["macro_nowcast_score"],
                    analyst_tone_score=result["analyst_tone_score"],
                    enhanced_composite=result["enhanced_composite"],
                    classic_composite_alpha=result["classic_composite"],
                    blended_alpha=result["blended"],
                )
                session.add(signal)

            results[cid] = result

        session.commit()
        logger.info(
            "Enhanced composition complete: %d companies scored for %s",
            len(results), scoring_date,
        )
        return results

    def _get_weights(self, portfolio: str) -> dict:
        """Return the weights dict for a portfolio."""
        if portfolio == "beta":
            return self._beta_weights
        elif portfolio == "delta":
            return self._delta_weights
        return self._alpha_weights

    @staticmethod
    def _weighted_average(
        weights: dict,
        factor_scores: dict[str, Optional[float]],
    ) -> Optional[float]:
        """Compute weighted average, redistributing None weights.

        Factors with None scores have their weight proportionally
        redistributed among factors that have values.
        """
        factor_keys = ["event_score", "macro_nowcast", "analyst_tone"]

        available = {}
        for key in factor_keys:
            score = factor_scores.get(key)
            w = weights.get(key, 0.0)
            if score is not None and w > 0:
                available[key] = (score, w)

        if not available:
            return None

        total_weight = sum(w for _, w in available.values())
        if total_weight <= 0:
            return None

        # Normalize weights to sum to 1.0
        composite = sum(
            score * (w / total_weight)
            for score, w in available.values()
        )

        return composite
