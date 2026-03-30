"""Macro overlay module.

Classifies the market environment into RISK_OFF, RISK_ON, or NEUTRAL based on
macroeconomic indicators (CDS, USD/TRY, CPI, Real Rates).
Provides weight multipliers to adjust portfolio scoring based on the regime.
"""

import logging
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from bist_picker.db.schema import MacroRegime

logger = logging.getLogger(__name__)

# Thresholds
_CDS_RISK_OFF = 400.0
_CDS_RISK_ON = 300.0
_REAL_RATE_RISK_OFF = -5.0  # Real rate < -5% is very loose/inflationary -> risk off for currency
_REAL_RATE_RISK_ON = 0.0    # Positive real rates -> stability -> risk on

class MacroRegimeClassifier:
    """Classifies market regime and adjusts scoring weights."""

    def __init__(self, session: Session):
        self.session = session

    def classify(self, target_date: Optional[date] = None) -> str:
        """Determine the macro regime for the given date.

        If explicit regime is stored in DB, use it.
        Otherwise, calculate based on indicators.

        Returns:
            "RISK_ON", "RISK_OFF", or "NEUTRAL"
        """
        if target_date is None:
            target_date = date.today()

        # Get latest macro data on or before target_date
        row = (
            self.session.query(MacroRegime)
            .filter(MacroRegime.date <= target_date)
            .order_by(MacroRegime.date.desc())
            .first()
        )

        if not row:
            logger.warning("No macro data found. Defaulting to NEUTRAL.")
            return "NEUTRAL"

        if row.regime:
            return row.regime

        # Calculate inferred regime
        score = 0
        
        # CDS Check
        if row.turkey_cds_5y:
            if row.turkey_cds_5y > _CDS_RISK_OFF:
                score -= 2
            elif row.turkey_cds_5y < _CDS_RISK_ON:
                score += 1

        # Real Rate Check
        # Approximate Real Rate = Policy Rate - CPI / 2 (simple heuristic)
        # or just check CPI trend. 
        # Here we use policy_rate and cpi_yoy if available.
        if row.policy_rate_pct is not None and row.cpi_yoy_pct is not None:
             real_rate = row.policy_rate_pct - row.cpi_yoy_pct
             if real_rate < _REAL_RATE_RISK_OFF:
                 score -= 1 # Deep negative rates = inflationary instability
             elif real_rate > _REAL_RATE_RISK_ON:
                 score += 1 # Positive real rates = stability

        if score <= -2:
            return "RISK_OFF"
        elif score >= 1:
            return "RISK_ON"
        else:
            return "NEUTRAL"

    def get_weight_multipliers(self, regime: str) -> dict[str, float]:
        """Get weight multipliers for the given regime.

        Returns:
            Dict mapping factor names (e.g., 'growth', 'value_graham_dcf')
            to multipliers (e.g., 1.2, 0.8).
        """
        if regime == "RISK_OFF":
            # In risk-off, we prefer High Quality, Low Valuation, Dividends
            return {
                "quality_buffett": 1.2,
                "value_graham_dcf": 1.2,
                "piotroski": 1.1,
                "dividend_yield": 1.2,
                "growth": 0.8,
                "momentum": 0.5, # Momentum kills in bear markets
                "insider": 1.2, # Insider buying is a critical safety signal in bear markets
            }
        elif regime == "RISK_ON":
            # In risk-on, we prefer Growth, Momentum
            return {
                "growth": 1.3,
                "momentum": 1.2,
                "quality_buffett": 0.9,
                "value_graham_dcf": 0.8,
                "dividend_yield": 0.8,
                "technical": 1.1, # Trend tracking becomes more reliable
            }
        else:
            return {} # No adjustment for NEUTRAL
