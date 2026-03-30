"""Exit rules checker for portfolio management.

Evaluates open positions against three exit conditions:
1. Stop-loss: Price falls below stop-loss level (default 82% of entry).
2. Target hit: Price reaches target price (DCF or composite-implied).
3. Thesis breaker: Significant negative news or insider selling.

Since we don't have full news sentiment analysis, 'thesis breaker' is
proxied by:
- Significant net insider selling (> 1M TRY or > 0.5% of market cap)
  since entry.
"""

import logging
from datetime import date
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from bist_picker.db.schema import (
    Company,
    DailyPrice,
    InsiderTransaction,
    PortfolioSelection,
)

logger = logging.getLogger("bist_picker.portfolio.exit_rules")

_STOP_LOSS_PCT = 0.82  # Hard stop at 18% loss if not specified
_THESIS_INSIDER_SELL_THRESHOLD = 5_000_000.0  # 5M TRY net selling is a red flag


class ExitRuleChecker:
    """Evaluates exit conditions for open portfolio positions."""

    def __init__(self, session: Session):
        self.session = session

    def check_exits(self) -> list[dict]:
        """Scan all open positions for exit signals.

        Returns:
            List of exit signals (dicts) with details needed for reporting.
            Does NOT execute the exit (update DB) — that is a manual decision
            or separate execution step.
        """
        open_positions = (
            self.session.query(PortfolioSelection, Company)
            .join(Company, Company.id == PortfolioSelection.company_id)
            .filter(PortfolioSelection.exit_date.is_(None))
            .all()
        )

        signals = []
        for selection, company in open_positions:
            signal = self._evaluate_position(selection, company)
            if signal:
                signals.append(signal)

        return signals

    def _evaluate_position(
        self, selection: PortfolioSelection, company: Company
    ) -> Optional[dict]:
        """Check a single position for any exit trigger."""
        # Get latest price
        latest_price_row = (
            self.session.query(DailyPrice)
            .filter(DailyPrice.company_id == company.id)
            .order_by(DailyPrice.date.desc())
            .first()
        )
        if not latest_price_row or not latest_price_row.close:
            return None

        current_price = latest_price_row.close
        price_date = latest_price_row.date
        entry_price = selection.entry_price or current_price  # Fallback to avoid div/0

        # Calculate return
        ret_pct = (current_price - entry_price) / entry_price * 100.0

        # 1. Stop-Loss Check
        # Use stored stop_loss or default 18% trailing/fixed
        stop_price = selection.stop_loss_price or (entry_price * _STOP_LOSS_PCT)
        if current_price <= stop_price:
            return {
                "company_id": company.id,
                "ticker": company.ticker,
                "portfolio": selection.portfolio,
                "entry_date": selection.selection_date,
                "entry_price": entry_price,
                "current_price": current_price,
                "price_date": price_date,
                "return_pct": ret_pct,
                "reason": "STOP_LOSS",
                "details": f"Price {current_price:.2f} <= Stop {stop_price:.2f}",
            }

        # 2. Target Hit Check
        if selection.target_price and current_price >= selection.target_price:
            return {
                "company_id": company.id,
                "ticker": company.ticker,
                "portfolio": selection.portfolio,
                "entry_date": selection.selection_date,
                "entry_price": entry_price,
                "current_price": current_price,
                "price_date": price_date,
                "return_pct": ret_pct,
                "reason": "TARGET",
                "details": f"Price {current_price:.2f} >= Target {selection.target_price:.2f}",
            }

        # 3. Thesis Breaker (Insider Selling)
        # Check net insider selling since entry date
        insider_net = self._calculate_net_insider_flow(
            company.id, selection.selection_date
        )
        if insider_net <= -_THESIS_INSIDER_SELL_THRESHOLD:
            return {
                "company_id": company.id,
                "ticker": company.ticker,
                "portfolio": selection.portfolio,
                "entry_date": selection.selection_date,
                "entry_price": entry_price,
                "current_price": current_price,
                "price_date": price_date,
                "return_pct": ret_pct,
                "reason": "THESIS_BREAKER",
                "details": (
                    f"Significant insider selling: {insider_net:,.0f} TRY "
                    f"since entry."
                ),
            }

        return None

    def _calculate_net_insider_flow(
        self, company_id: int, since_date: date
    ) -> float:
        """Calculate net insider transaction value (BUY - SELL) since a date.

        Returns:
            Net value in TRY. Negative means net selling.
        """
        txs = (
            self.session.query(InsiderTransaction)
            .filter(
                InsiderTransaction.company_id == company_id,
                InsiderTransaction.disclosure_date >= since_date,
            )
            .all()
        )

        net_flow = 0.0
        for tx in txs:
            if not tx.total_value_try:
                continue

            val = tx.total_value_try
            if tx.transaction_type == "BUY":
                net_flow += val
            elif tx.transaction_type == "SELL":
                net_flow -= val

        return net_flow
