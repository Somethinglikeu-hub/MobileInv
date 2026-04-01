import logging
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from bist_picker.db.schema import PortfolioSelection, DailyPrice, Company

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks and calculates portfolio performance metrics."""

    def __init__(self, session: Session):
        self.session = session

    def calculate_portfolio_performance(self, portfolio_name: str) -> Dict[str, float]:
        """
        Calculates total realized and unrealized performance for a portfolio.
        Returns a dict with 'total_return_avg', 'active_return_avg', 'win_rate'.
        """
        try:
            # Get all selections for this portfolio
            selections = (
                self.session.query(PortfolioSelection)
                .filter(PortfolioSelection.portfolio == portfolio_name.upper())
                .all()
            )

            if not selections:
                return {"total_return_avg": 0.0, "active_return_avg": 0.0, "win_rate": 0.0}

            # Pre-fetch current prices for active positions to avoid N+1 queries
            active_ids = [s.company_id for s in selections if s.exit_date is None]
            current_prices = self._get_current_prices(active_ids)

            total_pct_sum = 0.0
            unrealized_pct_sum = 0.0
            active_count = 0
            wins = 0
            
            for s in selections:
                entry = float(s.entry_price or 0)
                if entry <= 0: continue
                
                if s.exit_date:
                    exit_p = float(s.exit_price or entry)
                else:
                    exit_p = current_prices.get(s.company_id, entry)
                    unrealized_pct_sum += (exit_p - entry) / entry
                    active_count += 1
                
                trade_ret = (exit_p - entry) / entry
                total_pct_sum += trade_ret
                if trade_ret > 0:
                    wins += 1

            count = len(selections)
            avg_return = (total_pct_sum / count) if count > 0 else 0.0
            active_avg = (unrealized_pct_sum / active_count) if active_count > 0 else 0.0
            win_rate = (wins / count * 100) if count > 0 else 0.0

            return {
                "total_return_avg": avg_return * 100,
                "active_return_avg": active_avg * 100,
                "win_rate": win_rate
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {}

    def fetch_benchmark_performance(self) -> float:
        """Calculates XU100 YTD return (from Jan 1 of current year)."""
        try:
            xu100 = self.session.query(Company).filter(Company.ticker == "XU100").first()
            if not xu100:
                return 0.0

            today = date.today()
            start_of_year = date(today.year, 1, 1)

            latest = (
                self.session.query(DailyPrice)
                .filter(DailyPrice.company_id == xu100.id)
                .order_by(DailyPrice.date.desc())
                .first()
            )
            start = (
                self.session.query(DailyPrice)
                .filter(DailyPrice.company_id == xu100.id)
                .filter(DailyPrice.date >= start_of_year)
                .order_by(DailyPrice.date.asc())
                .first()
            )

            if latest and start and float(start.close) > 0:
                return ((float(latest.close) - float(start.close)) / float(start.close)) * 100
            
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching benchmark performance: {e}")
            return 0.0

    def _get_current_prices(self, company_ids: List[int]) -> Dict[int, float]:
        """Fetches the latest available close price for given companies."""
        if not company_ids:
            return {}
        
        # Get latest price for each unique company_id
        unique_ids = list(set(company_ids))
        prices = {}
        for cid in unique_ids:
            price = (
                self.session.query(DailyPrice)
                .filter(DailyPrice.company_id == cid)
                .order_by(DailyPrice.date.desc())
                .first()
            )
            if price:
                prices[cid] = float(price.adjusted_close or price.close)
        return prices
