
import json
import logging
from datetime import date
from collections import defaultdict
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session
from sqlalchemy import func

from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    DailyPrice,
    FinancialStatement,
)

logger = logging.getLogger(__name__)

class ScoringContext:
    """Holds pre-fetched data for a batch of companies to avoid N+1 queries.
    
    Loads AdjustedMetrics, FinancialStatements, and latest Prices in bulk.
    """

    def __init__(self, session: Session, scoring_date: Optional[date] = None):
        self.session = session
        self.scoring_date = scoring_date
        
        # Data caches: {company_id: [objects]}
        self._metrics: Dict[int, List[AdjustedMetric]] = defaultdict(list)
        self._statements: Dict[int, Dict[str, List[FinancialStatement]]] = defaultdict(lambda: defaultdict(list))
        self._prices: Dict[int, float] = {}
        self._company_types: Dict[int, str] = {}
        
        # Track loaded IDs to avoid reload
        self._loaded_ids = set()

    def load_data(self, company_ids: List[int]) -> None:
        """Bulk load data for the given company IDs."""
        ids_to_load = [cid for cid in company_ids if cid not in self._loaded_ids]
        if not ids_to_load:
            return

        logger.info(f"Pre-fetching data for {len(ids_to_load)} companies...")

        # 1. Company Types (for filtering)
        companies = (
            self.session.query(Company.id, Company.company_type)
            .filter(Company.id.in_(ids_to_load))
            .all()
        )
        for cid, ctype in companies:
            self._company_types[cid] = (ctype or "").upper()

        # 2. Adjusted Metrics
        from datetime import timedelta
        cutoff_date = self.scoring_date or date.today()
        # Fallback 45-day lag for metrics since they don't have publication_date
        lagged_cutoff = cutoff_date - timedelta(days=76)
        
        query = (
            self.session.query(AdjustedMetric)
            .filter(
                AdjustedMetric.company_id.in_(ids_to_load),
                AdjustedMetric.period_end <= lagged_cutoff,
            )
        )
        
        metrics = query.order_by(AdjustedMetric.period_end).all()
        for m in metrics:
            self._metrics[m.company_id].append(m)

        # 3. Financial Statements (Annual only for now, as most scorers use annual)
        cutoff = self.scoring_date or date.today()
        stmt_query = (
            self.session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id.in_(ids_to_load),
                FinancialStatement.period_type == "ANNUAL",
            )
        )
        if self.scoring_date:
            from sqlalchemy import or_
            lagged_cutoff_stmt = self.scoring_date - timedelta(days=76)
            stmt_query = stmt_query.filter(
                or_(
                    FinancialStatement.publication_date <= self.scoring_date,
                    # Fallback for old/missing pub dates: 45 days after period_end
                    (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff_stmt)),
                )
            )
        else:
            # If no scoring_date, we still don't want to use future unfiled statements unconditionally
            lagged_cutoff_stmt = date.today() - timedelta(days=76)
            from sqlalchemy import or_
            stmt_query = stmt_query.filter(
                or_(
                    FinancialStatement.publication_date <= date.today(),
                    (FinancialStatement.publication_date.is_(None) & (FinancialStatement.period_end <= lagged_cutoff_stmt)),
                )
            )
            
        statements = stmt_query.order_by(FinancialStatement.period_end).all()
        for s in statements:
            # Skip shell records with all-null values (e.g. future period 2025/12
            # fetched before the company has filed its annual report).
            if s.data_json:
                try:
                    import json
                    items = json.loads(s.data_json)
                    if isinstance(items, list) and not any(
                        item.get("value") is not None for item in items
                    ):
                        continue  # All values null — skip this record
                except (json.JSONDecodeError, TypeError):
                    pass
            self._statements[s.company_id][s.statement_type].append(s)

        # 4. Latest Prices
        # We need the latest price ON or BEFORE scoring_date.
        # Subquery strategy for bulk latest price is efficient.
        # SELECT company_id, close FROM daily_prices WHERE (company_id, date) IN ...
        # Or simpler: Window function? SQLite supports window functions.
        
        # Max date per company <= scoring_date
        max_date_sq = (
            self.session.query(
                DailyPrice.company_id, 
                func.max(DailyPrice.date).label("max_date")
            )
            .filter(DailyPrice.company_id.in_(ids_to_load))
            .filter(DailyPrice.close.isnot(None))
        )
        if self.scoring_date:
            max_date_sq = max_date_sq.filter(DailyPrice.date <= self.scoring_date)
            
        max_date_sq = max_date_sq.group_by(DailyPrice.company_id).subquery()
        
        prices = (
            self.session.query(DailyPrice.company_id, DailyPrice.close)
            .join(max_date_sq, 
                  (DailyPrice.company_id == max_date_sq.c.company_id) & 
                  (DailyPrice.date == max_date_sq.c.max_date))
            .all()
        )
        
        for cid, close_price in prices:
            self._prices[cid] = close_price

        self._loaded_ids.update(ids_to_load)

    # --- Accessors ---

    def get_company_type(self, company_id: int) -> str:
        return self._company_types.get(company_id, "")

    def get_metrics(self, company_id: int) -> List[AdjustedMetric]:
        """Get adjusted metrics for company, ordered by date."""
        return self._metrics.get(company_id, [])

    def get_statements(self, company_id: int, statement_type: str) -> List[FinancialStatement]:
        """Get financial statements of specific type (e.g. 'BALANCE'), ordered by date."""
        return self._statements[company_id].get(statement_type, [])

    def get_latest_price(self, company_id: int) -> Optional[float]:
        return self._prices.get(company_id)
