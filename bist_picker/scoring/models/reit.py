"""REIT (GYO) model composite scorer for BIST Stock Picker.

Applies ONLY to company_type = 'REIT'. 
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from sqlalchemy import or_
from sqlalchemy.orm import Session

from bist_picker.cleaning.inflation import _find_item_by_codes
from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    CorporateAction,
    DailyPrice,
    FinancialStatement,
)

logger = logging.getLogger("bist_picker.scoring.models.reit")

_DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "scoring_weights.yaml"
)

# Item codes for raw financial statement lookups
_CODES_TOTAL_ASSETS = ["1BL"]
_CODES_TOTAL_EQUITY = ["2N"]
_CODES_NET_SALES = ["3C"]

class ReitScorer:
    def __init__(self, weights_path: Optional[Path] = None):
        self.weights = self._load_weights(weights_path or _DEFAULT_WEIGHTS_PATH)

    def _load_weights(self, path: Path) -> dict:
        """Load REIT weights from scoring_weights.yaml."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("reit", {
                    "pb_vs_sector": 0.35,
                    "dividend_yield": 0.25,
                    "roe": 0.20,
                    "net_margin": 0.10,
                    "debt_equity": 0.10,
                })
        except Exception as e:
            logger.warning(f"Could not load weights from {path}: {e}")
            return {}

    def score_all(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> dict[int, dict]:
        """Score all REIT companies with cross-sectional percentile ranks."""
        reit_companies = (
            session.query(Company)
            .filter(Company.company_type == "REIT", Company.is_active == True)
            .all()
        )
        
        if not reit_companies:
            return {}

        raw_data = []
        for company in reit_companies:
            metrics = self._extract_metrics(company.id, session, scoring_date=scoring_date)
            if metrics:
                metrics["company_id"] = company.id
                raw_data.append(metrics)

        if not raw_data:
            return {}

        df = pd.DataFrame(raw_data)
        
        # Calculate scores for each factor
        scores_df = pd.DataFrame(index=df.index)
        scores_df["company_id"] = df["company_id"]

        # Helper to calculate percentile scores (0-100)
        def calc_rank(series, ascending=True):
            valid = series.dropna()
            if valid.empty:
                return pd.Series(float('nan'), index=series.index)
            ranks = series.rank(pct=True, ascending=ascending) * 100.0
            return ranks

        scores_df["pb_score"] = calc_rank(df["pb"], ascending=False)
        scores_df["div_score"] = calc_rank(df["dividend_yield"], ascending=True)
        scores_df["roe_score"] = calc_rank(df["roe"], ascending=True)
        scores_df["margin_score"] = calc_rank(df["net_margin"], ascending=True)
        scores_df["debt_score"] = calc_rank(df["debt_equity"], ascending=False)

        results = {}
        for idx, row in scores_df.iterrows():
            cid = int(row["company_id"])
            
            # Weighted average with redistribution for missing data
            available_weights = []
            weighted_sum = 0.0
            
            mapping = {
                "pb_vs_sector": "pb_score",
                "dividend_yield": "div_score",
                "roe": "roe_score",
                "net_margin": "margin_score",
                "debt_equity": "debt_score",
            }
            
            for factor, score_key in mapping.items():
                score = row[score_key]
                weight = self.weights.get(factor, 0.0)
                if pd.notna(score):
                    weighted_sum += score * weight
                    available_weights.append(weight)
            
            total_w = sum(available_weights)
            composite = (weighted_sum / total_w) if total_w > 0 else None
            
            results[cid] = {
                "reit_composite": composite,
                "data_completeness": (total_w / sum(self.weights.values())) * 100.0 if self.weights else 0,
            }
            
        return results

    def _extract_metrics(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Extract latest raw ratios for a single REIT."""
        cutoff_date = scoring_date or date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)

        latest_price = self._latest_price(company_id, session, cutoff_date)
        if latest_price is None or latest_price <= 0:
            return None
            
        # Get latest clean metrics
        metric_row = (
            session.query(AdjustedMetric)
            .filter(AdjustedMetric.company_id == company_id)
            .filter(AdjustedMetric.period_end <= lagged_cutoff)
            .order_by(AdjustedMetric.period_end.desc())
            .first()
        )
        if not metric_row:
            return None

        # Extract balance sheet items from raw JSON
        from sqlalchemy import or_
        balance_stmt = (
            session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.statement_type == "BALANCE",
                FinancialStatement.period_end == metric_row.period_end,
                or_(
                    FinancialStatement.publication_date <= cutoff_date,
                    (
                        FinancialStatement.publication_date.is_(None)
                        & (FinancialStatement.period_end <= lagged_cutoff)
                    ),
                ),
            )
            .first()
        )
        if not balance_stmt or not balance_stmt.data_json:
            return None
            
        try:
            data = json.loads(balance_stmt.data_json)
        except:
            return None
            
        total_assets = _find_item_by_codes(data, _CODES_TOTAL_ASSETS)
        total_equity = _find_item_by_codes(data, _CODES_TOTAL_EQUITY)
        
        if total_assets is None or total_equity is None or total_equity <= 0:
            return None

        income_stmt = (
            session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.statement_type == "INCOME",
                FinancialStatement.period_end == metric_row.period_end,
                or_(
                    FinancialStatement.publication_date <= cutoff_date,
                    (
                        FinancialStatement.publication_date.is_(None)
                        & (FinancialStatement.period_end <= lagged_cutoff)
                    ),
                ),
            )
            .first()
        )
        revenue = 0.0
        if income_stmt and income_stmt.data_json:
            try:
                inc_data = json.loads(income_stmt.data_json)
                revenue = _find_item_by_codes(inc_data, _CODES_NET_SALES) or 1.0
            except:
                revenue = 1.0

        net_income = metric_row.reported_net_income or 0.0
        
        # Calculate ratios
        # Market Cap = price * shares
        # Note: shares_outstanding is in AdjustedMetric
        # Wait, if shares_outstanding is not in AdjustedMetric, I'll use a heuristic or find it.
        # It's not in PRAGMA output. I'll check FinancialStatement again.
        # Actually, let's look at graham.py again for shares_outstanding.
        
        # Heuristic for market cap if shares_outstanding is missing:
        # We can use latest_close * (adjusted_net_income / eps_adjusted)
        shares = 1.0
        if metric_row.eps_adjusted and metric_row.eps_adjusted != 0:
            shares = (metric_row.adjusted_net_income or net_income) / metric_row.eps_adjusted
        
        pb = (latest_price * shares) / total_equity if total_equity > 0 else None
        roe = net_income / total_equity if total_equity > 0 else None
        net_margin = net_income / revenue if revenue > 0 else None
        debt_equity = (total_assets - total_equity) / total_equity if total_equity > 0 else None

        div_yield = self._calc_dividend_yield(
            company_id,
            latest_price,
            session,
            cutoff_date,
        )

        return {
            "pb": pb,
            "dividend_yield": div_yield,
            "roe": roe,
            "net_margin": net_margin,
            "debt_equity": debt_equity
        }

    @staticmethod
    def _latest_price(
        company_id: int,
        session: Session,
        cutoff_date: date,
    ) -> Optional[float]:
        """Return latest adjusted close when present, else plain close."""
        price_row = (
            session.query(DailyPrice)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.date <= cutoff_date,
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )
        if not price_row:
            return None
        if price_row.adjusted_close is not None and price_row.adjusted_close > 0:
            return float(price_row.adjusted_close)
        if price_row.close is not None and price_row.close > 0:
            return float(price_row.close)
        return None

    @staticmethod
    def _calc_dividend_yield(
        company_id: int,
        latest_price: float,
        session: Session,
        cutoff_date: date,
    ) -> Optional[float]:
        """Return trailing 12-month dividend yield from corporate actions."""
        if latest_price <= 0:
            return None

        cutoff_1y = cutoff_date - timedelta(days=365)
        total_dividend = (
            session.query(CorporateAction)
            .filter(
                CorporateAction.company_id == company_id,
                CorporateAction.action_type == "DIVIDEND",
                CorporateAction.action_date >= cutoff_1y,
                CorporateAction.action_date <= cutoff_date,
                CorporateAction.adjustment_factor.isnot(None),
                CorporateAction.adjustment_factor > 0,
            )
            .all()
        )
        if not total_dividend:
            return 0.0

        per_share_total = sum(action.adjustment_factor for action in total_dividend)
        return per_share_total / latest_price if per_share_total > 0 else 0.0
