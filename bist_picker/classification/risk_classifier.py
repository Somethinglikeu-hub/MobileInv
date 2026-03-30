"""Risk classification module for BIST Stock Picker.

Assigns each active company a risk tier (HIGH / MEDIUM / LOW) using five
market-data and financial-statement dimensions:

  1. Volatility   (25%) - 252-day annualised std dev of daily log-returns
  2. Beta         (25%) - vs BIST-100 over trailing 252 trading days
  3. Market cap   (20%) - latest close price x shares outstanding (item 2OA)
  4. Liquidity    (15%) - 20-day average daily TRY turnover
                           (source-aware: IsYatirim already stores TRY turnover;
                           Yahoo rows use close x share volume)
  5. Leverage     (15%) - Debt/Equity from most recent annual balance sheet

Each dimension is scored 1 (LOW risk) / 2 (MEDIUM) / 3 (HIGH risk).
Missing dimensions are excluded and their weights redistributed proportionally.
The weighted composite is mapped to the final tier via thresholds in
config/thresholds.yaml (risk_classifier section).

Final tier boundaries (defaults):
  composite <= 1.5  -> LOW
  composite <= 2.2  -> MEDIUM
  composite >  2.2  -> HIGH

classify_all() updates the risk_tier column in existing scoring_results rows
for the given scoring_date. Companies without a scoring_results row are
inserted with only risk_tier populated so other scoring modules can then
update the remaining columns.
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sqlalchemy.orm import Session

from bist_picker.db.schema import Company, DailyPrice, FinancialStatement, ScoringResult

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"
)

# Balance sheet item codes reused from existing conventions.
_CODES_TOTAL_ASSETS = ["1BL"]
_CODES_EQUITY = ["2N"]
_CODES_SHARE_CAPITAL = ["2OA"]   # par value 1 TRY -> value == shares outstanding

# BIST-100 benchmark ticker as stored in the companies table.
_BIST100_TICKER = "XU100"

# Minimum observations required for statistics to be meaningful.
_MIN_PRICE_OBSERVATIONS = 20
_MIN_BETA_OBSERVATIONS = 20


class RiskClassifier:
    """Classifies BIST companies into HIGH / MEDIUM / LOW risk tiers.

    Args:
        thresholds_path: Path to thresholds.yaml. Defaults to
            bist_picker/config/thresholds.yaml.
    """

    def __init__(self, thresholds_path: Optional[Path] = None) -> None:
        path = thresholds_path or _DEFAULT_THRESHOLDS_PATH
        with path.open("r", encoding="utf-8") as fh:
            all_thresholds = yaml.safe_load(fh)

        cfg = all_thresholds["risk_classifier"]
        self._weights: dict[str, float] = cfg["weights"]
        self._vol = cfg["volatility"]
        self._beta = cfg["beta"]
        self._mcap = cfg["market_cap"]
        self._liq = cfg["liquidity"]
        self._lev = cfg["leverage"]
        self._comp = cfg["composite"]

    # ------------------------------------------------------------------
    # Dimension scorers (return 1=LOW, 2=MEDIUM, 3=HIGH)
    # ------------------------------------------------------------------

    def _score_volatility(self, vol: float) -> int:
        if vol < self._vol["low_max"]:
            return 1
        if vol > self._vol["high_min"]:
            return 3
        return 2

    def _score_beta(self, beta: float) -> int:
        if beta < self._beta["low_max"]:
            return 1
        if beta > self._beta["high_min"]:
            return 3
        return 2

    def _score_market_cap(self, mcap: float) -> int:
        """Small cap = HIGH risk (3); large cap = LOW risk (1)."""
        if mcap >= self._mcap["large_min"]:
            return 1
        if mcap < self._mcap["small_max"]:
            return 3
        return 2

    def _score_liquidity(self, avg_turnover: float) -> int:
        """Low liquidity = HIGH risk (3); high liquidity = LOW risk (1)."""
        if avg_turnover >= self._liq["high_min"]:
            return 1
        if avg_turnover < self._liq["low_max"]:
            return 3
        return 2

    def _score_leverage(self, de_ratio: float) -> int:
        if de_ratio < self._lev["low_max"]:
            return 1
        if de_ratio > self._lev["high_min"]:
            return 3
        return 2

    # ------------------------------------------------------------------
    # Data fetchers
    # ------------------------------------------------------------------

    def _fetch_close_series(
        self,
        company_id: int,
        session: Session,
        days: int,
        scoring_date: Optional[date] = None,
    ) -> Optional["np.ndarray"]:
        """Return a numpy array of close prices ordered oldest -> newest.

        Args:
            company_id: Company PK.
            session: Active SQLAlchemy session.
            days: Number of most-recent trading days to fetch.

        Returns:
            1-D float array, or None if fewer than _MIN_PRICE_OBSERVATIONS rows.
        """
        rows = (
            session.query(DailyPrice.close)
            .filter(DailyPrice.company_id == company_id)
            .filter(DailyPrice.close.isnot(None))
            .filter(DailyPrice.date <= scoring_date if scoring_date else True)
            .order_by(DailyPrice.date.desc())
            .limit(days)
            .all()
        )
        if len(rows) < _MIN_PRICE_OBSERVATIONS:
            return None
        # rows are newest-first; reverse to chronological order.
        return np.array([r[0] for r in reversed(rows)], dtype=float)

    def _fetch_close_series_with_dates(
        self,
        company_id: int,
        session: Session,
        days: int,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Return a {date: close} dict for the most-recent ``days`` rows.

        Args:
            company_id: Company PK.
            session: Active SQLAlchemy session.
            days: Number of most-recent trading days.

        Returns:
            Dict mapping date -> float, or None if insufficient data.
        """
        rows = (
            session.query(DailyPrice.date, DailyPrice.close)
            .filter(DailyPrice.company_id == company_id)
            .filter(DailyPrice.close.isnot(None))
            .filter(DailyPrice.date <= scoring_date if scoring_date else True)
            .order_by(DailyPrice.date.desc())
            .limit(days)
            .all()
        )
        if len(rows) < _MIN_PRICE_OBSERVATIONS:
            return None
        return {r[0]: r[1] for r in rows}

    def _get_balance_item(
        self,
        company_id: int,
        session: Session,
        item_codes: list[str],
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Extract one item from the most recent annual balance sheet JSON.

        Args:
            company_id: Company PK.
            session: Active SQLAlchemy session.
            item_codes: List of item_code strings to match (first found wins).

        Returns:
            Float value, or None if not found.
        """
        from sqlalchemy import or_

        cutoff_date = scoring_date or date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        stmt = (
            session.query(FinancialStatement.data_json)
            .filter(FinancialStatement.company_id == company_id)
            .filter(FinancialStatement.statement_type == "BALANCE")
            .filter(FinancialStatement.period_type == "ANNUAL")
            .filter(FinancialStatement.data_json.isnot(None))
            .filter(
                or_(
                    FinancialStatement.publication_date <= cutoff_date,
                    (
                        FinancialStatement.publication_date.is_(None)
                        & (FinancialStatement.period_end <= lagged_cutoff)
                    ),
                )
            )
            .order_by(FinancialStatement.period_end.desc())
            .first()
        )
        if stmt is None:
            return None
        try:
            data = json.loads(stmt[0])
        except (json.JSONDecodeError, TypeError):
            return None
        return _find_item_by_codes(data, item_codes)

    # ------------------------------------------------------------------
    # Metric calculators
    # ------------------------------------------------------------------

    def _compute_volatility(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """252-day annualised volatility from daily log-returns.

        Args:
            company_id: Company PK.
            session: Active SQLAlchemy session.

        Returns:
            Annualised volatility (e.g. 0.35 for 35%), or None.
        """
        prices = self._fetch_close_series(
            company_id, session, days=253, scoring_date=scoring_date
        )
        if prices is None or len(prices) < _MIN_PRICE_OBSERVATIONS + 1:
            return None
        log_ret = np.diff(np.log(prices))
        if len(log_ret) < _MIN_PRICE_OBSERVATIONS:
            return None
        return float(np.std(log_ret, ddof=1) * np.sqrt(252))

    def _compute_beta(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Beta vs BIST-100 over trailing 252 trading days.

        Looks for the XU100 benchmark company in the companies table.
        Returns None if benchmark data is unavailable.

        Args:
            company_id: Company PK.
            session: Active SQLAlchemy session.

        Returns:
            Beta coefficient, or None.
        """
        bm_company = (
            session.query(Company.id)
            .filter(Company.ticker == _BIST100_TICKER)
            .first()
        )
        if bm_company is None:
            logger.debug("BIST-100 benchmark ticker '%s' not found in DB", _BIST100_TICKER)
            return None

        stock_map = self._fetch_close_series_with_dates(
            company_id, session, days=253, scoring_date=scoring_date
        )
        bm_map = self._fetch_close_series_with_dates(
            bm_company[0], session, days=253, scoring_date=scoring_date
        )

        if stock_map is None or bm_map is None:
            return None

        # Align on shared dates.
        common_dates = sorted(set(stock_map) & set(bm_map))
        if len(common_dates) < _MIN_BETA_OBSERVATIONS + 1:
            return None

        stock_prices = np.array([stock_map[d] for d in common_dates], dtype=float)
        bm_prices = np.array([bm_map[d] for d in common_dates], dtype=float)

        stock_ret = np.diff(np.log(stock_prices))
        bm_ret = np.diff(np.log(bm_prices))

        if len(stock_ret) < _MIN_BETA_OBSERVATIONS:
            return None

        bm_var = np.var(bm_ret, ddof=1)
        if bm_var == 0:
            return None

        cov = np.cov(stock_ret, bm_ret, ddof=1)
        return float(cov[0, 1] / bm_var)

    def _compute_market_cap(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Market cap in TRY = latest close price x shares outstanding.

        Shares outstanding is derived from balance sheet item 2OA (share
        capital at par value 1 TRY), so 2OA value equals share count in TRY.

        Args:
            company_id: Company PK.
            session: Active SQLAlchemy session.

        Returns:
            Market cap in TRY, or None.
        """
        price_row = (
            session.query(DailyPrice.close)
            .filter(DailyPrice.company_id == company_id)
            .filter(DailyPrice.close.isnot(None))
            .filter(DailyPrice.date <= scoring_date if scoring_date else True)
            .order_by(DailyPrice.date.desc())
            .first()
        )
        if price_row is None:
            return None
        latest_price = price_row[0]

        share_capital = self._get_balance_item(
            company_id, session, _CODES_SHARE_CAPITAL, scoring_date=scoring_date
        )
        if share_capital is None or share_capital <= 0:
            return None

        # share_capital in TRY (par value 1 TRY) == shares outstanding
        return float(latest_price * share_capital)

    def _compute_liquidity(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """20-day average daily TRY turnover.

        Args:
            company_id: Company PK.
            session: Active SQLAlchemy session.

        Returns:
            Average daily TRY turnover, or None.
        """
        rows = (
            session.query(DailyPrice.close, DailyPrice.volume, DailyPrice.source)
            .filter(DailyPrice.company_id == company_id)
            .filter(DailyPrice.close.isnot(None))
            .filter(DailyPrice.volume.isnot(None))
            .filter(DailyPrice.date <= scoring_date if scoring_date else True)
            .order_by(DailyPrice.date.desc())
            .limit(20)
            .all()
        )
        turnovers: list[float] = []
        for close_price, volume_value, source in rows:
            if volume_value is None:
                continue
            if (source or "").upper().startswith("YAHOO"):
                if close_price is None:
                    continue
                turnovers.append(float(close_price * volume_value))
            else:
                turnovers.append(float(volume_value))
        if not turnovers:
            return None
        return float(sum(turnovers) / len(turnovers))

    def _compute_leverage(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Debt/Equity from the most recent annual balance sheet.

        D/E = (total_assets - equity) / equity.

        Args:
            company_id: Company PK.
            session: Active SQLAlchemy session.

        Returns:
            D/E ratio (non-negative), or None if equity <= 0.
        """
        total_assets = self._get_balance_item(
            company_id, session, _CODES_TOTAL_ASSETS, scoring_date=scoring_date
        )
        equity = self._get_balance_item(
            company_id, session, _CODES_EQUITY, scoring_date=scoring_date
        )

        if total_assets is None or equity is None or equity <= 0:
            return None
        debt = total_assets - equity
        return float(max(debt, 0.0) / equity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> str:
        """Compute the risk tier for a single company.

        Missing dimensions are excluded and their weights redistributed
        proportionally among the remaining dimensions. If all five dimensions
        are missing, defaults to 'MEDIUM'.

        Args:
            company_id: Company PK in the companies table.
            session: Active SQLAlchemy session.

        Returns:
            'HIGH', 'MEDIUM', or 'LOW'.
        """
        # Collect (dimension_name, raw_metric, scorer_fn) tuples.
        metrics = {
            "volatility": self._compute_volatility(
                company_id, session, scoring_date=scoring_date
            ),
            "beta": self._compute_beta(
                company_id, session, scoring_date=scoring_date
            ),
            "market_cap": self._compute_market_cap(
                company_id, session, scoring_date=scoring_date
            ),
            "liquidity": self._compute_liquidity(
                company_id, session, scoring_date=scoring_date
            ),
            "leverage": self._compute_leverage(
                company_id, session, scoring_date=scoring_date
            ),
        }
        scorers = {
            "volatility": self._score_volatility,
            "beta": self._score_beta,
            "market_cap": self._score_market_cap,
            "liquidity": self._score_liquidity,
            "leverage": self._score_leverage,
        }

        # Score each available dimension.
        available: dict[str, tuple[int, float]] = {}  # dim -> (score, weight)
        for dim, metric in metrics.items():
            if metric is not None:
                score = scorers[dim](metric)
                available[dim] = (score, self._weights[dim])

        if not available:
            logger.warning(
                "company_id=%d: no dimensions computable, defaulting to MEDIUM",
                company_id,
            )
            return "MEDIUM"

        # Redistribute weights proportionally among available dimensions.
        total_weight = sum(w for _, w in available.values())
        composite = sum(
            score * (weight / total_weight)
            for score, weight in available.values()
        )

        if composite <= self._comp["low_max"]:
            tier = "LOW"
        elif composite > self._comp["high_min"]:
            tier = "HIGH"
        else:
            tier = "MEDIUM"

        logger.debug(
            "company_id=%d: composite=%.3f (%d/%d dims) -> %s",
            company_id,
            composite,
            len(available),
            len(metrics),
            tier,
        )
        return tier

    def classify_all(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> dict:
        """Classify all active companies and persist risk_tier to scoring_results.

        For each active company:
          - Compute risk tier via classify().
          - If a ScoringResult row exists for (company_id, scoring_date), update it.
          - Otherwise create a minimal ScoringResult row with risk_tier set.

        Args:
            session: Active SQLAlchemy session.
            scoring_date: Date to stamp on scoring_results rows. Defaults to today.

        Returns:
            Stats dict: {total, by_tier: {tier: count}}.
        """
        if scoring_date is None:
            scoring_date = date.today()

        companies = (
            session.query(Company)
            .filter(Company.is_active.is_(True))
            .filter(Company.company_type != "INDEX")
            .all()
        )

        # Build a lookup of existing scoring_results for this date.
        existing: dict[int, ScoringResult] = {}
        rows = (
            session.query(ScoringResult)
            .filter(ScoringResult.scoring_date == scoring_date)
            .all()
        )
        for row in rows:
            existing[row.company_id] = row

        stats: dict = {"total": len(companies), "by_tier": {}}

        for company in companies:
            tier = self.classify(company.id, session, scoring_date=scoring_date)

            if company.id in existing:
                existing[company.id].risk_tier = tier
            else:
                new_row = ScoringResult(
                    company_id=company.id,
                    scoring_date=scoring_date,
                    model_used=company.company_type or "OPERATING",
                    risk_tier=tier,
                )
                session.add(new_row)

            stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + 1

        session.flush()
        session.commit()

        logger.info(
            "classify_all: %d companies classified for %s -> %s",
            stats["total"],
            scoring_date,
            stats["by_tier"],
        )
        return stats


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_item_by_codes(
    data: list[dict], codes: list[str]
) -> Optional[float]:
    """Return the first item value matching any of the given item_codes.

    Replicates the helper from cleaning/inflation.py to avoid a circular import.

    Args:
        data: List of financial statement row dicts (item_code, value, ...).
        codes: Ordered list of item_code strings; first match is returned.

    Returns:
        Float value, or None if no match found.
    """
    for code in codes:
        for item in data:
            if item.get("item_code") == code:
                val = item.get("value")
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
    return None
