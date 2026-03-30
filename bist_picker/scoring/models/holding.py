"""Holding model composite scorer for BIST Stock Picker.

Applies ONLY to company_type = 'HOLDING'.  Operating companies use the
factor pipeline (buffett, graham, piotroski, ...) instead.

Key design rule from CLAUDE.md:
  "Holdings are different: Do NOT apply P/E, gross margin, EBITDA to
   holdings.  Use NAV discount model."
  Do NOT strip IAS 29 monetary gain/loss for holdings — their subsidiary
  mix determines the appropriate treatment, not a blanket rule.

Four sub-factors, each cross-sectionally percentile-ranked (0-100) within
the holding universe.  Missing data causes that sub-factor's weight to be
redistributed among available sub-factors.

Factor             Weight  Direction   Notes
-----------------  ------  ---------   -------------------------------------------
nav_discount       0.35    lower=good  Market cap / estimated NAV; deep discount=buy
portfolio_quality  0.25    higher      Weighted avg composite score of listed subs
dividend_yield     0.20    higher      Annual dividend yield + 5-year consistency
governance         0.20    higher      Low related-party %, manual score fallback

NAV estimation strategy (priority order):
  1. Sum-of-parts: listed subsidiary market caps * ownership stake %
     Stakes read from config/subsidiaries.yaml.
  2. Fallback: book equity (balance sheet 2N) as a NAV proxy.
  nav_ratio = market_cap / estimated_nav   (lower = deeper discount = better)

Portfolio quality strategy:
  1. Find most-recent ScoringResult.composite_alpha for each subsidiary.
  2. Weighted average by stake_pct.
  3. If no subsidiary scoring data in DB yet, returns None (weight
     redistributed to other factors).

Dividend yield strategy:
  - Sum CorporateAction rows (action_type='DIVIDEND') in last 12 months.
  - adjustment_factor assumed to be dividend-per-share (TRY).
  - Consistency: proportion of last 5 calendar years with >= 1 dividend.
  - Combined score: yield_score * 0.6 + consistency_score * 0.4.

Governance strategy:
  - Primary: manual governance_score from config/subsidiaries.yaml (0-100).
  - Secondary: derive from related_party_revenue_pct in AdjustedMetric
      0% related-party -> 100;  >= 50% related-party -> 0.
  - Default: 50.0 (neutral) when no data is available.

Cross-sectional normalisation in score_all():
  - Collect raw ratios from every HOLDING company.
  - Percentile-rank within the holding universe.
  - 'lower is better' metrics are inverted (rank 0 -> score 100).
  - holding_composite = weighted average of available percentile scores.

Weights are read from config/scoring_weights.yaml -> 'holding' section.
Subsidiary stakes are read from config/subsidiaries.yaml -> 'holdings' section.
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from bist_picker.cleaning.inflation import _find_item_by_codes
from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    CorporateAction,
    DailyPrice,
    FinancialStatement,
    ScoringResult,
)

logger = logging.getLogger("bist_picker.scoring.models.holding")

_DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "scoring_weights.yaml"
)
_DEFAULT_SUBSIDIARIES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "subsidiaries.yaml"
)

# Company type this model handles
_HOLDING_TYPE = "HOLDING"

# Balance sheet item codes
_CODES_EQUITY = ["2N"]                # Total shareholders' equity
_CODES_SHARE_CAPITAL = ["2OA"]        # Paid-in share capital (par 1 TRY = 1 share)
_CODES_NET_INCOME = ["3L"]            # Net income (reported, not adjusted)

# Factor direction for cross-sectional ranking
_FACTOR_DIRECTION: dict[str, str] = {
    "nav_discount":       "lower",   # lower market_cap/NAV ratio = deeper discount = better
    "portfolio_quality":  "higher",
    "dividend_yield":     "higher",
    "governance":         "higher",
}

# Dividend yield scoring constants
_YIELD_FULL_SCORE = 0.08   # 8% yield -> 100
_YIELD_ZERO_SCORE = 0.00   # 0% yield -> 0
_CONSISTENCY_YEARS = 5     # lookback window for dividend consistency
_YIELD_WEIGHT = 0.60       # within the combined dividend score
_CONSISTENCY_WEIGHT = 0.40

# Governance scoring from related-party revenue
_RP_ZERO_PCT = 0.00        # 0% related-party -> governance 100
_RP_FULL_PCT = 0.50        # >=50% related-party -> governance 0

# NAV discount scoring
_NAV_RATIO_IDEAL = 0.50    # 50% discount -> score 100
_NAV_RATIO_FAIR  = 1.00    # no discount (at NAV) -> score 0


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class HoldingScorer:
    """Composite scorer for HOLDING companies.

    Usage:
        scorer = HoldingScorer()
        # Single company — returns raw ratios (not yet normalised)
        raw = scorer.score(company_id, session)

        # Full universe — returns percentile-ranked sub-scores + composite
        all_scores = scorer.score_all(session)

    Args:
        weights_path: Path to scoring_weights.yaml.
        subsidiaries_path: Path to subsidiaries.yaml.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        subsidiaries_path: Optional[Path] = None,
    ) -> None:
        self._weights: dict[str, float] = self._load_yaml(
            weights_path or _DEFAULT_WEIGHTS_PATH
        ).get("holding", {})
        sub_cfg = self._load_yaml(subsidiaries_path or _DEFAULT_SUBSIDIARIES_PATH)
        self._subsidiaries: dict[str, dict] = sub_cfg.get("holdings", {})

        if not self._weights:
            logger.warning(
                "No 'holding' section in scoring_weights.yaml — using equal weights"
            )
            self._weights = {k: 1.0 / len(_FACTOR_DIRECTION) for k in _FACTOR_DIRECTION}

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Extract raw holding metrics for a single company.

        Returns raw (un-normalised) values that score_all() will cross-
        sectionally rank.  No 0-100 normalisation happens here.

        Args:
            company_id: Database ID of the company.
            session: Active SQLAlchemy session.

        Returns:
            Dict with raw metric values, or None if the company is not
            a holding or has insufficient data.
        """
        company = session.get(Company, company_id)
        if company is None:
            logger.warning("Company id=%d not found", company_id)
            return None

        ctype = (company.company_type or "").upper()
        if ctype != _HOLDING_TYPE:
            logger.debug(
                "Skipping %s: company_type=%s (not a holding)",
                company.ticker, ctype,
            )
            return None

        ticker = company.ticker

        # Load most-recent annual BALANCE statement
        balance_rows = self._load_statements(
            company_id, "BALANCE", session, limit=1, scoring_date=scoring_date
        )
        cur_balance = balance_rows[0] if balance_rows else []

        # ── Market cap ────────────────────────────────────────────────────
        market_cap = self._calc_market_cap(
            company_id, cur_balance, session, scoring_date=scoring_date
        )
        if market_cap is None:
            logger.debug("Cannot compute market cap for holding %s", ticker)
            # Fallthrough; NAV discount will be None but other factors can still score

        # ── NAV (sum-of-parts or book equity) ────────────────────────────
        equity = _find_item_by_codes(cur_balance, _CODES_EQUITY)
        nav_ratio = self._calc_nav_ratio(
            ticker, market_cap, equity, session, scoring_date=scoring_date
        )

        # ── Portfolio quality (subsidiary composite scores) ───────────────
        portfolio_quality = self._calc_portfolio_quality(
            ticker, session, scoring_date=scoring_date
        )

        # ── Dividend yield + consistency ──────────────────────────────────
        div_score = self._calc_dividend_score(
            company_id, market_cap, session, scoring_date=scoring_date
        )

        # ── Governance ────────────────────────────────────────────────────
        governance = self._calc_governance(
            company_id, ticker, session, scoring_date=scoring_date
        )

        return {
            "nav_ratio":         nav_ratio,        # raw ratio (lower = better)
            "portfolio_quality": portfolio_quality, # raw 0-100 from subsidiaries
            "dividend_score":    div_score,         # pre-blended 0-100
            "governance":        governance,        # pre-scored 0-100
            # Diagnostics (not directly scored)
            "market_cap":        market_cap,
            "equity":            equity,
        }

    def score_all(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> dict[int, dict]:
        """Score all HOLDING companies with cross-sectional percentile ranks.

        Steps:
          1. Extract raw metrics for every active holding.
          2. Percentile-rank each metric within the holding universe.
             nav_discount is rank-inverted (lower raw = higher score).
          3. Compute holding_composite = weighted average of sub-scores.

        Returns:
            Dict mapping company_id -> score dict containing raw values,
            percentile sub-scores (*_score keys, 0-100), and
            holding_composite (0-100 or None).
        """
        holding_ids = [
            cid
            for (cid,) in session.query(Company.id)
            .filter(
                Company.is_active.is_(True),
                Company.company_type == _HOLDING_TYPE,
            )
            .all()
        ]

        raw: dict[int, dict] = {}
        for cid in holding_ids:
            if scoring_date is None:
                result = self.score(cid, session)
            else:
                result = self.score(cid, session, scoring_date=scoring_date)
            if result is not None:
                raw[cid] = result

        if not raw:
            return {}

        # Map score key -> raw dict key (and inversion flag)
        factor_to_raw = {
            "nav_discount":      "nav_ratio",
            "portfolio_quality": "portfolio_quality",
            "dividend_yield":    "dividend_score",
            "governance":        "governance",
        }

        # Percentile-rank each factor across the holding universe
        for factor, raw_key in factor_to_raw.items():
            direction = _FACTOR_DIRECTION[factor]
            values = {cid: raw[cid].get(raw_key) for cid in raw}
            percentiles = _cross_percentile(values, invert=(direction == "lower"))
            for cid, pct in percentiles.items():
                raw[cid][f"{factor}_score"] = pct

        # Compute weighted holding_composite and data_completeness per company
        total_factors = len(factor_to_raw)
        for cid in raw:
            raw[cid]["holding_composite"] = self._compute_composite(raw[cid])
            # Data completeness: how many of the 4 factors have a score
            n_available = sum(
                1 for f in factor_to_raw
                if raw[cid].get(f"{f}_score") is not None
            )
            raw[cid]["data_completeness"] = round(
                (n_available / total_factors) * 100.0, 1
            )

        return raw

    # ── Composite calculation ─────────────────────────────────────────────────

    def _compute_composite(self, scores: dict) -> Optional[float]:
        """Weighted average of available factor sub-scores (0-100 each).

        Missing sub-scores have their weight redistributed among available
        factors.  Returns None if no sub-scores are available.

        Args:
            scores: Dict containing <factor>_score keys.

        Returns:
            Weighted composite 0-100, or None.
        """
        parts: list[tuple[float, float]] = []
        for factor, weight in self._weights.items():
            key = f"{factor}_score"
            val = scores.get(key)
            if val is not None:
                parts.append((val, weight))

        if not parts:
            return None

        total_w = sum(w for _, w in parts)
        return round(sum(v * w for v, w in parts) / total_w, 2)

    # ── NAV estimation ────────────────────────────────────────────────────────

    def _calc_nav_ratio(
        self,
        ticker: str,
        market_cap: Optional[float],
        book_equity: Optional[float],
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Compute market_cap / estimated_NAV ratio.

        NAV is estimated via sum-of-parts (listed subsidiaries) when possible,
        otherwise falls back to book equity.

        Args:
            ticker: Parent holding ticker.
            market_cap: Current market capitalisation (TRY).
            book_equity: Book equity from latest balance sheet (thousands TRY).
            session: Active SQLAlchemy session.

        Returns:
            nav_ratio >= 0, or None if market_cap and NAV are unavailable.
        """
        if market_cap is None:
            return None

        # Attempt sum-of-parts NAV
        sop_nav = self._sum_of_parts_nav(ticker, session, scoring_date=scoring_date)

        if sop_nav is not None and sop_nav > 0:
            nav = sop_nav
            logger.debug(
                "%s NAV via sum-of-parts: %.0f M TRY", ticker, nav / 1e6
            )
        elif book_equity is not None and book_equity > 0:
            # book_equity from financial statements is in thousands TRY
            nav = book_equity * 1000.0
            logger.debug(
                "%s NAV via book equity: %.0f M TRY", ticker, nav / 1e6
            )
        else:
            logger.debug("%s: no NAV estimate available", ticker)
            return None

        return market_cap / nav

    def _sum_of_parts_nav(
        self,
        ticker: str,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Estimate holding NAV by summing listed subsidiary market caps.

        Each subsidiary's market cap is weighted by the parent's ownership stake.
        Subsidiaries not found in the DB are skipped.

        Returns:
            Estimated NAV in TRY, or None if no subsidiary data found.
        """
        holding_cfg = self._subsidiaries.get(ticker, {})
        subsidiaries = holding_cfg.get("subsidiaries", [])
        if not subsidiaries:
            return None

        total_nav = 0.0
        found_any = False

        for sub in subsidiaries:
            sub_ticker = sub.get("ticker")
            stake = sub.get("stake_pct", 0.0) / 100.0
            if not sub_ticker or stake <= 0:
                continue

            sub_company = (
                session.query(Company)
                .filter_by(ticker=sub_ticker, is_active=True)
                .first()
            )
            if sub_company is None:
                logger.debug("Subsidiary %s not found in DB", sub_ticker)
                continue

            sub_mcap = self._calc_market_cap(
                sub_company.id, [], session, scoring_date=scoring_date
            )
            if sub_mcap is None:
                logger.debug("Cannot get market cap for subsidiary %s", sub_ticker)
                continue

            total_nav += sub_mcap * stake
            found_any = True

        return total_nav if found_any else None

    # ── Portfolio quality ─────────────────────────────────────────────────────

    def _calc_portfolio_quality(
        self,
        ticker: str,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Weighted average composite score of known listed subsidiaries.

        Looks up ScoringResult.composite_alpha for each subsidiary in the
        most recent scoring run.  Returns None if no subsidiary scores exist
        yet (scoring hasn't run, or no listed subs).

        Args:
            ticker: Parent holding ticker.
            session: Active SQLAlchemy session.

        Returns:
            Weighted average score 0-100, or None.
        """
        holding_cfg = self._subsidiaries.get(ticker, {})
        subsidiaries = holding_cfg.get("subsidiaries", [])
        if not subsidiaries:
            return None  # weight redistributed

        weighted_sum = 0.0
        total_stake = 0.0

        for sub in subsidiaries:
            sub_ticker = sub.get("ticker")
            stake = sub.get("stake_pct", 0.0) / 100.0
            if not sub_ticker or stake <= 0:
                continue

            sub_company = (
                session.query(Company)
                .filter_by(ticker=sub_ticker, is_active=True)
                .first()
            )
            if sub_company is None:
                continue

            # Most recent scoring result for this subsidiary
            scoring = (
                session.query(ScoringResult)
                .filter(
                    ScoringResult.company_id == sub_company.id,
                    ScoringResult.scoring_date <= scoring_date if scoring_date else True,
                    ScoringResult.composite_alpha.isnot(None),
                )
                .order_by(ScoringResult.scoring_date.desc())
                .first()
            )
            if scoring is None:
                continue

            weighted_sum += scoring.composite_alpha * stake
            total_stake += stake

        if total_stake == 0:
            return None

        return weighted_sum / total_stake

    # ── Dividend score ────────────────────────────────────────────────────────

    def _calc_dividend_score(
        self,
        company_id: int,
        market_cap: Optional[float],
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Blend of dividend yield (60%) and dividend consistency (40%).

        Dividend per share is read from CorporateAction.adjustment_factor
        where action_type = 'DIVIDEND'.

        Args:
            company_id: Database ID of the holding company.
            market_cap: Current market cap in TRY (used only if we need per-share
                        yield from total dividends — prefer per-share amounts).
            session: Active SQLAlchemy session.

        Returns:
            Blended dividend score 0-100, or None if no dividend history.
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
            return None  # no dividend history

        # ── Yield (last 12 months) ────────────────────────────────────────
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
                (raw_yield - _YIELD_ZERO_SCORE)
                / (_YIELD_FULL_SCORE - _YIELD_ZERO_SCORE)
                * 100.0,
            )
            yield_score = max(0.0, yield_score)
        else:
            yield_score = 0.0

        # ── Consistency (last 5 years) ────────────────────────────────────
        years_with_dividends = {a.action_date.year for a in dividend_actions}
        # Only count years within the 5-year window
        current_year = as_of.year
        relevant_years = {
            y for y in years_with_dividends if y >= (current_year - _CONSISTENCY_YEARS)
        }
        consistency = len(relevant_years) / _CONSISTENCY_YEARS
        consistency_score = min(100.0, consistency * 100.0)

        return round(
            _YIELD_WEIGHT * yield_score + _CONSISTENCY_WEIGHT * consistency_score, 2
        )

    # ── Governance score ──────────────────────────────────────────────────────

    def _calc_governance(
        self,
        company_id: int,
        ticker: str,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> float:
        """Governance score 0-100 (higher = better).

        Priority:
          1. Manual governance_score from subsidiaries.yaml.
          2. Derived from related_party_revenue_pct in AdjustedMetric.
          3. Neutral default 50.0.

        Args:
            company_id: Database ID of the holding company.
            ticker: Company ticker for config lookup.
            session: Active SQLAlchemy session.

        Returns:
            Governance score 0-100 (always returns a value, never None).
        """
        # 1. Manual config override
        holding_cfg = self._subsidiaries.get(ticker, {})
        manual_score = holding_cfg.get("governance_score")
        if manual_score is not None:
            return float(manual_score)

        # 2. Derive from related-party revenue %
        cutoff_date = scoring_date or date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        metric = (
            session.query(AdjustedMetric)
            .filter(
                AdjustedMetric.company_id == company_id,
                AdjustedMetric.period_end <= lagged_cutoff,
                AdjustedMetric.related_party_revenue_pct.isnot(None),
            )
            .order_by(AdjustedMetric.period_end.desc())
            .first()
        )
        if metric is not None:
            rp_pct = metric.related_party_revenue_pct / 100.0  # convert to 0-1
            rp_pct = max(0.0, min(rp_pct, _RP_FULL_PCT))
            score = (1.0 - rp_pct / _RP_FULL_PCT) * 100.0
            return round(score, 2)

        # 3. Neutral default
        return 50.0

    # ── Market cap calculation ────────────────────────────────────────────────

    def _calc_market_cap(
        self,
        company_id: int,
        balance_data: list[dict],
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Estimate current market cap = latest close price * shares outstanding.

        Share count estimation (priority):
          1. Paid-in share capital (2OA) from balance sheet.
             Turkey par value = 1 TRY/share, values stored in thousands TRY.
             shares = 2OA_value * 1000
          2. AdjustedMetric: adjusted_net_income / eps_adjusted.

        Args:
            company_id: Database ID.
            balance_data: Decoded balance sheet data_json list.
            session: Active SQLAlchemy session.

        Returns:
            Market cap in TRY, or None if insufficient data.
        """
        cutoff_date = scoring_date or date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        price_row = (
            session.query(DailyPrice.close)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.date <= cutoff_date,
                DailyPrice.close.isnot(None),
                DailyPrice.close > 0,
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )
        if price_row is None:
            return None
        close = price_row[0]

        # Method 1: share capital from balance sheet (in thousands TRY)
        if balance_data:
            share_capital_k = _find_item_by_codes(balance_data, _CODES_SHARE_CAPITAL)
            if share_capital_k is not None and share_capital_k > 0:
                shares = share_capital_k * 1_000.0  # thousands -> units
                return close * shares

        # Method 2: AdjustedMetric (eps-based)
        metric = (
            session.query(AdjustedMetric)
            .filter(
                AdjustedMetric.company_id == company_id,
                AdjustedMetric.period_end <= lagged_cutoff,
                AdjustedMetric.eps_adjusted.isnot(None),
                AdjustedMetric.adjusted_net_income.isnot(None),
            )
            .order_by(AdjustedMetric.period_end.desc())
            .first()
        )
        if metric is not None and metric.eps_adjusted and metric.eps_adjusted != 0:
            shares = abs(metric.adjusted_net_income / metric.eps_adjusted)
            if shares > 0:
                return close * shares

        return None

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_statements(
        self,
        company_id: int,
        statement_type: str,
        session: Session,
        limit: int = 2,
        scoring_date: Optional[date] = None,
    ) -> list[list[dict]]:
        """Load the most recent `limit` annual statements of a given type.

        Returns a list of data_json arrays (already decoded), most recent first.
        Each inner list is a list of financial statement line items.

        Args:
            company_id: Database ID.
            statement_type: 'INCOME', 'BALANCE', or 'CASHFLOW'.
            session: Active SQLAlchemy session.
            limit: Number of periods to return.

        Returns:
            List of decoded statement data arrays.
        """
        from sqlalchemy import or_

        cutoff_date = scoring_date or date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        rows = (
            session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.statement_type == statement_type,
                FinancialStatement.period_type == "ANNUAL",
                FinancialStatement.data_json.isnot(None),
                or_(
                    FinancialStatement.publication_date <= cutoff_date,
                    (
                        FinancialStatement.publication_date.is_(None)
                        & (FinancialStatement.period_end <= lagged_cutoff)
                    ),
                ),
            )
            .order_by(FinancialStatement.period_end.desc())
            .limit(limit)
            .all()
        )

        result = []
        for row in rows:
            try:
                data = json.loads(row.data_json)
                if isinstance(data, list):
                    result.append(data)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "Failed to decode data_json for company_id=%d: %s",
                    company_id,
                    exc,
                )
        return result

    # ── Config loading ────────────────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load and parse a YAML config file.

        Args:
            path: Absolute path to YAML file.

        Returns:
            Parsed dict, or empty dict on error.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("Config file not found: %s — using defaults", path)
            return {}
        except yaml.YAMLError as exc:
            logger.error("Failed to parse %s: %s", path, exc)
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cross_percentile(
    values: dict[int, Optional[float]], invert: bool = False
) -> dict[int, Optional[float]]:
    """Cross-sectionally percentile-rank a dict of company_id -> raw value.

    Companies with None values receive None (not ranked).
    All other companies receive a 0-100 percentile score.
    If invert=True, the lowest raw value gets 100 and the highest gets 0.

    Args:
        values: Mapping of company_id -> raw metric (may contain None).
        invert: If True, lower raw value is better (e.g. nav_discount ratio).

    Returns:
        Mapping of company_id -> percentile score (0-100) or None.
    """
    valid = {cid: v for cid, v in values.items() if v is not None}
    result: dict[int, Optional[float]] = {cid: None for cid in values}

    if not valid:
        return result

    sorted_ids = sorted(valid.keys(), key=lambda c: valid[c])
    if invert:
        sorted_ids = list(reversed(sorted_ids))

    n = len(sorted_ids)
    for i, cid in enumerate(sorted_ids):
        result[cid] = (i / (n - 1) * 100.0) if n > 1 else 50.0

    return result
