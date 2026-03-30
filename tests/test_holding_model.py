"""Unit tests for scoring/models/holding.py.

Tests verify:
  1.  score() returns None for non-holding company types
  2.  score() returns None when company not found in DB
  3.  score() returns a dict (possibly with None sub-values) for a HOLDING
  4.  _calc_market_cap uses share capital (2OA) from balance sheet
  5.  _calc_market_cap falls back to AdjustedMetric eps-based shares
  6.  _calc_market_cap returns None when price is unavailable
  7.  _calc_nav_ratio returns None when market_cap is None
  8.  _calc_nav_ratio uses book equity as fallback when no listed subs
  9.  _calc_nav_ratio returns market_cap / (equity * 1000) for equity fallback
  10. _calc_portfolio_quality returns None when no subsidiaries configured
  11. _calc_portfolio_quality returns None when subsidiary not in DB
  12. _calc_portfolio_quality returns weighted avg from ScoringResult
  13. _calc_dividend_score returns None when no CorporateAction dividends
  14. _calc_dividend_score computes yield and consistency correctly
  15. _calc_dividend_score consistency counts distinct calendar years
  16. _calc_governance uses manual governance_score from config
  17. _calc_governance derives score from related_party_revenue_pct
  18. _calc_governance returns 50.0 when no data is available
  19. _compute_composite redistributes weight for missing sub-scores
  20. _compute_composite returns None when all sub-scores are None
  21. score_all() skips non-HOLDING companies
  22. score_all() returns empty dict when no holdings in DB
  23. score_all() cross-sectionally ranks nav_discount (lower=better, inverted)
  24. score_all() cross-sectionally ranks portfolio_quality (higher=better)
  25. score_all() adds holding_composite to each result
  26. _cross_percentile handles all-None inputs
  27. _cross_percentile single entry gets 50
"""

from datetime import date, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from bist_picker.scoring.models.holding import (
    HoldingScorer,
    _cross_percentile,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_WEIGHTS = {
    "nav_discount":      0.35,
    "portfolio_quality": 0.25,
    "dividend_yield":    0.20,
    "governance":        0.20,
}

_DEFAULT_SUBSIDIARIES = {
    "SAHOL": {
        "name": "Sabanci Holding",
        "governance_score": 60,
        "subsidiaries": [
            {"ticker": "AKBNK", "stake_pct": 40.9},
            {"ticker": "AKCNS", "stake_pct": 39.8},
        ],
    },
    "KCHOL": {
        "name": "Koc Holding",
        "governance_score": 65,
        "subsidiaries": [
            {"ticker": "TUPRS", "stake_pct": 51.0},
            {"ticker": "ARCLK", "stake_pct": 40.9},
        ],
    },
    "DOHOL": {
        "name": "Dogan Holding",
        "subsidiaries": [],  # no listed subs
    },
}


def _make_scorer(
    weights: Optional[dict] = None,
    subsidiaries: Optional[dict] = None,
) -> HoldingScorer:
    """Return a HoldingScorer bypassing file I/O."""
    scorer = HoldingScorer.__new__(HoldingScorer)
    scorer._weights = weights or _DEFAULT_WEIGHTS
    scorer._subsidiaries = subsidiaries if subsidiaries is not None else _DEFAULT_SUBSIDIARIES
    return scorer


def _company(cid: int = 1, company_type: str = "HOLDING", ticker: str = "SAHOL") -> MagicMock:
    c = MagicMock()
    c.id = cid
    c.company_type = company_type
    c.ticker = ticker
    return c


def _balance_item(code: str = "", value: float = 0.0) -> dict:
    return {"item_code": code, "desc_tr": "", "desc_eng": "", "value": value}


def _dividend_action(action_date: date, dps: float) -> MagicMock:
    a = MagicMock()
    a.action_date = action_date
    a.action_type = "DIVIDEND"
    a.adjustment_factor = dps
    return a


# ─────────────────────────────────────────────────────────────────────────────
# _cross_percentile helper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossPercentile:
    def test_normal_ranking(self):
        values = {1: 10.0, 2: 20.0, 3: 30.0}
        result = _cross_percentile(values, invert=False)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(50.0)
        assert result[3] == pytest.approx(100.0)

    def test_inverted_ranking(self):
        """Lower raw value = higher score (for nav_discount)."""
        values = {1: 0.4, 2: 0.7, 3: 1.0}  # nav ratios
        result = _cross_percentile(values, invert=True)
        assert result[1] == pytest.approx(100.0)  # deepest discount → best
        assert result[2] == pytest.approx(50.0)
        assert result[3] == pytest.approx(0.0)

    def test_none_values_receive_none_score(self):
        values = {1: 10.0, 2: None, 3: 30.0}
        result = _cross_percentile(values, invert=False)
        assert result[2] is None
        assert result[1] == pytest.approx(0.0)
        assert result[3] == pytest.approx(100.0)

    def test_all_none(self):
        values = {1: None, 2: None}
        result = _cross_percentile(values, invert=False)
        assert result[1] is None
        assert result[2] is None

    def test_single_entry_gets_50(self):
        values = {1: 42.0}
        result = _cross_percentile(values)
        assert result[1] == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────────
# score() gate-keeping tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreGating:
    def test_returns_none_for_operating(self):
        scorer = _make_scorer()
        session = MagicMock()
        session.get.return_value = _company(1, "OPERATING", "THYAO")
        assert scorer.score(1, session) is None

    def test_returns_none_for_bank(self):
        scorer = _make_scorer()
        session = MagicMock()
        session.get.return_value = _company(1, "BANK", "GARAN")
        assert scorer.score(1, session) is None

    def test_returns_none_when_company_not_found(self):
        scorer = _make_scorer()
        session = MagicMock()
        session.get.return_value = None
        assert scorer.score(1, session) is None

    def test_returns_dict_for_holding(self):
        """score() should return a dict (not None) for a HOLDING, even with no data."""
        scorer = _make_scorer()
        session = MagicMock()
        session.get.return_value = _company(1, "HOLDING", "DOHOL")
        with patch.object(scorer, "_load_statements", return_value=[]), \
             patch.object(scorer, "_calc_market_cap", return_value=None), \
             patch.object(scorer, "_calc_nav_ratio", return_value=None), \
             patch.object(scorer, "_calc_portfolio_quality", return_value=None), \
             patch.object(scorer, "_calc_dividend_score", return_value=None), \
             patch.object(scorer, "_calc_governance", return_value=50.0):
            result = scorer.score(1, session)
        assert result is not None
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# _calc_market_cap tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcMarketCap:
    def test_uses_share_capital_2OA(self):
        """Primary method: 2OA (thousands TRY) * 1000 * close."""
        scorer = _make_scorer()
        # balance_data with 2OA = 1_000 (thousands TRY) → 1M shares
        balance_data = [_balance_item(code="2OA", value=1_000.0)]

        session = MagicMock()
        price_q = MagicMock()
        price_q.filter.return_value.order_by.return_value.first.return_value = (50.0,)
        session.query.return_value = price_q

        result = scorer._calc_market_cap(1, balance_data, session)
        # shares = 1000 * 1000 = 1_000_000; market_cap = 1_000_000 * 50 = 50_000_000
        assert result == pytest.approx(50_000_000.0)

    def test_falls_back_to_eps_when_no_2OA(self):
        """Fallback: adjusted_net_income / eps_adjusted."""
        scorer = _make_scorer()
        balance_data: list = []  # no 2OA

        session = MagicMock()

        # Price query returns 100 TRY.
        # NOTE: _calc_market_cap calls session.query(DailyPrice.close) — the argument
        # is a SQLAlchemy column attribute, not the DailyPrice class itself.
        # We can't dispatch on `model_class is DailyPrice`, so we dispatch by exclusion.
        price_q = MagicMock()
        price_q.filter.return_value.order_by.return_value.first.return_value = (100.0,)

        # AdjustedMetric: ni=50000, eps=5.0 → shares=10000
        metric = MagicMock()
        metric.eps_adjusted = 5.0
        metric.adjusted_net_income = 50_000.0
        adj_q = MagicMock()
        adj_q.filter.return_value.order_by.return_value.first.return_value = metric

        def query_side(model_class):
            from bist_picker.db.schema import AdjustedMetric
            if model_class is AdjustedMetric:
                return adj_q
            # DailyPrice.close column query arrives here (not the class)
            return price_q

        session.query.side_effect = query_side

        result = scorer._calc_market_cap(1, balance_data, session)
        # shares = 50000 / 5.0 = 10000; market_cap = 100 * 10000 = 1_000_000
        assert result == pytest.approx(1_000_000.0)

    def test_returns_none_when_no_price(self):
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.order_by.return_value.first.return_value = None
        session.query.return_value = q
        assert scorer._calc_market_cap(1, [], session) is None


# ─────────────────────────────────────────────────────────────────────────────
# _calc_nav_ratio tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcNavRatio:
    def test_returns_none_when_market_cap_is_none(self):
        scorer = _make_scorer()
        session = MagicMock()
        result = scorer._calc_nav_ratio("SAHOL", None, 5000.0, session)
        assert result is None

    def test_falls_back_to_book_equity(self):
        """When sum-of-parts unavailable, use book equity (thousands TRY)."""
        scorer = _make_scorer()
        session = MagicMock()

        with patch.object(scorer, "_sum_of_parts_nav", return_value=None):
            # equity = 10_000 (thousands) = 10_000_000 TRY
            # market_cap = 4_000_000 → ratio = 0.4
            result = scorer._calc_nav_ratio("DOHOL", 4_000_000.0, 10_000.0, session)

        assert result == pytest.approx(0.40)

    def test_returns_none_when_no_equity_and_no_sop(self):
        scorer = _make_scorer()
        session = MagicMock()
        with patch.object(scorer, "_sum_of_parts_nav", return_value=None):
            result = scorer._calc_nav_ratio("DOHOL", 5_000_000.0, None, session)
        assert result is None

    def test_returns_none_when_equity_zero(self):
        scorer = _make_scorer()
        session = MagicMock()
        with patch.object(scorer, "_sum_of_parts_nav", return_value=None):
            result = scorer._calc_nav_ratio("DOHOL", 5_000_000.0, 0.0, session)
        assert result is None

    def test_uses_sum_of_parts_when_available(self):
        """Sum-of-parts NAV should take priority over book equity."""
        scorer = _make_scorer()
        session = MagicMock()
        sop_nav = 20_000_000.0  # from listed subs
        with patch.object(scorer, "_sum_of_parts_nav", return_value=sop_nav):
            result = scorer._calc_nav_ratio("SAHOL", 10_000_000.0, 5_000.0, session)
        # nav_ratio = 10_000_000 / 20_000_000 = 0.5
        assert result == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# _calc_portfolio_quality tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcPortfolioQuality:
    def test_returns_none_when_no_subsidiaries(self):
        """Holdings with empty subsidiary list return None (weight redistributed)."""
        scorer = _make_scorer()
        session = MagicMock()
        result = scorer._calc_portfolio_quality("DOHOL", session)
        assert result is None

    def test_returns_none_when_subsidiary_not_in_db(self):
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter_by.return_value.first.return_value = None  # not in DB
        session.query.return_value = q
        result = scorer._calc_portfolio_quality("SAHOL", session)
        assert result is None

    def test_returns_none_when_no_scoring_results(self):
        scorer = _make_scorer()
        session = MagicMock()

        sub_company = MagicMock()
        sub_company.id = 99

        def query_side(model_class):
            from bist_picker.db.schema import Company, ScoringResult
            q = MagicMock()
            if model_class is Company:
                q.filter_by.return_value.first.return_value = sub_company
            elif model_class is ScoringResult:
                q.filter.return_value.order_by.return_value.first.return_value = None
            return q

        session.query.side_effect = query_side
        result = scorer._calc_portfolio_quality("SAHOL", session)
        assert result is None

    def test_computes_weighted_average(self):
        """Weighted average of subsidiary composite scores by stake %."""
        scorer = _make_scorer(subsidiaries={
            "PARENT": {
                "name": "Parent Holding",
                "subsidiaries": [
                    {"ticker": "SUB1", "stake_pct": 60.0},
                    {"ticker": "SUB2", "stake_pct": 40.0},
                ],
            }
        })

        sub1 = MagicMock(); sub1.id = 10
        sub2 = MagicMock(); sub2.id = 20
        scoring1 = MagicMock(); scoring1.composite_alpha = 80.0
        scoring2 = MagicMock(); scoring2.composite_alpha = 40.0

        call_count = [0]

        def query_side(model_class):
            from bist_picker.db.schema import Company, ScoringResult
            q = MagicMock()
            if model_class is Company:
                def filter_by_side(**kw):
                    fq = MagicMock()
                    if kw.get("ticker") == "SUB1":
                        fq.first.return_value = sub1
                    elif kw.get("ticker") == "SUB2":
                        fq.first.return_value = sub2
                    else:
                        fq.first.return_value = None
                    return fq
                q.filter_by.side_effect = filter_by_side
            elif model_class is ScoringResult:
                call_count[0] += 1
                inner = MagicMock()
                if call_count[0] == 1:
                    inner.filter.return_value.order_by.return_value.first.return_value = scoring1
                else:
                    inner.filter.return_value.order_by.return_value.first.return_value = scoring2
                return inner
            return q

        session = MagicMock()
        session.query.side_effect = query_side

        result = scorer._calc_portfolio_quality("PARENT", session)
        # weighted avg: (80 * 0.6 + 40 * 0.4) / (0.6 + 0.4) = (48 + 16) / 1.0 = 64
        assert result == pytest.approx(64.0)


# ─────────────────────────────────────────────────────────────────────────────
# _calc_dividend_score tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcDividendScore:
    def test_returns_none_when_no_dividends(self):
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.order_by.return_value.all.return_value = []
        session.query.return_value = q
        result = scorer._calc_dividend_score(1, None, session)
        assert result is None

    def test_zero_yield_when_no_recent_dividends(self):
        """Dividends outside 12-month window give 0 yield, but consistency may still score."""
        scorer = _make_scorer()

        today = date.today()
        old_action = _dividend_action(today - timedelta(days=400), dps=2.0)  # >1 year ago

        session = MagicMock()
        div_q = MagicMock()
        div_q.filter.return_value.order_by.return_value.all.return_value = [old_action]
        price_q = MagicMock()
        price_q.filter.return_value.order_by.return_value.first.return_value = (100.0,)

        def query_side(model_class):
            from bist_picker.db.schema import CorporateAction, DailyPrice
            if model_class is CorporateAction:
                return div_q
            if model_class is DailyPrice:
                return price_q
            return MagicMock()

        session.query.side_effect = query_side
        result = scorer._calc_dividend_score(1, None, session)
        # Yield = 0 (no dividends in last 12m), consistency = 1 year out of 5 = 20
        # score = 0.6*0 + 0.4*20 = 8.0
        assert result == pytest.approx(8.0, abs=1.0)

    def test_yield_and_consistency_calculation(self):
        """Full path: 8% yield + 5/5 year consistency."""
        scorer = _make_scorer()

        today = date.today()
        # 5 consecutive years of dividends; this year and 4 prior
        actions = []
        for yr_offset in range(5):
            action_date = today.replace(year=today.year - yr_offset) - timedelta(days=30)
            actions.append(_dividend_action(action_date, dps=8.0))

        session = MagicMock()
        div_q = MagicMock()
        div_q.filter.return_value.order_by.return_value.all.return_value = actions
        price_q = MagicMock()
        price_q.filter.return_value.order_by.return_value.first.return_value = (100.0,)

        # NOTE: _calc_dividend_score calls session.query(DailyPrice.close) — a column
        # attribute, not the class — so dispatch by exclusion, not identity.
        def query_side(model_class):
            from bist_picker.db.schema import CorporateAction
            if model_class is CorporateAction:
                return div_q
            return price_q  # DailyPrice.close column attribute arrives here

        session.query.side_effect = query_side
        result = scorer._calc_dividend_score(1, None, session)
        # Last 12 months: only the most recent action (yr=0) is within 365 days
        # yield = 8/100 = 0.08 → yield_score = 100
        # consistency: this year + 4 prior = 5 years → consistency_score = 100
        # score = 0.6 * 100 + 0.4 * 100 = 100
        assert result == pytest.approx(100.0, abs=2.0)

    def test_consistency_counts_distinct_years(self):
        """Multiple dividends in same year count as one year for consistency."""
        scorer = _make_scorer()

        today = date.today()
        # Two dividends in the same year, nothing in prior years
        # Use small deltas to ensure we don't cross year boundary unless it's Jan 1
        d1 = today - timedelta(days=2)
        d2 = today - timedelta(days=5)
        
        # If today is very early in year, these might still cross if we are unlucky (e.g. Jan 2)
        # So let's force them to be the same year explicitly
        current_year = today.year
        d1 = date(current_year, 6, 1)
        d2 = date(current_year, 8, 1)

        actions = [
            _dividend_action(d1, dps=3.0),
            _dividend_action(d2, dps=3.0),
        ]

        session = MagicMock()
        div_q = MagicMock()
        div_q.filter.return_value.order_by.return_value.all.return_value = actions
        price_q = MagicMock()
        price_q.filter.return_value.order_by.return_value.first.return_value = (100.0,)

        # DailyPrice.close is a column attribute — dispatch by exclusion.
        def query_side(model_class):
            from bist_picker.db.schema import CorporateAction
            if model_class is CorporateAction:
                return div_q
            return price_q

        session.query.side_effect = query_side
        result = scorer._calc_dividend_score(1, None, session)
        # yield = (3+3)/100 = 6% → yield_score = 75
        # consistency: only 1 distinct year → 1/5 = 20%
        # score = 0.6*75 + 0.4*20 = 45 + 8 = 53
        assert result == pytest.approx(53.0, abs=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# _calc_governance tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcGovernance:
    def test_uses_manual_score_from_config(self):
        """governance_score from subsidiaries.yaml takes priority."""
        scorer = _make_scorer()
        session = MagicMock()
        result = scorer._calc_governance(1, "SAHOL", session)
        assert result == pytest.approx(60.0)

    def test_uses_related_party_pct_when_no_manual(self):
        """Derive governance from related_party_revenue_pct when no config score."""
        scorer = _make_scorer(subsidiaries={
            "NOTHOL": {"name": "No Governance", "subsidiaries": []}
            # no governance_score key
        })

        metric = MagicMock()
        metric.related_party_revenue_pct = 25.0  # 25% → score = (1 - 0.25/0.5) * 100 = 50

        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.order_by.return_value.first.return_value = metric
        session.query.return_value = q

        result = scorer._calc_governance(1, "NOTHOL", session)
        assert result == pytest.approx(50.0)

    def test_returns_50_when_no_data(self):
        """Neutral default 50.0 when no config score and no related-party data."""
        scorer = _make_scorer(subsidiaries={
            "NOTHOL": {"name": "No Data", "subsidiaries": []}
        })

        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.order_by.return_value.first.return_value = None
        session.query.return_value = q

        result = scorer._calc_governance(1, "NOTHOL", session)
        assert result == pytest.approx(50.0)

    def test_returns_50_for_unknown_ticker(self):
        """Unknown ticker (not in config) falls through to default 50."""
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.order_by.return_value.first.return_value = None
        session.query.return_value = q
        result = scorer._calc_governance(1, "UNKNOWN", session)
        assert result == pytest.approx(50.0)

    def test_zero_related_party_gives_100(self):
        scorer = _make_scorer(subsidiaries={
            "NOTHOL": {"name": "Clean", "subsidiaries": []}
        })
        metric = MagicMock()
        metric.related_party_revenue_pct = 0.0
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.order_by.return_value.first.return_value = metric
        session.query.return_value = q
        result = scorer._calc_governance(1, "NOTHOL", session)
        assert result == pytest.approx(100.0)

    def test_50pct_related_party_gives_0(self):
        scorer = _make_scorer(subsidiaries={
            "NOTHOL": {"name": "Opaque", "subsidiaries": []}
        })
        metric = MagicMock()
        metric.related_party_revenue_pct = 50.0
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.order_by.return_value.first.return_value = metric
        session.query.return_value = q
        result = scorer._calc_governance(1, "NOTHOL", session)
        assert result == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# _compute_composite tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeComposite:
    def test_all_factors_present(self):
        scorer = _make_scorer()
        scores = {
            "nav_discount_score":      80.0,
            "portfolio_quality_score": 60.0,
            "dividend_yield_score":    70.0,
            "governance_score":        50.0,
        }
        expected = (
            80.0 * 0.35
            + 60.0 * 0.25
            + 70.0 * 0.20
            + 50.0 * 0.20
        )
        result = scorer._compute_composite(scores)
        assert result == pytest.approx(expected)

    def test_weight_redistribution_with_one_factor(self):
        """Single available factor should receive weight 1.0 effectively."""
        scorer = _make_scorer()
        scores = {"governance_score": 80.0}
        result = scorer._compute_composite(scores)
        assert result == pytest.approx(80.0)

    def test_returns_none_when_all_missing(self):
        scorer = _make_scorer()
        result = scorer._compute_composite({})
        assert result is None

    def test_raw_keys_without_score_suffix_ignored(self):
        scorer = _make_scorer()
        scores = {
            "nav_discount_score": 70.0,
            "nav_ratio": 0.6,        # raw — not a sub-score
            "market_cap": 1e9,       # raw — not a sub-score
        }
        # Only nav_discount_score contributes → 70.0 (redistributed to full weight)
        result = scorer._compute_composite(scores)
        assert result == pytest.approx(70.0)

    def test_partial_factors_redistribution(self):
        """Missing portfolio_quality (0.25) and dividend_yield (0.20) → remaining sum=0.55."""
        scorer = _make_scorer()
        scores = {
            "nav_discount_score": 100.0,  # weight 0.35
            "governance_score":    60.0,  # weight 0.20
        }
        # total_weight = 0.35 + 0.20 = 0.55
        # composite = (100*0.35 + 60*0.20) / 0.55 = (35 + 12) / 0.55 = 47/0.55 ≈ 85.45
        expected = (100.0 * 0.35 + 60.0 * 0.20) / 0.55
        result = scorer._compute_composite(scores)
        assert result == pytest.approx(expected, rel=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# score_all() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreAll:
    def test_empty_universe(self):
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.all.return_value = []
        session.query.return_value = q
        assert scorer.score_all(session) == {}

    def test_skips_companies_where_score_returns_none(self):
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.all.return_value = [(1,), (2,)]
        session.query.return_value = q

        with patch.object(scorer, "score", return_value=None):
            results = scorer.score_all(session)
        assert results == {}

    def test_nav_discount_inverted_ranking(self):
        """Lower nav_ratio (deeper discount) = higher nav_discount_score."""
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.all.return_value = [(1,), (2,), (3,)]
        session.query.return_value = q

        raw_map = {
            1: {"nav_ratio": 0.4, "portfolio_quality": None,
                "dividend_score": None, "governance": 50.0,
                "market_cap": 1e9, "equity": 10_000.0},
            2: {"nav_ratio": 0.7, "portfolio_quality": None,
                "dividend_score": None, "governance": 50.0,
                "market_cap": 1e9, "equity": 10_000.0},
            3: {"nav_ratio": 1.0, "portfolio_quality": None,
                "dividend_score": None, "governance": 50.0,
                "market_cap": 1e9, "equity": 10_000.0},
        }

        with patch.object(scorer, "score", side_effect=lambda cid, sess: raw_map.get(cid)):
            results = scorer.score_all(session)

        assert results[1]["nav_discount_score"] == pytest.approx(100.0)  # deepest discount
        assert results[3]["nav_discount_score"] == pytest.approx(0.0)    # at NAV

    def test_portfolio_quality_normal_ranking(self):
        """Higher portfolio_quality raw value = higher score (higher=better)."""
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.all.return_value = [(1,), (2,)]
        session.query.return_value = q

        raw_map = {
            1: {"nav_ratio": None, "portfolio_quality": 30.0,
                "dividend_score": None, "governance": 50.0,
                "market_cap": None, "equity": None},
            2: {"nav_ratio": None, "portfolio_quality": 70.0,
                "dividend_score": None, "governance": 50.0,
                "market_cap": None, "equity": None},
        }

        with patch.object(scorer, "score", side_effect=lambda cid, sess: raw_map.get(cid)):
            results = scorer.score_all(session)

        assert results[2]["portfolio_quality_score"] == pytest.approx(100.0)
        assert results[1]["portfolio_quality_score"] == pytest.approx(0.0)

    def test_holding_composite_present_in_output(self):
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.all.return_value = [(1,)]
        session.query.return_value = q

        raw_map = {
            1: {
                "nav_ratio":         0.5,
                "portfolio_quality": 60.0,
                "dividend_score":    40.0,
                "governance":        55.0,
                "market_cap":        5e9,
                "equity":            20_000.0,
            }
        }

        with patch.object(scorer, "score", side_effect=lambda cid, sess: raw_map.get(cid)):
            results = scorer.score_all(session)

        assert "holding_composite" in results[1]
        composite = results[1]["holding_composite"]
        assert composite is not None
        assert 0.0 <= composite <= 100.0

    def test_none_nav_ratio_receives_none_score(self):
        """Company with None nav_ratio should get None for nav_discount_score."""
        scorer = _make_scorer()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.all.return_value = [(1,), (2,)]
        session.query.return_value = q

        raw_map = {
            1: {"nav_ratio": None, "portfolio_quality": None,
                "dividend_score": None, "governance": 50.0,
                "market_cap": None, "equity": None},
            2: {"nav_ratio": 0.6, "portfolio_quality": None,
                "dividend_score": None, "governance": 70.0,
                "market_cap": 5e9, "equity": 10_000.0},
        }

        with patch.object(scorer, "score", side_effect=lambda cid, sess: raw_map.get(cid)):
            results = scorer.score_all(session)

        assert results[1].get("nav_discount_score") is None
        assert results[2]["nav_discount_score"] == pytest.approx(50.0)  # single valid entry
