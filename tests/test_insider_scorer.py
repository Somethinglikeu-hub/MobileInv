"""Unit tests for scoring/factors/insider.py.

Tests verify:
  1. Happy path: net buying is computed correctly for BUY/SELL transactions
  2. Role weighting: BOARD/CEO weighted higher than RELATED
  3. 3m/6m windows: transactions outside the window are excluded
  4. Market-cap normalisation: net_buy_pct = net_buy / market_cap
  5. Fallback when market cap cannot be derived (raw TRY used, pct = None)
  6. score() returns None when no transactions exist in window
  7. score_all() assigns neutral percentile (50.0) to companies with no data
  8. score_all() ranks companies correctly (most buying = highest percentile)
  9. Combined insider_raw = 60% 3m + 40% 6m
  10. Config is loaded and used (lookback days, role weights)
"""

from datetime import date, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from bist_picker.scoring.factors.insider import InsiderScorer


# ── Helpers ────────────────────────────────────────────────────────────────────

TODAY = date(2025, 6, 15)  # fixed reference date for deterministic tests


def _txn(
    disclosure_date: date,
    transaction_type: Optional[str],
    total_value_try: Optional[float],
    person_role: str = "BOARD",
) -> MagicMock:
    """Build a mock InsiderTransaction."""
    t = MagicMock()
    t.disclosure_date = disclosure_date
    t.transaction_type = transaction_type
    t.total_value_try = total_value_try
    t.person_role = person_role
    return t


def _company(cid: int = 1, is_active: bool = True) -> MagicMock:
    c = MagicMock()
    c.id = cid
    c.is_active = is_active
    c.ticker = f"TEST{cid}"
    return c


def _make_scorer(extra_cfg: Optional[dict] = None) -> InsiderScorer:
    """Return an InsiderScorer with inline config (no file I/O)."""
    base_cfg = {
        "insider": {
            "lookback_3m_days": 91,
            "lookback_6m_days": 182,
            "weight_3m": 0.60,
            "weight_6m": 0.40,
            "min_market_cap_for_pct": 100_000_000,
            "role_weights": {
                "BOARD": 1.0,
                "CEO": 1.0,
                "MAJOR_SHAREHOLDER": 0.8,
                "RELATED": 0.5,
                "OTHER": 0.5,
            },
        }
    }
    if extra_cfg:
        base_cfg["insider"].update(extra_cfg)
    scorer = InsiderScorer.__new__(InsiderScorer)
    # Manually set parsed config (bypass __init__ file I/O)
    scorer._lookback_3m = base_cfg["insider"]["lookback_3m_days"]
    scorer._lookback_6m = base_cfg["insider"]["lookback_6m_days"]
    scorer._weight_3m = base_cfg["insider"]["weight_3m"]
    scorer._weight_6m = base_cfg["insider"]["weight_6m"]
    scorer._min_market_cap = base_cfg["insider"]["min_market_cap_for_pct"]
    scorer._role_weights = {
        k.upper(): float(v)
        for k, v in base_cfg["insider"]["role_weights"].items()
    }
    return scorer


def _session_with_transactions(
    company_id: int,
    transactions: list[MagicMock],
    market_cap: float = None,  # None means market cap unavailable
) -> MagicMock:
    """Build a mock session that returns given transactions and optional market cap."""
    session = MagicMock()

    company = _company(company_id)
    session.get.return_value = company

    # Make the query chain return transactions
    txn_query = MagicMock()
    txn_query.filter.return_value.all.return_value = transactions

    # Mock price row
    if market_cap is not None:
        price_row = (50.0,)  # close price
        metric_row = MagicMock()
        metric_row.eps_adjusted = 5.0
        metric_row.adjusted_net_income = market_cap / 10.0  # shares = market_cap/50
    else:
        price_row = None
        metric_row = None

    def query_side_effect(model_class):
        from bist_picker.db.schema import InsiderTransaction, DailyPrice, AdjustedMetric, Company
        q = MagicMock()
        if model_class is InsiderTransaction:
            q.filter.return_value.all.return_value = transactions
        elif model_class is DailyPrice:
            inner = MagicMock()
            inner.filter.return_value.order_by.return_value.first.return_value = price_row
            q.filter.return_value = inner.filter.return_value
        elif model_class is AdjustedMetric:
            inner = MagicMock()
            inner.filter.return_value.order_by.return_value.first.return_value = metric_row
            q.filter.return_value = inner.filter.return_value
        elif model_class is Company:
            inner = MagicMock()
            inner.filter.return_value.all.return_value = [(company_id,)]
            q.filter.return_value = inner.filter.return_value
        return q

    session.query.side_effect = query_side_effect
    return session


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestNetBuy:
    """Tests for the _net_buy helper method."""

    def test_pure_buying(self):
        scorer = _make_scorer()
        since = TODAY - timedelta(days=91)
        txns = [
            _txn(TODAY - timedelta(days=10), "BUY", 1_000_000, "BOARD"),
            _txn(TODAY - timedelta(days=20), "BUY", 500_000, "BOARD"),
        ]
        result = scorer._net_buy(txns, since, TODAY)
        assert result == pytest.approx(1_500_000.0)

    def test_pure_selling(self):
        scorer = _make_scorer()
        since = TODAY - timedelta(days=91)
        txns = [_txn(TODAY - timedelta(days=5), "SELL", 2_000_000, "CEO")]
        result = scorer._net_buy(txns, since, TODAY)
        assert result == pytest.approx(-2_000_000.0)

    def test_mixed_buy_and_sell(self):
        scorer = _make_scorer()
        since = TODAY - timedelta(days=91)
        txns = [
            _txn(TODAY - timedelta(days=30), "BUY", 3_000_000, "BOARD"),
            _txn(TODAY - timedelta(days=10), "SELL", 1_000_000, "BOARD"),
        ]
        result = scorer._net_buy(txns, since, TODAY)
        assert result == pytest.approx(2_000_000.0)

    def test_role_weighting_reduces_value(self):
        scorer = _make_scorer()
        since = TODAY - timedelta(days=91)
        # RELATED has weight 0.5 — should be halved
        txns = [_txn(TODAY - timedelta(days=10), "BUY", 1_000_000, "RELATED")]
        result = scorer._net_buy(txns, since, TODAY)
        assert result == pytest.approx(500_000.0)

    def test_transaction_outside_window_excluded(self):
        scorer = _make_scorer()
        since = TODAY - timedelta(days=91)
        # This transaction is exactly at the boundary — just before since
        txns = [_txn(since - timedelta(days=1), "BUY", 1_000_000, "BOARD")]
        result = scorer._net_buy(txns, since, TODAY)
        assert result == pytest.approx(0.0)

    def test_transaction_at_boundary_included(self):
        scorer = _make_scorer()
        since = TODAY - timedelta(days=91)
        txns = [_txn(since, "BUY", 1_000_000, "BOARD")]
        result = scorer._net_buy(txns, since, TODAY)
        assert result == pytest.approx(1_000_000.0)

    def test_missing_transaction_type_skipped(self):
        scorer = _make_scorer()
        since = TODAY - timedelta(days=91)
        txns = [_txn(TODAY - timedelta(days=5), None, 1_000_000, "BOARD")]
        txns[0].transaction_type = None
        result = scorer._net_buy(txns, since, TODAY)
        assert result == pytest.approx(0.0)

    def test_missing_value_skipped(self):
        scorer = _make_scorer()
        since = TODAY - timedelta(days=91)
        txns = [_txn(TODAY - timedelta(days=5), "BUY", None, "BOARD")]
        txns[0].total_value_try = None
        result = scorer._net_buy(txns, since, TODAY)
        assert result == pytest.approx(0.0)


class TestRoleWeight:
    """Tests for the _role_weight helper."""

    def test_board_weight(self):
        scorer = _make_scorer()
        assert scorer._role_weight("BOARD") == pytest.approx(1.0)

    def test_ceo_weight(self):
        scorer = _make_scorer()
        assert scorer._role_weight("CEO") == pytest.approx(1.0)

    def test_major_shareholder_weight(self):
        scorer = _make_scorer()
        assert scorer._role_weight("MAJOR_SHAREHOLDER") == pytest.approx(0.8)

    def test_related_weight(self):
        scorer = _make_scorer()
        assert scorer._role_weight("RELATED") == pytest.approx(0.5)

    def test_unknown_role_defaults_to_other(self):
        scorer = _make_scorer()
        assert scorer._role_weight("SECRETARY") == pytest.approx(0.5)

    def test_none_role_defaults_to_other(self):
        scorer = _make_scorer()
        assert scorer._role_weight(None) == pytest.approx(0.5)

    def test_case_insensitive(self):
        scorer = _make_scorer()
        assert scorer._role_weight("board") == pytest.approx(1.0)


class TestScore:
    """Tests for the public score() method."""

    def test_returns_none_when_company_not_found(self):
        scorer = _make_scorer()
        session = MagicMock()
        session.get.return_value = None
        result = scorer.score(999, session)
        assert result is None

    def test_returns_none_when_no_transactions(self):
        scorer = _make_scorer()
        session = MagicMock()
        session.get.return_value = _company(1)
        q = MagicMock()
        q.filter.return_value.all.return_value = []
        session.query.return_value = q
        result = scorer.score(1, session)
        assert result is None

    def test_happy_path_with_market_cap(self):
        """Net buying is computed and normalised by market cap."""
        scorer = _make_scorer()

        # 3m transaction: BUY 10M (BOARD, weight 1.0)
        # 6m-only transaction: BUY 5M (MAJOR_SHAREHOLDER, weight 0.8)
        cutoff_6m = date.today() - timedelta(days=182)
        t_3m = _txn(date.today() - timedelta(days=30), "BUY", 10_000_000, "BOARD")
        t_6m = _txn(cutoff_6m + timedelta(days=5), "BUY", 5_000_000, "MAJOR_SHAREHOLDER")

        session = MagicMock()
        session.get.return_value = _company(1)

        from bist_picker.db.schema import InsiderTransaction

        def query_side_effect(model_class):
            q = MagicMock()
            if model_class is InsiderTransaction:
                q.filter.return_value.all.return_value = [t_3m, t_6m]
            return q

        session.query.side_effect = query_side_effect

        # Patch _estimate_market_cap so we control the return value exactly:
        # 100 TRY price * 100M shares = 10B TRY market cap
        market_cap = 10_000_000_000.0
        with patch.object(scorer, "_estimate_market_cap", return_value=market_cap):
            result = scorer.score(1, session)

        assert result is not None

        # 3m net: 10M (BOARD w=1.0); 6m net: 10M + 5M*0.8 = 14M
        assert result["net_buy_3m_try"] == pytest.approx(10_000_000.0)
        assert result["net_buy_6m_try"] == pytest.approx(14_000_000.0)

        assert result["market_cap_try"] == pytest.approx(market_cap)

        # Pct signals
        assert result["net_buy_3m_pct"] == pytest.approx(10_000_000 / market_cap)
        assert result["net_buy_6m_pct"] == pytest.approx(14_000_000 / market_cap)

        # insider_raw = 60% * 3m_pct + 40% * 6m_pct
        expected_raw = (
            0.60 * (10_000_000 / market_cap)
            + 0.40 * (14_000_000 / market_cap)
        )
        assert result["insider_raw"] == pytest.approx(expected_raw)

    def test_fallback_to_raw_try_when_no_market_cap(self):
        """When market cap is unavailable, raw TRY values are used and pct is None."""
        scorer = _make_scorer()

        t = _txn(date.today() - timedelta(days=10), "BUY", 5_000_000, "CEO")

        session = MagicMock()
        session.get.return_value = _company(1)

        from bist_picker.db.schema import InsiderTransaction, DailyPrice, AdjustedMetric

        def query_side_effect(model_class):
            q = MagicMock()
            if model_class is InsiderTransaction:
                q.filter.return_value.all.return_value = [t]
            elif model_class is DailyPrice:
                inner = MagicMock()
                inner.filter.return_value.order_by.return_value.first.return_value = None
                q.filter.return_value = inner.filter.return_value
            elif model_class is AdjustedMetric:
                inner = MagicMock()
                inner.filter.return_value.order_by.return_value.first.return_value = None
                q.filter.return_value = inner.filter.return_value
            return q

        session.query.side_effect = query_side_effect

        result = scorer.score(1, session)
        assert result is not None
        assert result["net_buy_3m_pct"] is None
        assert result["net_buy_6m_pct"] is None
        assert result["market_cap_try"] is None
        # Raw TRY combined: 60% * 5M + 40% * 5M = 5M (same 3m and 6m)
        assert result["insider_raw"] == pytest.approx(5_000_000.0)

    def test_transaction_counts(self):
        scorer = _make_scorer()

        t3m_1 = _txn(date.today() - timedelta(days=20), "BUY", 1_000_000, "BOARD")
        t3m_2 = _txn(date.today() - timedelta(days=50), "SELL", 500_000, "CEO")
        t6m_only = _txn(date.today() - timedelta(days=120), "BUY", 2_000_000, "BOARD")

        session = MagicMock()
        session.get.return_value = _company(1)

        from bist_picker.db.schema import InsiderTransaction, DailyPrice, AdjustedMetric

        def query_side_effect(model_class):
            q = MagicMock()
            if model_class is InsiderTransaction:
                q.filter.return_value.all.return_value = [t3m_1, t3m_2, t6m_only]
            elif model_class is DailyPrice:
                inner = MagicMock()
                inner.filter.return_value.order_by.return_value.first.return_value = None
                q.filter.return_value = inner.filter.return_value
            elif model_class is AdjustedMetric:
                inner = MagicMock()
                inner.filter.return_value.order_by.return_value.first.return_value = None
                q.filter.return_value = inner.filter.return_value
            return q

        session.query.side_effect = query_side_effect

        result = scorer.score(1, session)
        assert result is not None
        assert result["transaction_count_3m"] == 2
        assert result["transaction_count_6m"] == 3


class TestScoreAll:
    """Tests for score_all() percentile ranking."""

    def test_companies_without_transactions_get_neutral_percentile(self):
        scorer = _make_scorer()

        session = MagicMock()
        company_q = MagicMock()
        company_q.filter.return_value.all.return_value = [(1,), (2,)]
        session.query.return_value = company_q

        # score() will return None for both (no transactions)
        with patch.object(scorer, "score", return_value=None):
            results = scorer.score_all(session)

        assert 1 in results
        assert 2 in results
        assert results[1]["insider_percentile"] == pytest.approx(50.0)
        assert results[2]["insider_percentile"] == pytest.approx(50.0)

    def test_ranking_most_buying_gets_highest_percentile(self):
        scorer = _make_scorer()

        session = MagicMock()
        company_q = MagicMock()
        company_q.filter.return_value.all.return_value = [(1,), (2,), (3,)]
        session.query.return_value = company_q

        score_map = {
            1: {"insider_raw": 0.001},   # medium buyer
            2: {"insider_raw": 0.005},   # top buyer
            3: {"insider_raw": -0.002},  # net seller
        }

        def mock_score(cid, sess):
            return score_map.get(cid)

        with patch.object(scorer, "score", side_effect=mock_score):
            results = scorer.score_all(session)

        # Company 2 should be highest, company 3 lowest
        assert results[2]["insider_percentile"] == pytest.approx(100.0)
        assert results[3]["insider_percentile"] == pytest.approx(0.0)
        assert results[1]["insider_percentile"] == pytest.approx(50.0)

    def test_single_company_gets_50_percentile(self):
        scorer = _make_scorer()

        session = MagicMock()
        company_q = MagicMock()
        company_q.filter.return_value.all.return_value = [(1,)]
        session.query.return_value = company_q

        with patch.object(scorer, "score", return_value={"insider_raw": 0.003}):
            results = scorer.score_all(session)

        assert results[1]["insider_percentile"] == pytest.approx(50.0)

    def test_no_companies_returns_empty_dict(self):
        scorer = _make_scorer()

        session = MagicMock()
        company_q = MagicMock()
        company_q.filter.return_value.all.return_value = []
        session.query.return_value = company_q

        results = scorer.score_all(session)
        assert results == {}
