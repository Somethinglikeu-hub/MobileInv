"""Tests for classification/risk_classifier.py.

Tests RiskClassifier with synthetic in-memory data covering:
- THYAO profile (large cap, decent liquidity) -> LOW or MEDIUM
- Small-cap illiquid stock -> HIGH
- Beta calculation against a manual benchmark
- Each individual dimension scorer (unit)
- Missing dimension weight redistribution
- classify_all() DB integration
"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bist_picker.classification.risk_classifier import RiskClassifier
from bist_picker.db.schema import Base, Company, DailyPrice, FinancialStatement, ScoringResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    Session_ = sessionmaker(bind=engine)
    sess = Session_()
    yield sess
    sess.close()


@pytest.fixture
def clf() -> RiskClassifier:
    return RiskClassifier()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_company(
    session: Session,
    ticker: str,
    company_type: str = "OPERATING",
    is_active: bool = True,
) -> Company:
    c = Company(ticker=ticker, name=ticker, company_type=company_type, is_active=is_active)
    session.add(c)
    session.flush()
    return c


def _add_prices(
    session: Session,
    company_id: int,
    prices: list[float],
    start_date: date = date(2025, 1, 1),
    volume: int = 1_000_000,
    source: str = "YAHOO_TEST",
) -> None:
    """Add daily price rows (one per trading day) with given close prices."""
    d = start_date
    for price in prices:
        session.add(DailyPrice(
            company_id=company_id,
            date=d,
            close=price,
            volume=volume,
            source=source,
        ))
        d += timedelta(days=1)
    session.flush()


def _add_balance_sheet(
    session: Session,
    company_id: int,
    total_assets: float,
    equity: float,
    share_capital: float,
    period_end: date = date(2024, 12, 31),
) -> None:
    data = [
        {"item_code": "1BL", "desc_tr": "TOPLAM VARLIKLAR", "value": total_assets},
        {"item_code": "2N", "desc_tr": "OZKAYNAKLAR", "value": equity},
        {"item_code": "2OA", "desc_tr": "SERMAYE", "value": share_capital},
    ]
    session.add(FinancialStatement(
        company_id=company_id,
        period_end=period_end,
        period_type="ANNUAL",
        statement_type="BALANCE",
        data_json=json.dumps(data),
    ))
    session.flush()


def _make_low_vol_prices(n: int = 252, seed: int = 0) -> list[float]:
    """Generate a price series with ~12% annualised volatility (LOW risk)."""
    rng = np.random.default_rng(seed)
    # daily std = 0.12 / sqrt(252) ~ 0.0076
    daily_ret = rng.normal(0.0005, 0.0076, n)
    prices = 100.0 * np.exp(np.cumsum(daily_ret))
    return prices.tolist()


def _make_high_vol_prices(n: int = 252, seed: int = 1) -> list[float]:
    """Generate a price series with ~50% annualised volatility (HIGH risk)."""
    rng = np.random.default_rng(seed)
    # daily std = 0.50 / sqrt(252) ~ 0.0315
    daily_ret = rng.normal(0.0003, 0.0315, n)
    prices = 100.0 * np.exp(np.cumsum(daily_ret))
    return prices.tolist()


# ---------------------------------------------------------------------------
# Unit tests: dimension scorers
# ---------------------------------------------------------------------------


class TestDimensionScorers:
    """Unit tests for each individual dimension scoring method."""

    def test_volatility_low(self, clf):
        assert clf._score_volatility(0.10) == 1

    def test_volatility_medium(self, clf):
        assert clf._score_volatility(0.27) == 2

    def test_volatility_high(self, clf):
        assert clf._score_volatility(0.45) == 3

    def test_volatility_boundary_low_max(self, clf):
        # Exactly at low_max (0.20) is LOW per < check.
        assert clf._score_volatility(0.20) == 2  # not < 0.20, so MEDIUM

    def test_volatility_boundary_high_min(self, clf):
        # Exactly at high_min (0.35) is not HIGH (needs > 0.35).
        assert clf._score_volatility(0.35) == 2

    def test_beta_low(self, clf):
        assert clf._score_beta(0.5) == 1

    def test_beta_medium(self, clf):
        assert clf._score_beta(1.0) == 2

    def test_beta_high(self, clf):
        assert clf._score_beta(1.5) == 3

    def test_market_cap_large_is_low_risk(self, clf):
        """Large cap (>10B TRY) = LOW risk = score 1."""
        assert clf._score_market_cap(15_000_000_000) == 1

    def test_market_cap_mid_is_medium_risk(self, clf):
        assert clf._score_market_cap(5_000_000_000) == 2

    def test_market_cap_small_is_high_risk(self, clf):
        """Small cap (<2B TRY) = HIGH risk = score 3."""
        assert clf._score_market_cap(500_000_000) == 3

    def test_liquidity_high_is_low_risk(self, clf):
        """High liquidity (>25M TRY/day) = LOW risk = score 1."""
        assert clf._score_liquidity(30_000_000) == 1

    def test_liquidity_medium(self, clf):
        assert clf._score_liquidity(10_000_000) == 2

    def test_liquidity_low_is_high_risk(self, clf):
        """Low liquidity (<5M TRY/day) = HIGH risk = score 3."""
        assert clf._score_liquidity(1_000_000) == 3

    def test_leverage_low(self, clf):
        assert clf._score_leverage(0.3) == 1

    def test_leverage_medium(self, clf):
        assert clf._score_leverage(1.0) == 2

    def test_leverage_high(self, clf):
        assert clf._score_leverage(2.0) == 3


# ---------------------------------------------------------------------------
# Unit tests: beta calculation
# ---------------------------------------------------------------------------


class TestBetaCalculation:
    """Tests for _compute_beta() with known synthetic data."""

    def test_beta_of_one_for_identical_series(self, clf, session):
        """A stock with returns identical to the benchmark has beta = 1.0."""
        # Create benchmark (XU100).
        bm = _add_company(session, "XU100")
        stock = _add_company(session, "TSTOCK")

        prices = _make_low_vol_prices(n=253, seed=42)
        _add_prices(session, bm.id, prices)
        _add_prices(session, stock.id, prices)  # same series

        beta = clf._compute_beta(stock.id, session)

        assert beta is not None
        assert beta == pytest.approx(1.0, abs=1e-6)

    def test_beta_of_zero_for_uncorrelated_series(self, clf, session):
        """A stock uncorrelated with benchmark should have beta near 0."""
        rng = np.random.default_rng(99)
        bm = _add_company(session, "XU100")
        stock = _add_company(session, "UNCORR")

        bm_prices = _make_low_vol_prices(n=253, seed=10)
        # Independent series (uncorrelated with BM)
        stock_ret = rng.normal(0, 0.01, 252)
        stock_prices = (100.0 * np.exp(np.cumsum(stock_ret))).tolist()

        _add_prices(session, bm.id, bm_prices)
        _add_prices(session, stock.id, stock_prices)

        beta = clf._compute_beta(stock.id, session)

        # Beta should be near 0 for uncorrelated series (not exactly 0 due to noise).
        assert beta is not None
        assert abs(beta) < 0.4, f"Expected near-zero beta, got {beta:.3f}"

    def test_beta_none_without_benchmark(self, clf, session):
        """Beta returns None when no XU100 company exists in DB."""
        stock = _add_company(session, "THYAO")
        _add_prices(session, stock.id, _make_low_vol_prices())
        # No XU100 in DB
        beta = clf._compute_beta(stock.id, session)
        assert beta is None

    def test_beta_none_with_insufficient_data(self, clf, session):
        """Beta returns None when fewer than 20 shared dates."""
        bm = _add_company(session, "XU100")
        stock = _add_company(session, "STKX")
        # Only 15 prices (below minimum)
        short_prices = _make_low_vol_prices(n=15)
        _add_prices(session, bm.id, short_prices)
        _add_prices(session, stock.id, short_prices)

        beta = clf._compute_beta(stock.id, session)
        assert beta is None

    def test_beta_manual_verification(self, clf, session):
        """Verify beta formula: cov(stock, mkt) / var(mkt)."""
        rng = np.random.default_rng(7)
        # Market returns
        mkt_ret = rng.normal(0, 0.01, 100)
        # Stock returns = 1.5 * mkt + small noise -> expected beta ~ 1.5
        noise = rng.normal(0, 0.002, 100)
        stock_ret = 1.5 * mkt_ret + noise

        mkt_prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(mkt_ret))]).tolist()
        stock_prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(stock_ret))]).tolist()

        bm = _add_company(session, "XU100")
        stock = _add_company(session, "HBETA")
        _add_prices(session, bm.id, mkt_prices)
        _add_prices(session, stock.id, stock_prices)

        beta = clf._compute_beta(stock.id, session)

        assert beta is not None
        # With beta=1.5 construction, should be within reasonable range.
        assert 1.2 <= beta <= 1.8, f"Expected beta ~1.5, got {beta:.3f}"


# ---------------------------------------------------------------------------
# Integration tests: classify() happy paths
# ---------------------------------------------------------------------------


class TestClassify:
    """Tests for RiskClassifier.classify() with full synthetic profiles."""

    def _build_thyao_profile(self, session: Session) -> int:
        """Large-cap, liquid, low-leverage stock profile (like THYAO)."""
        c = _add_company(session, "THYAO")
        # Low volatility prices (~12% annual vol)
        prices = _make_low_vol_prices(n=253, seed=5)
        _add_prices(session, c.id, prices, volume=10_000_000)  # 10M shares -> ~1B/day TRY
        # Large cap: price ~100 TRY, 2B shares outstanding = 200B TRY mcap
        # => stock price * shares = 100 * 2B = 200B > 10B threshold
        _add_balance_sheet(
            session, c.id,
            total_assets=500_000_000_000,   # 500B TRY
            equity=250_000_000_000,          # 250B TRY -> D/E = 1.0 (MEDIUM)
            share_capital=2_000_000_000,     # 2B shares
        )
        return c.id

    def _build_small_illiquid_profile(self, session: Session) -> int:
        """Small-cap, illiquid, high-leverage stock (HIGH risk profile)."""
        c = _add_company(session, "XSML")
        # High volatility prices (~55% annual vol)
        prices = _make_high_vol_prices(n=253, seed=2)
        _add_prices(session, c.id, prices, volume=10_000)  # 10K shares -> tiny volume
        # Small cap: price ~100 TRY, 5M shares -> 500M TRY < 2B threshold
        _add_balance_sheet(
            session, c.id,
            total_assets=1_000_000_000,   # 1B TRY
            equity=300_000_000,            # 300M TRY -> D/E ~ 2.33 (HIGH)
            share_capital=5_000_000,       # 5M shares
        )
        return c.id

    def test_thyao_profile_is_low_or_medium(self, clf, session):
        """Large-cap liquid stock should classify as LOW or MEDIUM (not HIGH)."""
        cid = self._build_thyao_profile(session)
        tier = clf.classify(cid, session)
        assert tier in ("LOW", "MEDIUM"), f"Expected LOW or MEDIUM for THYAO profile, got {tier}"

    def test_small_illiquid_profile_is_high(self, clf, session):
        """Small-cap illiquid high-leverage stock should classify as HIGH."""
        cid = self._build_small_illiquid_profile(session)
        tier = clf.classify(cid, session)
        assert tier == "HIGH", f"Expected HIGH for small illiquid profile, got {tier}"

    def test_classify_returns_valid_tier(self, clf, session):
        """classify() always returns one of HIGH/MEDIUM/LOW."""
        cid = self._build_thyao_profile(session)
        tier = clf.classify(cid, session)
        assert tier in ("HIGH", "MEDIUM", "LOW")

    def test_no_price_data_defaults_to_medium(self, clf, session):
        """Company with no price or balance data defaults to MEDIUM."""
        c = _add_company(session, "XNEW")
        tier = clf.classify(c.id, session)
        assert tier == "MEDIUM"

    def test_nonexistent_company_defaults_to_medium(self, clf, session):
        """Company ID not in DB defaults to MEDIUM (all dims None)."""
        tier = clf.classify(99999, session)
        assert tier == "MEDIUM"

    def test_compute_liquidity_uses_isyatirim_turnover_directly(self, clf, session):
        """IsYatirim rows already carry TRY turnover and must not be multiplied by price."""
        company = _add_company(session, "TURN1")
        session.add_all(
            [
                DailyPrice(
                    company_id=company.id,
                    date=date(2026, 1, 1),
                    close=100.0,
                    volume=12_000_000,
                    source="ISYATIRIM",
                ),
                DailyPrice(
                    company_id=company.id,
                    date=date(2026, 1, 2),
                    close=120.0,
                    volume=18_000_000,
                    source="ISYATIRIM",
                ),
            ]
        )
        session.commit()

        liquidity = clf._compute_liquidity(company.id, session)

        assert liquidity == pytest.approx(15_000_000.0)


# ---------------------------------------------------------------------------
# Integration tests: classify_all()
# ---------------------------------------------------------------------------


class TestClassifyAll:
    """Tests for RiskClassifier.classify_all() DB integration."""

    def test_risk_tier_stored_in_new_scoring_row(self, clf, session):
        """classify_all creates a scoring_results row when none exists."""
        today = date(2026, 2, 1)
        c = _add_company(session, "ASELS")
        session.commit()

        stats = clf.classify_all(session, scoring_date=today)

        row = session.query(ScoringResult).filter_by(company_id=c.id).first()
        assert row is not None
        assert row.risk_tier in ("HIGH", "MEDIUM", "LOW")
        assert row.scoring_date == today

    def test_risk_tier_updates_existing_scoring_row(self, clf, session):
        """classify_all updates risk_tier on an existing scoring_results row."""
        today = date(2026, 2, 1)
        c = _add_company(session, "GARAN")
        existing = ScoringResult(
            company_id=c.id,
            scoring_date=today,
            model_used="BANK",
            buffett_score=70.0,
        )
        session.add(existing)
        session.commit()

        clf.classify_all(session, scoring_date=today)

        row = session.query(ScoringResult).filter_by(company_id=c.id).first()
        assert row.risk_tier is not None
        assert row.buffett_score == pytest.approx(70.0)  # untouched

    def test_stats_dict_returned(self, clf, session):
        """classify_all returns a stats dict with total and by_tier."""
        today = date(2026, 2, 1)
        for ticker in ("BIMAS", "THYAO", "SAHOL"):
            _add_company(session, ticker)
        session.commit()

        stats = clf.classify_all(session, scoring_date=today)

        assert stats["total"] == 3
        assert "by_tier" in stats
        assert sum(stats["by_tier"].values()) == 3

    def test_inactive_companies_skipped(self, clf, session):
        """classify_all ignores companies with is_active=False."""
        today = date(2026, 2, 1)
        active = _add_company(session, "TCELL", is_active=True)
        _add_company(session, "DELIST", is_active=False)
        session.commit()

        stats = clf.classify_all(session, scoring_date=today)

        assert stats["total"] == 1
        rows = session.query(ScoringResult).filter_by(scoring_date=today).all()
        assert len(rows) == 1
        assert rows[0].company_id == active.id

    def test_defaults_to_today(self, clf, session):
        """classify_all with no scoring_date uses today's date."""
        from datetime import date as date_cls
        today = date_cls.today()
        c = _add_company(session, "KCHOL")
        session.commit()

        clf.classify_all(session)

        row = session.query(ScoringResult).filter_by(company_id=c.id).first()
        assert row is not None
        assert row.scoring_date == today
