"""Tests for Buffett quality scoring factor.

Tests BuffettScorer with mock adjusted_metrics data for:
- Perfect Buffett stock (high ROE, stable margins, low debt) -> score > 80
- Bad stock (negative earnings, high debt) -> score < 30
- Bank company -> returns None
- Real data test with BIMAS (if DB available)
"""

import json
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bist_picker.db.schema import (
    AdjustedMetric,
    Base,
    Company,
    FinancialStatement,
)
from bist_picker.scoring.factors.buffett import BuffettScorer, _linear_scale, _linear_scale_inverse


# ---- Fixtures ----


@pytest.fixture
def engine():
    """Create in-memory SQLite database."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    """Create a database session."""
    Session_ = sessionmaker(bind=engine)
    sess = Session_()
    yield sess
    sess.close()


@pytest.fixture
def scorer():
    """Create a BuffettScorer instance."""
    return BuffettScorer()


def _make_income_json(gross_profit: float, net_sales: float) -> str:
    """Build minimal income statement JSON."""
    return json.dumps([
        {"item_code": "3C", "desc_tr": "Satis Gelirleri", "desc_eng": "Net Sales", "value": net_sales},
        {"item_code": "3D", "desc_tr": "BRUT KAR", "desc_eng": "GROSS PROFIT", "value": gross_profit},
    ])


def _make_balance_json(total_assets: float, total_equity: float) -> str:
    """Build minimal balance sheet JSON."""
    return json.dumps([
        {"item_code": "1BL", "desc_tr": "TOPLAM VARLIKLAR", "desc_eng": "TOTAL ASSETS", "value": total_assets},
        {"item_code": "2N", "desc_tr": "OZKAYNAKLAR", "desc_eng": "SHAREHOLDERS EQUITY", "value": total_equity},
    ])


def _add_perfect_stock(session: Session) -> int:
    """Add a perfect Buffett stock: high ROE, stable margins, low debt, positive earnings.

    Returns:
        Company ID.
    """
    company = Company(
        ticker="PERF",
        name="Perfect Quality A.S.",
        company_type="OPERATING",
        sector_bist="Sanayi",
        is_active=True,
    )
    session.add(company)
    session.flush()

    years = [2019, 2020, 2021, 2022, 2023]
    for i, year in enumerate(years):
        period_end = date(year, 12, 31)

        # Adjusted metrics: high ROE (~30%), positive EPS, positive FCF, growing OE
        session.add(AdjustedMetric(
            company_id=company.id,
            period_end=period_end,
            reported_net_income=50_000_000 + i * 5_000_000,
            monetary_gain_loss=0.0,
            adjusted_net_income=50_000_000 + i * 5_000_000,
            owner_earnings=40_000_000 + i * 6_000_000,
            free_cash_flow=35_000_000 + i * 4_000_000,
            roe_adjusted=0.28 + i * 0.01,    # 28-32%
            roa_adjusted=0.12 + i * 0.005,
            eps_adjusted=5.0 + i * 0.5,
            real_eps_growth_pct=0.08 if i > 0 else None,
        ))

        # Income: high gross margin (~40%)
        session.add(FinancialStatement(
            company_id=company.id,
            period_end=period_end,
            period_type="ANNUAL",
            statement_type="INCOME",
            data_json=_make_income_json(
                gross_profit=40_000_000 + i * 2_000_000,
                net_sales=100_000_000 + i * 5_000_000,
            ),
        ))

        # Balance: low D/E (~0.25)
        equity = 170_000_000 + i * 10_000_000
        total_assets = equity * 1.25  # D/E = 0.25
        session.add(FinancialStatement(
            company_id=company.id,
            period_end=period_end,
            period_type="ANNUAL",
            statement_type="BALANCE",
            data_json=_make_balance_json(
                total_assets=total_assets,
                total_equity=equity,
            ),
        ))

    session.flush()
    return company.id


def _add_bad_stock(session: Session) -> int:
    """Add a bad stock: negative earnings, high debt, volatile margins.

    Returns:
        Company ID.
    """
    company = Company(
        ticker="BADS",
        name="Bad Quality A.S.",
        company_type="OPERATING",
        sector_bist="Sanayi",
        is_active=True,
    )
    session.add(company)
    session.flush()

    years = [2019, 2020, 2021, 2022, 2023]
    for i, year in enumerate(years):
        period_end = date(year, 12, 31)

        # Negative or very low earnings, negative FCF
        ni = -10_000_000 + i * 3_000_000  # mostly negative
        session.add(AdjustedMetric(
            company_id=company.id,
            period_end=period_end,
            reported_net_income=ni,
            monetary_gain_loss=0.0,
            adjusted_net_income=ni,
            owner_earnings=-15_000_000 + i * 2_000_000,  # all negative
            free_cash_flow=-20_000_000 + i * 1_000_000,  # all negative
            roe_adjusted=-0.05 + i * 0.02,  # mostly negative, maxes at 0.03
            roa_adjusted=-0.03 + i * 0.01,
            eps_adjusted=-1.0 + i * 0.3,    # mostly negative
            real_eps_growth_pct=None,
        ))

        # Income: volatile low margin (5%-25% swings)
        margin = 0.05 + (i % 3) * 0.10  # 5%, 15%, 25%, 5%, 15%
        net_sales = 80_000_000
        session.add(FinancialStatement(
            company_id=company.id,
            period_end=period_end,
            period_type="ANNUAL",
            statement_type="INCOME",
            data_json=_make_income_json(
                gross_profit=net_sales * margin,
                net_sales=net_sales,
            ),
        ))

        # Balance: very high D/E (~3.0)
        equity = 20_000_000
        total_assets = equity * 4.0  # D/E = 3.0
        session.add(FinancialStatement(
            company_id=company.id,
            period_end=period_end,
            period_type="ANNUAL",
            statement_type="BALANCE",
            data_json=_make_balance_json(
                total_assets=total_assets,
                total_equity=equity,
            ),
        ))

    session.flush()
    return company.id


def _add_bank(session: Session) -> int:
    """Add a bank company.

    Returns:
        Company ID.
    """
    company = Company(
        ticker="XBNK",
        name="Test Bankasi A.S.",
        company_type="BANK",
        sector_bist="Bankacilik",
        is_active=True,
    )
    session.add(company)
    session.flush()

    # Add some metrics so it's not skipped for lack of data
    for year in [2020, 2021, 2022, 2023]:
        session.add(AdjustedMetric(
            company_id=company.id,
            period_end=date(year, 12, 31),
            reported_net_income=100_000_000,
            adjusted_net_income=100_000_000,
            roe_adjusted=0.20,
            eps_adjusted=3.0,
        ))

    session.flush()
    return company.id


# ---- Tests: linear scale utilities ----


class TestLinearScale:
    """Tests for _linear_scale and _linear_scale_inverse."""

    def test_linear_scale_at_low(self):
        assert _linear_scale(5.0, 5.0, 25.0) == 0.0

    def test_linear_scale_at_high(self):
        assert _linear_scale(25.0, 5.0, 25.0) == 100.0

    def test_linear_scale_midpoint(self):
        assert _linear_scale(15.0, 5.0, 25.0) == 50.0

    def test_linear_scale_below_low(self):
        assert _linear_scale(0.0, 5.0, 25.0) == 0.0

    def test_linear_scale_above_high(self):
        assert _linear_scale(50.0, 5.0, 25.0) == 100.0

    def test_inverse_at_low(self):
        assert _linear_scale_inverse(0.3, 0.3, 2.0) == 100.0

    def test_inverse_at_high(self):
        assert _linear_scale_inverse(2.0, 0.3, 2.0) == 0.0

    def test_inverse_midpoint(self):
        result = _linear_scale_inverse(1.15, 0.3, 2.0)
        assert 45.0 < result < 55.0  # roughly 50%


# ---- Tests: BuffettScorer ----


class TestBuffettScorer:
    """Tests for BuffettScorer.score()."""

    def test_perfect_stock_scores_above_80(self, scorer, session):
        """Perfect Buffett stock should score > 80."""
        cid = _add_perfect_stock(session)
        result = scorer.score(cid, session)

        assert result is not None
        assert result["buffett_combined"] > 80.0, (
            f"Perfect stock combined={result['buffett_combined']:.1f}, expected >80"
        )

        # Individual sub-scores should all be high
        assert result["roe_level"] is not None and result["roe_level"] > 80
        assert result["earnings_quality"] == 100.0  # all years positive
        assert result["fcf_quality"] == 100.0  # all years positive
        assert result["debt_safety"] is not None and result["debt_safety"] > 80

    def test_bad_stock_scores_below_30(self, scorer, session):
        """Bad stock should score < 30."""
        cid = _add_bad_stock(session)
        result = scorer.score(cid, session)

        assert result is not None
        assert result["buffett_combined"] < 30.0, (
            f"Bad stock combined={result['buffett_combined']:.1f}, expected <30"
        )

        # Debt safety should be 0 (D/E = 3.0 > 2.0 max)
        assert result["debt_safety"] == 0.0

    def test_bank_returns_none(self, scorer, session):
        """Bank company should return None."""
        cid = _add_bank(session)
        result = scorer.score(cid, session)
        assert result is None

    def test_holding_returns_none(self, scorer, session):
        """Holding company should return None."""
        company = Company(
            ticker="XHOL",
            name="Test Holding A.S.",
            company_type="HOLDING",
            is_active=True,
        )
        session.add(company)
        session.flush()
        result = scorer.score(company.id, session)
        assert result is None

    def test_insufficient_data_returns_none(self, scorer, session):
        """Company with < 3 years data should return None for combined."""
        company = Company(
            ticker="XNEW",
            name="New IPO A.S.",
            company_type="OPERATING",
            is_active=True,
        )
        session.add(company)
        session.flush()

        # Only 2 years
        for year in [2022, 2023]:
            session.add(AdjustedMetric(
                company_id=company.id,
                period_end=date(year, 12, 31),
                roe_adjusted=0.20,
                eps_adjusted=3.0,
            ))
        session.flush()

        result = scorer.score(company.id, session)
        assert result is None

    def test_nonexistent_company_returns_none(self, scorer, session):
        """Nonexistent company ID should return None."""
        result = scorer.score(99999, session)
        assert result is None

    def test_result_has_all_keys(self, scorer, session):
        """Result dict should have all expected keys."""
        cid = _add_perfect_stock(session)
        result = scorer.score(cid, session)

        expected_keys = {
            "roe_level", "roe_consistency", "gross_margin", "margin_stability",
            "debt_safety", "earnings_quality", "fcf_quality",
            "owner_earnings_trend", "buffett_combined",
        }
        assert set(result.keys()) == expected_keys

    def test_scores_are_in_range(self, scorer, session):
        """All non-None scores should be in 0-100 range."""
        cid = _add_perfect_stock(session)
        result = scorer.score(cid, session)

        for key, value in result.items():
            if value is not None:
                assert 0.0 <= value <= 100.0, f"{key}={value} out of range"


# ---- Test with real BIMAS data (integration, skip if no DB) ----


class TestBuffettReal:
    """Integration test using real DB data for BIMAS."""

    @pytest.fixture
    def real_session(self):
        """Try to connect to the real database."""
        from pathlib import Path
        db_path = Path(__file__).resolve().parent.parent.parent / "data" / "bist_picker.db"
        if not db_path.exists():
            pytest.skip("Real DB not found")

        eng = create_engine(f"sqlite:///{db_path}")
        Session_ = sessionmaker(bind=eng)
        sess = Session_()
        yield sess
        sess.close()

    def test_bimas_score(self, scorer, real_session):
        """BIMAS (BIM) should be scoreable as a good quality company."""
        company = (
            real_session.query(Company)
            .filter(Company.ticker == "BIMAS")
            .first()
        )
        if company is None:
            pytest.skip("BIMAS not in database")

        # Ensure BIMAS is classified as OPERATING
        if company.company_type not in ("OPERATING", None, ""):
            pytest.skip(f"BIMAS classified as {company.company_type}")

        # Check we have enough adjusted metrics
        metric_count = (
            real_session.query(AdjustedMetric)
            .filter(AdjustedMetric.company_id == company.id)
            .count()
        )
        if metric_count < 3:
            pytest.skip(f"BIMAS has only {metric_count} metric periods")

        result = scorer.score(company.id, real_session)
        if result is None:
            pytest.skip("BIMAS returned None (insufficient data)")

        # BIMAS is a quality company — combined score should be reasonable
        assert result["buffett_combined"] is not None
        assert result["buffett_combined"] > 0, "BIMAS should have a positive Buffett score"
