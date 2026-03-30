"""Tests for Graham and Piotroski scoring factors.

Tests:
- Piotroski perfect score (all 9 signals pass) -> fscore_total = 9
- Piotroski worst case (all signals fail) -> fscore_total = 0
- Piotroski inflation nuance: margin ratio comparison is inflation-neutral
- Graham: stock trading below Graham Number -> high score
- Graham: bank exclusion -> returns None
"""

import json
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.db.schema import (
    AdjustedMetric,
    Base,
    Company,
    DailyPrice,
    FinancialStatement,
)
from bist_picker.scoring.factors.graham import GrahamScorer
from bist_picker.scoring.factors.piotroski import (
    PiotroskiScorer,
    _current_ratio,
    _gross_margin,
    _leverage_ratio,
)


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
def graham_scorer():
    return GrahamScorer()


@pytest.fixture
def piotroski_scorer():
    return PiotroskiScorer()


# ---- JSON builders ----


def _balance_json(
    total_assets: float,
    current_assets: float,
    current_liabilities: float,
    lt_liabilities: float,
    total_equity: float,
    share_capital: float,
) -> str:
    """Build balance sheet JSON."""
    return json.dumps([
        {"item_code": "1BL", "desc_tr": "TOPLAM VARLIKLAR", "desc_eng": "TOTAL ASSETS", "value": total_assets},
        {"item_code": "1A", "desc_tr": "DONEN VARLIKLAR", "desc_eng": "CURRENT ASSETS", "value": current_assets},
        {"item_code": "2A", "desc_tr": "KISA VADELI YUKUMLULUKLER", "desc_eng": "SHORT TERM LIABILITIES", "value": current_liabilities},
        {"item_code": "2B", "desc_tr": "UZUN VADELI YUKUMLULUKLER", "desc_eng": "LONG TERM LIABILITIES", "value": lt_liabilities},
        {"item_code": "2N", "desc_tr": "OZKAYNAKLAR", "desc_eng": "SHAREHOLDERS EQUITY", "value": total_equity},
        {"item_code": "2OA", "desc_tr": "ODENMIS SERMAYE", "desc_eng": "SHARE CAPITAL", "value": share_capital},
    ])


def _income_json(gross_profit: float, net_sales: float) -> str:
    """Build income statement JSON."""
    return json.dumps([
        {"item_code": "3C", "desc_tr": "Satis Gelirleri", "desc_eng": "Net Sales", "value": net_sales},
        {"item_code": "3D", "desc_tr": "BRUT KAR", "desc_eng": "GROSS PROFIT", "value": gross_profit},
    ])


def _cashflow_json(cfo: float) -> str:
    """Build cash flow statement JSON."""
    return json.dumps([
        {"item_code": "4C", "desc_tr": "ISLETME FAALIYETLERINDEN NAKITLER", "desc_eng": "Net Cash from Operations", "value": cfo},
    ])


# ---- Data setup helpers ----


def _add_piotroski_perfect(session) -> int:
    """Add a company that scores 9/9 on Piotroski F-Score.

    Previous -> Current improvements:
    - ROA: 0.08 -> 0.10 (F1, F3)
    - CFO: positive (F2)
    - CFO > adjusted NI (F4)
    - LT debt/assets: decreased (F5)
    - Current ratio: improved (F6)
    - Shares: same (F7)
    - Gross margin: improved (F8)
    - Asset turnover: improved (F9)
    """
    company = Company(
        ticker="PRFT", name="Perfect Piotroski A.S.",
        company_type="OPERATING", is_active=True,
    )
    session.add(company)
    session.flush()

    # Previous year (2022)
    session.add(AdjustedMetric(
        company_id=company.id, period_end=date(2022, 12, 31),
        roa_adjusted=0.08, adjusted_net_income=8_000_000,
        eps_adjusted=2.0,
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2022, 12, 31),
        period_type="ANNUAL", statement_type="BALANCE",
        data_json=_balance_json(
            total_assets=100_000_000, current_assets=50_000_000,
            current_liabilities=25_000_000, lt_liabilities=30_000_000,
            total_equity=45_000_000, share_capital=10_000_000,
        ),
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2022, 12, 31),
        period_type="ANNUAL", statement_type="INCOME",
        data_json=_income_json(gross_profit=30_000_000, net_sales=100_000_000),
    ))

    # Current year (2023) - everything improved
    session.add(AdjustedMetric(
        company_id=company.id, period_end=date(2023, 12, 31),
        roa_adjusted=0.10, adjusted_net_income=12_000_000,
        reported_net_income=10_000_000,  # < CFO (15M) → F4=1
        eps_adjusted=3.0,
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2023, 12, 31),
        period_type="ANNUAL", statement_type="BALANCE",
        data_json=_balance_json(
            total_assets=120_000_000, current_assets=70_000_000,
            current_liabilities=30_000_000, lt_liabilities=25_000_000,  # decreased LT debt
            total_equity=65_000_000, share_capital=10_000_000,  # same shares
        ),
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2023, 12, 31),
        period_type="ANNUAL", statement_type="INCOME",
        data_json=_income_json(gross_profit=42_000_000, net_sales=130_000_000),  # higher margin
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2023, 12, 31),
        period_type="ANNUAL", statement_type="CASHFLOW",
        data_json=_cashflow_json(cfo=15_000_000),  # positive, > reported NI
    ))

    session.flush()
    return company.id


def _add_piotroski_worst(session) -> int:
    """Add a company that scores 0/9 on Piotroski F-Score.

    Everything deteriorated or negative.
    """
    company = Company(
        ticker="WRST", name="Worst Piotroski A.S.",
        company_type="OPERATING", is_active=True,
    )
    session.add(company)
    session.flush()

    # Previous year (2022) - was okay
    session.add(AdjustedMetric(
        company_id=company.id, period_end=date(2022, 12, 31),
        roa_adjusted=0.05, adjusted_net_income=5_000_000,
        eps_adjusted=1.0,
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2022, 12, 31),
        period_type="ANNUAL", statement_type="BALANCE",
        data_json=_balance_json(
            total_assets=100_000_000, current_assets=50_000_000,
            current_liabilities=20_000_000, lt_liabilities=20_000_000,
            total_equity=60_000_000, share_capital=10_000_000,
        ),
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2022, 12, 31),
        period_type="ANNUAL", statement_type="INCOME",
        data_json=_income_json(gross_profit=35_000_000, net_sales=100_000_000),
    ))

    # Current year (2023) - everything deteriorated
    session.add(AdjustedMetric(
        company_id=company.id, period_end=date(2023, 12, 31),
        roa_adjusted=-0.02,  # F1: negative ROA
        adjusted_net_income=-2_000_000,
        reported_net_income=5_000_000,  # > CFO (-5M) → F4=0
        eps_adjusted=-0.2,
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2023, 12, 31),
        period_type="ANNUAL", statement_type="BALANCE",
        data_json=_balance_json(
            total_assets=100_000_000, current_assets=30_000_000,
            current_liabilities=35_000_000,  # F6: worse current ratio
            lt_liabilities=30_000_000,  # F5: higher leverage
            total_equity=35_000_000,
            share_capital=15_000_000,  # F7: dilution
        ),
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2023, 12, 31),
        period_type="ANNUAL", statement_type="INCOME",
        data_json=_income_json(gross_profit=20_000_000, net_sales=90_000_000),  # F8, F9: worse
    ))
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2023, 12, 31),
        period_type="ANNUAL", statement_type="CASHFLOW",
        data_json=_cashflow_json(cfo=-5_000_000),  # F2: negative, F4: < adj NI
    ))

    session.flush()
    return company.id


def _add_graham_undervalued(session) -> int:
    """Add a company trading well below Graham Number.

    EPS = 5.0, BVPS = 50.0
    Graham Number = sqrt(22.5 * 5 * 50) = sqrt(5625) = 75.0
    Price = 40.0 -> Graham Number ratio = 75/40 = 1.875 (deeply undervalued)
    """
    company = Company(
        ticker="GVAL", name="Graham Value A.S.",
        company_type="OPERATING", is_active=True,
    )
    session.add(company)
    session.flush()

    # Adjusted metrics with growth data
    for i, year in enumerate([2020, 2021, 2022, 2023]):
        session.add(AdjustedMetric(
            company_id=company.id, period_end=date(year, 12, 31),
            eps_adjusted=3.0 + i * 0.67,  # 3.0, 3.67, 4.34, 5.0
            real_eps_growth_pct=0.10 if i > 0 else None,
            roa_adjusted=0.08 + i * 0.005,
            adjusted_net_income=30_000_000 + i * 5_000_000,
        ))

    # Balance sheet: equity = 500M, shares = 10M -> BVPS = 50
    session.add(FinancialStatement(
        company_id=company.id, period_end=date(2023, 12, 31),
        period_type="ANNUAL", statement_type="BALANCE",
        data_json=_balance_json(
            total_assets=800_000_000, current_assets=300_000_000,
            current_liabilities=150_000_000, lt_liabilities=150_000_000,
            total_equity=500_000_000, share_capital=10_000_000,
        ),
    ))

    # Price well below Graham Number
    session.add(DailyPrice(
        company_id=company.id, date=date(2024, 1, 15),
        close=40.0, source="ISYATIRIM",
    ))

    session.flush()
    return company.id


def _add_bank(session) -> int:
    """Add a bank company."""
    company = Company(
        ticker="XBNK", name="Test Bankasi A.S.",
        company_type="BANK", is_active=True,
    )
    session.add(company)
    session.flush()

    session.add(AdjustedMetric(
        company_id=company.id, period_end=date(2023, 12, 31),
        eps_adjusted=3.0, roa_adjusted=0.02,
        adjusted_net_income=100_000_000,
    ))
    session.flush()
    return company.id


# ---- Piotroski tests ----


class TestPiotroskiScorer:
    """Tests for PiotroskiScorer.score()."""

    def test_perfect_score_is_9(self, piotroski_scorer, session):
        """Company with all improving metrics should score 9/9."""
        cid = _add_piotroski_perfect(session)
        result = piotroski_scorer.score(cid, session)

        assert result is not None
        assert result["fscore_total"] == 9
        assert result["fscore_normalized"] == 100.0

        # Check each signal individually
        assert result["f1_positive_roa"] == 1
        assert result["f2_positive_cfo"] == 1
        assert result["f3_improving_roa"] == 1
        assert result["f4_accruals"] == 1
        assert result["f5_declining_leverage"] == 1
        assert result["f6_improving_liquidity"] == 1
        assert result["f7_no_dilution"] == 1
        assert result["f8_improving_margin"] == 1
        assert result["f9_improving_turnover"] == 1

    def test_worst_score_is_0(self, piotroski_scorer, session):
        """Company with all deteriorating metrics should score 0/9."""
        cid = _add_piotroski_worst(session)
        result = piotroski_scorer.score(cid, session)

        assert result is not None
        assert result["fscore_total"] == 0
        assert result["fscore_normalized"] == 0.0

        assert result["f1_positive_roa"] == 0
        assert result["f2_positive_cfo"] == 0
        assert result["f3_improving_roa"] == 0
        assert result["f4_accruals"] == 0
        assert result["f5_declining_leverage"] == 0
        assert result["f6_improving_liquidity"] == 0
        assert result["f7_no_dilution"] == 0
        assert result["f8_improving_margin"] == 0
        assert result["f9_improving_turnover"] == 0

    def test_insufficient_data_returns_none(self, piotroski_scorer, session):
        """Company with only 1 period should return None."""
        company = Company(
            ticker="XNEW", name="New A.S.",
            company_type="OPERATING", is_active=True,
        )
        session.add(company)
        session.flush()

        session.add(AdjustedMetric(
            company_id=company.id, period_end=date(2023, 12, 31),
            roa_adjusted=0.10, eps_adjusted=3.0,
        ))
        session.flush()

        result = piotroski_scorer.score(company.id, session)
        assert result is None

    def test_inflation_neutral_margin_comparison(self, piotroski_scorer, session):
        """Margin ratio comparison should be inflation-neutral.

        If nominal gross margin went from 30% to 33%, this is an improvement
        regardless of inflation (20% or any other level) because margin is
        a ratio where both numerator and denominator inflate together.
        """
        company = Company(
            ticker="XINF", name="Inflation Test A.S.",
            company_type="OPERATING", is_active=True,
        )
        session.add(company)
        session.flush()

        # Year 1: margin = 30M/100M = 30% (pre-inflation prices)
        session.add(AdjustedMetric(
            company_id=company.id, period_end=date(2022, 12, 31),
            roa_adjusted=0.05, adjusted_net_income=5_000_000,
            eps_adjusted=1.0,
        ))
        session.add(FinancialStatement(
            company_id=company.id, period_end=date(2022, 12, 31),
            period_type="ANNUAL", statement_type="INCOME",
            data_json=_income_json(gross_profit=30_000_000, net_sales=100_000_000),
        ))
        session.add(FinancialStatement(
            company_id=company.id, period_end=date(2022, 12, 31),
            period_type="ANNUAL", statement_type="BALANCE",
            data_json=_balance_json(
                total_assets=100_000_000, current_assets=40_000_000,
                current_liabilities=20_000_000, lt_liabilities=20_000_000,
                total_equity=60_000_000, share_capital=10_000_000,
            ),
        ))

        # Year 2: margin = 39.6M/120M = 33% (with 20% inflation, both inflated)
        # Nominal values are higher due to inflation, but margin ratio improved
        session.add(AdjustedMetric(
            company_id=company.id, period_end=date(2023, 12, 31),
            roa_adjusted=0.06, adjusted_net_income=7_000_000,
            eps_adjusted=1.4,
        ))
        session.add(FinancialStatement(
            company_id=company.id, period_end=date(2023, 12, 31),
            period_type="ANNUAL", statement_type="INCOME",
            data_json=_income_json(gross_profit=39_600_000, net_sales=120_000_000),
        ))
        session.add(FinancialStatement(
            company_id=company.id, period_end=date(2023, 12, 31),
            period_type="ANNUAL", statement_type="BALANCE",
            data_json=_balance_json(
                total_assets=120_000_000, current_assets=50_000_000,
                current_liabilities=25_000_000, lt_liabilities=20_000_000,
                total_equity=75_000_000, share_capital=10_000_000,
            ),
        ))
        session.add(FinancialStatement(
            company_id=company.id, period_end=date(2023, 12, 31),
            period_type="ANNUAL", statement_type="CASHFLOW",
            data_json=_cashflow_json(cfo=10_000_000),
        ))

        session.flush()
        result = piotroski_scorer.score(company.id, session)

        assert result is not None
        # 30% -> 33% is an improvement regardless of inflation
        assert result["f8_improving_margin"] == 1, (
            "Margin improvement 30%->33% should score as improvement "
            "regardless of inflation because margin is a ratio"
        )

    def test_result_has_all_keys(self, piotroski_scorer, session):
        """Result should have all expected keys."""
        cid = _add_piotroski_perfect(session)
        result = piotroski_scorer.score(cid, session)

        expected_keys = {
            "f1_positive_roa", "f2_positive_cfo", "f3_improving_roa",
            "f4_accruals", "f5_declining_leverage", "f6_improving_liquidity",
            "f7_no_dilution", "f8_improving_margin", "f9_improving_turnover",
            "fscore_total", "fscore_normalized",
        }
        assert set(result.keys()) == expected_keys


# ---- Piotroski helper function tests ----


class TestPiotroskiHelpers:
    """Tests for Piotroski helper functions."""

    def test_leverage_ratio(self):
        bal = {"lt_liabilities": 30.0, "total_assets": 100.0}
        assert _leverage_ratio(bal) == 0.3

    def test_leverage_ratio_missing_data(self):
        assert _leverage_ratio({}) is None

    def test_current_ratio(self):
        bal = {"current_assets": 50.0, "current_liabilities": 25.0}
        assert _current_ratio(bal) == 2.0

    def test_gross_margin(self):
        inc = {"gross_profit": 30.0, "net_sales": 100.0}
        assert _gross_margin(inc) == 0.3


# ---- Graham tests ----


class TestGrahamScorer:
    """Tests for GrahamScorer.score()."""

    def test_undervalued_stock_high_score(self, graham_scorer, session):
        """Stock trading below Graham Number should score high."""
        cid = _add_graham_undervalued(session)
        result = graham_scorer.score(cid, session)

        assert result is not None
        assert result["graham_combined"] is not None

        # Graham Number ratio should be high (price=40, GN=75)
        assert result["graham_number_ratio"] is not None
        assert result["graham_number_ratio"] > 70, (
            f"Graham number ratio score={result['graham_number_ratio']:.1f}, expected >70"
        )

    def test_bank_returns_none(self, graham_scorer, session):
        """Bank company should return None."""
        cid = _add_bank(session)
        result = graham_scorer.score(cid, session)
        assert result is None

    def test_insurance_returns_none(self, graham_scorer, session):
        """Insurance company should return None."""
        company = Company(
            ticker="XSGR", name="Test Sigorta A.S.",
            company_type="INSURANCE", is_active=True,
        )
        session.add(company)
        session.flush()
        result = graham_scorer.score(company.id, session)
        assert result is None

    def test_no_price_returns_none(self, graham_scorer, session):
        """Company without price data should return None."""
        company = Company(
            ticker="XNOP", name="No Price A.S.",
            company_type="OPERATING", is_active=True,
        )
        session.add(company)
        session.flush()

        session.add(AdjustedMetric(
            company_id=company.id, period_end=date(2023, 12, 31),
            eps_adjusted=3.0, roa_adjusted=0.10,
        ))
        session.add(FinancialStatement(
            company_id=company.id, period_end=date(2023, 12, 31),
            period_type="ANNUAL", statement_type="BALANCE",
            data_json=_balance_json(
                total_assets=100_000_000, current_assets=50_000_000,
                current_liabilities=20_000_000, lt_liabilities=20_000_000,
                total_equity=60_000_000, share_capital=10_000_000,
            ),
        ))
        session.flush()

        result = graham_scorer.score(company.id, session)
        assert result is None

    def test_result_has_all_keys(self, graham_scorer, session):
        """Result dict should have all expected keys."""
        cid = _add_graham_undervalued(session)
        result = graham_scorer.score(cid, session)

        expected_keys = {
            "graham_number_ratio", "ncav_ratio", "pe_pb_product",
            "graham_growth_value", "graham_combined",
        }
        assert set(result.keys()) == expected_keys

    def test_scores_in_range(self, graham_scorer, session):
        """All non-None scores should be 0-100."""
        cid = _add_graham_undervalued(session)
        result = graham_scorer.score(cid, session)

        for key, value in result.items():
            if value is not None:
                assert 0.0 <= value <= 100.0, f"{key}={value} out of range"

    def test_nonexistent_company(self, graham_scorer, session):
        """Nonexistent company ID should return None."""
        result = graham_scorer.score(99999, session)
        assert result is None
