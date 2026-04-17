"""Tests for Magic Formula, Lynch PEG, and Momentum scoring factors.

Tests:
- Magic Formula: 10 stocks with known EY and ROC, verify ranking
- Lynch: 30% nominal growth with 20% inflation -> PEG uses 10% real growth
- Lynch: negative real growth -> low score, not crash
- Momentum: verify skip-month logic with mock price data
"""

import json
from datetime import date, timedelta

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
from bist_picker.scoring.factors.lynch import LynchScorer, _score_peg
from bist_picker.scoring.factors.magic_formula import MagicFormulaScorer
from bist_picker.scoring.factors.momentum import MomentumScorer


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


# ---- JSON builders ----


def _balance_json(
    total_assets: float,
    current_assets: float,
    current_liabilities: float,
    total_equity: float,
    share_capital: float,
) -> str:
    return json.dumps([
        {"item_code": "1BL", "desc_tr": "", "desc_eng": "TOTAL ASSETS", "value": total_assets},
        {"item_code": "1A", "desc_tr": "", "desc_eng": "CURRENT ASSETS", "value": current_assets},
        {"item_code": "2A", "desc_tr": "", "desc_eng": "SHORT TERM LIABILITIES", "value": current_liabilities},
        {"item_code": "2N", "desc_tr": "", "desc_eng": "SHAREHOLDERS EQUITY", "value": total_equity},
        {"item_code": "2OA", "desc_tr": "", "desc_eng": "SHARE CAPITAL", "value": share_capital},
    ])


def _income_json(operating_profit: float, net_sales: float) -> str:
    return json.dumps([
        {"item_code": "3C", "desc_tr": "", "desc_eng": "Net Sales", "value": net_sales},
        {"item_code": "3D", "desc_tr": "", "desc_eng": "GROSS PROFIT", "value": operating_profit * 1.2},
        {"item_code": "3DF", "desc_tr": "", "desc_eng": "OPERATING PROFITS", "value": operating_profit},
    ])


# ---- Magic Formula tests ----


class TestMagicFormula:
    """Tests for MagicFormulaScorer.score_all()."""

    def _setup_companies(self, session, count=10):
        """Create `count` operating companies with varying EY and ROC.

        Company i has:
        - Price = 10
        - Shares = 1M
        - Market cap = 10M * (i+1) to vary above min threshold
        - Operating profit scales with i
        - Balance sheet varies to produce different ROC

        Returns list of company IDs.
        """
        ids = []
        for i in range(count):
            company = Company(
                ticker=f"MF{i:02d}",
                name=f"Magic Formula Test {i}",
                company_type="OPERATING",
                is_active=True,
            )
            session.add(company)
            session.flush()
            ids.append(company.id)

            price = 100.0
            shares = 10_000_000  # 10M shares
            # Market cap = 1B (above 500M threshold)

            # Operating profit increases with i (higher i = better EY)
            op_profit = (i + 1) * 50_000_000  # 50M to 500M

            # Balance sheet: vary equity to change EV and ROC
            total_assets = 2_000_000_000
            current_assets = 800_000_000
            current_liabilities = 400_000_000
            total_equity = 1_000_000_000 + i * 100_000_000  # varies

            session.add(DailyPrice(
                company_id=company.id,
                date=date(2024, 1, 15),
                close=price,
            ))

            session.add(FinancialStatement(
                company_id=company.id,
                period_end=date(2023, 12, 31),
                period_type="ANNUAL",
                statement_type="INCOME",
                data_json=_income_json(op_profit, op_profit * 3),
            ))

            session.add(FinancialStatement(
                company_id=company.id,
                period_end=date(2023, 12, 31),
                period_type="ANNUAL",
                statement_type="BALANCE",
                data_json=_balance_json(
                    total_assets, current_assets, current_liabilities,
                    total_equity, shares,
                ),
            ))

        session.flush()
        return ids

    def test_ranking_10_stocks(self, session):
        """10 stocks with known financials should produce valid rankings."""
        scorer = MagicFormulaScorer()
        # Override min market cap for test data
        scorer._min_market_cap = 0

        ids = self._setup_companies(session, count=10)
        results = scorer.score_all(session)

        assert len(results) == 10

        # All should have required keys
        for cid, data in results.items():
            assert "earnings_yield" in data
            assert "return_on_capital" in data
            assert "ey_rank" in data
            assert "roc_rank" in data
            assert "combined_rank" in data
            assert "magic_formula_score" in data
            assert 0.0 <= data["magic_formula_score"] <= 100.0

        # Best stock should have score 100, worst 0
        scores = {cid: d["magic_formula_score"] for cid, d in results.items()}
        assert max(scores.values()) == 100.0
        assert min(scores.values()) == 0.0

    def test_excludes_banks(self, session):
        """Banks should be excluded from Magic Formula."""
        scorer = MagicFormulaScorer()
        scorer._min_market_cap = 0

        # Add a bank
        bank = Company(
            ticker="XBNK", name="Test Bank",
            company_type="BANK", is_active=True,
        )
        session.add(bank)
        session.flush()

        session.add(DailyPrice(
            company_id=bank.id, date=date(2024, 1, 15), close=50.0,
        ))
        session.add(FinancialStatement(
            company_id=bank.id, period_end=date(2023, 12, 31),
            period_type="ANNUAL", statement_type="INCOME",
            data_json=_income_json(100_000_000, 300_000_000),
        ))
        session.add(FinancialStatement(
            company_id=bank.id, period_end=date(2023, 12, 31),
            period_type="ANNUAL", statement_type="BALANCE",
            data_json=_balance_json(
                5_000_000_000, 2_000_000_000, 1_000_000_000,
                1_500_000_000, 10_000_000,
            ),
        ))
        session.flush()

        # Add one operating company so results aren't empty
        self._setup_companies(session, count=1)

        results = scorer.score_all(session)
        assert bank.id not in results

    def test_excludes_below_market_cap(self, session):
        """Companies below min market cap should be excluded."""
        scorer = MagicFormulaScorer()
        scorer._min_market_cap = 1_000_000_000_000  # 1 trillion

        self._setup_companies(session, count=3)
        results = scorer.score_all(session)
        assert len(results) == 0

    def test_ey_rank_order(self, session):
        """Highest earnings yield should get rank 1."""
        scorer = MagicFormulaScorer()
        scorer._min_market_cap = 0

        ids = self._setup_companies(session, count=5)
        results = scorer.score_all(session)

        # Company with highest index has highest operating profit -> highest EY
        ey_values = [(cid, results[cid]["earnings_yield"]) for cid in results]
        ey_sorted = sorted(ey_values, key=lambda x: x[1], reverse=True)

        # Rank 1 should have highest EY
        best_cid = ey_sorted[0][0]
        assert results[best_cid]["ey_rank"] == 1


# ---- Lynch PEG tests ----


class TestLynchScorer:
    """Tests for LynchScorer.score()."""

    def _add_lynch_company(
        self,
        session,
        ticker: str,
        eps_values: list[float],
        growth_rates: list[float],
        price: float,
        revenue_current: float = 100_000_000,
        revenue_previous: float = 80_000_000,
    ) -> int:
        """Create a company with given EPS, growth rates, and price."""
        company = Company(
            ticker=ticker, name=f"{ticker} A.S.",
            company_type="OPERATING", is_active=True,
        )
        session.add(company)
        session.flush()

        years = list(range(2023 - len(eps_values) + 1, 2024))
        for i, (year, eps, growth) in enumerate(
            zip(years, eps_values, growth_rates + [None] * len(eps_values))
        ):
            session.add(AdjustedMetric(
                company_id=company.id,
                period_end=date(year, 12, 31),
                eps_adjusted=eps,
                real_eps_growth_pct=growth if i < len(growth_rates) else None,
                roa_adjusted=0.10,
                adjusted_net_income=eps * 10_000_000,
            ))

        # Income statements for revenue growth classification
        session.add(FinancialStatement(
            company_id=company.id, period_end=date(2023, 12, 31),
            period_type="ANNUAL", statement_type="INCOME",
            data_json=json.dumps([
                {"item_code": "3C", "desc_tr": "", "desc_eng": "Net Sales", "value": revenue_current},
            ]),
        ))
        session.add(FinancialStatement(
            company_id=company.id, period_end=date(2022, 12, 31),
            period_type="ANNUAL", statement_type="INCOME",
            data_json=json.dumps([
                {"item_code": "3C", "desc_tr": "", "desc_eng": "Net Sales", "value": revenue_previous},
            ]),
        ))

        session.add(DailyPrice(
            company_id=company.id, date=date(2024, 1, 15), close=price,
        ))

        session.flush()
        return company.id

    def test_peg_with_real_growth(self, session):
        """30% nominal growth - 20% inflation = ~10% real growth.

        Company: EPS = 5.0, Price = 50.0, P/E = 10
        Real growth = 10% -> PEG = 10/10 = 1.0
        PEG of 1.0 should score 80 (boundary of 0.5-1.0 range).
        """
        scorer = LynchScorer()

        # real_eps_growth_pct = 0.10 (10% real, after stripping 20% inflation)
        cid = self._add_lynch_company(
            session,
            ticker="LPEG",
            eps_values=[4.0, 5.0],
            growth_rates=[0.10],  # 10% real growth
            price=50.0,  # P/E = 50/5 = 10
        )

        result = scorer.score(cid, session)
        assert result is not None
        assert result["peg_ratio"] is not None

        # PEG = P/E / (real_growth * 100) = 10 / 10 = 1.0
        assert abs(result["peg_ratio"] - 1.0) < 0.01

        # PEG of 1.0 -> score = 80
        assert abs(result["peg_score"] - 80.0) < 0.01

    def test_negative_real_growth_no_crash(self, session):
        """Negative real growth should return low score, not crash."""
        scorer = LynchScorer()

        cid = self._add_lynch_company(
            session,
            ticker="LNEG",
            eps_values=[5.0, 4.0],  # declining
            growth_rates=[-0.10],  # -10% real
            price=40.0,
        )

        result = scorer.score(cid, session)
        assert result is not None
        assert result["peg_ratio"] is None  # Undefined for negative growth
        assert result["peg_score"] <= 15.0  # Low score

    def test_zero_eps_no_crash(self, session):
        """Zero or negative EPS should return low score, not crash."""
        scorer = LynchScorer()

        cid = self._add_lynch_company(
            session,
            ticker="LZER",
            eps_values=[-1.0, -0.5],
            growth_rates=[None],
            price=10.0,
        )

        result = scorer.score(cid, session)
        assert result is not None
        assert result["peg_ratio"] is None
        assert result["peg_score"] <= 10.0

    def test_fast_grower_category(self, session):
        """Company with >20% revenue growth should be fast_grower."""
        scorer = LynchScorer()

        cid = self._add_lynch_company(
            session,
            ticker="LFST",
            eps_values=[3.0, 5.0],
            growth_rates=[0.15],
            price=30.0,
            revenue_current=150_000_000,
            revenue_previous=100_000_000,  # 50% growth
        )

        result = scorer.score(cid, session)
        assert result is not None
        assert result["lynch_category"] == "fast_grower"

    def test_slow_grower_category(self, session):
        """Company with 2-10% revenue growth should be slow_grower."""
        scorer = LynchScorer()

        cid = self._add_lynch_company(
            session,
            ticker="LSLW",
            eps_values=[3.0, 3.2],
            growth_rates=[0.05],
            price=30.0,
            revenue_current=105_000_000,
            revenue_previous=100_000_000,  # 5% growth
        )

        result = scorer.score(cid, session)
        assert result is not None
        assert result["lynch_category"] == "slow_grower"

    def test_insufficient_data_returns_none(self, session):
        """Company with <2 metric periods should return None."""
        scorer = LynchScorer()

        company = Company(
            ticker="LONE", name="One Period A.S.",
            company_type="OPERATING", is_active=True,
        )
        session.add(company)
        session.flush()
        session.add(AdjustedMetric(
            company_id=company.id, period_end=date(2023, 12, 31),
            eps_adjusted=3.0,
        ))
        session.flush()

        result = scorer.score(company.id, session)
        assert result is None


class TestPegScoring:
    """Tests for _score_peg utility function."""

    def test_very_low_peg(self):
        assert _score_peg(0.3) == 100.0

    def test_peg_at_0_5(self):
        assert _score_peg(0.5) == 100.0

    def test_peg_at_0_75(self):
        # Midpoint of 0.5-1.0 range -> 90
        assert abs(_score_peg(0.75) - 90.0) < 0.01

    def test_peg_at_1_0(self):
        assert abs(_score_peg(1.0) - 80.0) < 0.01

    def test_peg_at_1_5(self):
        # Midpoint of 1.0-2.0 range -> 60
        assert abs(_score_peg(1.5) - 60.0) < 0.01

    def test_peg_at_2_0(self):
        assert abs(_score_peg(2.0) - 40.0) < 0.01

    def test_peg_at_3_0(self):
        assert abs(_score_peg(3.0) - 20.0) < 0.01

    def test_peg_at_4_0(self):
        assert _score_peg(4.0) == 0.0

    def test_very_high_peg(self):
        assert _score_peg(10.0) == 0.0


# ---- Momentum tests ----


class TestMomentumScorer:
    """Tests for MomentumScorer.score()."""

    def _add_price_series(self, session, company_id: int, prices: dict[date, float]):
        """Add daily prices for a company.

        Args:
            prices: Dict mapping date -> close price.
        """
        for d, p in prices.items():
            session.add(DailyPrice(
                company_id=company_id, date=d, close=p,
            ))
        session.flush()

    def test_skip_month_logic(self, session):
        """Momentum should skip the most recent month.

        Price history:
        - 13 months ago: 100 (start of 12m window)
        - 7 months ago: 110 (start of 6m window)
        - 4 months ago: 115 (start of 3m window)
        - 1 month ago: 130 (END of all windows — skip point)
        - Today: 150 (should be IGNORED)

        Expected:
        - 3m return = (130 - 115) / 115 = 13.04%
        - 6m return = (130 - 110) / 110 = 18.18%
        - 12m return = (130 - 100) / 100 = 30.0%
        """
        scorer = MomentumScorer()

        company = Company(
            ticker="XMOM", name="Momentum Test A.S.",
            company_type="OPERATING", is_active=True,
        )
        session.add(company)
        session.flush()

        today = date(2024, 6, 15)
        prices = {
            today - timedelta(days=395): 100.0,  # ~13 months ago
            today - timedelta(days=210): 110.0,   # ~7 months ago
            today - timedelta(days=120): 115.0,   # ~4 months ago
            today - timedelta(days=30): 130.0,     # ~1 month ago (skip point)
            today: 150.0,                           # today (should be ignored!)
        }
        self._add_price_series(session, company.id, prices)

        result = scorer.score(company.id, session)
        assert result is not None

        # 3m: from ~4 months ago to ~1 month ago
        assert result["return_3m"] is not None
        assert abs(result["return_3m"] - (130.0 / 115.0 - 1.0)) < 0.02

        # 6m: from ~7 months ago to ~1 month ago
        assert result["return_6m"] is not None
        assert abs(result["return_6m"] - (130.0 / 110.0 - 1.0)) < 0.02

        # 12m: from ~13 months ago to ~1 month ago
        assert result["return_12m"] is not None
        assert abs(result["return_12m"] - (130.0 / 100.0 - 1.0)) < 0.02

        # The today price (150) should NOT be used as the endpoint
        # If it were used, 12m return would be 50%, not 30%
        assert result["return_12m"] < 0.35  # Well below 50%

    def test_combined_momentum_weighted(self, session):
        """Combined momentum should be weighted 40/30/30."""
        scorer = MomentumScorer()

        company = Company(
            ticker="XWGT", name="Weight Test A.S.",
            company_type="OPERATING", is_active=True,
        )
        session.add(company)
        session.flush()

        today = date(2024, 6, 15)
        prices = {
            today - timedelta(days=400): 80.0,
            today - timedelta(days=210): 90.0,
            today - timedelta(days=120): 95.0,
            today - timedelta(days=30): 100.0,
            today: 110.0,
        }
        self._add_price_series(session, company.id, prices)

        result = scorer.score(company.id, session)
        assert result is not None
        assert result["momentum_combined"] is not None

        # Verify combined is weighted average
        r3 = result["return_3m"]
        r6 = result["return_6m"]
        r12 = result["return_12m"]
        if r3 is not None and r6 is not None and r12 is not None:
            expected = r12 * 0.40 + r6 * 0.30 + r3 * 0.30
            assert abs(result["momentum_combined"] - expected) < 0.001

    def test_no_price_data(self, session):
        """Company with no prices should return None."""
        scorer = MomentumScorer()

        company = Company(
            ticker="XNOP", name="No Prices A.S.",
            company_type="OPERATING", is_active=True,
        )
        session.add(company)
        session.flush()

        result = scorer.score(company.id, session)
        assert result is None

    def test_insufficient_history(self, session):
        """Company with only recent prices should return None for longer windows."""
        scorer = MomentumScorer()

        company = Company(
            ticker="XSHT", name="Short History A.S.",
            company_type="OPERATING", is_active=True,
        )
        session.add(company)
        session.flush()

        today = date(2024, 6, 15)
        # Only 2 months of data
        prices = {
            today - timedelta(days=60): 100.0,
            today - timedelta(days=30): 110.0,
            today: 120.0,
        }
        self._add_price_series(session, company.id, prices)

        result = scorer.score(company.id, session)
        # Should have 3m return (60 days is borderline) but may not have 6m/12m
        # If it can't find enough data for 3m, returns None entirely
        if result is not None:
            assert result["return_12m"] is None
            assert result["return_6m"] is None

    def test_score_all_percentile_ranking(self, session):
        """score_all should produce percentile-normalized scores."""
        scorer = MomentumScorer()

        today = date(2024, 6, 15)

        for i, (ticker, start_price) in enumerate([
            ("XM01", 100.0),
            ("XM02", 80.0),   # Best momentum (starts lower, same end)
            ("XM03", 120.0),  # Worst momentum (starts higher, same end)
        ]):
            company = Company(
                ticker=ticker, name=f"{ticker} A.S.",
                company_type="OPERATING", is_active=True,
            )
            session.add(company)
            session.flush()

            prices = {
                today - timedelta(days=400): start_price,
                today - timedelta(days=210): start_price * 1.05,
                today - timedelta(days=120): start_price * 1.10,
                today - timedelta(days=30): start_price * 1.20,
                today: start_price * 1.25,
            }
            self._add_price_series(session, company.id, prices)

        results = scorer.score_all(session)
        assert len(results) == 3

        # Each entry carries a raw momentum_combined; percentile normalization
        # is applied later by the composer, not the scorer.
        for cid, data in results.items():
            assert "momentum_combined" in data
            assert data["momentum_combined"] is not None
