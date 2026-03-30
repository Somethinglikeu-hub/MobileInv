"""Tests for dashboard data access helpers."""

from datetime import date, timedelta

import pandas as pd
import pytest
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import bist_picker.dashboard.data_access as data_access
from bist_picker.dashboard.data_access import (
    get_alpha_eligibility_reasons,
    _load_latest_macro,
    get_alpha_eligible_company_ids,
    get_all_tickers,
    get_latest_scoring_date,
    get_open_positions,
    get_scoring_results,
)
from bist_picker.db.schema import (
    AdjustedMetric,
    Base,
    Company,
    DailyPrice,
    MacroRegime,
    PortfolioSelection,
    ScoringResult,
)


def _make_engine():
    """Create an isolated in-memory SQLite engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_session():
    """Create an isolated in-memory database session."""
    engine = _make_engine()
    return sessionmaker(bind=engine)()


def _session_for(engine):
    """Create a session bound to an existing engine."""
    return sessionmaker(bind=engine)()


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    """Keep cached dashboard helpers isolated across tests."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def dashboard_engine(monkeypatch):
    """Provide a test engine to cached dashboard helpers."""
    engine = _make_engine()
    monkeypatch.setattr(data_access, "get_engine", lambda: engine)
    return engine


class TestLoadLatestMacro:

    def test_backfills_missing_macro_fields_from_prior_dates(self):
        """Latest snapshot should use the newest non-null value per metric."""
        session = _make_session()
        session.add_all(
            [
                MacroRegime(
                    date=date(2026, 3, 16),
                    policy_rate_pct=0.40,
                    cpi_yoy_pct=0.3064,
                    usdtry_rate=44.021,
                    regime="RISK_OFF",
                ),
                MacroRegime(
                    date=date(2026, 3, 17),
                    policy_rate_pct=0.40,
                    cpi_yoy_pct=0.3064,
                    usdtry_rate=None,
                    regime=None,
                ),
            ]
        )
        session.commit()

        result = _load_latest_macro(session)

        assert result["date"] == date(2026, 3, 17)
        assert result["policy_rate_pct"] == pytest.approx(0.40)
        assert result["policy_rate_date"] == date(2026, 3, 17)
        assert result["cpi_yoy_pct"] == pytest.approx(0.3064)
        assert result["cpi_yoy_date"] == date(2026, 3, 17)
        assert result["usdtry_rate"] == pytest.approx(44.021)
        assert result["usdtry_date"] == date(2026, 3, 16)
        assert result["regime"] == "RISK_OFF"
        assert result["regime_date"] == date(2026, 3, 16)

        session.close()

    def test_returns_none_when_macro_table_is_empty(self):
        """Empty macro table should not raise."""
        session = _make_session()

        assert _load_latest_macro(session) is None

        session.close()


class TestDashboardCachingHelpers:

    def test_get_latest_scoring_date_returns_most_recent_date(self, dashboard_engine):
        """Latest scoring date helper should return the newest available snapshot."""
        session = _session_for(dashboard_engine)
        company = Company(
            ticker="ALPHA1",
            name="Alpha One",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            is_active=True,
        )
        session.add(company)
        session.flush()
        session.add_all(
            [
                ScoringResult(company_id=company.id, scoring_date=date(2026, 3, 15)),
                ScoringResult(company_id=company.id, scoring_date=date(2026, 3, 19)),
            ]
        )
        session.commit()

        assert get_latest_scoring_date() == date(2026, 3, 19)

        session.close()

    def test_get_scoring_results_reuses_latest_scoring_date_helper(
        self,
        dashboard_engine,
        monkeypatch,
    ):
        """Filtering without an explicit date should delegate to the cached helper."""
        session = _session_for(dashboard_engine)
        company = Company(
            ticker="SKOR1",
            name="Skor One",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            is_active=True,
        )
        session.add(company)
        session.flush()
        session.add(
            ScoringResult(
                company_id=company.id,
                scoring_date=date(2026, 3, 19),
                composite_alpha=88.0,
                risk_tier="LOW",
                data_completeness=80.0,
                piotroski_fscore_raw=6,
            )
        )
        session.commit()
        session.close()

        seen = {"called": False}

        def fake_latest_scoring_date():
            seen["called"] = True
            return date(2026, 3, 19)

        monkeypatch.setattr(data_access, "get_latest_scoring_date", fake_latest_scoring_date)
        df = get_scoring_results()

        assert seen["called"] is True
        assert list(df["ticker"]) == ["SKOR1"]
        assert list(df["scoring_date"]) == [date(2026, 3, 19)]

    def test_get_scoring_results_includes_unscored_companies_when_alpha_view_is_off(
        self,
        dashboard_engine,
        monkeypatch,
    ):
        """The full dashboard view should include companies lacking scores."""
        session = _session_for(dashboard_engine)
        scored = Company(
            ticker="SKOR1",
            name="Scored Co",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            is_active=True,
        )
        unscored = Company(
            ticker="DEAD1",
            name="Inactive Co",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            is_active=False,
        )
        session.add_all([scored, unscored])
        session.flush()
        scored_id = scored.id
        session.add(
            ScoringResult(
                company_id=scored.id,
                scoring_date=date(2026, 3, 19),
                composite_alpha=88.0,
                risk_tier="LOW",
                data_completeness=80.0,
                piotroski_fscore_raw=6,
            )
        )
        session.commit()
        session.close()

        monkeypatch.setattr(
            data_access,
            "get_latest_scoring_date",
            lambda: date(2026, 3, 19),
        )
        df = get_scoring_results(alpha_eligible_only=False)

        assert list(df["ticker"]) == ["SKOR1", "DEAD1"]
        assert df.loc[df["ticker"] == "DEAD1", "has_score"].item() is False
        assert df.loc[df["ticker"] == "DEAD1", "is_active"].item() is False
        assert pd.isna(df.loc[df["ticker"] == "DEAD1", "alpha"].item())

    def test_get_all_tickers_includes_inactive_companies(self, dashboard_engine):
        """Ticker selector should allow opening inactive/delisted names too."""
        session = _session_for(dashboard_engine)
        session.add_all(
            [
                Company(ticker="ACTV1", name="Active One", is_active=True),
                Company(ticker="PASS1", name="Passive One", is_active=False),
            ]
        )
        session.commit()
        session.close()

        assert get_all_tickers() == ["ACTV1", "PASS1"]

    def test_get_alpha_eligible_company_ids_returns_investable_alpha_universe(
        self,
        dashboard_engine,
    ):
        """Universe helper should cache the ALPHA-eligible company ids for a date."""
        scoring_date = date(2026, 3, 19)
        session = _session_for(dashboard_engine)

        eligible = Company(
            ticker="GOOD1",
            name="Eligible Co",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=35.0,
            is_active=True,
        )
        ineligible = Company(
            ticker="BADF1",
            name="Ineligible Co",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=10.0,
            is_active=True,
        )
        session.add_all([eligible, ineligible])
        session.flush()
        eligible_id = eligible.id

        session.add_all(
            [
                DailyPrice(
                    company_id=eligible.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                ),
                DailyPrice(
                    company_id=ineligible.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                ),
                ScoringResult(
                    company_id=eligible.id,
                    scoring_date=scoring_date,
                    risk_tier="LOW",
                    data_completeness=80.0,
                    piotroski_fscore_raw=6,
                ),
                ScoringResult(
                    company_id=ineligible.id,
                    scoring_date=scoring_date,
                    risk_tier="LOW",
                    data_completeness=80.0,
                    piotroski_fscore_raw=6,
                ),
            ]
        )
        session.commit()
        session.close()

        assert get_alpha_eligible_company_ids(scoring_date) == {eligible_id}

    def test_get_alpha_eligibility_reasons_flags_free_float_failures(
        self,
        dashboard_engine,
    ):
        """Diagnostics should expose the binding ALPHA failure reason."""
        scoring_date = date(2026, 3, 19)
        session = _session_for(dashboard_engine)

        company = Company(
            ticker="LOWFF",
            name="Low Float Co",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=10.0,
            is_active=True,
        )
        session.add(company)
        session.flush()
        company_id = company.id

        session.add_all(
            [
                DailyPrice(
                    company_id=company.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                ),
                ScoringResult(
                    company_id=company.id,
                    scoring_date=scoring_date,
                    risk_tier="LOW",
                    data_completeness=80.0,
                    piotroski_fscore_raw=6,
                ),
            ]
        )
        session.commit()
        session.close()

        reasons = get_alpha_eligibility_reasons(scoring_date)

        assert reasons[company_id] == "Halka aciklik <%25"

    def test_get_alpha_eligibility_reasons_flags_latest_loss_and_negative_owner_earnings(
        self,
        dashboard_engine,
    ):
        """Latest-period loss plus negative owner earnings should block ALPHA Core."""
        scoring_date = date(2026, 3, 19)
        session = _session_for(dashboard_engine)

        company = Company(
            ticker="LOSS1",
            name="Loss Maker",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=35.0,
            is_active=True,
        )
        session.add(company)
        session.flush()
        company_id = company.id

        session.add_all(
            [
                DailyPrice(
                    company_id=company.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                ),
                AdjustedMetric(
                    company_id=company.id,
                    period_end=date(2025, 12, 31),
                    adjusted_net_income=-50_000_000.0,
                    owner_earnings=-10_000_000.0,
                    eps_adjusted=-1.2,
                ),
                ScoringResult(
                    company_id=company.id,
                    scoring_date=scoring_date,
                    composite_alpha=95.0,
                    risk_tier="LOW",
                    data_completeness=90.0,
                    piotroski_fscore_raw=6,
                    piotroski_fscore=66.0,
                ),
            ]
        )
        session.commit()
        session.close()

        reasons = get_alpha_eligibility_reasons(scoring_date)
        df = get_scoring_results(scoring_date=scoring_date, alpha_eligible_only=False)

        assert reasons[company_id] == "Son donem zarar; Owner earnings <= 0"
        row = df.loc[df["ticker"] == "LOSS1"].iloc[0]
        assert bool(row["alpha_core_eligible"]) is False
        assert row["alpha_primary_blocker"] == "Son donem zarar"
        assert row["alpha_reason"] == "Son donem zarar"
        assert "Owner earnings <= 0" in row["alpha_all_blockers"]
        assert row["alpha_research_bucket"] == "Excluded"

    def test_get_scoring_results_exposes_financial_piotroski_gap_in_alpha_reason(
        self,
        dashboard_engine,
    ):
        """Full view should classify non-operating alpha names as research only."""
        scoring_date = date(2026, 3, 19)
        session = _session_for(dashboard_engine)

        company = Company(
            ticker="FINA1",
            name="Finance One",
            company_type="FINANCIAL",
            sector_bist="Financial",
            sector_custom="Financial",
            free_float_pct=35.0,
            is_active=True,
        )
        session.add(company)
        session.flush()
        company_id = company.id

        session.add_all(
            [
                DailyPrice(
                    company_id=company.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                ),
                ScoringResult(
                    company_id=company.id,
                    scoring_date=scoring_date,
                    model_used="FINANCIAL",
                    composite_alpha=88.0,
                    banking_composite=85.0,
                    risk_tier="LOW",
                    data_completeness=100.0,
                    momentum_score=70.0,
                    technical_score=75.0,
                ),
            ]
        )
        session.commit()
        session.close()

        df = get_scoring_results(scoring_date=scoring_date, alpha_eligible_only=False)

        row = df.loc[df["ticker"] == "FINA1"].iloc[0]
        assert bool(row["alpha_eligible"]) is False
        assert row["alpha_reason"] == "Research only"
        assert row["alpha_primary_blocker"] == "ALPHA Core sadece OPERATING"
        assert row["alpha_research_bucket"] == "Non-Core Research"
        assert "Piotroski yok (FINANCIAL modeli)" in row["alpha_all_blockers"]

    def test_get_scoring_results_adds_model_specific_ranking_fields(
        self,
        dashboard_engine,
    ):
        """Non-operating types should expose native and fallback ranking metadata."""
        scoring_date = date(2026, 3, 19)
        session = _session_for(dashboard_engine)

        bank_native = Company(
            ticker="BANK1",
            name="Bank Native",
            company_type="BANK",
            sector_bist="Bank",
            sector_custom="Bank",
            free_float_pct=35.0,
            is_active=True,
        )
        bank_fallback = Company(
            ticker="BANK2",
            name="Bank Fallback",
            company_type="BANK",
            sector_bist="Bank",
            sector_custom="Bank",
            free_float_pct=35.0,
            is_active=True,
        )
        insurer = Company(
            ticker="INSR1",
            name="Insurance Fallback",
            company_type="INSURANCE",
            sector_bist="Insurance",
            sector_custom="Insurance",
            free_float_pct=35.0,
            is_active=True,
        )
        session.add_all([bank_native, bank_fallback, insurer])
        session.flush()

        for company in (bank_native, bank_fallback, insurer):
            session.add(
                DailyPrice(
                    company_id=company.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                )
            )

        session.add_all(
            [
                ScoringResult(
                    company_id=bank_native.id,
                    scoring_date=scoring_date,
                    model_used="BANK",
                    banking_composite=82.0,
                    composite_alpha=70.0,
                    risk_tier="LOW",
                    data_completeness=85.0,
                ),
                ScoringResult(
                    company_id=bank_fallback.id,
                    scoring_date=scoring_date,
                    model_used="BANK",
                    momentum_score=40.0,
                    technical_score=80.0,
                    risk_tier="LOW",
                    data_completeness=35.0,
                ),
                ScoringResult(
                    company_id=insurer.id,
                    scoring_date=scoring_date,
                    model_used="INSURANCE",
                    composite_alpha=77.0,
                    momentum_score=60.0,
                    technical_score=65.0,
                    risk_tier="LOW",
                    data_completeness=66.0,
                ),
            ]
        )
        session.commit()
        session.close()

        df = get_scoring_results(scoring_date=scoring_date, alpha_eligible_only=False)

        bank_native_row = df.loc[df["ticker"] == "BANK1"].iloc[0]
        assert bank_native_row["model_family"] == "Financials"
        assert bank_native_row["model_score"] == pytest.approx(82.0)
        assert bank_native_row["ranking_score"] == pytest.approx(82.0)
        assert bank_native_row["ranking_source"] == "Banking Model"
        assert bool(bank_native_row["has_native_model_score"]) is True
        assert bool(bank_native_row["ranking_uses_fallback"]) is False

        bank_fallback_row = df.loc[df["ticker"] == "BANK2"].iloc[0]
        assert pd.isna(bank_fallback_row["model_score"])
        assert bank_fallback_row["ranking_source"] == "Market Fallback"
        assert bank_fallback_row["ranking_score"] == pytest.approx(57.5)
        assert bool(bank_fallback_row["ranking_uses_fallback"]) is True

        insurer_row = df.loc[df["ticker"] == "INSR1"].iloc[0]
        assert pd.isna(insurer_row["model_score"])
        assert insurer_row["ranking_source"] == "Alpha Fallback"
        assert insurer_row["ranking_score"] == pytest.approx(77.0)
        assert bool(insurer_row["ranking_uses_fallback"]) is True

        bank_rows = (
            df[df["type"] == "BANK"]
            .sort_values("type_rank")
            .reset_index(drop=True)
        )
        assert list(bank_rows["ticker"]) == ["BANK1", "BANK2"]
        assert list(bank_rows["type_rank"]) == pytest.approx([1.0, 2.0])

    def test_get_scoring_results_builds_alpha_x_cross_type_universe(
        self,
        dashboard_engine,
    ):
        """ALPHA X should keep operating as anchor while admitting only mature non-operating signals."""
        scoring_date = date(2026, 3, 19)
        session = _session_for(dashboard_engine)

        operating = Company(
            ticker="OPER1",
            name="Operating One",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=35.0,
            is_active=True,
        )
        reit = Company(
            ticker="REIT1",
            name="REIT One",
            company_type="REIT",
            sector_bist="REIT",
            sector_custom="REIT",
            free_float_pct=35.0,
            is_active=True,
        )
        holding = Company(
            ticker="HOLD1",
            name="Holding One",
            company_type="HOLDING",
            sector_bist="Holding",
            sector_custom="Holding",
            free_float_pct=35.0,
            is_active=True,
        )
        insurer = Company(
            ticker="INSX1",
            name="Insurance Shadow",
            company_type="INSURANCE",
            sector_bist="Insurance",
            sector_custom="Insurance",
            free_float_pct=35.0,
            is_active=True,
        )
        bank = Company(
            ticker="BANKX",
            name="Bank Excluded",
            company_type="BANK",
            sector_bist="Bank",
            sector_custom="Bank",
            free_float_pct=35.0,
            is_active=True,
        )
        session.add_all([operating, reit, holding, insurer, bank])
        session.flush()

        for company in (operating, reit, holding, insurer, bank):
            session.add(
                DailyPrice(
                    company_id=company.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                )
            )

        session.add_all(
            [
                ScoringResult(
                    company_id=operating.id,
                    scoring_date=scoring_date,
                    model_used="OPERATING",
                    composite_alpha=91.0,
                    risk_tier="LOW",
                    data_completeness=85.0,
                    piotroski_fscore_raw=6,
                    piotroski_fscore=66.0,
                    momentum_score=70.0,
                    technical_score=72.0,
                ),
                ScoringResult(
                    company_id=reit.id,
                    scoring_date=scoring_date,
                    model_used="REIT",
                    composite_alpha=80.0,
                    reit_composite=88.0,
                    risk_tier="MEDIUM",
                    data_completeness=100.0,
                    momentum_score=60.0,
                    technical_score=55.0,
                ),
                ScoringResult(
                    company_id=holding.id,
                    scoring_date=scoring_date,
                    model_used="HOLDING",
                    composite_alpha=45.0,
                    holding_composite=82.0,
                    risk_tier="MEDIUM",
                    data_completeness=100.0,
                    momentum_score=40.0,
                    technical_score=35.0,
                ),
                ScoringResult(
                    company_id=insurer.id,
                    scoring_date=scoring_date,
                    model_used="INSURANCE",
                    composite_alpha=85.0,
                    risk_tier="LOW",
                    data_completeness=66.0,
                    momentum_score=70.0,
                    technical_score=75.0,
                ),
                ScoringResult(
                    company_id=bank.id,
                    scoring_date=scoring_date,
                    model_used="BANK",
                    composite_alpha=92.0,
                    banking_composite=99.0,
                    risk_tier="LOW",
                    data_completeness=90.0,
                    momentum_score=80.0,
                    technical_score=82.0,
                ),
            ]
        )
        session.commit()
        session.close()

        df = get_scoring_results(scoring_date=scoring_date, alpha_eligible_only=False)

        operating_row = df.loc[df["ticker"] == "OPER1"].iloc[0]
        assert bool(operating_row["alpha_x_eligible"]) is True
        assert operating_row["alpha_x_reason"] == "ALPHA X Uygun"

        reit_row = df.loc[df["ticker"] == "REIT1"].iloc[0]
        assert bool(reit_row["alpha_x_eligible"]) is True
        assert reit_row["alpha_x_reason"] == "ALPHA X Uygun"
        assert reit_row["alpha_x_score"] < reit_row["model_score"]
        assert reit_row["alpha_x_confidence"] < 1.0

        holding_row = df.loc[df["ticker"] == "HOLD1"].iloc[0]
        assert bool(holding_row["alpha_x_eligible"]) is True
        assert holding_row["alpha_x_reason"] == "ALPHA X Uygun"

        insurer_row = df.loc[df["ticker"] == "INSX1"].iloc[0]
        assert bool(insurer_row["alpha_x_eligible"]) is False
        assert insurer_row["alpha_x_bucket"] == "Native Shadow"
        assert insurer_row["alpha_x_reason"] == "Native model yok"

        bank_row = df.loc[df["ticker"] == "BANKX"].iloc[0]
        assert bool(bank_row["alpha_x_eligible"]) is False
        assert bank_row["alpha_x_bucket"] == "Type Excluded"
        assert bank_row["alpha_x_reason"] == "BANK ALPHA X disi"

        eligible = (
            df[df["alpha_x_eligible"]]
            .sort_values("alpha_x_rank")
            .reset_index(drop=True)
        )
        assert list(eligible["ticker"]) == ["OPER1", "REIT1", "HOLD1"]
        assert list(eligible["alpha_x_rank"]) == pytest.approx([1.0, 2.0, 3.0])

    def test_get_scoring_results_exposes_operating_debug_groups(
        self,
        dashboard_engine,
    ):
        """Operating rows should expose grouped factor diagnostics for debugging odd ranks."""
        scoring_date = date(2026, 3, 19)
        session = _session_for(dashboard_engine)

        company = Company(
            ticker="DBG01",
            name="Debug Alpha",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=35.0,
            is_active=True,
        )
        session.add(company)
        session.flush()

        session.add(
            DailyPrice(
                company_id=company.id,
                date=scoring_date,
                close=100.0,
                adjusted_close=100.0,
                high=101.0,
                low=99.0,
                volume=200_000,
                source="YAHOO_TEST",
            )
        )
        session.add(
            ScoringResult(
                company_id=company.id,
                scoring_date=scoring_date,
                model_used="OPERATING",
                composite_alpha=88.0,
                buffett_score=40.0,
                graham_score=None,
                dcf_margin_of_safety_pct=None,
                piotroski_fscore=90.0,
                piotroski_fscore_raw=6,
                magic_formula_rank=80.0,
                lynch_peg_score=60.0,
                momentum_score=70.0,
                technical_score=50.0,
                risk_tier="LOW",
                data_completeness=83.33,
            )
        )
        session.commit()
        session.close()

        df = get_scoring_results(scoring_date=scoring_date, alpha_eligible_only=False)

        row = df.loc[df["ticker"] == "DBG01"].iloc[0]
        assert pd.isna(row["alpha_value_group"])
        assert row["alpha_growth_group"] == pytest.approx(70.0)
        assert row["alpha_missing_groups"] == "Value"
        assert row["alpha_x_delta"] == pytest.approx(row["alpha_x_score"] - row["alpha"])

    def test_get_scoring_results_flags_quality_shadow_candidates(
        self,
        dashboard_engine,
    ):
        """Operating names with raw Piotroski 4 should land in Quality Shadow."""
        scoring_date = date(2026, 3, 19)
        session = _session_for(dashboard_engine)

        company = Company(
            ticker="QSHDW",
            name="Quality Shadow",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=35.0,
            is_active=True,
        )
        session.add(company)
        session.flush()

        session.add_all(
            [
                DailyPrice(
                    company_id=company.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                ),
                ScoringResult(
                    company_id=company.id,
                    scoring_date=scoring_date,
                    composite_alpha=92.0,
                    risk_tier="LOW",
                    data_completeness=85.0,
                    piotroski_fscore_raw=4,
                    piotroski_fscore=44.0,
                    momentum_score=70.0,
                    technical_score=72.0,
                ),
            ]
        )
        session.commit()
        session.close()

        df = get_scoring_results(scoring_date=scoring_date, alpha_eligible_only=False)

        row = df.loc[df["ticker"] == "QSHDW"].iloc[0]
        assert bool(row["alpha_core_eligible"]) is False
        assert row["alpha_primary_blocker"] == "Piotroski < 5"
        assert row["alpha_research_bucket"] == "Quality Shadow"
        assert bool(row["alpha_relaxed_p4_eligible"]) is True
        assert row["alpha_reason"] == "Quality Shadow adayi"

    def test_get_scoring_results_flags_free_float_shadow_candidates(
        self,
        dashboard_engine,
    ):
        """Operating names blocked only by free-float should land in Free-Float Shadow."""
        scoring_date = date(2026, 3, 19)
        session = _session_for(dashboard_engine)

        company = Company(
            ticker="FSHDW",
            name="Float Shadow",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=20.0,
            is_active=True,
        )
        session.add(company)
        session.flush()

        session.add_all(
            [
                DailyPrice(
                    company_id=company.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                ),
                ScoringResult(
                    company_id=company.id,
                    scoring_date=scoring_date,
                    composite_alpha=91.0,
                    risk_tier="LOW",
                    data_completeness=90.0,
                    piotroski_fscore_raw=6,
                    piotroski_fscore=66.0,
                    momentum_score=68.0,
                    technical_score=70.0,
                ),
            ]
        )
        session.commit()
        session.close()

        df = get_scoring_results(scoring_date=scoring_date, alpha_eligible_only=False)

        row = df.loc[df["ticker"] == "FSHDW"].iloc[0]
        assert bool(row["alpha_core_eligible"]) is False
        assert row["alpha_primary_blocker"] == "Halka aciklik <%25"
        assert row["alpha_research_bucket"] == "Free-Float Shadow"
        assert bool(row["alpha_relaxed_p4_eligible"]) is False
        assert row["alpha_reason"] == "Free-Float Shadow adayi"

    def test_get_scoring_results_uses_exact_snapshot_for_missing_scores(
        self,
        dashboard_engine,
    ):
        """Older scores should not leak into an exact-date dashboard snapshot."""
        session = _session_for(dashboard_engine)
        company = Company(
            ticker="LATE1",
            name="Late Snapshot",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=35.0,
            is_active=True,
        )
        session.add(company)
        session.flush()

        session.add(
            ScoringResult(
                company_id=company.id,
                scoring_date=date(2026, 3, 18),
                composite_alpha=90.0,
                risk_tier="LOW",
                data_completeness=85.0,
                piotroski_fscore_raw=6,
            )
        )
        session.commit()
        session.close()

        df = get_scoring_results(
            scoring_date=date(2026, 3, 19),
            alpha_eligible_only=False,
        )

        row = df.loc[df["ticker"] == "LATE1"].iloc[0]
        assert bool(row["has_score"]) is False
        assert row["alpha_reason"] == "Data-Unscorable"
        assert row["alpha_primary_blocker"] == "Skor snapshot yok"
        assert row["alpha_research_bucket"] == "Data-Unscorable"
        assert row["alpha_snapshot_streak"] == 0

    def test_get_scoring_results_reports_snapshot_streak_for_same_bucket(
        self,
        dashboard_engine,
    ):
        """Snapshot streak should count consecutive dates with the same bucket."""
        session = _session_for(dashboard_engine)
        company = Company(
            ticker="STREAK",
            name="Streak Co",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=35.0,
            is_active=True,
        )
        session.add(company)
        session.flush()

        for scoring_date in (date(2026, 3, 18), date(2026, 3, 19)):
            session.add(
                DailyPrice(
                    company_id=company.id,
                    date=scoring_date,
                    close=100.0,
                    adjusted_close=100.0,
                    high=101.0,
                    low=99.0,
                    volume=200_000,
                    source="YAHOO_TEST",
                )
            )
            session.add(
                ScoringResult(
                    company_id=company.id,
                    scoring_date=scoring_date,
                    composite_alpha=93.0,
                    risk_tier="LOW",
                    data_completeness=88.0,
                    piotroski_fscore_raw=4,
                    piotroski_fscore=44.0,
                    momentum_score=70.0,
                    technical_score=71.0,
                )
            )

        session.commit()
        session.close()

        df = get_scoring_results(
            scoring_date=date(2026, 3, 19),
            alpha_eligible_only=False,
        )

        row = df.loc[df["ticker"] == "STREAK"].iloc[0]
        assert row["alpha_research_bucket"] == "Quality Shadow"
        assert row["alpha_snapshot_streak"] == 2

    def test_get_open_positions_preserves_output_with_batched_latest_prices(
        self,
        dashboard_engine,
    ):
        """Open positions should still show the same prices and P&L after batching."""
        today = date.today()
        selection_date = today - timedelta(days=5)
        session = _session_for(dashboard_engine)

        company_a = Company(
            ticker="PICK1",
            name="Pick One",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            is_active=True,
        )
        company_b = Company(
            ticker="PICK2",
            name="Pick Two",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            is_active=True,
        )
        session.add_all([company_a, company_b])
        session.flush()

        session.add_all(
            [
                DailyPrice(
                    company_id=company_a.id,
                    date=today,
                    close=109.0,
                    adjusted_close=110.0,
                    high=111.0,
                    low=108.0,
                    volume=1000,
                ),
                DailyPrice(
                    company_id=company_b.id,
                    date=today,
                    close=180.0,
                    adjusted_close=None,
                    high=181.0,
                    low=179.0,
                    volume=1000,
                ),
                PortfolioSelection(
                    portfolio="ALPHA",
                    selection_date=selection_date,
                    company_id=company_a.id,
                    entry_price=100.0,
                    composite_score=90.0,
                ),
                PortfolioSelection(
                    portfolio="ALPHA",
                    selection_date=selection_date,
                    company_id=company_b.id,
                    entry_price=200.0,
                    composite_score=80.0,
                ),
            ]
        )
        session.commit()
        session.close()

        df = get_open_positions()

        assert list(df["ticker"]) == ["PICK1", "PICK2"]
        assert list(df["current_price"]) == [110.0, 180.0]
        assert list(df["entry_price"]) == [100.0, 200.0]
        assert list(df["days_held"]) == [5, 5]
        assert df["pnl_pct"].tolist() == pytest.approx([10.0, -10.0])
