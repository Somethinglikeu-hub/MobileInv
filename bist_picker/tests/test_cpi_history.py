"""Tests for CPI history storage and the inflation-adjusted growth path.

Verifies the 2026-04-30 audit fix: `MetricsCalculator._get_cpi_series` now
reads CPI index levels from the dedicated `cpi_history` table (populated
from TCMB TP.FG.J0). Previously it read YoY rates from
`macro_regime.cpi_yoy_pct` and fed them into `calculate_real_growth`,
which expects index levels — producing meaningless `real_eps_growth_pct`
values. These tests pin the fixed behavior.
"""

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.cleaning.financial_prep import MetricsCalculator
from bist_picker.cleaning.inflation import InflationAdjuster
from bist_picker.db.schema import Base, CpiHistory, MacroRegime


@pytest.fixture
def session():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    SessionFactory = sessionmaker(bind=eng)
    sess = SessionFactory()
    yield sess
    sess.close()


def _seed_cpi_history(session, points: list[tuple[date, float]]) -> None:
    for d, v in points:
        session.add(CpiHistory(date=d, cpi_index=v))
    session.flush()


class TestGetCpiSeries:
    def test_reads_cpi_history_levels(self, session):
        _seed_cpi_history(session, [
            (date(2022, 12, 31), 1000.0),
            (date(2023, 12, 31), 1650.0),
            (date(2024, 12, 31), 2300.0),
        ])
        calc = MetricsCalculator(session)
        series = calc._get_cpi_series()

        assert series is not None
        assert len(series) == 3
        # Stored values are CPI INDEX LEVELS, not YoY rates.
        # If this comes back as 0.65 instead of 1650 we've regressed.
        assert series.iloc[0] == pytest.approx(1000.0)
        assert series.iloc[1] == pytest.approx(1650.0)
        assert series.iloc[2] == pytest.approx(2300.0)

    def test_falls_back_to_macro_regime_when_history_empty(self, session):
        # Seed only the legacy field; CpiHistory is empty.
        session.add(MacroRegime(date=date(2024, 1, 1), cpi_yoy_pct=0.65))
        session.flush()

        calc = MetricsCalculator(session)
        series = calc._get_cpi_series()

        # Legacy fallback returns YoY rates as-is. Keeping this branch alive
        # is intentional so a stale DB doesn't crash before fetch_macro
        # repopulates cpi_history; a warning is logged.
        assert series is not None
        assert len(series) == 1
        assert series.iloc[0] == pytest.approx(0.65)

    def test_returns_none_when_both_sources_empty(self, session):
        calc = MetricsCalculator(session)
        assert calc._get_cpi_series() is None

    def test_real_growth_with_cpi_history_is_correct(self, session):
        # Real ground truth: 65% inflation between 2023-2024 (CPI 1000 -> 1650)
        # and a stock posting 80% nominal EPS growth.
        # Real growth = 1.80 / 1.65 - 1 = 0.0909 (9.09% real).
        _seed_cpi_history(session, [
            (date(2023, 12, 31), 1000.0),
            (date(2024, 12, 31), 1650.0),
        ])
        calc = MetricsCalculator(session)
        cpi = calc._get_cpi_series()

        result = InflationAdjuster.calculate_real_growth(
            current=180.0,
            previous=100.0,
            current_date=date(2024, 12, 31),
            previous_date=date(2023, 12, 31),
            cpi_series=cpi,
        )
        assert result == pytest.approx(0.0909, rel=1e-2)

    def test_real_growth_breaks_under_legacy_fallback(self, session):
        # Pin the documented bug: when only the legacy YoY-rate path is
        # available, real_growth math is meaningless. We assert the wrong
        # answer to make it crystal clear that the fallback is only there
        # to avoid crashing -- a stale DB will produce nonsense until
        # fetch_macro populates cpi_history.
        session.add(MacroRegime(date=date(2023, 12, 31), cpi_yoy_pct=0.50))
        session.add(MacroRegime(date=date(2024, 12, 31), cpi_yoy_pct=0.65))
        session.flush()

        calc = MetricsCalculator(session)
        cpi = calc._get_cpi_series()

        result = InflationAdjuster.calculate_real_growth(
            current=180.0,
            previous=100.0,
            current_date=date(2024, 12, 31),
            previous_date=date(2023, 12, 31),
            cpi_series=cpi,
        )
        # The formula does (cpi_current/cpi_previous - 1) on the YoY rates
        # 0.65/0.50 = 1.30 -> "inflation" = 0.30, then real = 1.80/1.30 - 1 = 0.385.
        # That's not right -- actual real growth depends on the unknown CPI
        # index levels behind the YoY rates. Test asserts the broken value
        # so future refactors don't accidentally "fix" the fallback in a way
        # that hides the warning we want users to see.
        assert result == pytest.approx(0.385, rel=1e-2)
