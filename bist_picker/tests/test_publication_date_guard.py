"""Tests for the AdjustedMetric point-in-time guard (audit CRITICAL #1).

Verifies the centralized ``_adjusted_metric_pit_filter`` SQLAlchemy filter:
  * Rows with ``publication_date <= scoring_date`` are visible.
  * Rows with ``publication_date > scoring_date`` are filtered out.
  * Legacy rows (``publication_date IS NULL``) fall back to the 76-day
    period_end heuristic so years of historical data don't disappear.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.db.schema import AdjustedMetric, Base, Company
from bist_picker.scoring.context import (
    _adjusted_metric_pit_filter,
    _LEGACY_PUBLICATION_LAG_DAYS,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    sess = SessionLocal()
    yield sess
    sess.close()


def _add_company(session, ticker: str = "TEST") -> int:
    c = Company(ticker=ticker, name=f"{ticker} A.S.",
                company_type="OPERATING", is_active=True)
    session.add(c); session.flush()
    return c.id


def _add_metric(
    session,
    cid: int,
    period_end: date,
    publication_date: date | None = None,
):
    """Add a minimal AdjustedMetric row with the given dates."""
    m = AdjustedMetric(
        company_id=cid,
        period_end=period_end,
        publication_date=publication_date,
        adjusted_net_income=1.0,
        eps_adjusted=1.0,
    )
    session.add(m); session.flush()
    return m


class TestAdjustedMetricPitFilter:
    """Confirm the filter ships rows that were knowable on the scoring date."""

    def test_filed_before_scoring_date_is_visible(self, session):
        """Row with publication_date <= scoring_date passes the filter."""
        cid = _add_company(session)
        _add_metric(
            session, cid,
            period_end=date(2024, 12, 31),
            publication_date=date(2025, 3, 15),  # filed on time
        )
        session.commit()

        scoring_date = date(2025, 4, 1)
        rows = (
            session.query(AdjustedMetric)
            .filter(_adjusted_metric_pit_filter(scoring_date))
            .all()
        )
        assert len(rows) == 1, "Filed row must be visible after publication_date"

    def test_filed_after_scoring_date_is_hidden(self, session):
        """Row with publication_date > scoring_date is invisible (no leak)."""
        cid = _add_company(session)
        _add_metric(
            session, cid,
            period_end=date(2024, 12, 31),
            publication_date=date(2025, 5, 20),  # late filer
        )
        session.commit()

        # Scoring "as of" April 1 — the late Q4-2024 filing isn't out yet.
        scoring_date = date(2025, 4, 1)
        rows = (
            session.query(AdjustedMetric)
            .filter(_adjusted_metric_pit_filter(scoring_date))
            .all()
        )
        assert rows == [], (
            "Row filed AFTER scoring_date must not leak into the past."
        )

    def test_legacy_row_uses_76_day_heuristic(self, session):
        """publication_date IS NULL → falls back to period_end + 76 days."""
        cid = _add_company(session)
        scoring_date = date(2025, 4, 1)
        # period_end well over 76 days before scoring → visible by heuristic.
        _add_metric(
            session, cid,
            period_end=scoring_date - timedelta(days=120),
            publication_date=None,
        )
        # period_end too recent → filter says "could not have been knowable".
        _add_metric(
            session, cid,
            period_end=scoring_date - timedelta(days=30),
            publication_date=None,
        )
        session.commit()

        rows = (
            session.query(AdjustedMetric)
            .filter(_adjusted_metric_pit_filter(scoring_date))
            .order_by(AdjustedMetric.period_end)
            .all()
        )
        assert len(rows) == 1
        assert rows[0].period_end == scoring_date - timedelta(days=120)

    def test_mixed_legacy_and_new_rows_are_combined(self, session):
        """Both code paths coexist on the same query.

        Three different companies × one annual period each, so the
        ``(company_id, period_end)`` unique constraint isn't tripped.
        """
        scoring_date = date(2025, 4, 1)

        # Company A: legacy row (no pub date), old period — passes heuristic.
        cid_a = _add_company(session, "AAA")
        _add_metric(
            session, cid_a,
            period_end=date(2023, 12, 31),
            publication_date=None,
        )
        # Company B: new row, on-time filing — passes strict path.
        cid_b = _add_company(session, "BBB")
        _add_metric(
            session, cid_b,
            period_end=date(2024, 12, 31),
            publication_date=date(2025, 3, 10),
        )
        # Company C: new row, late filing (after scoring_date) — must be hidden.
        cid_c = _add_company(session, "CCC")
        _add_metric(
            session, cid_c,
            period_end=date(2024, 12, 31),
            publication_date=date(2025, 5, 5),
        )
        session.commit()

        rows = (
            session.query(AdjustedMetric)
            .filter(_adjusted_metric_pit_filter(scoring_date))
            .order_by(AdjustedMetric.period_end)
            .all()
        )
        # 2 of 3 should pass: AAA (legacy heuristic) and BBB (strict path).
        # CCC is filed AFTER scoring_date so it must not leak.
        assert len(rows) == 2
        cids = sorted(r.company_id for r in rows)
        assert cids == sorted([cid_a, cid_b])

    def test_legacy_lag_constant_is_76_days(self):
        """Document the heuristic lag for future maintainers."""
        assert _LEGACY_PUBLICATION_LAG_DAYS == 76
