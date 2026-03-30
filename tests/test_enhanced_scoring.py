"""Tests for Phase 2 scoring factors and enhanced composer.

Tests use in-memory SQLite to verify:
  - EventScorer impact calculation and scoring
  - MacroNowcastScorer component weighting
  - InsiderScorer cluster and drawdown detection
  - EnhancedComposer weighted average and blending logic
"""

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bist_picker.db.schema import Base, Company, KapEvent, MacroNowcast


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def db_session():
    """Create an in-memory SQLite session with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Seed a test company
    company = Company(
        id=1,
        ticker="TEST",
        name="Test Company",
        is_active=True,
        sector_bist="XBANK",
    )
    session.add(company)
    session.commit()

    yield session
    session.close()


# ── EventScorer Tests ────────────────────────────────────────────────────────

class TestEventScorer:
    """Test KAP event scoring factor."""

    def test_no_events_returns_none(self, db_session):
        """Company with no events returns None."""
        from bist_picker.scoring.factors.event_score import EventScorer

        scorer = EventScorer()
        result = scorer.score(1, db_session, date.today())
        assert result is None

    def test_positive_event_scores_above_50(self, db_session):
        """Positive sentiment events should produce scores above 50."""
        from bist_picker.scoring.factors.event_score import EventScorer

        # Add a positive event
        event = KapEvent(
            company_id=1,
            disclosure_date=date.today() - timedelta(days=5),
            event_type="NEW_CONTRACT",
            sentiment_score=0.8,
            monetary_value=100_000_000,
            currency="TRY",
            confidence=0.9,
            raw_text_hash="abc123" + "0" * 58,
        )
        db_session.add(event)
        db_session.commit()

        scorer = EventScorer()
        result = scorer.score(1, db_session, date.today())

        assert result is not None
        assert result["event_score"] > 50.0
        assert result["event_count"] == 1
        assert result["top_event_type"] == "NEW_CONTRACT"

    def test_negative_event_scores_below_50(self, db_session):
        """Negative sentiment events should produce scores below 50."""
        from bist_picker.scoring.factors.event_score import EventScorer

        event = KapEvent(
            company_id=1,
            disclosure_date=date.today() - timedelta(days=3),
            event_type="LAWSUIT",
            sentiment_score=-0.9,
            confidence=0.85,
            raw_text_hash="def456" + "0" * 58,
        )
        db_session.add(event)
        db_session.commit()

        scorer = EventScorer()
        result = scorer.score(1, db_session, date.today())

        assert result is not None
        assert result["event_score"] < 50.0

    def test_old_events_decay(self, db_session):
        """Events older than lookback window should be excluded."""
        from bist_picker.scoring.factors.event_score import EventScorer

        event = KapEvent(
            company_id=1,
            disclosure_date=date.today() - timedelta(days=100),
            event_type="NEW_CONTRACT",
            sentiment_score=0.8,
            confidence=0.9,
            raw_text_hash="old123" + "0" * 58,
        )
        db_session.add(event)
        db_session.commit()

        scorer = EventScorer()
        result = scorer.score(1, db_session, date.today())

        assert result is None  # Beyond 90-day lookback


# ── MacroNowcastScorer Tests ────────────────────────────────────────────────

class TestMacroNowcastScorer:
    """Test macro nowcast scoring factor."""

    def test_no_data_returns_none(self, db_session):
        """No macro nowcast data returns None."""
        from bist_picker.scoring.factors.macro_nowcast_score import MacroNowcastScorer

        scorer = MacroNowcastScorer()
        result = scorer.score_macro(db_session, date.today())
        assert result is None

    def test_bullish_scores_high(self, db_session):
        """Bullish BONC with positive credit card growth → high score."""
        from bist_picker.scoring.factors.macro_nowcast_score import MacroNowcastScorer

        nowcast = MacroNowcast(
            date=date.today() - timedelta(days=1),
            bonc_index=105.0,
            bonc_change_mom=2.0,
            bonc_trend="RISING",
            credit_card_total_change_pct=8.0,
            llm_macro_sentiment="BULLISH",
            llm_confidence=0.85,
        )
        db_session.add(nowcast)
        db_session.commit()

        scorer = MacroNowcastScorer()
        result = scorer.score_macro(db_session, date.today())

        assert result is not None
        assert result["macro_nowcast_score"] > 70.0
        assert result["bonc_trend"] == "RISING"

    def test_bearish_scores_low(self, db_session):
        """Bearish BONC with negative spending → low score."""
        from bist_picker.scoring.factors.macro_nowcast_score import MacroNowcastScorer

        nowcast = MacroNowcast(
            date=date.today() - timedelta(days=1),
            bonc_index=95.0,
            bonc_change_mom=-3.0,
            bonc_trend="FALLING",
            credit_card_total_change_pct=-5.0,
            llm_macro_sentiment="BEARISH",
            llm_confidence=0.8,
        )
        db_session.add(nowcast)
        db_session.commit()

        scorer = MacroNowcastScorer()
        result = scorer.score_macro(db_session, date.today())

        assert result is not None
        assert result["macro_nowcast_score"] < 30.0

    def test_sector_adjustment(self, db_session):
        """Sector-specific impacts should adjust company score."""
        from bist_picker.scoring.factors.macro_nowcast_score import MacroNowcastScorer

        nowcast = MacroNowcast(
            date=date.today() - timedelta(days=1),
            bonc_index=100.0,
            bonc_trend="FLAT",
            llm_macro_sentiment="NEUTRAL",
            llm_confidence=0.7,
            sector_impacts_json=json.dumps({"XBANK": 0.5, "XUTEK": -0.3}),
        )
        db_session.add(nowcast)
        db_session.commit()

        scorer = MacroNowcastScorer()
        result = scorer.score_for_company(1, db_session, date.today())

        assert result is not None
        # Company sector is XBANK with +0.5 impact → should boost score
        assert result["sector_adjustment"] > 0


# ── EnhancedComposer Tests ──────────────────────────────────────────────────

class TestEnhancedComposerWeightedAverage:
    """Test the static weighted average method."""

    def test_all_factors_present(self):
        """All factors available → normal weighted average."""
        from bist_picker.scoring.enhanced_composer import EnhancedComposer

        weights = {
            "event_score": 0.35,
            "insider_cluster": 0.30,
            "macro_nowcast": 0.20,
            "analyst_tone": 0.15,
        }
        scores = {
            "event_score": 80.0,
            "insider_cluster": 60.0,
            "macro_nowcast": 70.0,
            "analyst_tone": 50.0,
        }

        result = EnhancedComposer._weighted_average(weights, scores)
        assert result is not None
        # 80*0.35 + 60*0.30 + 70*0.20 + 50*0.15 = 28+18+14+7.5 = 67.5
        assert abs(result - 67.5) < 0.1

    def test_missing_factor_redistributes(self):
        """Missing factor → weight redistributed to others."""
        from bist_picker.scoring.enhanced_composer import EnhancedComposer

        weights = {
            "event_score": 0.35,
            "insider_cluster": 0.30,
            "macro_nowcast": 0.20,
            "analyst_tone": 0.15,
        }
        scores = {
            "event_score": 80.0,
            "insider_cluster": 60.0,
            "macro_nowcast": None,  # Missing!
            "analyst_tone": None,   # Missing!
        }

        result = EnhancedComposer._weighted_average(weights, scores)
        assert result is not None
        # Only event (0.35) and insider (0.30) available
        # Normalized: event=0.35/0.65=0.538, insider=0.30/0.65=0.462
        expected = 80.0 * (0.35/0.65) + 60.0 * (0.30/0.65)
        assert abs(result - expected) < 0.1

    def test_all_missing_returns_none(self):
        """All factors missing → returns None."""
        from bist_picker.scoring.enhanced_composer import EnhancedComposer

        weights = {"event_score": 0.35, "insider_cluster": 0.30}
        scores = {"event_score": None, "insider_cluster": None}

        result = EnhancedComposer._weighted_average(weights, scores)
        assert result is None
