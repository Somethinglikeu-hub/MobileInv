"""Integration tests for technical scorer with real config loading.

Tests verify:
  - TechnicalScorer loads config from actual thresholds.yaml
  - All indicators can be calculated with real-world data patterns
  - Enhanced mode can be toggled via config
  - Backward compatibility with enhanced disabled
"""

from unittest.mock import MagicMock

from bist_picker.scoring.factors.technical import TechnicalScorer


# ── Helpers ────────────────────────────────────────────────────────────────────


def _session_with_mock_company():
    """Create a mock session with a company."""
    session = MagicMock()
    company = MagicMock()
    company.id = 1
    session.get.return_value = company

    # Mock query chain for DailyPrice
    q = MagicMock()
    session.query.return_value = q
    return session, q


def _create_price_rows(n_rows: int, pattern: str = "uptrend"):
    """Create mock price rows with specified pattern.

    Returns rows in DESCENDING order (newest first) to match DB query behavior.
    The scorer will reverse them internally to get oldest→newest for calculations.
    """
    rows = []
    for i in range(n_rows):
        r = MagicMock()
        # i=0 is newest, i=n_rows-1 is oldest (reversed index)
        rev_i = n_rows - 1 - i

        if pattern == "uptrend":
            # Uptrend: oldest (rev_i=n_rows-1) starts low, newest (rev_i=0) ends high
            r.close = 100.0 + rev_i * 0.3
            r.high = 100.5 + rev_i * 0.3
            r.low = 99.5 + rev_i * 0.3
            r.volume = 1_000_000
        elif pattern == "downtrend":
            # Downtrend: oldest starts high, newest ends low
            r.close = 200.0 - rev_i * 0.3
            r.high = 200.5 - rev_i * 0.3
            r.low = 199.5 - rev_i * 0.3
            r.volume = 1_000_000
        elif pattern == "sideways":
            # Sideways: oscillate around center
            r.close = 100.0 + (rev_i % 10 - 5) * 0.5
            r.high = 100.5 + (rev_i % 10 - 5) * 0.5
            r.low = 99.5 + (rev_i % 10 - 5) * 0.5
            r.volume = 1_000_000
        rows.append(r)
    return rows


# ── Integration tests ──────────────────────────────────────────────────────────


class TestTechnicalScorerIntegration:
    def test_loads_config_from_file(self):
        """TechnicalScorer loads config from thresholds.yaml."""
        # This will load the actual thresholds.yaml file
        scorer = TechnicalScorer()

        # Verify config loaded
        assert hasattr(scorer, "_enhanced_enabled")
        assert hasattr(scorer, "_weight_ma")
        assert hasattr(scorer, "_macd_ema_short")
        # Enhanced should be disabled by default in config
        assert scorer._enhanced_enabled is False

    def test_score_with_enhanced_disabled(self):
        """Verify scoring works with enhanced disabled (backward compatibility)."""
        scorer = TechnicalScorer()
        session, q = _session_with_mock_company()

        # Create 260 rows of uptrend data
        rows = _create_price_rows(260, pattern="uptrend")
        q.filter.return_value.order_by.return_value.limit.return_value.all.return_value = rows

        result = scorer.score(1, session)

        assert result is not None
        # Classic signals should be calculated
        assert "technical_score_classic" in result
        assert result["technical_score_classic"] is not None
        # Enhanced signals should be None
        assert result["technical_score_enhanced"] is None
        assert result["macd_value"] is None
        # Final score should equal classic
        assert result["technical_score"] == result["technical_score_classic"]

    def test_all_indicators_calculable(self):
        """Verify all 8 indicators can be calculated with sufficient data."""
        # Create scorer with enhanced enabled (override config)
        scorer = TechnicalScorer()
        scorer._enhanced_enabled = True  # Force enable for test

        session, q = _session_with_mock_company()

        # Create 260 rows (enough for all indicators including S/R)
        rows = _create_price_rows(260, pattern="uptrend")
        q.filter.return_value.order_by.return_value.limit.return_value.all.return_value = rows

        result = scorer.score(1, session)

        assert result is not None
        # Classic signals
        assert result["above_200ma"] is not None
        assert result["rsi_14"] is not None
        assert result["vol_ratio_20_60"] is not None
        # Enhanced signals
        assert result["macd_value"] is not None
        assert result["bb_upper"] is not None
        assert result["adx_value"] is not None
        assert result["obv_value"] is not None
        assert result["support_52w"] is not None
        # Scores
        assert result["technical_score_classic"] is not None
        assert result["technical_score_enhanced"] is not None
        assert result["technical_score"] is not None

    def test_uptrend_pattern_scores_well(self):
        """Uptrend pattern should score reasonably well."""
        scorer = TechnicalScorer()
        scorer._enhanced_enabled = True

        session, q = _session_with_mock_company()
        rows = _create_price_rows(260, pattern="uptrend")
        q.filter.return_value.order_by.return_value.limit.return_value.all.return_value = rows

        result = scorer.score(1, session)

        assert result is not None
        # Uptrend should have:
        # - above_200ma = True
        # - RSI might be overbought (depends on acceleration)
        # - MACD positive
        assert result["above_200ma"] is True
        assert result["macd_value"] is not None
        # Score should be > 0 at least
        assert result["technical_score"] > 0

    def test_downtrend_pattern_scores_poorly(self):
        """Downtrend pattern should score poorly."""
        scorer = TechnicalScorer()
        scorer._enhanced_enabled = True

        session, q = _session_with_mock_company()
        rows = _create_price_rows(260, pattern="downtrend")
        q.filter.return_value.order_by.return_value.limit.return_value.all.return_value = rows

        result = scorer.score(1, session)

        assert result is not None
        # Downtrend should have:
        # - above_200ma = False
        # - MACD negative
        assert result["above_200ma"] is False
        assert result["macd_value"] is not None
        # Score should be relatively low (< 50)
        assert result["technical_score"] < 50

    def test_insufficient_data_returns_partial_score(self):
        """With insufficient data for some indicators, should still return score."""
        scorer = TechnicalScorer()
        scorer._enhanced_enabled = True

        session, q = _session_with_mock_company()
        # Only 50 rows - enough for RSI, MACD, Bollinger but not ADX, OBV, S/R
        rows = _create_price_rows(50, pattern="uptrend")
        q.filter.return_value.order_by.return_value.limit.return_value.all.return_value = rows

        result = scorer.score(1, session)

        assert result is not None
        # Some indicators should be calculated
        assert result["rsi_14"] is not None
        assert result["macd_value"] is not None
        # Some should be None
        assert result["support_52w"] is None  # Needs 252 days
        # Should still have a valid score (dynamic weight rescaling)
        assert result["technical_score"] is not None
        assert 0 <= result["technical_score"] <= 100
