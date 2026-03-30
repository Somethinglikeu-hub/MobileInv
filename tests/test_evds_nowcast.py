"""Tests for EVDS Nowcast client — BONC and credit card data.

Tests use mocked EVDS API responses to verify:
  - Data parsing from EVDS JSON responses
  - MoM change calculations
  - BONC trend classification
  - Regime interpretation logic
"""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_api_key():
    """Ensure TCMB API key is available for all tests."""
    with patch.dict("os.environ", {"TCMB_API_KEY": "test-evds-key"}):
        yield


def _make_evds_response(items: list[dict]) -> MagicMock:
    """Create a mock EVDS API response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"items": items}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ── BONC Tests ───────────────────────────────────────────────────────────────


class TestFetchBONC:
    """Test BONC composite leading indicator fetching."""

    @patch("bist_picker.data.sources.evds_nowcast.requests.get")
    @patch("bist_picker.data.sources.evds_nowcast.time.sleep")
    def test_valid_bonc_data(self, mock_sleep, mock_get):
        """Parse valid BONC EVDS response into DataFrame."""
        from bist_picker.data.sources.evds_nowcast import EVDSNowcastClient

        items = [
            {"Tarih": "01-01-2025", "TP.BONC.G.I01": "102.5"},
            {"Tarih": "01-02-2025", "TP.BONC.G.I01": "103.2"},
            {"Tarih": "01-03-2025", "TP.BONC.G.I01": "104.1"},
            {"Tarih": "01-04-2025", "TP.BONC.G.I01": "103.8"},
        ]
        mock_get.return_value = _make_evds_response(items)

        client = EVDSNowcastClient()
        df = client.fetch_bonc_index(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 4, 1),
        )

        assert df is not None
        assert len(df) == 4
        assert "bonc_index" in df.columns
        assert "bonc_change_mom" in df.columns
        assert "bonc_trend" in df.columns
        assert df.iloc[0]["bonc_index"] == 102.5

    @patch("bist_picker.data.sources.evds_nowcast.requests.get")
    @patch("bist_picker.data.sources.evds_nowcast.time.sleep")
    def test_bonc_trend_rising(self, mock_sleep, mock_get):
        """BONC above 3-month MA should be classified as RISING."""
        from bist_picker.data.sources.evds_nowcast import EVDSNowcastClient

        # Steady uptrend
        items = [
            {"Tarih": "01-01-2025", "TP.BONC.G.I01": "100.0"},
            {"Tarih": "01-02-2025", "TP.BONC.G.I01": "102.0"},
            {"Tarih": "01-03-2025", "TP.BONC.G.I01": "104.0"},
            {"Tarih": "01-04-2025", "TP.BONC.G.I01": "107.0"},
        ]
        mock_get.return_value = _make_evds_response(items)

        client = EVDSNowcastClient()
        df = client.fetch_bonc_index(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 4, 1),
        )

        # Last row should show RISING trend
        assert df.iloc[-1]["bonc_trend"] == "RISING"

    @patch("bist_picker.data.sources.evds_nowcast.requests.get")
    @patch("bist_picker.data.sources.evds_nowcast.time.sleep")
    def test_empty_response(self, mock_sleep, mock_get):
        """Empty EVDS response should return None."""
        from bist_picker.data.sources.evds_nowcast import EVDSNowcastClient

        mock_get.return_value = _make_evds_response([])

        client = EVDSNowcastClient()
        df = client.fetch_bonc_index(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 4, 1),
        )

        assert df is None


# ── Credit Card Tests ────────────────────────────────────────────────────────


class TestFetchCreditCard:
    """Test credit card spending data fetching."""

    @patch("bist_picker.data.sources.evds_nowcast.requests.get")
    @patch("bist_picker.data.sources.evds_nowcast.time.sleep")
    def test_valid_credit_card_data(self, mock_sleep, mock_get):
        """Parse valid credit card spending response."""
        from bist_picker.data.sources.evds_nowcast import EVDSNowcastClient

        items = [
            {"Tarih": "01-01-2025", "TP.AB.B1": "450000000"},
            {"Tarih": "01-02-2025", "TP.AB.B1": "470000000"},
            {"Tarih": "01-03-2025", "TP.AB.B1": "495000000"},
        ]
        mock_get.return_value = _make_evds_response(items)

        client = EVDSNowcastClient()
        df = client.fetch_credit_card_spending(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 1),
        )

        assert df is not None
        assert len(df) == 3
        assert "total_spending" in df.columns
        assert "spending_change_mom" in df.columns


# ── Regime Interpretation Tests ──────────────────────────────────────────────


class TestRegimeInterpretation:
    """Test BONC-based regime interpretation."""

    def test_risk_on_signal(self):
        """Rising BONC with positive MoM → RISK_ON."""
        from bist_picker.data.sources.evds_nowcast import EVDSNowcastClient

        result = EVDSNowcastClient.interpret_bonc_for_regime({
            "bonc_trend": "RISING",
            "bonc_change_mom": 2.5,
        })
        assert result == "RISK_ON"

    def test_risk_off_signal(self):
        """Falling BONC with negative MoM → RISK_OFF."""
        from bist_picker.data.sources.evds_nowcast import EVDSNowcastClient

        result = EVDSNowcastClient.interpret_bonc_for_regime({
            "bonc_trend": "FALLING",
            "bonc_change_mom": -3.0,
        })
        assert result == "RISK_OFF"

    def test_neutral_signal(self):
        """Flat BONC → NEUTRAL."""
        from bist_picker.data.sources.evds_nowcast import EVDSNowcastClient

        result = EVDSNowcastClient.interpret_bonc_for_regime({
            "bonc_trend": "FLAT",
            "bonc_change_mom": 0.2,
        })
        assert result == "NEUTRAL"

    def test_mixed_signal_neutral(self):
        """Rising trend but small MoM (< 1%) → NEUTRAL."""
        from bist_picker.data.sources.evds_nowcast import EVDSNowcastClient

        result = EVDSNowcastClient.interpret_bonc_for_regime({
            "bonc_trend": "RISING",
            "bonc_change_mom": 0.5,
        })
        assert result == "NEUTRAL"
