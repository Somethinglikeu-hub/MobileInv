"""Tests for LLM Analyzer — Gemini API wrapper.

Tests use mocked Gemini API responses to verify:
  - JSON parsing and validation
  - Error handling for malformed responses
  - Rate limiting behavior
  - All 3 analysis functions (KAP event, analyst tone, macro headlines)
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# We need to mock the API key loading before importing the module
@pytest.fixture(autouse=True)
def mock_api_key():
    """Ensure API key is available for all tests."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-12345"}):
        yield


@pytest.fixture
def mock_genai_client():
    """Create a mock Gemini client that returns controlled responses."""
    with patch("google.genai.Client") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        yield mock_client


def _make_response(json_data: dict):
    """Helper to create a mock Gemini response object."""
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(json_data)
    mock_resp.usage_metadata = MagicMock()
    mock_resp.usage_metadata.prompt_token_count = 100
    mock_resp.usage_metadata.candidates_token_count = 50
    return mock_resp


class TestAnalyzeKapEvent:
    """Test KAP disclosure event extraction."""

    def test_valid_new_contract(self, mock_genai_client):
        """Parse a typical NEW_CONTRACT disclosure."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        expected = {
            "event_type": "NEW_CONTRACT",
            "sentiment": 0.8,
            "monetary_value": 150000000,
            "currency": "USD",
            "counterparty": "NATO",
            "duration_months": 36,
            "confidence": 0.9,
            "summary": "ASELSAN won a $150M defense contract from NATO.",
        }
        mock_genai_client.models.generate_content.return_value = _make_response(expected)

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.analyze_kap_event("ASELSAN NATO sözleşmesi...")

        assert result is not None
        assert result["event_type"] == "NEW_CONTRACT"
        assert result["sentiment"] == 0.8
        assert result["monetary_value"] == 150000000
        assert result["currency"] == "USD"
        assert result["confidence"] == 0.9

    def test_clamps_sentiment_range(self, mock_genai_client):
        """Sentiment values outside [-1, 1] should be clamped."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        response_data = {
            "event_type": "PENALTY",
            "sentiment": -5.0,  # out of range
            "confidence": 2.0,  # out of range
        }
        mock_genai_client.models.generate_content.return_value = _make_response(response_data)

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.analyze_kap_event("Test text")

        assert result["sentiment"] == -1.0
        assert result["confidence"] == 1.0

    def test_invalid_event_type_defaults_to_other(self, mock_genai_client):
        """Unknown event types should default to OTHER."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        response_data = {
            "event_type": "SPACE_LAUNCH",  # not valid
            "sentiment": 0.5,
            "confidence": 0.8,
        }
        mock_genai_client.models.generate_content.return_value = _make_response(response_data)

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.analyze_kap_event("Test text")

        assert result["event_type"] == "OTHER"

    def test_empty_response_returns_none(self, mock_genai_client):
        """Empty LLM response should return None gracefully."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_resp.usage_metadata = None
        mock_genai_client.models.generate_content.return_value = mock_resp

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.analyze_kap_event("Test text")

        assert result is None


class TestScoreAnalystTone:
    """Test analyst report tone scoring."""

    def test_valid_buy_report(self, mock_genai_client):
        """Parse a typical bullish analyst report."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        expected = {
            "tone_score": 7.5,
            "key_themes": ["revenue_growth", "margin_expansion"],
            "risk_flags": ["currency_risk"],
            "target_price_mentioned": 45.50,
            "recommendation": "BUY",
            "confidence": 0.85,
        }
        mock_genai_client.models.generate_content.return_value = _make_response(expected)

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.score_analyst_tone("Analyst report text...")

        assert result is not None
        assert result["tone_score"] == 7.5
        assert result["recommendation"] == "BUY"
        assert len(result["key_themes"]) == 2

    def test_clamps_tone_score(self, mock_genai_client):
        """Tone score outside [1, 10] should be clamped."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        response_data = {
            "tone_score": 15.0,  # out of range
            "recommendation": "STRONG_BUY",
            "confidence": 0.7,
        }
        mock_genai_client.models.generate_content.return_value = _make_response(response_data)

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.score_analyst_tone("Bullish report")

        assert result["tone_score"] == 10.0

    def test_invalid_recommendation(self, mock_genai_client):
        """Invalid recommendations should become None."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        response_data = {
            "tone_score": 5.0,
            "recommendation": "OUTPERFORM",  # not in our valid set
            "confidence": 0.6,
        }
        mock_genai_client.models.generate_content.return_value = _make_response(response_data)

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.score_analyst_tone("Report text")

        assert result["recommendation"] is None


class TestClassifyMacroHeadlines:
    """Test macro headline classification."""

    def test_valid_cautious_classification(self, mock_genai_client):
        """Parse a typical cautious macro classification."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        expected = {
            "macro_sentiment": "CAUTIOUS",
            "key_drivers": ["interest_rate_uncertainty", "usd_try_pressure"],
            "sector_impacts": {
                "XBANK": -0.3,
                "XUTEK": 0.1,
                "XTRZM": -0.2,
            },
            "confidence": 0.75,
        }
        mock_genai_client.models.generate_content.return_value = _make_response(expected)

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.classify_macro_headlines([
            "TCMB faiz kararı belirsizliği sürüyor",
            "Dolar/TL 35 seviyesini test etti",
        ])

        assert result is not None
        assert result["macro_sentiment"] == "CAUTIOUS"
        assert result["sector_impacts"]["XBANK"] == -0.3
        assert result["confidence"] == 0.75

    def test_sector_impact_clamping(self, mock_genai_client):
        """Sector impact scores outside [-1, 1] should be clamped."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        response_data = {
            "macro_sentiment": "BULLISH",
            "sector_impacts": {"XBANK": 5.0, "XUTEK": -3.0},
            "confidence": 0.8,
        }
        mock_genai_client.models.generate_content.return_value = _make_response(response_data)

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.classify_macro_headlines(["Test headline"])

        assert result["sector_impacts"]["XBANK"] == 1.0
        assert result["sector_impacts"]["XUTEK"] == -1.0

    def test_empty_headlines_returns_none(self, mock_genai_client):
        """Empty headlines list should return None."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.classify_macro_headlines([])

        assert result is None

    def test_invalid_sentiment_defaults_to_neutral(self, mock_genai_client):
        """Invalid sentiment should default to NEUTRAL."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        response_data = {
            "macro_sentiment": "VERY_WORRIED",  # not valid
            "confidence": 0.5,
        }
        mock_genai_client.models.generate_content.return_value = _make_response(response_data)

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.classify_macro_headlines(["Headline"])

        assert result["macro_sentiment"] == "NEUTRAL"


class TestUsageTracking:
    """Test usage statistics and rate limiting."""

    def test_usage_stats_increment(self, mock_genai_client):
        """Usage stats should increment after each call."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        response_data = {"event_type": "OTHER", "sentiment": 0.0, "confidence": 0.5}
        mock_genai_client.models.generate_content.return_value = _make_response(response_data)

        analyzer = LLMAnalyzer(api_key="test-key")
        stats_before = analyzer.get_usage_stats()
        assert stats_before["requests_made"] == 0

        analyzer.analyze_kap_event("Test text")
        stats_after = analyzer.get_usage_stats()
        assert stats_after["requests_made"] == 1
        assert stats_after["total_input_tokens"] == 100
        assert stats_after["total_output_tokens"] == 50

    def test_text_hash(self, mock_genai_client):
        """Text hashing for deduplication should be deterministic."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        hash1 = LLMAnalyzer.text_hash("test disclosure")
        hash2 = LLMAnalyzer.text_hash("test disclosure")
        hash3 = LLMAnalyzer.text_hash("different text")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 hex digest


class TestJSONParsing:
    """Test handling of various JSON formats from LLM."""

    def test_markdown_wrapped_json(self, mock_genai_client):
        """Handle JSON wrapped in ```json ... ``` markers."""
        from bist_picker.data.sources.llm_analyzer import LLMAnalyzer

        mock_resp = MagicMock()
        mock_resp.text = '```json\n{"event_type": "DIVIDEND", "sentiment": 0.6, "confidence": 0.9}\n```'
        mock_resp.usage_metadata = None
        mock_genai_client.models.generate_content.return_value = mock_resp

        analyzer = LLMAnalyzer(api_key="test-key")
        result = analyzer.analyze_kap_event("Temettü dağıtım kararı")

        assert result is not None
        assert result["event_type"] == "DIVIDEND"
