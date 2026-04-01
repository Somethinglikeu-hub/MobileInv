"""Gemini LLM Analyzer for BIST Stock Picker Enhanced Pipeline.

Wraps Google Gemini API for 3 structured analysis functions:
  1. analyze_kap_event()    — Extract structured data from KAP disclosures
  2. score_analyst_tone()   — Score forward-looking tone of analyst reports
  3. classify_macro_headlines() — Classify macro sentiment from headlines

All functions return structured Python dicts parsed from LLM JSON output.
Free tier: 1,500 req/day on Gemini 2.5 Flash.
"""

import hashlib
import json
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("bist_picker.data.sources.llm_analyzer")

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "llm_config.yaml"
_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


def _load_config() -> dict:
    """Load LLM configuration from llm_config.yaml."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("llm_config.yaml not found at %s", _CONFIG_PATH)
        return {}


def _load_api_key() -> Optional[str]:
    """Load Gemini API key from env var, settings.yaml, or APIKEY_FOLDER."""
    # 1. Environment variable
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    # 2. settings.yaml
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
        key = settings.get("gemini", {}).get("api_key", "")
        if key:
            return key
    except Exception:
        pass

    # 3. APIKEY_FOLDER
    api_folder = Path(__file__).resolve().parent.parent.parent.parent / "APIKEY_FOLDER"
    gemini_key_file = api_folder / "gemini_api_key.txt"
    if gemini_key_file.exists():
        key = gemini_key_file.read_text(encoding="utf-8").strip()
        if key:
            return key

    return None


class LLMAnalyzer:
    """Wraps Gemini API for structured financial text analysis.

    Usage::

        analyzer = LLMAnalyzer()
        result = analyzer.analyze_kap_event("KAP disclosure text here...")
        print(result)  # {"event_type": "NEW_CONTRACT", "sentiment": 0.8, ...}
    """

    def __init__(self, api_key: Optional[str] = None):
        self._config = _load_config()
        self._llm_config = self._config.get("llm", {})
        self._prompts = self._config.get("prompts", {})

        # Resolve API key
        self._api_key = api_key or _load_api_key()
        if not self._api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY env var, "
                "add to settings.yaml under gemini.api_key, or place in "
                "APIKEY_FOLDER/gemini_api_key.txt"
            )

        # Model config — Flash for routine, Pro for deep reasoning
        self._model_flash = self._llm_config.get("model", "gemini-2.5-flash")
        self._model_pro = self._llm_config.get("model_pro", "gemini-2.5-pro")
        self._model_name = self._model_flash  # default
        self._max_requests = self._llm_config.get("max_requests_per_day", 1400)
        self._timeout = self._llm_config.get("timeout_seconds", 30)
        self._retry_attempts = self._llm_config.get("retry_attempts", 3)

        # Task-based model routing: maps task_type -> "flash" or "pro"
        self._model_routing: dict[str, str] = self._llm_config.get("model_routing", {
            "kap_event": "flash",
            "macro_headlines": "flash",
            "analyst_tone": "flash",
            "stock_report": "pro",
        })

        # Cost tracking
        self._request_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Initialize Gemini client
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Google GenAI client."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            logger.info("Gemini client initialized with model=%s", self._model_name)
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install 'google-genai>=1.0'"
            )
        except Exception as e:
            logger.error("Failed to initialize Gemini client: %s", e)
            raise

    def _call_llm(
        self,
        system_prompt: str,
        user_content: str,
        max_output_tokens: int = 1024,
        task_type: Optional[str] = None,
    ) -> Optional[dict]:
        """Make a structured LLM call and parse JSON response.

        Args:
            system_prompt: System instruction for the model.
            user_content: User message (the text to analyze).
            max_output_tokens: Maximum tokens for the response (default 1024).
            task_type: Optional task identifier for model routing. When set,
                the routing map determines whether to use Flash (cheap) or
                Pro (expensive) model. Unrecognized task types use Flash.

        Returns:
            Parsed JSON dict, or None if the call fails.
        """
        import time as _time

        if self._request_count >= self._max_requests:
            logger.warning(
                "Daily request limit reached (%d/%d). Skipping.",
                self._request_count, self._max_requests,
            )
            return None

        # Model routing: select Flash or Pro based on task_type
        if task_type:
            tier = self._model_routing.get(task_type, "flash")
            model_to_use = self._model_pro if tier == "pro" else self._model_flash
        else:
            model_to_use = self._model_flash

        from google.genai import types

        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=model_to_use,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,  # Low temperature for structured output
                        max_output_tokens=max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )

                self._request_count += 1

                # Track token usage if available
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    self._total_input_tokens += getattr(um, "prompt_token_count", 0)
                    self._total_output_tokens += getattr(um, "candidates_token_count", 0)

                # Parse response text as JSON
                raw_text = response.text
                if not raw_text:
                    logger.warning("Empty response from LLM (attempt %d)", attempt)
                    continue

                # Clean potential markdown fencing
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    # Remove ```json ... ``` wrapper
                    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

                parsed = json.loads(cleaned)
                return parsed

            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON parse error on attempt %d: %s. Raw: %s",
                    attempt, e, raw_text[:200] if 'raw_text' in dir() else "N/A",
                )
            except Exception as e:
                err_str = str(e)
                # Handle rate limiting (429) with backoff
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_secs = 15 * attempt  # 15s, 30s, 45s
                    logger.warning(
                        "Rate limited (attempt %d). Waiting %ds...",
                        attempt, wait_secs,
                    )
                    _time.sleep(wait_secs)
                    continue
                logger.warning(
                    "LLM call failed on attempt %d: %s", attempt, e,
                )

        logger.error("All %d LLM attempts failed.", self._retry_attempts)
        return None

    # ── Function 1: KAP Event Extraction ─────────────────────────────────────

    def analyze_kap_event(self, disclosure_text: str) -> Optional[dict]:
        """Extract structured event data from a KAP disclosure.

        Args:
            disclosure_text: Raw KAP disclosure text (Turkish).

        Returns:
            Dict with: event_type, sentiment, monetary_value, currency,
            counterparty, duration_months, confidence, summary.
            None if extraction fails.
        """
        system_prompt = self._prompts.get("kap_event", "")
        if not system_prompt:
            logger.error("KAP event system prompt not found in llm_config.yaml")
            return None

        result = self._call_llm(system_prompt, disclosure_text, task_type="kap_event")
        if result is None:
            return None

        # Validate required fields
        validated = {
            "event_type": result.get("event_type", "OTHER"),
            "sentiment": self._clamp(result.get("sentiment"), -1.0, 1.0),
            "monetary_value": result.get("monetary_value"),
            "currency": result.get("currency"),
            "counterparty": result.get("counterparty"),
            "duration_months": result.get("duration_months"),
            "confidence": self._clamp(result.get("confidence"), 0.0, 1.0),
            "summary": result.get("summary", ""),
        }

        # Validate event_type
        valid_types = {
            "NEW_CONTRACT", "DIVIDEND", "SHARE_BUYBACK", "CAPITAL_INCREASE",
            "LAWSUIT", "PENALTY", "BOARD_CHANGE", "MERGER_ACQUISITION",
            "CAPACITY_EXPANSION", "RATING_CHANGE", "PARTNERSHIP",
            "ASSET_SALE", "DEBT_ISSUANCE", "OTHER",
        }
        if validated["event_type"] not in valid_types:
            validated["event_type"] = "OTHER"

        return validated

    # ── Function 2: Analyst Tone Scoring ─────────────────────────────────────

    def score_analyst_tone(self, report_text: str) -> Optional[dict]:
        """Score the forward-looking tone of an analyst report.

        Args:
            report_text: Analyst report text (Turkish or English).

        Returns:
            Dict with: tone_score (1-10), key_themes, risk_flags,
            target_price_mentioned, recommendation, confidence.
            None if scoring fails.
        """
        system_prompt = self._prompts.get("analyst_tone", "")
        if not system_prompt:
            logger.error("Analyst tone system prompt not found in llm_config.yaml")
            return None

        result = self._call_llm(system_prompt, report_text, task_type="analyst_tone")
        if result is None:
            return None

        validated = {
            "tone_score": self._clamp(result.get("tone_score"), 1.0, 10.0),
            "key_themes": result.get("key_themes", [])[:3],
            "risk_flags": result.get("risk_flags", [])[:3],
            "target_price_mentioned": result.get("target_price_mentioned"),
            "recommendation": result.get("recommendation"),
            "confidence": self._clamp(result.get("confidence"), 0.0, 1.0),
        }

        valid_recs = {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", None}
        if validated["recommendation"] not in valid_recs:
            validated["recommendation"] = None

        return validated

    # ── Function 3: Macro Headline Classification ────────────────────────────

    def classify_macro_headlines(
        self, headlines: list[str]
    ) -> Optional[dict]:
        """Classify macro sentiment from financial headlines.

        Args:
            headlines: List of financial headline strings (Turkish).

        Returns:
            Dict with: macro_sentiment, key_drivers, sector_impacts,
            confidence. None if classification fails.
        """
        system_prompt = self._prompts.get("macro_headlines", "")
        if not system_prompt:
            logger.error("Macro headlines system prompt not found in llm_config.yaml")
            return None

        # Format headlines as numbered list
        formatted = "\n".join(
            f"{i+1}. {h.strip()}" for i, h in enumerate(headlines) if h.strip()
        )
        if not formatted:
            return None

        result = self._call_llm(system_prompt, formatted, task_type="macro_headlines")
        if result is None:
            return None

        validated = {
            "macro_sentiment": result.get("macro_sentiment", "NEUTRAL"),
            "key_drivers": result.get("key_drivers", [])[:3],
            "sector_impacts": result.get("sector_impacts", {}),
            "confidence": self._clamp(result.get("confidence"), 0.0, 1.0),
        }

        valid_sentiments = {"BULLISH", "CAUTIOUS", "BEARISH", "NEUTRAL"}
        if validated["macro_sentiment"] not in valid_sentiments:
            validated["macro_sentiment"] = "NEUTRAL"

        # Clamp sector impact scores
        if isinstance(validated["sector_impacts"], dict):
            validated["sector_impacts"] = {
                k: self._clamp(v, -1.0, 1.0)
                for k, v in validated["sector_impacts"].items()
                if isinstance(v, (int, float))
            }

        return validated

    # ── Utility Methods ──────────────────────────────────────────────────────

    @staticmethod
    def text_hash(text: str) -> str:
        """Generate SHA-256 hash of text for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _clamp(
        value: Optional[float], min_val: float, max_val: float
    ) -> Optional[float]:
        """Clamp a numeric value to [min_val, max_val], or None."""
        if value is None or not isinstance(value, (int, float)):
            return None
        return max(min_val, min(max_val, float(value)))

    def get_usage_stats(self) -> dict:
        """Return current session usage statistics."""
        return {
            "requests_made": self._request_count,
            "requests_remaining": self._max_requests - self._request_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "model": self._model_name,
        }
