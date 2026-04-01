"""EVDS Nowcasting data source for BIST Stock Picker Enhanced Pipeline.

Fetches forward-looking macro indicators from TCMB's EVDS system:
  - BONC (Bileşik Öncü Göstergeler) composite leading indicator
  - Credit card sectoral spending data

Follows the same pattern as the existing TCMBClient in tcmb.py.
"""

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

logger = logging.getLogger("bist_picker.data.sources.evds_nowcast")

_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
_EVDS_BASE_URL = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"

# EVDS Series Codes
# BONC: Composite Leading Indicators (Bileşik Öncü Göstergeler Endeksi)
# Old code was TP.BONC.G.I01 (archived); new datagroup is bie_cli2
_BONC_SERIES = "TP.CLI2.A01"

# Credit card spending by sector
# These cover total card spending volumes — sectoral breakdown available
_CREDIT_CARD_TOTAL = "TP.AB.B1"  # Total card transaction volume


def _load_evds_api_key() -> Optional[str]:
    """Load EVDS/TCMB API key from env, settings.yaml, or APIKEY_FOLDER."""
    # 1. Environment variable
    key = os.environ.get("TCMB_API_KEY")
    if key:
        return key

    # 2. settings.yaml
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
        key = settings.get("tcmb", {}).get("api_key", "")
        if key:
            return key
    except Exception:
        pass

    # 3. APIKEY_FOLDER (base64-encoded)
    api_folder = Path(__file__).resolve().parent.parent.parent.parent / "APIKEY_FOLDER"
    key_file = api_folder / "api_key.txt"
    if key_file.exists():
        try:
            import base64
            raw = key_file.read_text(encoding="utf-8").strip()
            if raw:
                key = base64.b64decode(raw).decode("utf-8")
                if key:
                    return key
        except Exception:
            pass

    return None


class EVDSNowcastClient:
    """Fetches forward-looking macro indicators from TCMB EVDS.

    Usage::

        client = EVDSNowcastClient()
        bonc = client.fetch_bonc_index()
        print(bonc)  # pd.DataFrame with date, value, mom_change, trend
    """

    def __init__(self, api_key: Optional[str] = None, delay: float = 1.0):
        self._api_key = api_key or _load_evds_api_key()
        if not self._api_key:
            raise ValueError(
                "EVDS API key not found. Set TCMB_API_KEY env var "
                "or add to settings.yaml under tcmb.api_key"
            )
        self._delay = delay

    def _fetch_series(
        self,
        series_code: str,
        start_date: date,
        end_date: date,
        frequency: int = 5,  # 5 = monthly
    ) -> Optional[pd.DataFrame]:
        """Fetch a single EVDS data series.

        Uses the evds3 endpoint format where parameters are embedded
        in the URL path (no '?' separator). API key sent as HTTP header.

        Args:
            series_code: EVDS series identifier.
            start_date: Start date for data range.
            end_date: End date for data range.
            frequency: 1=daily, 2=b-daily, 3=weekly, 4=biweekly, 5=monthly

        Returns:
            DataFrame with columns [date, value], or None on failure.
        """
        # Build URL with params embedded in path (evds3 format)
        url = (
            f"{_EVDS_BASE_URL}"
            f"series={series_code}"
            f"&startDate={start_date.strftime('%d-%m-%Y')}"
            f"&endDate={end_date.strftime('%d-%m-%Y')}"
            f"&type=json"
            f"&frequency={frequency}"
            f"&aggregationTypes=avg"
        )

        try:
            time.sleep(self._delay)

            headers = {"key": self._api_key}
            resp = requests.get(url, headers=headers, timeout=30)

            # Check we got JSON, not HTML
            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type:
                logger.error(
                    "EVDS returned HTML instead of JSON for %s", series_code
                )
                return None

            resp.raise_for_status()
            data = resp.json()

            if "items" not in data or not data["items"]:
                logger.warning("No data returned for series %s", series_code)
                return None

            rows = []
            # Value key: dots replaced with underscores (e.g. TP.BONC.G.I01 -> TP_BONC_G_I01)
            value_key = series_code.replace(".", "_")

            for item in data["items"]:
                date_str = item.get("Tarih", "")
                dt = _parse_evds_date(date_str)
                if dt is None:
                    continue

                raw_value = item.get(value_key)
                if raw_value is None:
                    # Try alternative key formats (original dots, or scan)
                    raw_value = item.get(series_code)
                    if raw_value is None:
                        for k, v in item.items():
                            if k not in ("Tarih", "UNIXTIME") and v is not None:
                                raw_value = v
                                break

                if raw_value is not None:
                    try:
                        value = float(str(raw_value).replace(",", "."))
                        rows.append({"date": dt, "value": value})
                    except (ValueError, TypeError):
                        continue

            if not rows:
                return None

            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            return df

        except requests.exceptions.RequestException as e:
            logger.error("EVDS request failed for %s: %s", series_code, e)
            return None
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", series_code, e)
            return None

    # ── BONC (Composite Leading Indicators) ──────────────────────────────────

    def fetch_bonc_index(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        lookback_months: int = 24,
    ) -> Optional[pd.DataFrame]:
        """Fetch BONC composite leading indicator index.

        Returns DataFrame with columns:
            date, bonc_index, bonc_change_mom, bonc_trend

        The BONC index leads economic activity by ~3-6 months.
        A rising BONC = expanding economy ahead = RISK_ON signal.
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=lookback_months * 30)

        df = self._fetch_series(_BONC_SERIES, start_date, end_date)
        if df is None or df.empty:
            logger.warning("Failed to fetch BONC data")
            return None

        df = df.rename(columns={"value": "bonc_index"})

        # Calculate month-over-month change
        df["bonc_change_mom"] = df["bonc_index"].pct_change() * 100

        # Determine trend (3-month moving direction)
        df["bonc_ma3"] = df["bonc_index"].rolling(3, min_periods=2).mean()
        df["bonc_trend"] = "FLAT"
        df.loc[df["bonc_index"] > df["bonc_ma3"] * 1.005, "bonc_trend"] = "RISING"
        df.loc[df["bonc_index"] < df["bonc_ma3"] * 0.995, "bonc_trend"] = "FALLING"

        df = df.drop(columns=["bonc_ma3"])

        logger.info("Fetched %d BONC observations (%s to %s)",
                     len(df), df["date"].min(), df["date"].max())
        return df

    # ── Credit Card Spending ─────────────────────────────────────────────────

    def fetch_credit_card_spending(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        lookback_months: int = 12,
    ) -> Optional[pd.DataFrame]:
        """Fetch total credit card spending data.

        Returns DataFrame with columns:
            date, total_spending, spending_change_mom

        Credit card spending is a real-time proxy for consumer demand.
        Rising spending in a sector → revenue growth signal for related stocks.
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=lookback_months * 30)

        df = self._fetch_series(_CREDIT_CARD_TOTAL, start_date, end_date)
        if df is None or df.empty:
            logger.warning("Failed to fetch credit card spending data")
            return None

        df = df.rename(columns={"value": "total_spending"})

        # Calculate month-over-month change
        df["spending_change_mom"] = df["total_spending"].pct_change() * 100

        logger.info("Fetched %d credit card spending observations", len(df))
        return df

    # ── Convenience Methods ──────────────────────────────────────────────────

    def fetch_all_nowcast_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict:
        """Fetch all nowcast data sources at once.

        Returns:
            Dict with 'bonc' and 'credit_card' DataFrames.
        """
        return {
            "bonc": self.fetch_bonc_index(start_date, end_date),
            "credit_card": self.fetch_credit_card_spending(start_date, end_date),
        }

    @staticmethod
    def interpret_bonc_for_regime(latest_bonc_row: dict) -> str:
        """Interpret the latest BONC reading for macro regime classification.

        Args:
            latest_bonc_row: Dict with bonc_trend and bonc_change_mom.

        Returns:
            One of: RISK_ON, RISK_OFF, NEUTRAL
        """
        trend = latest_bonc_row.get("bonc_trend", "FLAT")
        mom = latest_bonc_row.get("bonc_change_mom", 0.0)

        if trend == "RISING" and (mom is not None and mom > 1.0):
            return "RISK_ON"
        elif trend == "FALLING" and (mom is not None and mom < -1.0):
            return "RISK_OFF"
        else:
            return "NEUTRAL"


def _parse_evds_date(date_str: str) -> Optional[date]:
    """Parse EVDS date string to date object.

    Handles multiple formats returned by EVDS:
    - Daily: "DD-MM-YYYY" (e.g., "01-01-2024")
    - Monthly: "YYYY-M" (e.g., "2024-1") or "YYYY-MM" (e.g., "2024-01")

    Args:
        date_str: Date string from EVDS response.

    Returns:
        date object, or None if parsing fails.
    """
    if not date_str:
        return None

    # Try daily format: "DD-MM-YYYY"
    try:
        return pd.to_datetime(date_str, format="%d-%m-%Y").date()
    except (ValueError, TypeError):
        pass

    # Try monthly format: "YYYY-M" or "YYYY-MM"
    try:
        parts = date_str.split("-")
        if len(parts) == 2:
            year = int(parts[0])
            month = int(parts[1])
            return date(year, month, 1)
    except (ValueError, TypeError, IndexError):
        pass

    return None
