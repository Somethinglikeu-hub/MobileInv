"""TCMB (Central Bank of Turkey) EVDS data source client.

Fetches macroeconomic data from TCMB's Electronic Data Delivery System (EVDS):
- Consumer Price Index (CPI) for inflation calculations
- USD/TRY and EUR/TRY exchange rates
- Policy interest rate (weighted average funding cost)

Requires a free API key from evds3.tcmb.gov.tr. If no key is available,
methods return empty results with a warning.

API history:
- Pre-2024: evds2.tcmb.gov.tr/service/evds with key as query param
- Apr 2024: key moved to HTTP header
- Late 2024+: evds2 decommissioned, evds3 with new path /igmevdsms-dis/
- The new endpoint embeds params in the URL path (no '?' separator)
- Monthly dates returned as 'YYYY-M' instead of 'DD-MM-YYYY'
"""

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

from bist_picker.data.cache import FileCache
from bist_picker.utils.rate_limiter import RateLimiter

logger = logging.getLogger("bist_picker.data.sources.tcmb")

_EVDS_BASE_URL = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"
_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
_REQUEST_TIMEOUT = 15  # seconds


class TCMBClient:
    """Client for fetching macroeconomic data from TCMB EVDS.

    Args:
        rate_limiter: RateLimiter instance. Default 1.0s delay.
        api_key: EVDS API key. If None, tries TCMB_API_KEY env var,
            then settings.yaml tcmb.api_key.
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._rate_limiter = rate_limiter or RateLimiter(
            min_delay=1.0, name="tcmb"
        )
        self._session = requests.Session()
        self._cache = FileCache()
        self._api_key = api_key or self._load_api_key()

        if not self._api_key:
            logger.warning(
                "No TCMB EVDS API key found. Set TCMB_API_KEY env var "
                "or configure tcmb.api_key in settings.yaml. "
                "Register at https://evds3.tcmb.gov.tr/login for a key. "
                "TCMB data will not be available."
            )

    def fetch_cpi_index(
        self, start_date: date, end_date: date
    ) -> pd.Series:
        """Fetch Consumer Price Index from EVDS.

        Series: TP.FG.J0 (monthly CPI index, 2003=100).

        Args:
            start_date: Start date for data range.
            end_date: End date for data range.

        Returns:
            pd.Series indexed by date with CPI values.
            Empty Series if unavailable.
        """
        if not self._api_key:
            return pd.Series(dtype=float, name="cpi")

        # Check cache
        cached = self._cache.load_raw_response(
            "tcmb", "MACRO", "cpi", max_age_hours=24
        )
        if cached:
            return self._items_to_series(cached, "TP_FG_J0", "cpi")

        items = self._fetch_evds(
            "TP.FG.J0", start_date, end_date, frequency=5
        )

        if not items:
            # Try evdspy fallback
            df = self._fetch_evds_fallback(
                "TP.FG.J0", start_date, end_date, frequency="monthly"
            )
            if df is not None and not df.empty:
                series = self._dataframe_to_series(df, "cpi")
                return series

            return pd.Series(dtype=float, name="cpi")

        self._cache.save_raw_response("tcmb", "MACRO", "cpi", items)
        return self._items_to_series(items, "TP_FG_J0", "cpi")

    def fetch_exchange_rates(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Fetch USD/TRY and EUR/TRY daily exchange rates.

        Series: TP.DK.USD.A (USD buying), TP.DK.EUR.A (EUR buying).

        Args:
            start_date: Start date for data range.
            end_date: End date for data range.

        Returns:
            DataFrame with columns: date, usd_try, eur_try.
            Empty DataFrame if unavailable.
        """
        if not self._api_key:
            return pd.DataFrame(columns=["date", "usd_try", "eur_try"])

        # Check cache
        cached = self._cache.load_raw_response(
            "tcmb", "MACRO", "fx_rates", max_age_hours=24
        )
        if cached:
            return self._items_to_fx_df(cached)

        items = self._fetch_evds(
            "TP.DK.USD.A-TP.DK.EUR.A", start_date, end_date
        )

        if not items:
            return pd.DataFrame(columns=["date", "usd_try", "eur_try"])

        self._cache.save_raw_response("tcmb", "MACRO", "fx_rates", items)
        return self._items_to_fx_df(items)

    def fetch_policy_rate(self) -> Optional[float]:
        """Fetch current TCMB policy rate (weighted avg funding cost).

        Series: TP.APIFON4. Fetches last 90 days and returns the most
        recent non-null value.

        Returns:
            Policy rate as float (e.g., 0.50 for 50%), or None.
        """
        if not self._api_key:
            return None

        end = date.today()
        start = end - timedelta(days=90)
        items = self._fetch_evds("TP.APIFON4", start, end)

        if not items:
            return None

        # Find most recent non-null value
        for item in reversed(items):
            val = item.get("TP_APIFON4")
            if val is not None and val != "":
                try:
                    return float(val) / 100.0  # Convert percentage to decimal
                except (ValueError, TypeError):
                    continue

        return None

    # Candidate series codes for the TCMB Market Participants Survey
    # 24-month-ahead CPI expectation. Tried in order; first non-empty wins.
    # TCMB has renamed these over the years — defensive multi-try avoids
    # pipeline failure when one code is retired.
    _INFLATION_EXP_24M_SERIES = (
        "TP.ENF.BEK24",       # Türkçe EVDS standard naming
        "TP.BEKMP.SM24",      # Survey of Market Participants, 24m mean
        "TP.BEKODTUFEYENI.BT02",  # Post-2018 inflation expectations block
    )

    def fetch_inflation_expectations_24m(self) -> Optional[float]:
        """Fetch the TCMB 24-month-ahead CPI expectation (Market Participants Survey).

        Tries several known series codes for robustness. Returns the most
        recent non-null value as a decimal (e.g., 0.18 = 18%), or None if
        none of the candidate series return data.

        Falls back gracefully so DCF terminal growth can keep using its
        static config default when EVDS is unavailable or series codes
        have been renamed.
        """
        if not self._api_key:
            return None

        end = date.today()
        start = end - timedelta(days=180)  # survey is monthly, 6 months headroom

        for series in self._INFLATION_EXP_24M_SERIES:
            items = self._fetch_evds(series, start, end, frequency=5)
            if not items:
                continue

            # Key in the response is the series with dots → underscores
            value_key = series.replace(".", "_")
            for item in reversed(items):
                val = item.get(value_key)
                if val is None or val == "":
                    continue
                try:
                    rate = float(val) / 100.0
                    logger.info(
                        "Fetched 24m inflation expectation %.2f%% from %s",
                        rate * 100, series,
                    )
                    return rate
                except (ValueError, TypeError):
                    continue

        logger.warning(
            "No 24m inflation expectation series returned data from EVDS. "
            "Tried: %s. DCF will fall back to static terminal_growth_try. "
            "If this persists, update _INFLATION_EXP_24M_SERIES with the "
            "current EVDS code from https://evds3.tcmb.gov.tr/.",
            ", ".join(self._INFLATION_EXP_24M_SERIES),
        )
        return None

    def get_inflation_rate(self, months_back: int = 12) -> Optional[float]:
        """Calculate YoY CPI inflation rate.

        Fetches CPI data for the required period and computes
        (CPI_latest / CPI_N_months_ago) - 1.

        Args:
            months_back: Number of months for YoY comparison. Default 12.

        Returns:
            Inflation rate as float (e.g., 0.65 for 65%), or None.
        """
        end = date.today()
        # Fetch extra months to ensure we have enough data
        start = date(end.year - 2, end.month, 1)
        cpi = self.fetch_cpi_index(start, end)

        if cpi.empty or len(cpi) < months_back + 1:
            return None

        latest = cpi.iloc[-1]
        earlier = cpi.iloc[-(months_back + 1)]

        if earlier == 0 or pd.isna(earlier) or pd.isna(latest):
            return None

        return (latest / earlier) - 1.0

    # --- Private helpers ---

    def _fetch_evds(
        self,
        series: str,
        start_date: date,
        end_date: date,
        frequency: Optional[int] = None,
    ) -> list[dict]:
        """Fetch data from EVDS REST API (evds3 endpoint).

        Uses the new evds3 endpoint format where parameters are embedded
        in the URL path (no '?' separator). API key is sent as HTTP header.

        Args:
            series: Series code(s), dash-separated for multiple.
            start_date: Start date.
            end_date: End date.
            frequency: Data frequency (1=daily, 5=monthly, 8=annual).

        Returns:
            List of data items from EVDS response.
        """
        if not self._api_key:
            return []

        self._rate_limiter.wait()

        # Build URL with params embedded in path (evds3 format)
        url = (
            f"{_EVDS_BASE_URL}"
            f"series={series}"
            f"&startDate={start_date.strftime('%d-%m-%Y')}"
            f"&endDate={end_date.strftime('%d-%m-%Y')}"
            f"&type=json"
        )
        if frequency is not None:
            url += f"&frequency={frequency}"

        headers = {"key": self._api_key}

        try:
            response = self._session.get(
                url, headers=headers, timeout=_REQUEST_TIMEOUT,
            )

            # Check we got JSON, not HTML (SPA fallback page)
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                logger.warning(
                    "EVDS returned HTML instead of JSON for %s "
                    "-- endpoint may have changed again", series,
                )
                return []

            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            if not items:
                logger.debug(
                    "No data returned from EVDS for series %s", series
                )
            return items

        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(
                "EVDS connection failed for %s: %s", series, e,
            )
        except requests.HTTPError as e:
            logger.warning(
                "EVDS HTTP error for %s: %s", series, e,
            )
        except ValueError as e:
            logger.warning(
                "Failed to parse EVDS response for %s: %s", series, e,
            )

        return []

    def _fetch_evds_fallback(
        self,
        series: str,
        start_date: date,
        end_date: date,
        frequency: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Try fetching data using evdspy library as fallback.

        Args:
            series: EVDS series code.
            start_date: Start date.
            end_date: End date.
            frequency: Frequency string for evdspy (monthly, daily, etc.).

        Returns:
            DataFrame from evdspy, or None if not available.
        """
        try:
            from evdspy import get_series
        except ImportError:
            logger.debug("evdspy not installed; skipping fallback")
            return None

        try:
            kwargs = {
                "start_date": start_date.strftime("%d-%m-%Y"),
                "end_date": end_date.strftime("%d-%m-%Y"),
            }
            if frequency:
                kwargs["frequency"] = frequency
            if self._api_key:
                kwargs["api_key"] = self._api_key

            df = get_series(series, **kwargs)
            if df is not None and not df.empty:
                logger.info(
                    "evdspy fallback returned %d rows for %s",
                    len(df), series,
                )
                return df
        except Exception as e:
            logger.warning("evdspy fallback failed for %s: %s", series, e)

        return None

    @staticmethod
    def _load_api_key() -> Optional[str]:
        """Load EVDS API key from environment, settings.yaml, or key file.

        Returns:
            API key string, or None if not found.
        """
        # Try environment variable first
        key = os.environ.get("TCMB_API_KEY", "").strip()
        if key:
            return key

        # Try settings.yaml
        if _SETTINGS_PATH.exists():
            try:
                with open(_SETTINGS_PATH, encoding="utf-8") as f:
                    settings = yaml.safe_load(f)
                key = (settings.get("tcmb", {}).get("api_key", "") or "").strip()
                if key:
                    return key
            except (OSError, yaml.YAMLError) as e:
                logger.debug("Failed to read settings.yaml: %s", e)

        # Try APIKEY_FOLDER/api_key.txt (base64-encoded)
        key_file = Path(__file__).resolve().parent.parent.parent.parent / "APIKEY_FOLDER" / "api_key.txt"
        if key_file.exists():
            try:
                import base64
                raw = key_file.read_text(encoding="utf-8").strip()
                if raw:
                    key = base64.b64decode(raw).decode("utf-8")
                    if key:
                        return key
            except (OSError, Exception) as e:
                logger.debug("Failed to read api_key.txt: %s", e)

        return None

    @staticmethod
    def _items_to_series(
        items: list[dict], value_key: str, name: str
    ) -> pd.Series:
        """Convert EVDS items to a pandas Series.

        Handles multiple date formats from EVDS:
        - Daily: "DD-MM-YYYY" (e.g., "01-01-2024")
        - Monthly: "YYYY-M" (e.g., "2024-1")

        Args:
            items: List of EVDS response items.
            value_key: Key for the value field (dots replaced with underscores).
            name: Name for the resulting Series.

        Returns:
            pd.Series indexed by date.
        """
        dates = []
        values = []
        for item in items:
            date_str = item.get("Tarih", "")
            val = item.get(value_key)
            if not date_str or val is None or val == "":
                continue
            try:
                d = _parse_evds_date(date_str)
                if d is None:
                    continue
                v = float(val)
                dates.append(d)
                values.append(v)
            except (ValueError, TypeError):
                continue

        series = pd.Series(values, index=dates, name=name, dtype=float)
        return series.sort_index()

    @staticmethod
    def _dataframe_to_series(df: pd.DataFrame, name: str) -> pd.Series:
        """Convert evdspy DataFrame to a simple Series.

        evdspy returns DataFrames with date index and value columns.

        Args:
            df: DataFrame from evdspy.
            name: Name for the resulting Series.

        Returns:
            pd.Series with the first value column.
        """
        if df.empty:
            return pd.Series(dtype=float, name=name)

        # evdspy returns columns like "TP_FG_J0" or the series code
        value_col = [c for c in df.columns if c != "Tarih" and c != "date"]
        if not value_col:
            return pd.Series(dtype=float, name=name)

        series = df[value_col[0]].astype(float)
        series.name = name
        return series.sort_index()

    @staticmethod
    def _items_to_fx_df(items: list[dict]) -> pd.DataFrame:
        """Convert EVDS FX items to a DataFrame.

        Args:
            items: List of EVDS response items with USD and EUR rates.

        Returns:
            DataFrame with columns: date, usd_try, eur_try.
        """
        rows = []
        for item in items:
            date_str = item.get("Tarih", "")
            usd = item.get("TP_DK_USD_A")
            eur = item.get("TP_DK_EUR_A")
            if not date_str:
                continue
            d = _parse_evds_date(date_str)
            if d is None:
                continue

            row = {"date": d}
            if usd is not None and usd != "":
                try:
                    row["usd_try"] = float(usd)
                except (ValueError, TypeError):
                    row["usd_try"] = None
            else:
                row["usd_try"] = None

            if eur is not None and eur != "":
                try:
                    row["eur_try"] = float(eur)
                except (ValueError, TypeError):
                    row["eur_try"] = None
            else:
                row["eur_try"] = None

            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["date", "usd_try", "eur_try"])

        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)
        return df


def _parse_evds_date(date_str: str) -> Optional[pd.Timestamp]:
    """Parse EVDS date string to Timestamp.

    Handles multiple formats returned by EVDS:
    - Daily: "DD-MM-YYYY" (e.g., "01-01-2024")
    - Monthly: "YYYY-M" (e.g., "2024-1") or "YYYY-MM" (e.g., "2024-01")

    Args:
        date_str: Date string from EVDS response.

    Returns:
        pd.Timestamp, or None if parsing fails.
    """
    if not date_str:
        return None

    # Try daily format first: "DD-MM-YYYY"
    try:
        return pd.to_datetime(date_str, format="%d-%m-%Y")
    except (ValueError, TypeError):
        pass

    # Try monthly format: "YYYY-M" or "YYYY-MM"
    try:
        parts = date_str.split("-")
        if len(parts) == 2:
            year = int(parts[0])
            month = int(parts[1])
            return pd.Timestamp(year=year, month=month, day=1)
    except (ValueError, TypeError, IndexError):
        pass

    return None
