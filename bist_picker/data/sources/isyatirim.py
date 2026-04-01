"""Is Yatirim (isyatirim.com.tr) data source client.

Fetches BIST stock data from Is Yatirim's public JSON API. No authentication required.

Endpoints used:
- HisseTekil: Historical OHLCV price data per ticker
- GetHissePerformans: Ticker lists with fundamentals (all BIST / BIST-100)
- MaliTablo: Financial statements (income, balance sheet, cash flow)
- SirketBilgileri: Company info and pre-calculated ratios
"""

import json
import logging
import urllib3
from datetime import date
from typing import Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from bist_picker.data.cache import FileCache
from bist_picker.utils.rate_limiter import RateLimiter
from bist_picker.utils.turkish import convert_turkish_number

logger = logging.getLogger("bist_picker.data.sources.isyatirim")

# Suppress InsecureRequestWarning when SSL verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Base URLs
_HISSE_TEKIL_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/"
    "Isyatirim.Website/Common/Data.aspx/HisseTekil"
)
_HISSE_PERFORMANS_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/"
    "IsYatirim.Website/StockInfo/CompanyInfoAjax.aspx/GetHissePerformans"
)
_MALI_TABLO_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/"
    "IsYatirim.Website/Common/Data.aspx/MaliTablo"
)
_SIRKET_BILGILERI_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/"
    "IsYatirim.Website/Common/Data.aspx/SirketBilgileri"
)

# Index codes for GetHissePerformans
_INDEX_ALL_SHARES = "09"   # BIST National All Shares
_INDEX_BIST100 = "01"      # BIST-100

_REQUEST_TIMEOUT = 15  # seconds


class IsYatirimClient:
    """Client for fetching BIST data from Is Yatirim's public API.

    Args:
        rate_limiter: RateLimiter instance to control request frequency.
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        verify_ssl: bool = True,
    ) -> None:
        self._rate_limiter = rate_limiter or RateLimiter(min_delay=1.0, name="isyatirim")
        self._session = requests.Session()
        self._cache = FileCache()
        self._verify_ssl = verify_ssl
        # Test SSL and fall back to unverified if cert chain fails
        if verify_ssl:
            try:
                self._session.get(
                    "https://www.isyatirim.com.tr", timeout=5, verify=True
                )
            except requests.exceptions.SSLError:
                logger.info(
                    "SSL verification failed for isyatirim.com.tr; "
                    "disabling SSL verification for this source (using verification fallback)"
                )
                self._verify_ssl = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _get(self, url: str, params: dict) -> dict:
        """Make a rate-limited GET request with retry logic.

        Args:
            url: Request URL.
            params: Query parameters.

        Returns:
            Parsed JSON response as dict.
        """
        self._rate_limiter.wait()
        response = self._session.get(
            url, params=params, timeout=_REQUEST_TIMEOUT, verify=self._verify_ssl
        )
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _post_json(self, url: str, payload: dict) -> dict:
        """Make a rate-limited POST request with retry logic.

        Args:
            url: Request URL.
            payload: JSON body.

        Returns:
            Parsed JSON response as dict.
        """
        self._rate_limiter.wait()
        response = self._session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=_REQUEST_TIMEOUT,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        return response.json()

    def fetch_price_data(
        self, ticker: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Fetch historical price data for a single ticker.

        Uses the HisseTekil endpoint. Returns adjusted (corporate-action
        corrected) prices. Note: the API does not provide an explicit
        opening price; the VWAP (AOF) field is used as a proxy.

        Args:
            ticker: BIST ticker code (e.g., "THYAO").
            start_date: Start date for historical data.
            end_date: End date for historical data.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume,
            adjusted_close, source. Empty DataFrame if fetch fails.
        """
        params = {
            "hisse": ticker.upper(),
            "startdate": start_date.strftime("%d-%m-%Y"),
            "enddate": end_date.strftime("%d-%m-%Y"),
        }

        try:
            data = self._get(_HISSE_TEKIL_URL, params)
        except requests.RequestException as e:
            logger.warning("Failed to fetch prices for %s: %s", ticker, e)
            return self._empty_price_df()

        values = data.get("value")
        if not values:
            logger.warning("No price data returned for %s", ticker)
            return self._empty_price_df()

        rows = []
        for item in values:
            try:
                row_date = self._parse_api_date(item.get("HGDG_TARIH"))
                if row_date is None:
                    continue

                rows.append({
                    "date": row_date,
                    # API has no Open price; use VWAP (AOF) as proxy
                    "open": self._to_float(item.get("HGDG_AOF")),
                    "high": self._to_float(item.get("HGDG_MAX")),
                    "low": self._to_float(item.get("HGDG_MIN")),
                    "close": self._to_float(item.get("HGDG_KAPANIS")),
                    # Volume from API is TRY turnover; keep as-is
                    "volume": self._to_int(item.get("HGDG_HACIM")),
                    "adjusted_close": self._to_float(item.get("HGDG_KAPANIS")),
                    "source": "ISYATIRIM",
                })
            except (ValueError, TypeError) as e:
                logger.debug("Skipping malformed row for %s: %s", ticker, e)
                continue

        if not rows:
            return self._empty_price_df()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def fetch_all_tickers(self) -> list[str]:
        """Fetch list of all BIST stock tickers.

        Uses GetHissePerformans with endeksKodu="09" (all shares).

        Returns:
            Sorted list of ticker strings. Empty list if fetch fails.
        """
        return self._fetch_ticker_list(_INDEX_ALL_SHARES)

    def fetch_bist100_tickers(self) -> list[str]:
        """Fetch current BIST 100 composition.

        Uses GetHissePerformans with endeksKodu="01" (BIST-100).

        Returns:
            Sorted list of ticker strings. Empty list if fetch fails.
        """
        return self._fetch_ticker_list(_INDEX_BIST100)

    def fetch_company_overview(self, index_code: str = _INDEX_ALL_SHARES) -> pd.DataFrame:
        """Fetch overview data for all companies in an index.

        Returns a DataFrame with ticker, name, sector, market_cap,
        free_float_pct, pe, pb, and other fundamentals.

        Args:
            index_code: Index code ("09" for all, "01" for BIST-100).

        Returns:
            DataFrame with company overview data. Empty if fetch fails.
        """
        raw = self._fetch_performans_raw(index_code)
        if not raw:
            return pd.DataFrame()

        rows = []
        for item in raw:
            rows.append({
                "ticker": (item.get("HISSE_KODU") or "").strip(),
                "name": (item.get("HISSE_TANIM") or "").strip(),
                "name_en": (item.get("HISSE_TANIM_YD") or "").strip(),
                "sector": (item.get("AS_ALT_SEKTOR_TANIMI") or "").strip(),
                "sector_en": (item.get("AS_ALT_SEKTOR_TANIMI_YD") or "").strip(),
                "price": self._to_float(item.get("PRICE_TL")),
                "market_cap": self._to_float(item.get("MARKET_CAP_TL")),
                "free_float_pct": self._to_float(item.get("FREE_FLOAT_RATE")),
                "pe_ratio": self._to_float(item.get("PE")),
                "pb_ratio": self._to_float(item.get("MV_BV")),
                "ev_ebitda": self._to_float(item.get("EV_EBITDA")),
                "paid_up_capital": self._to_float(item.get("PAID_UP_CAP")),
                "latest_period": (item.get("SON_DONEM") or "").strip(),
            })

        df = pd.DataFrame(rows)
        df = df[df["ticker"] != ""].sort_values("ticker").reset_index(drop=True)
        return df

    def fetch_financials(
        self, ticker: str, financial_group: str = "1"
    ) -> dict:
        """Fetch financial statements (income, balance sheet, cash flow).

        Fetches the latest 4 annual periods from the MaliTablo endpoint.
        Automatically detects bank vs non-bank companies: tries XI_29 first,
        falls back to UFRS if no data returned.

        Args:
            ticker: BIST ticker code (e.g., "THYAO").
            financial_group: Accounting standard code.
                '1' = XI_29 (IAS 29, non-banks), '2' = UFRS (banks consolidated),
                '3' = UFRS_K (banks solo). Default '1'.

        Returns:
            Dict with keys 'income', 'balance', 'cashflow' (each a DataFrame),
            'raw' (list of raw items), and 'financial_group' (label used).
            Empty dict if fetch fails.
        """
        ticker = ticker.upper()
        current_year = date.today().year

        # Check cache first
        cached = self._cache.load_raw_response("isyatirim", ticker, "financials", max_age_hours=168)
        if cached:
            raw_items = cached.get("raw", [])
            fg_label = cached.get("financial_group", "XI_29")
            periods = cached.get("periods", [])
            if raw_items:
                result = self._parse_mali_tablo(raw_items, periods)
                result["raw"] = raw_items
                result["financial_group"] = fg_label
                return result

        # Try fetching with the requested financial_group, auto-detect if needed
        fg_label = self._financial_group_label(financial_group)
        raw_items, fg_label, periods = self._fetch_mali_tablo_raw(
            ticker, fg_label, current_year
        )

        if not raw_items:
            return {}

        # Cache raw response
        self._cache.save_raw_response("isyatirim", ticker, "financials", {
            "raw": raw_items,
            "financial_group": fg_label,
            "periods": periods,
        })

        result = self._parse_mali_tablo(raw_items, periods)
        result["raw"] = raw_items
        result["financial_group"] = fg_label
        return result

    def fetch_financials_deep(
        self, ticker: str, financial_group: str = "1", num_years: int = 5
    ) -> dict:
        """Fetch deep historical financial statements (up to 5 years quarterly).

        Unlike fetch_financials which only gets 4 annual periods, this fetches
        up to 20 quarterly periods covering *num_years* years of history.
        Bypasses cache to ensure fresh data.

        Use this for one-time historical backfill. Results include quarterly
        data (periods 3, 6, 9, 12) which enables Graham, Lynch PEG, and DCF
        scorers that need multi-year earnings history.

        Args:
            ticker: BIST ticker code.
            financial_group: Accounting standard code ('1'=XI_29, '2'=UFRS, '3'=UFRS_K).
            num_years: How many years of history to fetch (max 5 = 20 quarters).

        Returns:
            Same dict format as fetch_financials.
        """
        ticker = ticker.upper()
        current_year = date.today().year
        fg_label = self._financial_group_label(financial_group)

        raw_items, fg_label, periods = self._fetch_mali_tablo_raw(
            ticker, fg_label, current_year, num_years=num_years, quarterly=True
        )

        if not raw_items:
            return {}

        result = self._parse_mali_tablo(raw_items, periods)
        result["raw"] = raw_items
        result["financial_group"] = fg_label
        return result

    def fetch_company_info(self, ticker: str) -> dict:
        """Fetch company information and key ratios.

        Uses the SirketBilgileri endpoint. Returns the most recent
        annual reported data (index 0 of the response array).

        Args:
            ticker: BIST ticker code (e.g., "THYAO").

        Returns:
            Dict with company info fields. Empty dict if fetch fails.
        """
        ticker = ticker.upper()
        raw = self._fetch_sirket_bilgileri_raw(ticker)
        if not raw:
            return {}

        # Index 0 = latest annual reported data
        item = raw[0]
        return {
            "ticker": ticker,
            "name": (item.get("Title") or "").strip(),
            "sector": (item.get("AS_ALT_SEKTOR_TANIMI") or "").strip(),
            "sector_en": (item.get("AS_ALT_SEKTOR_TANIMI_YD") or "").strip(),
            "market_cap": self._to_float(item.get("MARKET_CAP_TL")),
            "pe_ratio": self._to_float(item.get("F_K")),
            "pb_ratio": self._to_float(item.get("F_DD")),
            "ev_ebitda": self._to_float(item.get("FD_FAVOK")),
            "roe": self._to_float(item.get("ROE")),
            "roa": self._to_float(item.get("ROA")),
            "target_price": self._to_float(item.get("Son_Hedef_Fiyat_Target_Price")),
            "foreign_ownership_pct": self._to_float(item.get("YABANCI_ORAN")),
            # NET_NAKIT uses Turkish comma-decimal format
            "net_cash": convert_turkish_number(item.get("NET_NAKIT")),
            "recommendation": (item.get("ONERI_ACIKLAMA_ENG") or "").strip(),
            "price": self._to_float(item.get("Fiyat_TL_Price_TL")),
            "net_income": self._to_float(item.get("Net_Kar")),
            "revenue": self._to_float(item.get("Satislar")),
            "ebitda": self._to_float(item.get("FAVOK")),
            "equity": self._to_float(item.get("Ozsermaye")),
            # Trailing (current) ratios — use Turkish number conversion
            "trailing_pe": convert_turkish_number(item.get("CARI_FK")),
            "trailing_pb": convert_turkish_number(item.get("CARI_PD_DD")),
            "trailing_ev_ebitda": convert_turkish_number(item.get("CARI_FD_FAVOK")),
        }

    def fetch_ratios(self, ticker: str) -> dict:
        """Fetch pre-calculated financial ratios.

        Uses the SirketBilgileri endpoint. Returns reported annual ratios
        and trailing (current) ratios for comparison.

        Args:
            ticker: BIST ticker code (e.g., "THYAO").

        Returns:
            Dict with ratio fields. Empty dict if fetch fails.
        """
        ticker = ticker.upper()
        raw = self._fetch_sirket_bilgileri_raw(ticker)
        if not raw:
            return {}

        item = raw[0]
        result = {
            "ticker": ticker,
            "pe": self._to_float(item.get("F_K")),
            "pb": self._to_float(item.get("F_DD")),
            "ev_ebitda": self._to_float(item.get("FD_FAVOK")),
            "ev_sales": self._to_float(item.get("FD_Satislar")),
            "roe": self._to_float(item.get("ROE")),
            "roa": self._to_float(item.get("ROA")),
        }

        # Trailing ratios use Turkish comma-decimal format
        result["trailing_pe"] = convert_turkish_number(item.get("CARI_FK"))
        result["trailing_pb"] = convert_turkish_number(item.get("CARI_PD_DD"))
        result["trailing_ev_ebitda"] = convert_turkish_number(item.get("CARI_FD_FAVOK"))
        result["trailing_ev_sales"] = convert_turkish_number(item.get("CARI_FD_SATIS"))

        # Compute net_margin from revenue and net_income if available
        revenue = self._to_float(item.get("Satislar"))
        net_income = self._to_float(item.get("Net_Kar"))
        if revenue and net_income and revenue != 0:
            result["net_margin"] = net_income / revenue
        else:
            result["net_margin"] = None

        # debt/equity not directly available from this endpoint;
        # would need balance sheet data from fetch_financials
        result["debt_equity"] = None

        return result

    # --- Private helpers ---

    def _fetch_mali_tablo_raw(
        self,
        ticker: str,
        fg_label: str,
        current_year: int,
        num_years: int = 4,
        quarterly: bool = False,
    ) -> tuple[list[dict], str, list[str]]:
        """Fetch raw MaliTablo data, with auto-detection for banks.

        Tries the given financial_group first. If it returns no data,
        tries UFRS and then UFRS_K (for bank detection).

        Args:
            ticker: BIST ticker code.
            fg_label: Financial group label (XI_29, UFRS, UFRS_K).
            current_year: Current calendar year.
            num_years: Number of years of history to request (default 4).
            quarterly: If True, fetch quarterly periods (3,6,9,12) instead
                       of annual-only (12). Max 20 period slots.

        Returns:
            Tuple of (raw_items, financial_group_label, period_labels).
        """
        last_year = current_year - 1

        # Build period list
        if quarterly:
            # Quarterly: 4 quarters per year, newest first
            years_list = []
            periods_list = []
            for y_offset in range(num_years):
                yr = last_year - y_offset
                for q in [12, 9, 6, 3]:  # newest quarter first
                    years_list.append(yr)
                    periods_list.append(q)
            # MaliTablo supports up to 20 period slots (year1..year20)
            years_list = years_list[:20]
            periods_list = periods_list[:20]
            period_labels = [f"{y}/{p}" for y, p in zip(years_list, periods_list)]
        else:
            # Annual only (original behaviour)
            years_list = [last_year - i for i in range(num_years)]
            periods_list = [12] * num_years
            years_list = years_list[:20]
            periods_list = periods_list[:20]
            period_labels = [f"{y}/12" for y in years_list]

        num_slots = len(years_list)

        # Try requested group first
        for try_fg in [fg_label, "UFRS", "UFRS_K", "XI_29K", "UFRS_B"]:
            params = {
                "companyCode": ticker,
                "exchange": "TRY",
                "financialGroup": try_fg,
            }
            for i in range(num_slots):
                params[f"year{i + 1}"] = years_list[i]
                params[f"period{i + 1}"] = periods_list[i]

            try:
                data = self._get(_MALI_TABLO_URL, params)
            except requests.RequestException as e:
                logger.warning("Failed to fetch financials for %s (fg=%s): %s", ticker, try_fg, e)
                continue

            items = data.get("value", [])
            if items:
                if try_fg != fg_label:
                    logger.info("%s: no data with %s, using %s instead", ticker, fg_label, try_fg)
                return items, try_fg, period_labels

            # If the requested group returned nothing and it's the same as what
            # we'd try next, skip to avoid duplicate requests
            if try_fg == fg_label:
                continue

        logger.debug("No financial data found for %s with any financial group", ticker)
        return [], fg_label, []

    def _fetch_sirket_bilgileri_raw(self, ticker: str) -> list[dict]:
        """Fetch raw SirketBilgileri response."""
        # Check cache
        cached = self._cache.load_raw_response("isyatirim", ticker, "company_info", max_age_hours=24)
        if cached:
            return cached

        try:
            data = self._get(_SIRKET_BILGILERI_URL, {"hisse": ticker})
        except requests.RequestException as e:
            logger.warning("Failed to fetch company info for %s: %s", ticker, e)
            return []

        items = data.get("value", [])
        if not items:
            logger.warning("No company info returned for %s", ticker)
            return []

        # Cache raw response
        self._cache.save_raw_response("isyatirim", ticker, "company_info", items)
        return items

    @staticmethod
    def _financial_group_label(code: str) -> str:
        """Map numeric financial_group code to API label string.

        Args:
            code: '1', '2', '3', or the label itself.

        Returns:
            API label: 'XI_29', 'UFRS', or 'UFRS_K'.
        """
        mapping = {"1": "XI_29", "2": "UFRS", "3": "UFRS_K"}
        return mapping.get(code, code)

    @staticmethod
    def _parse_mali_tablo(
        raw_items: list[dict], periods: list[str]
    ) -> dict:
        """Parse raw MaliTablo items into income/balance/cashflow DataFrames.

        Splits items by itemCode prefix:
        - 1*, 2* = Balance sheet
        - 3* = Income statement
        - 4B* = Supplementary data (D&A, FX position, etc.)
        - 4C* = Cash flow statement

        Handles variable number of value columns (value1..valueN) matching
        the number of requested periods.

        Args:
            raw_items: List of dicts from MaliTablo API.
            periods: Period labels for column names.

        Returns:
            Dict with 'income', 'balance', 'cashflow' DataFrames.
        """
        income_rows = []
        balance_rows = []
        cashflow_rows = []

        # Dynamically build value column names to match requested periods
        num_periods = len(periods)
        value_cols = [f"value{i + 1}" for i in range(num_periods)]
        col_names = periods if periods else [f"period_{i+1}" for i in range(num_periods)]

        for item in raw_items:
            code = item.get("itemCode", "")
            row = {
                "item_code": code,
                "desc_tr": item.get("itemDescTr", ""),
                "desc_eng": item.get("itemDescEng") or "",
            }
            # Convert string values to float
            for vc, cn in zip(value_cols, col_names):
                val = item.get(vc)
                if val is not None and val != "":
                    try:
                        row[cn] = float(val)
                    except (ValueError, TypeError):
                        row[cn] = None
                else:
                    row[cn] = None

            # Route to appropriate statement
            if code.startswith("1") or code.startswith("2"):
                balance_rows.append(row)
            elif code.startswith("3"):
                income_rows.append(row)
            elif code.startswith("4C"):
                cashflow_rows.append(row)
            elif code.startswith("4B"):
                # Supplementary data (D&A, exports, FX) — add to income
                income_rows.append(row)
            elif code.startswith("4"):
                # Any other 4* prefix
                cashflow_rows.append(row)

        return {
            "income": pd.DataFrame(income_rows) if income_rows else pd.DataFrame(),
            "balance": pd.DataFrame(balance_rows) if balance_rows else pd.DataFrame(),
            "cashflow": pd.DataFrame(cashflow_rows) if cashflow_rows else pd.DataFrame(),
        }

    def _fetch_ticker_list(self, index_code: str) -> list[str]:
        """Fetch ticker list for a given index code."""
        raw = self._fetch_performans_raw(index_code)
        if not raw:
            return []

        tickers = []
        for item in raw:
            code = item.get("HISSE_KODU", "").strip()
            if code:
                tickers.append(code)

        return sorted(set(tickers))

    def _fetch_performans_raw(self, index_code: str) -> list[dict]:
        """Call GetHissePerformans and return the parsed Oran array."""
        payload = {
            "endeksKodu": index_code,
            "sektorKodu": "",
            "exchange": "TRY",
        }

        try:
            data = self._post_json(_HISSE_PERFORMANS_URL, payload)
        except requests.RequestException as e:
            logger.warning("Failed to fetch ticker list (index=%s): %s", index_code, e)
            return []

        # Response has {"d": "<json string>"} — needs double parse
        d_str = data.get("d", "")
        if not d_str:
            logger.warning("Empty response from GetHissePerformans (index=%s)", index_code)
            return []

        try:
            inner = json.loads(d_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to parse inner JSON from GetHissePerformans: %s", e)
            return []

        return inner.get("Oran", [])

    @staticmethod
    def _parse_api_date(date_str: Optional[str]) -> Optional[date]:
        """Parse dd-mm-yyyy date string from the API."""
        if not date_str:
            return None
        try:
            parts = date_str.strip().split("-")
            if len(parts) == 3:
                return date(int(parts[2]), int(parts[1]), int(parts[0]))
        except (ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _to_float(value) -> Optional[float]:
        """Safely convert API value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_int(value) -> Optional[int]:
        """Safely convert API value to int."""
        if value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _empty_price_df() -> pd.DataFrame:
        """Return an empty DataFrame with the expected price columns."""
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume",
                      "adjusted_close", "source"]
        )
