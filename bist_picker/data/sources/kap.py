"""KAP (Kamuyu Aydinlatma Platformu) data source client.

Scrapes company data from kap.org.tr - the Turkish Public Disclosure Platform.
KAP uses a Next.js frontend with embedded JSON data in RSC (React Server
Components) payloads. No public REST API is available; data is extracted
from server-rendered HTML pages.

Data available:
- Company list with basic info (ticker, name, city, audit firm)
- Company detail (sector, market segment, indices, paid capital, trade reg date)
- Company documents (articles of association, sustainability reports)

Limitations:
- Insider transaction and disclosure search APIs are internal only (proxied
  via kapsitebackend.mkk.com.tr) and not accessible externally.
- The "last five notifications" section on company pages is loaded dynamically
  via client-side JS, not embedded in the SSR HTML.
- Encoding issues with Turkish chars in RSC payloads (displayed as replacement
  chars in some contexts but data is structurally intact).
"""

import json
import logging
import re
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from bist_picker.data.cache import FileCache
from bist_picker.utils.rate_limiter import RateLimiter

logger = logging.getLogger("bist_picker.data.sources.kap")

_BASE_URL = "https://www.kap.org.tr"
_BIST_SIRKETLER_URL = f"{_BASE_URL}/tr/bist-sirketler"
_BILDIRIM_SORGU_URL = f"{_BASE_URL}/tr/bildirim-sorgu"
_SIRKET_OZET_URL = f"{_BASE_URL}/tr/sirket-bilgileri/ozet"
_SIRKET_GENEL_URL = f"{_BASE_URL}/tr/sirket-bilgileri/genel"

_REQUEST_TIMEOUT = 30  # seconds - KAP pages are large


class KAPClient:
    """Client for scraping BIST company data from KAP (kap.org.tr).

    Args:
        rate_limiter: RateLimiter instance. Default 2s delay for KAP.
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._rate_limiter = rate_limiter or RateLimiter(
            min_delay=2.0, name="kap"
        )
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        self._cache = FileCache()
        # OID lookup: ticker -> mkkMemberOid (populated on first company list fetch)
        self._oid_map: dict[str, str] = {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=3, max=15),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout)
        ),
        reraise=True,
    )
    def _get_page(self, url: str) -> str:
        """Fetch an HTML page with rate limiting and retry.

        Args:
            url: Full URL to fetch.

        Returns:
            Response text (HTML).
        """
        self._rate_limiter.wait()
        response = self._session.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text

    def fetch_company_list(self) -> pd.DataFrame:
        """Fetch list of all BIST-listed companies from KAP.

        Scrapes the bist-sirketler page and extracts embedded JSON data
        from the Next.js RSC payload.

        Returns:
            DataFrame with columns: ticker, name, city, audit_firm,
            detail_url, mkk_oid. Empty DataFrame if fetch fails.
        """
        # Check cache first
        cached = self._cache.load_raw_response(
            "kap", "ALL", "company_list", max_age_hours=168
        )
        if cached:
            df = pd.DataFrame(cached)
            self._build_oid_map(df)
            return df

        try:
            html = self._get_page(_BIST_SIRKETLER_URL)
        except requests.RequestException as e:
            logger.warning("Failed to fetch KAP company list: %s", e)
            return pd.DataFrame()

        companies = self._parse_company_list(html)
        if not companies:
            logger.warning("No companies found in KAP page")
            return pd.DataFrame()

        df = pd.DataFrame(companies)
        self._build_oid_map(df)

        # Cache the raw data
        self._cache.save_raw_response(
            "kap", "ALL", "company_list", companies
        )

        logger.info("Fetched %d companies from KAP", len(df))
        return df

    def fetch_company_detail(self, ticker: str) -> dict[str, Any]:
        """Fetch detailed company information from KAP.

        Uses the bildirim-sorgu page's embedded data (which has richer
        fields than bist-sirketler) and the ozet page for indices/market.

        Args:
            ticker: BIST ticker code (e.g., "THYAO").

        Returns:
            Dict with: ticker, name, city, audit_firm, sector,
            market_segment, indices, paid_capital, trade_reg_date,
            mkk_oid. Empty dict if not found.
        """
        ticker = ticker.upper()

        # Check cache
        cached = self._cache.load_raw_response(
            "kap", ticker, "company_detail", max_age_hours=168
        )
        if cached:
            return cached

        # Ensure we have the OID map
        oid = self._get_oid(ticker)
        if not oid:
            logger.warning("No KAP OID found for %s", ticker)
            return {}

        result: dict[str, Any] = {"ticker": ticker, "mkk_oid": oid}

        # Get basic info from bildirim-sorgu embedded data
        enriched = self._fetch_enriched_company_data(ticker)
        if enriched:
            result.update(enriched)

        # Get indices and market from ozet page
        ozet_data = self._fetch_ozet_data(oid)
        if ozet_data:
            result.update(ozet_data)

        if len(result) > 2:  # More than just ticker and oid
            # Convert date objects to strings for JSON serialization
            cacheable: dict[str, Any] = {}
            for k, v in result.items():
                if isinstance(v, date):
                    cacheable[k] = v.isoformat()
                else:
                    cacheable[k] = v
            self._cache.save_raw_response(
                "kap", ticker, "company_detail", cacheable
            )

        return result

    def fetch_insider_transactions(
        self, ticker: str, days_back: int = 180
    ) -> pd.DataFrame:
        """Fetch insider buying/selling disclosures for a company.

        NOTE: KAP's disclosure search API is internal only (proxied via
        kapsitebackend.mkk.com.tr) and not accessible from external
        scrapers. The "last five notifications" section on company pages
        is loaded dynamically via client-side JS, not embedded in HTML.

        This method extracts company document metadata (articles of
        association, sustainability reports) that IS embedded in the
        ozet page. For actual insider transaction data, use the
        IsYatirimClient or monitor KAP disclosures manually.

        Args:
            ticker: BIST ticker code.
            days_back: Look back period in days (for cache key only).

        Returns:
            DataFrame with columns: disclosure_index, file_name,
            process_name. Empty DataFrame if fetch fails or no data.
        """
        ticker = ticker.upper()

        # Check cache
        cached = self._cache.load_raw_response(
            "kap", ticker, "documents", max_age_hours=24
        )
        if cached:
            return pd.DataFrame(cached)

        oid = self._get_oid(ticker)
        if not oid:
            logger.warning("No KAP OID found for %s", ticker)
            return pd.DataFrame()

        try:
            html = self._get_page(f"{_SIRKET_OZET_URL}/{oid}")
        except requests.RequestException as e:
            logger.warning(
                "Failed to fetch KAP ozet for %s: %s", ticker, e
            )
            return pd.DataFrame()

        documents = self._parse_company_documents(html)
        if documents:
            self._cache.save_raw_response(
                "kap", ticker, "documents", documents
            )

        return pd.DataFrame(documents) if documents else pd.DataFrame()

    def fetch_financial_publication_dates(
        self, ticker: str
    ) -> list[dict]:
        """Fetch when financial statements were published on KAP.

        NOTE: Full disclosure history is NOT available via scraping.
        KAP's notification search API is internal only. This method
        returns an empty list and logs a warning.

        For backtesting, financial publication dates should be estimated
        from Is Yatirim's data or from the known regulatory deadlines
        (Q1: May 15, H1: Aug 15, 9M: Nov 15, Annual: Mar 1).

        Args:
            ticker: BIST ticker code.

        Returns:
            Empty list. Financial publication dates are not available
            via KAP scraping.
        """
        logger.info(
            "Financial publication dates not available via KAP scraping "
            "for %s. Use regulatory deadline estimates instead.",
            ticker.upper(),
        )
        return []

    # --- Private: Parsing helpers ---

    def _parse_company_list(self, html: str) -> list[dict]:
        """Parse company list from embedded Next.js RSC data.

        The bist-sirketler page embeds company data in a script tag as
        part of the RSC flight response. The data is organized as
        alphabetical sections with company objects.

        Args:
            html: Full HTML of the bist-sirketler page.

        Returns:
            List of company dicts.
        """
        # Find the script containing company data (look for stockCode)
        scripts = re.findall(
            r"<script>(.*?)</script>", html, re.DOTALL
        )

        data_script = None
        for sc in scripts:
            if "stockCode" in sc and len(sc) > 50000:
                data_script = sc
                break

        if not data_script:
            logger.warning("Could not find company data script in KAP HTML")
            return []

        # Extract the JSON data array from RSC payload
        # The data is embedded as: "data":[{sections}]
        # We need to unescape the RSC format first
        unescaped = data_script.replace('\\"', '"')

        data_idx = unescaped.find('"data":[')
        if data_idx < 0:
            logger.warning("Could not find data array in KAP script")
            return []

        # Extract the JSON array by matching brackets
        bracket_start = unescaped.find("[", data_idx)
        depth = 0
        end = bracket_start
        for j in range(bracket_start, len(unescaped)):
            if unescaped[j] == "[":
                depth += 1
            elif unescaped[j] == "]":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break

        try:
            sections = json.loads(unescaped[bracket_start:end])
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse KAP company data JSON: %s", e)
            return []

        companies = []
        for section in sections:
            for item in section.get("content", []):
                stock_code = (item.get("stockCode") or "").strip()
                if not stock_code:
                    continue

                # Handle multi-ticker entries (e.g. "THYAO, THYAP")
                for code in stock_code.split(","):
                    code = code.strip().upper()
                    if not code:
                        continue

                    companies.append({
                        "ticker": code,
                        "name": (
                            item.get("kapMemberTitle") or ""
                        ).strip(),
                        "city": (
                            item.get("cityName") or ""
                        ).strip(),
                        "audit_firm": (
                            item.get("relatedMemberTitle") or ""
                        ).strip(),
                        "mkk_oid": (
                            item.get("mkkMemberOid") or ""
                        ).strip(),
                        "detail_url": (
                            f"{_SIRKET_OZET_URL}/"
                            f"{item.get('mkkMemberOid', '')}"
                        ),
                    })

        return companies

    def _fetch_enriched_company_data(self, ticker: str) -> dict[str, Any]:
        """Fetch enriched company data from bildirim-sorgu page.

        The bildirim-sorgu page embeds ALL companies with extra fields
        not available on bist-sirketler: paidCapital, tradeRegDate,
        kayitliSermayeTavani, faaliyetDurumu, etc.

        Args:
            ticker: BIST ticker code.

        Returns:
            Dict with enriched company fields.
        """
        # Check if we have cached enriched data
        cached = self._cache.load_raw_response(
            "kap", "ALL", "enriched_companies", max_age_hours=168
        )

        if not cached:
            try:
                html = self._get_page(_BILDIRIM_SORGU_URL)
            except requests.RequestException as e:
                logger.warning(
                    "Failed to fetch KAP bildirim-sorgu: %s", e
                )
                return {}

            cached = self._parse_enriched_companies(html)
            if cached:
                self._cache.save_raw_response(
                    "kap", "ALL", "enriched_companies", cached
                )

        if not cached:
            return {}

        # Find the ticker in the enriched data
        for item in cached:
            if (item.get("stockCode") or "").strip().upper() == ticker:
                return self._map_enriched_fields(item)
            # Handle multi-ticker
            codes = (item.get("stockCode") or "").split(",")
            for code in codes:
                if code.strip().upper() == ticker:
                    return self._map_enriched_fields(item)

        return {}

    def _parse_enriched_companies(self, html: str) -> list[dict]:
        """Parse enriched company data from bildirim-sorgu page.

        This page has a ~1MB script with detailed company objects.

        Args:
            html: Full HTML of bildirim-sorgu page.

        Returns:
            List of raw company dicts from the embedded data.
        """
        scripts = re.findall(
            r"<script>(.*?)</script>", html, re.DOTALL
        )

        # Find the big script with company data
        for sc in scripts:
            if "stockCode" not in sc or len(sc) < 100000:
                continue

            # The data has double-escaped JSON objects
            # Extract all company objects
            unescaped = sc.replace('\\"', '"')
            companies: list[dict] = []

            # Find objects starting with "kapMemberOid"
            pattern = r'\{"kapMemberOid":"[^"]+","kapMemberType":'
            for match in re.finditer(pattern, unescaped):
                start = match.start()
                # Find the closing brace
                depth = 0
                end = start
                for j in range(start, min(start + 2000, len(unescaped))):
                    if unescaped[j] == "{":
                        depth += 1
                    elif unescaped[j] == "}":
                        depth -= 1
                        if depth == 0:
                            end = j + 1
                            break

                try:
                    obj = json.loads(unescaped[start:end])
                    if obj.get("stockCode"):
                        companies.append(obj)
                except json.JSONDecodeError:
                    continue

            if companies:
                logger.info(
                    "Parsed %d enriched companies from bildirim-sorgu",
                    len(companies),
                )
                return companies

        return []

    @staticmethod
    def _map_enriched_fields(item: dict) -> dict[str, Any]:
        """Map raw enriched company fields to our schema.

        Args:
            item: Raw company dict from bildirim-sorgu data.

        Returns:
            Dict with mapped field names.
        """
        result: dict[str, Any] = {
            "name": (item.get("kapMemberTitle") or "").strip(),
            "city": (item.get("cityName") or "").strip(),
            "audit_firm": (
                item.get("relatedMemberTitle") or ""
            ).strip(),
            "paid_capital": item.get("paidCapital"),
            "authorized_capital_ceiling": item.get(
                "kayitliSermayeTavani"
            ),
            "tax_no": (item.get("taxNo") or "").strip(),
            "tax_office": (item.get("taxOffice") or "").strip(),
            "company_code": (
                item.get("companyCode") or ""
            ).strip(),
            "kap_member_type": (
                item.get("kapMemberType") or ""
            ).strip(),
            "kap_member_state": (
                item.get("kapMemberState") or ""
            ).strip(),
        }

        # Parse trade registration date
        trade_reg_date_str = item.get("tradeRegDate", "")
        if trade_reg_date_str:
            result["trade_reg_date"] = _parse_kap_datetime(
                trade_reg_date_str
            )
        else:
            result["trade_reg_date"] = None

        return result

    def _fetch_ozet_data(self, oid: str) -> dict[str, Any]:
        """Fetch company summary (ozet) page and extract indices/market.

        Args:
            oid: mkkMemberOid for the company.

        Returns:
            Dict with: sector, market_segment, indices (list of str).
        """
        try:
            html = self._get_page(f"{_SIRKET_OZET_URL}/{oid}")
        except requests.RequestException as e:
            logger.warning(
                "Failed to fetch KAP ozet for OID %s: %s", oid, e
            )
            return {}

        result: dict[str, Any] = {}

        # Extract indices from the page
        # Indices appear as link text like "BIST 100", "BIST 30", etc.
        indices = re.findall(
            r'(?:BIST\s+\w+|XU\d+|XUTEK|XBANK)',
            html,
        )
        if indices:
            result["indices"] = sorted(set(indices))

        # Extract market segment
        # Market appears near "Pazar" section - encoded in link href
        # Yildiz Pazar, Ana Pazar, Alt Pazar
        for market_name in [
            "YILDIZ PAZAR", "ANA PAZAR", "ALT PAZAR",
            "Yildiz Pazar", "Ana Pazar", "Alt Pazar",
        ]:
            if market_name.upper() in html.upper():
                result["market_segment"] = market_name.upper()
                break

        # Extract sector from the page
        # Sector appears as "ULASTIRMA VE DEPOLAMA" etc.
        sector_match = re.search(
            r'(?:sektor|sector)[^"]*"value":"([^"]+)"',
            html,
        )
        if sector_match:
            result["sector"] = sector_match.group(1)

        return result

    def _parse_company_documents(self, html: str) -> list[dict]:
        """Parse company documents from the ozet page.

        The ozet page embeds document metadata (articles of association,
        sustainability reports, etc.) with disclosure indices, file names,
        and process types. These are the only embedded disclosure-like
        data; actual notifications are loaded via client-side JS.

        Args:
            html: Full HTML of the ozet page.

        Returns:
            List of document dicts with: disclosure_index, file_name,
            process_name, extension.
        """
        documents: list[dict] = []
        unescaped = html.replace('\\"', '"')

        # Find document entries in the embedded RSC data
        # Pattern: {"fileObjId":"...","disclosureIndex":123,"fileName":"..."}
        for match in re.finditer(
            r'"disclosureIndex":(\d+),"fileName":"([^"]*)",'
            r'"extension":"([^"]*)",'
            r'"processName":"([^"]*)"',
            unescaped,
        ):
            doc = {
                "disclosure_index": int(match.group(1)),
                "file_name": match.group(2),
                "extension": match.group(3),
                "process_name": match.group(4),
            }

            # Avoid duplicates
            if not any(
                d["disclosure_index"] == doc["disclosure_index"]
                and d["file_name"] == doc["file_name"]
                for d in documents
            ):
                documents.append(doc)

        return documents

    def _get_oid(self, ticker: str) -> Optional[str]:
        """Get mkkMemberOid for a ticker, fetching company list if needed.

        Args:
            ticker: BIST ticker code.

        Returns:
            OID string, or None if not found.
        """
        if not self._oid_map:
            self.fetch_company_list()

        return self._oid_map.get(ticker.upper())

    def _build_oid_map(self, df: pd.DataFrame) -> None:
        """Build ticker -> OID lookup from company list DataFrame.

        Args:
            df: Company list DataFrame with ticker and mkk_oid columns.
        """
        if df.empty:
            return
        for _, row in df.iterrows():
            ticker = row.get("ticker", "")
            oid = row.get("mkk_oid", "")
            if ticker and oid:
                self._oid_map[ticker] = oid


def _parse_kap_datetime(date_str: str) -> Optional[date]:
    """Parse KAP datetime string to date.

    KAP dates come as "DD/MM/YYYY HH:MM:SS" or "DD.MM.YYYY".

    Args:
        date_str: Date string from KAP.

    Returns:
        date object, or None if parsing fails.
    """
    if not date_str:
        return None

    for fmt in [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y",
        "%d-%m-%Y",
    ]:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue

    return None
