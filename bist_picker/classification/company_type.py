"""Company type classifier for BIST Stock Picker.

Classifies companies into one of seven types based on sector, company name,
and known ticker lists: OPERATING, HOLDING, BANK, INSURANCE, REIT, SPORT, FINANCIAL.

The classification drives which scoring model is used downstream:
- BANK: do NOT strip monetary gain/loss, use banking scoring model
- HOLDING: use NAV discount model, skip P/E and margin ratios
- INSURANCE: separate model (similar to banking)
- REIT: separate model (NAV-based)
- SPORT: football clubs / sports companies — excluded from all scoring models
- FINANCIAL: leasing, factoring, brokerage, asset management — use banking scoring model
- OPERATING: standard Buffett/Graham/Piotroski scoring
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from bist_picker.db.schema import Company

logger = logging.getLogger("bist_picker.classification.company_type")

# Known tickers for each type (override any heuristic)
_KNOWN_BANKS = {
    "GARAN", "AKBNK", "YKBNK", "HALKB", "VAKBN", "ISCTR",
    "DENIZ", "QNBFB", "SKBNK", "TSKB", "ALBRK",
}

_KNOWN_HOLDINGS = {
    "SAHOL", "KCHOL", "DOHOL", "TAVHL", "TKFEN", "ECZYT",
    "GLYHO", "NTHOL", "KOZAL",
}

_KNOWN_SPORTS = {
    "GSRAY",    # Galatasaray Sportif A.Ş.
    "BJKAS",    # Beşiktaş Futbol Yatırımları A.Ş.
    "TRABZON",  # Trabzonspor Sportif Yatırım A.Ş.
    "ALTAY",    # Altay
}

_KNOWN_FINANCIALS = {
    "ISFIN",    # İş Finansal Kiralama
    "KLNMA",    # Türkiye Kalkınma ve Yatırım Bankası (development bank / financial)
    "CRDFA",    # Creditwest Faktoring
    "GARFA",    # Garanti Faktoring
    "YKFKT",    # Yapı Kredi Faktoring
    "ISGSY",    # İş Girişim Sermayesi
}

# Turkish sector keywords (case-insensitive substring match)
_BANK_SECTOR_KEYWORDS = ["banka"]
_BANK_NAME_KEYWORDS = ["bank"]
_INSURANCE_SECTOR_KEYWORDS = ["sigorta"]
_REIT_SECTOR_KEYWORDS = ["gyo", "gayrimenkul yatirim"]
_HOLDING_SECTOR_KEYWORDS = ["holding"]
_HOLDING_NAME_KEYWORDS = ["holding"]
_SPORT_SECTOR_KEYWORDS = ["futbol", "spor kulübü", "spor yatirim"]
_FINANCIAL_SECTOR_KEYWORDS = [
    "kiralama", "faktoring", "araci kurum", "varlik yonetim",
    "finansal hizmet", "menkul kiymet", "girisim sermayesi",
]
_FINANCIAL_NAME_KEYWORDS = [
    "leasing", "faktoring", "factoring", "finansal kiralama",
    "araci kurum", "menkul kiymet", "varlik yonetim", "girisim sermayesi",
]


class CompanyClassifier:
    """Classifies BIST companies by type for scoring model selection.

    Classification rules (applied in priority order):
    1. Known ticker lists (highest priority)
    2. Sector name keyword matching
    3. Company name keyword matching
    4. Default to OPERATING
    """

    def classify(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        financials: Optional[dict] = None,
    ) -> str:
        """Classify a single company into a type.

        Args:
            ticker: BIST ticker code (e.g., "THYAO").
            company_name: Full company name.
            sector: BIST sector name.
            financials: Optional dict of financial data (reserved for future use).

        Returns:
            One of: "OPERATING", "HOLDING", "BANK", "INSURANCE", "REIT",
            "SPORT", "FINANCIAL".
        """
        ticker_upper = ticker.strip().upper()

        # 1. Known ticker lists (highest priority)
        if ticker_upper in _KNOWN_BANKS:
            return "BANK"
        if ticker_upper in _KNOWN_HOLDINGS:
            return "HOLDING"
        if ticker_upper in _KNOWN_SPORTS:
            return "SPORT"
        if ticker_upper in _KNOWN_FINANCIALS:
            return "FINANCIAL"

        # Normalize for keyword matching
        name_lower = (company_name or "").lower()
        sector_lower = (sector or "").lower()

        # 2. BANK: sector contains "Banka" or name contains "Bank"
        if _matches_any(sector_lower, _BANK_SECTOR_KEYWORDS):
            return "BANK"
        if _matches_any(name_lower, _BANK_NAME_KEYWORDS):
            return "BANK"

        # 3. INSURANCE: sector contains "Sigorta"
        if _matches_any(sector_lower, _INSURANCE_SECTOR_KEYWORDS):
            return "INSURANCE"

        # 4. FINANCIAL: leasing, factoring, brokerage, asset management
        # Must come AFTER bank/insurance check to avoid misclassifying
        # banks or holdings that have "finansal" in their name.
        if _matches_any(sector_lower, _FINANCIAL_SECTOR_KEYWORDS):
            # Guard: don't reclassify if sector also matches holding
            if not _matches_any(sector_lower, _HOLDING_SECTOR_KEYWORDS):
                return "FINANCIAL"
        if _matches_any(name_lower, _FINANCIAL_NAME_KEYWORDS):
            # Guard: don't reclassify if name also matches holding or bank
            if (not _matches_any(name_lower, _HOLDING_NAME_KEYWORDS)
                    and not _matches_any(name_lower, _BANK_NAME_KEYWORDS)):
                return "FINANCIAL"

        # 5. REIT: sector contains "GYO" or "Gayrimenkul Yatirim"
        if _matches_any(sector_lower, _REIT_SECTOR_KEYWORDS):
            return "REIT"

        # 6. SPORT: football clubs and sports investment companies
        if _matches_any(sector_lower, _SPORT_SECTOR_KEYWORDS):
            return "SPORT"
        if _matches_any(name_lower, _SPORT_SECTOR_KEYWORDS):
            return "SPORT"

        # 7. HOLDING: name contains "Holding" or sector contains "Holding"
        if _matches_any(name_lower, _HOLDING_NAME_KEYWORDS):
            return "HOLDING"
        if _matches_any(sector_lower, _HOLDING_SECTOR_KEYWORDS):
            return "HOLDING"

        # 8. Default
        return "OPERATING"

    def classify_all(self, session: Session) -> dict:
        """Classify all active companies in the database and update company_type.

        Args:
            session: SQLAlchemy session.

        Returns:
            Stats dict: {total, updated, by_type: {type: count}}.
        """
        companies = (
            session.query(Company)
            .filter(Company.is_active.is_(True))
            .all()
        )

        stats = {
            "total": len(companies),
            "updated": 0,
            "by_type": {},
        }

        for company in companies:
            company_type = self.classify(
                ticker=company.ticker,
                company_name=company.name or "",
                sector=company.sector_bist or "",
            )

            if company.company_type != company_type:
                company.company_type = company_type
                stats["updated"] += 1

            stats["by_type"][company_type] = stats["by_type"].get(company_type, 0) + 1

        if stats["updated"] > 0:
            session.flush()

        logger.info(
            "Classified %d companies: %s (updated %d)",
            stats["total"],
            stats["by_type"],
            stats["updated"],
        )
        return stats


def _matches_any(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the keywords (case-insensitive).

    Args:
        text: Lowercased text to search in.
        keywords: List of lowercased keywords.

    Returns:
        True if any keyword is found in text.
    """
    return any(kw in text for kw in keywords)
