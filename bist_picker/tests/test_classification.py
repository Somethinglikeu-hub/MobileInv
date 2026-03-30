"""Tests for company type classification and custom sector mapping.

Tests CompanyClassifier (company_type.py) and SectorMapper (sector_mapper.py)
with real BIST tickers and known expected classifications.
"""

from pathlib import Path

import pytest

from bist_picker.classification.company_type import CompanyClassifier
from bist_picker.classification.sector_mapper import SectorMapper


# ---- CompanyClassifier tests ----


class TestCompanyClassifier:
    """Tests for CompanyClassifier.classify()."""

    @pytest.fixture
    def classifier(self):
        return CompanyClassifier()

    def test_thyao_is_operating(self, classifier):
        """THYAO (Turkish Airlines) should be OPERATING."""
        result = classifier.classify("THYAO", "Turk Hava Yollari A.O.", "Ulastirma")
        assert result == "OPERATING"

    def test_garan_is_bank(self, classifier):
        """GARAN (Garanti Bank) should be BANK via known list."""
        result = classifier.classify("GARAN", "T. Garanti Bankasi A.S.", "Bankacilik")
        assert result == "BANK"

    def test_sahol_is_holding(self, classifier):
        """SAHOL (Sabanci Holding) should be HOLDING via known list."""
        result = classifier.classify("SAHOL", "Haci Omer Sabanci Holding A.S.", "Holding")
        assert result == "HOLDING"

    def test_ekgyo_is_reit(self, classifier):
        """EKGYO (Emlak Konut GYO) should be REIT via sector keyword."""
        result = classifier.classify("EKGYO", "Emlak Konut GYO A.S.", "GYO")
        assert result == "REIT"

    def test_all_known_banks(self, classifier):
        """All known bank tickers should classify as BANK."""
        banks = ["GARAN", "AKBNK", "YKBNK", "HALKB", "VAKBN", "ISCTR",
                 "DENIZ", "QNBFB", "SKBNK", "TSKB", "ALBRK"]
        for ticker in banks:
            result = classifier.classify(ticker, "", "")
            assert result == "BANK", f"{ticker} should be BANK"

    def test_all_known_holdings(self, classifier):
        """All known holding tickers should classify as HOLDING."""
        holdings = ["SAHOL", "KCHOL", "DOHOL", "TAVHL", "TKFEN", "ECZYT",
                     "GLYHO", "NTHOL", "KOZAL"]
        for ticker in holdings:
            result = classifier.classify(ticker, "", "")
            assert result == "HOLDING", f"{ticker} should be HOLDING"

    def test_bank_by_sector_keyword(self, classifier):
        """Unknown ticker with 'Banka' in sector -> BANK."""
        result = classifier.classify("XBNK", "Some Bank A.S.", "Bankacilik")
        assert result == "BANK"

    def test_bank_by_name_keyword(self, classifier):
        """Unknown ticker with 'Bank' in name -> BANK."""
        result = classifier.classify("XBNK", "XYZ Bankasi A.S.", "Finans")
        assert result == "BANK"

    def test_insurance_by_sector(self, classifier):
        """Sector containing 'Sigorta' -> INSURANCE."""
        result = classifier.classify("ANHYT", "Anadolu Hayat Emeklilik", "Sigorta")
        assert result == "INSURANCE"

    def test_reit_by_gyo_sector(self, classifier):
        """Sector containing 'GYO' -> REIT."""
        result = classifier.classify("ISGYO", "Is GYO A.S.", "GYO")
        assert result == "REIT"

    def test_reit_by_gayrimenkul_sector(self, classifier):
        """Sector containing 'gayrimenkul yatirim' -> REIT."""
        result = classifier.classify("XGYO", "Test GYO", "Gayrimenkul Yatirim Ortakligi")
        assert result == "REIT"

    def test_holding_by_name(self, classifier):
        """Company name containing 'Holding' -> HOLDING."""
        result = classifier.classify("XHOL", "Test Holding A.S.", "Sanayi")
        assert result == "HOLDING"

    def test_holding_by_sector(self, classifier):
        """Sector containing 'Holding' -> HOLDING."""
        result = classifier.classify("XHOL", "Test Sirketi A.S.", "Holding ve Yatirim")
        assert result == "HOLDING"

    def test_operating_default(self, classifier):
        """Unknown ticker with generic sector -> OPERATING."""
        result = classifier.classify("ASELS", "Aselsan Elektronik", "Savunma")
        assert result == "OPERATING"

    def test_bimas_is_operating(self, classifier):
        """BIMAS (BIM) should be OPERATING."""
        result = classifier.classify("BIMAS", "BIM Birlesik Magazalar A.S.", "Perakende")
        assert result == "OPERATING"

    def test_case_insensitive_sector(self, classifier):
        """Sector matching should be case-insensitive."""
        result = classifier.classify("XBNK", "Test", "BANKACILIK")
        assert result == "BANK"

    def test_case_insensitive_name(self, classifier):
        """Name matching should be case-insensitive."""
        result = classifier.classify("XHOL", "ABC HOLDING A.S.", "Sanayi")
        assert result == "HOLDING"

    def test_priority_bank_over_holding(self, classifier):
        """Known bank should take priority even if name says 'Holding'."""
        # TSKB is a known bank
        result = classifier.classify("TSKB", "Some Holding Bank", "Finans")
        assert result == "BANK"


# ---- SectorMapper tests ----


class TestSectorMapper:
    """Tests for SectorMapper.map_sector()."""

    @pytest.fixture
    def mapper(self):
        return SectorMapper()

    def test_thyao_airlines(self, mapper):
        """THYAO -> airlines via manual mapping."""
        assert mapper.map_sector("THYAO", "") == "airlines"

    def test_garan_banking_private(self, mapper):
        """GARAN -> banking_private via manual mapping."""
        assert mapper.map_sector("GARAN", "") == "banking_private"

    def test_halkb_banking_state(self, mapper):
        """HALKB -> banking_state via manual mapping."""
        assert mapper.map_sector("HALKB", "") == "banking_state"

    def test_sahol_holding_diversified(self, mapper):
        """SAHOL -> holding_diversified via manual mapping."""
        assert mapper.map_sector("SAHOL", "") == "holding_diversified"

    def test_tkfen_holding_industrial(self, mapper):
        """TKFEN -> holding_industrial via manual mapping."""
        assert mapper.map_sector("TKFEN", "") == "holding_industrial"

    def test_bimas_food_retail(self, mapper):
        """BIMAS -> food_retail via manual mapping."""
        assert mapper.map_sector("BIMAS", "") == "food_retail"

    def test_asels_defense(self, mapper):
        """ASELS -> defense via manual mapping."""
        assert mapper.map_sector("ASELS", "") == "defense"

    def test_tcell_telecom(self, mapper):
        """TCELL -> telecom via manual mapping."""
        assert mapper.map_sector("TCELL", "") == "telecom"

    def test_toaso_automotive(self, mapper):
        """TOASO -> automotive via manual mapping."""
        assert mapper.map_sector("TOASO", "") == "automotive"

    def test_eregl_steel(self, mapper):
        """EREGL -> steel via manual mapping."""
        assert mapper.map_sector("EREGL", "") == "steel"

    def test_ekgyo_real_estate(self, mapper):
        """EKGYO -> real_estate via manual mapping."""
        assert mapper.map_sector("EKGYO", "") == "real_estate"

    def test_tuprs_energy_oil_gas(self, mapper):
        """TUPRS -> energy_oil_gas via manual mapping."""
        assert mapper.map_sector("TUPRS", "") == "energy_oil_gas"

    def test_enkai_energy_power(self, mapper):
        """ENKAI -> energy_power via manual mapping."""
        assert mapper.map_sector("ENKAI", "") == "energy_power"

    def test_anhyt_insurance(self, mapper):
        """ANHYT -> insurance via manual mapping."""
        assert mapper.map_sector("ANHYT", "") == "insurance"

    def test_logo_technology_software(self, mapper):
        """LOGO -> technology_software via manual mapping."""
        assert mapper.map_sector("LOGO", "") == "technology_software"

    def test_vestl_technology_hardware(self, mapper):
        """VESTL -> technology_hardware via manual mapping."""
        assert mapper.map_sector("VESTL", "") == "technology_hardware"

    def test_akcns_cement(self, mapper):
        """AKCNS -> cement via manual mapping."""
        assert mapper.map_sector("AKCNS", "") == "cement"

    def test_ulker_food_production(self, mapper):
        """ULKER -> food_production via manual mapping."""
        assert mapper.map_sector("ULKER", "") == "food_production"

    def test_kozaa_mining(self, mapper):
        """KOZAA -> mining via manual mapping."""
        assert mapper.map_sector("KOZAA", "") == "mining"

    def test_mavi_retail_general(self, mapper):
        """MAVI -> retail_general via manual mapping."""
        assert mapper.map_sector("MAVI", "") == "retail_general"

    def test_at_least_20_manual_mappings(self, mapper):
        """Verify at least 20 tickers have manual mappings."""
        # These are all in sectors.yaml
        tickers_and_expected = {
            "THYAO": "airlines",
            "PGSUS": "airlines",
            "ASELS": "defense",
            "BIMAS": "food_retail",
            "GARAN": "banking_private",
            "HALKB": "banking_state",
            "SAHOL": "holding_diversified",
            "TKFEN": "holding_industrial",
            "TCELL": "telecom",
            "TOASO": "automotive",
            "EREGL": "steel",
            "EKGYO": "real_estate",
            "TUPRS": "energy_oil_gas",
            "ENKAI": "energy_power",
            "ANHYT": "insurance",
            "LOGO": "technology_software",
            "VESTL": "technology_hardware",
            "AKCNS": "cement",
            "ULKER": "food_production",
            "KOZAA": "mining",
            "MAVI": "retail_general",
            "CLEBI": "logistics",
            "SISE": "glass",
            "GUBRF": "chemicals",
        }
        assert len(tickers_and_expected) >= 20

        for ticker, expected in tickers_and_expected.items():
            result = mapper.map_sector(ticker, "")
            assert result == expected, f"{ticker}: expected {expected}, got {result}"

    def test_fallback_to_bist_sector(self, mapper):
        """Unknown ticker should fall back to BIST sector matching."""
        # Not in manual mapping, but sector matches "Sigorta" -> insurance
        result = mapper.map_sector("XSGR", "Sigorta")
        assert result == "insurance"

    def test_fallback_substring_match(self, mapper):
        """BIST sector fallback should work with substring matching."""
        result = mapper.map_sector("XBNK", "Bankacilik ve Finans")
        # "Banka" is a key in fallback, should match substring
        assert result == "banking_private"

    def test_unknown_defaults_to_other(self, mapper):
        """Completely unknown ticker and sector -> 'other'."""
        result = mapper.map_sector("XYZZ", "Bilinmeyen Sektor")
        assert result == "other"

    def test_case_insensitive_ticker(self, mapper):
        """Ticker lookup should be case-insensitive."""
        result = mapper.map_sector("thyao", "")
        assert result == "airlines"
