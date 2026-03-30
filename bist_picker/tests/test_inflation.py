"""Tests for IAS 29 inflation adjustment and financial metrics calculation.

Tests the InflationAdjuster class (monetary gain/loss stripping, CPI deflation,
real growth calculation) and the MetricsCalculator class (adjusted metrics
from raw financial statements).
"""

from datetime import date

import pandas as pd
import pytest

from bist_picker.cleaning.inflation import InflationAdjuster


# ---- Test data fixtures ----

def _make_income_with_monetary_gl():
    """Income statement JSON with monetary gain/loss line item."""
    return [
        {"item_code": "3C", "desc_tr": "Satis Gelirleri", "desc_eng": "Net Sales", "value": 100_000_000},
        {"item_code": "3D", "desc_tr": "BRUT KAR", "desc_eng": "GROSS PROFIT", "value": 40_000_000},
        {"item_code": "3DF", "desc_tr": "FAALIYET KARI", "desc_eng": "OPERATING PROFITS", "value": 25_000_000},
        {"item_code": "3I", "desc_tr": "VERGI ONCESI KAR", "desc_eng": "PROFIT BEFORE TAX", "value": 22_000_000},
        {"item_code": "3L", "desc_tr": "DONEM KARI", "desc_eng": "NET PROFIT AFTER TAXES", "value": 20_000_000},
        {"item_code": "3Z", "desc_tr": "Ana Ortaklik Paylari", "desc_eng": "Parent Shares", "value": 19_500_000},
        # Monetary gain/loss line (IAS 29)
        {"item_code": "3CK", "desc_tr": "NET PARASAL POZISYON KARI/ZARARI",
         "desc_eng": "Gain Loss on Net Monetary Position", "value": 5_000_000},
        {"item_code": "4B", "desc_tr": "Amortisman", "desc_eng": "Depreciation & Amortization", "value": 3_000_000},
    ]


def _make_income_without_monetary_gl():
    """Income statement JSON without monetary gain/loss line item (pre-2022)."""
    return [
        {"item_code": "3C", "desc_tr": "Satis Gelirleri", "desc_eng": "Net Sales", "value": 80_000_000},
        {"item_code": "3D", "desc_tr": "BRUT KAR", "desc_eng": "GROSS PROFIT", "value": 30_000_000},
        {"item_code": "3DF", "desc_tr": "FAALIYET KARI", "desc_eng": "OPERATING PROFITS", "value": 18_000_000},
        {"item_code": "3L", "desc_tr": "DONEM KARI", "desc_eng": "NET PROFIT AFTER TAXES", "value": 15_000_000},
        {"item_code": "3Z", "desc_tr": "Ana Ortaklik Paylari", "desc_eng": "Parent Shares", "value": 14_800_000},
        {"item_code": "4B", "desc_tr": "Amortisman", "desc_eng": "Depreciation & Amortization", "value": 2_500_000},
    ]


def _make_income_with_turkish_label_monetary():
    """Income statement with monetary gain/loss using Turkish label only (no known item code)."""
    return [
        {"item_code": "3C", "desc_tr": "Satis Gelirleri", "desc_eng": "Net Sales", "value": 50_000_000},
        {"item_code": "3L", "desc_tr": "DONEM KARI", "desc_eng": "NET PROFIT AFTER TAXES", "value": 12_000_000},
        {"item_code": "3Z", "desc_tr": "Ana Ortaklik Paylari", "desc_eng": "Parent Shares", "value": 11_800_000},
        # Monetary gain/loss with a non-standard item code but Turkish label
        {"item_code": "3XX", "desc_tr": "Net Parasal Pozisyon Kazanci/Kaybi",
         "desc_eng": None, "value": 3_000_000},
    ]


def _make_cpi_series():
    """CPI index series for testing deflation.

    Simulates: Jan 2023 CPI=1000, Jan 2024 CPI=1650 (65% inflation).
    """
    dates = pd.DatetimeIndex([
        date(2022, 12, 1),
        date(2023, 1, 1),
        date(2023, 6, 1),
        date(2023, 12, 1),
        date(2024, 1, 1),
        date(2024, 6, 1),
    ])
    values = [950.0, 1000.0, 1250.0, 1550.0, 1650.0, 1800.0]
    return pd.Series(values, index=dates, name="cpi_index")


# ---- InflationAdjuster.strip_monetary_gain_loss tests ----

class TestStripMonetaryGainLoss:

    def setup_method(self):
        self.adjuster = InflationAdjuster()

    def test_with_monetary_gl_found(self):
        """When monetary gain/loss line exists, strip it from net income."""
        data = _make_income_with_monetary_gl()
        adjusted, monetary_gl = self.adjuster.strip_monetary_gain_loss(data)

        assert monetary_gl == 5_000_000
        # adjusted = parent_shares (19.5M) - monetary_gl (5M) = 14.5M
        assert adjusted == 14_500_000

    def test_without_monetary_gl(self):
        """When no monetary gain/loss line, assume zero."""
        data = _make_income_without_monetary_gl()
        adjusted, monetary_gl = self.adjuster.strip_monetary_gain_loss(data)

        assert monetary_gl == 0.0
        # adjusted = reported parent shares
        assert adjusted == 14_800_000

    def test_with_turkish_label_only(self):
        """Find monetary gain/loss via Turkish label when item code is non-standard."""
        data = _make_income_with_turkish_label_monetary()
        adjusted, monetary_gl = self.adjuster.strip_monetary_gain_loss(data)

        assert monetary_gl == 3_000_000
        assert adjusted == 11_800_000 - 3_000_000  # 8.8M

    def test_empty_data(self):
        """Empty income data returns (None, 0.0)."""
        adjusted, monetary_gl = self.adjuster.strip_monetary_gain_loss([])
        assert adjusted is None
        assert monetary_gl == 0.0

    def test_none_data(self):
        """None income data returns (None, 0.0)."""
        adjusted, monetary_gl = self.adjuster.strip_monetary_gain_loss(None)
        assert adjusted is None
        assert monetary_gl == 0.0

    def test_no_net_income_found(self):
        """If net income line is missing entirely, return (None, 0.0)."""
        data = [
            {"item_code": "3C", "desc_tr": "Satis", "desc_eng": "Sales", "value": 100_000},
            {"item_code": "3D", "desc_tr": "Brut Kar", "desc_eng": "Gross Profit", "value": 50_000},
        ]
        adjusted, monetary_gl = self.adjuster.strip_monetary_gain_loss(data)
        assert adjusted is None
        assert monetary_gl == 0.0

    def test_negative_monetary_gl(self):
        """Negative monetary gain/loss (a loss) is subtracted correctly."""
        data = [
            {"item_code": "3Z", "desc_tr": "Ana Ortaklik", "desc_eng": "Parent Shares", "value": 10_000_000},
            {"item_code": "3CK", "desc_tr": "NET PARASAL POZISYON",
             "desc_eng": "Monetary", "value": -2_000_000},
        ]
        adjusted, monetary_gl = self.adjuster.strip_monetary_gain_loss(data)

        assert monetary_gl == -2_000_000
        # adjusted = 10M - (-2M) = 12M (loss removed, so income goes UP)
        assert adjusted == 12_000_000


# ---- InflationAdjuster.deflate_to_real tests ----

class TestDeflateToReal:

    def test_basic_deflation(self):
        """1000 TRY in Jan 2023 -> Jan 2024 with CPI 1000->1650."""
        cpi = _make_cpi_series()
        result = InflationAdjuster.deflate_to_real(
            1000.0, date(2023, 1, 1), date(2024, 1, 1), cpi
        )
        # real = 1000 * (1650 / 1000) = 1650
        assert result == pytest.approx(1650.0, rel=1e-6)

    def test_deflation_same_date(self):
        """Same from/to date should return the same value."""
        cpi = _make_cpi_series()
        result = InflationAdjuster.deflate_to_real(
            5000.0, date(2023, 6, 1), date(2023, 6, 1), cpi
        )
        assert result == pytest.approx(5000.0, rel=1e-6)

    def test_deflation_with_empty_cpi(self):
        """Empty CPI series returns None."""
        result = InflationAdjuster.deflate_to_real(
            1000.0, date(2023, 1, 1), date(2024, 1, 1), pd.Series(dtype=float)
        )
        assert result is None

    def test_deflation_with_none_cpi(self):
        """None CPI series returns None."""
        result = InflationAdjuster.deflate_to_real(
            1000.0, date(2023, 1, 1), date(2024, 1, 1), None
        )
        assert result is None

    def test_deflation_nearest_date(self):
        """When exact date not in CPI series, use nearest date."""
        cpi = _make_cpi_series()
        # Feb 2023 not in series -> should use nearest (Jan 2023 = 1000)
        result = InflationAdjuster.deflate_to_real(
            1000.0, date(2023, 2, 15), date(2024, 1, 1), cpi
        )
        # Nearest to Feb 15 is Jan 1 (1000), to_date Jan 1 2024 (1650)
        assert result == pytest.approx(1650.0, rel=1e-6)


# ---- InflationAdjuster.calculate_real_growth tests ----

class TestCalculateRealGrowth:

    def test_basic_real_growth(self):
        """30% nominal growth with 20% inflation = ~8.33% real growth."""
        cpi = pd.Series(
            [100.0, 120.0],
            index=pd.DatetimeIndex([date(2023, 12, 31), date(2024, 12, 31)]),
        )
        result = InflationAdjuster.calculate_real_growth(
            current=130.0,    # 30% nominal growth
            previous=100.0,
            current_date=date(2024, 12, 31),
            previous_date=date(2023, 12, 31),
            cpi_series=cpi,
        )
        # real = (1 + 0.30) / (1 + 0.20) - 1 = 1.3/1.2 - 1 = 0.08333...
        assert result == pytest.approx(0.08333, rel=1e-3)

    def test_negative_real_growth(self):
        """20% nominal growth with 50% inflation = negative real growth."""
        cpi = pd.Series(
            [100.0, 150.0],
            index=pd.DatetimeIndex([date(2023, 12, 31), date(2024, 12, 31)]),
        )
        result = InflationAdjuster.calculate_real_growth(
            current=120.0,    # 20% nominal
            previous=100.0,
            current_date=date(2024, 12, 31),
            previous_date=date(2023, 12, 31),
            cpi_series=cpi,
        )
        # real = (1.20 / 1.50) - 1 = -0.20
        assert result == pytest.approx(-0.20, rel=1e-3)

    def test_zero_previous_returns_none(self):
        """Zero previous value returns None (can't calculate growth)."""
        cpi = _make_cpi_series()
        result = InflationAdjuster.calculate_real_growth(
            current=100.0, previous=0.0,
            current_date=date(2024, 12, 31),
            previous_date=date(2023, 12, 31),
            cpi_series=cpi,
        )
        assert result is None

    def test_none_previous_returns_none(self):
        """None previous value returns None."""
        result = InflationAdjuster.calculate_real_growth(
            current=100.0, previous=None,
            current_date=date(2024, 12, 31),
            previous_date=date(2023, 12, 31),
            cpi_series=_make_cpi_series(),
        )
        assert result is None

    def test_none_current_returns_none(self):
        """None current value returns None."""
        result = InflationAdjuster.calculate_real_growth(
            current=None, previous=100.0,
            current_date=date(2024, 12, 31),
            previous_date=date(2023, 12, 31),
            cpi_series=_make_cpi_series(),
        )
        assert result is None

    def test_no_cpi_returns_nominal(self):
        """Without CPI data, return nominal growth as fallback."""
        result = InflationAdjuster.calculate_real_growth(
            current=130.0, previous=100.0,
            current_date=date(2024, 12, 31),
            previous_date=date(2023, 12, 31),
            cpi_series=None,
        )
        # Should return nominal growth = 0.30
        assert result == pytest.approx(0.30, rel=1e-6)


# ---- InflationAdjuster.is_inflation_adjusted tests ----

class TestIsInflationAdjusted:

    def test_statement_with_monetary_gl(self):
        """Statement with non-zero monetary gain/loss -> True."""
        data = _make_income_with_monetary_gl()
        assert InflationAdjuster.is_inflation_adjusted(data) is True

    def test_statement_without_monetary_gl(self):
        """Statement without monetary gain/loss -> False."""
        data = _make_income_without_monetary_gl()
        assert InflationAdjuster.is_inflation_adjusted(data) is False

    def test_statement_with_turkish_label(self):
        """Statement with Turkish monetary label -> True."""
        data = _make_income_with_turkish_label_monetary()
        assert InflationAdjuster.is_inflation_adjusted(data) is True

    def test_empty_data(self):
        """Empty statement -> False."""
        assert InflationAdjuster.is_inflation_adjusted([]) is False
        assert InflationAdjuster.is_inflation_adjusted(None) is False

    def test_zero_monetary_gl_code_present(self):
        """Monetary code present but value=0 -> False (not restated)."""
        data = [
            {"item_code": "3CK", "desc_tr": "NET PARASAL POZISYON KARI/ZARARI",
             "desc_eng": "Monetary Position", "value": 0},
            {"item_code": "3Z", "desc_tr": "Ana Ortaklik", "desc_eng": "Parent Shares", "value": 10_000_000},
        ]
        assert InflationAdjuster.is_inflation_adjusted(data) is False
