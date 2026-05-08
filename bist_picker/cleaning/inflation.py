"""IAS 29 inflation adjustment utilities for BIST financial statements.

Handles stripping monetary gain/loss from net income, deflating nominal
values to real (constant purchasing power) using CPI, and calculating
inflation-adjusted growth rates. This is the most critical cleaning module
because all downstream scoring depends on correctly adjusted financials.

Key rules:
- NEVER use reported net income directly for ratios
- For BANKS: do NOT strip monetary gain/loss (caller's responsibility)
- If monetary gain/loss line not found: assume zero
- All YoY comparisons must be inflation-adjusted using CPI
"""

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger("bist_picker.cleaning.inflation")

# Item codes known to contain monetary gain/loss on net monetary position
_MONETARY_GAIN_LOSS_CODES = [
    "3CK",   # Bank format: "NET PARASAL POZISYON KARI/ZARARI"
    # Note: 3HCA is "Other Income Before Tax" — NOT monetary gain/loss
]

# Turkish label keywords for monetary gain/loss (case-insensitive search)
_MONETARY_TR_KEYWORDS = [
    "parasal pozisyon",
    "net parasal",
    "parasal kazanc",
    "parasal kayip",
    "parasal kar",
    "parasal zarar",
]

# English label keywords for monetary gain/loss (case-insensitive search)
_MONETARY_EN_KEYWORDS = [
    "monetary position",
    "net monetary",
    "monetary gain",
    "monetary loss",
    "gain loss on net monetary",
]

# Item codes for net income
_NET_INCOME_PARENT_CODES = ["3Z"]   # Parent Shares (preferred)
_NET_INCOME_TOTAL_CODES = ["3L"]    # NET PROFIT AFTER TAXES


class InflationAdjuster:
    """Handles IAS 29 monetary gain/loss stripping and CPI deflation.

    Usage:
        adjuster = InflationAdjuster()
        adj_income, monetary_gl = adjuster.strip_monetary_gain_loss(income_data)
        real_value = adjuster.deflate_to_real(nominal, from_date, to_date, cpi)
    """

    def strip_monetary_gain_loss(
        self, income_data: list[dict]
    ) -> tuple[Optional[float], float]:
        """Strip monetary gain/loss from net income in an income statement.

        Searches the income statement JSON for the IAS 29 "monetary gain/loss
        on net monetary position" line item using multiple code and label patterns.

        Args:
            income_data: Parsed JSON array from INCOME statement's data_json.
                Each item has keys: item_code, desc_tr, desc_eng, value.

        Returns:
            Tuple of (adjusted_net_income, monetary_gain_loss_amount).
            If net income is not found, returns (None, 0.0).
            If monetary gain/loss is not found, returns (reported_net_income, 0.0).
        """
        if not income_data:
            return None, 0.0

        reported_net_income = self._find_net_income(income_data)
        if reported_net_income is None:
            logger.debug("Net income not found in income statement")
            return None, 0.0

        monetary_gl = self._find_monetary_gain_loss(income_data)

        adjusted = reported_net_income - monetary_gl
        if monetary_gl != 0.0:
            logger.debug(
                "Stripped monetary gain/loss: reported=%.0f, monetary_gl=%.0f, adjusted=%.0f",
                reported_net_income, monetary_gl, adjusted,
            )

        return adjusted, monetary_gl

    @staticmethod
    def deflate_to_real(
        nominal_value: float,
        from_date: date,
        to_date: date,
        cpi_series: pd.Series,
    ) -> Optional[float]:
        """Convert a nominal TRY value to real (constant purchasing power).

        Formula: real = nominal * (CPI_to / CPI_from)

        Args:
            nominal_value: Nominal TRY amount.
            from_date: Date of the nominal value.
            to_date: Target date for constant purchasing power.
            cpi_series: Series with date index and CPI index values.

        Returns:
            Real (deflated) value, or None if CPI data is unavailable.
        """
        if cpi_series is None or cpi_series.empty:
            return None

        cpi_from = _get_nearest_cpi(cpi_series, from_date)
        cpi_to = _get_nearest_cpi(cpi_series, to_date)

        if cpi_from is None or cpi_to is None or cpi_from == 0:
            logger.warning(
                "CPI data unavailable for deflation: from=%s, to=%s", from_date, to_date
            )
            return None

        return nominal_value * (cpi_to / cpi_from)

    @staticmethod
    def deflate_rate(
        nominal_rate: Optional[float],
        period_end: date,
        cpi_series: pd.Series,
        lookback_months: int = 12,
    ) -> Optional[float]:
        """Convert a nominal rate (e.g. ROE, ROA) to a real rate via Fisher.

        Formula: ``real = (1 + nominal) / (1 + inflation) - 1`` where
        ``inflation`` is the trailing-12-month CPI growth ending at
        ``period_end``. Use this for nominal *return* metrics — for
        nominal level values (TRY amounts), use ``deflate_value``.

        Args:
            nominal_rate: Nominal rate as a decimal (e.g. 0.30 for 30% ROE).
            period_end: Reporting period end (e.g. 2025-12-31).
            cpi_series: CPI index series with date index.
            lookback_months: Months back for CPI YoY comparison. Default 12.

        Returns:
            Real rate as a decimal, or None if inputs / CPI insufficient.
        """
        if nominal_rate is None:
            return None
        if cpi_series is None or cpi_series.empty:
            return None

        # Approximate "1 year ago" by subtracting lookback_months * 30 days.
        # CPI series is monthly; _get_nearest_cpi tolerates the rounding.
        from datetime import timedelta
        prior_date = period_end - timedelta(days=lookback_months * 30)

        cpi_now = _get_nearest_cpi(cpi_series, period_end)
        cpi_prior = _get_nearest_cpi(cpi_series, prior_date)
        if cpi_now is None or cpi_prior is None or cpi_prior <= 0:
            return None

        inflation = (cpi_now / cpi_prior) - 1.0
        if inflation <= -1.0:
            return None

        return (1.0 + nominal_rate) / (1.0 + inflation) - 1.0

    @staticmethod
    def calculate_real_growth(
        current: float,
        previous: float,
        current_date: date,
        previous_date: date,
        cpi_series: pd.Series,
    ) -> Optional[float]:
        """Calculate real (inflation-adjusted) growth rate.

        Formula: real_growth = (1 + nominal_growth) / (1 + inflation) - 1

        Args:
            current: Current period value.
            previous: Previous period value.
            current_date: Date of current period.
            previous_date: Date of previous period.
            cpi_series: Series with date index and CPI index values.

        Returns:
            Real growth rate as a decimal (e.g., 0.083 for 8.3%),
            or None if calculation is not possible.
        """
        if previous is None or previous == 0:
            return None
        if current is None:
            return None

        nominal_growth = (current / previous) - 1.0

        if cpi_series is None or cpi_series.empty:
            logger.debug("No CPI data; returning nominal growth as-is")
            return nominal_growth

        cpi_current = _get_nearest_cpi(cpi_series, current_date)
        cpi_previous = _get_nearest_cpi(cpi_series, previous_date)

        if cpi_current is None or cpi_previous is None or cpi_previous == 0:
            logger.warning(
                "CPI data unavailable for growth calc: current=%s, previous=%s",
                current_date, previous_date,
            )
            return nominal_growth

        inflation = (cpi_current / cpi_previous) - 1.0

        if inflation == -1.0:
            return None

        real_growth = (1.0 + nominal_growth) / (1.0 + inflation) - 1.0
        return real_growth

    @staticmethod
    def is_inflation_adjusted(statement_data: list[dict]) -> bool:
        """Detect if a financial statement has been restated per IAS 29.

        Checks for the presence of a monetary gain/loss line item or
        other IAS 29 markers in the statement data.

        Args:
            statement_data: Parsed JSON array from any statement's data_json.

        Returns:
            True if the statement appears to be IAS 29 restated.
        """
        if not statement_data:
            return False

        for item in statement_data:
            code = (item.get("item_code") or "").strip()
            desc_tr = (item.get("desc_tr") or "").lower()
            desc_eng = (item.get("desc_eng") or "").lower()

            # Check known item codes
            if code in _MONETARY_GAIN_LOSS_CODES:
                val = item.get("value")
                if val is not None and val != 0:
                    return True

            # Check Turkish labels (only if value is non-zero)
            val = item.get("value")
            has_nonzero_value = val is not None and val != 0

            for kw in _MONETARY_TR_KEYWORDS:
                if kw in desc_tr and has_nonzero_value:
                    return True

            # Check English labels (only if value is non-zero)
            for kw in _MONETARY_EN_KEYWORDS:
                if kw in desc_eng and has_nonzero_value:
                    return True

        return False

    def _find_net_income(self, data: list[dict]) -> Optional[float]:
        """Find net income (parent shares) from income statement data.

        Tries 3Z (Parent Shares) first, then falls back to 3L (Net Profit).

        Args:
            data: Income statement JSON items.

        Returns:
            Net income value, or None if not found.
        """
        # Prefer parent shares (3Z)
        val = _find_item_by_codes(data, _NET_INCOME_PARENT_CODES)
        if val is not None:
            return val

        # Fallback to total net income (3L)
        val = _find_item_by_codes(data, _NET_INCOME_TOTAL_CODES)
        if val is not None:
            return val

        return None

    def _find_monetary_gain_loss(self, data: list[dict]) -> float:
        """Find the monetary gain/loss on net monetary position.

        Searches by item codes first, then by Turkish/English label keywords.
        Returns 0.0 if not found (pre-2022 or company doesn't report separately).

        Args:
            data: Income statement JSON items.

        Returns:
            Monetary gain/loss amount (positive = gain, negative = loss).
            Returns 0.0 if not found.
        """
        # Try known item codes first
        val = _find_item_by_codes(data, _MONETARY_GAIN_LOSS_CODES)
        if val is not None:
            return val

        # Try Turkish label search
        val = _find_item_by_labels(data, _MONETARY_TR_KEYWORDS, lang="tr")
        if val is not None:
            return val

        # Try English label search
        val = _find_item_by_labels(data, _MONETARY_EN_KEYWORDS, lang="en")
        if val is not None:
            return val

        logger.debug(
            "Monetary gain/loss line not found in income statement; "
            "assuming zero (normal for non-IAS 29 filers or pre-2022 periods)"
        )
        return 0.0


def _find_item_by_codes(data: list[dict], codes: list[str]) -> Optional[float]:
    """Search for a financial line item by item_code.

    Args:
        data: List of statement items with item_code and value keys.
        codes: List of item codes to search for (exact match).

    Returns:
        The value of the first matching item, or None if not found.
    """
    for item in data:
        code = (item.get("item_code") or "").strip()
        if code in codes:
            val = item.get("value")
            if val is not None:
                return float(val)
    return None


def _find_item_by_labels(
    data: list[dict], keywords: list[str], lang: str = "tr"
) -> Optional[float]:
    """Search for a financial line item by description keyword match.

    Args:
        data: List of statement items.
        keywords: List of keywords to search for (case-insensitive substring).
        lang: Language to search in ("tr" for desc_tr, "en" for desc_eng).

    Returns:
        The value of the first matching item, or None if not found.
    """
    field = "desc_tr" if lang == "tr" else "desc_eng"
    for item in data:
        desc = (item.get(field) or "").lower()
        for kw in keywords:
            if kw in desc:
                val = item.get("value")
                if val is not None:
                    return float(val)
    return None


def _get_nearest_cpi(cpi_series: pd.Series, target_date: date) -> Optional[float]:
    """Get the CPI value for the nearest available date.

    Args:
        cpi_series: Series with date index and CPI values.
        target_date: Target date to look up.

    Returns:
        CPI value, or None if series is empty.
    """
    if cpi_series is None or cpi_series.empty:
        return None

    # Convert target_date to same type as index
    if hasattr(cpi_series.index, 'tz'):
        target = pd.Timestamp(target_date)
    else:
        target = pd.Timestamp(target_date)

    # Try exact match first
    if target in cpi_series.index:
        return float(cpi_series[target])

    # Find nearest date using searchsorted
    idx = cpi_series.index.searchsorted(target)
    if idx == 0:
        return float(cpi_series.iloc[0])
    if idx >= len(cpi_series):
        return float(cpi_series.iloc[-1])

    # Return the closer of the two neighbors
    before = cpi_series.index[idx - 1]
    after = cpi_series.index[idx]
    if abs(target - before) <= abs(target - after):
        return float(cpi_series.iloc[idx - 1])
    return float(cpi_series.iloc[idx])
