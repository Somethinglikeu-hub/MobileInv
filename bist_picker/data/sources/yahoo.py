"""Yahoo Finance data source client.

Provides BIST stock price data via yfinance as a fallback/validation source.
BIST tickers use the .IS suffix on Yahoo Finance (e.g., THYAO.IS).

Primary uses:
- Cross-validate Is Yatirim price data
- Fetch BIST 100 index history
- Fallback price source when Is Yatirim is unavailable
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from bist_picker.data.cache import FileCache
from bist_picker.utils.rate_limiter import RateLimiter


def _yf():
    """Lazy import of yfinance.

    yfinance pulls in protobuf, which has C-extension issues on Python
    3.14 (TypeError: Metaclasses with custom tp_new). Importing it at
    module level breaks the whole pipeline even when the macro/Damodaran
    paths never need Yahoo. Importing on first use scopes the breakage
    to the methods that actually need yfinance.
    """
    import yfinance as yf
    return yf

logger = logging.getLogger("bist_picker.data.sources.yahoo")

_BIST_SUFFIX = ".IS"
_BIST100_SYMBOL = "XU100.IS"


class YahooClient:
    """Client for fetching BIST data from Yahoo Finance via yfinance.

    Args:
        rate_limiter: RateLimiter instance. Default 0.5s delay.
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._rate_limiter = rate_limiter or RateLimiter(
            min_delay=0.5, name="yahoo"
        )
        self._cache = FileCache()

    def fetch_price_data(
        self, ticker: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Fetch historical price data for a BIST ticker.

        Uses yfinance with .IS suffix. Returns data in the same column
        format as IsYatirimClient.fetch_price_data() for easy comparison.

        Args:
            ticker: BIST ticker code (e.g., "THYAO"). No .IS suffix needed.
            start_date: Start date for historical data.
            end_date: End date for historical data.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume,
            adjusted_close, source. Empty DataFrame if fetch fails.
        """
        ticker = ticker.upper()
        yahoo_ticker = f"{ticker}{_BIST_SUFFIX}"

        self._rate_limiter.wait()

        try:
            yf_ticker = _yf().Ticker(yahoo_ticker)
            # yfinance end_date is exclusive, so add 1 day
            hist = yf_ticker.history(
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
            )
        except Exception as e:
            logger.warning(
                "Failed to fetch Yahoo prices for %s: %s", yahoo_ticker, e
            )
            return self._empty_price_df()

        if hist.empty:
            logger.warning("No Yahoo price data for %s", yahoo_ticker)
            return self._empty_price_df()

        return self._convert_yf_dataframe(hist, ticker)

    def validate_prices(
        self, isyatirim_df: pd.DataFrame, ticker: str
    ) -> dict:
        """Compare Is Yatirim prices against Yahoo Finance for validation.

        Fetches Yahoo prices for the same date range and compares closing
        prices. Flags divergences exceeding the threshold (default 2%).

        Args:
            isyatirim_df: Price DataFrame from IsYatirimClient.
            ticker: BIST ticker code.

        Returns:
            Dict with: match_pct, max_divergence, mean_divergence,
            divergent_dates (list), total_compared, is_valid (bool).
        """
        if isyatirim_df.empty:
            return {
                "match_pct": 0.0,
                "max_divergence": 0.0,
                "mean_divergence": 0.0,
                "divergent_dates": [],
                "total_compared": 0,
                "is_valid": False,
            }

        # Get date range from isyatirim data
        dates = pd.to_datetime(isyatirim_df["date"])
        start_date = dates.min().date()
        end_date = dates.max().date()

        yahoo_df = self.fetch_price_data(ticker, start_date, end_date)
        if yahoo_df.empty:
            logger.warning(
                "Cannot validate %s: no Yahoo data available", ticker
            )
            return {
                "match_pct": 0.0,
                "max_divergence": 0.0,
                "mean_divergence": 0.0,
                "divergent_dates": [],
                "total_compared": 0,
                "is_valid": False,
            }

        # Prepare for merge — normalize dates to date only
        isy = isyatirim_df.copy()
        isy["merge_date"] = pd.to_datetime(isy["date"]).dt.date

        yah = yahoo_df.copy()
        yah["merge_date"] = pd.to_datetime(yah["date"]).dt.date

        merged = pd.merge(
            isy[["merge_date", "close"]],
            yah[["merge_date", "close"]],
            on="merge_date",
            suffixes=("_isy", "_yah"),
        )

        if merged.empty:
            return {
                "match_pct": 0.0,
                "max_divergence": 0.0,
                "mean_divergence": 0.0,
                "divergent_dates": [],
                "total_compared": 0,
                "is_valid": False,
            }

        # Calculate divergence as absolute percentage difference
        merged["divergence"] = abs(
            (merged["close_isy"] - merged["close_yah"]) / merged["close_isy"]
        )

        threshold = 0.02  # 2% divergence threshold
        divergent = merged[merged["divergence"] > threshold]
        divergent_dates = [
            d.isoformat() for d in divergent["merge_date"].tolist()
        ]

        total = len(merged)
        matching = total - len(divergent)

        result = {
            "match_pct": matching / total if total > 0 else 0.0,
            "max_divergence": float(merged["divergence"].max()),
            "mean_divergence": float(merged["divergence"].mean()),
            "divergent_dates": divergent_dates,
            "total_compared": total,
            "is_valid": len(divergent) == 0,
        }

        if divergent_dates:
            logger.info(
                "%s price validation: %.1f%% match, %d divergent days (>2%%)",
                ticker,
                result["match_pct"] * 100,
                len(divergent_dates),
            )

        return result

    def fetch_index_data(
        self,
        index: str = _BIST100_SYMBOL,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fetch BIST 100 (or other) index history from Yahoo Finance.

        Args:
            index: Yahoo Finance symbol. Default "XU100.IS" (BIST 100).
            start_date: Start date. Default 2 years ago.
            end_date: End date. Default today.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume,
            adjusted_close, source. Empty DataFrame if fetch fails.
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=730)

        self._rate_limiter.wait()

        try:
            yf_ticker = _yf().Ticker(index)
            hist = yf_ticker.history(
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
            )
        except Exception as e:
            logger.warning(
                "Failed to fetch Yahoo index data for %s: %s", index, e
            )
            return self._empty_price_df()

        if hist.empty:
            logger.warning("No Yahoo index data for %s", index)
            return self._empty_price_df()

        # Use index symbol as source identifier
        source = index.replace(_BIST_SUFFIX, "").replace(".", "_")
        return self._convert_yf_dataframe(hist, source)

    def fetch_fx_rates(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fetch USD/TRY and EUR/TRY exchange rates from Yahoo Finance.

        Used as fallback when TCMB EVDS is unavailable.

        Args:
            start_date: Start date. Default 2 years ago.
            end_date: End date. Default today.

        Returns:
            DataFrame with columns: date, usd_try, eur_try.
            Empty DataFrame if fetch fails.
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=730)

        usd_df = self._fetch_currency_pair("USDTRY=X", start_date, end_date)
        eur_df = self._fetch_currency_pair("EURTRY=X", start_date, end_date)

        if usd_df.empty and eur_df.empty:
            return pd.DataFrame(columns=["date", "usd_try", "eur_try"])

        # Merge USD and EUR data on date
        if not usd_df.empty and not eur_df.empty:
            merged = pd.merge(
                usd_df[["date", "close"]].rename(columns={"close": "usd_try"}),
                eur_df[["date", "close"]].rename(columns={"close": "eur_try"}),
                on="date",
                how="outer",
            )
        elif not usd_df.empty:
            merged = usd_df[["date", "close"]].rename(
                columns={"close": "usd_try"}
            )
            merged["eur_try"] = None
        else:
            merged = eur_df[["date", "close"]].rename(
                columns={"close": "eur_try"}
            )
            merged["usd_try"] = None

        merged = merged.sort_values("date").reset_index(drop=True)
        logger.info(
            "Yahoo FX fallback: %d rows (USD/TRY + EUR/TRY)", len(merged)
        )
        return merged

    def _fetch_currency_pair(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Fetch a single currency pair from Yahoo Finance.

        Args:
            symbol: Yahoo Finance symbol (e.g., 'USDTRY=X').
            start_date: Start date.
            end_date: End date.

        Returns:
            DataFrame with date and close columns, or empty DataFrame.
        """
        self._rate_limiter.wait()
        try:
            yf_ticker = _yf().Ticker(symbol)
            hist = yf_ticker.history(
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
            )
        except Exception as e:
            logger.warning(
                "Failed to fetch Yahoo FX data for %s: %s", symbol, e
            )
            return pd.DataFrame()

        if hist.empty:
            logger.warning("No Yahoo FX data for %s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame()
        df["date"] = (
            hist.index.tz_localize(None) if hist.index.tz else hist.index
        )
        df["close"] = hist["Close"].values
        return df.sort_values("date").reset_index(drop=True)

    # --- Private helpers ---

    @staticmethod
    def _convert_yf_dataframe(
        hist: pd.DataFrame, source_label: str
    ) -> pd.DataFrame:
        """Convert yfinance history DataFrame to our standard format.

        yfinance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        with a DatetimeIndex (possibly timezone-aware).

        Args:
            hist: yfinance history DataFrame.
            source_label: Source identifier for the source column.

        Returns:
            DataFrame with standard columns.
        """
        df = pd.DataFrame()
        # Normalize timezone-aware index to date
        df["date"] = hist.index.tz_localize(None) if hist.index.tz else hist.index
        df["open"] = hist["Open"].values
        df["high"] = hist["High"].values
        df["low"] = hist["Low"].values
        df["close"] = hist["Close"].values
        df["volume"] = hist["Volume"].values.astype(int)
        df["adjusted_close"] = hist["Close"].values
        df["source"] = f"YAHOO_{source_label}"

        df = df.sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _empty_price_df() -> pd.DataFrame:
        """Return an empty DataFrame with the expected price columns."""
        return pd.DataFrame(
            columns=[
                "date", "open", "high", "low", "close",
                "volume", "adjusted_close", "source",
            ]
        )
