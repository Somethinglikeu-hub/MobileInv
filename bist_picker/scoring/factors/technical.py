"""Technical confirmation scoring factor for BIST Stock Picker.

A two-tier trend-confirmation filter combining classic and enhanced indicators.
The combined technical_score (0–100) is used as a 5–10% weight in the composite
model — it is not intended to drive stock selection on its own, only to avoid
catching falling knives on otherwise good fundamental picks.

Two-Tier Blended Scoring System
---------------------------------
When enhanced mode is DISABLED (default):
  • Uses only 3 classic signals (200-day MA, RSI, Volume)
  • technical_score = technical_score_classic
  • Maintains full backward compatibility

When enhanced mode is ENABLED (technical_enhanced.enabled: true):
  • Calculates 3 classic signals (weighted 60% by default)
  • Calculates 5 enhanced signals (weighted 40% by default)
  • technical_score = 60% classic + 40% enhanced
  • All 8 signals available in output dict

Classic Signal Definitions (3 signals)
---------------------------------------
1. above_200ma (weight 50%)
   Binary: 1.0 if latest close > 200-day SMA, else 0.0.
   Requires at least 200 rows of close data.

2. rsi_signal (weight 30%)
   RSI(14) using Wilder's smoothing.
     RSI in [30, 70]  -> 1.0   neutral zone, acceptable entry
     RSI < 30         -> 0.5   oversold — extra risk, but possible bounce
     RSI > 70         -> 0.0   overbought — chase territory, avoid
   Requires at least 15 rows of close data.

3. volume_trend (weight 20%)
   Ratio of 20-day average volume to 60-day average volume.
     ratio > 1.10   -> 1.0   rising interest
     ratio [0.90–1.10]  -> 0.5   stable
     ratio < 0.90   -> 0.0   fading interest
   Requires at least 60 rows of non-zero volume data.

Enhanced Signal Definitions (5 signals, requires enabled: true)
----------------------------------------------------------------
4. macd_signal (weight 25%)
   MACD(12,26,9) crossover and position.
     MACD > 0 AND MACD > signal_line  -> 1.0   bullish crossover
     otherwise                         -> 0.5   neutral/transition
     MACD < 0 AND MACD < signal_line  -> 0.0   bearish crossover
   Requires at least 35 rows of close data.

5. bollinger_signal (weight 20%)
   Bollinger Bands (20-day SMA ± 2σ) position.
     price in middle 50% of band width  -> 1.0   neutral zone
     price in outer zones (near bands)  -> 0.5   caution zone
     price outside bands                -> 0.0   extreme volatility
   Requires at least 20 rows of close data.

6. adx_signal (weight 20%)
   ADX(14) trend strength (not direction).
     ADX > 25.0        -> 1.0   strong trend
     ADX [20.0–25.0]   -> 0.5   developing trend
     ADX < 20.0        -> 0.0   weak/choppy trend
   Requires at least 28 rows of OHLC data.

7. obv_signal (weight 20%)
   On-Balance Volume trend (cumulative volume flow).
     OBV slope > 0.10   -> 1.0   rising volume flow
     OBV slope [−0.05 to 0.10]  -> 0.5   flat
     OBV slope < −0.05  -> 0.0   falling volume flow
   Requires at least 60 rows of close and volume data.

8. support_resistance_signal (weight 15%)
   52-week high/low proximity.
     price in lower half of 52w range  -> 1.0   near support
     price [75%–95%] of range          -> 0.5   upper zone
     price in top 5% of range          -> 0.0   at resistance
   Requires at least 252 rows of close data.

Dynamic Weight Rescaling
-------------------------
If a signal cannot be computed (insufficient data), it is excluded from
the weighted average and the remaining weights are rescaled to sum to 1.
If NO signal can be computed, score() returns None.

All thresholds are configurable via config/thresholds.yaml.
Classic signals: 'technical' section
Enhanced signals: 'technical_enhanced' section

Output keys from score()
------------------------
Classic signals (always calculated):
  above_200ma              bool | None   — True if price > 200-day SMA
  sma_200                  float | None  — 200-day simple moving average
  rsi_14                   float | None  — RSI value (0–100)
  rsi_signal               float | None  — discretised RSI (0/0.5/1.0)
  vol_ratio_20_60          float | None  — 20d / 60d volume ratio
  volume_trend             float | None  — discretised volume (0/0.5/1.0)
  technical_score_classic  float | None  — classic 3-signal score (0–100)

Enhanced signals (only when enabled: true):
  macd_value               float | None  — MACD line value
  macd_signal_line         float | None  — MACD signal line value
  macd_signal              float | None  — discretised MACD (0/0.5/1.0)
  bb_upper                 float | None  — Bollinger upper band
  bb_lower                 float | None  — Bollinger lower band
  bollinger_signal         float | None  — discretised BB (0/0.5/1.0)
  adx_value                float | None  — ADX value (0–100)
  adx_signal               float | None  — discretised ADX (0/0.5/1.0)
  obv_value                float | None  — OBV cumulative value
  obv_signal               float | None  — discretised OBV (0/0.5/1.0)
  support_52w              float | None  — 52-week low
  resistance_52w           float | None  — 52-week high
  sr_signal                float | None  — discretised S/R (0/0.5/1.0)
  technical_score_enhanced float | None  — enhanced 5-signal score (0–100)

Final output (always present):
  technical_score          float | None  — final blended score (0–100)
  components_used          int           — number of signals contributing
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from bist_picker.db.schema import Company, DailyPrice

logger = logging.getLogger("bist_picker.scoring.factors.technical")

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"
)

# Number of most-recent rows to fetch from DB (covers 200 trading days + buffer)
_QUERY_LIMIT = 300

# RSI
_RSI_PERIOD = 14
_RSI_OVERSOLD = 30.0
_RSI_OVERBOUGHT = 70.0

# Volume
_VOL_SHORT = 20   # days for short-term average
_VOL_LONG = 60    # days for long-term average


class TechnicalScorer:
    """Calculates trend-confirmation signals for a BIST company.

    Uses only DailyPrice data (close, volume); no financial statement data
    is required.  Scores are self-contained per company — no cross-sectional
    normalisation step is needed because the 0/0.5/1.0 discretisation
    already puts all companies on the same 0–100 scale.

    Args:
        config_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        path = config_path or _DEFAULT_CONFIG_PATH
        cfg = self._load_config(path).get("technical", {})

        self._weight_ma = float(cfg.get("weight_ma", 0.50))
        self._weight_rsi = float(cfg.get("weight_rsi", 0.30))
        self._weight_vol = float(cfg.get("weight_vol", 0.20))

        self._rsi_oversold = float(cfg.get("rsi_oversold", _RSI_OVERSOLD))
        self._rsi_overbought = float(cfg.get("rsi_overbought", _RSI_OVERBOUGHT))

        self._vol_rising = float(cfg.get("vol_rising_threshold", 1.10))
        self._vol_flat_lo = float(cfg.get("vol_flat_lo_threshold", 0.90))

        self._min_rows_ma = int(cfg.get("min_rows_200ma", 200))
        self._min_rows_rsi = int(cfg.get("min_rows_rsi", 15))
        self._min_rows_vol = int(cfg.get("min_rows_volume", 60))

        # Enhanced indicators configuration
        enhanced_cfg = self._load_config(path).get("technical_enhanced", {})

        self._enhanced_enabled = bool(enhanced_cfg.get("enabled", False))
        self._classic_weight = float(enhanced_cfg.get("classic_weight", 0.60))
        self._enhanced_weight = float(enhanced_cfg.get("enhanced_weight", 0.40))

        # Enhanced component weights
        self._weight_macd = float(enhanced_cfg.get("weight_macd", 0.25))
        self._weight_bb = float(enhanced_cfg.get("weight_bollinger", 0.20))
        self._weight_adx = float(enhanced_cfg.get("weight_adx", 0.20))
        self._weight_obv = float(enhanced_cfg.get("weight_obv", 0.20))
        self._weight_sr = float(enhanced_cfg.get("weight_support_resistance", 0.15))

        # MACD thresholds
        self._macd_ema_short = int(enhanced_cfg.get("macd_ema_short", 12))
        self._macd_ema_long = int(enhanced_cfg.get("macd_ema_long", 26))
        self._macd_signal_period = int(enhanced_cfg.get("macd_signal", 9))
        self._min_rows_macd = int(enhanced_cfg.get("min_rows_macd", 35))

        # Bollinger Bands thresholds
        self._bb_period = int(enhanced_cfg.get("bb_period", 20))
        self._bb_std_dev = float(enhanced_cfg.get("bb_std_dev", 2.0))
        self._bb_middle_pct = float(enhanced_cfg.get("bb_middle_pct", 0.50))
        self._min_rows_bb = int(enhanced_cfg.get("min_rows_bb", 20))

        # ADX thresholds
        self._adx_period = int(enhanced_cfg.get("adx_period", 14))
        self._adx_strong = float(enhanced_cfg.get("adx_strong", 25.0))
        self._adx_developing = float(enhanced_cfg.get("adx_developing", 20.0))
        self._min_rows_adx = int(enhanced_cfg.get("min_rows_adx", 28))

        # OBV thresholds
        self._obv_short = int(enhanced_cfg.get("obv_short_period", 20))
        self._obv_long = int(enhanced_cfg.get("obv_long_period", 60))
        self._obv_rising = float(enhanced_cfg.get("obv_rising_threshold", 0.10))
        self._obv_flat = float(enhanced_cfg.get("obv_flat_threshold", -0.05))
        self._min_rows_obv = int(enhanced_cfg.get("min_rows_obv", 60))

        # Support/Resistance thresholds
        self._sr_lookback = int(enhanced_cfg.get("sr_lookback_days", 252))
        self._sr_resistance = float(enhanced_cfg.get("sr_resistance_zone", 0.95))
        self._sr_upper = float(enhanced_cfg.get("sr_upper_zone", 0.75))
        self._min_rows_sr = int(enhanced_cfg.get("min_rows_sr", 252))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, company_id: int, session: Session, scoring_date: Optional[date] = None) -> Optional[dict]:
        """Calculate technical signals for a single company.

        Args:
            company_id: Database ID of the company.
            session: Active SQLAlchemy session.
            scoring_date: Reference date for historical analysis (PIT).
        """
        company = session.get(Company, company_id)
        if company is None:
            logger.debug("Company id=%d not found", company_id)
            return None

        ref_date = scoring_date or date.today()

        # Fetch most-recent rows up to ref_date
        rows = (
            session.query(DailyPrice)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.date <= ref_date,
            )
            .order_by(DailyPrice.date.desc())
            .limit(_QUERY_LIMIT)
            .all()
        )
        rows = list(reversed(rows))  # oldest first

        if not rows:
            return None

        # Extract close prices and volumes, replacing None with sentinel
        closes: list[float] = [r.close for r in rows if r.close is not None]
        volumes: list[Optional[float]] = [
            float(r.volume) if r.volume is not None else None for r in rows
        ]

        if not closes:
            return None

        # --- CLASSIC SIGNALS (existing 3 indicators) ---

        # Signal 1: Price vs 200-day MA
        above_200ma, sma_200 = self._calc_200ma_signal(closes)

        # Signal 2: RSI(14)
        rsi_14 = self._calc_rsi(closes)
        rsi_signal = self._rsi_to_signal(rsi_14)

        # Signal 3: Volume trend (20d vs 60d)
        vol_ratio = self._calc_volume_ratio(volumes)
        volume_trend = self._ratio_to_volume_signal(vol_ratio)

        # Combine classic signals. The MA contribution is the smooth
        # distance-based signal, not the binary above/below.
        last_close_val = closes[-1] if closes else None
        ma_signal = self._ma_distance_to_signal(last_close_val, sma_200)
        technical_score_classic, classic_used = self._combine(
            above_200ma_val=ma_signal,
            rsi_signal_val=rsi_signal,
            volume_trend_val=volume_trend,
        )

        # --- ENHANCED SIGNALS (new 5 indicators) ---

        # Initialize all to None
        macd_val, macd_sig, macd_signal = None, None, None
        bb_upper, bb_lower, bb_signal = None, None, None
        adx_val, adx_signal = None, None
        obv_val, obv_signal = None, None
        sr_high, sr_low, sr_signal = None, None, None
        technical_score_enhanced, enhanced_used = None, 0

        if self._enhanced_enabled:
            # Extract highs and lows for ADX (filter out Nones)
            highs: list[float] = [r.high for r in rows if r.high is not None]
            lows: list[float] = [r.low for r in rows if r.low is not None]

            # Signal 4: MACD
            macd_val, macd_sig, macd_signal = self._calc_macd_signal(closes)

            # Signal 5: Bollinger Bands
            bb_upper, bb_lower, bb_signal = self._calc_bollinger_signal(closes)

            # Signal 6: ADX (requires highs/lows)
            if len(highs) >= self._min_rows_adx and len(lows) >= self._min_rows_adx:
                adx_val, adx_signal = self._calc_adx_signal(highs, lows, closes)
            else:
                adx_val, adx_signal = None, None

            # Signal 7: OBV
            obv_val, obv_signal = self._calc_obv_signal(closes, volumes)

            # Signal 8: Support/Resistance
            sr_high, sr_low, sr_signal = self._calc_support_resistance_signal(closes)

            # Combine enhanced signals
            technical_score_enhanced, enhanced_used = self._combine_enhanced(
                macd_signal_val=macd_signal,
                bb_signal_val=bb_signal,
                adx_signal_val=adx_signal,
                obv_signal_val=obv_signal,
                sr_signal_val=sr_signal,
            )

        # --- BLEND CLASSIC + ENHANCED ---

        if self._enhanced_enabled and technical_score_enhanced is not None and enhanced_used > 0:
            # Blend: 60% classic + 40% enhanced (configurable)
            technical_score = round(
                technical_score_classic * self._classic_weight
                + technical_score_enhanced * self._enhanced_weight,
                2,
            )
        else:
            # Enhanced disabled or no enhanced signals available → classic only
            technical_score = technical_score_classic

        return {
            # Classic signals (existing)
            "above_200ma": above_200ma,
            "sma_200": sma_200,
            "rsi_14": rsi_14,
            "rsi_signal": rsi_signal,
            "vol_ratio_20_60": vol_ratio,
            "volume_trend": volume_trend,
            "technical_score_classic": technical_score_classic,
            "components_classic_used": classic_used,
            # Enhanced signals (new)
            "macd_value": macd_val,
            "macd_signal_line": macd_sig,
            "macd_signal": macd_signal,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_signal": bb_signal,
            "adx_value": adx_val,
            "adx_signal": adx_signal,
            "obv_value": obv_val,
            "obv_signal": obv_signal,
            "support_52w": sr_low,
            "resistance_52w": sr_high,
            "sr_signal": sr_signal,
            "technical_score_enhanced": technical_score_enhanced,
            "components_enhanced_used": enhanced_used,
            # Final blended score
            "technical_score": technical_score,
            "components_used": classic_used + enhanced_used,
        }

    def score_all(self, session: Session, scoring_date: Optional[date] = None) -> dict[int, dict]:
        """Score all active companies as of scoring_date."""
        company_ids = [
            cid
            for (cid,) in session.query(Company.id)
            .filter(Company.is_active.is_(True))
            .all()
        ]

        results: dict[int, dict] = {}
        for cid in company_ids:
            if scoring_date is None:
                result = self.score(cid, session)
            else:
                result = self.score(cid, session, scoring_date=scoring_date)
            if result is not None:
                results[cid] = result
            else:
                logger.debug("No price data for company_id=%d — skipped", cid)

        return results

    # ------------------------------------------------------------------
    # Signal calculations (pure functions on plain lists)
    # ------------------------------------------------------------------

    def _calc_200ma_signal(
        self, closes: list[float]
    ) -> tuple[Optional[bool], Optional[float]]:
        """Return (above_200ma, sma_200).

        Args:
            closes: Chronological list of closing prices (no Nones).

        Returns:
            Tuple of (True/False/None, sma_value/None).
            Binary boolean kept for the snapshot schema; the actual scoring
            uses ``_ma_distance_to_signal`` on the pct distance for a smooth
            signal instead of a cliff at the MA crossover.
        """
        if len(closes) < self._min_rows_ma:
            return None, None
        sma = sum(closes[-self._min_rows_ma:]) / self._min_rows_ma
        return closes[-1] > sma, sma

    @staticmethod
    def _ma_distance_to_signal(
        last_close: Optional[float], sma: Optional[float]
    ) -> Optional[float]:
        """Map signed % distance from 200MA to a 0–1 signal.

        Replaces the old binary ``above_200ma ? 1.0 : 0.0`` mapping. That
        cliff meant a stock one kuruş above its SMA scored the same as one
        20% above, and vice versa below. Here we linearly interpolate over
        a +/-20% band centered on the SMA, clamped to [0, 1]:

            distance = (close − sma) / sma
            signal   = clamp(0.5 + distance × 2.5, 0, 1)

        So distance +20% → 1.0, distance 0 → 0.5, distance −20% → 0.0. That
        rewards strong uptrends and penalises deep breakdowns proportionally
        while keeping values near the MA ambiguous (close to 0.5).
        """
        if last_close is None or sma is None or sma <= 0:
            return None
        distance = (last_close - sma) / sma
        signal = 0.5 + distance * 2.5
        return max(0.0, min(1.0, signal))

    def _calc_rsi(self, closes: list[float]) -> Optional[float]:
        """Calculate RSI using Wilder's smoothing (exponential moving average).

        Wilder initialises with a simple average of the first `period` changes,
        then applies exponential smoothing: avg = (prev * (n-1) + new) / n.

        Args:
            closes: Chronological list of closing prices (no Nones).

        Returns:
            RSI value 0-100, or None if not enough data.
        """
        if len(closes) < self._min_rows_rsi + 1:
            return None

        # Use all available closes for warm-up; only the final value matters
        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        period = _RSI_PERIOD

        # Seed: simple average of first `period` changes
        gains = [max(c, 0.0) for c in changes[:period]]
        losses = [abs(min(c, 0.0)) for c in changes[:period]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        # Wilder's smoothing for remaining changes
        for change in changes[period:]:
            avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
            avg_loss = (avg_loss * (period - 1) + abs(min(change, 0.0))) / period

        if avg_loss == 0.0:
            return 100.0  # pure uptrend

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _calc_volume_ratio(
        self, volumes: list[Optional[float]]
    ) -> Optional[float]:
        """Ratio of 20-day average volume to 60-day average volume.

        Args:
            volumes: Chronological list of volume values (may contain None).

        Returns:
            vol_20d_avg / vol_60d_avg, or None if not enough valid data.
        """
        valid = [v for v in volumes if v is not None and v > 0]
        if len(valid) < self._min_rows_vol:
            return None

        avg_short = sum(valid[-_VOL_SHORT:]) / _VOL_SHORT
        avg_long = sum(valid[-_VOL_LONG:]) / _VOL_LONG

        if avg_long == 0.0:
            return None

        return avg_short / avg_long

    # ------------------------------------------------------------------
    # Enhanced indicators (MACD, Bollinger, ADX, OBV, Support/Resistance)
    # ------------------------------------------------------------------

    def _calc_ema(self, values: list[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average.

        Uses standard EMA formula: EMA_today = (Price_today * k) + (EMA_yesterday * (1 - k))
        where k = 2 / (period + 1)

        Args:
            values: Chronological list of values (no Nones).
            period: EMA period (e.g., 12, 26).

        Returns:
            Final EMA value, or None if insufficient data.
        """
        if len(values) < period:
            return None

        # Seed: simple average of first `period` values
        ema = sum(values[:period]) / period
        k = 2.0 / (period + 1)

        # Apply EMA formula to remaining values
        for value in values[period:]:
            ema = (value * k) + (ema * (1 - k))

        return ema

    def _calc_macd_signal(
        self, closes: list[float]
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate MACD indicator and return signal score.

        MACD = 12-day EMA - 26-day EMA
        Signal line = 9-day EMA of MACD
        Signal mapping:
          - 1.0 = bullish (MACD > signal AND both positive)
          - 0.5 = neutral/transition (mixed signs)
          - 0.0 = bearish (MACD < signal AND both negative)

        Args:
            closes: Chronological list of closing prices (no Nones).

        Returns:
            Tuple of (macd_value, signal_line, macd_signal_score).
            All None if insufficient data.
        """
        if len(closes) < self._min_rows_macd:
            return None, None, None

        # Calculate 12-day and 26-day EMAs
        ema_short = self._calc_ema(closes, self._macd_ema_short)
        ema_long = self._calc_ema(closes, self._macd_ema_long)

        if ema_short is None or ema_long is None:
            return None, None, None

        # MACD line = fast EMA - slow EMA
        macd_line = ema_short - ema_long

        # Need to calculate EMA of MACD line (signal line)
        # Build MACD history for the last self._macd_signal_period values
        macd_history = []
        for i in range(len(closes) - self._min_rows_macd, len(closes)):
            subset = closes[: i + 1]
            short = self._calc_ema(subset, self._macd_ema_short)
            long = self._calc_ema(subset, self._macd_ema_long)
            if short is not None and long is not None:
                macd_history.append(short - long)

        if len(macd_history) < self._macd_signal_period:
            return macd_line, None, None

        # Signal line = 9-day EMA of MACD
        signal_line = self._calc_ema(macd_history, self._macd_signal_period)

        if signal_line is None:
            return macd_line, None, None

        # Map to signal score (0.0 / 0.5 / 1.0)
        if macd_line > signal_line and macd_line > 0 and signal_line > 0:
            # Strong bullish: MACD above signal, both positive
            score = 1.0
        elif macd_line < signal_line and macd_line < 0 and signal_line < 0:
            # Strong bearish: MACD below signal, both negative
            score = 0.0
        else:
            # Neutral/transition: mixed signals
            score = 0.5

        return macd_line, signal_line, score

    def _calc_bollinger_signal(
        self, closes: list[float]
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate Bollinger Bands and return signal score.

        Bollinger Bands = 20-day SMA ± (2 × standard deviation)
        Signal mapping based on price position within bands:
          - 1.0 = price in middle 50% of band width (healthy range)
          - 0.5 = price in outer 25% but inside bands (caution zone)
          - 0.0 = price outside bands (extreme, avoid)

        Args:
            closes: Chronological list of closing prices (no Nones).

        Returns:
            Tuple of (bb_upper, bb_lower, bb_signal_score).
            All None if insufficient data.
        """
        if len(closes) < self._min_rows_bb:
            return None, None, None

        # Calculate 20-day simple moving average
        recent = closes[-self._bb_period :]
        sma = sum(recent) / self._bb_period

        # Calculate standard deviation
        variance = sum((x - sma) ** 2 for x in recent) / self._bb_period
        std_dev = variance ** 0.5

        # Upper and lower bands
        bb_upper = sma + (self._bb_std_dev * std_dev)
        bb_lower = sma - (self._bb_std_dev * std_dev)

        # Current price position
        current_price = closes[-1]
        band_width = bb_upper - bb_lower

        if band_width == 0:
            # Degenerate case: no volatility (all prices identical)
            return bb_upper, bb_lower, 0.5

        # Calculate position within bands (0.0 = lower band, 1.0 = upper band)
        position = (current_price - bb_lower) / band_width

        # Map position to signal score
        if position < 0 or position > 1.0:
            # Outside bands (extreme) → avoid
            score = 0.0
        elif 0.25 <= position <= 0.75:
            # Middle 50% of bands → healthy
            score = 1.0
        else:
            # Outer 25% but inside bands → caution
            score = 0.5

        return bb_upper, bb_lower, score

    def _calc_adx_signal(
        self, highs: list[float], lows: list[float], closes: list[float]
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate ADX (Average Directional Index) and return signal score.

        ADX measures trend strength (not direction). Uses Wilder's smoothing.
        Signal mapping:
          - 1.0 = ADX > 25 (strong trend exists)
          - 0.5 = ADX 20-25 (developing trend)
          - 0.0 = ADX < 20 (weak/choppy trend, avoid)

        Args:
            highs: Chronological list of high prices (no Nones).
            lows: Chronological list of low prices (no Nones).
            closes: Chronological list of closing prices (no Nones).

        Returns:
            Tuple of (adx_value, adx_signal_score).
            Both None if insufficient data.
        """
        if len(highs) < self._min_rows_adx or len(lows) < self._min_rows_adx or len(closes) < self._min_rows_adx:
            return None, None

        period = self._adx_period

        # Calculate True Range (TR)
        tr_values = []
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return None, None

        # Calculate +DM and -DM (Directional Movement)
        plus_dm_values = []
        minus_dm_values = []
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm = up_move
            else:
                plus_dm = 0.0

            if down_move > up_move and down_move > 0:
                minus_dm = down_move
            else:
                minus_dm = 0.0

            plus_dm_values.append(plus_dm)
            minus_dm_values.append(minus_dm)

        # Smooth TR, +DM, -DM using Wilder's method (like RSI)
        # Seed with simple average
        atr = sum(tr_values[:period]) / period
        smoothed_plus_dm = sum(plus_dm_values[:period]) / period
        smoothed_minus_dm = sum(minus_dm_values[:period]) / period

        # Apply Wilder's smoothing to remaining values
        for i in range(period, len(tr_values)):
            atr = (atr * (period - 1) + tr_values[i]) / period
            smoothed_plus_dm = (smoothed_plus_dm * (period - 1) + plus_dm_values[i]) / period
            smoothed_minus_dm = (smoothed_minus_dm * (period - 1) + minus_dm_values[i]) / period

        # Calculate +DI and -DI
        if atr == 0:
            return None, None

        plus_di = 100 * (smoothed_plus_dm / atr)
        minus_di = 100 * (smoothed_minus_dm / atr)

        # Calculate DX (Directional Index)
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return None, None

        dx = 100 * abs(plus_di - minus_di) / di_sum

        # For full ADX calculation, we need to smooth DX over period days
        # Build DX history
        dx_history = []
        for end_idx in range(period, len(tr_values) + 1):
            # Recalculate smoothed values for this window
            tr_window = tr_values[:end_idx]
            plus_window = plus_dm_values[:end_idx]
            minus_window = minus_dm_values[:end_idx]

            atr_temp = sum(tr_window[:period]) / period
            plus_temp = sum(plus_window[:period]) / period
            minus_temp = sum(minus_window[:period]) / period

            for j in range(period, len(tr_window)):
                atr_temp = (atr_temp * (period - 1) + tr_window[j]) / period
                plus_temp = (plus_temp * (period - 1) + plus_window[j]) / period
                minus_temp = (minus_temp * (period - 1) + minus_window[j]) / period

            if atr_temp > 0:
                di_p = 100 * (plus_temp / atr_temp)
                di_m = 100 * (minus_temp / atr_temp)
                di_sum_temp = di_p + di_m
                if di_sum_temp > 0:
                    dx_history.append(100 * abs(di_p - di_m) / di_sum_temp)

        if len(dx_history) < period:
            return None, None

        # ADX = Wilder's smoothed average of DX
        adx = sum(dx_history[:period]) / period
        for dx_val in dx_history[period:]:
            adx = (adx * (period - 1) + dx_val) / period

        # Map ADX to signal score
        if adx >= self._adx_strong:
            score = 1.0
        elif adx >= self._adx_developing:
            score = 0.5
        else:
            score = 0.0

        return adx, score

    def _calc_obv_signal(
        self, closes: list[float], volumes: list[float]
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate OBV (On-Balance Volume) trend and return signal score.

        OBV is cumulative volume flow (add on up days, subtract on down days).
        Signal based on slope comparison (20d vs 60d).
        Signal mapping:
          - 1.0 = OBV slope rising (volume supporting price)
          - 0.5 = OBV slope flat (neutral volume)
          - 0.0 = OBV slope falling (divergence warning)

        Args:
            closes: Chronological list of closing prices (no Nones).
            volumes: Chronological list of volume values (may contain None).

        Returns:
            Tuple of (obv_current, obv_signal_score).
            Both None if insufficient data.
        """
        # Filter out None volumes
        valid_data = [(c, v) for c, v in zip(closes, volumes) if v is not None and v > 0]

        if len(valid_data) < self._min_rows_obv:
            return None, None

        # Calculate OBV
        obv = 0.0
        obv_values = []

        for i, (close, volume) in enumerate(valid_data):
            if i == 0:
                obv_values.append(obv)
            else:
                prev_close = valid_data[i - 1][0]
                if close > prev_close:
                    obv += volume
                elif close < prev_close:
                    obv -= volume
                # If close == prev_close, obv stays same
                obv_values.append(obv)

        # Calculate slope of OBV (simple linear regression slope approximation)
        # Slope = (mean(y) of last N - mean(y) of prev N) / N
        recent_short = obv_values[-self._obv_short :]
        recent_long = obv_values[-self._obv_long :]

        avg_short = sum(recent_short) / len(recent_short)
        avg_long = sum(recent_long) / len(recent_long)

        # Normalize by length to get slope per period
        slope = (avg_short - avg_long) / (self._obv_long - self._obv_short)

        # Map slope to signal score
        if slope >= self._obv_rising:
            score = 1.0
        elif slope >= self._obv_flat:
            score = 0.5
        else:
            score = 0.0

        return obv_values[-1], score

    def _calc_support_resistance_signal(
        self, closes: list[float]
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate Support/Resistance levels and return signal score.

        Uses 52-week (252-day) high/low as resistance/support.
        Signal mapping based on price position in range:
          - 1.0 = price near support or in middle range (good entry)
          - 0.5 = price in upper quartile (approaching resistance)
          - 0.0 = price at/near 52-week high (poor risk/reward)

        Args:
            closes: Chronological list of closing prices (no Nones).

        Returns:
            Tuple of (resistance_52w, support_52w, sr_signal_score).
            All None if insufficient data.
        """
        if len(closes) < self._min_rows_sr:
            return None, None, None

        # Use last sr_lookback days
        recent = closes[-self._sr_lookback :]

        support_52w = min(recent)
        resistance_52w = max(recent)
        current_price = closes[-1]

        range_width = resistance_52w - support_52w

        if range_width == 0:
            # Degenerate case: all prices identical in lookback
            return resistance_52w, support_52w, 0.5

        # Calculate position within range (0.0 = support, 1.0 = resistance)
        position = (current_price - support_52w) / range_width

        # Map position to signal score
        if position >= self._sr_resistance:
            # At resistance (top 5%) → avoid
            score = 0.0
        elif position >= self._sr_upper:
            # Upper zone (top 25%) → caution
            score = 0.5
        else:
            # Lower 75% → good entry zone
            score = 1.0

        return resistance_52w, support_52w, score

    # ------------------------------------------------------------------
    # Signal-to-score mappers
    # ------------------------------------------------------------------

    def _rsi_to_signal(self, rsi: Optional[float]) -> Optional[float]:
        """Map RSI value to a contrarian entry-quality signal in [0, 1].

        The old mapping penalised oversold RSI with 0.5, which is
        *opposite* of what value investing wants: an oversold quality
        name is a classic contrarian setup, not a weaker one. New tiers:

            rsi > 70          → 0.0    overbought, poor entry point
            30 <= rsi <= 70   → 1.0    neutral / healthy tape
            rsi < 30          → 0.85   oversold — contrarian BUY signal
                                       (not 1.0, because an oversold name
                                        that keeps falling is a knife, but
                                        clearly better than neutral-but-
                                        expensive for long-term value picks)

        Returns None if rsi is None.
        """
        if rsi is None:
            return None
        if rsi > self._rsi_overbought:
            return 0.0
        if rsi < self._rsi_oversold:
            return 0.85
        return 1.0

    def _ratio_to_volume_signal(self, ratio: Optional[float]) -> Optional[float]:
        """Map 20d/60d volume ratio to a 0/0.5/1.0 signal.

        Returns None if ratio is None.
        """
        if ratio is None:
            return None
        if ratio >= self._vol_rising:
            return 1.0   # rising interest
        if ratio >= self._vol_flat_lo:
            return 0.5   # stable
        return 0.0        # fading interest

    # ------------------------------------------------------------------
    # Combination
    # ------------------------------------------------------------------

    def _combine(
        self,
        above_200ma_val: Optional[float],
        rsi_signal_val: Optional[float],
        volume_trend_val: Optional[float],
    ) -> tuple[float, int]:
        """Combine available signals into a 0-100 score.

        If one or more components are unavailable (None), the remaining
        weights are rescaled to sum to 1.0.

        Args:
            above_200ma_val: MA signal (0.0 or 1.0), or None.
            rsi_signal_val:  RSI signal (0/0.5/1.0), or None.
            volume_trend_val: Volume signal (0/0.5/1.0), or None.

        Returns:
            (technical_score 0-100, number of components used).
        """
        parts: list[tuple[float, float]] = []  # (value, weight)

        if above_200ma_val is not None:
            parts.append((above_200ma_val, self._weight_ma))
        if rsi_signal_val is not None:
            parts.append((rsi_signal_val, self._weight_rsi))
        if volume_trend_val is not None:
            parts.append((volume_trend_val, self._weight_vol))

        if not parts:
            return 0.0, 0

        total_weight = sum(w for _, w in parts)
        raw = sum(v * w for v, w in parts) / total_weight  # rescaled weighted avg
        return round(raw * 100.0, 2), len(parts)

    def _combine_enhanced(
        self,
        macd_signal_val: Optional[float],
        bb_signal_val: Optional[float],
        adx_signal_val: Optional[float],
        obv_signal_val: Optional[float],
        sr_signal_val: Optional[float],
    ) -> tuple[float, int]:
        """Combine enhanced signals into a 0-100 score.

        Same pattern as _combine(): dynamic weight rescaling when components
        are unavailable (None).

        Args:
            macd_signal_val: MACD signal (0/0.5/1.0), or None.
            bb_signal_val: Bollinger Band signal (0/0.5/1.0), or None.
            adx_signal_val: ADX signal (0/0.5/1.0), or None.
            obv_signal_val: OBV signal (0/0.5/1.0), or None.
            sr_signal_val: Support/Resistance signal (0/0.5/1.0), or None.

        Returns:
            (technical_score_enhanced 0-100, number of components used).
        """
        parts: list[tuple[float, float]] = []  # (value, weight)

        if macd_signal_val is not None:
            parts.append((macd_signal_val, self._weight_macd))
        if bb_signal_val is not None:
            parts.append((bb_signal_val, self._weight_bb))
        if adx_signal_val is not None:
            parts.append((adx_signal_val, self._weight_adx))
        if obv_signal_val is not None:
            parts.append((obv_signal_val, self._weight_obv))
        if sr_signal_val is not None:
            parts.append((sr_signal_val, self._weight_sr))

        if not parts:
            return 0.0, 0

        total_weight = sum(w for _, w in parts)
        raw = sum(v * w for v, w in parts) / total_weight  # rescaled weighted avg
        return round(raw * 100.0, 2), len(parts)

    # ------------------------------------------------------------------
    # Config loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> dict:
        """Load YAML config, returning empty dict on any error."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("Config file not found: %s — using defaults", path)
            return {}
        except yaml.YAMLError as exc:
            logger.error("Failed to parse config %s: %s", path, exc)
            return {}
