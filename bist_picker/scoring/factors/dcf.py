"""DCF (Discounted Cash Flow) intrinsic value scorer for BIST Stock Picker.

Projects owner earnings 10 years into the future, discounts them at the
Turkey TRY discount rate (~35%), adds a Gordon Growth terminal value, and
computes the margin of safety vs. the current market price.

Key design choices for Turkish equities:
  - Discount rate: ~35% TRY (covers currency risk + country risk + real rate)
  - Terminal growth: ~12% TRY (~3% real + ~9% long-run structural inflation)
  - Owner Earnings base: from AdjustedMetric (IAS 29-adjusted, D&A added back,
    maintenance capex and WC change deducted)
  - Per-share conversion: OE_per_share = OE x (eps_adjusted / adjusted_net_income)
  - Growth rate: nominal CAGR of eps_adjusted over available history, capped at
    max_growth_rate from config (default 30%)
  - Returns None for negative/zero OE, banks, or holdings (separate models)

Output column: dcf_margin_of_safety_pct in ScoringResult
  Positive  -> stock is undervalued (MoS > 0)
  Negative  -> stock is overvalued  (MoS < 0)
  None      -> insufficient data or negative OE

All parameters come from config/thresholds.yaml section 'dcf'.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sqlalchemy.orm import Session

from bist_picker.db.schema import AdjustedMetric, Company, DailyPrice

logger = logging.getLogger("bist_picker.scoring.factors.dcf")

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"
)

# Company types this scorer is applicable to
_APPLICABLE_TYPES = {"OPERATING", None, ""}

# Minimum years of positive OE history required before projecting
_MIN_YEARS = 1
# Minimum coverage of sub-scores needed for a valid combined score
_MIN_WEIGHT_COVERAGE = 0.50


class DCFScorer:
    """Computes DCF intrinsic value and margin of safety for OPERATING companies.

    Skips BANK, HOLDING, INSURANCE, and REIT companies, which require
    separate valuation models.

    Args:
        config_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._cfg: dict = {}
        self._load_config()

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Run DCF valuation for a single company.

        Args:
            company_id: Database ID of the company.
            session: Active SQLAlchemy session.
            scoring_date: PiT date.

        Returns:
            Dict with DCF outputs, or None if the company is ineligible or
            has insufficient data. Dict keys:

              intrinsic_value_per_share  (float, TRY)
              base_oe_per_share          (float, TRY) - most recent annual OE/share
              growth_rate_used           (float) - fraction, e.g. 0.15 for 15%
              years_projected            (int)
              current_price              (float | None, TRY)
              margin_of_safety_pct       (float | None) - positive = undervalued
              dcf_combined               (float | None) - same as margin_of_safety_pct;
                                           stored in ScoringResult.dcf_margin_of_safety_pct
        """
        company = session.get(Company, company_id)
        if company is None:
            logger.warning("Company ID %d not found", company_id)
            return None

        ctype = (company.company_type or "").upper()
        if ctype in ("BANK", "HOLDING", "INSURANCE", "REIT", "SPORT", "FINANCIAL"):
            logger.debug("Skipping %s: company_type=%s (not applicable for DCF)", company.ticker, ctype)
            return None

        # Load adjusted metrics ordered oldest -> newest.
        # Filter out future periods with all-null data, and handle the fact
        # that publication_date is always NULL (IsYatirim doesn't provide it).
        from datetime import date as _date, timedelta
        cutoff_date = scoring_date or _date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        query = session.query(AdjustedMetric).filter(
            AdjustedMetric.company_id == company_id,
            AdjustedMetric.period_end <= lagged_cutoff,
        )

        metrics = query.order_by(AdjustedMetric.period_end).all()

        if not metrics:
            logger.debug("Skipping %s: no adjusted metrics", company.ticker)
            return None

        # Compute per-share owner earnings for each year
        oe_per_share_series = self._compute_oe_per_share_series(metrics)
        if not oe_per_share_series:
            logger.debug("Skipping %s: cannot compute OE/share (missing data)", company.ticker)
            return None

        # Need at least one positive OE year
        positive_oe = [v for v in oe_per_share_series if v > 0]
        if not positive_oe:
            logger.debug("Skipping %s: all owner earnings <= 0", company.ticker)
            return None

        base_oe = positive_oe[-1]  # Most recent positive OE/share

        # Estimate nominal growth rate from historical eps_adjusted CAGR
        growth_rate = self._estimate_growth_rate(metrics)

        # Project and discount owner earnings
        intrinsic = self._compute_intrinsic_value(base_oe, growth_rate)
        if intrinsic is None or intrinsic <= 0:
            return None

        # Get current market price
        current_price = self._get_latest_price(company_id, session, scoring_date)

        # Margin of safety (percentage), clamped to [-100, +200] to prevent
        # extreme outliers from distorting composite scores.
        mos_pct: Optional[float] = None
        if current_price and current_price > 0:
            raw_mos = (intrinsic - current_price) / intrinsic * 100.0
            mos_pct = max(-100.0, min(200.0, raw_mos))

        return {
            "intrinsic_value_per_share": round(intrinsic, 2),
            "base_oe_per_share": round(base_oe, 4),
            "growth_rate_used": round(growth_rate, 4),
            "years_projected": self._cfg.get("projection_years", 10),
            "current_price": current_price,
            "margin_of_safety_pct": mos_pct,
            "dcf_combined": mos_pct,  # stored in ScoringResult.dcf_margin_of_safety_pct
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_oe_per_share_series(
        self, metrics: list[AdjustedMetric]
    ) -> list[float]:
        """Return per-share owner earnings for each year with sufficient data.

        Per-share OE = owner_earnings * (eps_adjusted / adjusted_net_income).
        This is dimensionally consistent because all three values come from
        the same unit base (the financial statements) and the ratio is
        unitless, leaving the result in TRY/share.

        Years where adjusted_net_income <= 0 or eps_adjusted is None are skipped.
        """
        series = []
        for m in metrics:
            oe = m.owner_earnings
            ani = m.adjusted_net_income
            eps = m.eps_adjusted

            if oe is None or ani is None or eps is None:
                continue
            if ani <= 0:
                # Cannot derive shares from negative net income
                continue

            oe_per_share = oe * (eps / ani)
            series.append(oe_per_share)

        return series

    def _estimate_growth_rate(self, metrics: list[AdjustedMetric]) -> float:
        """Estimate nominal growth rate from historical eps_adjusted CAGR.

        Uses positive-EPS endpoints for the CAGR ratio, but counts the FULL
        calendar span including loss years. Additionally applies a loss-year
        penalty: growth is reduced by (loss_years / total_years) * 50%.

        The result is capped at [min_growth_rate, max_growth_rate] from config.
        """
        min_g = self._cfg.get("min_growth_rate", 0.05)
        max_g = self._cfg.get("max_growth_rate", 0.35)
        default_g = self._cfg.get("conservative_growth_rate", 0.10)

        # ALL EPS observations (including negative) for full date span
        all_eps = [
            (m.eps_adjusted, m.period_end)
            for m in metrics
            if m.eps_adjusted is not None
        ]

        # Positive EPS only — used for CAGR endpoints
        eps_pairs = [(eps, dt) for eps, dt in all_eps if eps > 0]

        if len(eps_pairs) < 2:
            logger.debug("Insufficient EPS history; using conservative growth %.0f%%", default_g * 100)
            return max(min_g, min(max_g, default_g))

        first_eps, _ = eps_pairs[0]
        last_eps, _ = eps_pairs[-1]

        # Use FULL date span (including loss years) for the CAGR denominator
        n_years = (all_eps[-1][1] - all_eps[0][1]).days / 365.25
        if n_years < 0.5:
            return max(min_g, min(max_g, default_g))

        try:
            cagr = (last_eps / first_eps) ** (1.0 / n_years) - 1.0
        except (ZeroDivisionError, ValueError):
            return max(min_g, min(max_g, default_g))

        if not np.isfinite(cagr):
            return max(min_g, min(max_g, default_g))

        # Loss-year penalty: reduce growth when many years had negative EPS.
        # E.g., 2 loss years out of 5 total → penalty = 1 - (0.4 * 0.5) = 0.80
        loss_years = sum(1 for eps, _ in all_eps if eps <= 0)
        loss_penalty = 1.0 - (loss_years / len(all_eps)) * 0.50
        cagr = cagr * loss_penalty

        capped = max(min_g, min(max_g, cagr))
        logger.debug(
            "EPS CAGR over %.1f years (incl. %d loss years): %.1f%% -> capped to %.1f%%",
            n_years, loss_years, cagr * 100, capped * 100,
        )
        return capped

    def _compute_intrinsic_value(
        self, base_oe_per_share: float, growth_rate: float
    ) -> Optional[float]:
        """Compute intrinsic value per share via 10-year DCF + terminal value.

        Formula:
          PV = sum(OE_t / (1+r)^t,  t=1..N)
          TV = OE_N * (1+g_terminal) / (r - g_terminal)
          PV_TV = TV / (1+r)^N
          intrinsic = PV + PV_TV

        where:
          r   = discount_rate_try  (config)
          g   = growth_rate        (estimated from history)
          g_t = terminal_growth_try (config)
          N   = projection_years   (config, default 10)

        Returns None if the spread (r - g_terminal) would be zero or negative,
        which would cause a division by zero in the terminal value formula.
        """
        r = self._cfg.get("discount_rate_try", 0.35)
        g_terminal = self._cfg.get("terminal_growth_try", 0.12)
        n = self._cfg.get("projection_years", 10)

        spread = r - g_terminal
        if spread <= 0:
            logger.warning(
                "Discount rate (%.2f) <= terminal growth (%.2f); skipping DCF",
                r, g_terminal,
            )
            return None

        # Project owner earnings: OE_t = base * (1+g)^t
        pv_sum = 0.0
        oe_t = base_oe_per_share
        for t in range(1, n + 1):
            oe_t = oe_t * (1.0 + growth_rate)
            pv_sum += oe_t / ((1.0 + r) ** t)

        # Terminal value at end of projection period (OE after year N)
        # oe_t here is OE at year N
        tv = oe_t * (1.0 + g_terminal) / spread
        pv_tv = tv / ((1.0 + r) ** n)

        return pv_sum + pv_tv

    def _get_latest_price(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """Return the most recent adjusted_close (or close) for a company."""
        query = session.query(DailyPrice).filter(
            DailyPrice.company_id == company_id,
            DailyPrice.adjusted_close.isnot(None),
        )

        if scoring_date:
            query = query.filter(DailyPrice.date <= scoring_date)
            
        row = query.order_by(DailyPrice.date.desc()).first()
        if row:
            return row.adjusted_close

        query = session.query(DailyPrice).filter(
            DailyPrice.company_id == company_id,
            DailyPrice.close.isnot(None),
        )

        if scoring_date:
            query = query.filter(DailyPrice.date <= scoring_date)

        row = query.order_by(DailyPrice.date.desc()).first()
        return row.close if row else None

    def _load_config(self) -> None:
        """Load DCF parameters from thresholds.yaml."""
        if not self._config_path.exists():
            logger.warning("thresholds.yaml not found at %s; using built-in defaults", self._config_path)
            return
        with self._config_path.open("r", encoding="utf-8") as fh:
            full = yaml.safe_load(fh) or {}
        self._cfg = full.get("dcf", {})
        logger.debug("DCFScorer loaded config: %s", self._cfg)
