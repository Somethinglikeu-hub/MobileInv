"""DCF (Discounted Cash Flow) intrinsic value scorer for BIST Stock Picker.

Projects owner earnings 10 years into the future, discounts them at a
dynamic TRY cost-of-equity (policy rate + equity risk premium), adds a
Gordon Growth terminal value, and computes the margin of safety vs. the
current market price.

Key design choices for Turkish equities:
  - Discount rate: TCMB policy rate (from MacroRegime) + equity_risk_premium.
    Falls back to static ``discount_rate_try`` when macro data is unavailable
    or ``dynamic_discount_rate`` is disabled in config.
  - Terminal growth: ``terminal_growth_try`` (config, ~long-run nominal TRY).
    Re-checked against the dynamic rate so (r - g_terminal) stays positive.
  - Owner Earnings base: from AdjustedMetric (IAS 29-adjusted, D&A added back,
    maintenance capex and WC change deducted).
  - Per-share conversion: OE_per_share = OE x (eps_adjusted / adjusted_net_income).
  - Growth rate: log-linear regression of positive eps_adjusted over time
    (robust to single-year outliers), with a loss-year penalty, capped at
    [min_growth_rate, max_growth_rate].
  - Returns None for negative/zero OE, banks, or holdings (separate models).

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

from bist_picker.db.schema import AdjustedMetric, Company, DailyPrice, MacroRegime

logger = logging.getLogger("bist_picker.scoring.factors.dcf")

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"
)
_DEFAULT_MACRO_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "macro.yaml"
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

    def __init__(
        self,
        config_path: Optional[Path] = None,
        macro_config_path: Optional[Path] = None,
    ) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._macro_config_path = macro_config_path or _DEFAULT_MACRO_CONFIG_PATH
        self._cfg: dict = {}
        self._macro_cfg: dict = {}
        self._load_config()
        # Cache: (scoring_date, resolved_rate). Invalidated when scoring_date changes.
        self._rate_cache: Optional[tuple[Optional[date], float, str]] = None
        # Cache: (scoring_date, g_terminal, source).
        self._terminal_growth_cache: Optional[tuple[Optional[date], float, str]] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Run DCF valuation for a single company."""
        company = session.get(Company, company_id)
        if company is None:
            logger.warning("Company ID %d not found", company_id)
            return None

        ctype = (company.company_type or "").upper()
        if ctype in ("BANK", "HOLDING", "INSURANCE", "REIT", "SPORT", "FINANCIAL"):
            logger.debug("Skipping %s: company_type=%s (not applicable for DCF)", company.ticker, ctype)
            return None

        # Centralized point-in-time guard (audit CRITICAL #1, 2026-05-07).
        from datetime import date as _date
        from bist_picker.scoring.context import _adjusted_metric_pit_filter
        cutoff_date = scoring_date or _date.today()
        query = session.query(AdjustedMetric).filter(
            AdjustedMetric.company_id == company_id,
            _adjusted_metric_pit_filter(cutoff_date),
        )

        metrics = query.order_by(AdjustedMetric.period_end).all()

        if not metrics:
            logger.debug("Skipping %s: no adjusted metrics", company.ticker)
            return None

        oe_per_share_series = self._compute_oe_per_share_series(metrics)
        if not oe_per_share_series:
            logger.debug("Skipping %s: cannot compute OE/share (missing data)", company.ticker)
            return None

        positive_oe = [v for v in oe_per_share_series if v > 0]
        if not positive_oe:
            logger.debug("Skipping %s: all owner earnings <= 0", company.ticker)
            return None

        base_oe = positive_oe[-1]

        growth_rate = self._estimate_growth_rate(metrics)

        discount_rate, rate_source = self._resolve_discount_rate(session, scoring_date)
        g_terminal, terminal_source = self._resolve_terminal_growth(session, scoring_date)

        intrinsic = self._compute_intrinsic_value(
            base_oe, growth_rate, discount_rate, g_terminal=g_terminal,
        )
        if intrinsic is None or intrinsic <= 0:
            return None

        current_price = self._get_latest_price(company_id, session, scoring_date)

        mos_pct: Optional[float] = None
        if current_price and current_price > 0:
            raw_mos = (intrinsic - current_price) / intrinsic * 100.0
            mos_pct = max(-100.0, min(200.0, raw_mos))

        return {
            "intrinsic_value_per_share": round(intrinsic, 2),
            "base_oe_per_share": round(base_oe, 4),
            "growth_rate_used": round(growth_rate, 4),
            "discount_rate_used": round(discount_rate, 4),
            "discount_rate_source": rate_source,
            "terminal_growth_used": round(g_terminal, 4),
            "terminal_growth_source": terminal_source,
            "years_projected": self._cfg.get("projection_years", 10),
            "current_price": current_price,
            "margin_of_safety_pct": mos_pct,
            "dcf_combined": mos_pct,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _resolve_discount_rate(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> tuple[float, str]:
        """Return (discount_rate, source) for the current scoring run.

        When ``dcf.dynamic_discount_rate`` is truthy, reads the most recent
        MacroRegime.policy_rate_pct on or before ``scoring_date`` and adds
        ``dcf.equity_risk_premium_try``. Falls back to the static
        ``dcf.discount_rate_try`` when macro data is missing or dynamic mode
        is disabled.

        The spread (r - terminal_growth_try) is validated: if dynamic r would
        fall at or below terminal growth, falls back to static.

        Result is cached per scoring_date within a single scorer instance.
        """
        if self._rate_cache is not None and self._rate_cache[0] == scoring_date:
            return self._rate_cache[1], self._rate_cache[2]

        static_r = float(self._cfg.get("discount_rate_try", 0.35))
        # Terminal growth comes from _resolve_terminal_growth (dynamic + clamped).
        g_terminal, _ = self._resolve_terminal_growth(session, scoring_date)
        dynamic_enabled = bool(self._cfg.get("dynamic_discount_rate", True))
        erp = self._get_erp(session=session, scoring_date=scoring_date)
        min_spread = float(self._cfg.get("min_rate_terminal_spread", 0.05))

        if not dynamic_enabled:
            self._rate_cache = (scoring_date, static_r, "static_config")
            return static_r, "static_config"

        query = session.query(MacroRegime).filter(
            MacroRegime.policy_rate_pct.isnot(None),
        )
        if scoring_date is not None:
            query = query.filter(MacroRegime.date <= scoring_date)
        latest = query.order_by(MacroRegime.date.desc()).first()

        if latest is None or latest.policy_rate_pct is None:
            logger.info("No MacroRegime policy rate found; using static %.2f%%", static_r * 100)
            self._rate_cache = (scoring_date, static_r, "static_fallback_no_macro")
            return static_r, "static_fallback_no_macro"

        policy = float(latest.policy_rate_pct)
        # policy_rate stored as fraction (e.g., 0.425 = 42.5%)
        dyn_r = policy + erp

        if dyn_r - g_terminal < min_spread:
            logger.warning(
                "Dynamic rate %.2f%% (policy %.2f%% + ERP %.2f%%) too close to "
                "terminal growth %.2f%%; using static %.2f%%",
                dyn_r * 100, policy * 100, erp * 100, g_terminal * 100, static_r * 100,
            )
            self._rate_cache = (scoring_date, static_r, "static_fallback_thin_spread")
            return static_r, "static_fallback_thin_spread"

        logger.info(
            "DCF discount rate: %.2f%% (policy %.2f%% + ERP %.2f%%, as of %s)",
            dyn_r * 100, policy * 100, erp * 100, latest.date,
        )
        self._rate_cache = (scoring_date, dyn_r, "dynamic_policy_plus_erp")
        return dyn_r, "dynamic_policy_plus_erp"

    def _compute_oe_per_share_series(
        self, metrics: list[AdjustedMetric]
    ) -> list[float]:
        series = []
        for m in metrics:
            oe = m.owner_earnings
            ani = m.adjusted_net_income
            eps = m.eps_adjusted

            if oe is None or ani is None or eps is None:
                continue
            if ani <= 0:
                continue

            oe_per_share = oe * (eps / ani)
            series.append(oe_per_share)

        return series

    def _estimate_growth_rate(self, metrics: list[AdjustedMetric]) -> float:
        """Estimate nominal growth rate from eps_adjusted history.

        Uses log-linear regression on positive EPS observations vs. time-in-years:
            log(eps_t) = a + b * t   ->   annual growth = exp(b) - 1

        This is robust to single-year outliers in a way that endpoint-CAGR is
        not: a freak first or last year cannot dominate the estimate.

        Additionally applies a loss-year penalty based on the share of
        observations with eps <= 0: penalty = 1 - (loss_years / total) * 0.50.

        The result is clamped to [min_growth_rate, max_growth_rate] from config.
        """
        min_g = self._cfg.get("min_growth_rate", 0.05)
        max_g = self._cfg.get("max_growth_rate", 0.35)
        default_g = self._cfg.get("conservative_growth_rate", 0.10)

        all_eps = [
            (m.eps_adjusted, m.period_end)
            for m in metrics
            if m.eps_adjusted is not None
        ]
        positive = [(eps, dt) for eps, dt in all_eps if eps > 0]

        if len(positive) < 2:
            logger.debug("Insufficient positive EPS history; using conservative %.0f%%", default_g * 100)
            return max(min_g, min(max_g, default_g))

        base_date = all_eps[0][1]
        t_years = np.array([(dt - base_date).days / 365.25 for _, dt in positive], dtype=float)
        log_eps = np.array([np.log(eps) for eps, _ in positive], dtype=float)

        if t_years[-1] - t_years[0] < 0.5:
            return max(min_g, min(max_g, default_g))

        try:
            slope, _intercept = np.polyfit(t_years, log_eps, 1)
        except (np.linalg.LinAlgError, ValueError):
            return max(min_g, min(max_g, default_g))

        if not np.isfinite(slope):
            return max(min_g, min(max_g, default_g))

        # Continuous-compounding slope -> annual growth rate
        annual_growth = float(np.exp(slope) - 1.0)

        loss_years = sum(1 for eps, _ in all_eps if eps <= 0)
        loss_penalty = 1.0 - (loss_years / len(all_eps)) * 0.50
        annual_growth *= loss_penalty

        capped = max(min_g, min(max_g, annual_growth))
        logger.debug(
            "EPS log-linear growth over %.1f years (incl. %d loss years): "
            "%.1f%% -> capped to %.1f%%",
            t_years[-1] - t_years[0], loss_years, annual_growth * 100, capped * 100,
        )
        return capped

    def _compute_intrinsic_value(
        self,
        base_oe_per_share: float,
        growth_rate: float,
        discount_rate: Optional[float] = None,
        g_terminal: Optional[float] = None,
    ) -> Optional[float]:
        """Compute intrinsic value per share via N-year DCF + Gordon terminal value.

        If ``discount_rate`` is None, falls back to the static ``discount_rate_try``
        from config. Callers that want the dynamic rate should resolve it first
        via ``_resolve_discount_rate`` and pass the result explicitly.

        If ``g_terminal`` is None, falls back to the static ``terminal_growth_try``
        from config. Callers that want the dynamic rate should resolve it via
        ``_resolve_terminal_growth`` and pass the result explicitly.
        """
        if g_terminal is None:
            g_terminal = float(self._cfg.get("terminal_growth_try", 0.08))
        n = self._cfg.get("projection_years", 10)
        r = discount_rate if discount_rate is not None else float(
            self._cfg.get("discount_rate_try", 0.35)
        )

        spread = r - g_terminal
        if spread <= 0:
            logger.warning(
                "Discount rate (%.2f) <= terminal growth (%.2f); skipping DCF",
                r, g_terminal,
            )
            return None

        pv_sum = 0.0
        oe_t = base_oe_per_share
        for t in range(1, n + 1):
            oe_t = oe_t * (1.0 + growth_rate)
            pv_sum += oe_t / ((1.0 + r) ** t)

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
        """Load DCF parameters from thresholds.yaml and macro.yaml."""
        if not self._config_path.exists():
            logger.warning("thresholds.yaml not found at %s; using built-in defaults", self._config_path)
        else:
            with self._config_path.open("r", encoding="utf-8") as fh:
                full = yaml.safe_load(fh) or {}
            self._cfg = full.get("dcf", {})
            logger.debug("DCFScorer loaded dcf config: %s", self._cfg)

        if self._macro_config_path.exists():
            with self._macro_config_path.open("r", encoding="utf-8") as fh:
                self._macro_cfg = yaml.safe_load(fh) or {}
            logger.debug("DCFScorer loaded macro config: %s", self._macro_cfg)
        else:
            logger.info(
                "macro.yaml not found at %s; DCF will use thresholds.yaml "
                "equity_risk_premium_try / terminal_growth_try only",
                self._macro_config_path,
            )

    def _get_erp(
        self,
        session: Optional[Session] = None,
        scoring_date: Optional[date] = None,
    ) -> float:
        """Return the equity risk premium for the dynamic discount rate.

        Priority (2026-05-07: DB-first, YAML is pure fallback):
          1. ``MacroRegime.equity_risk_premium_pct`` on or before ``scoring_date``
             (auto-fetched from Damodaran via ``data/sources/damodaran.py``).
          2. ``config/macro.yaml`` ``erp.equity_risk_premium_try`` (legacy
             manual entry — still useful when DB is empty / first run).
          3. ``config/thresholds.yaml`` ``dcf.equity_risk_premium_try``.
          4. Hard-coded 0.06.
        """
        if session is not None:
            query = session.query(MacroRegime).filter(
                MacroRegime.equity_risk_premium_pct.isnot(None),
            )
            if scoring_date is not None:
                query = query.filter(MacroRegime.date <= scoring_date)
            latest = query.order_by(MacroRegime.date.desc()).first()
            if latest is not None and latest.equity_risk_premium_pct is not None:
                return float(latest.equity_risk_premium_pct)

        erp_block = self._macro_cfg.get("erp", {}) or {}
        if "equity_risk_premium_try" in erp_block:
            return float(erp_block["equity_risk_premium_try"])
        return float(self._cfg.get("equity_risk_premium_try", 0.06))

    def _resolve_terminal_growth(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> tuple[float, str]:
        """Return (terminal_growth, source) for the current scoring run.

        Priority:
          1. MacroRegime.inflation_expectation_24m_pct + macro.yaml real_growth_pct
          2. thresholds.yaml terminal_growth_try (static fallback)

        Clamped to macro.yaml [min_floor, max_ceiling].
        Result is cached per scoring_date within a single scorer instance.
        """
        if (
            self._terminal_growth_cache is not None
            and self._terminal_growth_cache[0] == scoring_date
        ):
            return self._terminal_growth_cache[1], self._terminal_growth_cache[2]

        static_g = float(self._cfg.get("terminal_growth_try", 0.08))
        tg_block = self._macro_cfg.get("terminal_growth", {}) or {}
        real_growth = float(tg_block.get("real_growth_pct", 0.02))
        floor = float(tg_block.get("min_floor", 0.05))
        ceiling = float(tg_block.get("max_ceiling", 0.15))

        query = session.query(MacroRegime).filter(
            MacroRegime.inflation_expectation_24m_pct.isnot(None),
        )
        if scoring_date is not None:
            query = query.filter(MacroRegime.date <= scoring_date)
        latest = query.order_by(MacroRegime.date.desc()).first()

        if latest is None or latest.inflation_expectation_24m_pct is None:
            clamped = max(floor, min(ceiling, static_g))
            logger.info(
                "No TCMB 24m inflation expectation found; terminal growth = "
                "static %.2f%% (clamped)", clamped * 100,
            )
            self._terminal_growth_cache = (scoring_date, clamped, "static_fallback")
            return clamped, "static_fallback"

        exp = float(latest.inflation_expectation_24m_pct)
        dyn_g = exp + real_growth
        clamped = max(floor, min(ceiling, dyn_g))
        logger.info(
            "DCF terminal growth: %.2f%% (24m CPI exp %.2f%% + real %.2f%%, "
            "clamped to [%.2f%%, %.2f%%], as of %s)",
            clamped * 100, exp * 100, real_growth * 100,
            floor * 100, ceiling * 100, latest.date,
        )
        self._terminal_growth_cache = (scoring_date, clamped, "dynamic_cpi_plus_real")
        return clamped, "dynamic_cpi_plus_real"
