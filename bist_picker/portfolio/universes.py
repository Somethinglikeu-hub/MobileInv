"""Universe construction for the BIST Stock Picker.

Defines which companies are eligible for each portfolio by applying
investability and quality filters on top of the model scores.

ALPHA now has an explicit production contract:
  - company_type == OPERATING
  - risk_tier != HIGH
  - free_float >= 25%
  - avg_turnover >= 10M TRY
  - data_completeness >= 70%
  - raw Piotroski >= 5
  - latest adjusted net income and owner earnings are not both <= 0

The dashboard also uses this module to explain why a name misses ALPHA
Core and to place it into research buckets such as Quality Shadow,
Free-Float Shadow, Non-Core Research, or Data-Unscorable.
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from bist_picker.db.schema import AdjustedMetric, Company, DailyPrice, ScoringResult

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"
)

_UNIVERSE_CONFIG: dict[str, dict] = {
    "ALPHA": {
        "risk_tier": None,
        "exclude_high_risk": True,
        "ipo_eligible": False,
        "bist100_only": False,
        "free_float_min": 25.0,
        "avg_volume_try_min": 10_000_000,
        "data_completeness_min": 70.0,
        "fscore_min": 5,
    },
    "BETA": {
        "risk_tier": "MEDIUM",
        "exclude_high_risk": False,
        "ipo_eligible": True,
        "bist100_only": False,
        "free_float_min": 25.0,
        "avg_volume_try_min": 10_000_000,
        "data_completeness_min": 60.0,
        "fscore_min": 3,
    },
    "DELTA": {
        "risk_tier": None,
        "exclude_high_risk": False,
        "ipo_eligible": False,
        "bist100_only": True,
        "free_float_min": 30.0,
        "avg_volume_try_min": 25_000_000,
        "data_completeness_min": 60.0,
        "fscore_min": 3,
    },
}

_LOOKBACK_DAYS: int = 30
_IPO_MIN_AGE_MONTHS: int = 3

_ALPHA_CORE_TYPES = {"OPERATING"}
_ALPHA_NON_CORE_RESEARCH_TYPES = {
    "BANK",
    "FINANCIAL",
    "HOLDING",
    "REIT",
    "INSURANCE",
}
_ALPHA_CORE_BUCKET = "ALPHA Core"
_ALPHA_QUALITY_SHADOW_BUCKET = "Quality Shadow"
_ALPHA_FREE_FLOAT_SHADOW_BUCKET = "Free-Float Shadow"
_ALPHA_NON_CORE_BUCKET = "Non-Core Research"
_ALPHA_DATA_UNSCORABLE_BUCKET = "Data-Unscorable"
_ALPHA_EXCLUDED_BUCKET = "Excluded"
_ALPHA_CORE_ONLY_REASON = "ALPHA Core sadece OPERATING"

_ALPHA_RESEARCH_DEFAULTS: dict[str, float | int] = {
    "quality_shadow_raw_fscore": 4,
    "free_float_shadow_min": 15.0,
    "free_float_shadow_alpha_min": 85.0,
    "scenario_relaxed_operating_fscore": 4,
}


class UniverseBuilder:
    """Build the eligible stock universe for a given portfolio."""

    def __init__(self, scoring_date: Optional[date] = None) -> None:
        self.scoring_date: date = scoring_date or date.today()
        self._full_cfg = self._load_config()
        self._filters_cfg = self._full_cfg.get("filters", {}) or {}
        self._alpha_research_cfg = self._load_alpha_research_config()

    def _load_config(self) -> dict:
        """Load thresholds.yaml when available."""
        if not _DEFAULT_THRESHOLDS_PATH.exists():
            return {}
        try:
            with _DEFAULT_THRESHOLDS_PATH.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            return {}

    @staticmethod
    def _normalize_pct(value: Optional[float], fallback: float) -> float:
        """Accept both 0-1 and 0-100 style percentage configs."""
        if value is None:
            return fallback
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        return parsed * 100.0 if 0.0 <= parsed <= 1.0 else parsed

    def _load_alpha_research_config(self) -> dict[str, float | int]:
        """Load ALPHA research helper thresholds from thresholds.yaml."""
        cfg = dict(_ALPHA_RESEARCH_DEFAULTS)
        overrides = self._full_cfg.get("alpha_research", {}) or {}

        if "quality_shadow_raw_fscore" in overrides:
            try:
                cfg["quality_shadow_raw_fscore"] = int(
                    overrides.get("quality_shadow_raw_fscore")
                )
            except (TypeError, ValueError):
                pass

        if "free_float_shadow_min" in overrides:
            cfg["free_float_shadow_min"] = self._normalize_pct(
                overrides.get("free_float_shadow_min"),
                float(cfg["free_float_shadow_min"]),
            )

        if "free_float_shadow_alpha_min" in overrides:
            try:
                cfg["free_float_shadow_alpha_min"] = float(
                    overrides.get("free_float_shadow_alpha_min")
                )
            except (TypeError, ValueError):
                pass

        if "scenario_relaxed_operating_fscore" in overrides:
            try:
                cfg["scenario_relaxed_operating_fscore"] = int(
                    overrides.get("scenario_relaxed_operating_fscore")
                )
            except (TypeError, ValueError):
                pass

        return cfg

    def _portfolio_config(self, portfolio: str) -> dict:
        """Merge YAML overrides onto the repo defaults for a portfolio."""
        cfg = dict(_UNIVERSE_CONFIG[portfolio])
        overrides = self._filters_cfg.get(portfolio.lower(), {}) or {}

        if "risk_tier" in overrides:
            cfg["risk_tier"] = overrides.get("risk_tier")
        if "exclude_high_risk" in overrides:
            cfg["exclude_high_risk"] = bool(overrides.get("exclude_high_risk"))
        if "ipo_eligible" in overrides:
            cfg["ipo_eligible"] = bool(overrides.get("ipo_eligible"))
        if "bist100_only" in overrides:
            cfg["bist100_only"] = bool(overrides.get("bist100_only"))
        if "min_free_float" in overrides:
            cfg["free_float_min"] = self._normalize_pct(
                overrides.get("min_free_float"),
                cfg["free_float_min"],
            )
        if "min_avg_volume_try" in overrides:
            cfg["avg_volume_try_min"] = float(
                overrides.get("min_avg_volume_try", cfg["avg_volume_try_min"])
            )
        if "min_data_completeness" in overrides:
            cfg["data_completeness_min"] = self._normalize_pct(
                overrides.get("min_data_completeness"),
                cfg["data_completeness_min"],
            )
        if "min_fscore" in overrides:
            cfg["fscore_min"] = int(overrides.get("min_fscore", cfg["fscore_min"]))

        return cfg

    def get_universe(
        self,
        portfolio: str,
        session: Session,
        exact_date: bool = False,
    ) -> list[int]:
        """Return company_ids eligible for *portfolio*."""
        portfolio = portfolio.upper()
        if portfolio not in _UNIVERSE_CONFIG:
            raise ValueError(
                f"Unknown portfolio {portfolio!r}. Valid options: {sorted(_UNIVERSE_CONFIG)}"
            )

        cfg = self._portfolio_config(portfolio)
        logger.info(
            "Building universe for %s (scoring_date=%s, exact_date=%s)",
            portfolio,
            self.scoring_date,
            exact_date,
        )

        scored = self._get_scores(session, exact_date=exact_date)
        avg_volumes = self._get_avg_volumes(session)
        latest_metrics = self._get_latest_adjusted_metrics(session)

        eligible: list[int] = []
        for company_id, score_row, company in scored:
            vol = avg_volumes.get(company_id, 0.0)
            if self._passes_filters(
                company=company,
                score=score_row,
                latest_metric=latest_metrics.get(company_id),
                avg_volume_try=vol,
                cfg=cfg,
                portfolio=portfolio,
            ):
                eligible.append(company_id)

        logger.info(
            "Universe for %s: %d eligible (from %d scored companies)",
            portfolio,
            len(eligible),
            len(scored),
        )
        return eligible

    def get_universe_diagnostics(
        self,
        portfolio: str,
        session: Session,
        exact_date: bool = False,
    ) -> dict[int, dict[str, object]]:
        """Return eligibility diagnostics for scored active companies."""
        portfolio = portfolio.upper()
        if portfolio not in _UNIVERSE_CONFIG:
            raise ValueError(
                f"Unknown portfolio {portfolio!r}. Valid options: {sorted(_UNIVERSE_CONFIG)}"
            )

        cfg = self._portfolio_config(portfolio)
        scored = self._get_scores(session, exact_date=exact_date)
        avg_volumes = self._get_avg_volumes(session)
        latest_metrics = self._get_latest_adjusted_metrics(session)

        diagnostics: dict[int, dict[str, object]] = {}
        for company_id, score_row, company in scored:
            avg_volume_try = avg_volumes.get(company_id, 0.0)
            reasons = self._failure_reasons(
                company=company,
                score=score_row,
                latest_metric=latest_metrics.get(company_id),
                avg_volume_try=avg_volume_try,
                cfg=cfg,
                portfolio=portfolio,
            )
            entry: dict[str, object] = {
                "eligible": len(reasons) == 0,
                "reasons": reasons,
            }
            if portfolio == "ALPHA":
                entry.update(
                    self._alpha_diagnostics(
                        company=company,
                        score=score_row,
                        latest_metric=latest_metrics.get(company_id),
                        avg_volume_try=avg_volume_try,
                        cfg=cfg,
                        reasons=reasons,
                    )
                )
            diagnostics[company_id] = entry
        return diagnostics

    def _get_scores(
        self,
        session: Session,
        exact_date: bool = False,
    ) -> list[tuple[int, "ScoringResult", "Company"]]:
        """Return score rows for the requested snapshot behavior."""
        if exact_date:
            return (
                session.query(ScoringResult.company_id, ScoringResult, Company)
                .join(Company, Company.id == ScoringResult.company_id)
                .filter(
                    ScoringResult.scoring_date == self.scoring_date,
                    Company.is_active == True,
                )
                .all()
            )
        return self._get_latest_scores(session)

    def _get_latest_scores(
        self, session: Session
    ) -> list[tuple[int, "ScoringResult", "Company"]]:
        """Return one latest-on-or-before score row per active company."""
        subq = (
            session.query(
                ScoringResult.company_id,
                func.max(ScoringResult.scoring_date).label("max_date"),
            )
            .filter(ScoringResult.scoring_date <= self.scoring_date)
            .group_by(ScoringResult.company_id)
            .subquery()
        )

        return (
            session.query(ScoringResult.company_id, ScoringResult, Company)
            .join(
                subq,
                (ScoringResult.company_id == subq.c.company_id)
                & (ScoringResult.scoring_date == subq.c.max_date),
            )
            .join(Company, Company.id == ScoringResult.company_id)
            .filter(Company.is_active == True)
            .all()
        )

    def _get_avg_volumes(self, session: Session) -> dict[int, float]:
        """Return average daily TRY turnover per company."""
        cutoff = self.scoring_date - timedelta(days=_LOOKBACK_DAYS)

        turnover_expr = case(
            (DailyPrice.source.ilike("YAHOO%"), DailyPrice.close * DailyPrice.volume),
            else_=DailyPrice.volume,
        )

        rows = (
            session.query(
                DailyPrice.company_id,
                func.avg(turnover_expr).label("avg_turnover"),
            )
            .filter(DailyPrice.date >= cutoff)
            .filter(DailyPrice.date <= self.scoring_date)
            .filter(DailyPrice.close.isnot(None))
            .filter(DailyPrice.volume.isnot(None))
            .group_by(DailyPrice.company_id)
            .all()
        )
        return {row.company_id: float(row.avg_turnover or 0.0) for row in rows}

    def _get_latest_adjusted_metrics(
        self,
        session: Session,
    ) -> dict[int, "AdjustedMetric"]:
        """Return the latest adjusted metrics row on or before scoring_date."""
        subq = (
            session.query(
                AdjustedMetric.company_id,
                func.max(AdjustedMetric.period_end).label("max_period_end"),
            )
            .filter(AdjustedMetric.period_end <= self.scoring_date)
            .group_by(AdjustedMetric.company_id)
            .subquery()
        )

        rows = (
            session.query(AdjustedMetric.company_id, AdjustedMetric)
            .join(
                subq,
                (AdjustedMetric.company_id == subq.c.company_id)
                & (AdjustedMetric.period_end == subq.c.max_period_end),
            )
            .all()
        )
        return {company_id: metric for company_id, metric in rows}

    def _passes_filters(
        self,
        company: "Company",
        score: "ScoringResult",
        latest_metric: Optional["AdjustedMetric"],
        avg_volume_try: float,
        cfg: dict,
        portfolio: str = "ALPHA",
    ) -> bool:
        """Return True if *company* passes all filters defined in *cfg*."""
        return len(
            self._failure_reasons(
                company=company,
                score=score,
                latest_metric=latest_metric,
                avg_volume_try=avg_volume_try,
                cfg=cfg,
                portfolio=portfolio,
            )
        ) == 0

    def _failure_reasons(
        self,
        company: "Company",
        score: "ScoringResult",
        latest_metric: Optional["AdjustedMetric"],
        avg_volume_try: float,
        cfg: dict,
        portfolio: str = "ALPHA",
    ) -> list[str]:
        """Return human-readable reasons why *company* fails *portfolio* filters."""
        reasons: list[str] = []
        company_type = self._company_type(company)

        if cfg["bist100_only"] and not company.is_bist100:
            reasons.append("BIST-100 disi")

        if portfolio == "ALPHA" and company_type not in _ALPHA_CORE_TYPES:
            if company_type == "SPORT":
                reasons.append("SPORT disarida")
            else:
                reasons.append(_ALPHA_CORE_ONLY_REASON)

        if cfg["risk_tier"] is not None and score.risk_tier != cfg["risk_tier"]:
            is_eligible_ipo = (
                cfg["ipo_eligible"]
                and company.is_ipo
                and (company.ipo_age_months or 0) > _IPO_MIN_AGE_MONTHS
            )
            if not is_eligible_ipo:
                reasons.append(f"Risk tier != {cfg['risk_tier']}")

        if cfg.get("exclude_high_risk") and score.risk_tier == "HIGH":
            reasons.append("Risk tier = HIGH")

        if company.free_float_pct is None:
            reasons.append("Halka aciklik verisi yok")
        elif company.free_float_pct < cfg["free_float_min"]:
            reasons.append(f"Halka aciklik <%{cfg['free_float_min']:.0f}")

        if avg_volume_try < cfg["avg_volume_try_min"]:
            reasons.append(
                f"Likidite < {cfg['avg_volume_try_min'] / 1_000_000:.0f}M TRY"
            )

        if score.data_completeness is None:
            reasons.append("Veri kapsami yok")
        elif score.data_completeness < cfg["data_completeness_min"]:
            reasons.append(f"Veri kapsami <%{cfg['data_completeness_min']:.0f}")

        if portfolio == "ALPHA":
            reasons.extend(self._alpha_profitability_failure_reasons(latest_metric))

        raw_fscore = getattr(score, "piotroski_fscore_raw", None)
        if raw_fscore is not None:
            if raw_fscore < cfg["fscore_min"]:
                reasons.append(f"Piotroski < {cfg['fscore_min']}")
        elif score.piotroski_fscore is None or score.piotroski_fscore < cfg["fscore_min"]:
            if company_type in {"BANK", "FINANCIAL", "HOLDING", "REIT", "SPORT"}:
                reasons.append(f"Piotroski yok ({company_type} modeli)")
            elif portfolio == "ALPHA":
                reasons.append("Piotroski yok")
            else:
                reasons.append(f"Piotroski < {cfg['fscore_min']}")

        if company_type == "SPORT" and "SPORT disarida" not in reasons:
            reasons.append("SPORT disarida")

        return reasons

    @staticmethod
    def _company_type(company: "Company") -> str:
        """Return the normalized company type."""
        return (getattr(company, "company_type", None) or "").upper()

    @staticmethod
    def _has_usable_score_signal(score: "ScoringResult") -> bool:
        """Return True when a score row carries at least one usable signal."""
        fields = (
            "buffett_score",
            "graham_score",
            "piotroski_fscore",
            "piotroski_fscore_raw",
            "magic_formula_rank",
            "lynch_peg_score",
            "dcf_margin_of_safety_pct",
            "momentum_score",
            "insider_score",
            "technical_score",
            "dividend_score",
            "banking_composite",
            "holding_composite",
            "reit_composite",
            "composite_alpha",
        )
        return any(getattr(score, field, None) is not None for field in fields)

    def _is_data_unscorable(self, score: Optional["ScoringResult"]) -> bool:
        """Return True when the snapshot exists but is not strategically usable."""
        if score is None:
            return True
        completeness = getattr(score, "data_completeness", None)
        if completeness is None or completeness <= 0:
            return True
        return not self._has_usable_score_signal(score)

    def _alpha_score(
        self,
        company: "Company",
        score: Optional["ScoringResult"],
    ) -> Optional[float]:
        """Return the effective ALPHA score used by dashboard diagnostics."""
        if score is None:
            return None
        if self._company_type(company) == "SPORT":
            return None
        return getattr(score, "composite_alpha", None)

    @staticmethod
    def _alpha_profitability_failure_reasons(
        latest_metric: Optional["AdjustedMetric"],
    ) -> list[str]:
        """Return ALPHA-only blockers for latest earnings plus cash-generation stress."""
        if latest_metric is None:
            return []

        adjusted_net_income = getattr(latest_metric, "adjusted_net_income", None)
        owner_earnings = getattr(latest_metric, "owner_earnings", None)
        if adjusted_net_income is None or owner_earnings is None:
            return []
        if adjusted_net_income > 0 or owner_earnings > 0:
            return []
        return ["Son donem zarar", "Owner earnings <= 0"]

    def _passes_alpha_common_gates(
        self,
        company: "Company",
        score: "ScoringResult",
        latest_metric: Optional["AdjustedMetric"],
        avg_volume_try: float,
        cfg: dict,
    ) -> bool:
        """Return True when non-Piotroski ALPHA Core gates all pass."""
        if self._company_type(company) != "OPERATING":
            return False
        if getattr(score, "risk_tier", None) == "HIGH":
            return False

        free_float = getattr(company, "free_float_pct", None)
        if free_float is None or free_float < cfg["free_float_min"]:
            return False

        if avg_volume_try < cfg["avg_volume_try_min"]:
            return False

        completeness = getattr(score, "data_completeness", None)
        if completeness is None or completeness < cfg["data_completeness_min"]:
            return False

        if self._alpha_profitability_failure_reasons(latest_metric):
            return False

        return True

    def _is_quality_shadow(
        self,
        company: "Company",
        score: Optional["ScoringResult"],
        latest_metric: Optional["AdjustedMetric"],
        avg_volume_try: float,
        cfg: dict,
    ) -> bool:
        """Return True when a name only misses ALPHA Core by raw Piotroski 4."""
        if score is None:
            return False
        if not self._passes_alpha_common_gates(
            company,
            score,
            latest_metric,
            avg_volume_try,
            cfg,
        ):
            return False
        return getattr(score, "piotroski_fscore_raw", None) == int(
            self._alpha_research_cfg["quality_shadow_raw_fscore"]
        )

    def _is_free_float_shadow(
        self,
        company: "Company",
        score: Optional["ScoringResult"],
        latest_metric: Optional["AdjustedMetric"],
        avg_volume_try: float,
        cfg: dict,
    ) -> bool:
        """Return True when a name only misses ALPHA Core by free-float."""
        if score is None:
            return False
        if self._alpha_profitability_failure_reasons(latest_metric):
            return False

        if self._company_type(company) != "OPERATING":
            return False
        if getattr(score, "risk_tier", None) == "HIGH":
            return False

        free_float = getattr(company, "free_float_pct", None)
        if free_float is None:
            return False
        if free_float < float(self._alpha_research_cfg["free_float_shadow_min"]):
            return False
        if free_float >= cfg["free_float_min"]:
            return False

        if avg_volume_try < cfg["avg_volume_try_min"]:
            return False

        completeness = getattr(score, "data_completeness", None)
        if completeness is None or completeness < cfg["data_completeness_min"]:
            return False

        raw_fscore = getattr(score, "piotroski_fscore_raw", None)
        if raw_fscore is None or raw_fscore < cfg["fscore_min"]:
            return False

        alpha_score = self._alpha_score(company, score)
        if alpha_score is None:
            return False
        return alpha_score >= float(self._alpha_research_cfg["free_float_shadow_alpha_min"])

    def _passes_alpha_relaxed_p4(
        self,
        company: "Company",
        score: Optional["ScoringResult"],
        latest_metric: Optional["AdjustedMetric"],
        avg_volume_try: float,
        cfg: dict,
    ) -> bool:
        """Return True when a name passes the comparison-only Piotroski-4 scenario."""
        if score is None:
            return False
        if not self._passes_alpha_common_gates(
            company,
            score,
            latest_metric,
            avg_volume_try,
            cfg,
        ):
            return False
        raw_fscore = getattr(score, "piotroski_fscore_raw", None)
        if raw_fscore is None:
            return False
        return raw_fscore >= int(self._alpha_research_cfg["scenario_relaxed_operating_fscore"])

    def _is_non_core_research(
        self,
        company: "Company",
        score: Optional["ScoringResult"],
    ) -> bool:
        """Return True when a scored non-operating name belongs in research only."""
        if self._company_type(company) not in _ALPHA_NON_CORE_RESEARCH_TYPES:
            return False
        return self._alpha_score(company, score) is not None

    def _alpha_diagnostics(
        self,
        company: "Company",
        score: "ScoringResult",
        latest_metric: Optional["AdjustedMetric"],
        avg_volume_try: float,
        cfg: dict,
        reasons: list[str],
    ) -> dict[str, object]:
        """Return ALPHA Core status, blockers, and research bucket metadata."""
        core_eligible = len(reasons) == 0
        research_bucket = _ALPHA_EXCLUDED_BUCKET
        primary_blocker: Optional[str] = reasons[0] if reasons else None

        if core_eligible:
            research_bucket = _ALPHA_CORE_BUCKET
            primary_blocker = None
        elif self._is_data_unscorable(score):
            research_bucket = _ALPHA_DATA_UNSCORABLE_BUCKET
            primary_blocker = _ALPHA_DATA_UNSCORABLE_BUCKET
        elif self._is_quality_shadow(company, score, latest_metric, avg_volume_try, cfg):
            research_bucket = _ALPHA_QUALITY_SHADOW_BUCKET
            primary_blocker = f"Piotroski < {cfg['fscore_min']}"
        elif self._is_free_float_shadow(
            company,
            score,
            latest_metric,
            avg_volume_try,
            cfg,
        ):
            research_bucket = _ALPHA_FREE_FLOAT_SHADOW_BUCKET
            primary_blocker = f"Halka aciklik <%{cfg['free_float_min']:.0f}"
        elif self._is_non_core_research(company, score):
            research_bucket = _ALPHA_NON_CORE_BUCKET
            primary_blocker = reasons[0] if reasons else _ALPHA_CORE_ONLY_REASON

        return {
            "alpha_core_eligible": core_eligible,
            "alpha_primary_blocker": primary_blocker,
            "alpha_all_blockers": reasons,
            "alpha_research_bucket": research_bucket,
            "alpha_relaxed_p4_eligible": self._passes_alpha_relaxed_p4(
                company=company,
                score=score,
                latest_metric=latest_metric,
                avg_volume_try=avg_volume_try,
                cfg=cfg,
            ),
        }
