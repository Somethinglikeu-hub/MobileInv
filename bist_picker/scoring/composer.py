"""Score composition module for the BIST Stock Picker.

Combines individual normalized factor scores (0-100 each) into a single
weighted composite score (0-100) per portfolio type (Alpha / Beta / Delta).

Weight schemes are loaded from config/scoring_weights.yaml. Weights must sum
to 1.0 per section — this is validated on load and will raise on failure.

Missing factors (None) have their weight redistributed proportionally among
the factors that do have a score, so the composite always reflects whatever
data is available rather than silently defaulting to zero.

Model-type routing:
  OPERATING  -> portfolio-specific weights (alpha / beta / delta)
  BANK       -> banking weights (same composite for all portfolios)
  HOLDING    -> holding weights (same composite for all portfolios)
  IPO        -> ipo weights (same composite for all portfolios)
"""

import copy
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from sqlalchemy.orm import Session

from bist_picker.db.schema import Company, ScoringResult
from bist_picker.scoring.normalizer import ScoreNormalizer

logger = logging.getLogger(__name__)

# Default path to the weights config (relative to this file's package root).
_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_weights.yaml"

# Portfolios that use model-type-specific weight sections regardless of
# which portfolio the company is being scored for.
# REIT → uses holding weights (NAV / dividend focus)
# INSURANCE → treated as OPERATING (no special section; uses portfolio weights)
_MODEL_OVERRIDES = {
    "BANK": "banking",
    "INSURANCE": "banking",
    "HOLDING": "holding",
    "IPO": "ipo",
    "REIT": "holding",  # REITs: NAV/dividend-focused, same as holdings
    "FINANCIAL": "banking",  # leasing, factoring, brokerage → banking model
}

# How ScoringResult DB columns map to the weight key names used in the YAML.
# Composite weight keys (growth, value_graham_dcf) are derived by averaging
# the listed column names. When a column is None it is simply excluded from
# the average, which may cascade to the factor being None if all sub-columns
# are None (triggering weight redistribution in compose()).
_DB_COL_TO_FACTOR: dict[str, list[str]] = {
    # Single-column mappings
    "quality_buffett": ["buffett_score"],
    "piotroski": ["piotroski_fscore"],
    "momentum": ["momentum_score"],
    "technical": ["technical_score"],
    # Composite mappings (averaged from multiple columns)
    "growth": ["magic_formula_rank", "lynch_peg_score"],
    "value_graham_dcf": ["graham_score", "dcf_margin_of_safety_pct"],
    # Banking model — individual sub-factors are NOT stored as separate
    # ScoringResult columns. Instead, BankingScorer pre-computes a single
    # banking_composite score, which the composer uses directly (see
    # compose_all lines 388-421). Empty lists here are intentional.
    "pb_vs_sector": [],
    "nim": [],
    "npl_ratio": [],
    "car": [],
    "cost_income": [],
    "loan_growth": [],
    "roe": [],
    # Holding model
    "nav_discount": [],
    "portfolio_quality": [],
    "dividend_yield": ["dividend_score"],
    "governance": [],
    # IPO model
    "revenue_growth": [],
    "gross_margin_vs_sector": [],
    "ps_ratio_vs_sector": [],
    "insider_retention": [],
    "liquidity": [],
}


class ScoreComposer:
    """Composes weighted composite scores from individual normalized factor scores.

    Usage::

        composer = ScoreComposer()
        score = composer.compose(
            company_id=1,
            factor_scores={"quality_buffett": 75.0, "momentum": 60.0, ...},
            portfolio="alpha",
            model_type="OPERATING",
        )
    """

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        """Load and validate scoring weights from YAML.

        Args:
            weights_path: Path to scoring_weights.yaml. Defaults to
                bist_picker/config/scoring_weights.yaml.

        Raises:
            FileNotFoundError: If the weights file cannot be found.
            ValueError: If any weight section does not sum to 1.0.
        """
        path = weights_path or _DEFAULT_WEIGHTS_PATH
        self.weights = self._load_weights(path)
        self._validate_weights(self.weights)
        logger.debug("ScoreComposer loaded weights from %s", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_weights(self, path: Path) -> dict:
        """Load scoring_weights.yaml and return the parsed dict.

        Args:
            path: Absolute path to the YAML file.

        Returns:
            Parsed YAML as a nested dict.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"scoring_weights.yaml not found at {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _validate_weights(self, weights: dict) -> None:
        """Assert every portfolio section sums to 1.0 (+/- 1e-6 tolerance).

        Skips non-portfolio sections like buffett_subfactors (they are
        sub-weightings used internally by buffett.py, not by the composer).

        Args:
            weights: Parsed YAML dict.

        Raises:
            ValueError: If any section's weights do not sum to 1.0.
        """
        # Sections that must sum to 1.0
        required_sections = {"alpha", "beta", "delta", "banking", "holding", "ipo"}
        for section in required_sections:
            if section not in weights:
                logger.warning("scoring_weights.yaml missing section '%s'", section)
                continue
            total = sum(weights[section].values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"scoring_weights.yaml: section '{section}' weights sum to "
                    f"{total:.6f}, expected 1.0"
                )
        
        # Validate regime weights if present
        if "regime_weights" in weights:
            for regime, r_weights in weights["regime_weights"].items():
                total = sum(r_weights.values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"scoring_weights.yaml: regime_weights '{regime}' sum to "
                        f"{total:.6f}, expected 1.0"
                    )

    # Valid model types: overrides + types that use portfolio-based weights.
    _VALID_MODEL_TYPES = frozenset(_MODEL_OVERRIDES) | {"OPERATING", "SPORT"}

    def _get_weight_section(self, portfolio: str, model_type: str) -> str:
        """Return the YAML section name for the given portfolio/model combination.

        Args:
            portfolio: 'alpha', 'beta', or 'delta'.
            model_type: 'OPERATING', 'BANK', 'FINANCIAL', 'INSURANCE',
                        'HOLDING', 'REIT', 'IPO', or 'SPORT'.

        Returns:
            YAML section name string.

        Raises:
            ValueError: If model_type is not a recognised company type.
        """
        if model_type in _MODEL_OVERRIDES:
            return _MODEL_OVERRIDES[model_type]
        if model_type not in ("OPERATING", "SPORT"):
            raise ValueError(
                f"Unknown model_type {model_type!r}. Must be one of: "
                f"{sorted(self._VALID_MODEL_TYPES)}"
            )
        return portfolio.lower()

    def _extract_factor_scores(
        self,
        row: ScoringResult,
        factor_overrides: Optional[dict[str, Optional[float]]] = None,
    ) -> dict[str, Optional[float]]:
        """Build a factor_scores dict from a ScoringResult ORM row.

        Composite factors (growth, value_graham_dcf) are computed by averaging
        whatever sub-columns have non-None values. If all sub-columns are None
        the composite is also None, which triggers weight redistribution in
        compose().

        Args:
            row: ScoringResult ORM object.

        Returns:
            Dict mapping weight-key names to float scores (or None).
        """
        # Map column name -> current value from the ORM row.
        col_values: dict[str, Optional[float]] = {
            "buffett_score": row.buffett_score,
            "graham_score": row.graham_score,
            "piotroski_fscore": row.piotroski_fscore,
            "magic_formula_rank": row.magic_formula_rank,
            "lynch_peg_score": row.lynch_peg_score,
            "dcf_margin_of_safety_pct": row.dcf_margin_of_safety_pct,
            "momentum_score": row.momentum_score,
            "technical_score": row.technical_score,
            "dividend_score": row.dividend_score,
        }
        if factor_overrides:
            col_values.update(factor_overrides)

        factor_scores: dict[str, Optional[float]] = {}
        for factor_key, db_cols in _DB_COL_TO_FACTOR.items():
            if not db_cols:
                # Not yet implemented — leave as None.
                factor_scores[factor_key] = None
                continue
            parts = [col_values[c] for c in db_cols if col_values.get(c) is not None]
            factor_scores[factor_key] = sum(parts) / len(parts) if parts else None

        return factor_scores

    def _build_dcf_factor_overrides(
        self,
        session: Session,
        rows: list[ScoringResult],
    ) -> dict[int, dict[str, Optional[float]]]:
        """Normalize raw DCF margins by sector without overwriting the raw DB values."""
        import pandas as pd

        dcf_rows = [row for row in rows if row.dcf_margin_of_safety_pct is not None]
        if not dcf_rows:
            return {}

        company_ids = {row.company_id for row in dcf_rows}
        sector_rows = (
            session.query(Company.id, Company.sector_custom, Company.sector_bist)
            .filter(Company.id.in_(company_ids))
            .all()
        )
        sector_lookup = {
            cid: (sector_custom or sector_bist or "UNKNOWN")
            for cid, sector_custom, sector_bist in sector_rows
        }

        df = pd.DataFrame(
            [
                {
                    "id": row.id,
                    "sector": sector_lookup.get(row.company_id, "UNKNOWN"),
                    "dcf_margin_of_safety_pct": row.dcf_margin_of_safety_pct,
                }
                for row in dcf_rows
            ]
        ).set_index("id")

        normalizer = ScoreNormalizer()
        normalized = normalizer.normalize_factor(df, "dcf_margin_of_safety_pct", "sector")

        overrides: dict[int, dict[str, Optional[float]]] = {}
        for row_id, value in normalized.items():
            overrides[int(row_id)] = {
                "dcf_margin_of_safety_pct": None if pd.isna(value) else float(value)
            }
        return overrides

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(
        self,
        company_id: int,
        factor_scores: dict,
        portfolio: str,
        model_type: str,
    ) -> Optional[float]:
        """Compute a weighted composite score (0-100) for one company.

        None factors have their weight redistributed proportionally among the
        remaining factors that do have a score. If all factors are None,
        returns None.

        Args:
            company_id: Database ID of the company (used for log messages).
            factor_scores: Dict mapping weight-key names to normalized scores
                (0-100 each) or None. Keys must match those in scoring_weights.yaml
                for the resolved section (portfolio + model_type).
            portfolio: 'alpha', 'beta', or 'delta'.
            model_type: 'OPERATING', 'BANK', 'HOLDING', or 'IPO'.

        Returns:
            Weighted composite score in [0, 100], or None if all factors are
            missing.
        """
        section = self._get_weight_section(portfolio, model_type)
        section_weights = self.weights.get(section)
        if section_weights is None:
            logger.warning(
                "company_id=%d: no weight section '%s' in scoring_weights.yaml",
                company_id,
                section,
            )
            return None

        # Separate available (non-None) from missing factors.
        available: dict[str, tuple[float, float]] = {}  # factor -> (score, weight)
        for factor, weight in section_weights.items():
            score = factor_scores.get(factor)
            if score is not None and not (isinstance(score, float) and __import__("math").isnan(score)):
                available[factor] = (float(score), float(weight))

        if not available:
            logger.debug(
                "company_id=%d portfolio=%s model=%s: all factors None",
                company_id, portfolio, model_type,
            )
            return None

        # Redistribute weights proportionally among available factors.
        total_avail_weight = sum(w for _, w in available.values())
        if total_avail_weight <= 0:
            return None
            
        composite = sum(
            score * (weight / total_avail_weight)
            for score, weight in available.values()
        )

        # Data-coverage penalty. Previously softened to 0.90 + 0.10 × coverage
        # which capped the penalty at 10% even for stocks with just one factor
        # available — a single-factor "score" looked almost identical to a
        # fully-populated score, making shoddy-data names drift into the top
        # of the ranked list. Hardening to 0.50 + 0.50 × coverage:
        #   100% coverage -> x1.00  (no penalty)
        #    50% coverage -> x0.75
        #    20% coverage -> x0.60
        # The 0.50 floor keeps partial-coverage names visible as candidates
        # but removes the false parity with fully-covered peers.
        total_possible_weight = sum(w for w in section_weights.values() if w > 0)
        coverage_ratio = total_avail_weight / total_possible_weight if total_possible_weight > 0 else 0.0
        data_penalty = 0.50 + 0.50 * coverage_ratio
        composite = composite * data_penalty

        logger.debug(
            "company_id=%d portfolio=%s model=%s: "
            "%d/%d factors available, coverage=%.0f%%, composite=%.2f",
            company_id, portfolio, model_type,
            len(available), len(section_weights),
            coverage_ratio * 100, composite,
        )
        return composite

    def _harmonize_composites(self, rows: list[ScoringResult]) -> None:
        """Re-rank all composite scores across the full universe.

        After operating, banking, and holding composites are computed
        independently, this step converts them to universe-wide percentile
        ranks (0-100). This ensures a bank/holding must earn its score
        relative to ALL companies, not just its small peer group.

        Before percentile-ranking, applies cross-model calibration penalties
        to prevent small peer groups (e.g. 14 banks) from producing
        artificially inflated percentile scores:

        1. Peer group size penalty: smaller peer groups = less reliable ranking.
           Formula: peer_factor = 0.60 + 0.40 * min(1.0, n_peers / 50)
           14 banks → 0.712, 40 holdings → 0.92, 468 operating → 1.0

        2. Minimum data threshold: if data_completeness < 40%, set score to None.
           This prevents low-data FINANCIAL companies (only momentum+technical)
           from polluting the rankings.
        """
        import pandas as pd

        # Count peer group sizes by model type
        model_counts: dict[str, int] = {}
        for row in rows:
            mt = row.model_used or "OPERATING"
            model_counts[mt] = model_counts.get(mt, 0) + 1

        for attr in ("composite_alpha", "composite_beta", "composite_delta"):
            values = pd.Series(
                {i: getattr(rows[i], attr) for i in range(len(rows))},
                dtype=float,
            )

            # Apply cross-model calibration before percentile ranking
            for i, row in enumerate(rows):
                if pd.isna(values.iloc[i]):
                    continue

                mt = row.model_used or "OPERATING"
                dc = row.data_completeness or 0.0

                # Companies with a precomputed model composite (banking/holding)
                # have data_completeness from their own model scorer. Their
                # composite already encodes weight redistribution for missing
                # sub-factors, so only exclude if they have NO model composite.
                has_model_composite = (
                    (mt in ("BANK", "FINANCIAL", "INSURANCE") and row.banking_composite is not None)
                    or (mt == "HOLDING" and row.holding_composite is not None)
                    or (mt == "REIT" and row.reit_composite is not None)
                )

                # Minimum data threshold: exclude if < 40% data completeness
                # AND has no precomputed model composite to rely on.
                if dc < 40.0 and not has_model_composite:
                    values.iloc[i] = float("nan")
                    setattr(row, attr, None)
                    continue

                # Peer group size penalty for non-OPERATING models
                if mt not in ("OPERATING", "SPORT"):
                    n_peers = model_counts.get(mt, 1)
                    peer_factor = 0.60 + 0.40 * min(1.0, n_peers / 50.0)
                    values.iloc[i] = values.iloc[i] * peer_factor

            # Skip if all None
            if values.dropna().empty:
                continue

            # Percentile rank across full universe
            ranks = values.rank(method="average", na_option="keep")
            n_valid = int(values.notna().sum())

            if n_valid <= 1:
                continue

            percentiles = (ranks - 1) / (n_valid - 1) * 100.0

            for i in range(len(rows)):
                if pd.notna(percentiles.iloc[i]):
                    setattr(rows[i], attr, round(float(percentiles.iloc[i]), 2))

        logger.info(
            "Harmonized composite scores across %d companies "
            "(universe-wide percentile with cross-model calibration, "
            "peer counts: %s)",
            len(rows),
            {k: v for k, v in sorted(model_counts.items())},
        )

    def compose_all(self, session: Session, scoring_date: Optional[date] = None, use_regime: bool = False) -> None:
        """Compute and persist composite scores for all companies in scoring_results.

        For each ScoringResult row that belongs to the given scoring_date
        (defaults to today), this method:
          1. Extracts factor scores from the ORM columns.
          2. Derives composite_alpha, composite_beta, composite_delta.
          3. Updates data_completeness (fraction of operating factors available).
          4. Commits the updated rows.

        Args:
            session: SQLAlchemy session bound to the bist_picker.db database.
            scoring_date: The scoring date to process. Defaults to today.
            use_regime: If True, uses MarketRegimeClassifier to pick weights
                dynamically from regime_weights section.
        """
        if scoring_date is None:
            scoring_date = date.today()

        rows = (
            session.query(ScoringResult)
            .filter(ScoringResult.scoring_date == scoring_date)
            .all()
        )

        if not rows:
            logger.warning("compose_all: no scoring_results for %s", scoring_date)
            return

        dcf_overrides = self._build_dcf_factor_overrides(session, rows)

        # Operating model factors used for data_completeness calculation.
        operating_factors = [
            f for f, w in self.weights.get("alpha", {}).items() if w > 0
        ]

        from bist_picker.portfolio.macro_overlay import MacroRegimeClassifier as LegacyMacroClassifier
        from bist_picker.portfolio.regime_classifier import MarketRegimeClassifier

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            working_weights = copy.deepcopy(self.weights)

            if use_regime:
                classifier = MarketRegimeClassifier(session)
                regime = classifier.classify(scoring_date)
                logger.info(f"Market Regime Detection (Dynamic): {regime}")
                print(f"\n[REGIME] Market Regime Detection (Dynamic): {regime}")
                
                regime_weights = self.weights.get("regime_weights", {}).get(regime)
                if regime_weights:
                    logger.info(f"Using weights for regime {regime}")
                    print(f"[REGIME] Using weights for regime {regime}")
                    # Apply to standard portfolios
                    working_weights["alpha"] = regime_weights
                    working_weights["beta"] = regime_weights
                    working_weights["delta"] = regime_weights
                else:
                    logger.warning(f"No weights defined for regime {regime}, falling back to defaults.")
                    print(f"[REGIME] No weights defined for regime {regime}, falling back to defaults.")
            else:
                # Legacy Macro Overlay logic
                legacy_classifier = LegacyMacroClassifier(session)
                regime = legacy_classifier.classify(scoring_date)
                multipliers = legacy_classifier.get_weight_multipliers(regime)

                if multipliers:
                    logger.info(f"Macro Regime (Legacy): {regime}. Adjusting weights...")
                    print(f"\n[REGIME] Macro Regime (Legacy): {regime}. Adjusting weights...")
                    for section, weights in working_weights.items():
                        if not isinstance(weights, dict) or section == "regime_weights":
                            continue
                        
                        new_weights = {}
                        total_new_weight = 0.0
                        for factor, weight in weights.items():
                            mult = multipliers.get(factor, 1.0)
                            w = weight * mult
                            new_weights[factor] = w
                            total_new_weight += w
                        
                        if total_new_weight > 0:
                            working_weights[section] = {
                                k: v / total_new_weight for k, v in new_weights.items()
                            }
                else:
                    logger.info(f"Macro Regime (Legacy): {regime}. No weight adjustments.")

            # Use the working copy for all scoring
            original_weights = self.weights
            self.weights = working_weights

            task = progress.add_task(
                f"Composing scores for {scoring_date}...", total=len(rows)
            )

            updated = 0
            for row in rows:
                model_type = row.model_used or "OPERATING"

                # ── BANK / FINANCIAL / HOLDING / REIT: hybrid specialized scoring ──
                if model_type in ("BANK", "FINANCIAL", "INSURANCE", "HOLDING", "REIT"):
                    col = "banking_composite" if model_type in ("BANK", "FINANCIAL", "INSURANCE") \
                          else "holding_composite" if model_type == "HOLDING" \
                          else "reit_composite"
                    
                    model_score = getattr(row, col, None)
                    if model_score is not None:
                        # Hybrid Bridge (as discussed): 
                        # 70% Specialized Model Score + 30% Universal Factors (Momentum, Technical)
                        # This brings "Equivalent Purchasing Power" to different sectors.
                        
                        universal_weights = {"momentum_score": 0.50, "technical_score": 0.50}
                        universal_total = 0.0
                        universal_avail_w = 0.0
                        for f, w in universal_weights.items():
                            val = getattr(row, f, None)
                            if val is not None:
                                universal_total += val * w
                                universal_avail_w += w
                        
                        universal_score = (universal_total / universal_avail_w) if universal_avail_w > 0 else model_score
                        
                        # Blend: 70% model, 30% universal
                        blended = (model_score * 0.70) + (universal_score * 0.30)
                        
                        if row.data_completeness is None or row.data_completeness < 10.0:
                            row.data_completeness = 100.0

                        # Apply data_penalty
                        coverage_ratio = (row.data_completeness or 0.0) / 100.0
                        data_penalty = 0.50 + 0.50 * coverage_ratio
                        penalized = round(blended * data_penalty, 2)

                        row.composite_alpha = penalized
                        row.composite_beta = penalized
                        row.composite_delta = penalized
                    else:
                        # Fallback: score with operating factors if model scorer didn't run
                        factor_scores = self._extract_factor_scores(
                            row,
                            factor_overrides=dcf_overrides.get(row.id),
                        )
                        row.composite_alpha = self.compose(
                            row.company_id, factor_scores, "alpha", "OPERATING"
                        )
                        row.composite_beta = self.compose(
                            row.company_id, factor_scores, "beta", "OPERATING"
                        )
                        row.composite_delta = self.compose(
                            row.company_id, factor_scores, "delta", "OPERATING"
                        )
                        operating_factors_local = [
                            f for f, w in self.weights.get("alpha", {}).items() if w > 0
                        ]
                        n_avail = sum(
                            1 for f in operating_factors_local if factor_scores.get(f) is not None
                        )
                        row.data_completeness = (
                            (n_avail / len(operating_factors_local)) * 100.0
                            if operating_factors_local else None
                        )
                    updated += 1
                    progress.advance(task)
                    continue

                # SPORTS clubs are tracked in the DB but are not investable for
                # this project's portfolio logic. Showing them with a full ALPHA
                # composite is misleading because they only carry a small subset
                # of the classic factors (mostly momentum/technical/growth).
                if model_type == "SPORT":
                    row.composite_alpha = None
                    row.composite_beta = None
                    row.composite_delta = None
                    row.data_completeness = None
                    updated += 1
                    progress.advance(task)
                    continue

                # ── OPERATING / INSURANCE: standard factor pipeline ──
                factor_scores = self._extract_factor_scores(
                    row,
                    factor_overrides=dcf_overrides.get(row.id),
                )

                row.composite_alpha = self.compose(
                    row.company_id, factor_scores, "alpha", model_type
                )
                row.composite_beta = self.compose(
                    row.company_id, factor_scores, "beta", model_type
                )
                row.composite_delta = self.compose(
                    row.company_id, factor_scores, "delta", model_type
                )

                # Data completeness: percentage (0-100) of operating factors with a value.
                n_available = sum(
                    1 for f in operating_factors if factor_scores.get(f) is not None
                )
                row.data_completeness = (
                    (n_available / len(operating_factors)) * 100.0 if operating_factors else None
                )

                updated += 1
                progress.advance(task)

        # Restore original weights so future calls start from pristine config
        self.weights = original_weights

        # ── Harmonize: re-rank composites across full universe ──
        self._harmonize_composites(rows)

        session.commit()
        logger.info(
            "compose_all: updated %d composite scores for %s", updated, scoring_date
        )
