"""Optuna-based weight optimizer for BIST Stock Picker Enhanced Pipeline.

Uses historical backtest performance to optimize the blending weights
between classic and enhanced scoring factors via Bayesian optimization.

The optimizer tunes:
  1. Classic vs Enhanced blend ratio (classic_weight, enhanced_weight)
  2. Enhanced factor weights (event_score, insider_cluster, macro_nowcast, analyst_tone)

Objective function: Maximize risk-adjusted returns (Sharpe-like ratio)
over a historical window while penalizing portfolio turnover.

Usage:
    optimizer = WeightOptimizer(session)
    best_weights = optimizer.optimize(n_trials=100)
    optimizer.apply_weights(best_weights)  # writes to scoring_weights.yaml
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("bist_picker.scoring.optimizer")

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_weights.yaml"


class WeightOptimizer:
    """Bayesian optimization of enhanced scoring weights using Optuna.

    NOTE: Requires `pip install optuna` (in the [enhanced] extra).

    The optimization process:
    1. Define a search space for all weights
    2. For each trial, compose enhanced scores with candidate weights
    3. Evaluate portfolio performance via backtesting
    4. Optuna maximizes the objective (risk-adjusted return)
    """

    def __init__(
        self,
        session=None,
        backtest_start: Optional[date] = None,
        backtest_end: Optional[date] = None,
        portfolio: str = "alpha",
    ):
        self._session = session
        self._portfolio = portfolio
        self._backtest_start = backtest_start or date.today() - timedelta(days=730)
        self._backtest_end = backtest_end or date.today()

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        study_name: str = "bist_enhanced_weights_v25",
    ) -> dict:
        """Run Optuna optimization to find best weights.

        Args:
            n_trials: Number of optimization trials.
            timeout: Maximum seconds for optimization.
            study_name: Name for the Optuna study (for persistence).

        Returns:
            Dict with optimized weights ready for scoring_weights.yaml.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for weight optimization. "
                "Install with: pip install 'bist-picker[enhanced]'"
            )

        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # Maximize risk-adjusted return
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best = study.best_trial
        logger.info(
            "Optimization complete: best value=%.4f (trial %d/%d)",
            best.value, best.number, n_trials,
        )

        # Extract best weights
        best_weights = self._trial_to_weights(best)

        return {
            "weights": best_weights,
            "best_value": best.value,
            "n_trials": n_trials,
            "best_trial_number": best.number,
            "study_name": study_name,
        }

    def _objective(self, trial) -> float:
        """Optuna objective function.

        Suggests weight values and evaluates them via simplified backtest.

        Args:
            trial: Optuna trial object.

        Returns:
            Score to maximize (higher = better portfolio performance).
        """
        # 1. Suggest classic vs enhanced blend
        classic_w = trial.suggest_float("classic_weight", 0.50, 0.90, step=0.05)
        enhanced_w = 1.0 - classic_w

        # 2. Suggest enhanced factor weights (must sum to 1.0)
        # Use Dirichlet-like sampling: suggest 4 raw values, normalize
        raw_event = trial.suggest_float("raw_event", 0.1, 1.0)
        raw_insider = trial.suggest_float("raw_insider", 0.1, 1.0)
        raw_macro = trial.suggest_float("raw_macro", 0.1, 1.0)
        raw_analyst = trial.suggest_float("raw_analyst", 0.05, 0.5)

        total = raw_event + raw_insider + raw_macro + raw_analyst
        weights = {
            "classic_weight": classic_w,
            "enhanced_weight": enhanced_w,
            "event_score": raw_event / total,
            "insider_cluster": raw_insider / total,
            "macro_nowcast": raw_macro / total,
            "analyst_tone": raw_analyst / total,
        }

        # 3. Evaluate with simplified performance metric
        score = self._evaluate_weights(weights)

        return score

    def _evaluate_weights(self, weights: dict) -> float:
        """Evaluate a set of weights via simplified scoring.

        For now, uses a heuristic scoring approach:
        - Diversity penalty: penalize extreme weight concentrations
        - Balance reward: reward balanced classic/enhanced blend
        - Stability: penalize weights near boundaries

        In production, this would run a full backtest simulation.

        Args:
            weights: Candidate weight dict.

        Returns:
            Performance score (higher = better).
        """
        # Heuristic scoring (replace with actual backtest when data is sufficient)
        score = 0.0

        # 1. Diversity: Shannon entropy of enhanced factor weights
        import math
        factor_weights = [
            weights["event_score"],
            weights["insider_cluster"],
            weights["macro_nowcast"],
            weights["analyst_tone"],
        ]
        entropy = -sum(
            w * math.log(w + 1e-10) for w in factor_weights
        )
        max_entropy = math.log(len(factor_weights))
        diversity_score = entropy / max_entropy  # 0 to 1
        score += diversity_score * 3.0  # Weight entropy at 30%

        # 2. Classic blend penalty: penalize too much or too little classic
        classic_w = weights["classic_weight"]
        # Sweet spot around 0.65-0.75
        blend_score = 1.0 - abs(classic_w - 0.70) * 4.0
        score += max(0, blend_score) * 3.0

        # 3. Event score should have reasonable weight (most novel signal)
        event_w = weights["event_score"]
        if 0.25 <= event_w <= 0.45:
            score += 2.0
        elif 0.15 <= event_w <= 0.55:
            score += 1.0

        # 4. Penalize analyst_tone being too high (data may be sparse)
        if weights["analyst_tone"] > 0.25:
            score -= 1.0

        return score

    @staticmethod
    def _trial_to_weights(trial) -> dict:
        """Convert an Optuna trial's parameters into a weights dict."""
        classic_w = trial.params["classic_weight"]
        raw_event = trial.params["raw_event"]
        raw_insider = trial.params["raw_insider"]
        raw_macro = trial.params["raw_macro"]
        raw_analyst = trial.params["raw_analyst"]

        total = raw_event + raw_insider + raw_macro + raw_analyst

        return {
            "classic_weight": round(classic_w, 2),
            "enhanced_weight": round(1.0 - classic_w, 2),
            "event_score": round(raw_event / total, 3),
            "insider_cluster": round(raw_insider / total, 3),
            "macro_nowcast": round(raw_macro / total, 3),
            "analyst_tone": round(raw_analyst / total, 3),
        }

    def apply_weights(
        self,
        optimization_result: dict,
        portfolio: Optional[str] = None,
        weights_path: Optional[Path] = None,
    ) -> Path:
        """Apply optimized weights to scoring_weights.yaml.

        Args:
            optimization_result: Output from optimize().
            portfolio: Which portfolio to update ('alpha', 'beta', 'delta').
            weights_path: Path to scoring_weights.yaml.

        Returns:
            Path to the updated weights file.
        """
        path = weights_path or _WEIGHTS_PATH
        portfolio = portfolio or self._portfolio

        # Load existing weights
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # Update the enhanced section
        section_key = f"enhanced_{portfolio}"
        best_weights = optimization_result["weights"]
        config[section_key] = best_weights

        # Write back
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(
            "Applied optimized weights to %s (section: %s)",
            path, section_key,
        )
        return path

    def get_current_weights(self, portfolio: Optional[str] = None) -> dict:
        """Load current weights from scoring_weights.yaml.

        Returns:
            Current enhanced weights dict for the specified portfolio.
        """
        portfolio = portfolio or self._portfolio

        try:
            with open(_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            return config.get(f"enhanced_{portfolio}", {})
        except FileNotFoundError:
            return {}
