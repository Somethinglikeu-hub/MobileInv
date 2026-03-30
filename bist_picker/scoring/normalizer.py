"""Score normalization pipeline for the BIST Stock Picker.

Implements the three-stage normalization pipeline that makes raw factor scores
comparable across companies and sectors:

  1. Winsorize at +/-n standard deviations (removes extreme outliers)
  2. Sector z-score (normalizes within custom sub-sector peer group;
     falls back to universe stats when sector has fewer than 3 members)
  3. Convert to 0-100 percentile rank

All methods handle NaN values gracefully and are vectorized with pandas/numpy.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScoreNormalizer:
    """Normalizes raw factor scores via winsorize -> sector z-score -> percentile.

    Each stage can be used independently or chained via normalize_factor().
    The class holds no state and is safe to reuse across calls.
    """

    def winsorize(self, values: pd.Series, n_std: float = 3.0) -> pd.Series:
        """Cap values at +/-n standard deviations from the mean.

        Extreme outliers (e.g., P/E of 5000) can dominate z-scores and
        percentile ranks. Winsorizing clamps them before further processing.
        NaN values are preserved unchanged.

        Args:
            values: Series of raw factor scores (may contain NaN).
            n_std: Number of standard deviations to use as the cap. Default 3.0.

        Returns:
            Series with the same index; values outside mean +/- n_std*std
            are clamped to the boundary. NaN values remain NaN.
        """
        valid = values.dropna()
        if valid.empty:
            return values.copy()

        mean = valid.mean()
        std = valid.std()

        if std == 0 or np.isnan(std):
            # All values identical — nothing to winsorize.
            return values.copy()

        lower = mean - n_std * std
        upper = mean + n_std * std

        return values.clip(lower=lower, upper=upper)

    def sector_zscore(
        self,
        values: pd.Series,
        sector_labels: pd.Series,
    ) -> pd.Series:
        """Compute z-score within each sector peer group.

        For sectors with fewer than 3 members with non-NaN values, falls back
        to full-universe mean/std to avoid unreliable sector statistics.
        NaN values in the input are preserved as NaN in the output.

        Args:
            values: Series of (optionally winsorized) factor scores.
            sector_labels: Series with the same index; each entry is the
                sector or sub-sector label for that company.

        Returns:
            Series of z-scores with the same index. Companies whose sector
            has <3 valid members are z-scored against the full universe.
        """
        result = pd.Series(np.nan, index=values.index, dtype=float)

        # Universe-wide fallback stats (computed once).
        valid_all = values.dropna()
        if valid_all.empty:
            return result

        universe_mean = float(valid_all.mean())
        universe_std = float(valid_all.std())
        if universe_std == 0 or np.isnan(universe_std):
            universe_std = 1.0

        # Iterate over each unique sector group.
        for sector in sector_labels.unique():
            group_idx = sector_labels[sector_labels == sector].index
            group_valid = values.loc[group_idx].dropna()
            n_valid = len(group_valid)

            if n_valid < 3:
                # Too few peers — use universe stats.
                s_mean = universe_mean
                s_std = universe_std
            else:
                s_mean = float(group_valid.mean())
                s_std = float(group_valid.std())
                if s_std == 0 or np.isnan(s_std):
                    # All sector peers identical — fall back to universe std.
                    s_std = universe_std

            # Assign z-scores only for members with non-NaN values.
            valid_mask = values.loc[group_idx].notna()
            valid_idx = group_idx[valid_mask]
            result.loc[valid_idx] = (values.loc[valid_idx] - s_mean) / s_std

        return result

    def to_percentile(self, values: pd.Series) -> pd.Series:
        """Convert values to 0-100 percentile rank.

        Uses the average-rank method for ties. The minimum value maps to 0
        and the maximum to 100 via linear interpolation:
            percentile = (rank - 1) / (n_valid - 1) * 100

        NaN values receive NaN and are excluded from the rank count. A single
        non-NaN value maps to 100.

        Args:
            values: Series of z-scores or any ordinal values.

        Returns:
            Series with values in [0, 100]. NaN inputs remain NaN.
        """
        ranks = values.rank(method="average", na_option="keep")
        n_valid = int(values.notna().sum())

        if n_valid == 0:
            return pd.Series(np.nan, index=values.index, dtype=float)

        if n_valid == 1:
            # Only one valid observation — map it to 100.
            result = pd.Series(np.nan, index=values.index, dtype=float)
            result[values.notna()] = 100.0
            return result

        return (ranks - 1) / (n_valid - 1) * 100.0

    def normalize_factor(
        self,
        raw_scores: pd.DataFrame,
        factor_col: str,
        sector_col: str,
    ) -> pd.Series:
        """Apply the full three-stage normalization pipeline to one factor column.

        Pipeline: winsorize -> sector z-score -> percentile (0-100).

        Args:
            raw_scores: DataFrame containing at least ``factor_col`` and
                ``sector_col`` columns, one row per company.
            factor_col: Name of the column with raw factor scores.
            sector_col: Name of the column with sector/sub-sector labels.

        Returns:
            Series of normalized scores in [0, 100] with the same index as
            ``raw_scores``. Companies with NaN in ``factor_col`` receive NaN.
        """
        raw = raw_scores[factor_col].copy()
        sectors = raw_scores[sector_col]

        winsorized = self.winsorize(raw)
        zscored = self.sector_zscore(winsorized, sectors)
        percentiled = self.to_percentile(zscored)

        logger.debug(
            "normalize_factor(%s): %d values, %d non-NaN -> "
            "min=%.1f max=%.1f mean=%.1f",
            factor_col,
            len(raw),
            raw.notna().sum(),
            percentiled.min() if percentiled.notna().any() else float("nan"),
            percentiled.max() if percentiled.notna().any() else float("nan"),
            percentiled.mean() if percentiled.notna().any() else float("nan"),
        )

        return percentiled
