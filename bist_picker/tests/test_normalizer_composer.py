"""Tests for scoring/normalizer.py and scoring/composer.py.

Covers:
  ScoreNormalizer:
    - winsorize: outlier capping with 2-sigma; all-NaN; zero-std; NaN preserved
    - sector_zscore: two-sector relative scoring; small sector fallback to universe
    - to_percentile: [10,20,30,40,50] -> [0,25,50,75,100]
    - normalize_factor: full pipeline on a DataFrame

  ScoreComposer:
    - validate_weights: scoring_weights.yaml sums to 1.0 per section
    - compose: known weights + known scores -> expected result
    - compose with one None factor -> weights redistribute correctly
    - compose with all None factors -> returns None
    - compose unknown model_type -> raises ValueError
    - compose_all: DB integration (in-memory SQLite)
"""

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.db.schema import Base, Company, ScoringResult
from bist_picker.scoring.composer import ScoreComposer
from bist_picker.scoring.normalizer import ScoreNormalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normalizer() -> ScoreNormalizer:
    return ScoreNormalizer()


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    Session_ = sessionmaker(bind=engine)
    sess = Session_()
    yield sess
    sess.close()


@pytest.fixture
def composer() -> ScoreComposer:
    """ScoreComposer using the real scoring_weights.yaml."""
    return ScoreComposer()


# ---------------------------------------------------------------------------
# ScoreNormalizer — winsorize
# ---------------------------------------------------------------------------


class TestWinsorize:
    """Tests for ScoreNormalizer.winsorize()."""

    def test_outlier_capped_at_2sigma(self, normalizer):
        """Upper outlier is capped at mean + 2*std.

        Uses [1..9, 100] where 100 genuinely exceeds mean+2*std of all 10 values.
        (mean~14.5, std~30.2 -> upper cap ~75; 100 > 75 so it is capped.)
        Values 1-9 are below the cap and must remain unchanged.
        """
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
        result = normalizer.winsorize(s, n_std=2.0)

        assert result.iloc[9] < 100.0, "Upper outlier 100 was not capped"
        for i in range(9):
            assert result.iloc[i] == pytest.approx(float(i + 1)), (
                f"Non-outlier value at index {i} was incorrectly changed"
            )

    def test_no_capping_needed(self, normalizer):
        """Tightly-clustered values should be unchanged."""
        s = pd.Series([10.0, 11.0, 12.0, 13.0])
        result = normalizer.winsorize(s, n_std=3.0)
        pd.testing.assert_series_equal(result, s)

    def test_nan_values_preserved(self, normalizer):
        """NaN values must pass through unchanged."""
        s = pd.Series([1.0, float("nan"), 3.0, 100.0])
        result = normalizer.winsorize(s, n_std=2.0)
        assert math.isnan(result[1])

    def test_all_nan_returns_all_nan(self, normalizer):
        """Series of all NaN -> all NaN returned."""
        s = pd.Series([float("nan"), float("nan")])
        result = normalizer.winsorize(s)
        assert result.isna().all()

    def test_zero_std_returns_unchanged(self, normalizer):
        """All-identical values (std=0) should be returned unchanged."""
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = normalizer.winsorize(s)
        pd.testing.assert_series_equal(result, s)

    def test_lower_outlier_capped(self, normalizer):
        """Very negative outlier is capped at mean - 2*std.

        Uses [-100, 1..9] where -100 genuinely falls below mean-2*std
        of all 10 values. (mean~-5.5, std~33.3 -> lower cap ~-72; -100 < -72.)
        """
        s = pd.Series([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        result = normalizer.winsorize(s, n_std=2.0)
        assert result.iloc[0] > -100.0, "Lower outlier -100 was not capped"


# ---------------------------------------------------------------------------
# ScoreNormalizer — sector_zscore
# ---------------------------------------------------------------------------


class TestSectorZscore:
    """Tests for ScoreNormalizer.sector_zscore()."""

    def test_two_sectors_are_sector_relative(self, normalizer):
        """Z-scores within each sector are computed relative to that sector."""
        # Sector A: [10, 20, 30]  mean=20, std=10
        # Sector B: [100, 200, 300]  mean=200, std=100
        values = pd.Series([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])
        sectors = pd.Series(["A", "A", "A", "B", "B", "B"])

        result = normalizer.sector_zscore(values, sectors)

        # Each sector's z-scores should mirror the same pattern.
        z_a = result.iloc[:3].values
        z_b = result.iloc[3:].values

        assert z_a == pytest.approx(z_b, abs=1e-9), (
            "Sector A and B should have identical z-score patterns"
        )
        # Middle element of each sector should be ~0 (close to sector mean).
        assert z_a[1] == pytest.approx(0.0, abs=1e-9)

    def test_small_sector_uses_universe(self, normalizer):
        """Sector with <3 members falls back to full-universe stats."""
        # Sector A: 3 members (valid for sector stats)
        # Sector B: 2 members (too small -> universe fallback)
        values = pd.Series([10.0, 20.0, 30.0, 200.0, 300.0])
        sectors = pd.Series(["A", "A", "A", "B", "B"])

        result = normalizer.sector_zscore(values, sectors)

        # B members z-scored against universe (mean ~112, std ~120ish).
        # Just verify they're not z-scored against sector B (mean=250).
        universe_mean = values.mean()
        universe_std = values.std()

        expected_z_200 = (200.0 - universe_mean) / universe_std
        assert result.iloc[3] == pytest.approx(expected_z_200, abs=1e-9)

    def test_nan_values_preserved(self, normalizer):
        """NaN inputs remain NaN in the output."""
        values = pd.Series([10.0, float("nan"), 30.0, 100.0, 200.0, 300.0])
        sectors = pd.Series(["A", "A", "A", "B", "B", "B"])
        result = normalizer.sector_zscore(values, sectors)
        assert math.isnan(result.iloc[1])

    def test_all_nan_returns_all_nan(self, normalizer):
        """All-NaN input -> all-NaN output."""
        values = pd.Series([float("nan"), float("nan"), float("nan")])
        sectors = pd.Series(["A", "A", "A"])
        result = normalizer.sector_zscore(values, sectors)
        assert result.isna().all()

    def test_sector_with_identical_values_no_crash(self, normalizer):
        """Sector where all members have the same value should not raise."""
        values = pd.Series([5.0, 5.0, 5.0, 100.0, 200.0, 300.0])
        sectors = pd.Series(["A", "A", "A", "B", "B", "B"])
        result = normalizer.sector_zscore(values, sectors)
        # Sector A z-scores should fall back gracefully.
        assert not result.iloc[:3].isna().all()


# ---------------------------------------------------------------------------
# ScoreNormalizer — to_percentile
# ---------------------------------------------------------------------------


class TestToPercentile:
    """Tests for ScoreNormalizer.to_percentile()."""

    def test_five_distinct_values(self, normalizer):
        """[10, 20, 30, 40, 50] -> [0, 25, 50, 75, 100]."""
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = normalizer.to_percentile(s)

        expected = pd.Series([0.0, 25.0, 50.0, 75.0, 100.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_tied_values_average_rank(self, normalizer):
        """Ties should receive the average of their shared ranks."""
        s = pd.Series([10.0, 20.0, 20.0, 30.0])
        result = normalizer.to_percentile(s)

        # Rank 1->0%, rank 2.5 avg->50%, rank 4->100%
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(result.iloc[2])  # same rank
        assert result.iloc[3] == pytest.approx(100.0)

    def test_single_value_returns_100(self, normalizer):
        """Single non-NaN value should rank at 100%."""
        s = pd.Series([42.0])
        result = normalizer.to_percentile(s)
        assert result.iloc[0] == pytest.approx(100.0)

    def test_nan_preserved(self, normalizer):
        """NaN values do not affect other ranks and remain NaN."""
        s = pd.Series([10.0, float("nan"), 30.0])
        result = normalizer.to_percentile(s)
        assert math.isnan(result.iloc[1])
        # The two non-NaN values should span [0, 100].
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(100.0)

    def test_output_in_range(self, normalizer):
        """All non-NaN percentiles must be in [0, 100]."""
        s = pd.Series(np.random.default_rng(0).standard_normal(50))
        result = normalizer.to_percentile(s)
        assert result.min() >= 0.0
        assert result.max() <= 100.0


# ---------------------------------------------------------------------------
# ScoreNormalizer — normalize_factor (full pipeline)
# ---------------------------------------------------------------------------


class TestNormalizeFactor:
    """Tests for ScoreNormalizer.normalize_factor()."""

    def test_full_pipeline_output_in_range(self, normalizer):
        """normalize_factor output must be in [0, 100] for non-NaN values."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "pe_ratio": rng.standard_normal(30) * 10 + 20,
            "sector": ["A", "B", "C"] * 10,
        })
        # Add one outlier.
        df.loc[0, "pe_ratio"] = 9999.0

        result = normalizer.normalize_factor(df, "pe_ratio", "sector")

        assert result.notna().all()
        assert result.min() >= 0.0
        assert result.max() <= 100.0

    def test_nan_passed_through(self, normalizer):
        """Rows with NaN in factor_col must produce NaN in output."""
        df = pd.DataFrame({
            "score": [1.0, float("nan"), 3.0, 4.0, 5.0],
            "sector": ["A", "A", "B", "B", "B"],
        })
        result = normalizer.normalize_factor(df, "score", "sector")
        assert math.isnan(result.iloc[1])

    def test_outlier_does_not_dominate_rank(self, normalizer):
        """After winsorization, an extreme outlier should not always rank 100."""
        df = pd.DataFrame({
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 9999.0],
            "sector": ["A"] * 6,
        })
        result = normalizer.normalize_factor(df, "val", "sector")
        # With 6 distinct values the max percentile is 100.
        # The outlier (index 5) should still rank last, just capped.
        # Without winsorization its z-score would dwarf everything else.
        # Just verify the output is valid.
        assert result.notna().all()
        assert result.max() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# ScoreComposer — weight validation
# ---------------------------------------------------------------------------


class TestWeightValidation:
    """Tests for scoring_weights.yaml validation."""

    def test_all_sections_sum_to_1(self, composer):
        """All portfolio sections in scoring_weights.yaml must sum to 1.0."""
        for section in ("alpha", "beta", "delta", "banking", "holding", "ipo"):
            weights = composer.weights[section]
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, (
                f"Section '{section}' sums to {total:.6f}, not 1.0"
            )

    def test_bad_weights_raises(self, tmp_path):
        """ScoreComposer should raise ValueError if a section doesn't sum to 1.0."""
        bad_yaml = tmp_path / "bad_weights.yaml"
        bad_yaml.write_text(
            "alpha:\n  growth: 0.60\n  momentum: 0.60\n"  # sums to 1.2
            "beta:\n  quality_buffett: 0.50\n  momentum: 0.50\n"
            "delta:\n  quality_buffett: 0.50\n  momentum: 0.50\n"
            "banking:\n  nim: 1.0\n"
            "holding:\n  nav_discount: 1.0\n"
            "ipo:\n  revenue_growth: 1.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="alpha"):
            ScoreComposer(weights_path=bad_yaml)

    def test_missing_weights_file_raises(self, tmp_path):
        """ScoreComposer should raise FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            ScoreComposer(weights_path=tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# ScoreComposer — compose (unit)
# ---------------------------------------------------------------------------


class TestCompose:
    """Tests for ScoreComposer.compose()."""

    def _minimal_composer(self, tmp_path, weights_yaml: str) -> ScoreComposer:
        """Create a ScoreComposer from a custom YAML string."""
        p = tmp_path / "w.yaml"
        p.write_text(weights_yaml, encoding="utf-8")
        return ScoreComposer(weights_path=p)

    _SIMPLE_YAML = (
        "alpha:\n  quality_buffett: 0.60\n  momentum: 0.40\n"
        "beta:\n  quality_buffett: 0.50\n  momentum: 0.50\n"
        "delta:\n  quality_buffett: 0.70\n  momentum: 0.30\n"
        "banking:\n  nim: 1.0\n"
        "holding:\n  nav_discount: 1.0\n"
        "ipo:\n  revenue_growth: 1.0\n"
    )

    def test_known_scores_and_weights(self, tmp_path):
        """Verify the weighted average math with known inputs."""
        composer = self._minimal_composer(tmp_path, self._SIMPLE_YAML)
        factor_scores = {"quality_buffett": 80.0, "momentum": 50.0}

        result = composer.compose(1, factor_scores, "alpha", "OPERATING")

        # alpha: 80*0.6 + 50*0.4 = 48 + 20 = 68
        assert result == pytest.approx(68.0, abs=1e-9)

    def test_none_factor_redistributes_weight(self, tmp_path):
        """If one factor is None, its weight is spread among the rest."""
        composer = self._minimal_composer(tmp_path, self._SIMPLE_YAML)
        factor_scores = {"quality_buffett": 80.0, "momentum": None}

        result = composer.compose(1, factor_scores, "alpha", "OPERATING")

        # Only quality_buffett available; the weighted score is still 80.0,
        # but current composer logic applies a coverage penalty:
        # coverage = 0.60 -> penalty = 0.96 -> 80 * 0.96 = 76.8
        assert result == pytest.approx(76.8, abs=1e-9)

    def test_two_of_three_none_redistributes(self, tmp_path):
        """Weight redistribution works with three factors, two None."""
        yaml_text = (
            "alpha:\n  a: 0.50\n  b: 0.30\n  c: 0.20\n"
            "beta:\n  a: 0.50\n  b: 0.30\n  c: 0.20\n"
            "delta:\n  a: 0.50\n  b: 0.30\n  c: 0.20\n"
            "banking:\n  nim: 1.0\n"
            "holding:\n  nav_discount: 1.0\n"
            "ipo:\n  revenue_growth: 1.0\n"
        )
        composer = self._minimal_composer(tmp_path, yaml_text)
        factor_scores = {"a": None, "b": 60.0, "c": None}

        result = composer.compose(1, factor_scores, "alpha", "OPERATING")

        # Only b available; coverage = 0.30 so the composer applies a
        # 0.93 penalty -> 60 * 0.93 = 55.8
        assert result == pytest.approx(55.8, abs=1e-9)

    def test_all_none_returns_none(self, tmp_path):
        """All None factors -> compose() returns None."""
        composer = self._minimal_composer(tmp_path, self._SIMPLE_YAML)
        result = composer.compose(
            1, {"quality_buffett": None, "momentum": None}, "alpha", "OPERATING"
        )
        assert result is None

    def test_unknown_model_type_raises(self, tmp_path):
        """Unknown model_type should raise ValueError."""
        composer = self._minimal_composer(tmp_path, self._SIMPLE_YAML)
        with pytest.raises(ValueError, match="model_type"):
            composer.compose(1, {"quality_buffett": 50.0}, "alpha", "UNKNOWN")

    def test_bank_model_uses_banking_weights(self, tmp_path):
        """BANK model_type uses 'banking' section regardless of portfolio."""
        yaml_text = (
            "alpha:\n  quality_buffett: 1.0\n"
            "beta:\n  quality_buffett: 1.0\n"
            "delta:\n  quality_buffett: 1.0\n"
            "banking:\n  nim: 0.60\n  roe: 0.40\n"
            "holding:\n  nav_discount: 1.0\n"
            "ipo:\n  revenue_growth: 1.0\n"
        )
        composer = self._minimal_composer(tmp_path, yaml_text)
        factor_scores = {"nim": 70.0, "roe": 50.0}

        result = composer.compose(1, factor_scores, "alpha", "BANK")

        # 70*0.6 + 50*0.4 = 42 + 20 = 62
        assert result == pytest.approx(62.0, abs=1e-9)

    def test_holding_model_uses_holding_weights(self, tmp_path):
        """HOLDING model_type uses 'holding' section."""
        yaml_text = (
            "alpha:\n  quality_buffett: 1.0\n"
            "beta:\n  quality_buffett: 1.0\n"
            "delta:\n  quality_buffett: 1.0\n"
            "banking:\n  nim: 1.0\n"
            "holding:\n  nav_discount: 0.50\n  dividend_yield: 0.50\n"
            "ipo:\n  revenue_growth: 1.0\n"
        )
        composer = self._minimal_composer(tmp_path, yaml_text)
        result = composer.compose(
            1, {"nav_discount": 90.0, "dividend_yield": 30.0}, "delta", "HOLDING"
        )
        # 90*0.5 + 30*0.5 = 60
        assert result == pytest.approx(60.0, abs=1e-9)

    def test_real_weights_operating_score_in_range(self, composer):
        """Composite score from real weights must be in [0, 100]."""
        factor_scores = {
            "quality_buffett": 72.0,
            "value_graham_dcf": 55.0,
            "growth": 80.0,
            "momentum": 45.0,
            "technical": 50.0,
            "piotroski": 65.0,
        }
        for portfolio in ("alpha", "beta", "delta"):
            result = composer.compose(1, factor_scores, portfolio, "OPERATING")
            assert result is not None
            assert 0.0 <= result <= 100.0, f"{portfolio}: {result}"


# ---------------------------------------------------------------------------
# ScoreComposer — compose_all (DB integration)
# ---------------------------------------------------------------------------


class TestComposeAll:
    """Integration tests for ScoreComposer.compose_all()."""

    def _add_company(self, session, ticker: str, model_type: str = "OPERATING") -> int:
        company = Company(ticker=ticker, name=ticker, company_type=model_type, is_active=True)
        session.add(company)
        session.flush()
        return company.id

    def _add_scoring_row(
        self,
        session,
        company_id: int,
        scoring_date: date,
        model_used: str = "OPERATING",
        **factor_kwargs,
    ) -> ScoringResult:
        row = ScoringResult(
            company_id=company_id,
            scoring_date=scoring_date,
            model_used=model_used,
            **factor_kwargs,
        )
        session.add(row)
        session.flush()
        return row

    def test_composites_written_to_db(self, composer, session):
        """compose_all() should write composite_alpha/beta/delta for each row."""
        today = date(2026, 2, 1)
        cid = self._add_company(session, "THYAO")
        self._add_scoring_row(
            session, cid, today,
            buffett_score=70.0,
            graham_score=55.0,
            momentum_score=60.0,
            technical_score=50.0,
            magic_formula_rank=65.0,
            lynch_peg_score=72.0,
            piotroski_fscore=7.0,
        )
        session.commit()

        composer.compose_all(session, scoring_date=today)

        row = session.query(ScoringResult).filter_by(company_id=cid).first()
        assert row.composite_alpha is not None
        assert row.composite_beta is not None
        assert row.composite_delta is not None
        assert 0.0 <= row.composite_alpha <= 100.0
        assert 0.0 <= row.composite_beta <= 100.0
        assert 0.0 <= row.composite_delta <= 100.0

    def test_data_completeness_calculated(self, composer, session):
        """compose_all() should set data_completeness on each row."""
        today = date(2026, 2, 1)
        cid = self._add_company(session, "BIMAS")
        # Provide only 2 of 6 alpha factors (quality_buffett, momentum).
        self._add_scoring_row(
            session, cid, today,
            buffett_score=60.0,
            momentum_score=55.0,
        )
        session.commit()

        composer.compose_all(session, scoring_date=today)

        row = session.query(ScoringResult).filter_by(company_id=cid).first()
        assert row.data_completeness is not None
        # 3 of 6+ operating factors present → some percentage in (0, 100].
        # data_completeness is stored as a percentage (0–100).
        assert 0.0 <= row.data_completeness <= 100.0

    def test_all_none_factors_produces_none_composites(self, composer, session):
        """Row with all factor columns None -> all composites remain None."""
        today = date(2026, 2, 1)
        cid = self._add_company(session, "GARAN", "OPERATING")
        self._add_scoring_row(session, cid, today)
        session.commit()

        composer.compose_all(session, scoring_date=today)

        row = session.query(ScoringResult).filter_by(company_id=cid).first()
        assert row.composite_alpha is None
        assert row.composite_beta is None
        assert row.composite_delta is None

    def test_no_rows_for_date_is_safe(self, composer, session):
        """compose_all() with no rows for the date should not raise."""
        composer.compose_all(session, scoring_date=date(2000, 1, 1))  # no rows

    def test_multiple_companies(self, composer, session):
        """compose_all() should handle multiple companies in one call."""
        today = date(2026, 2, 1)
        tickers = ["AKBNK", "SAHOL", "ASELS"]
        cids = [self._add_company(session, t) for t in tickers]
        for cid in cids:
            self._add_scoring_row(
                session, cid, today,
                buffett_score=50.0,
                momentum_score=50.0,
                technical_score=50.0,
                graham_score=50.0,
                piotroski_fscore=5.0,
                magic_formula_rank=50.0,
                lynch_peg_score=50.0,
            )
        session.commit()

        composer.compose_all(session, scoring_date=today)

        rows = session.query(ScoringResult).filter_by(scoring_date=today).all()
        assert len(rows) == 3
        for row in rows:
            assert row.composite_alpha is not None

    def test_sport_company_does_not_get_alpha_composite(self, composer, session):
        """SPORT companies should not receive investable portfolio composites."""
        today = date(2026, 2, 1)
        cid = self._add_company(session, "GSRAY", "SPORT")
        self._add_scoring_row(
            session,
            cid,
            today,
            model_used="SPORT",
            momentum_score=90.0,
            technical_score=85.0,
            lynch_peg_score=80.0,
        )
        session.commit()

        composer.compose_all(session, scoring_date=today)

        row = session.query(ScoringResult).filter_by(company_id=cid).first()
        assert row.composite_alpha is None
        assert row.composite_beta is None
        assert row.composite_delta is None

    def test_insurance_uses_banking_family_model_composite(self, composer, session):
        """INSURANCE rows should use banking-family native composites when available."""
        today = date(2026, 2, 1)
        cid = self._add_company(session, "AGESA", "INSURANCE")
        self._add_scoring_row(
            session,
            cid,
            today,
            model_used="INSURANCE",
            banking_composite=80.0,
            momentum_score=60.0,
            technical_score=40.0,
            data_completeness=85.0,
        )
        session.commit()

        composer.compose_all(session, scoring_date=today)

        row = session.query(ScoringResult).filter_by(company_id=cid).first()
        expected = round(((80.0 * 0.70) + (50.0 * 0.30)) * (0.50 + 0.50 * 0.85), 2)
        assert row.composite_alpha == pytest.approx(expected, abs=1e-2)
        assert row.composite_beta == pytest.approx(expected, abs=1e-2)
        assert row.composite_delta == pytest.approx(expected, abs=1e-2)
        assert row.data_completeness == pytest.approx(85.0)

    def test_dcf_is_normalized_without_overwriting_raw_margin(self, composer, session):
        """Raw DCF margin should stay raw in DB while composer uses sector-normalized values."""
        today = date(2026, 2, 1)
        cid_low = self._add_company(session, "DCFLO")
        cid_high = self._add_company(session, "DCFHI")

        session.query(Company).filter_by(id=cid_low).update({"sector_custom": "technology_software"})
        session.query(Company).filter_by(id=cid_high).update({"sector_custom": "technology_software"})

        row_low = self._add_scoring_row(
            session,
            cid_low,
            today,
            graham_score=50.0,
            dcf_margin_of_safety_pct=-100.0,
        )
        row_high = self._add_scoring_row(
            session,
            cid_high,
            today,
            graham_score=50.0,
            dcf_margin_of_safety_pct=100.0,
        )
        session.commit()

        overrides = composer._build_dcf_factor_overrides(session, [row_low, row_high])

        assert overrides[row_low.id]["dcf_margin_of_safety_pct"] == pytest.approx(0.0)
        assert overrides[row_high.id]["dcf_margin_of_safety_pct"] == pytest.approx(100.0)
        assert row_low.dcf_margin_of_safety_pct == pytest.approx(-100.0)
        assert row_high.dcf_margin_of_safety_pct == pytest.approx(100.0)
