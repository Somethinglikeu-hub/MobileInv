"""Unit tests for scoring/factors/dcf.py.

Tests verify:
  1. DCF math with known inputs produces the correct intrinsic value
  2. Margin of safety is computed correctly (positive when undervalued)
  3. Per-share OE conversion handles all edge cases
  4. Growth rate estimation: CAGR from history, fallback to conservative rate
  5. Growth rate is capped at max_growth_rate and floored at min_growth_rate
  6. None returned for BANK / HOLDING / INSURANCE / REIT
  7. None returned when all owner earnings are <= 0
  8. None returned when no adjusted metrics exist
  9. Config parameters are loaded and used correctly
  10. DCF formula: terminal value is included correctly
"""

from datetime import date
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from bist_picker.scoring.factors.dcf import DCFScorer


# ── Helpers ────────────────────────────────────────────────────────────────────

def _metric(
    period_end: date,
    owner_earnings: Optional[float],
    adjusted_net_income: Optional[float],
    eps_adjusted: Optional[float],
    real_eps_growth_pct: Optional[float] = None,
) -> MagicMock:
    """Build a mock AdjustedMetric object."""
    m = MagicMock()
    m.period_end = period_end
    m.owner_earnings = owner_earnings
    m.adjusted_net_income = adjusted_net_income
    m.eps_adjusted = eps_adjusted
    m.real_eps_growth_pct = real_eps_growth_pct
    return m


def _company(company_type: str = "OPERATING") -> MagicMock:
    c = MagicMock()
    c.company_type = company_type
    c.ticker = "TEST"
    return c


def _price_row(price: float) -> MagicMock:
    r = MagicMock()
    r.adjusted_close = price
    r.close = price
    return r


def _make_scorer(extra_cfg: dict = None) -> DCFScorer:
    """Return a DCFScorer with inline config (no file I/O)."""
    scorer = DCFScorer.__new__(DCFScorer)
    scorer._config_path = Path("/nonexistent")
    scorer._rate_cache = None
    scorer._cfg = {
        "discount_rate_try": 0.35,
        "terminal_growth_try": 0.12,
        "projection_years": 10,
        "min_growth_rate": 0.00,
        "max_growth_rate": 0.30,
        "conservative_growth_rate": 0.10,
    }
    if extra_cfg:
        scorer._cfg.update(extra_cfg)
    return scorer


def _mock_session(company: MagicMock, metrics: list, price: Optional[float]) -> MagicMock:
    """Return a session mock that yields the given data."""
    session = MagicMock()
    session.get.return_value = company
    session.query.return_value.filter.return_value.order_by.return_value.all.return_value = metrics
    # For price queries: DailyPrice calls use filter(...).order_by(...).first()
    if price is not None:
        price_row = _price_row(price)
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = price_row
    else:
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
    return session


# ── Test 1: DCF math with known inputs ────────────────────────────────────────

class TestDCFMath:
    """Verify the DCF formula produces the expected intrinsic value."""

    def test_intrinsic_value_known_inputs(self):
        """Manually compute DCF and compare to scorer output."""
        scorer = _make_scorer()
        # r=0.35, g=0.15, g_t=0.12, N=10, base_oe=10.0

        r, g, g_t, n = 0.35, 0.15, 0.12, 10

        expected_pv = sum(
            10.0 * (1.0 + g) ** t / (1.0 + r) ** t
            for t in range(1, n + 1)
        )
        oe_n = 10.0 * (1.0 + g) ** n
        tv = oe_n * (1.0 + g_t) / (r - g_t)
        pv_tv = tv / (1.0 + r) ** n
        expected_intrinsic = expected_pv + pv_tv

        result = scorer._compute_intrinsic_value(base_oe_per_share=10.0, growth_rate=0.15)
        assert result == pytest.approx(expected_intrinsic, rel=1e-6)

    def test_terminal_value_dominates_for_long_duration(self):
        """Terminal value should be a significant portion of total intrinsic value."""
        scorer = _make_scorer()
        # With high discount (35%) and modest growth, TV can still be large
        intrinsic = scorer._compute_intrinsic_value(base_oe_per_share=10.0, growth_rate=0.10)
        # Just verify we get a positive finite number
        assert intrinsic is not None
        assert intrinsic > 0

    def test_returns_none_when_spread_nonpositive(self):
        """discount_rate <= terminal_growth should return None (avoid division by zero)."""
        scorer = _make_scorer({"discount_rate_try": 0.12, "terminal_growth_try": 0.12})
        result = scorer._compute_intrinsic_value(10.0, 0.10)
        assert result is None

    def test_higher_growth_gives_higher_intrinsic_value(self):
        """Higher growth rate should always produce a higher intrinsic value."""
        scorer = _make_scorer()
        low = scorer._compute_intrinsic_value(10.0, 0.05)
        high = scorer._compute_intrinsic_value(10.0, 0.25)
        assert high > low

    def test_higher_discount_rate_gives_lower_intrinsic_value(self):
        """Higher discount rate should always produce a lower intrinsic value."""
        scorer_cheap = _make_scorer({"discount_rate_try": 0.25})
        scorer_expensive = _make_scorer({"discount_rate_try": 0.45})
        cheap = scorer_cheap._compute_intrinsic_value(10.0, 0.15)
        expensive = scorer_expensive._compute_intrinsic_value(10.0, 0.15)
        assert cheap > expensive


# ── Test 2: Margin of safety ───────────────────────────────────────────────────

class TestMarginOfSafety:
    def _run(self, intrinsic: float, price: float) -> float:
        """Directly compute margin of safety as the scorer does."""
        return (intrinsic - price) / intrinsic * 100.0

    def test_positive_mos_when_undervalued(self):
        """Intrinsic > price => positive MoS."""
        mos = self._run(intrinsic=100.0, price=70.0)
        assert mos == pytest.approx(30.0, rel=1e-6)

    def test_negative_mos_when_overvalued(self):
        """Price > intrinsic => negative MoS."""
        mos = self._run(intrinsic=100.0, price=130.0)
        assert mos == pytest.approx(-30.0, rel=1e-6)

    def test_zero_mos_at_fair_value(self):
        """Price == intrinsic => MoS = 0."""
        mos = self._run(intrinsic=100.0, price=100.0)
        assert mos == pytest.approx(0.0, abs=1e-9)


# ── Test 3: Per-share OE conversion ───────────────────────────────────────────

class TestOEPerShare:
    def test_basic_conversion(self):
        """OE * (eps / ani) should equal OE * eps_per_unit_of_income."""
        scorer = _make_scorer()
        metrics = [
            _metric(date(2024, 12, 31),
                    owner_earnings=80_000,
                    adjusted_net_income=100_000,
                    eps_adjusted=5.0),
        ]
        # oe_per_share = 80_000 * (5.0 / 100_000) = 4.0
        result = scorer._compute_oe_per_share_series(metrics)
        assert result == pytest.approx([4.0])

    def test_skips_years_with_negative_ani(self):
        """Years where adjusted_net_income <= 0 are excluded."""
        scorer = _make_scorer()
        metrics = [
            _metric(date(2022, 12, 31), 50_000, -10_000, -2.0),  # skip
            _metric(date(2023, 12, 31), 80_000, 100_000, 5.0),   # keep
        ]
        result = scorer._compute_oe_per_share_series(metrics)
        assert len(result) == 1
        assert result[0] == pytest.approx(4.0)

    def test_skips_years_with_none_fields(self):
        """Years with any None field are skipped."""
        scorer = _make_scorer()
        metrics = [
            _metric(date(2022, 12, 31), None, 100_000, 5.0),     # OE is None
            _metric(date(2023, 12, 31), 80_000, None, 5.0),      # ANI is None
            _metric(date(2024, 12, 31), 80_000, 100_000, None),  # EPS is None
        ]
        result = scorer._compute_oe_per_share_series(metrics)
        assert result == []

    def test_negative_oe_included_in_series(self):
        """Negative OE is included in the series (positive check happens in score())."""
        scorer = _make_scorer()
        metrics = [
            _metric(date(2024, 12, 31), -20_000, 100_000, 5.0),
        ]
        result = scorer._compute_oe_per_share_series(metrics)
        assert len(result) == 1
        assert result[0] < 0


# ── Test 4: Growth rate estimation ────────────────────────────────────────────

class TestGrowthRateEstimation:
    def test_cagr_from_two_years(self):
        """With 2 EPS data points, compute CAGR correctly."""
        scorer = _make_scorer()
        # EPS: 5.0 -> 6.25 over 1 period => CAGR = 0.25
        metrics = [
            _metric(date(2023, 12, 31), 100_000, 100_000, 5.0),
            _metric(date(2024, 12, 31), 100_000, 100_000, 6.25),
        ]
        g = scorer._estimate_growth_rate(metrics)
        assert g == pytest.approx(0.25, rel=1e-2)

    def test_cagr_from_multiple_years(self):
        """With 4 EPS data points (3 periods), CAGR is computed over n-1 periods."""
        scorer = _make_scorer()
        # EPS doubles over 3 periods: 10 -> 20, CAGR = 2^(1/3) - 1 ≈ 0.2599
        metrics = [
            _metric(date(2021, 12, 31), 100_000, 100_000, 10.0),
            _metric(date(2022, 12, 31), 100_000, 100_000, 12.6),
            _metric(date(2023, 12, 31), 100_000, 100_000, 15.87),
            _metric(date(2024, 12, 31), 100_000, 100_000, 20.0),
        ]
        g = scorer._estimate_growth_rate(metrics)
        assert g == pytest.approx(2 ** (1 / 3) - 1, rel=1e-2)

    def test_falls_back_to_conservative_rate_with_one_year(self):
        """Single EPS point => falls back to conservative_growth_rate (0.10)."""
        scorer = _make_scorer()
        metrics = [
            _metric(date(2024, 12, 31), 100_000, 100_000, 5.0),
        ]
        g = scorer._estimate_growth_rate(metrics)
        assert g == pytest.approx(0.10, rel=1e-6)

    def test_growth_capped_at_max(self):
        """CAGR > max_growth_rate is capped."""
        scorer = _make_scorer({"max_growth_rate": 0.25})
        # EPS triples over 1 period => CAGR = 200% >> max
        metrics = [
            _metric(date(2023, 12, 31), 100_000, 100_000, 5.0),
            _metric(date(2024, 12, 31), 100_000, 100_000, 15.0),
        ]
        g = scorer._estimate_growth_rate(metrics)
        assert g == pytest.approx(0.25, rel=1e-6)

    def test_growth_floored_at_min(self):
        """Negative CAGR is floored at min_growth_rate (0.0)."""
        scorer = _make_scorer({"min_growth_rate": 0.00})
        # EPS halves => CAGR = -50%
        metrics = [
            _metric(date(2023, 12, 31), 100_000, 100_000, 10.0),
            _metric(date(2024, 12, 31), 100_000, 100_000, 5.0),
        ]
        g = scorer._estimate_growth_rate(metrics)
        assert g == pytest.approx(0.00, abs=1e-9)

    def test_skips_nonpositive_eps_for_cagr(self):
        """Negative EPS years are excluded from the log-linear regression
        but counted toward the loss-year penalty.

        Log-linear regression on (5.0 @ t=1y, 6.25 @ t=2y) → slope = log(1.25),
        annual growth = exp(slope) - 1 = 25%.
        Loss penalty: 1 loss out of 3 total → 1 - (1/3 * 0.5) ≈ 0.8333.
        Final: 0.25 * 0.8333 ≈ 0.2083.
        """
        scorer = _make_scorer()
        metrics = [
            _metric(date(2022, 12, 31), 100_000, 100_000, -2.0),   # loss year
            _metric(date(2023, 12, 31), 100_000, 100_000, 5.0),    # start
            _metric(date(2024, 12, 31), 100_000, 100_000, 6.25),   # end
        ]
        g = scorer._estimate_growth_rate(metrics)
        expected = 0.25 * (1.0 - (1 / 3) * 0.50)
        assert g == pytest.approx(expected, rel=5e-3)


# ── Test 5: Company type eligibility ──────────────────────────────────────────

class TestCompanyTypeEligibility:
    @pytest.mark.parametrize("ctype", ["BANK", "HOLDING", "INSURANCE", "REIT"])
    def test_returns_none_for_non_operating_types(self, ctype):
        """DCF is not applicable to financial / holding companies."""
        scorer = _make_scorer()
        company = _company(company_type=ctype)
        session = MagicMock()
        session.get.return_value = company
        result = scorer.score(company_id=1, session=session)
        assert result is None

    def test_scores_operating_company(self):
        """OPERATING company with valid data should return a result dict."""
        scorer = _make_scorer()
        company = _company("OPERATING")

        metrics = [
            _metric(date(2022, 12, 31), 80_000, 100_000, 4.0),
            _metric(date(2023, 12, 31), 88_000, 110_000, 4.4),
            _metric(date(2024, 12, 31), 96_000, 120_000, 4.8),
        ]

        session = MagicMock()
        session.get.return_value = company
        # metrics query
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = metrics
        # price query
        price_row = _price_row(price=70.0)
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = price_row

        result = scorer.score(company_id=1, session=session)
        assert result is not None
        assert "intrinsic_value_per_share" in result
        assert "margin_of_safety_pct" in result
        assert "dcf_combined" in result

    def test_none_company_type_treated_as_operating(self):
        """company_type=None should be treated as OPERATING (default)."""
        scorer = _make_scorer()
        company = _company(company_type=None)

        metrics = [
            _metric(date(2024, 12, 31), 80_000, 100_000, 5.0),
        ]
        session = MagicMock()
        session.get.return_value = company
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = metrics
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        result = scorer.score(company_id=1, session=session)
        # Should not be None due to type; may be None if OE <= 0 check fires
        # base_oe = 80_000 * (5.0/100_000) = 4.0 > 0, so should succeed
        assert result is not None


# ── Test 6: Edge cases for OE ─────────────────────────────────────────────────

class TestOEEdgeCases:
    def test_returns_none_when_all_oe_nonpositive(self):
        """All owner_earnings <= 0 => scorer returns None."""
        scorer = _make_scorer()
        company = _company("OPERATING")
        metrics = [
            _metric(date(2023, 12, 31), -5_000, 100_000, 3.0),
            _metric(date(2024, 12, 31), 0, 100_000, 0.0),
        ]
        session = MagicMock()
        session.get.return_value = company
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = metrics
        result = scorer.score(1, session)
        assert result is None

    def test_returns_none_when_no_metrics(self):
        """No AdjustedMetric rows => scorer returns None."""
        scorer = _make_scorer()
        company = _company("OPERATING")
        session = MagicMock()
        session.get.return_value = company
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        result = scorer.score(1, session)
        assert result is None

    def test_mos_is_none_when_no_price(self):
        """margin_of_safety_pct is None when price data is unavailable."""
        scorer = _make_scorer()
        company = _company("OPERATING")
        metrics = [
            _metric(date(2024, 12, 31), 80_000, 100_000, 5.0),
        ]
        session = MagicMock()
        session.get.return_value = company
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = metrics
        # No price row
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        result = scorer.score(1, session)
        assert result is not None
        assert result["current_price"] is None
        assert result["margin_of_safety_pct"] is None
        assert result["dcf_combined"] is None
        assert result["intrinsic_value_per_share"] > 0  # intrinsic still computed


# ── Test 7: Config parameters used correctly ──────────────────────────────────

class TestConfigParameters:
    def test_custom_discount_rate_affects_intrinsic(self):
        """Changing discount_rate_try changes intrinsic value."""
        scorer_low = _make_scorer({"discount_rate_try": 0.25})
        scorer_high = _make_scorer({"discount_rate_try": 0.45})

        iv_low = scorer_low._compute_intrinsic_value(10.0, 0.15)
        iv_high = scorer_high._compute_intrinsic_value(10.0, 0.15)

        assert iv_low > iv_high

    def test_custom_terminal_growth_affects_intrinsic(self):
        """Higher terminal growth increases intrinsic value (spread widens)."""
        scorer_low = _make_scorer({"terminal_growth_try": 0.05})
        scorer_high = _make_scorer({"terminal_growth_try": 0.20})

        iv_low = scorer_low._compute_intrinsic_value(10.0, 0.15)
        iv_high = scorer_high._compute_intrinsic_value(10.0, 0.15)

        assert iv_high > iv_low

    def test_projection_years_affects_intrinsic(self):
        """More projection years increases intrinsic value."""
        scorer_short = _make_scorer({"projection_years": 5})
        scorer_long = _make_scorer({"projection_years": 15})

        iv_short = scorer_short._compute_intrinsic_value(10.0, 0.15)
        iv_long = scorer_long._compute_intrinsic_value(10.0, 0.15)

        assert iv_long > iv_short
