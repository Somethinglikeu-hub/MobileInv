"""Tests for DCFScorer dynamic terminal growth and macro.yaml ERP.

Phase 2 added two dynamic paths to DCF valuation:
  1. Terminal growth = MacroRegime.inflation_expectation_24m_pct + real_growth_pct
     (clamped to macro.yaml [min_floor, max_ceiling]), fallback to
     thresholds.yaml terminal_growth_try when that column is empty.
  2. Equity risk premium pulled from macro.yaml erp.equity_risk_premium_try
     in the dynamic discount rate calculation, fallback to thresholds.yaml.

These tests cover both paths + the fallback chain. No network / no API.
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.db.schema import Base, MacroRegime
from bist_picker.scoring.factors.dcf import DCFScorer


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


def _write_thresholds(tmp_path: Path, terminal_growth_try: float = 0.08) -> Path:
    path = tmp_path / "thresholds.yaml"
    payload = {
        "dcf": {
            "dynamic_discount_rate": True,
            "equity_risk_premium_try": 0.06,
            "min_rate_terminal_spread": 0.05,
            "discount_rate_try": 0.35,
            "terminal_growth_try": terminal_growth_try,
            "projection_years": 10,
        }
    }
    path.write_text(yaml.safe_dump(payload))
    return path


def _write_macro(
    tmp_path: Path,
    erp: float = 0.09,
    real_growth: float = 0.02,
    floor: float = 0.05,
    ceiling: float = 0.15,
) -> Path:
    path = tmp_path / "macro.yaml"
    payload = {
        "erp": {
            "equity_risk_premium_try": erp,
            "last_updated": "2025-07-01",
            "stale_after_days": 90,
        },
        "terminal_growth": {
            "real_growth_pct": real_growth,
            "min_floor": floor,
            "max_ceiling": ceiling,
        },
    }
    path.write_text(yaml.safe_dump(payload))
    return path


# ── Terminal growth tests ──────────────────────────────────────────────────


def test_terminal_growth_uses_macro_regime_when_available(session, tmp_path):
    """Dynamic path: 24m CPI expectation + real growth, clamped."""
    session.add(
        MacroRegime(date=date(2025, 6, 1), inflation_expectation_24m_pct=0.12)
    )
    session.commit()

    scorer = DCFScorer(
        config_path=_write_thresholds(tmp_path),
        macro_config_path=_write_macro(tmp_path, real_growth=0.02),
    )
    g, source = scorer._resolve_terminal_growth(session, scoring_date=date(2025, 7, 1))

    assert source == "dynamic_cpi_plus_real"
    assert g == pytest.approx(0.14)  # 0.12 + 0.02


def test_terminal_growth_clamped_to_ceiling(session, tmp_path):
    """Absurdly high CPI expectation is clamped to max_ceiling."""
    session.add(
        MacroRegime(date=date(2025, 6, 1), inflation_expectation_24m_pct=0.40)
    )
    session.commit()

    scorer = DCFScorer(
        config_path=_write_thresholds(tmp_path),
        macro_config_path=_write_macro(tmp_path, ceiling=0.15),
    )
    g, _ = scorer._resolve_terminal_growth(session, scoring_date=date(2025, 7, 1))

    assert g == pytest.approx(0.15)


def test_terminal_growth_falls_back_when_no_macro_row(session, tmp_path):
    """No MacroRegime row → static terminal_growth_try, clamped."""
    scorer = DCFScorer(
        config_path=_write_thresholds(tmp_path, terminal_growth_try=0.08),
        macro_config_path=_write_macro(tmp_path),
    )
    g, source = scorer._resolve_terminal_growth(session, scoring_date=date(2025, 7, 1))

    assert source == "static_fallback"
    assert g == pytest.approx(0.08)


def test_terminal_growth_falls_back_when_expectation_null(session, tmp_path):
    """MacroRegime row exists but expectation is NULL → fallback."""
    session.add(
        MacroRegime(date=date(2025, 6, 1), policy_rate_pct=0.50)
    )
    session.commit()

    scorer = DCFScorer(
        config_path=_write_thresholds(tmp_path),
        macro_config_path=_write_macro(tmp_path),
    )
    g, source = scorer._resolve_terminal_growth(session, scoring_date=date(2025, 7, 1))

    assert source == "static_fallback"


def test_terminal_growth_respects_scoring_date_cutoff(session, tmp_path):
    """Only uses rows on or before scoring_date."""
    session.add_all([
        MacroRegime(date=date(2025, 1, 1), inflation_expectation_24m_pct=0.10),
        MacroRegime(date=date(2025, 6, 1), inflation_expectation_24m_pct=0.20),
    ])
    session.commit()

    scorer = DCFScorer(
        config_path=_write_thresholds(tmp_path),
        macro_config_path=_write_macro(tmp_path, real_growth=0.02, ceiling=0.30),
    )
    # Before the June row → should use January (0.10 + 0.02 = 0.12)
    g, _ = scorer._resolve_terminal_growth(session, scoring_date=date(2025, 3, 1))
    assert g == pytest.approx(0.12)

    # After → should use June (0.20 + 0.02 = 0.22)
    scorer._terminal_growth_cache = None  # reset cache for second call
    g, _ = scorer._resolve_terminal_growth(session, scoring_date=date(2025, 7, 1))
    assert g == pytest.approx(0.22)


# ── ERP tests ──────────────────────────────────────────────────────────────


def test_erp_prefers_macro_yaml_over_thresholds(tmp_path):
    """macro.yaml erp.equity_risk_premium_try wins over thresholds.yaml."""
    scorer = DCFScorer(
        config_path=_write_thresholds(tmp_path),  # thresholds has ERP 0.06
        macro_config_path=_write_macro(tmp_path, erp=0.0952),
    )
    assert scorer._get_erp() == pytest.approx(0.0952)


def test_erp_falls_back_to_thresholds_when_macro_missing(tmp_path):
    """No macro.yaml → use thresholds.yaml equity_risk_premium_try."""
    thresholds = _write_thresholds(tmp_path)
    missing_macro = tmp_path / "missing_macro.yaml"  # not written
    scorer = DCFScorer(config_path=thresholds, macro_config_path=missing_macro)

    assert scorer._get_erp() == pytest.approx(0.06)


def test_dynamic_discount_rate_uses_macro_erp(session, tmp_path):
    """Discount rate = policy + macro ERP when both available."""
    session.add(MacroRegime(date=date(2025, 6, 1), policy_rate_pct=0.45))
    session.commit()

    scorer = DCFScorer(
        config_path=_write_thresholds(tmp_path),
        macro_config_path=_write_macro(tmp_path, erp=0.0952),
    )
    r, source = scorer._resolve_discount_rate(session, scoring_date=date(2025, 7, 1))

    assert source == "dynamic_policy_plus_erp"
    assert r == pytest.approx(0.45 + 0.0952)
