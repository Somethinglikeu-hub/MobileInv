"""Phase 5 transparency-surface tests.

Covers:
  A. ``scoring.red_flags`` — unit-level detect / serialize / deserialize
  B. DCF breakdown columns persist into ``ScoringResult`` and appear in the
     mobile snapshot's ``scoring_latest`` table
  C. ``portfolio.selector._compute_reason_top_factors`` — correct top-N chips
     with stable tie-break
  D. Mobile snapshot ``open_positions`` table carries all Phase 5 columns
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import bist_picker.read_service as read_service
from bist_picker.db.schema import (
    AdjustedMetric,
    Base,
    Company,
    DailyPrice,
    MacroRegime,
    PortfolioSelection,
    ScoringResult,
)
from bist_picker.mobile_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    export_mobile_snapshot,
)
from bist_picker.portfolio.selector import _compute_reason_top_factors
from bist_picker.scoring.red_flags import (
    DATA_COMPLETENESS_THRESHOLD,
    DCF_OVERVALUED_THRESHOLD,
    PIOTROSKI_LOW_THRESHOLD,
    WEAK_TECHNICAL_THRESHOLD,
    deserialize_flags,
    detect_flags,
    serialize_flags,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def source_engine(monkeypatch):
    """Isolated in-memory source DB with Phase 5 fields populated."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    monkeypatch.setattr(read_service, "get_engine", lambda: engine)

    today = date(2026, 4, 20)
    session = session_factory()
    try:
        company = Company(
            ticker="TUPRS",
            name="Tüpraş",
            company_type="OPERATING",
            sector_bist="Energy",
            sector_custom="Energy",
            free_float_pct=49.0,
            is_bist100=True,
            is_active=True,
        )
        session.add(company)
        session.flush()

        session.add_all([
            DailyPrice(
                company_id=company.id,
                date=today,
                close=160.0,
                adjusted_close=160.0,
                volume=5_000_000,
            ),
            ScoringResult(
                company_id=company.id,
                scoring_date=today,
                model_used="OPERATING",
                composite_alpha=85.0,
                composite_beta=70.0,
                composite_delta=60.0,
                buffett_score=88.0,
                graham_score=75.0,
                piotroski_fscore=70.0,
                piotroski_fscore_raw=7,
                magic_formula_rank=65.0,
                lynch_peg_score=60.0,
                dcf_margin_of_safety_pct=12.0,
                dcf_intrinsic_value=181.0,
                dcf_growth_rate_pct=8.5,
                dcf_discount_rate_pct=14.0,
                dcf_terminal_growth_pct=3.5,
                momentum_score=72.0,
                technical_score=68.0,
                dividend_score=55.0,
                risk_tier="LOW",
                data_completeness=90.0,
                quality_flags_json=None,
            ),
            PortfolioSelection(
                portfolio="ALPHA",
                selection_date=today - timedelta(days=5),
                company_id=company.id,
                entry_price=155.0,
                composite_score=85.0,
                target_price=195.0,
                stop_loss_price=135.0,
                reason_top_factors_json=json.dumps([
                    {"factor": "buffett_score", "label": "Buffett Quality", "value": 88.0},
                    {"factor": "graham_score", "label": "Graham Value", "value": 75.0},
                    {"factor": "piotroski_fscore", "label": "Piotroski", "value": 70.0},
                ]),
            ),
            MacroRegime(
                date=today,
                policy_rate_pct=0.47,
                cpi_yoy_pct=0.35,
                usdtry_rate=38.5,
                regime="RISK_ON",
            ),
        ])
        session.commit()
    finally:
        session.close()

    return engine


# ─── A. Red-flag unit tests ────────────────────────────────────────────────────

class TestDetectFlags:
    def test_no_flags_for_clean_row(self):
        row = {
            "piotroski_fscore_raw": PIOTROSKI_LOW_THRESHOLD,     # exactly threshold → no flag
            "data_completeness": DATA_COMPLETENESS_THRESHOLD,    # exactly threshold → no flag
            "dcf_margin_of_safety_pct": DCF_OVERVALUED_THRESHOLD,# exactly 0 → no flag
            "technical_score": WEAK_TECHNICAL_THRESHOLD,         # exactly 40 → no flag
        }
        assert detect_flags(row) == []

    def test_piotroski_low_fires_below_threshold(self):
        flags = detect_flags({"piotroski_fscore_raw": PIOTROSKI_LOW_THRESHOLD - 1})
        assert "PIOTROSKI_LOW" in flags

    def test_piotroski_low_does_not_fire_at_threshold(self):
        flags = detect_flags({"piotroski_fscore_raw": PIOTROSKI_LOW_THRESHOLD})
        assert "PIOTROSKI_LOW" not in flags

    def test_limited_data_fires_below_threshold(self):
        flags = detect_flags({"data_completeness": DATA_COMPLETENESS_THRESHOLD - 0.1})
        assert "LIMITED_DATA" in flags

    def test_dcf_overvalued_fires_when_negative(self):
        flags = detect_flags({"dcf_margin_of_safety_pct": -5.0})
        assert "DCF_OVERVALUED" in flags

    def test_dcf_overvalued_does_not_fire_at_zero(self):
        flags = detect_flags({"dcf_margin_of_safety_pct": 0.0})
        assert "DCF_OVERVALUED" not in flags

    def test_weak_technical_fires_below_threshold(self):
        flags = detect_flags({"technical_score": WEAK_TECHNICAL_THRESHOLD - 1})
        assert "WEAK_TECHNICAL" in flags

    def test_all_four_flags_fire_together(self):
        row = {
            "piotroski_fscore_raw": 2,
            "data_completeness": 40.0,
            "dcf_margin_of_safety_pct": -10.0,
            "technical_score": 20.0,
        }
        flags = detect_flags(row)
        assert flags == ["PIOTROSKI_LOW", "LIMITED_DATA", "DCF_OVERVALUED", "WEAK_TECHNICAL"]

    def test_flag_order_is_stable(self):
        """Flags always come out in declaration order, not insertion order."""
        row = {
            "technical_score": 10.0,               # WEAK_TECHNICAL (4th)
            "piotroski_fscore_raw": 1,              # PIOTROSKI_LOW (1st)
            "dcf_margin_of_safety_pct": -1.0,       # DCF_OVERVALUED (3rd)
            "data_completeness": 50.0,              # LIMITED_DATA (2nd)
        }
        assert detect_flags(row) == [
            "PIOTROSKI_LOW",
            "LIMITED_DATA",
            "DCF_OVERVALUED",
            "WEAK_TECHNICAL",
        ]

    def test_none_values_skip_silently(self):
        """Missing fields never fire a flag."""
        assert detect_flags({}) == []
        assert detect_flags({"piotroski_fscore_raw": None}) == []


class TestSerializeDeserializeFlags:
    def test_serialize_nonempty_produces_json_array(self):
        payload = serialize_flags(["PIOTROSKI_LOW", "LIMITED_DATA"])
        assert payload == '["PIOTROSKI_LOW","LIMITED_DATA"]'

    def test_serialize_empty_returns_none(self):
        assert serialize_flags([]) is None

    def test_deserialize_none_returns_empty(self):
        assert deserialize_flags(None) == []

    def test_deserialize_empty_string_returns_empty(self):
        assert deserialize_flags("") == []

    def test_roundtrip(self):
        flags = ["PIOTROSKI_LOW", "DCF_OVERVALUED"]
        assert deserialize_flags(serialize_flags(flags)) == flags

    def test_deserialize_malformed_returns_empty(self):
        assert deserialize_flags("not-json") == []
        assert deserialize_flags("{}") == []


# ─── B. DCF breakdown in mobile snapshot ──────────────────────────────────────

def test_dcf_breakdown_columns_in_snapshot(source_engine, tmp_path):
    """scoring_latest in the snapshot carries all four DCF breakdown columns."""
    out = tmp_path / "snap.db"
    export_mobile_snapshot(out)

    with sqlite3.connect(out) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT dcf_intrinsic_value, dcf_growth_rate_pct, "
            "dcf_discount_rate_pct, dcf_terminal_growth_pct "
            "FROM scoring_latest WHERE ticker = 'TUPRS'"
        ).fetchone()

    assert row is not None
    assert row["dcf_intrinsic_value"] == pytest.approx(181.0)
    assert row["dcf_growth_rate_pct"] == pytest.approx(8.5)
    assert row["dcf_discount_rate_pct"] == pytest.approx(14.0)
    assert row["dcf_terminal_growth_pct"] == pytest.approx(3.5)


# ─── C. Reason-top-factors chip selection ─────────────────────────────────────

class TestComputeReasonTopFactors:
    def test_returns_top_3_by_score(self):
        factors = {
            "buffett_score": 90.0,
            "graham_score": 80.0,
            "piotroski_fscore": 70.0,
            "magic_formula_rank": 60.0,
            "momentum_score": 50.0,
        }
        chips = _compute_reason_top_factors(factors, top_n=3)
        assert len(chips) == 3
        assert chips[0]["factor"] == "buffett_score"
        assert chips[0]["value"] == pytest.approx(90.0)
        assert chips[1]["factor"] == "graham_score"
        assert chips[2]["factor"] == "piotroski_fscore"

    def test_none_values_are_skipped(self):
        factors = {
            "buffett_score": None,
            "graham_score": 80.0,
            "piotroski_fscore": None,
            "momentum_score": 70.0,
        }
        chips = _compute_reason_top_factors(factors, top_n=3)
        factor_names = [c["factor"] for c in chips]
        assert "buffett_score" not in factor_names
        assert "piotroski_fscore" not in factor_names

    def test_stable_tiebreak_by_declaration_order(self):
        """When two factors score equally, declaration order decides."""
        # buffett_score is declared before graham_score in _REASON_FACTOR_LABELS
        factors = {"buffett_score": 75.0, "graham_score": 75.0}
        chips = _compute_reason_top_factors(factors, top_n=1)
        assert chips[0]["factor"] == "buffett_score"

    def test_fewer_factors_than_top_n(self):
        factors = {"buffett_score": 88.0}
        chips = _compute_reason_top_factors(factors, top_n=3)
        assert len(chips) == 1

    def test_label_is_populated(self):
        factors = {"buffett_score": 88.0}
        chips = _compute_reason_top_factors(factors)
        assert chips[0]["label"] == "Buffett Quality"

    def test_value_is_rounded_to_two_dp(self):
        factors = {"buffett_score": 88.123456}
        chips = _compute_reason_top_factors(factors)
        assert chips[0]["value"] == pytest.approx(88.12)


# ─── D. Phase 5 columns in open_positions table ───────────────────────────────

def test_mobile_snapshot_open_positions_phase5_columns(source_engine, tmp_path):
    """open_positions table in the snapshot carries Phase 5 transparency cols."""
    out = tmp_path / "snap2.db"
    export_mobile_snapshot(out)

    with sqlite3.connect(out) as conn:
        # Verify columns exist via PRAGMA
        col_names = {
            row[1]
            for row in conn.execute("PRAGMA table_info(open_positions)").fetchall()
        }

    expected_new_cols = {
        "stop_pct_from_entry",
        "reason_top_factors_json",
        "quality_flags_json",
        "dcf_margin_of_safety_pct",
        "dcf_intrinsic_value",
        "dcf_growth_rate_pct",
        "dcf_discount_rate_pct",
        "dcf_terminal_growth_pct",
    }
    assert expected_new_cols.issubset(col_names)


def test_mobile_snapshot_schema_version_matches_constant(source_engine, tmp_path):
    """Schema version in the exported file matches the module constant.

    Kept at 1 until the APK is updated to accept version 2 — bump both
    together when Phase 5 UI lands on the Android side.
    """
    out = tmp_path / "snap3.db"
    export_mobile_snapshot(out)

    with sqlite3.connect(out) as conn:
        version = conn.execute(
            "SELECT schema_version FROM snapshot_metadata WHERE id = 1"
        ).fetchone()[0]

    assert version == SNAPSHOT_SCHEMA_VERSION
    assert SNAPSHOT_SCHEMA_VERSION == 1
