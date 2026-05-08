"""Unit tests for portfolio/selector.py.

Tests verify:
  1. Basic selection: top picks chosen by composite score
  2. Sector constraint: max 2 stocks from the same custom sub-sector
  3. Bank constraint: max 1 bank per portfolio
  4. Turnover penalty: incumbents in top-10 are retained unless a new
     candidate beats them by >15%
  5. Turnover bypass: incumbents outside top-10 are replaced freely
  6. Turnover bypass: incumbents beaten by >15% are replaced

All tests bypass the DB and UniverseBuilder by patching
_fetch_candidates and get_universe so the selector operates on
controlled in-memory data.
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bist_picker.portfolio.selector import (
    PortfolioSelector,
    _MAX_BANKS,
    _MAX_PER_SECTOR,
    _PICKS_PER_PORTFOLIO,
    _STOP_LOSS_FACTOR,
    _TURNOVER_THRESHOLD,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_candidate(
    company_id: int,
    ticker: str,
    score: float,
    sector: str = "SECTOR_A",
    company_type: str = "OPERATING",
    dcf_mos: float | None = None,
    data_completeness: float | None = None,
    technical_score: float | None = None,
    above_200ma: bool | None = None,
) -> dict:
    """Build a minimal candidate dict matching what _fetch_candidates returns."""
    return {
        "company_id": company_id,
        "ticker": ticker,
        "score": score,
        "sector_custom": sector,
        "company_type": company_type,
        "dcf_mos": dcf_mos,
        "data_completeness": data_completeness,
        "technical_score": technical_score,
        "above_200ma": above_200ma,
    }


def _make_selector() -> PortfolioSelector:
    """Return a PortfolioSelector with a fixed scoring date (no DB needed)."""
    return PortfolioSelector(scoring_date=date(2026, 2, 1))


def _mock_session() -> MagicMock:
    """Return a bare-minimum mock SQLAlchemy session."""
    return MagicMock()


def _patch_selector(selector: PortfolioSelector, candidates: list[dict]) -> None:
    """Patch _fetch_candidates to return *candidates* and get_universe to return all ids."""
    all_ids = {c["company_id"] for c in candidates}
    selector._universe.get_universe = MagicMock(return_value=list(all_ids))
    selector._fetch_candidates = MagicMock(return_value=candidates)
    # No price data in unit tests — _get_latest_price returns a fixed value
    selector._get_latest_price = MagicMock(return_value=100.0)


# ── Helper to run selection with mocked data ──────────────────────────────────

def _select(candidates: list[dict], current_holdings: list[int] | None = None) -> list[dict]:
    """Run ALPHA selection on *candidates* with optional incumbents."""
    sel = _make_selector()
    session = _mock_session()
    _patch_selector(sel, candidates)
    return sel.select("ALPHA", session, current_holdings=current_holdings)


# ── Test 1: Happy path — top 5 by score ──────────────────────────────────────

class TestBasicSelection:
    def test_picks_top_n(self):
        """Selector should return the top _PICKS_PER_PORTFOLIO candidates by score."""
        candidates = [
            _make_candidate(i, f"STK{i:02d}", score=float(100 - i), sector=f"SEC{i}")
            for i in range(1, 21)
        ]
        picks = _select(candidates)

        assert len(picks) == _PICKS_PER_PORTFOLIO
        tickers = [p["ticker"] for p in picks]
        # Top _PICKS_PER_PORTFOLIO by score are STK01 (99) ... STK0N
        expected = [f"STK{i:02d}" for i in range(1, _PICKS_PER_PORTFOLIO + 1)]
        assert tickers == expected

    def test_ranks_are_sequential(self):
        """Ranks should be 1 through _PICKS_PER_PORTFOLIO in order."""
        candidates = [
            _make_candidate(i, f"STK{i:02d}", score=float(100 - i), sector=f"SEC{i}")
            for i in range(1, 10)
        ]
        picks = _select(candidates)
        assert [p["rank"] for p in picks] == list(range(1, _PICKS_PER_PORTFOLIO + 1))

    def test_fewer_than_target_count_candidates_returns_all(self):
        """If universe has fewer than the target count, return all available."""
        candidates = [
            _make_candidate(i, f"STK{i:02d}", score=float(90 - i), sector=f"SEC{i}")
            for i in range(1, 4)
        ]
        picks = _select(candidates)
        assert len(picks) == 3

    def test_stop_loss_is_entry_times_factor(self):
        """stop_loss == entry_price * 0.82 for every pick."""
        candidates = [
            _make_candidate(i, f"STK{i:02d}", score=float(90 - i), sector=f"SEC{i}")
            for i in range(1, 10)
        ]
        picks = _select(candidates)
        for pick in picks:
            assert pick["entry_price"] == 100.0
            assert pick["stop_loss"] == pytest.approx(100.0 * _STOP_LOSS_FACTOR)

    def test_target_price_dcf_path(self):
        """When dcf_mos is set, target_price = entry / (1 - mos/100)."""
        candidates = [
            _make_candidate(i, f"STK{i:02d}", score=float(90 - i), sector=f"SEC{i}", dcf_mos=25.0)
            for i in range(1, 10)
        ]
        picks = _select(candidates)
        # intrinsic = 100 / (1 - 0.25) = 133.33...
        for pick in picks:
            assert pick["target_price"] == pytest.approx(100.0 / 0.75, rel=1e-3)

    def test_target_price_score_implied_fallback(self):
        """Without dcf_mos, target_price uses score-implied upside (>=10%)."""
        candidates = [
            _make_candidate(i, f"STK{i:02d}", score=0.0, sector=f"SEC{i}")
            for i in range(1, 10)
        ]
        picks = _select(candidates)
        for pick in picks:
            # score=0 => upside = max(0.10, 0/400) = 0.10
            assert pick["target_price"] == pytest.approx(100.0 * 1.10, rel=1e-3)

    def test_target_price_penalizes_low_data_completeness_percentage(self):
        """Low completeness should reduce the fallback upside on the 0-100 scale."""
        selector = _make_selector()
        candidate = _make_candidate(
            1,
            "LOWDATA",
            score=100.0,
            sector="SEC1",
            data_completeness=50.0,
        )

        target = selector._compute_target_price(candidate, 100.0)

        assert target == pytest.approx(121.0, rel=1e-3)


# ── Test 2: Max-2-per-sector constraint ───────────────────────────────────────

class TestSectorConstraint:
    def test_max_2_per_sector(self):
        """No more than 2 picks from any single sub-sector."""
        # Top scorers in same sector, remaining names spread across other sectors
        candidates = [
            _make_candidate(i, f"SAME{i:02d}", score=float(100 - i), sector="CROWDED")
            for i in range(1, 6)
        ] + [
            _make_candidate(10 + i, f"OTH{i:02d}", score=float(60 - i), sector=f"SEC{i}")
            for i in range(1, 16)
        ]

        picks = _select(candidates)
        sector_counts: dict[str, int] = {}
        for p in picks:
            sector = next(
                c["sector_custom"] for c in candidates if c["company_id"] == p["company_id"]
            )
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        assert all(
            count <= _MAX_PER_SECTOR for count in sector_counts.values()
        ), f"Sector constraint violated: {sector_counts}"

    def test_crowded_sector_causes_fallthrough_to_lower_scorers(self):
        """When top _MAX_PER_SECTOR+1 are in same sector, overflow gets skipped."""
        # Build top (_MAX_PER_SECTOR + 1) candidates in CROWDED; the last must be skipped.
        crowded = [
            _make_candidate(i + 1, f"TOP{i + 1}", score=100.0 - i, sector="CROWDED")
            for i in range(_MAX_PER_SECTOR + 1)
        ]
        alts = [
            _make_candidate(100 + i, f"ALT{i + 1}", score=50.0 - i, sector=f"SECTOR_{i}")
            for i in range(_PICKS_PER_PORTFOLIO)
        ]
        picks = _select(crowded + alts)
        tickers = [p["ticker"] for p in picks]

        overflow_ticker = crowded[-1]["ticker"]
        for c in crowded[:-1]:
            assert c["ticker"] in tickers
        assert overflow_ticker not in tickers       # over-limit in CROWDED
        assert alts[0]["ticker"] in tickers         # fills slot instead
        assert len(picks) == _PICKS_PER_PORTFOLIO


# ── Test 3: Max-1-bank constraint ────────────────────────────────────────────

class TestBankConstraint:
    def test_max_1_bank(self):
        """Only one bank may appear in any portfolio."""
        candidates = [
            _make_candidate(1, "BANK1", score=100.0, sector="BANKING", company_type="BANK"),
            _make_candidate(2, "BANK2", score=99.0, sector="BANKING2", company_type="BANK"),
            _make_candidate(3, "BANK3", score=98.0, sector="BANKING3", company_type="BANK"),
            _make_candidate(4, "OP1", score=70.0, sector="RETAIL"),
            _make_candidate(5, "OP2", score=69.0, sector="ENERGY"),
            _make_candidate(6, "OP3", score=68.0, sector="TELECOM"),
            _make_candidate(7, "OP4", score=67.0, sector="FOOD"),
        ]
        picks = _select(candidates)
        bank_picks = [p for p in picks if any(
            c["company_type"] == "BANK" and c["company_id"] == p["company_id"]
            for c in candidates
        )]
        assert len(bank_picks) <= _MAX_BANKS

    def test_overflow_bank_replaced_by_operating_company(self):
        """Banks above _MAX_BANKS are skipped; operating companies fill the slots."""
        banks = [
            _make_candidate(
                i + 1, f"BANK{i + 1}", score=100.0 - i,
                sector=f"BANKING{i + 1}", company_type="BANK",
            )
            for i in range(_MAX_BANKS + 1)
        ]
        ops = [
            _make_candidate(100 + i, f"OP{i + 1}", score=80.0 - i, sector=f"OPS{i}")
            for i in range(_PICKS_PER_PORTFOLIO)
        ]
        picks = _select(banks + ops)
        tickers = [p["ticker"] for p in picks]

        overflow_bank = banks[-1]["ticker"]
        for b in banks[:-1]:
            assert b["ticker"] in tickers
        assert overflow_bank not in tickers
        assert ops[0]["ticker"] in tickers


# ── Test 4: Turnover penalty — incumbent retained ─────────────────────────────

class TestTurnoverPenalty:
    def test_incumbent_in_top10_is_kept_when_new_pick_not_much_better(self):
        """Incumbent in top-10 is retained when best new pick scores <=15% higher."""
        # Incumbent score = 80; best new candidate = 80 * 1.10 = 88 (not 15% above)
        incumbent_id = 99
        candidates = [
            _make_candidate(1, "NEW1", score=88.0, sector="SEC1"),   # 88/80 = 1.10 — not >15%
            _make_candidate(2, "NEW2", score=75.0, sector="SEC2"),
            _make_candidate(3, "NEW3", score=74.0, sector="SEC3"),
            _make_candidate(4, "NEW4", score=73.0, sector="SEC4"),
            _make_candidate(5, "NEW5", score=72.0, sector="SEC5"),
            _make_candidate(incumbent_id, "INCMB", score=80.0, sector="SEC6"),
        ]
        picks = _select(candidates, current_holdings=[incumbent_id])
        tickers = [p["ticker"] for p in picks]

        assert "INCMB" in tickers, "Protected incumbent should appear in picks"

    def test_incumbent_in_top10_replaced_when_new_pick_much_better(self):
        """Incumbent in top-10 is replaced when a new pick scores >15% higher."""
        # Incumbent score = 80; best new candidate = 80 * 1.20 = 96 (>15% above)
        incumbent_id = 99
        candidates = [
            _make_candidate(1, "NEW1", score=96.0, sector="SEC1"),   # 96/80 = 1.20 > 1.15
            _make_candidate(2, "NEW2", score=75.0, sector="SEC2"),
            _make_candidate(3, "NEW3", score=74.0, sector="SEC3"),
            _make_candidate(4, "NEW4", score=73.0, sector="SEC4"),
            _make_candidate(5, "NEW5", score=72.0, sector="SEC5"),
            _make_candidate(incumbent_id, "INCMB", score=80.0, sector="SEC6"),
        ]
        picks = _select(candidates, current_holdings=[incumbent_id])
        tickers = [p["ticker"] for p in picks]

        # INCMB may or may not appear (it is still eligible),
        # but the key invariant is that target-count picks are selected and NEW1 (highest)
        # must be there since the incumbent is NOT protected.
        assert "NEW1" in tickers
        assert len(picks) == _PICKS_PER_PORTFOLIO

    def test_incumbent_outside_top10_not_protected(self):
        """Incumbent ranked below 10 gets no turnover protection."""
        incumbent_id = 99
        # Incumbent score = 30, ranked 11th — should not receive protection
        candidates = [
            _make_candidate(i, f"STK{i:02d}", score=float(100 - i), sector=f"SEC{i}")
            for i in range(1, 11)
        ] + [
            _make_candidate(incumbent_id, "INCMB", score=30.0, sector="SEC99"),
        ]
        picks = _select(candidates, current_holdings=[incumbent_id])
        tickers = [p["ticker"] for p in picks]

        # Top 10 best candidates (all unique sectors) fill the target-count slots
        assert "INCMB" not in tickers
        assert len(picks) == _PICKS_PER_PORTFOLIO

    def test_no_current_holdings_no_turnover_effect(self):
        """When current_holdings is None, standard score-order selection applies."""
        candidates = [
            _make_candidate(i, f"STK{i:02d}", score=float(100 - i), sector=f"SEC{i}")
            for i in range(1, 21)
        ]
        picks = _select(candidates, current_holdings=None)
        tickers = [p["ticker"] for p in picks]
        expected = [f"STK{i:02d}" for i in range(1, _PICKS_PER_PORTFOLIO + 1)]
        assert tickers == expected

    def test_multiple_incumbents_both_protected_when_eligible(self):
        """Two incumbents in top-10 are both kept when no new pick beats them by 15%."""
        inc1, inc2 = 91, 92
        # Both incumbents score 85 and 83; best new = 88 (88/83 = 1.06, <1.15)
        candidates = [
            _make_candidate(1, "NEW1", score=88.0, sector="SEC1"),
            _make_candidate(2, "NEW2", score=70.0, sector="SEC2"),
            _make_candidate(3, "NEW3", score=69.0, sector="SEC3"),
            _make_candidate(inc1, "INC1", score=85.0, sector="SEC4"),
            _make_candidate(inc2, "INC2", score=83.0, sector="SEC5"),
        ]
        picks = _select(candidates, current_holdings=[inc1, inc2])
        tickers = [p["ticker"] for p in picks]

        assert "INC1" in tickers
        assert "INC2" in tickers


# ── Test 5: Combined constraints ──────────────────────────────────────────────

class TestCombinedConstraints:
    def test_sector_and_bank_constraints_together(self):
        """Sector and bank constraints are both enforced simultaneously."""
        candidates = [
            # Two top banks — only one should be selected
            _make_candidate(1, "BANK1", score=100.0, sector="BANKING", company_type="BANK"),
            _make_candidate(2, "BANK2", score=99.0, sector="BANKING2", company_type="BANK"),
            # Three in same sector — max 2 allowed
            _make_candidate(3, "CROWD1", score=95.0, sector="CROWDED"),
            _make_candidate(4, "CROWD2", score=94.0, sector="CROWDED"),
            _make_candidate(5, "CROWD3", score=93.0, sector="CROWDED"),
            # Fill candidates
            _make_candidate(6, "OP1", score=60.0, sector="RETAIL"),
            _make_candidate(7, "OP2", score=59.0, sector="ENERGY"),
            _make_candidate(8, "OP3", score=58.0, sector="TELECOM"),
        ]
        picks = _select(candidates)

        bank_count = sum(
            1 for p in picks
            if any(
                c["company_type"] == "BANK" and c["company_id"] == p["company_id"]
                for c in candidates
            )
        )
        sector_counts: dict[str, int] = {}
        for p in picks:
            sec = next(
                c["sector_custom"] for c in candidates if c["company_id"] == p["company_id"]
            )
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        assert bank_count <= _MAX_BANKS
        assert all(v <= _MAX_PER_SECTOR for v in sector_counts.values())
        assert len(picks) == _PICKS_PER_PORTFOLIO

    def test_empty_universe_returns_empty_list(self):
        """Empty universe yields empty pick list, not an error."""
        sel = _make_selector()
        session = _mock_session()
        sel._universe.get_universe = MagicMock(return_value=[])
        sel._fetch_candidates = MagicMock(return_value=[])
        sel._get_latest_price = MagicMock(return_value=100.0)

        picks = sel.select("ALPHA", session)
        assert picks == []

    def test_invalid_portfolio_raises(self):
        """Unknown portfolio name raises ValueError."""
        sel = _make_selector()
        with pytest.raises(ValueError, match="GAMMA"):
            sel.select("GAMMA", _mock_session())


# ── Sprint 1 §3.5 (2026-05-07): falling-knife uses raw + percentile ─────────


class TestFallingKnifeFilter:
    """Verify the audit MEDIUM #11 fix: combined percentile + 200MA filter."""

    def test_low_percentile_AND_below_200ma_is_rejected(self):
        """A genuine falling knife (low tech score AND below 200MA) is dropped."""
        candidates = [
            _make_candidate(1, "GOOD", 95.0, sector="A", technical_score=80.0, above_200ma=True),
            _make_candidate(2, "BAD",  90.0, sector="B", technical_score=20.0, above_200ma=False),
            _make_candidate(3, "OK1",  85.0, sector="C", technical_score=70.0, above_200ma=True),
            _make_candidate(4, "OK2",  80.0, sector="D", technical_score=65.0, above_200ma=True),
            _make_candidate(5, "OK3",  75.0, sector="E", technical_score=60.0, above_200ma=True),
            _make_candidate(6, "OK4",  70.0, sector="F", technical_score=55.0, above_200ma=True),
        ]
        picks = _select(candidates)
        tickers = [p["ticker"] for p in picks]
        assert "BAD" not in tickers, "Falling knife (low tech AND below 200MA) must be rejected"
        assert "GOOD" in tickers
        assert len(picks) == 5

    def test_low_percentile_BUT_above_200ma_is_kept(self):
        """Bull-market case: percentile low because cross-section is hot, but
        the stock is still above its 200MA → don't drop it."""
        candidates = [
            _make_candidate(1, "TOP",  95.0, sector="A", technical_score=80.0, above_200ma=True),
            # tech_score 20 (low percentile) but above_200ma=True (uptrend) → KEEP.
            _make_candidate(2, "DIP",  92.0, sector="B", technical_score=20.0, above_200ma=True),
            _make_candidate(3, "X1",   80.0, sector="C", technical_score=70.0, above_200ma=True),
            _make_candidate(4, "X2",   75.0, sector="D", technical_score=70.0, above_200ma=True),
            _make_candidate(5, "X3",   70.0, sector="E", technical_score=70.0, above_200ma=True),
        ]
        picks = _select(candidates)
        tickers = [p["ticker"] for p in picks]
        assert "DIP" in tickers, (
            "A stock with low percentile but above its 200MA should not be "
            "filtered as a falling knife (audit MEDIUM #11)."
        )

    def test_below_200ma_BUT_high_percentile_is_kept(self):
        """Bear-market case: tech percentile high (everyone's worse) but the
        stock is below 200MA. Without the AND, we'd drop too much; with the
        AND, the percentile saves it."""
        candidates = [
            _make_candidate(1, "STRONG", 95.0, sector="A", technical_score=80.0, above_200ma=False),
            _make_candidate(2, "X1",     85.0, sector="B", technical_score=70.0, above_200ma=True),
            _make_candidate(3, "X2",     80.0, sector="C", technical_score=70.0, above_200ma=True),
            _make_candidate(4, "X3",     75.0, sector="D", technical_score=70.0, above_200ma=True),
            _make_candidate(5, "X4",     70.0, sector="E", technical_score=70.0, above_200ma=True),
        ]
        picks = _select(candidates)
        tickers = [p["ticker"] for p in picks]
        assert "STRONG" in tickers, (
            "A stock with high tech percentile shouldn't be dropped just for "
            "being below 200MA — the AND combination is intentional."
        )

    def test_legacy_no_above_200ma_data_falls_back_to_percentile(self):
        """When the DB row predates Sprint 1 §3.5, above_200ma is None.
        Backward-compat: fall back to the original percentile-only filter."""
        candidates = [
            _make_candidate(1, "GOOD", 95.0, sector="A", technical_score=80.0, above_200ma=None),
            _make_candidate(2, "BAD",  90.0, sector="B", technical_score=20.0, above_200ma=None),
            _make_candidate(3, "OK1",  85.0, sector="C", technical_score=70.0, above_200ma=None),
            _make_candidate(4, "OK2",  80.0, sector="D", technical_score=65.0, above_200ma=None),
            _make_candidate(5, "OK3",  75.0, sector="E", technical_score=60.0, above_200ma=None),
            _make_candidate(6, "OK4",  70.0, sector="F", technical_score=55.0, above_200ma=None),
        ]
        picks = _select(candidates)
        tickers = [p["ticker"] for p in picks]
        assert "BAD" not in tickers, (
            "Legacy rows (no above_200ma) should still apply the percentile-"
            "only fallback so existing DBs keep behaving."
        )
