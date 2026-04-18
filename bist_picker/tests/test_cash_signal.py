"""Phase 4 — regression tests for the cash-out signal state machine.

The state machine is where whipsaw bugs hide, so these tests exhaustively
exercise:
  * raw-signal mapping from (market, macro) regime pairs
  * confirmation windows (up vs down) are asymmetric and enforced
  * the cooldown (``min_holding_days``) blocks transitions even when the
    confirmation window is satisfied
  * ``max_step_per_transition=1`` forces the machine to walk through
    intermediate states (NORMAL -> CAUTION -> DEFENSIVE) on sudden spikes
  * the kill-switch degrades to NORMAL without touching persistence
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.db.schema import Base, CashAllocationState
from bist_picker.portfolio.cash_signal import (
    CashSignalCalculator,
    CashSignalConfig,
    raw_signal_from_regimes,
    target_state_from_raw,
)


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
def cfg() -> CashSignalConfig:
    """Small hysteresis windows so tests stay readable."""
    return CashSignalConfig(
        enabled=True,
        cash_pct={"NORMAL": 0.0, "CAUTION": 0.25, "DEFENSIVE": 0.50, "RISK_OFF": 0.75},
        up_confirmation_days=3,
        down_confirmation_days=5,
        min_holding_days=10,
        max_step_per_transition=1,
        open_issue_on_change=True,
    )


def _run(calc: CashSignalCalculator, session, d: date, market: str, macro: str):
    return calc.compute(
        session, d, market_regime=market, macro_regime=macro
    )


# ── Raw signal tables ────────────────────────────────────────────────────────


def test_raw_signal_pairs():
    # Most benign pair
    assert raw_signal_from_regimes("BULL_LOW_VOL", "RISK_ON") == 0
    # Middle-of-the-road
    assert raw_signal_from_regimes("BULL_HIGH_VOL", "NEUTRAL") == 2
    # Worst case
    assert raw_signal_from_regimes("BEAR", "RISK_OFF") == 4


def test_target_state_mapping():
    assert target_state_from_raw(0) == "NORMAL"
    assert target_state_from_raw(1) == "NORMAL"       # stress 0 envelope
    assert target_state_from_raw(2) == "CAUTION"
    assert target_state_from_raw(3) == "DEFENSIVE"
    assert target_state_from_raw(4) == "RISK_OFF"


# ── Initial / steady behaviour ───────────────────────────────────────────────


def test_first_day_jumps_directly_to_target(session, cfg):
    calc = CashSignalCalculator(config=cfg)
    r = _run(calc, session, date(2026, 5, 1), "BEAR", "RISK_OFF")
    # Bootstrapping is the ONLY time we allow multi-rung jumps.
    assert r.state == "RISK_OFF"
    assert r.cash_pct == pytest.approx(0.75)
    assert r.transitioned_today is True


def test_steady_benign_signal_stays_normal(session, cfg):
    calc = CashSignalCalculator(config=cfg)
    start = date(2026, 5, 1)
    for i in range(5):
        r = _run(calc, session, start + timedelta(days=i), "BULL_LOW_VOL", "RISK_ON")
    assert r.state == "NORMAL"
    assert r.days_in_state == 5


# ── Up-confirmation (reacting to stress) ─────────────────────────────────────


def test_single_stress_day_does_not_move_up(session, cfg):
    """A lone day of stress must not trigger a transition."""
    calc = CashSignalCalculator(config=cfg)
    start = date(2026, 5, 1)
    # 4 days of calm, then one bad day
    for i in range(4):
        _run(calc, session, start + timedelta(days=i), "BULL_LOW_VOL", "RISK_ON")
    r = _run(calc, session, start + timedelta(days=4), "BULL_HIGH_VOL", "NEUTRAL")
    assert r.state == "NORMAL"
    assert r.transitioned_today is False
    assert "pending up-confirmation" in r.notes


def test_full_up_confirmation_window_triggers_transition(session, cfg):
    calc = CashSignalCalculator(config=cfg)
    start = date(2026, 5, 1)
    # Seed with 4 calm days so the machine is settled into NORMAL.
    for i in range(4):
        _run(calc, session, start + timedelta(days=i), "BULL_LOW_VOL", "RISK_ON")
    # Then up_confirmation_days (3) consecutive stress days.
    for i in range(4, 4 + cfg.up_confirmation_days - 1):
        r = _run(calc, session, start + timedelta(days=i), "BULL_HIGH_VOL", "NEUTRAL")
        assert r.state == "NORMAL", f"day {i}: transitioned too early"
    r = _run(
        calc,
        session,
        start + timedelta(days=4 + cfg.up_confirmation_days - 1),
        "BULL_HIGH_VOL",
        "NEUTRAL",
    )
    assert r.state == "CAUTION"
    assert r.transitioned_today is True


def test_step_limit_prevents_multi_rung_jump(session, cfg):
    """Even a sustained RISK_OFF (raw=4) from a NORMAL state only moves one
    step per transition event — the machine must walk through CAUTION first."""
    calc = CashSignalCalculator(config=cfg)
    start = date(2026, 5, 1)
    # Seed calm
    for i in range(4):
        _run(calc, session, start + timedelta(days=i), "BULL_LOW_VOL", "RISK_ON")
    # Slam to max stress for a full up-confirmation window
    for i in range(4, 4 + cfg.up_confirmation_days):
        r = _run(calc, session, start + timedelta(days=i), "BEAR", "RISK_OFF")
    assert r.state == "CAUTION"   # only ONE rung, not DEFENSIVE or RISK_OFF


# ── Cooldown ─────────────────────────────────────────────────────────────────


def test_cooldown_blocks_subsequent_transitions(session, cfg):
    """Once a transition fires, min_holding_days must elapse before another."""
    calc = CashSignalCalculator(config=cfg)
    start = date(2026, 5, 1)
    # Seed calm then push up to CAUTION
    for i in range(4):
        _run(calc, session, start + timedelta(days=i), "BULL_LOW_VOL", "RISK_ON")
    for i in range(4, 4 + cfg.up_confirmation_days):
        r = _run(calc, session, start + timedelta(days=i), "BEAR", "RISK_OFF")
    transition_day = start + timedelta(days=4 + cfg.up_confirmation_days - 1)
    assert r.state == "CAUTION"
    assert r.last_transition_date == transition_day

    # Now try to keep pushing further stress — cooldown should hold CAUTION.
    for i in range(1, cfg.min_holding_days):
        d = transition_day + timedelta(days=i)
        r = _run(calc, session, d, "BEAR", "RISK_OFF")
        assert r.state == "CAUTION", f"cooldown broken on day +{i}"
        assert r.transitioned_today is False
        assert "cooldown" in r.notes


def test_transition_allowed_after_cooldown(session, cfg):
    calc = CashSignalCalculator(config=cfg)
    start = date(2026, 5, 1)
    for i in range(4):
        _run(calc, session, start + timedelta(days=i), "BULL_LOW_VOL", "RISK_ON")
    for i in range(4, 4 + cfg.up_confirmation_days):
        _run(calc, session, start + timedelta(days=i), "BEAR", "RISK_OFF")
    transition_day = start + timedelta(days=4 + cfg.up_confirmation_days - 1)

    # Wait out the cooldown, keeping stress high so up-confirmation stays satisfied.
    for i in range(1, cfg.min_holding_days):
        _run(
            calc,
            session,
            transition_day + timedelta(days=i),
            "BEAR",
            "RISK_OFF",
        )

    # One more day past the cooldown should now allow another step up.
    r = _run(
        calc,
        session,
        transition_day + timedelta(days=cfg.min_holding_days),
        "BEAR",
        "RISK_OFF",
    )
    assert r.state == "DEFENSIVE"
    assert r.transitioned_today is True


# ── Down-confirmation (asymmetric re-risking) ────────────────────────────────


def test_down_confirmation_is_stricter_than_up(session, cfg):
    """After reaching CAUTION, a brief benign spell must not snap back to NORMAL."""
    calc = CashSignalCalculator(config=cfg)
    start = date(2026, 5, 1)

    # Reach CAUTION via up-confirmation (seed + 3 stress days = 7 days total).
    for i in range(4):
        _run(calc, session, start + timedelta(days=i), "BULL_LOW_VOL", "RISK_ON")
    for i in range(4, 4 + cfg.up_confirmation_days):
        _run(calc, session, start + timedelta(days=i), "BEAR", "RISK_OFF")
    transition_day = start + timedelta(days=4 + cfg.up_confirmation_days - 1)

    # Wait out the cooldown with a raw signal that matches CAUTION exactly
    # (raw=2). That way the machine neither tries to step up nor step down
    # while the cooldown is holding — a clean pause.
    for i in range(1, cfg.min_holding_days + 1):
        _run(
            calc,
            session,
            transition_day + timedelta(days=i),
            "BULL_HIGH_VOL",
            "NEUTRAL",
        )
    cooldown_end = transition_day + timedelta(days=cfg.min_holding_days)

    # Now turn benign. up_confirmation_days (3) of calm should NOT move us
    # back to NORMAL — we need down_confirmation_days (5).
    for i in range(1, cfg.up_confirmation_days + 1):
        r = _run(
            calc,
            session,
            cooldown_end + timedelta(days=i),
            "BULL_LOW_VOL",
            "RISK_ON",
        )
        assert r.state == "CAUTION", f"down-path tripped early on day +{i}"


def test_down_path_fires_after_full_window(session, cfg):
    calc = CashSignalCalculator(config=cfg)
    start = date(2026, 5, 1)
    for i in range(4):
        _run(calc, session, start + timedelta(days=i), "BULL_LOW_VOL", "RISK_ON")
    for i in range(4, 4 + cfg.up_confirmation_days):
        _run(calc, session, start + timedelta(days=i), "BEAR", "RISK_OFF")
    transition_day = start + timedelta(days=4 + cfg.up_confirmation_days - 1)
    # Hold at CAUTION's target stress (raw=2) through the cooldown — no
    # up/down pressure, just time passing.
    for i in range(1, cfg.min_holding_days + 1):
        _run(
            calc,
            session,
            transition_day + timedelta(days=i),
            "BULL_HIGH_VOL",
            "NEUTRAL",
        )
    cooldown_end = transition_day + timedelta(days=cfg.min_holding_days)

    # Feed down_confirmation_days of calm; transition must fire on the last.
    for i in range(1, cfg.down_confirmation_days):
        r = _run(
            calc,
            session,
            cooldown_end + timedelta(days=i),
            "BULL_LOW_VOL",
            "RISK_ON",
        )
        assert r.state == "CAUTION"
    r = _run(
        calc,
        session,
        cooldown_end + timedelta(days=cfg.down_confirmation_days),
        "BULL_LOW_VOL",
        "RISK_ON",
    )
    assert r.state == "NORMAL"
    assert r.transitioned_today is True


# ── Persistence ──────────────────────────────────────────────────────────────


def test_persisted_row_matches_result(session, cfg):
    calc = CashSignalCalculator(config=cfg)
    r = _run(calc, session, date(2026, 5, 1), "BEAR", "RISK_OFF")
    session.commit()
    row = session.query(CashAllocationState).filter_by(date=date(2026, 5, 1)).one()
    assert row.state == r.state
    assert row.cash_pct == pytest.approx(r.cash_pct)
    assert row.raw_signal == r.raw_signal
    assert row.target_state == r.target_state


def test_rerun_same_day_updates_in_place(session, cfg):
    calc = CashSignalCalculator(config=cfg)
    d = date(2026, 5, 1)
    _run(calc, session, d, "BEAR", "RISK_OFF")
    _run(calc, session, d, "BULL_LOW_VOL", "RISK_ON")
    rows = session.query(CashAllocationState).filter_by(date=d).all()
    assert len(rows) == 1
    # The second run rewrites the row. Because this is *still* the first
    # persisted day (there's no prior history), the calculator treats it as
    # a fresh bootstrap and picks NORMAL for the benign inputs.
    assert rows[0].state == "NORMAL"


# ── Kill-switch ──────────────────────────────────────────────────────────────


def test_kill_switch_forces_normal_but_logs_signal(session):
    killed = CashSignalConfig(
        enabled=False,
        cash_pct={"NORMAL": 0.0, "CAUTION": 0.25, "DEFENSIVE": 0.50, "RISK_OFF": 0.75},
        up_confirmation_days=3,
        down_confirmation_days=5,
        min_holding_days=10,
        max_step_per_transition=1,
        open_issue_on_change=True,
    )
    calc = CashSignalCalculator(config=killed)
    r = calc.compute(
        session,
        date(2026, 5, 1),
        market_regime="BEAR",
        macro_regime="RISK_OFF",
    )
    assert r.state == "NORMAL"
    assert r.cash_pct == 0.0
    # Raw signal is still captured so the UI can show "would be RISK_OFF"
    assert r.raw_signal == 4
    assert r.target_state == "RISK_OFF"
    assert "kill-switch" in r.notes
