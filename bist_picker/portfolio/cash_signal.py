"""Phase 4 — Cash-out signal state machine.

Blends Market Regime (XU100 price + volatility) and Macro Regime (Turkey CDS
+ real rates) into a 4-level sticky state machine that tells the portfolio
how much of the book to hold in cash.

States
------
NORMAL     -> 0%  cash (full risk)
CAUTION    -> 25% cash (trim exposure, still primarily long)
DEFENSIVE  -> 50% cash (half book in cash)
RISK_OFF   -> 75% cash (mostly cash; keep only highest-conviction picks)

Design notes
------------
* The raw signal is just the sum of two small integer scores, so the whole
  classifier is trivially testable without any fixtures.
* All stickiness lives in :func:`CashSignalCalculator._decide_state` so it
  can be exercised with synthetic history in unit tests.
* The kill-switch in ``cash_signal.yaml`` degrades gracefully: we still log
  the signal and target state, but force ``state=NORMAL`` so selector weights
  behave exactly as before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from bist_picker.db.schema import CashAllocationState
from bist_picker.portfolio.macro_overlay import MacroRegimeClassifier
from bist_picker.portfolio.regime_classifier import MarketRegimeClassifier

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "cash_signal.yaml"
)

# Ordered from lowest stress (NORMAL) to highest (RISK_OFF). The state
# machine only ever steps one rung at a time and the stress level is the
# index into this list.
_STATE_ORDER: tuple[str, ...] = ("NORMAL", "CAUTION", "DEFENSIVE", "RISK_OFF")

# Default cash percentages for each state (overridden by cash_signal.yaml).
_DEFAULT_CASH_PCT: dict[str, float] = {
    "NORMAL": 0.0,
    "CAUTION": 0.25,
    "DEFENSIVE": 0.50,
    "RISK_OFF": 0.75,
}

# Score table for the raw input signal.
_MARKET_SCORES: dict[str, int] = {
    "BULL_LOW_VOL": 0,
    "BULL_HIGH_VOL": 1,
    "BEAR": 2,
}
_MACRO_SCORES: dict[str, int] = {
    "RISK_ON": 0,
    "NEUTRAL": 1,
    "RISK_OFF": 2,
}


def _stress_level(state: str) -> int:
    try:
        return _STATE_ORDER.index(state)
    except ValueError as exc:
        raise ValueError(f"Unknown cash state {state!r}") from exc


def _state_for_stress(stress: int) -> str:
    clamped = max(0, min(stress, len(_STATE_ORDER) - 1))
    return _STATE_ORDER[clamped]


def raw_signal_from_regimes(market_regime: str, macro_regime: str) -> int:
    """Map a (market, macro) regime pair to a 0..4 raw stress signal."""
    market = _MARKET_SCORES.get(market_regime, 1)   # unknown -> mid
    macro = _MACRO_SCORES.get(macro_regime, 1)
    return market + macro


def target_state_from_raw(raw: int) -> str:
    """Map a raw 0..4 signal to its 'desired' state (before hysteresis)."""
    if raw <= 1:
        return "NORMAL"
    if raw == 2:
        return "CAUTION"
    if raw == 3:
        return "DEFENSIVE"
    return "RISK_OFF"


@dataclass(frozen=True)
class CashSignalConfig:
    """Typed view over cash_signal.yaml."""

    enabled: bool
    cash_pct: dict[str, float]
    up_confirmation_days: int
    down_confirmation_days: int
    min_holding_days: int
    max_step_per_transition: int
    open_issue_on_change: bool

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CashSignalConfig":
        path = path or _DEFAULT_CONFIG_PATH
        data: dict = {}
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("cash_signal.yaml unreadable (%s); using defaults", exc)

        states = data.get("states") or {}
        cash_pct = {
            name: float((states.get(name) or {}).get("cash_pct", default))
            for name, default in _DEFAULT_CASH_PCT.items()
        }
        hyst = data.get("hysteresis") or {}
        notif = data.get("notifications") or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            cash_pct=cash_pct,
            up_confirmation_days=int(hyst.get("up_confirmation_days", 5)),
            down_confirmation_days=int(hyst.get("down_confirmation_days", 10)),
            min_holding_days=int(hyst.get("min_holding_days", 20)),
            max_step_per_transition=int(hyst.get("max_step_per_transition", 1)),
            open_issue_on_change=bool(notif.get("open_issue_on_change", True)),
        )


@dataclass(frozen=True)
class CashSignalResult:
    """Outcome of a single day's evaluation."""

    date: date
    market_regime: str
    macro_regime: str
    raw_signal: int
    target_state: str
    state: str
    cash_pct: float
    days_in_state: int
    last_transition_date: Optional[date]
    transitioned_today: bool
    notes: str


class CashSignalCalculator:
    """Computes and persists the daily cash-out state.

    The class is intentionally stateless between calls — each ``compute`` call
    reads the prior persisted row(s) from ``cash_allocation_state`` and
    decides the current day based on that history plus today's regime inputs.
    """

    def __init__(self, config: Optional[CashSignalConfig] = None) -> None:
        self._cfg = config or CashSignalConfig.load()

    # ── Public API ────────────────────────────────────────────────────────

    def compute(
        self,
        session: Session,
        scoring_date: date,
        *,
        market_regime: Optional[str] = None,
        macro_regime: Optional[str] = None,
        persist: bool = True,
    ) -> CashSignalResult:
        """Evaluate the cash state for *scoring_date*.

        ``market_regime`` / ``macro_regime`` are accepted as overrides so tests
        can feed synthetic signals without populating the full price table.
        In production they default to the live classifiers.
        """
        if market_regime is None:
            market_regime = MarketRegimeClassifier(session).classify(scoring_date)
        if macro_regime is None:
            macro_regime = MacroRegimeClassifier(session).classify(scoring_date)

        raw = raw_signal_from_regimes(market_regime, macro_regime)
        target = target_state_from_raw(raw)

        prev = self._latest_prior(session, scoring_date)
        history = self._recent_signals(
            session,
            scoring_date,
            max(self._cfg.up_confirmation_days, self._cfg.down_confirmation_days),
        )

        decision = self._decide_state(
            today_raw=raw,
            prev=prev,
            history=history,
            today=scoring_date,
        )

        if not self._cfg.enabled:
            # Kill-switch: emit the signal but force NORMAL so selector is
            # untouched. Still persist so the UI can show "disabled but signal
            # would be X".
            decision = _Decision(
                state="NORMAL",
                transitioned_today=(prev is not None and prev.state != "NORMAL"),
                days_in_state=(prev.days_in_state + 1) if prev and prev.state == "NORMAL" else 1,
                last_transition_date=(
                    scoring_date
                    if (prev is not None and prev.state != "NORMAL")
                    else (prev.last_transition_date if prev else None)
                ),
                notes="kill-switch disabled; forcing NORMAL",
            )

        cash_pct = self._cfg.cash_pct[decision.state]

        result = CashSignalResult(
            date=scoring_date,
            market_regime=market_regime,
            macro_regime=macro_regime,
            raw_signal=raw,
            target_state=target,
            state=decision.state,
            cash_pct=cash_pct,
            days_in_state=decision.days_in_state,
            last_transition_date=decision.last_transition_date,
            transitioned_today=decision.transitioned_today,
            notes=decision.notes,
        )

        if persist:
            self._persist(session, result)
        return result

    # ── Internals ─────────────────────────────────────────────────────────

    def _latest_prior(
        self, session: Session, today: date
    ) -> Optional[CashAllocationState]:
        return (
            session.query(CashAllocationState)
            .filter(CashAllocationState.date < today)
            .order_by(CashAllocationState.date.desc())
            .first()
        )

    def _recent_signals(
        self, session: Session, today: date, lookback_days: int
    ) -> list[CashAllocationState]:
        """Return the last N persisted rows strictly before today, ascending."""
        if lookback_days <= 0:
            return []
        rows = (
            session.query(CashAllocationState)
            .filter(CashAllocationState.date < today)
            .order_by(CashAllocationState.date.desc())
            .limit(lookback_days)
            .all()
        )
        return list(reversed(rows))  # oldest first

    def _decide_state(
        self,
        *,
        today_raw: int,
        prev: Optional[CashAllocationState],
        history: list[CashAllocationState],
        today: date,
    ) -> "_Decision":
        # First-ever run: start from the target state directly. This is the
        # only time we jump multiple rungs; afterwards the hysteresis rules
        # keep transitions single-step.
        if prev is None:
            target = target_state_from_raw(today_raw)
            return _Decision(
                state=target,
                transitioned_today=target != "NORMAL",
                days_in_state=1,
                last_transition_date=today if target != "NORMAL" else None,
                notes="initial state",
            )

        current = prev.state
        current_stress = _stress_level(current)
        today_target_stress = _stress_level(target_state_from_raw(today_raw))

        # Cooldown — min_holding_days since last transition.
        last_tx = prev.last_transition_date
        if last_tx is not None:
            days_since_tx = (today - last_tx).days
            if days_since_tx < self._cfg.min_holding_days:
                return _Decision(
                    state=current,
                    transitioned_today=False,
                    days_in_state=prev.days_in_state + 1,
                    last_transition_date=last_tx,
                    notes=(
                        f"cooldown: {days_since_tx}d since last transition "
                        f"(min {self._cfg.min_holding_days}d)"
                    ),
                )

        # Determine direction.
        if today_target_stress > current_stress:
            window = self._cfg.up_confirmation_days
            needed_stress = current_stress + 1
            # Every day of the window (including today) must be AT OR ABOVE
            # the next higher stress level.
            tail = history[-(window - 1):] if window > 1 else []
            enough_history = len(history) >= window - 1
            tail_confirms = all(
                _stress_level(target_state_from_raw(row.raw_signal)) >= needed_stress
                for row in tail
            )
            if (
                today_target_stress >= needed_stress
                and enough_history
                and tail_confirms
            ):
                new_state = _state_for_stress(
                    current_stress + self._cfg.max_step_per_transition
                )
                return _Decision(
                    state=new_state,
                    transitioned_today=True,
                    days_in_state=1,
                    last_transition_date=today,
                    notes=f"upgraded after {window}d stress confirmation",
                )
            return _Decision(
                state=current,
                transitioned_today=False,
                days_in_state=prev.days_in_state + 1,
                last_transition_date=last_tx,
                notes=f"pending up-confirmation (need {window}d)",
            )

        if today_target_stress < current_stress:
            window = self._cfg.down_confirmation_days
            needed_stress = current_stress - 1
            tail = history[-(window - 1):] if window > 1 else []
            enough_history = len(history) >= window - 1
            tail_confirms = all(
                _stress_level(target_state_from_raw(row.raw_signal)) <= needed_stress
                for row in tail
            )
            if (
                today_target_stress <= needed_stress
                and enough_history
                and tail_confirms
            ):
                new_state = _state_for_stress(
                    current_stress - self._cfg.max_step_per_transition
                )
                return _Decision(
                    state=new_state,
                    transitioned_today=True,
                    days_in_state=1,
                    last_transition_date=today,
                    notes=f"downgraded after {window}d calm confirmation",
                )
            return _Decision(
                state=current,
                transitioned_today=False,
                days_in_state=prev.days_in_state + 1,
                last_transition_date=last_tx,
                notes=f"pending down-confirmation (need {window}d)",
            )

        # Target matches current state — stay put.
        return _Decision(
            state=current,
            transitioned_today=False,
            days_in_state=prev.days_in_state + 1,
            last_transition_date=last_tx,
            notes="steady",
        )

    def _persist(self, session: Session, result: CashSignalResult) -> None:
        existing = (
            session.query(CashAllocationState)
            .filter(CashAllocationState.date == result.date)
            .first()
        )
        if existing:
            existing.market_regime = result.market_regime
            existing.macro_regime = result.macro_regime
            existing.raw_signal = result.raw_signal
            existing.target_state = result.target_state
            existing.state = result.state
            existing.cash_pct = result.cash_pct
            existing.days_in_state = result.days_in_state
            existing.last_transition_date = result.last_transition_date
            existing.transitioned_today = result.transitioned_today
            existing.notes = result.notes
        else:
            session.add(
                CashAllocationState(
                    date=result.date,
                    market_regime=result.market_regime,
                    macro_regime=result.macro_regime,
                    raw_signal=result.raw_signal,
                    target_state=result.target_state,
                    state=result.state,
                    cash_pct=result.cash_pct,
                    days_in_state=result.days_in_state,
                    last_transition_date=result.last_transition_date,
                    transitioned_today=result.transitioned_today,
                    notes=result.notes,
                )
            )
        session.flush()


@dataclass(frozen=True)
class _Decision:
    state: str
    transitioned_today: bool
    days_in_state: int
    last_transition_date: Optional[date]
    notes: str


def next_possible_transition_date(
    prev: CashAllocationState, cfg: Optional[CashSignalConfig] = None
) -> Optional[date]:
    """Given the latest persisted row, return the earliest date the cooldown
    would allow the next transition (for UI display only)."""
    cfg = cfg or CashSignalConfig.load()
    if prev.last_transition_date is None:
        return None
    return prev.last_transition_date + timedelta(days=cfg.min_holding_days)
