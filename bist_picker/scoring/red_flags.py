"""Phase 5: red-flag detection for the transparency surface.

Deliberately minimal MVP: only flags that can be derived from fields already
present on ``ScoringResult`` — no new factor modules, no new fetches, no
multi-period financial analysis. Those heavier signals (Altman-Z, debt trend,
FCF deterioration) belong to a later phase that explicitly pulls historical
financials for red-flag purposes.

The output is a JSON-serialised list of short string codes that the APK can
map to human-readable labels and colours. We use codes (not prose) so the
wire format stays stable as translations / wording evolve.

Flag codes
----------
- ``PIOTROSKI_LOW``     — raw F-score < 4 (weak fundamentals)
- ``LIMITED_DATA``      — data completeness below ``min_completeness``
- ``DCF_OVERVALUED``    — DCF margin of safety < 0 (price above intrinsic)
- ``WEAK_TECHNICAL``    — normalized technical score < 40

Each entry in the list is one of these codes. An empty list means "no
issues detected from the data we have" (not the same as "safe" — this is
explicitly MVP coverage).
"""

from __future__ import annotations

import json
from typing import Any, Iterable

# Thresholds — kept as module-level constants so tests can import + patch if
# we ever want to tune without a config round-trip. These are deliberately
# conservative: if you're hitting these, something is worth a second look.
PIOTROSKI_LOW_THRESHOLD: int = 4           # raw 0-9 F-score, <4 is weak
DATA_COMPLETENESS_THRESHOLD: float = 60.0  # percent
DCF_OVERVALUED_THRESHOLD: float = 0.0      # MoS% below this -> overvalued
WEAK_TECHNICAL_THRESHOLD: float = 40.0     # 0-100 normalized


def detect_flags(row: dict[str, Any]) -> list[str]:
    """Return the list of red-flag codes applicable to ``row``.

    ``row`` is a plain dict-shaped view of a ScoringResult (or any object
    exposing the same keys). Missing keys are treated as "unknown" and do
    not fire a flag — the MVP never flags based on absence alone, except
    through the LIMITED_DATA check which uses ``data_completeness`` itself.

    Ordering is stable for reproducibility: flags come out in declaration
    order so the serialised JSON doesn't churn spuriously day to day.
    """
    flags: list[str] = []

    piotroski_raw = row.get("piotroski_fscore_raw")
    if piotroski_raw is not None and piotroski_raw < PIOTROSKI_LOW_THRESHOLD:
        flags.append("PIOTROSKI_LOW")

    completeness = row.get("data_completeness")
    if completeness is not None and completeness < DATA_COMPLETENESS_THRESHOLD:
        flags.append("LIMITED_DATA")

    mos = row.get("dcf_margin_of_safety_pct")
    if mos is not None and mos < DCF_OVERVALUED_THRESHOLD:
        flags.append("DCF_OVERVALUED")

    tech = row.get("technical_score")
    if tech is not None and tech < WEAK_TECHNICAL_THRESHOLD:
        flags.append("WEAK_TECHNICAL")

    return flags


def serialize_flags(flags: Iterable[str]) -> str | None:
    """Serialise a flags list to JSON, or return None when empty.

    Storing ``None`` instead of ``"[]"`` keeps the column semantically
    clean: "no flags computed yet" and "no flags fired" look different in
    the DB, and the APK can treat NULL as "not yet available".
    """
    flags_list = list(flags)
    if not flags_list:
        return None
    return json.dumps(flags_list, separators=(",", ":"))


def deserialize_flags(payload: str | None) -> list[str]:
    """Parse the persisted JSON payload back into a list.

    Defensive: any malformed value -> empty list, so the read path never
    crashes the snapshot / API due to corrupt history.
    """
    if not payload:
        return []
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, (str, int, float))]
