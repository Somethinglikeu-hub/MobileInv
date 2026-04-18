"""Macro staleness check for values that require periodic manual review.

Some macro inputs (notably Turkey Equity Risk Premium) do not have a reliable
scraper-less API. They live in ``bist_picker/config/macro.yaml`` with a
``last_updated`` date. This module exposes:

- :func:`check_macro_staleness`: returns a structured report for any stale
  fields, usable from both the CLI (:mod:`bist_picker.cli`) and the GitHub
  Actions workflow (opens an issue when stale).
- ``python -m bist_picker.macro_check``: prints a human-readable status line
  and exits non-zero when any field is stale — GitHub Actions uses the exit
  code to decide whether to open the issue.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml

_DEFAULT_MACRO_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "macro.yaml"


@dataclass
class MacroStaleField:
    """One stale field in macro.yaml — the pieces an issue body needs."""

    field: str
    last_updated: str
    age_days: int
    stale_after_days: int
    source_url: Optional[str]
    current_value: Optional[float]
    update_instructions: str


@dataclass
class MacroStalenessReport:
    """Full macro check result. Serialisable to JSON for the workflow."""

    checked_at: str
    is_stale: bool
    stale_fields: list[MacroStaleField]

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, ensure_ascii=False)


def check_macro_staleness(
    macro_config_path: Optional[Path] = None,
    today: Optional[date] = None,
) -> MacroStalenessReport:
    """Check macro.yaml fields against their ``last_updated`` / ``stale_after_days``.

    Currently only checks ``erp.equity_risk_premium_try`` (the one field
    that genuinely requires manual refresh). If more manually-curated
    fields are added to macro.yaml, extend the ``_CHECKS`` table below.
    """
    path = macro_config_path or _DEFAULT_MACRO_CONFIG_PATH
    today = today or date.today()

    if not path.exists():
        return MacroStalenessReport(
            checked_at=today.isoformat(),
            is_stale=False,
            stale_fields=[],
        )

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    stale: list[MacroStaleField] = []

    erp_block = cfg.get("erp", {}) or {}
    last_updated_str = erp_block.get("last_updated")
    stale_after = int(erp_block.get("stale_after_days", 90))
    if last_updated_str:
        try:
            last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d").date()
        except ValueError:
            last_updated = None
        if last_updated is not None:
            age = (today - last_updated).days
            if age > stale_after:
                stale.append(
                    MacroStaleField(
                        field="erp.equity_risk_premium_try",
                        last_updated=last_updated_str,
                        age_days=age,
                        stale_after_days=stale_after,
                        source_url=erp_block.get("source_url"),
                        current_value=erp_block.get("equity_risk_premium_try"),
                        update_instructions=(
                            "1. Open the Damodaran Country Risk page (source_url).\n"
                            "2. Find the Turkey row → copy the Equity Risk Premium.\n"
                            "3. Divide by 100 if the source shows a percent (9.52 → 0.0952).\n"
                            "4. Edit bist_picker/config/macro.yaml: update "
                            "`equity_risk_premium_try` and bump `last_updated` to today.\n"
                            "5. Commit + push; this issue will close automatically on the "
                            "next pipeline run."
                        ),
                    )
                )

    return MacroStalenessReport(
        checked_at=today.isoformat(),
        is_stale=bool(stale),
        stale_fields=stale,
    )


def _format_human(report: MacroStalenessReport) -> str:
    if not report.is_stale:
        return f"Macro config OK (checked {report.checked_at})."
    lines = [
        f"Macro config has {len(report.stale_fields)} stale field(s) "
        f"(checked {report.checked_at}):",
    ]
    for f in report.stale_fields:
        lines.append(
            f"  - {f.field}: updated {f.last_updated} "
            f"({f.age_days}d ago, threshold {f.stale_after_days}d); "
            f"current value = {f.current_value}"
        )
    return "\n".join(lines)


def _main() -> int:
    """CLI entry: prints report and exits 1 when stale, 0 when fresh.

    ``--json`` emits the structured report (used by GitHub Actions to build
    the issue body).
    """
    as_json = "--json" in sys.argv[1:]
    report = check_macro_staleness()
    if as_json:
        print(report.to_json())
    else:
        print(_format_human(report))
    return 1 if report.is_stale else 0


if __name__ == "__main__":
    raise SystemExit(_main())
