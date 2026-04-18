"""Tests for the macro-staleness check used by the CLI and GitHub Actions."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from bist_picker.macro_check import check_macro_staleness


def _write_macro(tmp_path: Path, last_updated: str, stale_after: int = 90) -> Path:
    path = tmp_path / "macro.yaml"
    payload = {
        "erp": {
            "equity_risk_premium_try": 0.0952,
            "last_updated": last_updated,
            "source_url": "https://example.test/erp",
            "stale_after_days": stale_after,
        }
    }
    path.write_text(yaml.safe_dump(payload))
    return path


def test_fresh_erp_is_not_stale(tmp_path):
    path = _write_macro(tmp_path, last_updated="2025-07-01")
    report = check_macro_staleness(macro_config_path=path, today=date(2025, 7, 15))

    assert report.is_stale is False
    assert report.stale_fields == []


def test_stale_erp_flagged_with_instructions(tmp_path):
    path = _write_macro(tmp_path, last_updated="2024-01-01", stale_after=90)
    report = check_macro_staleness(macro_config_path=path, today=date(2025, 7, 15))

    assert report.is_stale is True
    assert len(report.stale_fields) == 1
    field = report.stale_fields[0]
    assert field.field == "erp.equity_risk_premium_try"
    assert field.age_days > 90
    assert "Damodaran" in field.update_instructions
    assert field.source_url == "https://example.test/erp"


def test_missing_macro_yaml_is_not_stale(tmp_path):
    """No macro.yaml is a valid state (e.g., early bootstrap) — don't cry wolf."""
    report = check_macro_staleness(
        macro_config_path=tmp_path / "does_not_exist.yaml",
        today=date(2025, 7, 15),
    )
    assert report.is_stale is False


def test_to_json_round_trip(tmp_path):
    path = _write_macro(tmp_path, last_updated="2024-01-01")
    report = check_macro_staleness(macro_config_path=path, today=date(2025, 7, 15))

    import json
    payload = json.loads(report.to_json())
    assert payload["is_stale"] is True
    assert payload["stale_fields"][0]["field"] == "erp.equity_risk_premium_try"
