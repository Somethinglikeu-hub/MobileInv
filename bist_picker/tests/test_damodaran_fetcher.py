"""Tests for the Damodaran Country Risk Premium scraper.

Pin the HTML parser against a copy of the Turkey row so we catch column-order
or wrapper-tag changes from Damodaran without needing the live site. A
separate test exercises the real network and is skipped if offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bist_picker.data.sources.damodaran import (
    DamodaranTurkeyERP,
    fetch_turkey_erp,
    parse_turkey_row,
)


# ── Fixtures: snapshot of the live HTML around the Turkey row ──────────────

# Captured 2026-05-07 from
# https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html
# Trimmed to the relevant rows; whitespace/inline styles preserved on purpose
# so the parser is exercised against realistic markup.
_HTML_FIXTURE_TURKEY = """
<table>
  <tr>
    <td>Tunisia</td>
    <td>B3</td>
    <td>5.32%</td>
    <td>8.10%</td>
    <td>14.67%</td>
  </tr>
  <tr height=21 style='height:16.0pt'>
    <td height=21 class=xl7044 style='height:16.0pt;border-top:none'>Turkey (updated February 2026)</td>
    <td class=xl7144 style='border-top:none;border-left:none'>Ba3</td>
    <td class=xl7244 style='border-top:none;border-left:none'>3.06%</td>
    <td class=xl7341 style='border-top:none;border-left:none'>4.66%</td>
    <td class=xl7244 style='border-top:none;border-left:none'>8.89%</td>
    <td class=xl7341 style='border-top:none;border-left:none'>25.00%</td>
    <td class=xl6744 style='border-top:none;border-left:none'>2.85%</td>
    <td class=xl6744 style='border-top:none;border-left:none'>8.56%</td>
  </tr>
  <tr>
    <td>Turks and Caicos Islands</td>
    <td>NR</td>
  </tr>
</table>
"""


_HTML_FIXTURE_NO_TURKEY = """
<table>
  <tr>
    <td>Albania</td>
    <td>B1</td>
    <td>4.32%</td>
    <td>6.58%</td>
    <td>11.91%</td>
  </tr>
</table>
"""


# Should not match — "Turkmenistan" must not be treated as "Turkey".
_HTML_FIXTURE_TURKMENISTAN_ONLY = """
<table>
  <tr>
    <td>Turkmenistan</td>
    <td>NR</td>
    <td>10.00%</td>
    <td>15.00%</td>
    <td>20.00%</td>
  </tr>
</table>
"""


# ── parse_turkey_row ─────────────────────────────────────────────────────


class TestParseTurkeyRow:
    def test_extracts_erp_and_metadata(self):
        result = parse_turkey_row(_HTML_FIXTURE_TURKEY)
        assert result is not None
        # ERP is the 3rd percent column on the Turkey row.
        assert result["erp"] == pytest.approx(0.0889, abs=1e-6)
        # CRP is the 2nd percent column.
        assert result["crp"] == pytest.approx(0.0466, abs=1e-6)
        # Adj default spread is the 1st percent column.
        assert result["adj_default_spread"] == pytest.approx(0.0306, abs=1e-6)
        # Updated label captured from "(updated …)" suffix.
        assert result["updated_label"] == "February 2026"
        # First non-percent cell is the Moody's rating.
        assert result["rating"] == "Ba3"

    def test_returns_none_when_turkey_row_missing(self):
        assert parse_turkey_row(_HTML_FIXTURE_NO_TURKEY) is None

    def test_does_not_match_turkmenistan(self):
        # Turkmenistan must not satisfy the "Turkey" lookup — false positive
        # would silently give us the wrong country's ERP.
        assert parse_turkey_row(_HTML_FIXTURE_TURKMENISTAN_ONLY) is None


# ── fetch_turkey_erp (live network, skippable) ─────────────────────────────


@pytest.mark.skipif(
    os.getenv("BIST_SKIP_NETWORK_TESTS") == "1",
    reason="network tests disabled",
)
class TestFetchLive:
    """Hit the real Damodaran URL. Skipped when offline; opt-out via env var."""

    def test_returns_plausible_turkey_erp(self):
        result = fetch_turkey_erp(force_refresh=True)
        if result is None:
            pytest.skip("Damodaran unreachable; cannot verify")
        assert isinstance(result, DamodaranTurkeyERP)
        # Sanity bounds: Turkey ERP has been in [0.05, 0.20] for the last
        # ~10 years. If we ever see a value outside this, either we parsed
        # the wrong column or the world has changed dramatically.
        assert 0.04 <= result.equity_risk_premium_pct <= 0.25
        # CRP must be < ERP (ERP = US-base ERP + CRP, mathematically).
        assert result.country_risk_premium_pct < result.equity_risk_premium_pct
