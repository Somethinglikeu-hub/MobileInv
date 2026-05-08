"""Damodaran Country Risk Premium scraper for Turkey.

Replaces the manual `bist_picker/config/macro.yaml` `equity_risk_premium_try`
entry. Fetches the same value from
https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html
and returns it as a decimal fraction (e.g. 0.0889 for 8.89%).

The Damodaran page is plain HTML — Aswath Damodaran has been publishing it
in the same table layout for ~15 years. The Turkey row contains:

    <tr>
      <td>Turkey (updated <Month> <Year>)</td>   ← country name + tag
      <td>Ba3</td>                                ← Moody's rating
      <td>3.06%</td>                              ← Adj. Default Spread
      <td>4.66%</td>                              ← Country Risk Premium
      <td>8.89%</td>                              ← Equity Risk Premium  (← this)
      <td>25.00%</td>                             ← Tax Rate
      <td>2.85%</td>                              ← Sigma_Equity / something
      <td>8.56%</td>                              ← Sigma_Country
    </tr>

We extract the 5th percentage (column index 4 after country cell). If
Damodaran ever changes column order, the test
``tests/test_damodaran_fetcher.py`` will catch it.

Caching: the result is cached on disk for 24h via FileCache. Damodaran
updates monthly at most, so 24h is generous.
"""

from __future__ import annotations

import logging
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from bist_picker.data.cache import FileCache

logger = logging.getLogger("bist_picker.data.sources.damodaran")

_CTRYPREM_URL = (
    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html"
)
_USER_AGENT = "bist-picker/2.0 (+https://github.com/Somethinglikeu-hub/MobileInv)"
_REQUEST_TIMEOUT = 30  # seconds
_CACHE_SOURCE = "damodaran"
_CACHE_TICKER = "turkey"
_CACHE_DATA_TYPE = "ctryprem_html"
_CACHE_TTL_HOURS = 24.0


@dataclass(frozen=True)
class DamodaranTurkeyERP:
    """Parsed Turkey row from Damodaran's country risk page."""

    equity_risk_premium_pct: float  # decimal, e.g. 0.0889 for 8.89%
    country_risk_premium_pct: float  # decimal
    moodys_rating: Optional[str]
    last_updated_label: Optional[str]  # e.g. "February 2026"
    fetched_at: date


def fetch_turkey_erp(
    cache: Optional[FileCache] = None,
    *,
    force_refresh: bool = False,
) -> Optional[DamodaranTurkeyERP]:
    """Fetch Turkey's Equity Risk Premium from Damodaran.

    Returns ``None`` (and logs a warning) if:
      * Network fetch fails
      * The page does not contain a Turkey row
      * The Turkey row cannot be parsed (HTML structure changed)

    Callers should fall back to the static value in macro.yaml when this
    returns None — never crash the pipeline because Damodaran's site is down.
    """
    cache = cache or FileCache()

    html: Optional[str] = None
    if not force_refresh:
        cached = cache.load_raw_response(
            _CACHE_SOURCE, _CACHE_TICKER, _CACHE_DATA_TYPE,
            max_age_hours=_CACHE_TTL_HOURS,
        )
        if isinstance(cached, str):
            html = cached
            logger.debug("Damodaran HTML loaded from cache")

    if html is None:
        try:
            req = urllib.request.Request(_CTRYPREM_URL, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                raw = resp.read()
            html = raw.decode("utf-8", errors="replace")
            try:
                cache.save_raw_response(
                    _CACHE_SOURCE, _CACHE_TICKER, _CACHE_DATA_TYPE, html,
                )
            except Exception as exc:  # pragma: no cover — cache write is best-effort
                logger.debug("Damodaran cache write failed: %s", exc)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Damodaran fetch failed: %s", exc)
            return None

    parsed = parse_turkey_row(html)
    if parsed is None:
        logger.warning("Damodaran HTML did not yield a Turkey row")
        return None

    return DamodaranTurkeyERP(
        equity_risk_premium_pct=parsed["erp"],
        country_risk_premium_pct=parsed["crp"],
        moodys_rating=parsed.get("rating"),
        last_updated_label=parsed.get("updated_label"),
        fetched_at=date.today(),
    )


# ── HTML parser ────────────────────────────────────────────────────────────

# Match a <tr>…</tr> whose first cell contains "Turkey" but not "Turkey and",
# "Turks", "Turkmenistan" etc. The "(updated …)" suffix is optional but very
# common since 2020.
_TURKEY_ROW_RE = re.compile(
    r"<tr[^>]*>\s*"
    r"<td[^>]*>\s*Turkey(?:\s*\(updated\s+([^<)]+?)\))?\s*</td>"  # group 1: updated label
    r"(.*?)"                                                       # group 2: rest of cells
    r"</tr>",
    re.IGNORECASE | re.DOTALL,
)
_CELL_RE = re.compile(r"<td[^>]*>\s*([^<]*?)\s*</td>", re.IGNORECASE | re.DOTALL)
_PCT_RE = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*%")


def parse_turkey_row(html: str) -> Optional[dict]:
    """Locate the Turkey row in Damodaran's HTML and extract numeric fields.

    Returns a dict with keys ``erp``, ``crp``, ``rating``, ``updated_label``,
    or None if the row is missing or has fewer than 5 percentage cells.
    """
    m = _TURKEY_ROW_RE.search(html)
    if not m:
        return None
    updated_label = (m.group(1) or "").strip() or None
    cells = [c.strip() for c in _CELL_RE.findall(m.group(2))]

    rating: Optional[str] = None
    pct_values: list[float] = []
    for cell in cells:
        # Plain text cells: rating like "Ba3", percentages like "8.89%"
        pct_match = _PCT_RE.search(cell)
        if pct_match:
            pct_values.append(float(pct_match.group(1)) / 100.0)
            continue
        # First non-percent cell after country name is typically the rating.
        if rating is None and cell and len(cell) <= 8:
            rating = cell

    # Expected order: [adj_default_spread, crp, erp, tax_rate, sigma_e, sigma_c]
    if len(pct_values) < 3:
        return None

    return {
        "rating": rating,
        "updated_label": updated_label,
        "adj_default_spread": pct_values[0],
        "crp": pct_values[1],
        "erp": pct_values[2],
    }
