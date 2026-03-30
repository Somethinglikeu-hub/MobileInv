"""Styling utilities for the Streamlit dashboard.

Converts terminal.py color conventions to CSS/hex colors for Streamlit.
Color rules:
  Score >= 70  -> green
  Score 50-70  -> yellow/amber
  Score < 50   -> red
  Risk HIGH    -> red
  Risk MEDIUM  -> yellow/amber
  Risk LOW     -> green
  P&L >= 0     -> green
  P&L < 0      -> red
"""

from typing import Optional


# -- Color palette (hex) --

GREEN = "#28a745"
YELLOW = "#ffc107"
RED = "#dc3545"
DIM = "#6c757d"
WHITE = "#ffffff"
BLUE = "#17a2b8"


def score_color(score: Optional[float]) -> str:
    """Return hex color for a 0-100 score value."""
    if score is None:
        return DIM
    if score >= 70:
        return GREEN
    if score >= 50:
        return YELLOW
    return RED


def risk_color(tier: Optional[str]) -> str:
    """Return hex color for a risk tier string."""
    mapping = {"HIGH": RED, "MEDIUM": YELLOW, "LOW": GREEN}
    return mapping.get((tier or "").upper(), DIM)


def pnl_color(pct: Optional[float]) -> str:
    """Return hex color for a P&L percentage."""
    if pct is None:
        return DIM
    return GREEN if pct >= 0 else RED


def fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format a float as percentage string, or '--' if None."""
    if value is None:
        return "--"
    return f"{value:.{decimals}f}%"


def fmt_float(value: Optional[float], decimals: int = 2) -> str:
    """Format a float with fixed decimals, or '--' if None."""
    if value is None:
        return "--"
    return f"{value:,.{decimals}f}"


def fmt_price(value: Optional[float]) -> str:
    """Format a price in TRY with 2 decimal places."""
    if value is None:
        return "--"
    return f"{value:,.2f} TL"


def colored_score_html(score: Optional[float]) -> str:
    """Return an HTML span with color-coded score value."""
    if score is None:
        return f'<span style="color:{DIM}">--</span>'
    color = score_color(score)
    return f'<span style="color:{color};font-weight:bold">{score:.1f}</span>'


def colored_pnl_html(pct: Optional[float]) -> str:
    """Return an HTML span with color-coded P&L percentage."""
    if pct is None:
        return f'<span style="color:{DIM}">--</span>'
    color = pnl_color(pct)
    sign = "+" if pct >= 0 else ""
    return f'<span style="color:{color};font-weight:bold">{sign}{pct:.1f}%</span>'


def colored_risk_html(tier: Optional[str]) -> str:
    """Return an HTML span with color-coded risk tier."""
    if not tier:
        return f'<span style="color:{DIM}">--</span>'
    color = risk_color(tier)
    return f'<span style="color:{color};font-weight:bold">{tier}</span>'


def regime_emoji(regime: Optional[str]) -> str:
    """Return emoji for macro regime status."""
    mapping = {"RISK_ON": "🟢", "RISK_OFF": "🔴", "TRANSITION": "🟡"}
    return mapping.get((regime or "").upper(), "⚪")


def style_dataframe(df, score_columns=None, pnl_columns=None):
    """Apply conditional formatting to a pandas DataFrame for display.

    Args:
        df: pandas DataFrame to style.
        score_columns: List of column names with 0-100 score values.
        pnl_columns: List of column names with P&L percentage values.

    Returns:
        pandas Styler object.
    """
    styler = df.style

    def _score_bg(val):
        if val is None or (isinstance(val, float) and val != val):  # NaN check
            return ""
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
            
        if v >= 70:
            return "background-color: rgba(40, 167, 69, 0.25)"
        if v >= 50:
            return "background-color: rgba(255, 193, 7, 0.20)"
        return "background-color: rgba(220, 53, 69, 0.25)"

    def _pnl_bg(val):
        if val is None or (isinstance(val, float) and val != val):
            return ""
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v >= 0:
            return "color: #28a745; font-weight: bold"
        return "color: #dc3545; font-weight: bold"

    if score_columns:
        existing = [c for c in score_columns if c in df.columns]
        if existing:
            styler = styler.map(_score_bg, subset=existing)

    if pnl_columns:
        existing = [c for c in pnl_columns if c in df.columns]
        if existing:
            styler = styler.map(_pnl_bg, subset=existing)

    return styler
