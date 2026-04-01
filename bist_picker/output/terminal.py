"""Terminal output for the BIST Stock Picker.

Uses the Rich library to display portfolio picks, current holdings status,
and single-stock deep-dive information in well-formatted terminal tables.

Public interface:
  TerminalOutput.show_portfolio(portfolio, picks, session)
  TerminalOutput.show_status(session)
  TerminalOutput.show_inspect(ticker, session)

Color conventions:
  Score >= 70   -> green
  Score 50-70   -> yellow
  Score < 50    -> red
  Risk HIGH     -> red
  Risk MEDIUM   -> yellow
  Risk LOW      -> green
  P&L positive  -> green
  P&L negative  -> red
"""

import json
import logging
from datetime import date
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sqlalchemy.orm import Session

from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    DailyPrice,
    PortfolioSelection,
    ScoringResult,
)

logger = logging.getLogger(__name__)


def _score_color(score: Optional[float]) -> str:
    """Return a Rich color tag for a 0-100 composite score."""
    if score is None:
        return "dim"
    if score >= 70:
        return "bold green"
    if score >= 50:
        return "bold yellow"
    return "bold red"


def _risk_color(tier: Optional[str]) -> str:
    """Return a Rich color tag for a risk tier string."""
    mapping = {"HIGH": "bold red", "MEDIUM": "bold yellow", "LOW": "bold green"}
    return mapping.get((tier or "").upper(), "dim")


def _pnl_color(pct: Optional[float]) -> str:
    """Return a Rich color tag for a P&L percentage."""
    if pct is None:
        return "dim"
    return "bold green" if pct >= 0 else "bold red"


def _fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format a float as a percentage string, or '--' if None."""
    if value is None:
        return "--"
    return f"{value:.{decimals}f}%"


def _fmt_float(value: Optional[float], decimals: int = 2) -> str:
    """Format a float with fixed decimals, or '--' if None."""
    if value is None:
        return "--"
    return f"{value:,.{decimals}f}"


def _fmt_price(value: Optional[float]) -> str:
    """Format a price in TRY with 2 decimal places."""
    if value is None:
        return "--"
    return f"TRY {value:,.2f}"


def _fmt_target(price: Optional[float], entry: Optional[float]) -> str:
    """Format a target price with upside %, e.g. 'TRY 189 +46%'."""
    if price is None or entry is None or entry <= 0:
        return "--"
    upside = (price - entry) / entry * 100.0
    return f"TRY {price:,.0f} +{upside:.0f}%"


def _compute_horizon_targets(
    entry: Optional[float],
    dcf_mos: Optional[float],
    single_target: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (short_3_6m, mid_12m, long_2_3y) price targets.

    Gap-closure model:
      Short (3-6 months): 25% of DCF gap closed, capped at 1.30× entry
      Medium  (12 months): 50% of DCF gap closed, capped at 2.00× entry
      Long   (2-3 years): 100% of DCF gap = full intrinsic, capped at 2.50× entry

    Falls back to proportional scaling from single_target when DCF unavailable.
    """
    if entry is None or entry <= 0:
        return None, None, None

    if dcf_mos is not None and 0.0 < dcf_mos < 100.0:
        intrinsic = entry / (1.0 - dcf_mos / 100.0)
        gap = intrinsic - entry
        short = round(min(entry + 0.25 * gap, entry * 1.30), 2)
        mid   = round(min(entry + 0.50 * gap, entry * 2.00), 2)
        long_ = round(min(intrinsic,           entry * 2.50), 2)
        return short, mid, long_

    # No DCF — scale off single_target (score-implied 12M)
    if single_target and single_target > entry:
        upside = single_target - entry
        short = round(min(entry + 0.40 * upside, entry * 1.15), 2)
        mid   = round(single_target, 2)
        long_ = round(min(entry + 1.75 * upside, entry * 1.50), 2)
        return short, mid, long_

    return None, None, None


class TerminalOutput:
    """Renders portfolio data to the terminal using Rich.

    Args:
        console: Rich Console instance. A new one is created if omitted.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self._console = console or Console(force_terminal=True, legacy_windows=False)

    # ── Public methods ─────────────────────────────────────────────────────────

    def show_portfolio(
        self,
        portfolio: str,
        picks: list[dict],
        session: Session,
    ) -> None:
        """Display the selected picks for one portfolio in a Rich table.

        Args:
            portfolio: Portfolio name ('ALPHA', 'BETA', 'DELTA').
            picks: List of pick dicts from PortfolioSelector.select().
            session: Active SQLAlchemy session (used to load P/E and ROE).
        """
        portfolio_upper = portfolio.upper()
        month_label = date.today().strftime("%B %Y")
        title = f"{portfolio_upper} PORTFOLIO -- {month_label}"

        table = Table(
            title=title,
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            title_style="bold white",
            expand=False,
        )
        table.add_column("Rank", justify="center", style="bold white", min_width=4)
        table.add_column("Ticker", justify="center", style="bold white", min_width=8)
        table.add_column("Score", justify="right", min_width=6)
        table.add_column("Entry", justify="right", min_width=10)
        table.add_column("3-6M Target", justify="right", min_width=14)
        table.add_column("12M Target", justify="right", min_width=14)
        table.add_column("2-3Y Target", justify="right", min_width=14)
        table.add_column("Stop", justify="right", min_width=10)
        table.add_column("Risk", justify="center", min_width=8)
        table.add_column("ROE", justify="right", min_width=7)
        table.add_column("P/E", justify="right", min_width=7)

        if not picks:
            self._console.print(
                Panel(
                    "[yellow]No picks available for this portfolio.[/yellow]",
                    title=title,
                    border_style="yellow",
                )
            )
            return

        for pick in picks:
            extra = self._load_pick_extras(pick["company_id"], pick.get("entry_price"), session)
            score = pick.get("score")
            risk = extra.get("risk_tier")
            entry = pick.get("entry_price")
            # Prefer dcf_mos from pick dict (live run); fall back to scoring_results (report from DB)
            dcf_mos = pick.get("dcf_mos") if pick.get("dcf_mos") is not None else extra.get("dcf_mos")
            single_target = pick.get("target_price")

            t_short, t_mid, t_long = _compute_horizon_targets(entry, dcf_mos, single_target)

            table.add_row(
                str(pick.get("rank", "--")),
                pick.get("ticker", "--"),
                Text(f"{score:.1f}" if score is not None else "--", style=_score_color(score)),
                _fmt_price(entry),
                _fmt_target(t_short, entry),
                _fmt_target(t_mid,   entry),
                _fmt_target(t_long,  entry),
                _fmt_price(pick.get("stop_loss")),
                Text(risk or "--", style=_risk_color(risk)),
                _fmt_pct(extra.get("roe_pct")),
                _fmt_float(extra.get("pe_ratio"), decimals=1),
            )

        self._console.print(table)
        self._console.print()

    def show_all_portfolios(
        self,
        all_picks: dict[str, list[dict]],
        session: Session,
    ) -> None:
        """Display all three portfolios one after another.

        Args:
            all_picks: Dict from PortfolioSelector.select_all() with keys
                'alpha', 'beta', 'delta'.
            session: Active SQLAlchemy session.
        """
        # NOTE: BETA and DELTA commented out — only ALPHA portfolio is shown.
        # To re-enable, uncomment "beta" and "delta" in the tuple below.
        for portfolio_key in ("alpha",):  # "beta", "delta"):
            picks = all_picks.get(portfolio_key, [])
            self.show_portfolio(portfolio_key, picks, session)

    def show_status(self, session: Session) -> None:
        """Display all current open holdings across all three portfolios.

        Fetches all PortfolioSelection rows with no exit_date, looks up the
        latest price, and displays P&L alongside entry/target/stop values.

        Args:
            session: Active SQLAlchemy session.
        """
        # All open positions (no exit date)
        open_positions = (
            session.query(PortfolioSelection, Company)
            .join(Company, Company.id == PortfolioSelection.company_id)
            .filter(PortfolioSelection.exit_date.is_(None))
            .order_by(PortfolioSelection.portfolio, PortfolioSelection.composite_score.desc())
            .all()
        )

        if not open_positions:
            self._console.print(
                Panel(
                    "[yellow]No open holdings found in the database.[/yellow]",
                    title="Portfolio Status",
                    border_style="yellow",
                )
            )
            return

        table = Table(
            title="Current Holdings",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            title_style="bold white",
            expand=False,
        )
        table.add_column("Portfolio", justify="center", style="bold white", min_width=9)
        table.add_column("Ticker", justify="center", style="bold white", min_width=8)
        table.add_column("Entry", justify="right", min_width=10)
        table.add_column("Current", justify="right", min_width=10)
        table.add_column("P&L%", justify="right", min_width=8)
        table.add_column("Target", justify="right", min_width=10)
        table.add_column("Stop", justify="right", min_width=10)
        table.add_column("Score", justify="right", min_width=6)
        table.add_column("Days", justify="right", min_width=5)

        for pos, company in open_positions:
            current_price = self._get_latest_price(company.id, session)
            entry = pos.entry_price
            pnl_pct: Optional[float] = None
            if entry and current_price:
                pnl_pct = (current_price - entry) / entry * 100.0

            days_held: Optional[int] = None
            if pos.selection_date:
                days_held = (date.today() - pos.selection_date).days

            score = pos.composite_score
            pnl_color = _pnl_color(pnl_pct)

            table.add_row(
                pos.portfolio,
                company.ticker,
                _fmt_price(entry),
                _fmt_price(current_price),
                Text(_fmt_pct(pnl_pct), style=pnl_color),
                _fmt_price(pos.target_price),
                _fmt_price(pos.stop_loss_price),
                Text(f"{score:.1f}" if score else "--", style=_score_color(score)),
                str(days_held) if days_held is not None else "--",
            )

        self._console.print(table)
        self._console.print()

    def show_inspect(self, ticker: str, session: Session) -> None:
        """Deep dive on a single stock: company info, all factor scores, key metrics.

        Args:
            ticker: BIST ticker code (e.g., 'THYAO').
            session: Active SQLAlchemy session.
        """
        ticker_upper = ticker.strip().upper()
        company = (
            session.query(Company)
            .filter(Company.ticker == ticker_upper)
            .first()
        )

        if not company:
            self._console.print(f"[red]Ticker [bold]{ticker_upper}[/bold] not found in database.[/red]")
            return

        # ── Company header panel ──────────────────────────────────────────────
        info_lines = [
            f"[bold]{company.name or ticker_upper}[/bold]",
            f"Type: [bold]{company.company_type or '--'}[/bold]  "
            f"Sector: [bold]{company.sector_custom or company.sector_bist or '--'}[/bold]",
            f"BIST-100: {'[green]Yes[/green]' if company.is_bist100 else '[dim]No[/dim]'}  "
            f"IPO: {'[yellow]Yes[/yellow]' if company.is_ipo else '[dim]No[/dim]'}  "
            f"Free Float: [bold]{_fmt_pct(company.free_float_pct)}[/bold]",
            f"Listing Date: [bold]{company.listing_date or '--'}[/bold]  "
            f"Active: {'[green]Yes[/green]' if company.is_active else '[red]No[/red]'}",
        ]
        self._console.print(
            Panel(
                "\n".join(info_lines),
                title=f"[bold white]{ticker_upper}[/bold white]",
                border_style="cyan",
            )
        )

        # ── Latest scoring result ─────────────────────────────────────────────
        score_row = (
            session.query(ScoringResult)
            .filter(ScoringResult.company_id == company.id)
            .order_by(ScoringResult.scoring_date.desc())
            .first()
        )

        if score_row:
            self._show_factor_scores(score_row)
            self._show_composite_scores(score_row)
        else:
            self._console.print("[dim]No scoring results found.[/dim]\n")

        # ── Latest adjusted metrics ───────────────────────────────────────────
        metric_row = (
            session.query(AdjustedMetric)
            .filter(AdjustedMetric.company_id == company.id)
            .order_by(AdjustedMetric.period_end.desc())
            .first()
        )

        if metric_row:
            self._show_key_metrics(metric_row)
        else:
            self._console.print("[dim]No adjusted metrics found.[/dim]\n")

        # ── Quality flags ─────────────────────────────────────────────────────
        if score_row and score_row.quality_flags_json:
            self._show_quality_flags(score_row.quality_flags_json)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _show_factor_scores(self, score_row: ScoringResult) -> None:
        """Render a table of the 9 individual factor scores (0-100 percentile)."""
        table = Table(
            title="Factor Scores (0-100 percentile)",
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            expand=False,
        )
        table.add_column("Factor", style="bold white", min_width=20)
        table.add_column("Score", justify="right", min_width=8)

        factors = [
            ("Buffett Quality", score_row.buffett_score),
            ("Graham Value", score_row.graham_score),
            ("Piotroski F-Score", score_row.piotroski_fscore),
            ("Magic Formula Rank", score_row.magic_formula_rank),
            ("Lynch PEG", score_row.lynch_peg_score),
            ("DCF Margin of Safety", score_row.dcf_margin_of_safety_pct),
            ("Momentum", score_row.momentum_score),
            ("Insider Activity", score_row.insider_score),
            ("Technical", score_row.technical_score),
        ]

        for name, val in factors:
            score_text = Text(
                f"{val:.1f}" if val is not None else "--",
                style=_score_color(val),
            )
            table.add_row(name, score_text)

        scoring_date = score_row.scoring_date
        model = score_row.model_used or "--"
        self._console.print(
            f"[dim]Scoring date: {scoring_date}  Model: {model}  "
            f"Data completeness: {_fmt_pct(score_row.data_completeness)}[/dim]"
        )
        self._console.print(table)
        self._console.print()

    def _show_composite_scores(self, score_row: ScoringResult) -> None:
        """Render composite scores for all three portfolios."""
        table = Table(
            title="Composite Scores",
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            expand=False,
        )
        table.add_column("Portfolio", style="bold white", min_width=10)
        table.add_column("Score", justify="right", min_width=8)
        table.add_column("Risk Tier", justify="center", min_width=10)

        risk = score_row.risk_tier
        # NOTE: BETA and DELTA commented out — only ALPHA shown.
        # To re-enable, uncomment the BETA and DELTA lines below.
        for label, val in [
            ("ALPHA", score_row.composite_alpha),
            # ("BETA", score_row.composite_beta),
            # ("DELTA", score_row.composite_delta),
        ]:
            table.add_row(
                label,
                Text(f"{val:.1f}" if val is not None else "--", style=_score_color(val)),
                Text(risk or "--", style=_risk_color(risk)),
            )

        self._console.print(table)
        self._console.print()

    def _show_key_metrics(self, metric_row: AdjustedMetric) -> None:
        """Render a table of the key adjusted financial metrics."""
        table = Table(
            title=f"Key Metrics  (period: {metric_row.period_end})",
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            expand=False,
        )
        table.add_column("Metric", style="bold white", min_width=25)
        table.add_column("Value", justify="right", min_width=18)

        def _m(label: str, value: Optional[float], is_pct: bool = False) -> None:
            if is_pct:
                formatted = _fmt_pct(value)
            else:
                formatted = _fmt_float(value, decimals=2) if value is not None else "--"
            table.add_row(label, formatted)

        _m("Reported Net Income (TRY)", metric_row.reported_net_income)
        _m("Monetary G/L (TRY)", metric_row.monetary_gain_loss)
        _m("Adjusted Net Income (TRY)", metric_row.adjusted_net_income)
        _m("Owner Earnings (TRY)", metric_row.owner_earnings)
        _m("Free Cash Flow (TRY)", metric_row.free_cash_flow)
        _m("ROE (adjusted)", metric_row.roe_adjusted * 100.0 if metric_row.roe_adjusted is not None else None, is_pct=True)
        _m("ROA (adjusted)", metric_row.roa_adjusted * 100.0 if metric_row.roa_adjusted is not None else None, is_pct=True)
        _m("EPS (adjusted, TRY)", metric_row.eps_adjusted)
        _m("Real EPS Growth", metric_row.real_eps_growth_pct * 100.0 if metric_row.real_eps_growth_pct is not None else None, is_pct=True)
        _m("Related-Party Revenue %", metric_row.related_party_revenue_pct * 100.0 if metric_row.related_party_revenue_pct is not None else None, is_pct=True)

        self._console.print(table)
        self._console.print()

    def _show_quality_flags(self, quality_flags_json: str) -> None:
        """Render any quality flags stored in the scoring result."""
        try:
            flags = json.loads(quality_flags_json)
        except (json.JSONDecodeError, TypeError):
            return

        if not flags:
            return

        table = Table(
            title="Quality Flags",
            show_header=True,
            header_style="bold cyan",
            border_style="yellow",
            expand=False,
        )
        table.add_column("Flag", style="bold yellow", min_width=20)
        table.add_column("Detail", min_width=30)

        if isinstance(flags, dict):
            for key, detail in flags.items():
                table.add_row(str(key), str(detail))
        elif isinstance(flags, list):
            for item in flags:
                table.add_row(str(item), "")

        self._console.print(table)
        self._console.print()

    def _load_pick_extras(
        self,
        company_id: int,
        entry_price: Optional[float],
        session: Session,
    ) -> dict:
        """Load ROE, risk tier, P/E, and DCF MoS for a single pick from the DB.

        Returns a dict with keys: roe_pct, pe_ratio, risk_tier, dcf_mos.
        All values may be None if data is unavailable.
        """
        extras: dict = {"roe_pct": None, "pe_ratio": None, "risk_tier": None, "dcf_mos": None}

        # Latest adjusted metric
        metric = (
            session.query(AdjustedMetric)
            .filter(AdjustedMetric.company_id == company_id)
            .order_by(AdjustedMetric.period_end.desc())
            .first()
        )
        if metric:
            if metric.roe_adjusted is not None:
                extras["roe_pct"] = metric.roe_adjusted * 100.0
            if entry_price and metric.eps_adjusted and metric.eps_adjusted > 0:
                extras["pe_ratio"] = entry_price / metric.eps_adjusted

        # Latest scoring result for risk tier and DCF MoS
        score_row = (
            session.query(ScoringResult)
            .filter(ScoringResult.company_id == company_id)
            .order_by(ScoringResult.scoring_date.desc())
            .first()
        )
        if score_row:
            extras["risk_tier"] = score_row.risk_tier
            extras["dcf_mos"] = score_row.dcf_margin_of_safety_pct

        return extras

    def _get_latest_price(
        self, company_id: int, session: Session
    ) -> Optional[float]:
        """Return the most recent adjusted_close (or close) for a company."""
        row = (
            session.query(DailyPrice)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.adjusted_close.isnot(None),
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )
        if row:
            return row.adjusted_close

        row = (
            session.query(DailyPrice)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.close.isnot(None),
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )
        return row.close if row else None
