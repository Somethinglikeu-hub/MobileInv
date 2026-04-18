"""CLI entry point for BIST Stock Picker.

Uses Click to provide commands for the full analysis pipeline:
  menu        -> Interactive terminal menu (setup + run + daily ops)
  fetch       -> Stage 1: download price and financial data
  clean       -> Stage 2: inflation-adjust, compute metrics, classify companies
  score       -> Stage 3: calculate and normalize all factor scores
  pick        -> Stage 4: select portfolio stocks
  report      -> Stage 5: display portfolio tables
  run         -> All stages sequentially
  status      -> Current portfolio holdings with P&L
  inspect     -> Deep dive on a single ticker
  check-exits -> Mid-month exit check (stop-loss / target / thesis)

Global flags:
  --verbose   Enable DEBUG logging
  --dry-run   Skip all database writes (read-only pipeline)
"""

import logging
import os
from datetime import date
from pathlib import Path

import click
from rich.console import Console

# When run as subprocess from dashboard, PIPE_MODE=1 is set.
PIPE_MODE = os.environ.get("PIPE_MODE") == "1"
if PIPE_MODE:
    import sys
    # In pipe mode: no markup/highlight so output is plain text,
    # and write to a force-flushing stream.
    console = Console(markup=False, highlight=False, force_terminal=False, file=sys.stdout)
else:
    console = Console()


def _get_engine_and_tables():
    """Create DB engine and ensure all tables exist. Returns the engine."""
    from bist_picker.db.connection import ensure_runtime_db_ready, get_engine
    engine = get_engine()
    ensure_runtime_db_ready(engine)
    return engine


# ── Root group ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.option("--dry-run", is_flag=True, help="Skip all database writes.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, dry_run: bool) -> None:
    """BIST Stock Picker -- Buffett-style fundamental analysis for Borsa Istanbul."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run

    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── fetch ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker", multiple=True, help="Fetch specific ticker(s) only.")
@click.option("--prices-only", is_flag=True, help="Fetch only price data.")
@click.option("--limit", type=int, default=0, help="Limit number of tickers to fetch.")
@click.option(
    "--history",
    is_flag=True,
    help="One-time deep backfill: fetch 5 years of quarterly financial data.",
)
@click.pass_context
def fetch(ctx: click.Context, ticker: tuple, prices_only: bool, limit: int, history: bool) -> None:
    """Stage 1: Download data from all sources (prices + financials + macro)."""
    from bist_picker.db.connection import session_scope
    from bist_picker.data.fetcher import DataFetcher

    dry_run: bool = ctx.obj.get("dry_run", False)
    if dry_run:
        console.print("[yellow]--dry-run: skipping all fetches.[/yellow]")
        return

    engine = _get_engine_and_tables()
    with session_scope(engine) as session:
        fetcher = DataFetcher(session=session, console=console)
        tickers = list(ticker) if ticker else None

        if history:
            # One-time deep backfill — financials only, 5 years quarterly
            console.print(
                "[bold magenta]Historical backfill: fetching 5 years of "
                "quarterly financial data...[/bold magenta]"
            )
            if not tickers:
                # Need universe first to know which tickers exist
                fetcher.fetch_universe()
                session.commit()
            fetcher.fetch_history(tickers=tickers)
            return

        if prices_only:
            console.print("[bold blue]Fetching prices only...[/bold blue]")
            if tickers:
                for t in tickers:
                    fetcher._get_or_create_company(t)
                session.commit()
            fetcher.fetch_prices(tickers=tickers)
        elif tickers:
            console.print(
                f"[bold blue]Fetching data for {', '.join(tickers)}...[/bold blue]"
            )
            fetcher.fetch_universe()
            session.commit()
            fetcher.fetch_prices(tickers=tickers)
            session.commit()
            fetcher.fetch_financials(tickers=tickers)
            session.commit()
            fetcher.fetch_macro()
        else:
            console.print("[bold blue]Fetching all data...[/bold blue]")
            fetcher.fetch_all(limit=limit)


# ── clean ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def clean(ctx: click.Context) -> None:
    """Stage 2: Classify companies, adjust for inflation, compute clean metrics."""
    from bist_picker.db.connection import session_scope
    from bist_picker.classification.company_type import CompanyClassifier
    from bist_picker.classification.sector_mapper import SectorMapper
    from bist_picker.cleaning.financial_prep import MetricsCalculator

    dry_run: bool = ctx.obj.get("dry_run", False)
    engine = _get_engine_and_tables()

    with session_scope(engine) as session:
        # Step 2a: classify company types (OPERATING / BANK / HOLDING / ...)
        console.print("[bold blue]Classifying companies...[/bold blue]")
        classifier = CompanyClassifier()
        stats = classifier.classify_all(session)
        console.print(
            f"  Classified [bold]{stats.get('total', 0)}[/bold] companies: "
            + ", ".join(
                f"{t}: {n}" for t, n in stats.get("by_type", {}).items()
            )
        )

        console.print("[bold blue]Mapping custom sectors...[/bold blue]")
        sector_mapper = SectorMapper()
        sector_stats = sector_mapper.map_all(session)
        console.print(
            f"  Mapped [bold]{sector_stats.get('total', 0)}[/bold] companies "
            f"into [bold]{len(sector_stats.get('by_sector', {}))}[/bold] custom sectors"
        )

        if dry_run:
            console.print("[yellow]--dry-run: rolling back classification.[/yellow]")
            session.rollback()
            return

        session.commit()

        # Step 2b: calculate adjusted metrics
        console.print("[bold blue]Calculating adjusted financial metrics...[/bold blue]")
        calculator = MetricsCalculator(session=session, console=console)
        result = calculator.calculate_all()
        processed = result.get("calculated", 0)
        skipped = result.get("skipped", 0)
        errors = result.get("errors", 0)
        console.print(
            f"  Metrics: [green]{processed}[/green] processed, "
            f"[yellow]{skipped}[/yellow] skipped, "
            f"[red]{errors}[/red] errors"
        )


# ── score ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--use-regime", is_flag=True, help="Use dynamic regime-switching weights.")
@click.pass_context
def score(ctx: click.Context, use_regime: bool) -> None:
    """Stage 3: Calculate, normalize, and compose all factor scores."""
    import pandas as pd
    from bist_picker.db.connection import session_scope
    from bist_picker.db.schema import Company, ScoringResult
    from bist_picker.scoring.factors.buffett import BuffettScorer
    from bist_picker.scoring.factors.dcf import DCFScorer
    from bist_picker.scoring.factors.graham import GrahamScorer
    from bist_picker.scoring.factors.lynch import LynchScorer
    from bist_picker.scoring.factors.magic_formula import MagicFormulaScorer
    from bist_picker.scoring.factors.momentum import MomentumScorer
    from bist_picker.scoring.factors.piotroski import PiotroskiScorer
    from bist_picker.scoring.factors.technical import TechnicalScorer
    from bist_picker.scoring.normalizer import ScoreNormalizer
    from bist_picker.scoring.factors.piotroski import PiotroskiScorer
    from bist_picker.scoring.normalizer import ScoreNormalizer
    from bist_picker.scoring.composer import ScoreComposer
    from bist_picker.scoring.context import ScoringContext

    dry_run: bool = ctx.obj.get("dry_run", False)
    engine = _get_engine_and_tables()
    scoring_date = date.today()

    _FACTOR_COLS = [
        "buffett_score", "graham_score", "piotroski_fscore",
        "magic_formula_rank", "lynch_peg_score", "momentum_score",
        "technical_score",
    ]

    with session_scope(engine) as session:
        companies = (
            session.query(Company)
            .filter(Company.is_active == True)
            .all()
        )
        if not companies:
            console.print("[yellow]No active companies found -- run 'bist fetch' first.[/yellow]")
            return

        console.print(
            f"[bold blue]Scoring {len(companies)} companies "
            f"(date: {scoring_date})...[/bold blue]"
        )

        # Step 3a: Per-company factor scorers
        buffett = BuffettScorer()
        dcf = DCFScorer()
        graham = GrahamScorer()
        piotroski = PiotroskiScorer()
        lynch = LynchScorer()
        
        # Initialize ScoringContext for bulk data loading
        context = ScoringContext(session, scoring_date)
        all_ids = [c.id for c in companies]
        context.load_data(all_ids)

        raw_scores: dict[int, dict] = {}
        for company in companies:
            cid = company.id
            row: dict = {
                "company_id": cid,
                "model_used": company.company_type or "OPERATING",
            }

            b = buffett.score(cid, session, scoring_date=scoring_date, scoring_context=context)
            row["buffett_score"] = (b or {}).get("buffett_combined")

            d = dcf.score(cid, session, scoring_date=scoring_date) # DCF not yet context-aware
            # dcf_margin_of_safety_pct is stored raw (not normalized)
            row["dcf_margin_of_safety_pct"] = (d or {}).get("dcf_combined")

            g = graham.score(cid, session, scoring_date=scoring_date, scoring_context=context)
            row["graham_score"] = (g or {}).get("graham_combined")

            p = piotroski.score(cid, session, scoring_date=scoring_date, scoring_context=context)
            row["piotroski_fscore"] = (p or {}).get("fscore_total")
            row["piotroski_fscore_raw"] = int((p or {}).get("fscore_total", 0)) if p else None

            l = lynch.score(cid, session, scoring_date=scoring_date, scoring_context=context)
            row["lynch_peg_score"] = (l or {}).get("peg_score")

            raw_scores[cid] = row

        # Step 3b: Batch-scored factors
        console.print("[dim]  Running Magic Formula...[/dim]")
        mf_scores = MagicFormulaScorer().score_all(session, scoring_date=scoring_date)
        for cid, result in mf_scores.items():
            if cid in raw_scores:
                raw_scores[cid]["magic_formula_rank"] = result.get("magic_formula_score")

        console.print("[dim]  Running Momentum...[/dim]")
        mom_scores = MomentumScorer().score_all(session, scoring_date=scoring_date)
        for cid, result in mom_scores.items():
            if cid in raw_scores:
                raw_scores[cid]["momentum_score"] = result.get("momentum_combined")

        console.print("[dim]  Running Technical...[/dim]")
        tech_scores = TechnicalScorer().score_all(session, scoring_date=scoring_date)
        for cid, result in tech_scores.items():
            if cid in raw_scores:
                raw_scores[cid]["technical_score"] = result.get("technical_score")

        console.print("[dim]  Running Dividend Yield...[/dim]")
        from bist_picker.scoring.factors.dividend import DividendYieldScorer
        div_scores = DividendYieldScorer().score_all(session, scoring_date=scoring_date)
        for cid, result in div_scores.items():
            if cid in raw_scores:
                raw_scores[cid]["dividend_score"] = result.get("dividend_score")

        # Step 3b+: Sector-specific model scorers (banking, holding)
        console.print("[dim]  Running Banking model...[/dim]")
        from bist_picker.scoring.models.banking import BankingScorer
        bank_scores = BankingScorer().score_all(session, scoring_date=scoring_date)
        for cid, result in bank_scores.items():
            if cid in raw_scores:
                raw_scores[cid]["banking_composite"] = result.get("banking_composite")
                raw_scores[cid]["data_completeness"] = result.get("data_completeness")

        console.print("[dim]  Running Holding model...[/dim]")
        from bist_picker.scoring.models.holding import HoldingScorer
        hold_scores = HoldingScorer().score_all(session, scoring_date=scoring_date)
        for cid, result in hold_scores.items():
            if cid in raw_scores:
                raw_scores[cid]["holding_composite"] = result.get("holding_composite")
                raw_scores[cid]["data_completeness"] = result.get("data_completeness")

        console.print("[dim]  Running REIT model...[/dim]")
        from bist_picker.scoring.models.reit import ReitScorer
        reit_scores = ReitScorer().score_all(session, scoring_date=scoring_date)
        for cid, result in reit_scores.items():
            if cid in raw_scores:
                raw_scores[cid]["reit_composite"] = result.get("reit_composite")
                raw_scores[cid]["data_completeness"] = result.get("data_completeness")
                raw_scores[cid]["model_used"] = "REIT"

        if dry_run:
            console.print("[yellow]--dry-run: skipping DB writes for scores.[/yellow]")
            return

        # Step 3c: Upsert raw ScoringResult rows
        # Note: dcf_margin_of_safety_pct is stored raw so target-price logic can
        # still use the original margin-of-safety percentage. The composer
        # normalizes it on the fly when building composites.
        _EXTRA_COLS = ["model_used", "dcf_margin_of_safety_pct",
                       "dividend_score", "banking_composite", "holding_composite",
                       "reit_composite", "piotroski_fscore_raw", "data_completeness"]
        console.print("[dim]  Writing raw scores to DB...[/dim]")
        for cid, row in raw_scores.items():
            existing = (
                session.query(ScoringResult)
                .filter_by(company_id=cid, scoring_date=scoring_date)
                .first()
            )
            if existing:
                for col in _FACTOR_COLS + _EXTRA_COLS:
                    if col in row:
                        setattr(existing, col, row.get(col))
            else:
                session.add(ScoringResult(
                    company_id=cid,
                    scoring_date=scoring_date,
                    model_used=row.get("model_used"),
                    buffett_score=row.get("buffett_score"),
                    graham_score=row.get("graham_score"),
                    piotroski_fscore=row.get("piotroski_fscore"),
                    piotroski_fscore_raw=row.get("piotroski_fscore_raw"),
                    magic_formula_rank=row.get("magic_formula_rank"),
                    lynch_peg_score=row.get("lynch_peg_score"),
                    momentum_score=row.get("momentum_score"),
                    technical_score=row.get("technical_score"),
                    dcf_margin_of_safety_pct=row.get("dcf_margin_of_safety_pct"),
                    dividend_score=row.get("dividend_score"),
                    banking_composite=row.get("banking_composite"),
                    holding_composite=row.get("holding_composite"),
                    reit_composite=row.get("reit_composite"),
                    data_completeness=row.get("data_completeness"),
                ))
        session.commit()

        # Step 3d: Normalize each factor in-place
        console.print("[dim]  Normalizing factor scores...[/dim]")
        rows = (
            session.query(ScoringResult, Company)
            .join(Company, Company.id == ScoringResult.company_id)
            .filter(ScoringResult.scoring_date == scoring_date)
            .all()
        )
        if rows:
            df = pd.DataFrame([
                {
                    "id": sr.id,
                    "sector": c.sector_custom or c.sector_bist or "UNKNOWN",
                    **{col: getattr(sr, col) for col in _FACTOR_COLS},
                }
                for sr, c in rows
            ]).set_index("id")

            normalizer = ScoreNormalizer()
            for col in _FACTOR_COLS:
                if col in df.columns and df[col].notna().any():
                    df[col] = normalizer.normalize_factor(df, col, "sector")

            for row_id, row_data in df.iterrows():
                sr = session.get(ScoringResult, int(row_id))
                if sr:
                    for col in _FACTOR_COLS:
                        val = row_data.get(col)
                        setattr(sr, col, None if pd.isna(val) else float(val))
            session.commit()

        # Step 3e: Classify risk tiers
        console.print("[dim]  Classifying risk tiers...[/dim]")
        try:
            from bist_picker.classification.risk_classifier import RiskClassifier
            risk_clf = RiskClassifier()
            risk_clf.classify_all(session, scoring_date=scoring_date)
        except Exception as exc:
            console.print(f"[yellow]Warning: risk classification failed: {exc}[/yellow]")

        # Step 3f: Compute composite scores
        console.print("[dim]  Computing composite scores...[/dim]")
        try:
            composer = ScoreComposer()
            composer.compose_all(session, scoring_date=scoring_date, use_regime=use_regime)
        except Exception as exc:
            console.print(f"[yellow]Warning: composite scoring failed: {exc}[/yellow]")

        total_scored = sum(
            1 for r in raw_scores.values()
            if any(r.get(c) is not None for c in _FACTOR_COLS)
        )
        console.print(
            f"[green]Scored {total_scored}[/green] / {len(companies)} companies."
        )


# ── pick ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def pick(ctx: click.Context) -> None:
    """Stage 4: Select 3 stocks per portfolio using composite scores."""
    from bist_picker.db.connection import session_scope
    from bist_picker.portfolio.selector import PortfolioSelector

    dry_run: bool = ctx.obj.get("dry_run", False)
    engine = _get_engine_and_tables()

    with session_scope(engine) as session:
        console.print("[bold blue]Selecting portfolio stocks...[/bold blue]")
        selector = PortfolioSelector()

        if dry_run:
            all_picks = selector.select_all(session)
            for portfolio, picks in all_picks.items():
                tickers = [p["ticker"] for p in picks]
                console.print(
                    f"  {portfolio.upper()}: {', '.join(tickers) if tickers else 'no picks'}"
                    " [dim](dry-run, not stored)[/dim]"
                )
        else:
            all_picks = selector.select_and_store(session)
            for portfolio, picks in all_picks.items():
                tickers = [p["ticker"] for p in picks]
                console.print(
                    f"  {portfolio.upper()}: {', '.join(tickers) if tickers else 'no picks'}"
                )


# ── report ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--portfolio",
    type=click.Choice(["alpha", "all"], case_sensitive=False),
    default="all",
    help="Which portfolio to display.",
)
@click.option(
    "--format",
    type=click.Choice(["terminal", "excel"], case_sensitive=False),
    default="terminal",
    help="Output format: terminal table or Excel file.",
)
@click.pass_context
def report(ctx: click.Context, portfolio: str, format: str) -> None:
    """Stage 5: Display portfolio tables or generate Excel report."""
    from bist_picker.db.connection import session_scope
    from bist_picker.db.schema import Company, PortfolioSelection
    from bist_picker.output.terminal import TerminalOutput
    from bist_picker.portfolio.selector import get_selection_target_count
    from rich.table import Table
    from rich import box

    engine = _get_engine_and_tables()


    with session_scope(engine) as session:
        if format.lower() == "excel":
            from bist_picker.output.excel import ExcelReporter
            reporter = ExcelReporter()
            try:
                path = reporter.generate(session)
                console.print(f"[green]Excel report generated: {path}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to generate Excel report: {e}[/red]")
            return

        # Terminal Output Logic
        from bist_picker.output.performance import PerformanceTracker
        
        # Performance Summary
        tracker = PerformanceTracker(session)
        console.print("[bold underline]Performance Summary[/bold underline]")
        
        perf_table = Table(box=box.SIMPLE)
        perf_table.add_column("Portfolio", style="cyan")
        perf_table.add_column("Avg Return", justify="right")
        perf_table.add_column("Win Rate", justify="right")
        
        for p in ["ALPHA"]:
            stats = tracker.calculate_portfolio_performance(p)
            avg = stats.get("total_return_avg", 0.0)
            win = stats.get("win_rate", 0.0)
            color = "green" if avg >= 0 else "red"
            perf_table.add_row(p, f"[{color}]{avg:.1f}%[/{color}]", f"{win:.1f}%")
            
        console.print(perf_table)
        console.print()

        output = TerminalOutput(console=console)
        target_count = get_selection_target_count()
        portfolios_to_show = (
            ["alpha"]
            if portfolio.lower() == "all"
            else [portfolio.lower()]
        )

        for pname in portfolios_to_show:
            # Load most recent selections for this portfolio
            rows = (
                session.query(PortfolioSelection, Company)
                .join(Company, Company.id == PortfolioSelection.company_id)
                .filter(PortfolioSelection.portfolio == pname.upper())
                .order_by(
                    PortfolioSelection.selection_date.desc(),
                    PortfolioSelection.composite_score.desc(),
                )
                .limit(10)
                .all()
            )

            if not rows:
                console.print(
                    f"[yellow]No selections found for {pname.upper()} -- "
                    "run 'bist pick' first.[/yellow]"
                )
                continue

            # Use only picks from the most recent selection_date
            # Safe access if rows is not empty
            most_recent_date = rows[0][0].selection_date 
            picks = []
            for rank, (pos, company) in enumerate(rows, start=1):
                if pos.selection_date != most_recent_date:
                    break
                if len(picks) >= target_count:
                    break
                picks.append({
                    "company_id": company.id,
                    "ticker": company.ticker,
                    "score": pos.composite_score,
                    "rank": rank,
                    "entry_price": pos.entry_price,
                    "target_price": pos.target_price,
                    "stop_loss": pos.stop_loss_price,
                })

            output.show_portfolio(pname, picks, session)


# ── run ────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker", multiple=True, help="Limit pipeline to specific ticker(s).")
@click.option("--use-regime", is_flag=True, help="Use dynamic regime-switching weights.")
@click.pass_context
def run(ctx: click.Context, ticker: tuple, use_regime: bool) -> None:
    """Run all pipeline stages sequentially: fetch, clean, score, pick, report."""
    console.print("[bold blue]Running full pipeline...[/bold blue]")
    ctx.invoke(fetch, ticker=ticker, prices_only=False, limit=0)
    ctx.invoke(clean)
    ctx.invoke(score, use_regime=use_regime)
    ctx.invoke(pick)
    ctx.invoke(report, portfolio="all")
    console.print("[bold green]Pipeline complete.[/bold green]")


# ── status ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current portfolio holdings with entry price and P&L."""
    from bist_picker.db.connection import session_scope
    from bist_picker.output.terminal import TerminalOutput

    engine = _get_engine_and_tables()
    with session_scope(engine) as session:
        output = TerminalOutput(console=console)
        output.show_status(session)


# ── inspect ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.pass_context
def inspect(ctx: click.Context, ticker: str) -> None:
    """Deep dive on a single stock (e.g.: bist inspect THYAO)."""
    from bist_picker.db.connection import session_scope
    from bist_picker.output.terminal import TerminalOutput

    engine = _get_engine_and_tables()
    with session_scope(engine) as session:
        output = TerminalOutput(console=console)
        output.show_inspect(ticker, session)


# ── check-exits ────────────────────────────────────────────────────────────────

@cli.command(name="check-exits")
@click.pass_context
def check_exits(ctx: click.Context) -> None:
    """Mid-month exit check: evaluate stop-loss, target, and thesis-breaker conditions."""
    from rich.table import Table
    from bist_picker.db.connection import session_scope
    from bist_picker.portfolio.exit_rules import ExitRuleChecker

    engine = _get_engine_and_tables()
    with session_scope(engine) as session:
        checker = ExitRuleChecker(session)
        signals = checker.check_exits()

        if not signals:
            console.print("[green]No exit signals found. All positions holding steady.[/green]")
            return

        console.print(f"[bold red]Found {len(signals)} exit signals![/bold red]")
        
        table = Table(title="Exit Signals", style="red")
        table.add_column("Ticker", style="cyan")
        table.add_column("Portfolio", style="magenta")
        table.add_column("Entry Price", justify="right")
        table.add_column("Current Price", justify="right")
        table.add_column("Return %", justify="right")
        table.add_column("Reason", style="bold red")
        table.add_column("Details")

        for s in signals:
            color = "green" if s["return_pct"] >= 0 else "red"
            table.add_row(
                s["ticker"],
                s["portfolio"],
                f"{s['entry_price']:.2f}",
                f"{s['current_price']:.2f}",
                f"[{color}]{s['return_pct']:.1f}%[/{color}]",
                s["reason"],
                s["details"],
            )

        console.print(table)


# ── backtest (disabled) ────────────────────────────────────────────────────────
#
# Backtesting was intentionally removed per user request.
# Keep the old CLI command disabled so it does not reappear in menus or help
# output unless the user explicitly asks for it again.
#
# @cli.command()
# @click.option("--start-date", default="2022-01-01", help="Backtest start date (YYYY-MM-DD).")
# @click.option("--end-date", default=None, help="Backtest end date (YYYY-MM-DD). Defaults to today.")
# @click.option("--capital", default=100000.0, help="Initial capital.")
# @click.pass_context
# def backtest(ctx: click.Context, start_date: str, end_date: str, capital: float) -> None:
#     ...


# ── push-sheets ────────────────────────────────────────────────────────────────
 
@cli.command(name="push-sheets")
@click.option(
    "--portfolio",
    type=click.Choice(["alpha", "all"], case_sensitive=False),
    default="all",
    help="Which portfolio to push.",
)
@click.option("--sheet-name", default="BIST Portfolio Tracker", help="Name of the Google Sheet file.")
@click.pass_context
def push_sheets(ctx: click.Context, portfolio: str, sheet_name: str) -> None:
    """Stage 5b: Push portfolio picks to Google Sheets."""
    import os
    from bist_picker.db.connection import session_scope
    from bist_picker.db.schema import Company, PortfolioSelection
    from bist_picker.output.google_sheets import GoogleSheetsClient

    # Finding service_account.json
    creds_path = "service_account.json"
    if not os.path.exists(creds_path):
        # Try config folder
        creds_path = "config/service_account.json"
        if not os.path.exists(creds_path):
             console.print("[red]Error: service_account.json not found in root or config/ folder.[/red]")
             return

    client = GoogleSheetsClient(credentials_path=creds_path)
    if not client.client:
        return

    engine = _get_engine_and_tables()
    
    with session_scope(engine) as session:
        portfolios_to_push = (
            ["alpha"]
            if portfolio.lower() == "all"
            else [portfolio.lower()]
        )

        for pname in portfolios_to_push:
            # Load most recent selections
            rows = (
                session.query(PortfolioSelection, Company)
                .join(Company, Company.id == PortfolioSelection.company_id)
                .filter(PortfolioSelection.portfolio == pname.upper())
                .order_by(
                    PortfolioSelection.selection_date.desc(),
                    PortfolioSelection.composite_score.desc(),
                )
                .limit(10)
                .all()
            )

            if not rows:
                console.print(f"[yellow]No selections found for {pname.upper()}.[/yellow]")
                continue

            most_recent_date = rows[0][0].selection_date
            # Format tab name: "Apr 2023 - Alpha"
            tab_name = f"{most_recent_date.strftime('%b %Y')} - {pname.capitalize()}"
            
            picks = []
            for rank, (pos, company) in enumerate(rows, start=1):
                if pos.selection_date != most_recent_date:
                    break
                
                picks.append({
                    "Rank": rank,
                    "Ticker": company.ticker,
                    "Company": company.name,
                    "Score": float(f"{pos.composite_score:.2f}"),
                    "Entry Price": float(f"{pos.entry_price:.2f}"),
                    "Target": float(f"{pos.target_price:.2f}"),
                    "Stop Loss": float(f"{pos.stop_loss_price:.2f}"),
                    "Sector": company.sector_custom or company.sector_bist or "",
                    "Risk": "TBD" # Could join scoring_result to get risk_tier if needed
                })

            console.print(f"[bold blue]Pushing {len(picks)} picks to '{sheet_name}' / '{tab_name}'...[/bold blue]")
            success = client.push_portfolio(sheet_name, tab_name, picks)
            if success:
                console.print(f"[green]Successfully pushed {pname.upper()}.[/green]")
            else:
                console.print(f"[red]Failed to push {pname.upper()}.[/red]")


@cli.command(name="export-mobile-snapshot")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the offline Android snapshot to this SQLite file.",
)
def export_mobile_snapshot_command(output: Path | None) -> None:
    """Export the compact offline SQLite snapshot used by the Android app."""
    from bist_picker.mobile_snapshot import (
        DEFAULT_MOBILE_SNAPSHOT_PATH,
        export_mobile_snapshot,
    )

    target_path = output or DEFAULT_MOBILE_SNAPSHOT_PATH
    exported_path = export_mobile_snapshot(target_path)
    console.print(
        f"[green]Mobile snapshot exported to[/green] [cyan]{exported_path}[/cyan]"
    )


@cli.command(name="export-mobile-feed")
@click.option(
    "--feed-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write manifest.json and mobile_snapshot.db.gz into this directory.",
)
@click.option(
    "--base-download-url",
    type=str,
    default=None,
    help="Optional base URL prepended to the published snapshot filename in manifest.json.",
)
def export_mobile_feed_command(
    feed_dir: Path | None,
    base_download_url: str | None,
) -> None:
    """Export the cloud mobile feed used by the Android auto-sync flow."""
    from bist_picker.mobile_feed import DEFAULT_FEED_DIRECTORY, export_mobile_feed

    target_dir = feed_dir or DEFAULT_FEED_DIRECTORY
    result = export_mobile_feed(
        target_dir,
        base_download_url=base_download_url,
    )
    console.print(
        "[green]Mobile feed exported:[/green] "
        f"[cyan]{result.manifest_path}[/cyan] "
        f"and [cyan]{result.snapshot_path}[/cyan]"
    )


def _mask_secret(value: str) -> str:
    """Mask secrets for display without exposing full value."""
    if not value:
        return "(not set)"
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:3]}{'*' * (len(value) - 5)}{value[-2:]}"


def _parse_tickers(raw: str) -> tuple[str, ...]:
    """Parse comma-separated tickers into a unique uppercase tuple."""
    cleaned = [t.strip().upper() for t in raw.split(",") if t.strip()]
    # Preserve order while removing duplicates.
    unique = list(dict.fromkeys(cleaned))
    return tuple(unique)


def _prompt_portfolio(default: str = "all") -> str:
    """Prompt for portfolio selection."""
    return click.prompt(
        "Portfolio",
        type=click.Choice(["alpha", "all"], case_sensitive=False),
        default=default,
        show_default=True,
    ).lower()


def _prompt_api_key() -> str:
    """Prompt for TCMB API key using a terminal-compatible visible input."""
    console.print(
        "[dim]Enter/paste TCMB EVDS API key and press Enter "
        "(input is visible in terminal).[/dim]"
    )
    try:
        return click.prompt("TCMB API key", type=str).strip()
    except (click.Abort, EOFError):
        return ""


def _ensure_tcmb_api_key() -> None:
    """Prompt user for TCMB API key if missing in current session."""
    key = os.environ.get("TCMB_API_KEY", "").strip()
    if key:
        console.print(
            f"[green]TCMB_API_KEY is set for this session:[/green] "
            f"[cyan]{_mask_secret(key)}[/cyan]"
        )
        return

    console.print(
        "[yellow]TCMB_API_KEY is not set. TCMB macro data may be unavailable.[/yellow]"
    )
    if click.confirm("Set TCMB API key now for this terminal session?", default=True):
        attempts = 0
        while attempts < 3:
            new_key = _prompt_api_key()
            if new_key:
                os.environ["TCMB_API_KEY"] = new_key
                console.print("[green]TCMB_API_KEY set for current session.[/green]")
                return
            attempts += 1
            console.print("[yellow]Empty key entered.[/yellow]")
            if attempts < 3 and not click.confirm("Try entering API key again?", default=True):
                break
        console.print("[yellow]Keeping TCMB_API_KEY unset for now.[/yellow]")


def _run_full_pipeline_for_tickers(
    ctx: click.Context, tickers: tuple[str, ...],
) -> None:
    """Run full pipeline with optional ticker filter."""
    ctx.invoke(run, ticker=tickers)


def _stage_menu(ctx: click.Context) -> None:
    """Interactive stage-by-stage menu."""
    while True:
        console.print("\n[bold]Stage Menu[/bold]")
        console.print("  1) Fetch (all)")
        console.print("  2) Fetch (selected tickers)")
        console.print("  3) Clean")
        console.print("  4) Score")
        console.print("  5) Pick")
        console.print("  6) Report (terminal)")
        console.print("  7) Report (excel)")
        console.print("  8) Historical Backfill (5yr quarterly financials)")
        console.print("  0) Back")

        choice = click.prompt(
            "Select option",
            type=click.Choice(["1", "2", "3", "4", "5", "6", "7", "8", "0"]),
        )

        if choice == "0":
            return
        if choice == "1":
            ctx.invoke(fetch, ticker=(), prices_only=False, limit=0, history=False)
        elif choice == "2":
            raw = click.prompt("Enter tickers (comma-separated, e.g. THYAO,BIMAS)")
            tickers = _parse_tickers(raw)
            if not tickers:
                console.print("[yellow]No valid tickers entered.[/yellow]")
                continue
            ctx.invoke(fetch, ticker=tickers, prices_only=False, limit=0, history=False)
        elif choice == "3":
            ctx.invoke(clean)
        elif choice == "4":
            ctx.invoke(score)
        elif choice == "5":
            ctx.invoke(pick)
        elif choice == "6":
            portfolio = _prompt_portfolio(default="all")
            ctx.invoke(report, portfolio=portfolio, format="terminal")
        elif choice == "7":
            portfolio = _prompt_portfolio(default="all")
            ctx.invoke(report, portfolio=portfolio, format="excel")
        elif choice == "8":
            console.print(
                "\n[bold]This fetches 5 years of quarterly financial data "
                "from IsYatirim.[/bold]"
            )
            console.print("[dim]This is a one-time operation — historical data doesn't change.[/dim]")
            if click.confirm("Proceed with historical backfill?", default=True):
                ctx.invoke(fetch, ticker=(), prices_only=False, limit=0, history=True)


def _daily_ops_menu(ctx: click.Context) -> None:
    """Interactive daily operations menu."""
    while True:
        console.print("\n[bold]Daily Operations[/bold]")
        console.print("  1) Status")
        console.print("  2) Inspect ticker")
        console.print("  3) Check exits")
        console.print("  4) Push to Google Sheets")
        console.print("  0) Back")

        choice = click.prompt(
            "Select option",
            type=click.Choice(["1", "2", "3", "4", "0"]),
        )

        if choice == "0":
            return
        if choice == "1":
            ctx.invoke(status)
        elif choice == "2":
            ticker = click.prompt("Ticker").strip().upper()
            if not ticker:
                console.print("[yellow]Ticker cannot be empty.[/yellow]")
                continue
            ctx.invoke(inspect, ticker=ticker)
        elif choice == "3":
            ctx.invoke(check_exits)
        elif choice == "4":
            portfolio = _prompt_portfolio(default="all")
            sheet_name = click.prompt("Google Sheet name", default="BIST Portfolio Tracker")
            ctx.invoke(push_sheets, portfolio=portfolio, sheet_name=sheet_name)


def _setup_menu() -> None:
    """Interactive setup/config menu."""
    from pathlib import Path
    from bist_picker.db.connection import get_engine

    while True:
        console.print("\n[bold]Setup and Configuration[/bold]")
        console.print("  1) Set/replace TCMB API key (session only)")
        console.print("  2) Clear TCMB API key (session)")
        console.print("  3) Show setup status")
        console.print("  4) Show command to persist API key")
        console.print("  0) Back")

        choice = click.prompt(
            "Select option",
            type=click.Choice(["1", "2", "3", "4", "0"]),
        )

        if choice == "0":
            return
        if choice == "1":
            new_key = _prompt_api_key()
            if not new_key:
                console.print("[yellow]Empty key entered. No change made.[/yellow]")
                continue
            os.environ["TCMB_API_KEY"] = new_key
            console.print("[green]TCMB_API_KEY updated for current session.[/green]")
        elif choice == "2":
            os.environ.pop("TCMB_API_KEY", None)
            console.print("[yellow]TCMB_API_KEY removed from current session.[/yellow]")
        elif choice == "3":
            key = os.environ.get("TCMB_API_KEY", "").strip()
            root_creds = Path("service_account.json").exists()
            cfg_creds = Path("config/service_account.json").exists()
            engine = get_engine()
            console.print(f"TCMB_API_KEY: [cyan]{_mask_secret(key)}[/cyan]")
            console.print(
                "Google creds file: "
                f"[cyan]root={root_creds}, config={cfg_creds}[/cyan]"
            )
            console.print(f"DB path: [cyan]{engine.url}[/cyan]")
        elif choice == "4":
            console.print(
                "PowerShell (persist): [cyan]setx TCMB_API_KEY \"YOUR_EVDS_KEY\"[/cyan]"
            )
            console.print(
                "After running setx, open a new terminal for it to take effect."
            )


@cli.command()
@click.pass_context
def menu(ctx: click.Context) -> None:
    """Interactive menu for setup, pipeline runs, and daily operations."""
    console.print("[bold blue]BIST Stock Picker - Interactive Menu[/bold blue]")
    if ctx.obj.get("dry_run", False):
        console.print("[yellow]Global --dry-run is active.[/yellow]")

    _ensure_tcmb_api_key()

    while True:
        console.print("\n[bold]Main Menu[/bold]")
        console.print("  1) Quick test run (5 tickers)")
        console.print("  2) Full run (all stocks)")
        console.print("  3) Full run (selected tickers)")
        console.print("  4) Stage-by-stage tools")
        console.print("  5) Daily operations")
        console.print("  6) Setup and configuration")
        console.print("  0) Exit")

        choice = click.prompt(
            "Select option",
            type=click.Choice(["1", "2", "3", "4", "5", "6", "0"]),
        )

        try:
            if choice == "0":
                console.print("[green]Exiting menu.[/green]")
                return
            if choice == "1":
                tickers = ("THYAO", "BIMAS", "GARAN", "SAHOL", "ASELS")
                if click.confirm(
                    f"Run full pipeline for test tickers: {', '.join(tickers)}?",
                    default=True,
                ):
                    _run_full_pipeline_for_tickers(ctx, tickers)
            elif choice == "2":
                if click.confirm(
                    "Run full pipeline for all stocks? This may take a long time.",
                    default=False,
                ):
                    _run_full_pipeline_for_tickers(ctx, ())
            elif choice == "3":
                raw = click.prompt("Enter tickers (comma-separated, e.g. THYAO,BIMAS)")
                tickers = _parse_tickers(raw)
                if not tickers:
                    console.print("[yellow]No valid tickers entered.[/yellow]")
                    continue
                if click.confirm(
                    f"Run full pipeline for: {', '.join(tickers)}?",
                    default=True,
                ):
                    _run_full_pipeline_for_tickers(ctx, tickers)
            elif choice == "4":
                _stage_menu(ctx)
            elif choice == "5":
                _daily_ops_menu(ctx)
            elif choice == "6":
                _setup_menu()
        except Exception as exc:
            console.print(f"[red]Operation failed:[/red] {exc}")


if __name__ == "__main__":
    cli()
