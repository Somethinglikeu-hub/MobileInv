"""LLM-powered stock report generator for the BIST Stock Picker.

Generates concise, human-readable investment reports for each portfolio
pick using Gemini. Each report includes an investment thesis, key risks,
forward outlook, and a conviction rating.

Usage::

    from bist_picker.output.llm_report import LLMReportGenerator
    gen = LLMReportGenerator()
    reports = gen.generate_portfolio_reports("ALPHA", session)
    gen.display_reports(reports)

Token budget: ~500 input + ~300 output per stock = ~15 calls/month total.
"""

import json
import logging
from datetime import date
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
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

# Verdict color mapping
_VERDICT_COLORS = {
    "STRONG_CONVICTION": "bold green",
    "CONVICTION": "green",
    "MODERATE": "yellow",
    "CAUTIOUS": "bold red",
}


class LLMReportGenerator:
    """Generates LLM-powered investment reports for portfolio picks.

    Args:
        console: Rich Console for terminal display. Created if omitted.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self._console = console or Console(force_terminal=True, legacy_windows=False)
        self._analyzer = None

    def _get_analyzer(self):
        """Lazy-init the LLM analyzer to avoid import errors when not needed."""
        if self._analyzer is None:
            from bist_picker.data.sources.llm_analyzer import LLMAnalyzer
            self._analyzer = LLMAnalyzer()
        return self._analyzer

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_portfolio_reports(
        self,
        portfolio: str,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> list[dict]:
        """Generate LLM reports for all picks in a portfolio.

        Args:
            portfolio: One of 'ALPHA', 'BETA', 'DELTA'.
            session: Active SQLAlchemy session.
            scoring_date: Date to look up picks. Defaults to most recent.

        Returns:
            List of report dicts with keys: ticker, company_name, score,
            entry_price, target_price, stop_loss, llm_report (the parsed
            LLM response), data_payload (the input sent to LLM).
        """
        portfolio = portfolio.upper()
        picks = self._load_picks(portfolio, session, scoring_date)

        if not picks:
            logger.warning("No picks found for %s", portfolio)
            return []

        reports = []
        analyzer = self._get_analyzer()

        for i, pick_data in enumerate(picks):
            ticker = pick_data["ticker"]
            logger.info("Generating LLM report for %s (%s)", ticker, portfolio)

            # Rate limit: wait between calls (free tier = 5 RPM)
            if i > 0:
                import time
                time.sleep(13)  # ~4.6 calls/min, safely under 5 RPM

            # Build compact data payload for LLM
            payload = self._build_data_payload(pick_data)
            payload_text = self._format_payload_for_llm(payload)

            # Get the stock_report prompt
            system_prompt = analyzer._prompts.get("stock_report", "")
            if not system_prompt:
                logger.error("stock_report prompt not found in llm_config.yaml")
                continue

            # Call LLM with higher token limit for full reports
            llm_result = analyzer._call_llm(
                system_prompt, payload_text, max_output_tokens=2048,
                task_type="stock_report",
            )

            report = {
                "ticker": ticker,
                "company_name": pick_data.get("company_name", ticker),
                "portfolio": portfolio,
                "score": pick_data.get("score"),
                "entry_price": pick_data.get("entry_price"),
                "target_price": pick_data.get("target_price"),
                "stop_loss": pick_data.get("stop_loss"),
                "llm_report": llm_result,
                "data_payload": payload,
            }
            reports.append(report)

        usage = analyzer.get_usage_stats()
        logger.info(
            "Generated %d reports for %s (requests used: %d, tokens: %d in / %d out)",
            len(reports), portfolio,
            usage["requests_made"],
            usage["total_input_tokens"],
            usage["total_output_tokens"],
        )
        return reports

    def generate_all_reports(self, session: Session) -> dict[str, list[dict]]:
        """Generate reports for all three portfolios.

        Returns:
            Dict with keys 'ALPHA', 'BETA', 'DELTA', each a list of reports.
        """
        results = {}
        # NOTE: BETA and DELTA commented out — only ALPHA portfolio is active.
        # To re-enable, uncomment "BETA" and "DELTA" in the tuple below.
        for portfolio in ("ALPHA",):  # "BETA", "DELTA"):
            results[portfolio] = self.generate_portfolio_reports(portfolio, session)
        return results

    def display_reports(self, reports: list[dict]) -> None:
        """Display LLM reports in a beautifully formatted terminal output."""
        if not reports:
            self._console.print("[yellow]No reports to display.[/yellow]")
            return

        portfolio = reports[0].get("portfolio", "PORTFOLIO")
        self._console.print()
        self._console.print(
            f"[bold cyan]{'=' * 60}[/bold cyan]"
        )
        self._console.print(
            f"[bold white]  {portfolio} PORTFOLIO -- AI Analysis Report[/bold white]"
        )
        self._console.print(
            f"[bold cyan]{'=' * 60}[/bold cyan]"
        )
        self._console.print()

        for report in reports:
            self._display_single_report(report)

        # Usage summary
        try:
            usage = self._get_analyzer().get_usage_stats()
            self._console.print(
                f"[dim]LLM usage: {usage['requests_made']} requests, "
                f"{usage['total_input_tokens']} input / "
                f"{usage['total_output_tokens']} output tokens "
                f"({usage['model']})[/dim]"
            )
        except Exception:
            pass
        self._console.print()

    def display_all_reports(self, all_reports: dict[str, list[dict]]) -> None:
        """Display reports for all portfolios."""
        # NOTE: BETA and DELTA commented out — only ALPHA portfolio is shown.
        # To re-enable, uncomment "BETA" and "DELTA" in the tuple below.
        for portfolio in ("ALPHA",):  # "BETA", "DELTA"):
            reports = all_reports.get(portfolio, [])
            if reports:
                self.display_reports(reports)

    # ── Private: data loading ─────────────────────────────────────────────────

    def _load_picks(
        self,
        portfolio: str,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> list[dict]:
        """Load portfolio picks with all associated data from DB."""
        query = (
            session.query(PortfolioSelection, Company)
            .join(Company, Company.id == PortfolioSelection.company_id)
            .filter(PortfolioSelection.portfolio == portfolio)
        )

        if scoring_date:
            query = query.filter(PortfolioSelection.selection_date == scoring_date)

        rows = (
            query.order_by(
                PortfolioSelection.selection_date.desc(),
                PortfolioSelection.composite_score.desc(),
            )
            .limit(10)
            .all()
        )

        if not rows:
            return []

        # Only use picks from the most recent selection_date
        most_recent = rows[0][0].selection_date
        picks = []
        for pos, company in rows:
            if pos.selection_date != most_recent:
                break

            # Load scoring data
            score_row = (
                session.query(ScoringResult)
                .filter(ScoringResult.company_id == company.id)
                .order_by(ScoringResult.scoring_date.desc())
                .first()
            )

            # Load financial metrics
            metric = (
                session.query(AdjustedMetric)
                .filter(AdjustedMetric.company_id == company.id)
                .order_by(AdjustedMetric.period_end.desc())
                .first()
            )

            # Load price context (52-week high/low)
            price_ctx = self._get_price_context(company.id, session)

            picks.append({
                "company_id": company.id,
                "ticker": company.ticker,
                "company_name": company.name,
                "company_type": company.company_type,
                "sector_bist": company.sector_bist,
                "sector_custom": company.sector_custom,
                "is_bist100": company.is_bist100,
                "free_float_pct": company.free_float_pct,
                "score": pos.composite_score,
                "entry_price": pos.entry_price,
                "target_price": pos.target_price,
                "stop_loss": pos.stop_loss_price,
                "scoring": self._extract_scoring(score_row),
                "metrics": self._extract_metrics(metric),
                "price_ctx": price_ctx,
            })

        return picks

    def _get_price_context(self, company_id: int, session: Session) -> dict:
        """Get 52-week high, low, and latest price for context."""
        from datetime import timedelta

        cutoff = date.today() - timedelta(days=365)
        prices = (
            session.query(DailyPrice.close, DailyPrice.date)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.date >= cutoff,
                DailyPrice.close.isnot(None),
            )
            .order_by(DailyPrice.date.desc())
            .all()
        )

        if not prices:
            return {}

        closes = [p.close for p in prices]
        return {
            "latest_price": closes[0],
            "high_52w": max(closes),
            "low_52w": min(closes),
            "latest_date": str(prices[0].date),
        }

    def _extract_scoring(self, score_row) -> dict:
        """Extract key scoring fields into a compact dict."""
        if score_row is None:
            return {}
        return {
            "buffett": score_row.buffett_score,
            "graham": score_row.graham_score,
            "piotroski": score_row.piotroski_fscore,
            "magic_formula": score_row.magic_formula_rank,
            "lynch_peg": score_row.lynch_peg_score,
            "dcf_mos": score_row.dcf_margin_of_safety_pct,
            "momentum": score_row.momentum_score,
            "risk_tier": score_row.risk_tier,
            "data_completeness": score_row.data_completeness,
            "model_used": score_row.model_used,
            "composite_alpha": score_row.composite_alpha,
            "composite_beta": score_row.composite_beta,
            "composite_delta": score_row.composite_delta,
        }

    def _extract_metrics(self, metric) -> dict:
        """Extract key financial metrics into a compact dict."""
        if metric is None:
            return {}
        return {
            "period": str(metric.period_end) if metric.period_end else None,
            "net_income": metric.adjusted_net_income,
            "owner_earnings": metric.owner_earnings,
            "fcf": metric.free_cash_flow,
            "roe": metric.roe_adjusted,
            "roa": metric.roa_adjusted,
            "eps": metric.eps_adjusted,
            "real_eps_growth": metric.real_eps_growth_pct,
            "related_party_pct": metric.related_party_revenue_pct,
        }

    # ── Private: payload formatting ───────────────────────────────────────────

    def _build_data_payload(self, pick: dict) -> dict:
        """Build a compact data payload for the LLM. Keeps token count low."""
        scoring = pick.get("scoring", {})
        metrics = pick.get("metrics", {})
        price_ctx = pick.get("price_ctx", {})

        # Compute upside/downside from entry
        entry = pick.get("entry_price")
        target = pick.get("target_price")
        stop = pick.get("stop_loss")

        upside_pct = None
        if entry and target and entry > 0:
            upside_pct = round((target - entry) / entry * 100, 1)

        downside_pct = None
        if entry and stop and entry > 0:
            downside_pct = round((stop - entry) / entry * 100, 1)

        # Position vs 52-week range
        pct_from_high = None
        pct_from_low = None
        if price_ctx.get("latest_price") and price_ctx.get("high_52w"):
            latest = price_ctx["latest_price"]
            high = price_ctx["high_52w"]
            low = price_ctx["low_52w"]
            if high > 0:
                pct_from_high = round((latest - high) / high * 100, 1)
            if low > 0:
                pct_from_low = round((latest - low) / low * 100, 1)

        # Compute P/E from entry and EPS
        pe_ratio = None
        if entry and metrics.get("eps") and metrics["eps"] > 0:
            pe_ratio = round(entry / metrics["eps"], 1)

        return {
            "ticker": pick["ticker"],
            "name": pick.get("company_name"),
            "type": pick.get("company_type"),
            "sector": pick.get("sector_custom") or pick.get("sector_bist"),
            "bist100": pick.get("is_bist100"),
            "free_float": pick.get("free_float_pct"),
            "composite_score": pick.get("score"),
            "entry_price": entry,
            "target_price": target,
            "upside_pct": upside_pct,
            "stop_loss": stop,
            "downside_pct": downside_pct,
            "pct_from_52w_high": pct_from_high,
            "pct_from_52w_low": pct_from_low,
            "risk_tier": scoring.get("risk_tier"),
            "factor_scores": {
                "buffett_quality": scoring.get("buffett"),
                "graham_value": scoring.get("graham"),
                "piotroski_fscore": scoring.get("piotroski"),
                "magic_formula": scoring.get("magic_formula"),
                "lynch_peg": scoring.get("lynch_peg"),
                "dcf_margin_of_safety": scoring.get("dcf_mos"),
                "momentum": scoring.get("momentum"),
            },
            "financials": {
                "period": metrics.get("period"),
                "adj_net_income_try": self._compact_num(metrics.get("net_income")),
                "owner_earnings_try": self._compact_num(metrics.get("owner_earnings")),
                "free_cash_flow_try": self._compact_num(metrics.get("fcf")),
                "roe_pct": self._round_or_none(metrics.get("roe"), mult=100),
                "roa_pct": self._round_or_none(metrics.get("roa"), mult=100),
                "eps_try": self._round_or_none(metrics.get("eps")),
                "pe_ratio": pe_ratio,
                "real_eps_growth_pct": self._round_or_none(metrics.get("real_eps_growth"), mult=100),
                "related_party_revenue_pct": self._round_or_none(metrics.get("related_party_pct"), mult=100),
            },
            "data_completeness": scoring.get("data_completeness"),
        }

    def _format_payload_for_llm(self, payload: dict) -> str:
        """Format the data payload as a compact text block for the LLM."""
        lines = []
        lines.append(f"STOCK: {payload['ticker']} ({payload.get('name', 'N/A')})")
        lines.append(f"Type: {payload.get('type', 'N/A')} | Sector: {payload.get('sector', 'N/A')}")
        lines.append(f"BIST-100: {'Yes' if payload.get('bist100') else 'No'} | Free Float: {payload.get('free_float', 'N/A')}%")
        lines.append(f"Risk Tier: {payload.get('risk_tier', 'N/A')} | Data Completeness: {payload.get('data_completeness', 'N/A')}%")
        lines.append("")
        lines.append(f"COMPOSITE SCORE: {payload.get('composite_score', 'N/A')}/100")
        lines.append(f"Entry: TRY {payload.get('entry_price', 'N/A')} | Target: TRY {payload.get('target_price', 'N/A')} ({payload.get('upside_pct', 'N/A')}% upside)")
        lines.append(f"Stop Loss: TRY {payload.get('stop_loss', 'N/A')} ({payload.get('downside_pct', 'N/A')}% downside)")
        lines.append(f"52-week: {payload.get('pct_from_52w_high', 'N/A')}% from high, +{payload.get('pct_from_52w_low', 'N/A')}% from low")
        lines.append("")

        # Factor scores
        factors = payload.get("factor_scores", {})
        factor_parts = []
        for name, val in factors.items():
            if val is not None:
                factor_parts.append(f"{name}={val:.0f}")
        if factor_parts:
            lines.append("FACTOR SCORES: " + ", ".join(factor_parts))

        # Financials
        fin = payload.get("financials", {})
        lines.append("")
        lines.append(f"FINANCIALS (period: {fin.get('period', 'N/A')}):")
        fin_parts = []
        for key in ["adj_net_income_try", "owner_earnings_try", "free_cash_flow_try"]:
            val = fin.get(key)
            if val is not None:
                label = key.replace("_try", "").replace("_", " ").title()
                fin_parts.append(f"  {label}: TRY {val}")
        for part in fin_parts:
            lines.append(part)

        ratio_parts = []
        for key in ["roe_pct", "roa_pct", "eps_try", "pe_ratio", "real_eps_growth_pct", "related_party_revenue_pct"]:
            val = fin.get(key)
            if val is not None:
                label = key.replace("_pct", "%").replace("_try", " TRY").replace("_", " ").upper()
                ratio_parts.append(f"{label}={val}")
        if ratio_parts:
            lines.append("  " + " | ".join(ratio_parts))

        return "\n".join(lines)

    # ── Private: display ──────────────────────────────────────────────────────

    def _display_single_report(self, report: dict) -> None:
        """Render a single stock report as a Rich panel."""
        ticker = report["ticker"]
        name = report.get("company_name") or ticker
        score = report.get("score")
        entry = report.get("entry_price")
        target = report.get("target_price")
        stop = report.get("stop_loss")
        llm = report.get("llm_report")

        # Header line
        score_str = f"{score:.1f}" if score is not None else "--"
        entry_str = f"TRY {entry:,.2f}" if entry else "--"
        target_str = f"TRY {target:,.2f}" if target else "--"
        stop_str = f"TRY {stop:,.2f}" if stop else "--"

        header = (
            f"[bold white]{ticker}[/bold white] - {name}\n"
            f"Score: [bold cyan]{score_str}[/bold cyan] | "
            f"Entry: {entry_str} | Target: {target_str} | Stop: {stop_str}"
        )

        # LLM content
        if llm is None:
            body = "[dim]LLM analysis unavailable.[/dim]"
            border_color = "dim"
        else:
            verdict = llm.get("verdict", "N/A")
            llm_score = llm.get("score_out_of_10")
            thesis = llm.get("thesis", "No thesis available.")
            risks = llm.get("risks", [])
            outlook = llm.get("outlook", "No outlook available.")

            verdict_color = _VERDICT_COLORS.get(verdict, "white")
            score_10 = f"{llm_score:.1f}/10" if llm_score else "--"

            parts = []
            parts.append(f"\n[bold]Verdict:[/bold] [{verdict_color}]{verdict}[/{verdict_color}] ({score_10})\n")
            parts.append(f"[bold]Thesis:[/bold] {thesis}\n")

            if risks:
                parts.append("[bold]Risks:[/bold]")
                for risk in risks:
                    parts.append(f"  - {risk}")
                parts.append("")

            parts.append(f"[bold]Outlook:[/bold] {outlook}")

            body = "\n".join(parts)
            border_color = verdict_color.replace("bold ", "")

        self._console.print(
            Panel(
                f"{header}\n{body}",
                border_style=border_color,
                padding=(1, 2),
            )
        )

    # ── Private: utilities ────────────────────────────────────────────────────

    @staticmethod
    def _compact_num(value: Optional[float]) -> Optional[str]:
        """Format large numbers compactly: 1,234,567,890 -> '1.23B'."""
        if value is None:
            return None
        abs_val = abs(value)
        sign = "-" if value < 0 else ""
        if abs_val >= 1_000_000_000:
            return f"{sign}{abs_val / 1_000_000_000:.2f}B"
        if abs_val >= 1_000_000:
            return f"{sign}{abs_val / 1_000_000:.1f}M"
        if abs_val >= 1_000:
            return f"{sign}{abs_val / 1_000:.0f}K"
        return f"{sign}{abs_val:.0f}"

    @staticmethod
    def _round_or_none(value: Optional[float], decimals: int = 1, mult: float = 1.0) -> Optional[float]:
        """Round a value, optionally multiplying first (e.g. 0.15 * 100 = 15.0%)."""
        if value is None:
            return None
        return round(value * mult, decimals)
