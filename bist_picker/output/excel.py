
import logging
import os
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    DailyPrice,
    PortfolioSelection,
    ScoringResult,
)

logger = logging.getLogger(__name__)


class ExcelReporter:
    """Generates detailed Excel reports for the BIST Stock Picker.

    Reports are saved to output/reports/ with date-stamped filenames so
    users can compare historical results side-by-side.
    """

    def __init__(self, output_dir: str = "output/reports") -> None:
        self.output_dir = output_dir

    def generate(self, session: Session, output_filename: Optional[str] = None) -> str:
        """Generate a multi-sheet Excel report.

        Sheets:
          - Summary: High-level stats (date, counts, score dist).
          - Alpha / Beta / Delta: Portfolio picks with entry/target/stop.
          - All Scores: Every scored company with all factor + composite scores.

        Args:
            session: Active DB session.
            output_filename: Override filename. Defaults to bist_report_YYYY-MM-DD.xlsx.

        Returns:
            Absolute path to the generated file.
        """
        if not output_filename:
            output_filename = f"bist_report_{date.today()}.xlsx"

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_filename)
        logger.info("Generating Excel report at %s ...", output_path)

        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                self._create_summary_sheet(writer, session)
                self._create_portfolio_sheets(writer, session)
                self._create_scores_sheet(writer, session)

            logger.info("Report generated successfully: %s", output_path)
            return output_path
        except Exception as exc:
            logger.error("Failed to generate Excel report: %s", exc)
            raise

    # ── Sheet builders ──────────────────────────────────────────────────────

    def _create_summary_sheet(self, writer: pd.ExcelWriter, session: Session) -> None:
        """Summary dashboard with key statistics."""
        scoring_date = date.today()
        total = session.query(Company).filter(Company.is_active.is_(True)).count()
        scored = (
            session.query(ScoringResult)
            .filter(ScoringResult.scoring_date == scoring_date)
            .count()
        )
        bist100 = (
            session.query(Company)
            .filter(Company.is_active.is_(True), Company.is_bist100.is_(True))
            .count()
        )

        # Portfolio pick counts
        pick_counts = {}
        # NOTE: BETA and DELTA commented out — only ALPHA portfolio is active.
        # To re-enable, uncomment "BETA" and "DELTA" in the tuple below.
        for pname in ("ALPHA",):  # "BETA", "DELTA"):
            cnt = (
                session.query(PortfolioSelection)
                .filter(
                    PortfolioSelection.portfolio == pname,
                    PortfolioSelection.selection_date == scoring_date,
                )
                .count()
            )
            pick_counts[pname] = cnt

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Report Date",
                    "Total Active Companies",
                    "BIST-100 Companies",
                    "Scored Companies",
                    "ALPHA Picks",
                    # "BETA Picks",
                    # "DELTA Picks",
                    "Engine",
                ],
                "Value": [
                    str(scoring_date),
                    total,
                    bist100,
                    scored,
                    pick_counts.get("ALPHA", 0),
                    # pick_counts.get("BETA", 0),
                    # pick_counts.get("DELTA", 0),
                    "BIST Stock Picker V2",
                ],
            }
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)
        self._auto_width(writer, "Summary", summary)

    def _create_portfolio_sheets(self, writer: pd.ExcelWriter, session: Session) -> None:
        """One sheet per portfolio with picks, prices, and key metrics."""
        scoring_date = date.today()

        # NOTE: BETA and DELTA sheets commented out — only ALPHA is generated.
        # To re-enable, uncomment "BETA" and "DELTA" in the tuple below.
        for pname in ("ALPHA",):  # "BETA", "DELTA"):
            picks = (
                session.query(PortfolioSelection, Company)
                .join(Company, Company.id == PortfolioSelection.company_id)
                .filter(
                    PortfolioSelection.portfolio == pname,
                    PortfolioSelection.selection_date == scoring_date,
                )
                .order_by(PortfolioSelection.composite_score.desc())
                .all()
            )

            if not picks:
                # Write an empty sheet with a note
                df = pd.DataFrame({"Note": ["No picks for this portfolio."]})
                df.to_excel(writer, sheet_name=pname, index=False)
                continue

            rows = []
            for sel, comp in picks:
                # Get latest scoring result for extra data
                sr = (
                    session.query(ScoringResult)
                    .filter(
                        ScoringResult.company_id == comp.id,
                        ScoringResult.scoring_date == scoring_date,
                    )
                    .first()
                )

                # Get latest adjusted metric for fundamentals
                metric = (
                    session.query(AdjustedMetric)
                    .filter(AdjustedMetric.company_id == comp.id)
                    .order_by(AdjustedMetric.period_end.desc())
                    .first()
                )

                row = {
                    "Rank": len(rows) + 1,
                    "Ticker": comp.ticker,
                    "Company Name": comp.name or "",
                    "Sector": comp.sector_custom or comp.sector_bist or "",
                    "Type": comp.company_type or "",
                    "Composite Score": _round(sel.composite_score),
                    "Entry Price (TRY)": _round(sel.entry_price),
                    "Target Price (TRY)": _round(sel.target_price),
                    "Stop Loss (TRY)": _round(sel.stop_loss_price),
                    "Risk Tier": sr.risk_tier if sr else "",
                    "Data Completeness %": _round(sr.data_completeness) if sr else None,
                    "Buffett": _round(sr.buffett_score) if sr else None,
                    "Graham": _round(sr.graham_score) if sr else None,
                    "Piotroski": _round(sr.piotroski_fscore) if sr else None,
                    "Momentum": _round(sr.momentum_score) if sr else None,
                    "ROE %": _round(metric.roe_adjusted * 100) if metric and metric.roe_adjusted else None,
                    "EPS (adj)": _round(metric.eps_adjusted) if metric else None,
                    "Free Float %": _round(comp.free_float_pct),
                    "BIST-100": "Yes" if comp.is_bist100 else "No",
                    "Selection Date": str(sel.selection_date),
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=pname, index=False)
            self._auto_width(writer, pname, df)

    def _create_scores_sheet(self, writer: pd.ExcelWriter, session: Session) -> None:
        """Full factor-score table for all scored companies, sorted by Alpha."""
        scoring_date = date.today()

        results = (
            session.query(ScoringResult, Company)
            .join(Company, Company.id == ScoringResult.company_id)
            .filter(ScoringResult.scoring_date == scoring_date)
            .all()
        )

        if not results:
            return

        rows = []
        for sr, comp in results:
            rows.append(
                {
                    "Ticker": comp.ticker,
                    "Name": comp.name or "",
                    "Type": sr.model_used or "",
                    "Sector": comp.sector_custom or comp.sector_bist or "",
                    "Buffett": _round(sr.buffett_score),
                    "Graham": _round(sr.graham_score),
                    "Piotroski": _round(sr.piotroski_fscore),
                    "MagicFormula": _round(sr.magic_formula_rank),
                    "Lynch PEG": _round(sr.lynch_peg_score),
                    "DCF MoS %": _round(sr.dcf_margin_of_safety_pct),
                    "Momentum": _round(sr.momentum_score),
                    "Insider": _round(sr.insider_score),
                    "Technical": _round(sr.technical_score),
                    "Alpha": _round(sr.composite_alpha),
                    # "Beta": _round(sr.composite_beta),   # Commented out — only ALPHA active
                    # "Delta": _round(sr.composite_delta),  # Commented out — only ALPHA active
                    "Risk Tier": sr.risk_tier or "",
                    "Data %": _round(sr.data_completeness),
                    "Free Float %": _round(comp.free_float_pct),
                    "BIST-100": "Yes" if comp.is_bist100 else "No",
                }
            )

        df = pd.DataFrame(rows)
        df.sort_values(by="Alpha", ascending=False, na_position="last", inplace=True)
        df.to_excel(writer, sheet_name="All Scores", index=False)
        self._auto_width(writer, "All Scores", df)

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_width(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
        """Auto-adjust column widths based on content length."""
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col)),
            )
            # Cap at 40 chars, add padding
            worksheet.column_dimensions[chr(65 + idx) if idx < 26 else f"A{chr(65 + idx - 26)}"].width = min(
                max_len + 3, 40
            )


def _round(value, decimals: int = 2):
    """Round a value for Excel display, returning None if input is None."""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return value
