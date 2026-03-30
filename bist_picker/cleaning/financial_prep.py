"""Financial metrics calculator for BIST Stock Picker.

Calculates adjusted financial metrics from raw financial statements stored
in the database. Handles IAS 29 monetary gain/loss stripping, owner earnings,
free cash flow, adjusted ROE/ROA/EPS, and real (inflation-adjusted) EPS growth.

Critical rules:
- For BANKS: do NOT strip monetary gain/loss
- Missing data -> set metrics to None, not 0
- Defensive parsing: try multiple item codes and label patterns
"""

import json
import logging
from datetime import date
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

from bist_picker.cleaning.inflation import InflationAdjuster, _find_item_by_codes
from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    FinancialStatement,
    MacroRegime,
)

logger = logging.getLogger("bist_picker.cleaning.financial_prep")

# ---- Item code constants (verified from real Is Yatirim data) ----

# Income statement
CODES_NET_INCOME_PARENT = ["3Z"]      # Parent Shares
CODES_NET_INCOME_TOTAL = ["3L"]       # NET PROFIT AFTER TAXES
CODES_GROSS_PROFIT = ["3D"]           # GROSS PROFIT (LOSS)
CODES_OPERATING_PROFIT = ["3DF"]      # OPERATING PROFITS
CODES_PRE_TAX_PROFIT = ["3I"]         # PROFIT BEFORE TAX
CODES_NET_SALES = ["3C"]              # Net Sales

# Supplementary (stored with income data, 4B* prefix)
CODES_DA_INCOME = ["4B"]              # Depreciation & Amortization (income supp.)

# Balance sheet
CODES_TOTAL_ASSETS = ["1BL"]          # TOTAL ASSETS
CODES_CURRENT_ASSETS = ["1A"]         # CURRENT ASSETS
CODES_CURRENT_LIABILITIES = ["2A"]    # SHORT TERM LIABILITIES
CODES_TOTAL_EQUITY = ["2N"]           # SHAREHOLDERS EQUITY
CODES_PARENT_EQUITY = ["2O"]          # Parent Shareholders Capital
CODES_SHARE_CAPITAL = ["2OA"]         # Share Capital (Odenmi Sermaye) = shares * 1 TRY par
CODES_PP_AND_E = ["1BC"]              # Property, Plant & Equipment

# Cash flow statement
CODES_CFO = ["4C"]                    # Net Cash from Operations
CODES_DA_CASHFLOW = ["4CAB"]          # Depreciation & Amortisation (cashflow)
CODES_CAPEX = ["4CAI"]                # Capital Expenditures (CapEx) - typically negative
CODES_FCF = ["4CB"]                   # Free Cash Flow (reported)
CODES_WC_CHANGE = ["4CAF"]            # Change in Working Capital

# Income statement — deferred tax
CODES_DEFERRED_TAX = ["3HA"]          # Deferred tax income/expense

# Fallback maintenance capex ratio (when insufficient history for Greenwald)
MAINTENANCE_CAPEX_RATIO = 0.70


def _estimate_maintenance_capex(
    ppe_sales_history: list[tuple[float, float]],
    current_sales_growth: float | None,
    total_capex: float,
) -> float:
    """Bruce Greenwald method: split Total CapEx into Maintenance vs Growth.

    Growth CapEx = avg(PP&E / Sales) × ΔSales.
    Maintenance CapEx = Total CapEx - Growth CapEx.

    Falls back to 70% of Total CapEx if fewer than 3 years of data.
    """
    if len(ppe_sales_history) < 3 or current_sales_growth is None:
        return total_capex * MAINTENANCE_CAPEX_RATIO  # fallback

    ratios = [
        ppe / sales
        for ppe, sales in ppe_sales_history
        if sales is not None and sales > 0 and ppe is not None and ppe > 0
    ]
    if len(ratios) < 2:
        return total_capex * MAINTENANCE_CAPEX_RATIO

    avg_ratio = sum(ratios) / len(ratios)
    growth_capex = avg_ratio * max(0.0, current_sales_growth)
    maintenance = total_capex - growth_capex

    # Floor: maintenance can't be negative or exceed total capex
    return max(0.0, min(total_capex, maintenance))


def _estimate_excess_depreciation(
    da_sales_history: list[tuple[float, float]],
    current_da: float,
    current_sales: float,
) -> float:
    """Estimate excess depreciation caused by IAS 29 asset revaluation.

    If current D&A/Sales is more than 1.5× the historical median, the excess
    is likely inflation-driven and should be added back to owner earnings.
    Returns 0.0 when there is no detectable excess.
    """
    if len(da_sales_history) < 3 or current_sales <= 0 or current_da <= 0:
        return 0.0

    historical_ratios = [
        da / sales
        for da, sales in da_sales_history
        if sales is not None and sales > 0 and da is not None and da > 0
    ]
    if len(historical_ratios) < 2:
        return 0.0

    historical_ratios.sort()
    median_ratio = historical_ratios[len(historical_ratios) // 2]
    current_ratio = current_da / current_sales

    # If current D&A ratio is >1.5x of historical median, excess is inflationary
    if current_ratio > median_ratio * 1.5 and median_ratio > 0:
        expected_da = median_ratio * current_sales
        excess = current_da - expected_da
        logger.debug(
            "Excess D&A detected: current ratio %.3f vs median %.3f, excess=%.0f",
            current_ratio, median_ratio, excess,
        )
        return max(0.0, excess)

    return 0.0


class MetricsCalculator:
    """Calculates adjusted financial metrics from raw financial statements.

    Pulls financial statement JSON from the DB, applies IAS 29 adjustments,
    and stores results in the adjusted_metrics table.

    Args:
        session: SQLAlchemy session for DB access.
        console: Rich console for output. Optional.
    """

    def __init__(
        self, session: Session, console: Optional[Console] = None
    ) -> None:
        self._session = session
        self._console = console or Console()
        self._adjuster = InflationAdjuster()
        self._cpi_series: Optional[pd.Series] = None

    def calculate_adjusted_metrics(self, company_id: int) -> int:
        """Calculate and store adjusted metrics for a single company.

        Processes all ANNUAL financial statements for the company,
        calculates adjusted metrics, and upserts into the adjusted_metrics table.

        Args:
            company_id: Database ID of the company.

        Returns:
            Number of metric rows upserted.
        """
        company = self._session.query(Company).get(company_id)
        if company is None:
            logger.warning("Company ID %d not found", company_id)
            return 0

        is_bank = (company.company_type or "").upper() == "BANK"

        # Get all annual periods that have at least an income statement
        periods = (
            self._session.query(FinancialStatement.period_end)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.period_type == "ANNUAL",
                FinancialStatement.statement_type == "INCOME",
            )
            .distinct()
            .order_by(FinancialStatement.period_end)
            .all()
        )

        if not periods:
            logger.debug("No annual income statements for company %d", company_id)
            return 0

        period_dates = [p[0] for p in periods]
        upserted = 0
        prev_eps = None
        prev_period_end = None
        prev_equity: Optional[float] = None
        prev_assets: Optional[float] = None
        prev_sales: Optional[float] = None

        # Accumulators for Greenwald CapEx and excess D&A detection
        ppe_sales_history: list[tuple[float, float]] = []
        da_sales_history: list[tuple[float, float]] = []

        for period_end in period_dates:
            income_data = self._load_statement_json(
                company_id, period_end, "INCOME"
            )
            balance_data = self._load_statement_json(
                company_id, period_end, "BALANCE"
            )
            cashflow_data = self._load_statement_json(
                company_id, period_end, "CASHFLOW"
            )

            if not income_data:
                prev_eps = None
                prev_period_end = None
                continue

            # Extract fields
            income_fields = _extract_income_fields(income_data)

            # Skip periods where all income fields are None — this happens when
            # IsYatirim returns a placeholder statement for a period not yet
            # filed (e.g. 2025/12 in February 2026, before the annual report
            # deadline).  Using an empty statement would reset prev_eps to None
            # and produce 0-composite scores for all scorers.
            key_income_fields = [
                income_fields.get("net_income_parent"),
                income_fields.get("net_income_total"),
                income_fields.get("net_sales"),
                income_fields.get("operating_profit"),
            ]
            if all(v is None for v in key_income_fields):
                logger.debug(
                    "Skipping empty period %s for company %d (all key income fields None)",
                    period_end, company_id,
                )
                continue

            balance_fields = _extract_balance_fields(balance_data) if balance_data else {}
            cashflow_fields = _extract_cashflow_fields(cashflow_data) if cashflow_data else {}

            # Get reported net income
            reported_ni = income_fields.get("net_income_parent") or income_fields.get("net_income_total")

            # Strip monetary gain/loss (skip for banks)
            if is_bank:
                adjusted_ni = reported_ni
                monetary_gl = 0.0
            else:
                adjusted_ni, monetary_gl = self._adjuster.strip_monetary_gain_loss(
                    income_data
                )

            # ── Deferred tax stripping (non-cash IAS 29 artifact) ─────────
            deferred_tax_val = income_fields.get("deferred_tax")
            deferred_tax_stripped: Optional[float] = None
            if (
                not is_bank
                and deferred_tax_val is not None
                and deferred_tax_val > 0
                and adjusted_ni is not None
            ):
                # Positive deferred_tax = income (non-cash); strip it
                deferred_tax_stripped = deferred_tax_val
                adjusted_ni = adjusted_ni - deferred_tax_val

            # D&A: prefer cashflow version, fallback to income supplementary
            da = cashflow_fields.get("da") or income_fields.get("da")

            # CapEx (typically negative from cashflow)
            capex = cashflow_fields.get("capex")
            capex_abs = abs(capex) if capex is not None else None

            # Working capital change
            wc_change = cashflow_fields.get("wc_change")

            # CFO
            cfo = cashflow_fields.get("cfo")

            # Balance sheet items
            total_equity = balance_fields.get("total_equity")
            total_assets = balance_fields.get("total_assets")
            share_capital = balance_fields.get("share_capital")
            ppe = balance_fields.get("ppe")
            net_sales = income_fields.get("net_sales")

            # Sales growth for Greenwald method
            current_sales_growth = None
            if net_sales is not None and prev_sales is not None:
                current_sales_growth = net_sales - prev_sales

            # Calculate shares outstanding from share capital (par value = 1 TRY)
            shares_outstanding = share_capital  # 1 TRY par value in Turkey

            # ── Greenwald Maintenance CapEx decomposition ─────────────────
            maint_capex = 0.0
            growth_capex_val: Optional[float] = None
            if capex_abs is not None and capex_abs > 0:
                maint_capex = _estimate_maintenance_capex(
                    ppe_sales_history, current_sales_growth, capex_abs
                )
                growth_capex_val = capex_abs - maint_capex

            # ── IAS 29 excess depreciation detection ──────────────────────
            excess_da = 0.0
            if da is not None and net_sales is not None and not is_bank:
                excess_da = _estimate_excess_depreciation(
                    da_sales_history, da, net_sales
                )

            # ── Owner Earnings ────────────────────────────────────────────
            # OE = Adj NI + D&A + excess_da_addback - Maintenance CapEx - ΔWC
            owner_earnings = None
            if adjusted_ni is not None and da is not None:
                delta_wc = wc_change if wc_change is not None else 0.0
                owner_earnings = adjusted_ni + da + excess_da - maint_capex - delta_wc

            # Free Cash Flow = CFO - Total CapEx
            free_cash_flow = None
            if cfo is not None and capex_abs is not None:
                free_cash_flow = cfo - capex_abs

            # ROE adjusted = adj NI / AVERAGE equity (beginning + ending / 2)
            roe_adjusted = None
            if adjusted_ni is not None and total_equity and total_equity != 0:
                avg_equity = (total_equity + prev_equity) / 2.0 if prev_equity else total_equity
                roe_adjusted = adjusted_ni / avg_equity

            # ROA adjusted = adj NI / AVERAGE total assets
            roa_adjusted = None
            if adjusted_ni is not None and total_assets and total_assets != 0:
                avg_assets = (total_assets + prev_assets) / 2.0 if prev_assets else total_assets
                roa_adjusted = adjusted_ni / avg_assets

            # EPS adjusted = adj NI / shares outstanding
            eps_adjusted = None
            if adjusted_ni is not None and shares_outstanding and shares_outstanding != 0:
                eps_adjusted = adjusted_ni / shares_outstanding

            # Real EPS growth (inflation-adjusted YoY)
            real_eps_growth = None
            if eps_adjusted is not None and prev_eps is not None and prev_period_end is not None:
                cpi = self._get_cpi_series()
                real_eps_growth = self._adjuster.calculate_real_growth(
                    eps_adjusted, prev_eps, period_end, prev_period_end, cpi
                )

            # ── Update accumulators for next iteration ────────────────────
            prev_eps = eps_adjusted
            prev_period_end = period_end
            prev_equity = total_equity
            prev_assets = total_assets
            prev_sales = net_sales

            # Accumulate PP&E/Sales and D&A/Sales history for Greenwald and excess D&A
            if ppe is not None and net_sales is not None:
                ppe_sales_history.append((ppe, net_sales))
            if da is not None and net_sales is not None:
                da_sales_history.append((da, net_sales))

            # Upsert into adjusted_metrics
            existing = (
                self._session.query(AdjustedMetric)
                .filter(
                    AdjustedMetric.company_id == company_id,
                    AdjustedMetric.period_end == period_end,
                )
                .first()
            )

            if existing:
                existing.reported_net_income = reported_ni
                existing.monetary_gain_loss = monetary_gl
                existing.adjusted_net_income = adjusted_ni
                existing.owner_earnings = owner_earnings
                existing.free_cash_flow = free_cash_flow
                existing.roe_adjusted = roe_adjusted
                existing.roa_adjusted = roa_adjusted
                existing.eps_adjusted = eps_adjusted
                existing.real_eps_growth_pct = real_eps_growth
                existing.maintenance_capex = maint_capex if capex_abs else None
                existing.growth_capex = growth_capex_val
                existing.deferred_tax_stripped = deferred_tax_stripped
                existing.excess_depreciation_addback = excess_da if excess_da > 0 else None
            else:
                metric = AdjustedMetric(
                    company_id=company_id,
                    period_end=period_end,
                    reported_net_income=reported_ni,
                    monetary_gain_loss=monetary_gl,
                    adjusted_net_income=adjusted_ni,
                    owner_earnings=owner_earnings,
                    free_cash_flow=free_cash_flow,
                    roe_adjusted=roe_adjusted,
                    roa_adjusted=roa_adjusted,
                    eps_adjusted=eps_adjusted,
                    real_eps_growth_pct=real_eps_growth,
                    maintenance_capex=maint_capex if capex_abs else None,
                    growth_capex=growth_capex_val,
                    deferred_tax_stripped=deferred_tax_stripped,
                    excess_depreciation_addback=excess_da if excess_da > 0 else None,
                )
                self._session.add(metric)

            upserted += 1

        if upserted > 0:
            self._session.flush()

        return upserted

    def calculate_all(self) -> dict:
        """Calculate adjusted metrics for all active companies with financials.

        Returns:
            Stats dict: {total, calculated, skipped, errors}.
        """
        # Get all active companies that have at least one financial statement
        companies = (
            self._session.query(Company)
            .filter(Company.is_active.is_(True))
            .join(FinancialStatement)
            .distinct()
            .all()
        )

        stats = {"total": len(companies), "calculated": 0, "skipped": 0, "errors": 0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Calculating metrics", total=len(companies))

            for company in companies:
                progress.update(task, description=f"Metrics: {company.ticker}")
                try:
                    count = self.calculate_adjusted_metrics(company.id)
                    if count > 0:
                        stats["calculated"] += 1
                    else:
                        stats["skipped"] += 1
                except Exception:
                    logger.exception("Error calculating metrics for %s", company.ticker)
                    stats["errors"] += 1
                finally:
                    progress.advance(task)

        self._console.print(
            f"[green]Metrics calculated:[/green] {stats['calculated']} companies, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )
        return stats

    def _load_statement_json(
        self, company_id: int, period_end: date, statement_type: str
    ) -> Optional[list[dict]]:
        """Load and parse the data_json from a financial statement.

        Args:
            company_id: Company database ID.
            period_end: Statement period end date.
            statement_type: INCOME, BALANCE, or CASHFLOW.

        Returns:
            Parsed JSON list, or None if not found.
        """
        stmt = (
            self._session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.period_end == period_end,
                FinancialStatement.period_type == "ANNUAL",
                FinancialStatement.statement_type == statement_type,
            )
            .order_by(FinancialStatement.version.desc())
            .first()
        )

        if stmt is None or not stmt.data_json:
            return None

        try:
            return json.loads(stmt.data_json)
        except json.JSONDecodeError:
            logger.warning(
                "Invalid JSON in %s statement for company %d, period %s",
                statement_type, company_id, period_end,
            )
            return None

    def _get_cpi_series(self) -> Optional[pd.Series]:
        """Load CPI data from the macro_regime table.

        Returns:
            pandas Series with date index and CPI YoY % values,
            or None if no data is available.
        """
        if self._cpi_series is not None:
            return self._cpi_series

        rows = (
            self._session.query(MacroRegime.date, MacroRegime.cpi_yoy_pct)
            .filter(MacroRegime.cpi_yoy_pct.isnot(None))
            .order_by(MacroRegime.date)
            .all()
        )

        if not rows:
            logger.debug("No CPI data in macro_regime table")
            return None

        dates = [r[0] for r in rows]
        values = [r[1] for r in rows]
        self._cpi_series = pd.Series(
            values, index=pd.DatetimeIndex(dates), name="cpi_yoy_pct"
        )
        return self._cpi_series


def _extract_income_fields(data: list[dict]) -> dict:
    """Extract key fields from income statement JSON.

    Args:
        data: Parsed income statement JSON items.

    Returns:
        Dict with extracted values (may contain None).
    """
    return {
        "net_income_parent": _find_item_by_codes(data, CODES_NET_INCOME_PARENT),
        "net_income_total": _find_item_by_codes(data, CODES_NET_INCOME_TOTAL),
        "gross_profit": _find_item_by_codes(data, CODES_GROSS_PROFIT),
        "operating_profit": _find_item_by_codes(data, CODES_OPERATING_PROFIT),
        "pre_tax_profit": _find_item_by_codes(data, CODES_PRE_TAX_PROFIT),
        "net_sales": _find_item_by_codes(data, CODES_NET_SALES),
        "da": _find_item_by_codes(data, CODES_DA_INCOME),
        "deferred_tax": _find_item_by_codes(data, CODES_DEFERRED_TAX),
    }


def _extract_balance_fields(data: list[dict]) -> dict:
    """Extract key fields from balance sheet JSON.

    Args:
        data: Parsed balance sheet JSON items.

    Returns:
        Dict with extracted values (may contain None).
    """
    return {
        "total_assets": _find_item_by_codes(data, CODES_TOTAL_ASSETS),
        "current_assets": _find_item_by_codes(data, CODES_CURRENT_ASSETS),
        "current_liabilities": _find_item_by_codes(data, CODES_CURRENT_LIABILITIES),
        "total_equity": _find_item_by_codes(data, CODES_TOTAL_EQUITY),
        "parent_equity": _find_item_by_codes(data, CODES_PARENT_EQUITY),
        "share_capital": _find_item_by_codes(data, CODES_SHARE_CAPITAL),
        "ppe": _find_item_by_codes(data, CODES_PP_AND_E),
    }


def _extract_cashflow_fields(data: list[dict]) -> dict:
    """Extract key fields from cash flow statement JSON.

    Args:
        data: Parsed cash flow statement JSON items.

    Returns:
        Dict with extracted values (may contain None).
    """
    return {
        "cfo": _find_item_by_codes(data, CODES_CFO),
        "da": _find_item_by_codes(data, CODES_DA_CASHFLOW),
        "capex": _find_item_by_codes(data, CODES_CAPEX),
        "fcf": _find_item_by_codes(data, CODES_FCF),
        "wc_change": _find_item_by_codes(data, CODES_WC_CHANGE),
    }
