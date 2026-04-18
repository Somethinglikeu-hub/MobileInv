"""Data fetcher orchestrator for BIST Stock Picker.

Coordinates all data sources (IsYatirim, KAP, TCMB, Yahoo) and writes
fetched data into the SQLite database. Used by the `bist fetch` CLI command.
"""

import json
import logging
import os
import random
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import concurrent.futures
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from sqlalchemy.orm import Session

from bist_picker.data.sources.isyatirim import IsYatirimClient
from bist_picker.data.sources.kap import KAPClient
from bist_picker.data.sources.tcmb import TCMBClient
from bist_picker.data.sources.yahoo import YahooClient
from bist_picker.db.schema import (
    Company,
    DailyPrice,
    FinancialStatement,
    MacroRegime,
)
from bist_picker.utils.rate_limiter import RateLimiter

logger = logging.getLogger("bist_picker.data.fetcher")

_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
_BENCHMARK_TICKER = "XU100"
_BENCHMARK_NAME = "BIST 100"


def _load_settings() -> dict:
    """Load settings.yaml configuration."""
    if _SETTINGS_PATH.exists():
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class _PipeProgress:
    """Minimal Progress-like wrapper for pipe/subprocess mode.

    Uses print()+flush instead of Rich progress bars, which don't render
    correctly when stdout is piped on Windows. Output goes directly to
    stdout so the parent process (dashboard) can read it in real-time.
    """

    def __init__(self, description: str) -> None:
        self._description = description
        self._tasks: dict[int, dict] = {}
        self._next_id = 0

    def _emit(self, msg: str) -> None:
        """Print a line and immediately flush stdout."""
        import sys
        print(msg, flush=True)
        sys.stdout.flush()

    def __enter__(self) -> "_PipeProgress":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def add_task(self, name: str, total: int = 0) -> int:
        """Register a new task and return its ID."""
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = {"name": name, "completed": 0, "total": total}
        self._emit(f"{self._description} — {name}: 0/{total}")
        return task_id

    def advance(self, task_id: int, advance: int = 1) -> None:
        """Advance a task's completed count."""
        task = self._tasks.get(task_id)
        if task is None:
            return
        task["completed"] += advance
        done, total = task["completed"], task["total"]
        # Emit every ~10% or at completion
        if done == total or done % max(total // 10, 1) == 0:
            pct = int(done * 100 / total) if total else 0
            self._emit(f"{self._description} — {task['name']}: {done}/{total} ({pct}%)")

    def update(self, task_id: int, **kwargs) -> None:
        """Update task attributes (for compatibility)."""
        task = self._tasks.get(task_id)
        if task is None:
            return
        if "completed" in kwargs:
            task["completed"] = kwargs["completed"]
        if "total" in kwargs:
            task["total"] = kwargs["total"]


class DataFetcher:
    """Orchestrates data fetching from all sources into the database.

    Args:
        session: SQLAlchemy database session.
        console: Rich Console for output. Creates one if not provided.
    """

    def __init__(
        self,
        session: Session,
        console: Optional[Console] = None,
    ) -> None:
        self._session = session
        self._console = console or Console()
        settings = _load_settings()

        rate_limits = settings.get("rate_limits", {})
        isy_delay = rate_limits.get("isyatirim_delay_sec", 1.0)
        kap_delay = rate_limits.get("kap_delay_sec", 2.0)
        tcmb_delay = rate_limits.get("tcmb_delay_sec", 1.0)
        yahoo_delay = rate_limits.get("yahoo_delay_sec", 0.5)

        self._isy = IsYatirimClient(
            rate_limiter=RateLimiter(min_delay=isy_delay, name="isyatirim")
        )
        self._kap = KAPClient(
            rate_limiter=RateLimiter(min_delay=kap_delay, name="kap")
        )
        self._tcmb = TCMBClient(
            rate_limiter=RateLimiter(min_delay=tcmb_delay, name="tcmb")
        )
        self._yahoo = YahooClient(
            rate_limiter=RateLimiter(min_delay=yahoo_delay, name="yahoo")
        )

        self._fetch_settings = settings.get("fetch", {})

    def fetch_universe(self) -> dict:
        """Fetch company universe and update the companies table.

        Builds the union of the IsYatirim and KAP company lists so the
        database is not limited to whichever upstream source is smaller.
        IsYatirim remains the primary source for sector/free-float data,
        while KAP contributes additional listed tickers and company names.

        Returns:
            Stats dict: {total, new, updated, delisted, bist100_count}.
        """
        self._console.print("[bold]Fetching company universe...[/bold]")
        stats = {"total": 0, "new": 0, "updated": 0, "delisted": 0, "bist100_count": 0}

        # Fetch from IsYatirim
        overview_df = self._isy.fetch_company_overview()
        bist100_tickers = set(self._isy.fetch_bist100_tickers())

        # Fetch from KAP for enrichment
        kap_df = self._kap.fetch_company_list()
        kap_lookup: dict[str, dict] = {}
        if not kap_df.empty:
            for _, row in kap_df.iterrows():
                kap_lookup[row["ticker"]] = row.to_dict()

        overview_lookup: dict[str, dict] = {}
        if not overview_df.empty:
            for _, row in overview_df.iterrows():
                ticker = (row.get("ticker") or "").strip().upper()
                if ticker:
                    overview_lookup[ticker] = row.to_dict()

        all_tickers = sorted(set(overview_lookup) | set(kap_lookup))
        if not all_tickers:
            self._console.print(
                "[red]Failed to fetch company universe from both IsYatirim and KAP[/red]"
            )
            return stats

        # Get existing companies from DB
        existing = {
            c.ticker: c
            for c in self._session.query(Company).all()
        }
        seen_tickers: set[str] = set()

        for ticker in all_tickers:
            seen_tickers.add(ticker)

            row = overview_lookup.get(ticker, {})
            kap_data = kap_lookup.get(ticker, {})
            is_bist100 = ticker in bist100_tickers
            free_float_pct = row.get("free_float_pct")
            sector_bist = row.get("sector")
            company_name = kap_data.get("name") or row.get("name", "")

            if ticker in existing:
                # Update existing company
                company = existing[ticker]
                company.name = company_name or company.name
                company.sector_bist = sector_bist or company.sector_bist
                if free_float_pct is not None:
                    company.free_float_pct = free_float_pct
                company.is_bist100 = is_bist100
                company.is_active = True
                stats["updated"] += 1
            else:
                # Insert new company
                company = Company(
                    ticker=ticker,
                    name=company_name,
                    sector_bist=sector_bist or "",
                    free_float_pct=free_float_pct,
                    is_bist100=is_bist100,
                    is_active=True,
                )
                self._session.add(company)
                stats["new"] += 1

            if is_bist100:
                stats["bist100_count"] += 1

        # Mark delisted companies
        for ticker, company in existing.items():
            if ticker not in seen_tickers and company.is_active:
                company.is_active = False
                stats["delisted"] += 1

        self._session.flush()
        stats["total"] = len(seen_tickers)

        self._console.print(
            f"  Universe: {stats['total']} companies "
            f"({stats['new']} new, {stats['updated']} updated, "
            f"{stats['delisted']} delisted, {stats['bist100_count']} in BIST 100)"
        )
        return stats

    def fetch_prices(
        self,
        tickers: Optional[list[str]] = None,
        days_back: int = 730,
    ) -> dict:
        """Fetch historical price data and store in daily_prices table.
        Uses ThreadPoolExecutor for parallel fetching.
        """
        ticker_list = tickers or self._get_active_tickers()
        if not ticker_list:
            self._console.print("[yellow]No tickers to fetch prices for[/yellow]")
            return {"tickers_processed": 0, "rows_inserted": 0, "rows_skipped": 0, "failed": []}

        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        stats = {"tickers_processed": 0, "rows_inserted": 0, "rows_skipped": 0, "failed": []}

        # Max workers for IsYatirim 
        max_workers = 8 
        
        # Pre-fetch existing companies to avoid DB lookups in threads
        # We need company IDs.
        companies = {
            c.ticker: c.id 
            for c in self._session.query(Company).filter(Company.ticker.in_(ticker_list)).all()
        }
        
        # Helper for thread execution
        def fetch_one(tsk_ticker):
            try:
                # If company missing (shouldn't happen if we fetched universe), skip or handle
                cid = companies.get(tsk_ticker)
                if not cid:
                    # Try to get or create inside thread (careful with session)
                    # For safety, skipping concurrent creation here. 
                    return tsk_ticker, None, "No Company ID"
                
                df = self._isy.fetch_price_data(tsk_ticker, start_date, end_date)
                if df is None or df.empty:
                    df = self._yahoo.fetch_price_data(tsk_ticker, start_date, end_date)
                return tsk_ticker, df, None
            except Exception as e:
                return tsk_ticker, None, str(e)

        with self._progress_bar("Fetching prices (Parallel)") as progress:
            task = progress.add_task("Prices", total=len(ticker_list))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_ticker = {
                    executor.submit(fetch_one, t): t for t in ticker_list
                }
                
                batch_data = []
                BATCH_SIZE = 50  # Number of tickers to batch before insert

                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker, df, error = future.result()
                    
                    if error:
                        logger.warning("Failed to fetch prices for %s: %s", ticker, error)
                        stats["failed"].append(ticker)
                    elif df is not None and not df.empty:
                        # Prepare data for batch insert
                        count = self._prepare_prices_batch(companies[ticker], df, batch_data)
                        stats["rows_inserted"] += count
                    else:
                        stats["failed"].append(ticker)
                    
                    stats["tickers_processed"] += 1
                    progress.advance(task)
                    
                    # Periodic flush to keep memory low
                    if len(batch_data) >= 500:
                        self._bulk_insert_prices(batch_data)
                        batch_data.clear()

                # Final flush
                if batch_data:
                    self._bulk_insert_prices(batch_data)

        self._console.print(
            f"  Prices: {stats['tickers_processed']} tickers, "
            f"{stats['rows_inserted']} rows fetched"
        )
        if stats["failed"]:
            self._console.print(
                f"  [yellow]Failed: {len(stats['failed'])} tickers[/yellow]"
            )

        benchmark_stats = self.fetch_benchmark_prices(days_back=days_back)
        if benchmark_stats.get("rows_inserted"):
            stats["benchmark_rows_inserted"] = benchmark_stats["rows_inserted"]
        return stats

    def fetch_financials(
        self, tickers: Optional[list[str]] = None
    ) -> dict:
        """Fetch financial statements and store in financial_statements table.
        Uses ThreadPoolExecutor for parallel fetching.
        """
        ticker_list = tickers or self._get_active_tickers()
        if not ticker_list:
            self._console.print("[yellow]No tickers to fetch financials for[/yellow]")
            return {"tickers_processed": 0, "statements_inserted": 0, "failed": []}

        stats = {"tickers_processed": 0, "statements_inserted": 0, "failed": []}
        max_workers = 8
        
        companies = {
            c.ticker: c.id 
            for c in self._session.query(Company).filter(Company.ticker.in_(ticker_list)).all()
        }

        def fetch_one(tsk_ticker):
            try:
                fin_data = self._isy.fetch_financials(tsk_ticker)
                return tsk_ticker, fin_data, None
            except Exception as e:
                return tsk_ticker, None, str(e)

        with self._progress_bar("Fetching financials (Parallel)") as progress:
            task = progress.add_task("Financials", total=len(ticker_list))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(fetch_one, t): t for t in ticker_list
                }
                
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker, fin_data, error = future.result()
                    
                    if error:
                        logger.warning("Failed to fetch financials for %s: %s", ticker, error)
                        stats["failed"].append(ticker)
                    elif fin_data:
                        # Still inserting sequentially here as financial parsing is complex 
                        # and involves multiple inserts (income, balance, cashflow)
                        # We could batch this too but parsing logic is heavy.
                        # Optimization: We removed network wait, so this is CPU bound now.
                        cid = companies.get(ticker)
                        if cid:
                            inserted = self._upsert_financials(cid, fin_data)
                            stats["statements_inserted"] += inserted
                    
                    stats["tickers_processed"] += 1
                    progress.advance(task)

        # self._session.flush() # upsert_financials flushes internally if needed, or we do it at end 
        self._session.commit() # Commit batch of financials

        self._console.print(
            f"  Financials: {stats['tickers_processed']} tickers, "
            f"{stats['statements_inserted']} statements inserted"
        )
        if stats["failed"]:
            self._console.print(
                f"  [yellow]Failed: {len(stats['failed'])} tickers[/yellow]"
            )
        return stats

    def fetch_history(
        self, tickers: Optional[list[str]] = None, num_years: int = 5
    ) -> dict:
        """Fetch deep historical financial data (5-year quarterly).

        This is a one-time backfill operation. It fetches up to 20 quarterly
        periods per company from IsYatirim's MaliTablo endpoint and stores
        them in the financial_statements table.

        Historical financial data doesn't change, so this only needs to be
        run once to populate the database with enough depth for Graham,
        Lynch PEG, and DCF scorers.

        Args:
            tickers: If specified, only backfill these tickers.
            num_years: Number of years to go back (default 5, max 5).

        Returns:
            Stats dict: {tickers_processed, statements_inserted, failed, skipped}.
        """
        ticker_list = tickers or self._get_active_tickers()
        if not ticker_list:
            self._console.print("[yellow]No tickers to backfill[/yellow]")
            return {"tickers_processed": 0, "statements_inserted": 0, "failed": [], "skipped": 0}

        num_years = min(num_years, 5)  # Max 5 years = 20 quarters
        stats = {"tickers_processed": 0, "statements_inserted": 0, "failed": [], "skipped": 0}
        max_workers = 6  # Slightly fewer than normal to be gentle on API

        companies = {
            c.ticker: c.id
            for c in self._session.query(Company).filter(Company.ticker.in_(ticker_list)).all()
        }

        def fetch_one(tsk_ticker):
            try:
                fin_data = self._isy.fetch_financials_deep(
                    tsk_ticker, num_years=num_years
                )
                return tsk_ticker, fin_data, None
            except Exception as e:
                return tsk_ticker, None, str(e)

        self._console.print(
            f"  [bold]Historical backfill:[/bold] {len(ticker_list)} tickers, "
            f"{num_years} years quarterly data"
        )

        with self._progress_bar("Fetching history (Deep)") as progress:
            task = progress.add_task("History", total=len(ticker_list))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(fetch_one, t): t for t in ticker_list
                }

                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker, fin_data, error = future.result()

                    if error:
                        logger.warning("Failed to fetch history for %s: %s", ticker, error)
                        stats["failed"].append(ticker)
                    elif fin_data:
                        cid = companies.get(ticker)
                        if cid:
                            inserted = self._upsert_financials(cid, fin_data)
                            stats["statements_inserted"] += inserted
                    else:
                        stats["skipped"] += 1

                    stats["tickers_processed"] += 1
                    progress.advance(task)

        self._session.commit()

        self._console.print(
            f"  History: {stats['tickers_processed']} tickers processed, "
            f"{stats['statements_inserted']} new statements inserted, "
            f"{stats['skipped']} skipped"
        )
        if stats["failed"]:
            self._console.print(
                f"  [yellow]Failed: {len(stats['failed'])} tickers[/yellow]"
            )
        return stats

    def fetch_macro(self) -> dict:
        """Fetch macro data (CPI, FX, policy rate) from TCMB.

        Falls back to Yahoo Finance for FX rates when TCMB EVDS is
        unavailable (e.g., EVDS2 decommission).

        Returns:
            Stats dict: {cpi_points, fx_points, policy_rate, inflation_rate}.
        """
        self._console.print("[bold]Fetching macro data...[/bold]")
        stats: dict = {
            "cpi_points": 0,
            "fx_points": 0,
            "policy_rate": None,
            "inflation_rate": None,
        }

        end = date.today()
        start = date(end.year - 2, 1, 1)

        # CPI
        cpi = self._tcmb.fetch_cpi_index(start, end)
        stats["cpi_points"] = len(cpi)

        # Exchange rates — try TCMB first, fall back to Yahoo
        fx = self._tcmb.fetch_exchange_rates(start, end)
        fx_source = "TCMB"
        if fx.empty:
            self._console.print(
                "  [yellow]TCMB FX data unavailable — "
                "falling back to Yahoo Finance[/yellow]"
            )
            fx = self._yahoo.fetch_fx_rates(start, end)
            fx_source = "Yahoo"
        stats["fx_points"] = len(fx)

        # Policy rate
        policy_rate = self._tcmb.fetch_policy_rate()
        stats["policy_rate"] = policy_rate

        # Inflation
        inflation = self._tcmb.get_inflation_rate(months_back=12)
        stats["inflation_rate"] = inflation

        # 24-month-ahead CPI expectation (feeds DCF terminal growth)
        inflation_exp_24m = self._tcmb.fetch_inflation_expectations_24m()
        stats["inflation_expectation_24m"] = inflation_exp_24m

        # Store in macro_regime table
        if not fx.empty:
            for _, row in fx.iterrows():
                row_date = pd.Timestamp(row["date"]).date()
                # Convert pandas NaT/NaN to Python None for SQLAlchemy
                usd_val = row.get("usd_try")
                if pd.isna(usd_val):
                    usd_val = None
                else:
                    usd_val = float(usd_val)
                existing = (
                    self._session.query(MacroRegime)
                    .filter(MacroRegime.date == row_date)
                    .first()
                )
                if existing:
                    existing.usdtry_rate = usd_val
                else:
                    regime = MacroRegime(
                        date=row_date,
                        usdtry_rate=usd_val,
                    )
                    self._session.add(regime)

            self._session.flush()

        # Update latest macro_regime with CPI, policy rate, and 24m inflation expectation
        if policy_rate is not None or inflation is not None or inflation_exp_24m is not None:
            today = date.today()
            existing = (
                self._session.query(MacroRegime)
                .filter(MacroRegime.date == today)
                .first()
            )
            if existing:
                if policy_rate is not None:
                    existing.policy_rate_pct = policy_rate
                if inflation is not None:
                    existing.cpi_yoy_pct = inflation
                if inflation_exp_24m is not None:
                    existing.inflation_expectation_24m_pct = inflation_exp_24m
            else:
                regime = MacroRegime(
                    date=today,
                    policy_rate_pct=policy_rate,
                    cpi_yoy_pct=inflation,
                    inflation_expectation_24m_pct=inflation_exp_24m,
                )
                self._session.add(regime)
            self._session.flush()

        policy_str = f"{policy_rate:.1%}" if policy_rate else "N/A"
        inflation_str = f"{inflation:.1%}" if inflation else "N/A"
        exp24_str = f"{inflation_exp_24m:.1%}" if inflation_exp_24m else "N/A"
        self._console.print(
            f"  Macro: {stats['cpi_points']} CPI points, "
            f"{stats['fx_points']} FX points (source: {fx_source}), "
            f"policy rate={policy_str}, inflation={inflation_str}, "
            f"24m CPI exp={exp24_str}"
        )
        return stats

    def fetch_insiders(
        self,
        tickers: Optional[list[str]] = None,
        days_back: int = 180,
    ) -> dict:
        """Fetch insider/document data from KAP.

        Note: Real insider transaction data is not available via KAP scraping.
        This fetches company document metadata as the closest available data.

        Args:
            tickers: Specific tickers. If None, fetches all active.
            days_back: Look-back period in days.

        Returns:
            Stats dict: {tickers_processed, documents_found, failed}.
        """
        ticker_list = tickers or self._get_active_tickers()
        if not ticker_list:
            return {"tickers_processed": 0, "documents_found": 0, "failed": []}

        stats = {"tickers_processed": 0, "documents_found": 0, "failed": []}

        with self._progress_bar("Fetching insiders") as progress:
            task = progress.add_task("Insiders", total=len(ticker_list))

            for ticker in ticker_list:
                try:
                    docs = self._kap.fetch_insider_transactions(
                        ticker, days_back=days_back
                    )
                    if not docs.empty:
                        stats["documents_found"] += len(docs)
                    stats["tickers_processed"] += 1
                except Exception as e:
                    logger.warning("Failed to fetch insiders for %s: %s", ticker, e)
                    stats["failed"].append(ticker)

                progress.advance(task)

        self._console.print(
            f"  Insiders: {stats['tickers_processed']} tickers, "
            f"{stats['documents_found']} documents found"
        )
        return stats

    def fetch_all(
        self,
        tickers: Optional[list[str]] = None,
        limit: int = 0,
    ) -> dict:
        """Run all fetch stages in order.

        Args:
            tickers: If specified, only fetch these tickers.
            limit: If > 0, limit to this many tickers for prices/financials.

        Returns:
            Combined stats from all stages.
        """
        all_stats: dict = {}

        # Stage 1: Universe
        all_stats["universe"] = self.fetch_universe()
        self._session.commit()

        # Determine ticker list for subsequent stages
        fetch_tickers = tickers
        if not fetch_tickers and limit > 0:
            active = self._get_active_tickers()
            fetch_tickers = active[:limit]

        # Stage 2: Prices
        days_back = self._fetch_settings.get("price_history_days", 730)
        all_stats["prices"] = self.fetch_prices(fetch_tickers, days_back=days_back)
        self._session.commit()

        # Stage 3: Financials
        all_stats["financials"] = self.fetch_financials(fetch_tickers)
        self._session.commit()

        # Stage 4: Macro
        all_stats["macro"] = self.fetch_macro()
        self._session.commit()

        # Stage 5: Insiders — DISABLED
        # KAP insider data is not accessible via scraping (JS-only API).
        # IsYatirim has no insider endpoint either. Skipping to save ~10 min.
        # Re-enable when a real insider data source becomes available.

        # Print summary table
        self._print_summary(all_stats)
        return all_stats

    def validate_prices(self, sample_pct: float = 0.10) -> dict:
        """Validate a random sample of prices against Yahoo Finance.

        Args:
            sample_pct: Fraction of active tickers to validate.

        Returns:
            Dict with validation results per ticker.
        """
        active = self._get_active_tickers()
        sample_size = max(1, int(len(active) * sample_pct))
        sample = random.sample(active, min(sample_size, len(active)))

        self._console.print(
            f"[bold]Validating prices for {len(sample)} tickers...[/bold]"
        )
        results = {}

        end = date.today()
        start = end - timedelta(days=30)

        for ticker in sample:
            isy_df = self._isy.fetch_price_data(ticker, start, end)
            if isy_df.empty:
                continue
            result = self._yahoo.validate_prices(isy_df, ticker)
            results[ticker] = result
            self._console.print(
                f"  {ticker}: {result['match_pct']:.0%} match, "
                f"max div={result['max_divergence']:.2%}"
            )

        return results

    # --- Private helpers ---

    def fetch_benchmark_prices(self, days_back: int = 730) -> dict:
        """Fetch BIST 100 benchmark prices used by beta/risk scoring."""
        company = self._ensure_benchmark_company()
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        df = self._yahoo.fetch_index_data(start_date=start_date, end_date=end_date)

        if df.empty:
            logger.warning("Failed to refresh benchmark prices for %s", _BENCHMARK_TICKER)
            return {"rows_inserted": 0}

        batch_data: list[dict] = []
        count = self._prepare_prices_batch(company.id, df, batch_data)
        self._bulk_insert_prices(batch_data)
        self._console.print(
            f"  Benchmark: {_BENCHMARK_TICKER} refreshed ({count} rows)"
        )
        return {"rows_inserted": count}

    def _get_active_tickers(self) -> list[str]:
        """Get list of active tickers from the database."""
        companies = (
            self._session.query(Company.ticker)
            .filter(Company.is_active.is_(True))
            .order_by(Company.ticker)
            .all()
        )
        return [c.ticker for c in companies]

    def _get_or_create_company(self, ticker: str) -> Company:
        """Find a company by ticker, creating a minimal record if needed.

        Args:
            ticker: BIST ticker code.

        Returns:
            Company ORM instance.
        """
        company = (
            self._session.query(Company)
            .filter(Company.ticker == ticker.upper())
            .first()
        )
        if company:
            return company

        company = Company(ticker=ticker.upper(), is_active=True)
        self._session.add(company)
        self._session.flush()
        return company

    def _ensure_benchmark_company(self) -> Company:
        """Ensure the XU100 benchmark exists for beta/regime calculations."""
        company = (
            self._session.query(Company)
            .filter(Company.ticker == _BENCHMARK_TICKER)
            .first()
        )
        if company:
            company.name = company.name or _BENCHMARK_NAME
            company.company_type = "INDEX"
            company.is_active = False
            self._session.flush()
            return company

        company = Company(
            ticker=_BENCHMARK_TICKER,
            name=_BENCHMARK_NAME,
            company_type="INDEX",
            is_active=False,
        )
        self._session.add(company)
        self._session.flush()
        return company

    def _prepare_prices_batch(self, company_id: int, price_df: pd.DataFrame, batch_list: list) -> int:
        """Prepare price rows for bulk insert.
        Returns number of rows added to batch.
        """
        if price_df.empty:
            return 0
        
        count = 0
        for _, row in price_df.iterrows():
            row_date = pd.Timestamp(row["date"]).date()
            batch_list.append({
                "company_id": company_id,
                "date": row_date,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else None,
                "adjusted_close": row.get("adjusted_close"),
                "source": row.get("source", "ISYATIRIM"),
            })
            count += 1
        return count

    def _bulk_insert_prices(self, batch_data: list):
        """Perform bulk insert of price data."""
        if not batch_data:
            return
            
        try:
            # SQLAlchemy 1.4+ / 2.0 style bulk insert
            # We use insert().prefix_with("OR IGNORE") for SQLite to handle duplicates
            from sqlalchemy import insert
            stmt = insert(DailyPrice).values(batch_data).prefix_with("OR IGNORE")
            self._session.execute(stmt)
            self._session.commit()
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}", exc_info=True)
            self._session.rollback()
            raise e

    def _upsert_prices(
        self, company_id: int, price_df: pd.DataFrame
    ) -> tuple[int, int]:
        """Legacy method retained but unused in new parallel flow."""
        return 0, 0

    def _upsert_financials(
        self, company_id: int, fin_data: dict
    ) -> int:
        """Insert financial statement data into the database.

        Args:
            company_id: Company database ID.
            fin_data: Dict from IsYatirimClient.fetch_financials() with
                keys 'income', 'balance', 'cashflow', 'raw', 'financial_group'.

        Returns:
            Number of statement rows inserted.
        """
        raw_items = fin_data.get("raw", [])
        if not raw_items:
            return 0

        # Determine periods from the raw data
        # raw items have value1..value4 corresponding to periods
        # The financial_group and periods info is in fin_data
        fg = fin_data.get("financial_group", "XI_29")
        is_consolidated = fg != "UFRS_K"

        # Map statement types to their DataFrames
        stmt_map = {
            "INCOME": fin_data.get("income"),
            "BALANCE": fin_data.get("balance"),
            "CASHFLOW": fin_data.get("cashflow"),
        }

        inserted = 0

        for stmt_type, df in stmt_map.items():
            if df is None or df.empty:
                continue

            # Extract period columns (format: "YYYY/PP" like "2024/12")
            period_cols = [
                c for c in df.columns
                if c not in ("item_code", "desc_tr", "desc_eng")
            ]

            for period_col in period_cols:
                period_end, period_type = self._parse_period(period_col)
                if period_end is None:
                    continue

                # Check if this statement already exists
                existing = (
                    self._session.query(FinancialStatement)
                    .filter(
                        FinancialStatement.company_id == company_id,
                        FinancialStatement.period_end == period_end,
                        FinancialStatement.period_type == period_type,
                        FinancialStatement.statement_type == stmt_type,
                        FinancialStatement.version == 1,
                    )
                    .first()
                )

                # Build JSON data from the DataFrame column
                data_rows = []
                for _, row in df.iterrows():
                    val = row.get(period_col)
                    data_rows.append({
                        "item_code": row.get("item_code", ""),
                        "desc_tr": row.get("desc_tr", ""),
                        "desc_eng": row.get("desc_eng", ""),
                        "value": val if pd.notna(val) else None,
                    })

                data_json = json.dumps(data_rows, ensure_ascii=False)

                if existing:
                    existing.data_json = data_json
                    existing.is_consolidated = is_consolidated
                else:
                    stmt = FinancialStatement(
                        company_id=company_id,
                        period_end=period_end,
                        period_type=period_type,
                        statement_type=stmt_type,
                        is_consolidated=is_consolidated,
                        is_inflation_adj=(fg == "XI_29"),
                        version=1,
                        data_json=data_json,
                    )
                    self._session.add(stmt)
                    inserted += 1

        if inserted > 0:
            self._session.flush()

        return inserted

    @staticmethod
    def _parse_period(period_str: str) -> tuple[Optional[date], Optional[str]]:
        """Parse a period string like '2024/12' into (date, period_type).

        Args:
            period_str: Format 'YYYY/PP' where PP is 3, 6, 9, or 12.

        Returns:
            Tuple of (period_end_date, period_type_str) or (None, None).
        """
        try:
            parts = period_str.split("/")
            if len(parts) != 2:
                return None, None
            year = int(parts[0])
            period = int(parts[1])

            period_map = {
                3: ("Q1", date(year, 3, 31)),
                6: ("Q2", date(year, 6, 30)),
                9: ("Q3", date(year, 9, 30)),
                12: ("ANNUAL", date(year, 12, 31)),
            }

            if period in period_map:
                ptype, pend = period_map[period]
                return pend, ptype
        except (ValueError, TypeError):
            pass

        return None, None

    def _progress_bar(self, description: str) -> Progress:
        """Create a Rich progress bar, or a pipe-friendly fallback.

        When PIPE_MODE=1 (set by dashboard subprocess), uses a simple
        wrapper that prints 'description N/M' lines instead of Rich
        progress bars (which don't work in pipe mode on Windows).

        Args:
            description: Description shown above the progress bar.

        Returns:
            Progress-like context manager.
        """
        if os.environ.get("PIPE_MODE") == "1":
            return _PipeProgress(description)
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
        )

    def _print_summary(self, all_stats: dict) -> None:
        """Print a summary table of all fetch results.

        Args:
            all_stats: Combined stats from fetch_all().
        """
        table = Table(title="Fetch Summary")
        table.add_column("Stage", style="bold")
        table.add_column("Detail")

        if "universe" in all_stats:
            u = all_stats["universe"]
            table.add_row(
                "Universe",
                f"{u['total']} companies ({u['new']} new, "
                f"{u['bist100_count']} BIST 100)",
            )

        if "prices" in all_stats:
            p = all_stats["prices"]
            table.add_row(
                "Prices",
                f"{p['tickers_processed']} tickers, "
                f"{p['rows_inserted']} rows inserted",
            )

        if "financials" in all_stats:
            f = all_stats["financials"]
            table.add_row(
                "Financials",
                f"{f['tickers_processed']} tickers, "
                f"{f['statements_inserted']} statements",
            )

        if "macro" in all_stats:
            m = all_stats["macro"]
            pr = f"{m['policy_rate']:.1%}" if m.get("policy_rate") else "N/A"
            ir = f"{m['inflation_rate']:.1%}" if m.get("inflation_rate") else "N/A"
            table.add_row(
                "Macro",
                f"{m['cpi_points']} CPI, {m['fx_points']} FX, "
                f"rate={pr}, CPI YoY={ir}",
            )

        if "insiders" in all_stats:
            i = all_stats["insiders"]
            table.add_row(
                "Insiders",
                f"{i['tickers_processed']} tickers, "
                f"{i['documents_found']} documents",
            )

        self._console.print()
        self._console.print(table)
