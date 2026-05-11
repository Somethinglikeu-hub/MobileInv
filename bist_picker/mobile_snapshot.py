"""Export a compact offline snapshot for the Android app."""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import mkstemp
from typing import Any

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from bist_picker import read_service
from bist_picker.db.schema import (
    AdjustedMetric,
    Company,
    DailyPrice,
    ScoringResult,
)

SNAPSHOT_SCHEMA_VERSION = 1  # 2026-05-11: reverted from 2 — v1 APK rejects v2 ("snapshot sürümü desteklenmiyor: 2"). New alpha_x_* / factor_history_quarterly content is still emitted (additive), but the version stays at 1 until the v2 APK ships in Sprint 3.
PRICE_HISTORY_DAYS = 730
DEFAULT_MOBILE_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "mobile_snapshot.db"
REQUIRED_TABLES = (
    "snapshot_metadata",
    "home_summary",
    "open_positions",
    "portfolio_history",
    "companies",
    "scoring_latest",
    "adjusted_metrics_latest",
    "price_history_730d",
    # v2 (2026-05-08)
    "factor_history_quarterly",
)


def _sqlite_value(value: Any) -> Any:
    """Normalize values before writing them into SQLite."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _write_records(
    connection: sqlite3.Connection,
    table_name: str,
    columns: list[str],
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    connection.executemany(
        sql,
        [
            tuple(_sqlite_value(record.get(column)) for column in columns)
            for record in records
        ],
    )


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE snapshot_metadata (
            id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
            schema_version INTEGER NOT NULL,
            exported_at TEXT NOT NULL,
            snapshot_date TEXT,
            latest_price_date TEXT,
            source_db_path TEXT,
            company_count INTEGER NOT NULL,
            scoring_row_count INTEGER NOT NULL,
            price_history_days INTEGER NOT NULL
        );

        CREATE TABLE home_summary (
            id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
            total_return_avg REAL,
            active_return_avg REAL,
            win_rate REAL,
            benchmark_ytd REAL,
            macro_date TEXT,
            policy_rate_pct REAL,
            cpi_yoy_pct REAL,
            usdtry_rate REAL,
            regime TEXT,
            cash_state TEXT,
            cash_pct REAL,
            cash_days_in_state INTEGER,
            cash_last_transition_date TEXT,
            cash_target_state TEXT
        );

        CREATE TABLE open_positions (
            sort_order INTEGER NOT NULL PRIMARY KEY,
            portfolio TEXT,
            ticker TEXT NOT NULL,
            name TEXT,
            company_id INTEGER,
            entry_price REAL,
            current_price REAL,
            pnl_pct REAL,
            target_price REAL,
            stop_loss_price REAL,
            stop_pct_from_entry REAL,
            composite_score REAL,
            selection_date TEXT,
            days_held INTEGER,
            reason_top_factors_json TEXT,
            quality_flags_json TEXT,
            dcf_margin_of_safety_pct REAL,
            dcf_intrinsic_value REAL,
            dcf_growth_rate_pct REAL,
            dcf_discount_rate_pct REAL,
            dcf_terminal_growth_pct REAL
        );

        CREATE TABLE portfolio_history (
            sort_order INTEGER NOT NULL PRIMARY KEY,
            portfolio TEXT,
            ticker TEXT NOT NULL,
            name TEXT,
            selection_date TEXT,
            exit_date TEXT,
            entry_price REAL,
            exit_price REAL,
            pnl_pct REAL
        );

        CREATE TABLE companies (
            id INTEGER NOT NULL PRIMARY KEY,
            ticker TEXT NOT NULL UNIQUE,
            name TEXT,
            company_type TEXT,
            sector_bist TEXT,
            sector_custom TEXT,
            is_bist100 INTEGER NOT NULL,
            is_ipo INTEGER NOT NULL,
            free_float_pct REAL,
            listing_date TEXT,
            is_active INTEGER NOT NULL
        );

        CREATE TABLE scoring_latest (
            company_id INTEGER NOT NULL PRIMARY KEY,
            ticker TEXT NOT NULL UNIQUE,
            name TEXT,
            type TEXT,
            sector TEXT,
            is_bist100 INTEGER NOT NULL,
            is_active INTEGER NOT NULL,
            free_float_pct REAL,
            avg_volume_try REAL,
            -- v2 (2026-05-08): unified ranking surface for the APK Liste tab.
            -- ranking_score = composite_alpha for now; ranking_source labels the
            -- pipeline so future ML / blended scores can plug in without
            -- breaking APK assumptions.
            ranking_score REAL,
            ranking_source TEXT,
            -- model_score = the sector-specific composite when one applies
            -- (banking/holding/REIT). NULL for OPERATING. Lets the APK detail
            -- card show "Banking model: 95.3" when relevant.
            model_score REAL,
            alpha REAL,
            -- alpha_x_* (v2): research-quality variant. alpha_x_score weighs
            -- alpha by data confidence; alpha_x_rank is rank within the
            -- research-eligible set; alpha_x_eligible widens core to also
            -- include Quality / Free-Float Shadow buckets so the APK has a
            -- "wider net" view without losing the strict ALPHA Core gate.
            alpha_x_score REAL,
            alpha_x_rank REAL,
            alpha_x_confidence REAL,
            alpha_core_eligible INTEGER NOT NULL,
            alpha_x_eligible INTEGER NOT NULL,
            alpha_reason TEXT,
            alpha_primary_blocker TEXT,
            alpha_research_bucket TEXT,
            alpha_snapshot_streak INTEGER,
            risk TEXT,
            data_completeness REAL,
            scoring_date TEXT,
            model_used TEXT,
            buffett REAL,
            graham REAL,
            piotroski REAL,
            piotroski_raw INTEGER,
            magic_formula REAL,
            lynch_peg REAL,
            dcf_mos REAL,
            momentum REAL,
            insider REAL,
            technical REAL,
            dividend REAL,
            beta REAL,
            delta REAL,
            quality_flags_json TEXT,
            dcf_intrinsic_value REAL,
            dcf_growth_rate_pct REAL,
            dcf_discount_rate_pct REAL,
            dcf_terminal_growth_pct REAL
        );

        CREATE TABLE adjusted_metrics_latest (
            company_id INTEGER NOT NULL PRIMARY KEY,
            period_end TEXT,
            reported_net_income REAL,
            monetary_gain_loss REAL,
            adjusted_net_income REAL,
            owner_earnings REAL,
            free_cash_flow REAL,
            roe_adjusted REAL,
            roa_adjusted REAL,
            eps_adjusted REAL,
            real_eps_growth_pct REAL,
            related_party_revenue_pct REAL,
            maintenance_capex REAL,
            growth_capex REAL
        );

        -- Sprint 2 §5 (2026-05-08): quarter-end factor snapshots for the
        -- v2 mobile sparkline UI. Only the universe of currently-pickable
        -- companies plus open positions is included — keeps the snapshot
        -- small (~30 KB at 50 companies × 8 quarters).
        CREATE TABLE factor_history_quarterly (
            company_id INTEGER NOT NULL,
            quarter_end TEXT NOT NULL,
            scoring_date TEXT NOT NULL,
            buffett REAL,
            graham REAL,
            piotroski REAL,
            magic_formula REAL,
            lynch_peg REAL,
            dcf_mos REAL,
            momentum REAL,
            technical REAL,
            dividend REAL,
            composite_alpha REAL,
            data_completeness REAL,
            PRIMARY KEY (company_id, quarter_end)
        );

        CREATE TABLE price_history_730d (
            company_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adjusted_close REAL,
            PRIMARY KEY (company_id, date)
        );

        """
    )


def _load_companies(session) -> list[Company]:
    return session.query(Company).order_by(Company.ticker.asc()).all()


def _load_latest_scores(session, scoring_date: date) -> dict[int, ScoringResult]:
    rows = (
        session.query(ScoringResult)
        .filter(ScoringResult.scoring_date == scoring_date)
        .all()
    )
    return {row.company_id: row for row in rows}


def _load_latest_adjusted_metrics(session) -> dict[int, AdjustedMetric]:
    latest_periods = (
        session.query(
            AdjustedMetric.company_id.label("company_id"),
            func.max(AdjustedMetric.period_end).label("period_end"),
        )
        .group_by(AdjustedMetric.company_id)
        .subquery()
    )
    rows = (
        session.query(AdjustedMetric)
        .join(
            latest_periods,
            (AdjustedMetric.company_id == latest_periods.c.company_id)
            & (AdjustedMetric.period_end == latest_periods.c.period_end),
        )
        .all()
    )
    return {row.company_id: row for row in rows}


def _select_factor_history_universe(
    companies_by_ticker: dict,
    open_positions_frame,
    scoring_frame,
    top_alpha_n: int = 75,
) -> list[int]:
    """Pick the company set worth shipping quarter-end factor history for.

    Universe = currently-open ALPHA positions ∪ top-N alpha_x_eligible
    companies in the latest scoring snapshot. Anything outside this set
    won't get a sparkline in the APK, so storing 8 quarters of factor
    history for it is wasted bytes.

    Returns a list of unique company_ids; keeps the snapshot footprint
    bounded (~75 companies × 8 quarters × ~12 numeric columns ≈ 60 KB).
    """
    ids: set[int] = set()

    if open_positions_frame is not None and not open_positions_frame.empty:
        for ticker in open_positions_frame.get("ticker", []):
            company = companies_by_ticker.get(str(ticker))
            if company is not None:
                ids.add(company.id)

    if scoring_frame is not None and not scoring_frame.empty:
        eligible = scoring_frame
        if "alpha_x_eligible" in eligible.columns:
            eligible = eligible[eligible["alpha_x_eligible"].astype(bool)]
        if "alpha_x_score" in eligible.columns:
            eligible = eligible.sort_values(
                "alpha_x_score", ascending=False, na_position="last"
            )
        for ticker in eligible.head(top_alpha_n).get("ticker", []):
            company = companies_by_ticker.get(str(ticker))
            if company is not None:
                ids.add(company.id)

    return sorted(ids)


def _load_price_history(session) -> list[dict[str, Any]]:
    cutoff = date.today() - timedelta(days=PRICE_HISTORY_DAYS)
    rows = (
        session.query(DailyPrice)
        .filter(DailyPrice.date >= cutoff)
        .order_by(DailyPrice.company_id.asc(), DailyPrice.date.asc())
        .all()
    )
    return [
        {
            "company_id": row.company_id,
            "date": row.date,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "adjusted_close": row.adjusted_close,
        }
        for row in rows
    ]


def export_mobile_snapshot(output_path: str | Path = DEFAULT_MOBILE_SNAPSHOT_PATH) -> Path:
    """Export the latest offline mobile snapshot into a compact SQLite database."""
    latest_scoring_date = read_service.get_latest_scoring_date()
    if latest_scoring_date is None:
        raise RuntimeError("No scoring snapshot found. Run the scoring pipeline first.")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported_at = datetime.now(timezone.utc).replace(microsecond=0)
    source_engine = read_service.get_engine()
    source_db_path = getattr(source_engine.url, "database", None)
    session = sessionmaker(bind=source_engine)()

    try:
        performance = read_service.get_all_portfolio_performance() or {}
        macro = read_service.get_latest_macro() or {}
        cash_state = read_service.get_latest_cash_state() or {}
        open_positions_frame = read_service.get_open_positions()
        portfolio_history_frame = read_service.get_portfolio_history()
        scoring_frame = read_service.get_scoring_results(scoring_date=latest_scoring_date)
        companies = _load_companies(session)
        companies_by_ticker = {company.ticker: company for company in companies}
        latest_scores = _load_latest_scores(session, latest_scoring_date)
        latest_metrics = _load_latest_adjusted_metrics(session)
        price_history = _load_price_history(session)
        # v2 §5: quarter-end factor history for sparkline UI. Limit the
        # set to currently-open positions + the top alpha-X eligible
        # universe so the snapshot stays small. The selection mirrors
        # what the APK can actually drill into from the Liste tab.
        factor_history_company_ids = _select_factor_history_universe(
            companies_by_ticker,
            open_positions_frame,
            scoring_frame,
        )
        factor_history_frame = read_service.get_factor_history_quarterly(
            factor_history_company_ids,
            quarters=8,
            end_date=latest_scoring_date,
        )
    finally:
        session.close()

    latest_price_date = max(
        (row["date"] for row in price_history),
        default=None,
    )

    temp_fd, temp_name = mkstemp(
        prefix="mobile_snapshot_",
        suffix=".db",
        dir=str(output_path.parent),
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)
    connection: sqlite3.Connection | None = None

    try:
        connection = sqlite3.connect(temp_path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        _create_schema(connection)

        _write_records(
            connection,
            "snapshot_metadata",
            [
                "id",
                "schema_version",
                "exported_at",
                "snapshot_date",
                "latest_price_date",
                "source_db_path",
                "company_count",
                "scoring_row_count",
                "price_history_days",
            ],
            [
                {
                    "id": 1,
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
                    "exported_at": exported_at,
                    "snapshot_date": latest_scoring_date,
                    "latest_price_date": latest_price_date,
                    "source_db_path": source_db_path,
                    "company_count": len(companies),
                    "scoring_row_count": len(scoring_frame),
                    "price_history_days": PRICE_HISTORY_DAYS,
                }
            ],
        )

        _write_records(
            connection,
            "home_summary",
            [
                "id",
                "total_return_avg",
                "active_return_avg",
                "win_rate",
                "benchmark_ytd",
                "macro_date",
                "policy_rate_pct",
                "cpi_yoy_pct",
                "usdtry_rate",
                "regime",
                "cash_state",
                "cash_pct",
                "cash_days_in_state",
                "cash_last_transition_date",
                "cash_target_state",
            ],
            [
                {
                    "id": 1,
                    "total_return_avg": performance.get("total_return_avg"),
                    "active_return_avg": performance.get("active_return_avg"),
                    "win_rate": performance.get("win_rate"),
                    "benchmark_ytd": performance.get("benchmark_ytd"),
                    "macro_date": macro.get("date"),
                    "policy_rate_pct": macro.get("policy_rate_pct"),
                    "cpi_yoy_pct": macro.get("cpi_yoy_pct"),
                    "usdtry_rate": macro.get("usdtry_rate"),
                    "regime": macro.get("regime"),
                    "cash_state": cash_state.get("state"),
                    "cash_pct": cash_state.get("cash_pct"),
                    "cash_days_in_state": cash_state.get("days_in_state"),
                    "cash_last_transition_date": cash_state.get("last_transition_date"),
                    "cash_target_state": cash_state.get("target_state"),
                }
            ],
        )

        _write_records(
            connection,
            "open_positions",
            [
                "sort_order",
                "portfolio",
                "ticker",
                "name",
                "company_id",
                "entry_price",
                "current_price",
                "pnl_pct",
                "target_price",
                "stop_loss_price",
                "stop_pct_from_entry",
                "composite_score",
                "selection_date",
                "days_held",
                "reason_top_factors_json",
                "quality_flags_json",
                "dcf_margin_of_safety_pct",
                "dcf_intrinsic_value",
                "dcf_growth_rate_pct",
                "dcf_discount_rate_pct",
                "dcf_terminal_growth_pct",
            ],
            [
                {"sort_order": index, **record}
                for index, record in enumerate(
                    open_positions_frame.to_dict(orient="records"),
                    start=1,
                )
            ],
        )

        _write_records(
            connection,
            "portfolio_history",
            [
                "sort_order",
                "portfolio",
                "ticker",
                "name",
                "selection_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "pnl_pct",
            ],
            [
                {"sort_order": index, **record}
                for index, record in enumerate(
                    portfolio_history_frame.to_dict(orient="records"),
                    start=1,
                )
            ],
        )

        _write_records(
            connection,
            "companies",
            [
                "id",
                "ticker",
                "name",
                "company_type",
                "sector_bist",
                "sector_custom",
                "is_bist100",
                "is_ipo",
                "free_float_pct",
                "listing_date",
                "is_active",
            ],
            [
                {
                    "id": company.id,
                    "ticker": company.ticker,
                    "name": company.name,
                    "company_type": company.company_type,
                    "sector_bist": company.sector_bist,
                    "sector_custom": company.sector_custom,
                    "is_bist100": company.is_bist100,
                    "is_ipo": company.is_ipo,
                    "free_float_pct": company.free_float_pct,
                    "listing_date": company.listing_date,
                    "is_active": company.is_active,
                }
                for company in companies
            ],
        )

        _write_records(
            connection,
            "scoring_latest",
            [
                "company_id",
                "ticker",
                "name",
                "type",
                "sector",
                "is_bist100",
                "is_active",
                "free_float_pct",
                "avg_volume_try",
                "ranking_score",
                "ranking_source",
                "model_score",
                "alpha",
                "alpha_x_score",
                "alpha_x_rank",
                "alpha_x_confidence",
                "alpha_core_eligible",
                "alpha_x_eligible",
                "alpha_reason",
                "alpha_primary_blocker",
                "alpha_research_bucket",
                "alpha_snapshot_streak",
                "risk",
                "data_completeness",
                "scoring_date",
                "model_used",
                "buffett",
                "graham",
                "piotroski",
                "piotroski_raw",
                "magic_formula",
                "lynch_peg",
                "dcf_mos",
                "momentum",
                "insider",
                "technical",
                "dividend",
                "beta",
                "delta",
                "quality_flags_json",
                "dcf_intrinsic_value",
                "dcf_growth_rate_pct",
                "dcf_discount_rate_pct",
                "dcf_terminal_growth_pct",
            ],
            [
                {
                    "company_id": companies_by_ticker[str(record["ticker"])].id,
                    "ticker": record.get("ticker"),
                    "name": record.get("name"),
                    "type": record.get("type"),
                    "sector": record.get("sector"),
                    "is_bist100": record.get("bist100"),
                    "is_active": record.get("is_active"),
                    "free_float_pct": record.get("free_float_pct"),
                    "avg_volume_try": record.get("avg_volume_try"),
                    "ranking_score": record.get("ranking_score"),
                    "ranking_source": record.get("ranking_source"),
                    "model_score": record.get("model_score"),
                    "alpha": record.get("alpha"),
                    "alpha_x_score": record.get("alpha_x_score"),
                    "alpha_x_rank": record.get("alpha_x_rank"),
                    "alpha_x_confidence": record.get("alpha_x_confidence"),
                    "alpha_core_eligible": record.get("alpha_core_eligible"),
                    "alpha_x_eligible": record.get("alpha_x_eligible"),
                    "alpha_reason": record.get("alpha_reason"),
                    "alpha_primary_blocker": record.get("alpha_primary_blocker"),
                    "alpha_research_bucket": record.get("alpha_research_bucket"),
                    "alpha_snapshot_streak": record.get("alpha_snapshot_streak"),
                    "risk": record.get("risk"),
                    "data_completeness": record.get("data_completeness"),
                    "scoring_date": record.get("scoring_date"),
                    "model_used": (
                        latest_scores.get(companies_by_ticker[str(record["ticker"])].id).model_used
                        if latest_scores.get(companies_by_ticker[str(record["ticker"])].id)
                        else None
                    ),
                    "buffett": record.get("buffett"),
                    "graham": record.get("graham"),
                    "piotroski": record.get("piotroski"),
                    "piotroski_raw": record.get("piotroski_raw"),
                    "magic_formula": record.get("magic_formula"),
                    "lynch_peg": record.get("lynch_peg"),
                    "dcf_mos": record.get("dcf_mos"),
                    "momentum": record.get("momentum"),
                    "insider": record.get("insider"),
                    "technical": record.get("technical"),
                    "dividend": record.get("dividend"),
                    "beta": (
                        latest_scores.get(companies_by_ticker[str(record["ticker"])].id).composite_beta
                        if latest_scores.get(companies_by_ticker[str(record["ticker"])].id)
                        else None
                    ),
                    "delta": (
                        latest_scores.get(companies_by_ticker[str(record["ticker"])].id).composite_delta
                        if latest_scores.get(companies_by_ticker[str(record["ticker"])].id)
                        else None
                    ),
                    "quality_flags_json": (
                        latest_scores.get(companies_by_ticker[str(record["ticker"])].id).quality_flags_json
                        if latest_scores.get(companies_by_ticker[str(record["ticker"])].id)
                        else None
                    ),
                    "dcf_intrinsic_value": (
                        latest_scores.get(companies_by_ticker[str(record["ticker"])].id).dcf_intrinsic_value
                        if latest_scores.get(companies_by_ticker[str(record["ticker"])].id)
                        else None
                    ),
                    "dcf_growth_rate_pct": (
                        latest_scores.get(companies_by_ticker[str(record["ticker"])].id).dcf_growth_rate_pct
                        if latest_scores.get(companies_by_ticker[str(record["ticker"])].id)
                        else None
                    ),
                    "dcf_discount_rate_pct": (
                        latest_scores.get(companies_by_ticker[str(record["ticker"])].id).dcf_discount_rate_pct
                        if latest_scores.get(companies_by_ticker[str(record["ticker"])].id)
                        else None
                    ),
                    "dcf_terminal_growth_pct": (
                        latest_scores.get(companies_by_ticker[str(record["ticker"])].id).dcf_terminal_growth_pct
                        if latest_scores.get(companies_by_ticker[str(record["ticker"])].id)
                        else None
                    ),
                }
                for record in scoring_frame.to_dict(orient="records")
                if str(record.get("ticker")) in companies_by_ticker
            ],
        )

        _write_records(
            connection,
            "adjusted_metrics_latest",
            [
                "company_id",
                "period_end",
                "reported_net_income",
                "monetary_gain_loss",
                "adjusted_net_income",
                "owner_earnings",
                "free_cash_flow",
                "roe_adjusted",
                "roa_adjusted",
                "eps_adjusted",
                "real_eps_growth_pct",
                "related_party_revenue_pct",
                "maintenance_capex",
                "growth_capex",
            ],
            [
                {
                    "company_id": company_id,
                    "period_end": metric.period_end,
                    "reported_net_income": metric.reported_net_income,
                    "monetary_gain_loss": metric.monetary_gain_loss,
                    "adjusted_net_income": metric.adjusted_net_income,
                    "owner_earnings": metric.owner_earnings,
                    "free_cash_flow": metric.free_cash_flow,
                    "roe_adjusted": metric.roe_adjusted,
                    "roa_adjusted": metric.roa_adjusted,
                    "eps_adjusted": metric.eps_adjusted,
                    "real_eps_growth_pct": metric.real_eps_growth_pct,
                    "related_party_revenue_pct": metric.related_party_revenue_pct,
                    "maintenance_capex": metric.maintenance_capex,
                    "growth_capex": metric.growth_capex,
                }
                for company_id, metric in latest_metrics.items()
            ],
        )

        _write_records(
            connection,
            "price_history_730d",
            [
                "company_id",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjusted_close",
            ],
            price_history,
        )

        _write_records(
            connection,
            "factor_history_quarterly",
            [
                "company_id", "quarter_end", "scoring_date",
                "buffett", "graham", "piotroski", "magic_formula",
                "lynch_peg", "dcf_mos", "momentum", "technical",
                "dividend", "composite_alpha", "data_completeness",
            ],
            (
                factor_history_frame.to_dict(orient="records")
                if factor_history_frame is not None and not factor_history_frame.empty
                else []
            ),
        )

        connection.commit()
        connection.close()
        connection = None

        if output_path.exists():
            output_path.unlink()
        os.replace(temp_path, output_path)
    except Exception:
        if connection is not None:
            connection.close()
        if temp_path.exists():
            temp_path.unlink()
        raise

    return output_path


def validate_mobile_snapshot(snapshot_path: str | Path) -> dict[str, Any]:
    """Validate a mobile snapshot file and return its metadata row."""
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.exists():
        raise FileNotFoundError(snapshot_path)

    with sqlite3.connect(snapshot_path) as connection:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row[0] for row in table_rows}
        missing_tables = [table_name for table_name in REQUIRED_TABLES if table_name not in table_names]
        if missing_tables:
            raise RuntimeError(f"Snapshot missing required tables: {', '.join(missing_tables)}")

        connection.row_factory = sqlite3.Row
        metadata_row = connection.execute(
            "SELECT * FROM snapshot_metadata WHERE id = 1"
        ).fetchone()
        if metadata_row is None:
            raise RuntimeError("Snapshot metadata row is missing.")
        metadata = dict(metadata_row)
        if metadata.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Snapshot schema version {metadata.get('schema_version')} is not supported."
            )
        return metadata
