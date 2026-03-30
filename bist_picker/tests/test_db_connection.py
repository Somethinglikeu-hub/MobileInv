"""Tests for runtime DB bootstrap and query-plan hardening."""

import sqlite3
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.db.connection import ensure_runtime_db_ready
from bist_picker.db.schema import Base, Company, PortfolioSelection, ScoringResult

_RUNTIME_INDEXES = {
    "idx_scoring_results_scoring_date",
    "idx_scoring_results_scoring_date_composite_alpha",
    "idx_scoring_results_company_id_scoring_date",
    "idx_portfolio_selections_portfolio_exit_date_selection_date",
    "idx_portfolio_selections_portfolio_selection_date_company_id",
}


def _make_file_engine(tmp_path, name: str = "runtime.db"):
    """Create a file-backed SQLite engine for planner/index tests."""
    db_path = tmp_path / name
    engine = create_engine(f"sqlite:///{db_path}")
    return engine, db_path


def _sqlite_index_names(db_path) -> set[str]:
    """Return all explicit index names from sqlite_master."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    return {row[0] for row in rows}


def _plan_details(db_path, sql: str) -> list[str]:
    """Return the planner detail strings for a SQL statement."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    return [row[3] for row in rows]


def test_ensure_runtime_db_ready_adds_indexes_to_existing_sqlite_db(tmp_path):
    """Runtime bootstrap should add named indexes even when tables already exist."""
    engine, db_path = _make_file_engine(tmp_path, "existing.db")
    Base.metadata.create_all(engine)

    ensure_runtime_db_ready(engine)

    assert _RUNTIME_INDEXES.issubset(_sqlite_index_names(db_path))


def test_runtime_indexes_are_used_by_dashboard_query_plans(tmp_path):
    """Planner should pick the new runtime indexes for dashboard-critical reads."""
    engine, db_path = _make_file_engine(tmp_path)
    ensure_runtime_db_ready(engine)

    session = sessionmaker(bind=engine)()
    company = Company(
        ticker="PLAN1",
        name="Plan One",
        company_type="OPERATING",
        sector_bist="Industrial",
        sector_custom="Industrial",
        is_active=True,
    )
    session.add(company)
    session.flush()
    session.add_all(
        [
            ScoringResult(
                company_id=company.id,
                scoring_date=date(2026, 3, 16),
                composite_alpha=75.0,
            ),
            ScoringResult(
                company_id=company.id,
                scoring_date=date(2026, 3, 19),
                composite_alpha=88.0,
            ),
            PortfolioSelection(
                portfolio="ALPHA",
                selection_date=date(2026, 3, 19),
                company_id=company.id,
                entry_price=100.0,
            ),
        ]
    )
    session.commit()
    session.close()

    latest_date_plan = _plan_details(
        db_path,
        "SELECT scoring_date FROM scoring_results ORDER BY scoring_date DESC LIMIT 1",
    )
    scoring_rows_plan = _plan_details(
        db_path,
        "SELECT * FROM scoring_results "
        "WHERE scoring_date = '2026-03-19' ORDER BY composite_alpha DESC",
    )
    open_alpha_plan = _plan_details(
        db_path,
        "SELECT selection_date FROM portfolio_selections "
        "WHERE exit_date IS NULL AND portfolio = 'ALPHA' "
        "ORDER BY selection_date DESC LIMIT 1",
    )

    assert any("idx_scoring_results_" in detail for detail in latest_date_plan)
    assert any("idx_scoring_results_" in detail for detail in scoring_rows_plan)
    assert any(
        "idx_portfolio_selections_portfolio_exit_date_selection_date" in detail
        for detail in open_alpha_plan
    )
