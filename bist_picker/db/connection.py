"""Database connection utilities for BIST Stock Picker.

Provides get_engine() and get_session() for SQLite database access.
All database interactions should use these functions.
"""

import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
import yaml

from bist_picker.db.schema import Base

# Default database path: project root / data / bist_picker.db
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB_DIR = _PROJECT_ROOT / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "bist_picker.db"
_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None

# Idempotent column-add migrations for existing SQLite databases.
# `Base.metadata.create_all` only creates missing tables; it does not alter
# existing ones. Each entry here is applied only if the column is absent.
# Key format: (table, column) -> ALTER TABLE DDL.
_RUNTIME_SQLITE_COLUMN_ADDS: dict[tuple[str, str], str] = {
    ("macro_regime", "inflation_expectation_24m_pct"): (
        "ALTER TABLE macro_regime "
        "ADD COLUMN inflation_expectation_24m_pct REAL"
    ),
    # Phase 4: per-pick weight + cash state snapshot on portfolio_selections.
    ("portfolio_selections", "weight"): (
        "ALTER TABLE portfolio_selections ADD COLUMN weight REAL"
    ),
    ("portfolio_selections", "cash_state"): (
        "ALTER TABLE portfolio_selections ADD COLUMN cash_state VARCHAR(20)"
    ),
    ("portfolio_selections", "cash_pct"): (
        "ALTER TABLE portfolio_selections ADD COLUMN cash_pct REAL"
    ),
    # Phase 5: transparency surface — DCF breakdown + per-pick reason chips.
    ("scoring_results", "dcf_intrinsic_value"): (
        "ALTER TABLE scoring_results ADD COLUMN dcf_intrinsic_value REAL"
    ),
    ("scoring_results", "dcf_growth_rate_pct"): (
        "ALTER TABLE scoring_results ADD COLUMN dcf_growth_rate_pct REAL"
    ),
    ("scoring_results", "dcf_discount_rate_pct"): (
        "ALTER TABLE scoring_results ADD COLUMN dcf_discount_rate_pct REAL"
    ),
    ("scoring_results", "dcf_terminal_growth_pct"): (
        "ALTER TABLE scoring_results ADD COLUMN dcf_terminal_growth_pct REAL"
    ),
    ("portfolio_selections", "reason_top_factors_json"): (
        "ALTER TABLE portfolio_selections ADD COLUMN reason_top_factors_json TEXT"
    ),
    # Sprint 1 (2026-05-07) — picker fixes:
    # §3.1 Buffett inflation-aware ROE/ROA (Fisher-deflated).
    ("adjusted_metrics", "roe_real"): (
        "ALTER TABLE adjusted_metrics ADD COLUMN roe_real REAL"
    ),
    ("adjusted_metrics", "roa_real"): (
        "ALTER TABLE adjusted_metrics ADD COLUMN roa_real REAL"
    ),
    # §3.2 Point-in-time guard for AdjustedMetric.
    ("adjusted_metrics", "publication_date"): (
        "ALTER TABLE adjusted_metrics ADD COLUMN publication_date DATE"
    ),
    # §3.5 Raw above_200ma signal for falling-knife filter.
    ("scoring_results", "above_200ma"): (
        "ALTER TABLE scoring_results ADD COLUMN above_200ma BOOLEAN"
    ),
    # §3.7 Damodaran ERP auto-fetch persisted on MacroRegime.
    ("macro_regime", "equity_risk_premium_pct"): (
        "ALTER TABLE macro_regime ADD COLUMN equity_risk_premium_pct REAL"
    ),
    ("macro_regime", "erp_source"): (
        "ALTER TABLE macro_regime ADD COLUMN erp_source VARCHAR(64)"
    ),
}

_RUNTIME_SQLITE_INDEXES: dict[str, str] = {
    "idx_scoring_results_scoring_date": (
        "CREATE INDEX IF NOT EXISTS idx_scoring_results_scoring_date "
        "ON scoring_results (scoring_date)"
    ),
    "idx_scoring_results_scoring_date_composite_alpha": (
        "CREATE INDEX IF NOT EXISTS idx_scoring_results_scoring_date_composite_alpha "
        "ON scoring_results (scoring_date, composite_alpha)"
    ),
    "idx_scoring_results_company_id_scoring_date": (
        "CREATE INDEX IF NOT EXISTS idx_scoring_results_company_id_scoring_date "
        "ON scoring_results (company_id, scoring_date)"
    ),
    "idx_portfolio_selections_portfolio_exit_date_selection_date": (
        "CREATE INDEX IF NOT EXISTS "
        "idx_portfolio_selections_portfolio_exit_date_selection_date "
        "ON portfolio_selections (portfolio, exit_date, selection_date)"
    ),
    "idx_portfolio_selections_portfolio_selection_date_company_id": (
        "CREATE INDEX IF NOT EXISTS "
        "idx_portfolio_selections_portfolio_selection_date_company_id "
        "ON portfolio_selections (portfolio, selection_date, company_id)"
    ),
}


def _load_db_path_from_settings() -> Path | None:
    """Load database.path from bist_picker/config/settings.yaml if available."""
    if not _SETTINGS_PATH.exists():
        return None

    try:
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None

    db_path = (settings.get("database", {}).get("path", "") or "").strip()
    if not db_path:
        return None

    path = Path(db_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def get_engine(db_path: str | Path | None = None) -> Engine:
    """Get or create the SQLAlchemy engine for SQLite.

    Args:
        db_path: Optional path to the SQLite database file.
                 Resolution order when omitted:
                 1) BIST_DB_PATH env var
                 2) database.path in bist_picker/config/settings.yaml
                 3) data/bist_picker.db in the project root

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine
    if _engine is not None:
        return _engine

    if db_path is None:
        env_db_path = os.environ.get("BIST_DB_PATH", "").strip()
        if env_db_path:
            db_path = env_db_path
        else:
            db_path = _load_db_path_from_settings() or _DEFAULT_DB_PATH

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        future=True,
    )

    from sqlalchemy import event
    @event.listens_for(_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000") # 64MB cache
        cursor.close()

    return _engine


def ensure_runtime_db_ready(engine: Engine | None = None) -> Engine:
    """Ensure tables and runtime query indexes exist for the active database.

    The extra indexes are created explicitly because they are operational
    accelerators for the Streamlit dashboard's most common read paths rather
    than part of the ORM schema definition itself.
    """
    if engine is None:
        engine = get_engine()

    Base.metadata.create_all(engine)

    if engine.dialect.name != "sqlite":
        return engine

    with engine.begin() as connection:
        # Apply idempotent column-add migrations first so subsequent index
        # creation can reference the new columns if needed.
        for (table, column), ddl in _RUNTIME_SQLITE_COLUMN_ADDS.items():
            existing = connection.exec_driver_sql(
                f"PRAGMA table_info({table})"
            ).fetchall()
            existing_cols = {row[1] for row in existing}
            if column not in existing_cols:
                connection.exec_driver_sql(ddl)

        for ddl in _RUNTIME_SQLITE_INDEXES.values():
            connection.exec_driver_sql(ddl)

    return engine


def create_tables(engine: Engine | None = None) -> None:
    """Create all tables defined in schema.py.

    Args:
        engine: SQLAlchemy engine. Uses default if not provided.
    """
    ensure_runtime_db_ready(engine)


def get_session(engine: Engine | None = None) -> Session:
    """Create a new database session.

    Args:
        engine: SQLAlchemy engine. Uses default if not provided.

    Returns:
        A new SQLAlchemy Session instance.
    """
    global _SessionFactory
    if _SessionFactory is None:
        if engine is None:
            engine = get_engine()
        _SessionFactory = sessionmaker(bind=engine)
    return _SessionFactory()


@contextmanager
def session_scope(engine: Engine | None = None):
    """Context manager for database sessions with auto commit/rollback.

    Usage:
        with session_scope() as session:
            session.add(obj)
            # auto-commits on exit, rolls back on exception

    Args:
        engine: SQLAlchemy engine. Uses default if not provided.

    Yields:
        SQLAlchemy Session instance.
    """
    session = get_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
