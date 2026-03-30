"""Tests for the offline mobile snapshot export."""

from __future__ import annotations

import json
import sqlite3
import gzip
from datetime import date, timedelta

from click.testing import CliRunner
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

import bist_picker.read_service as read_service
from bist_picker.cli import cli
from bist_picker.db.schema import (
    AdjustedMetric,
    Base,
    Company,
    DailyPrice,
    MacroRegime,
    PortfolioSelection,
    ScoringResult,
)
from bist_picker.mobile_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    export_mobile_snapshot,
    validate_mobile_snapshot,
)
from bist_picker.mobile_feed import export_mobile_feed


@pytest.fixture
def source_engine(monkeypatch):
    """Provide an isolated source database for snapshot export tests."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    monkeypatch.setattr(read_service, "get_engine", lambda: engine)

    session = session_factory()
    try:
        company = Company(
            ticker="TEST1",
            name="Test One",
            company_type="OPERATING",
            sector_bist="Industrial",
            sector_custom="Industrial",
            free_float_pct=35.0,
            is_bist100=True,
            is_active=True,
        )
        session.add(company)
        session.flush()

        today = date(2026, 3, 19)
        session.add_all(
            [
                DailyPrice(
                    company_id=company.id,
                    date=today - timedelta(days=365),
                    close=80.0,
                    adjusted_close=82.0,
                    volume=15_000_000,
                ),
                DailyPrice(
                    company_id=company.id,
                    date=today,
                    close=100.0,
                    adjusted_close=101.0,
                    high=102.0,
                    low=99.0,
                    volume=20_000_000,
                ),
                PortfolioSelection(
                    portfolio="ALPHA",
                    selection_date=today - timedelta(days=7),
                    company_id=company.id,
                    entry_price=95.0,
                    composite_score=89.0,
                    target_price=120.0,
                    stop_loss_price=88.0,
                ),
                PortfolioSelection(
                    portfolio="ALPHA",
                    selection_date=today - timedelta(days=60),
                    company_id=company.id,
                    entry_price=75.0,
                    exit_date=today - timedelta(days=30),
                    exit_price=90.0,
                ),
                ScoringResult(
                    company_id=company.id,
                    scoring_date=today,
                    model_used="OPERATING",
                    composite_alpha=91.0,
                    composite_beta=77.0,
                    composite_delta=66.0,
                    buffett_score=80.0,
                    graham_score=72.0,
                    piotroski_fscore=66.0,
                    piotroski_fscore_raw=6,
                    magic_formula_rank=64.0,
                    lynch_peg_score=61.0,
                    dcf_margin_of_safety_pct=18.0,
                    momentum_score=68.0,
                    technical_score=70.0,
                    dividend_score=55.0,
                    risk_tier="LOW",
                    data_completeness=88.0,
                    quality_flags_json=json.dumps({"earnings": "clean"}),
                ),
                AdjustedMetric(
                    company_id=company.id,
                    period_end=today - timedelta(days=90),
                    adjusted_net_income=1_250_000.0,
                    owner_earnings=1_100_000.0,
                    free_cash_flow=950_000.0,
                    roe_adjusted=0.22,
                    roa_adjusted=0.11,
                    real_eps_growth_pct=0.18,
                ),
                MacroRegime(
                    date=today,
                    policy_rate_pct=0.42,
                    cpi_yoy_pct=0.31,
                    usdtry_rate=39.1,
                    regime="RISK_OFF",
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    return engine


def test_export_mobile_snapshot_writes_expected_tables(source_engine, tmp_path):
    output_path = tmp_path / "mobile_snapshot.db"

    exported_path = export_mobile_snapshot(output_path)
    metadata = validate_mobile_snapshot(exported_path)

    assert exported_path == output_path.resolve()
    assert metadata["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert metadata["snapshot_date"] == "2026-03-19"
    assert metadata["company_count"] == 1

    with sqlite3.connect(exported_path) as connection:
        connection.row_factory = sqlite3.Row

        scoring_row = connection.execute(
            "SELECT ticker, alpha, quality_flags_json, beta, delta FROM scoring_latest"
        ).fetchone()
        assert dict(scoring_row) == {
            "ticker": "TEST1",
            "alpha": pytest.approx(91.0),
            "quality_flags_json": json.dumps({"earnings": "clean"}),
            "beta": pytest.approx(77.0),
            "delta": pytest.approx(66.0),
        }

        adjusted_row = connection.execute(
            "SELECT adjusted_net_income, roe_adjusted FROM adjusted_metrics_latest"
        ).fetchone()
        assert adjusted_row["adjusted_net_income"] == pytest.approx(1_250_000.0)
        assert adjusted_row["roe_adjusted"] == pytest.approx(0.22)

        open_position = connection.execute(
            "SELECT ticker, current_price, target_price FROM open_positions"
        ).fetchone()
        assert dict(open_position) == {
            "ticker": "TEST1",
            "current_price": pytest.approx(101.0),
            "target_price": pytest.approx(120.0),
        }

        price_count = connection.execute(
            "SELECT COUNT(*) FROM price_history_730d"
        ).fetchone()[0]
        assert price_count == 2


def test_export_mobile_snapshot_cli_command(source_engine, tmp_path):
    output_path = tmp_path / "cli_snapshot.db"
    runner = CliRunner()

    result = runner.invoke(cli, ["export-mobile-snapshot", "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Mobile snapshot exported" in result.output


def test_export_mobile_feed_writes_manifest_and_gzip(source_engine, tmp_path):
    feed_dir = tmp_path / "feed"

    result = export_mobile_feed(
        feed_dir,
        base_download_url="https://example.test/mobile-feed",
    )

    manifest_path = feed_dir / "manifest.json"
    snapshot_gzip_path = feed_dir / "mobile_snapshot.db.gz"

    assert result.manifest_path == manifest_path
    assert result.snapshot_path == snapshot_gzip_path
    assert manifest_path.exists()
    assert snapshot_gzip_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["feedVersion"] == 1
    assert manifest["snapshotSchemaVersion"] == SNAPSHOT_SCHEMA_VERSION
    assert manifest["snapshotDate"] == "2026-03-19"
    assert manifest["compression"] == "gzip"
    assert manifest["downloadUrl"] == "https://example.test/mobile-feed/mobile_snapshot.db.gz"
    assert manifest["sizeBytes"] == snapshot_gzip_path.stat().st_size
    assert len(manifest["sha256"]) == 64

    extracted_snapshot = tmp_path / "extracted_snapshot.db"
    with gzip.open(snapshot_gzip_path, "rb") as source:
        extracted_snapshot.write_bytes(source.read())

    metadata = validate_mobile_snapshot(extracted_snapshot)
    assert metadata["snapshot_date"] == "2026-03-19"


def test_export_mobile_feed_cli_command(source_engine, tmp_path):
    feed_dir = tmp_path / "cli_feed"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "export-mobile-feed",
            "--feed-dir",
            str(feed_dir),
            "--base-download-url",
            "https://example.test/mobile-feed",
        ],
    )

    assert result.exit_code == 0
    assert (feed_dir / "manifest.json").exists()
    assert (feed_dir / "mobile_snapshot.db.gz").exists()
    assert "Mobile feed exported" in result.output
