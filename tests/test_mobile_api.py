"""Integration-style tests for the mobile FastAPI layer."""

from datetime import date, timedelta

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import bist_picker.api.app as api_app
import bist_picker.read_service as read_service
from bist_picker.db.schema import (
    Base,
    Company,
    DailyPrice,
    MacroRegime,
    PortfolioSelection,
    ScoringResult,
)


@pytest.fixture
def mobile_api_client(monkeypatch):
    """Provide a FastAPI client backed by an isolated in-memory database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    monkeypatch.setattr(read_service, "get_engine", lambda: engine)
    monkeypatch.setattr(api_app, "ensure_runtime_db_ready", lambda: engine)
    monkeypatch.setattr(api_app, "get_session", lambda: session_factory())

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
                ScoringResult(
                    company_id=company.id,
                    scoring_date=today,
                    model_used="OPERATING",
                    composite_alpha=91.0,
                    buffett_score=80.0,
                    graham_score=72.0,
                    piotroski_fscore=66.0,
                    piotroski_fscore_raw=6,
                    momentum_score=68.0,
                    technical_score=70.0,
                    risk_tier="LOW",
                    data_completeness=88.0,
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

    with TestClient(api_app.app) as client:
        yield client


def test_home_endpoint_matches_expected_shape(mobile_api_client):
    response = mobile_api_client.get("/v1/home")

    assert response.status_code == 200
    payload = response.json()
    assert payload["open_positions"][0]["ticker"] == "TEST1"
    assert payload["portfolio_history"] == []
    assert payload["macro"]["regime"] == "RISK_OFF"


def test_scoring_list_filters_and_returns_summary(mobile_api_client):
    response = mobile_api_client.get(
        "/v1/scoring/list",
        params={"view_mode": "ALPHA_CORE", "page": 1, "page_size": 20},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["summary"]["alpha_core"] == 1
    assert payload["items"][0]["ticker"] == "TEST1"
    assert payload["items"][0]["alpha"] == pytest.approx(91.0)


def test_stock_detail_and_price_history_endpoints(mobile_api_client):
    detail_response = mobile_api_client.get("/v1/stocks/TEST1")
    price_response = mobile_api_client.get("/v1/stocks/TEST1/prices", params={"days": 365})

    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["company"]["ticker"] == "TEST1"
    assert detail["open_position"]["target_price"] == pytest.approx(120.0)
    assert detail["latest_scores"]["alpha"] == pytest.approx(91.0)

    assert price_response.status_code == 200
    prices = price_response.json()
    assert len(prices) == 1
    assert prices[0]["adjusted_close"] == pytest.approx(101.0)
