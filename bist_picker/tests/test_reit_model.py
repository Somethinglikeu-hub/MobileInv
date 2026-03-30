import json
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.db.schema import (
    AdjustedMetric,
    Base,
    Company,
    CorporateAction,
    DailyPrice,
    FinancialStatement,
)
from bist_picker.scoring.models.reit import ReitScorer


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()


def _balance_json(total_assets: float, total_equity: float) -> str:
    return json.dumps(
        [
            {"item_code": "1BL", "desc_tr": "TOPLAM VARLIKLAR", "value": total_assets},
            {"item_code": "2N", "desc_tr": "OZKAYNAKLAR", "value": total_equity},
        ]
    )


def _income_json(net_sales: float) -> str:
    return json.dumps(
        [
            {"item_code": "3C", "desc_tr": "Satis Gelirleri", "value": net_sales},
        ]
    )


def test_reit_scorer_uses_close_price_and_trailing_dividend_yield(session):
    scoring_date = date(2026, 3, 25)
    period_end = date(2025, 12, 31)

    company = Company(
        ticker="YGYOX",
        name="Test REIT",
        company_type="REIT",
        sector_bist="REIT",
        sector_custom="REIT",
        is_active=True,
    )
    session.add(company)
    session.flush()

    session.add(
        DailyPrice(
            company_id=company.id,
            date=scoring_date,
            close=20.0,
            adjusted_close=None,
            high=20.5,
            low=19.5,
            volume=100_000,
            source="YAHOO_TEST",
        )
    )
    session.add(
        AdjustedMetric(
            company_id=company.id,
            period_end=period_end,
            reported_net_income=20_000_000.0,
            adjusted_net_income=20_000_000.0,
            eps_adjusted=2.0,
        )
    )
    session.add_all(
        [
            FinancialStatement(
                company_id=company.id,
                period_end=period_end,
                period_type="ANNUAL",
                statement_type="BALANCE",
                publication_date=date(2026, 3, 10),
                data_json=_balance_json(260_000_000.0, 200_000_000.0),
            ),
            FinancialStatement(
                company_id=company.id,
                period_end=period_end,
                period_type="ANNUAL",
                statement_type="INCOME",
                publication_date=date(2026, 3, 10),
                data_json=_income_json(50_000_000.0),
            ),
            CorporateAction(
                company_id=company.id,
                action_date=date(2025, 6, 1),
                action_type="DIVIDEND",
                adjustment_factor=1.2,
                source="TEST",
            ),
        ]
    )
    session.commit()

    scorer = ReitScorer()
    metrics = scorer._extract_metrics(company.id, session, scoring_date=scoring_date)

    assert metrics is not None
    assert metrics["pb"] == pytest.approx(1.0)
    assert metrics["roe"] == pytest.approx(0.10)
    assert metrics["net_margin"] == pytest.approx(0.40)
    assert metrics["debt_equity"] == pytest.approx(0.30)
    assert metrics["dividend_yield"] == pytest.approx(0.06)

    scores = scorer.score_all(session, scoring_date=scoring_date)
    assert company.id in scores
    assert scores[company.id]["reit_composite"] is not None
    assert scores[company.id]["data_completeness"] == pytest.approx(100.0)
