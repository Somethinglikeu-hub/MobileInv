"""Regression tests for the portfolio selector."""

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.db.schema import Base, Company, DailyPrice, ScoringResult
from bist_picker.portfolio.selector import PortfolioSelector


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    Session_ = sessionmaker(bind=engine)
    sess = Session_()
    yield sess
    sess.close()


class TestPortfolioSelector:
    def _add_company(
        self,
        session,
        ticker: str,
        sector_bist: str,
        sector_custom: str | None = None,
        company_type: str = "OPERATING",
    ) -> int:
        company = Company(
            ticker=ticker,
            name=ticker,
            company_type=company_type,
            sector_bist=sector_bist,
            sector_custom=sector_custom,
            is_active=True,
        )
        session.add(company)
        session.flush()
        return company.id

    def _add_price(self, session, company_id: int, as_of: date, close: float) -> None:
        session.add(
            DailyPrice(
                company_id=company_id,
                date=as_of,
                close=close,
                adjusted_close=close,
                high=close,
                low=close,
                volume=1000,
            )
        )

    def _add_score(
        self,
        session,
        company_id: int,
        as_of: date,
        composite_alpha: float,
        technical_score: float = 80.0,
        dcf_margin_of_safety_pct: float = 20.0,
    ) -> None:
        session.add(
            ScoringResult(
                company_id=company_id,
                scoring_date=as_of,
                model_used="OPERATING",
                composite_alpha=composite_alpha,
                technical_score=technical_score,
                dcf_margin_of_safety_pct=dcf_margin_of_safety_pct,
                data_completeness=100.0,
            )
        )

    def test_sector_cap_uses_bist_sector_when_custom_sector_missing(self, session, monkeypatch):
        """Selector should still diversify when sector_custom is absent in DB rows."""
        as_of = date(2026, 2, 1)
        company_ids = [
            self._add_company(session, "AAA1", "Technology"),
            self._add_company(session, "AAA2", "Technology"),
            self._add_company(session, "AAA3", "Technology"),
            self._add_company(session, "BBB1", "Industrial"),
        ]
        scores = [95.0, 90.0, 85.0, 80.0]
        for cid, score in zip(company_ids, scores):
            self._add_price(session, cid, as_of, close=10.0 + score)
            self._add_score(session, cid, as_of, composite_alpha=score)
        session.commit()

        selector = PortfolioSelector(scoring_date=as_of)
        monkeypatch.setattr(
            selector._universe,
            "get_universe",
            lambda portfolio, db_session: company_ids,
        )

        picks = selector.select("ALPHA", session)
        picked_tickers = {pick["ticker"] for pick in picks}

        # Phase 3 tightening: max_per_sector=2. Of the 3 Tech candidates only
        # the top 2 by composite score survive the sector cap; the single
        # Industrial candidate also makes it in.
        assert len(picks) == 3
        assert picked_tickers == {"AAA1", "AAA2", "BBB1"}
