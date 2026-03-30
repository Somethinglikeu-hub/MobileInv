"""Regression tests for point-in-time universe and selection behavior."""

from datetime import date
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bist_picker.db.schema import Base, Company, PortfolioSelection, ScoringResult
from bist_picker.portfolio.selector import PortfolioSelector
from bist_picker.portfolio.universes import UniverseBuilder, _UNIVERSE_CONFIG


def test_universe_builder_uses_latest_score_on_or_before_scoring_date():
    """Future scores must not leak into an earlier universe snapshot."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    company = Company(
        ticker="TEST1",
        is_active=True,
        free_float_pct=40.0,
        company_type="OPERATING",
    )
    session.add(company)
    session.commit()

    session.add_all(
        [
            ScoringResult(
                company_id=company.id,
                scoring_date=date(2025, 1, 1),
                composite_alpha=10.0,
            ),
            ScoringResult(
                company_id=company.id,
                scoring_date=date(2026, 1, 1),
                composite_alpha=90.0,
            ),
        ]
    )
    session.commit()

    rows = UniverseBuilder(scoring_date=date(2025, 6, 1))._get_latest_scores(session)

    assert len(rows) == 1
    _, score_row, _ = rows[0]
    assert score_row.scoring_date == date(2025, 1, 1)


def test_selector_skips_candidates_without_price_history():
    """A candidate with no entry price should not become a portfolio pick."""
    selector = PortfolioSelector(scoring_date=date(2026, 2, 1))
    selector._universe.get_universe = lambda portfolio, session: [1, 2, 3, 4, 5, 6]
    selector._fetch_candidates = lambda universe_ids, score_col, session: [
        {"company_id": 1, "ticker": "NOPRICE", "score": 100.0, "sector_custom": "A", "company_type": "OPERATING", "dcf_mos": None},
        {"company_id": 2, "ticker": "GOOD2", "score": 99.0, "sector_custom": "B", "company_type": "OPERATING", "dcf_mos": None},
        {"company_id": 3, "ticker": "GOOD3", "score": 98.0, "sector_custom": "C", "company_type": "OPERATING", "dcf_mos": None},
        {"company_id": 4, "ticker": "GOOD4", "score": 97.0, "sector_custom": "D", "company_type": "OPERATING", "dcf_mos": None},
        {"company_id": 5, "ticker": "GOOD5", "score": 96.0, "sector_custom": "E", "company_type": "OPERATING", "dcf_mos": None},
        {"company_id": 6, "ticker": "GOOD6", "score": 95.0, "sector_custom": "F", "company_type": "OPERATING", "dcf_mos": None},
    ]
    selector._get_latest_price = lambda company_id, session: None if company_id == 1 else 100.0
    selector._compute_atr_stop = lambda company_id, entry_price, session: 82.0
    selector._reduce_correlation = lambda picks, remaining_candidates, sector_counts, bank_count, max_corr, session: picks

    picks = selector.select("ALPHA", session=None, current_holdings=None)

    assert [pick["ticker"] for pick in picks] == ["GOOD2", "GOOD3", "GOOD4"]


def test_alpha_universe_rejects_high_risk_candidate_even_with_good_scores():
    """Research-driven ALPHA rules should exclude HIGH risk names."""
    builder = UniverseBuilder(scoring_date=date(2026, 2, 1))
    company = SimpleNamespace(
        is_bist100=False,
        free_float_pct=40.0,
        is_ipo=False,
        ipo_age_months=None,
        company_type="OPERATING",
    )
    score = SimpleNamespace(
        risk_tier="HIGH",
        data_completeness=85.0,
        piotroski_fscore_raw=7,
        piotroski_fscore=80.0,
    )

    assert builder._passes_filters(
        company=company,
        score=score,
        latest_metric=None,
        avg_volume_try=20_000_000.0,
        cfg=_UNIVERSE_CONFIG["ALPHA"],
    ) is False


def test_alpha_universe_requires_stronger_piotroski_floor():
    """ALPHA should require a higher raw Piotroski floor than before."""
    builder = UniverseBuilder(scoring_date=date(2026, 2, 1))
    company = SimpleNamespace(
        is_bist100=False,
        free_float_pct=40.0,
        is_ipo=False,
        ipo_age_months=None,
        company_type="OPERATING",
    )
    score = SimpleNamespace(
        risk_tier="MEDIUM",
        data_completeness=85.0,
        piotroski_fscore_raw=4,
        piotroski_fscore=80.0,
    )

    assert builder._passes_filters(
        company=company,
        score=score,
        latest_metric=None,
        avg_volume_try=20_000_000.0,
        cfg=_UNIVERSE_CONFIG["ALPHA"],
    ) is False


def test_alpha_universe_rejects_latest_loss_plus_negative_owner_earnings():
    """ALPHA should reject names whose latest earnings and owner earnings are both negative."""
    builder = UniverseBuilder(scoring_date=date(2026, 3, 25))
    company = SimpleNamespace(
        is_bist100=False,
        free_float_pct=40.0,
        is_ipo=False,
        ipo_age_months=None,
        company_type="OPERATING",
    )
    score = SimpleNamespace(
        risk_tier="LOW",
        data_completeness=85.0,
        piotroski_fscore_raw=6,
        piotroski_fscore=80.0,
    )
    latest_metric = SimpleNamespace(
        adjusted_net_income=-74_273_598.0,
        owner_earnings=-42_616_463.8,
    )

    assert builder._passes_filters(
        company=company,
        score=score,
        latest_metric=latest_metric,
        avg_volume_try=20_000_000.0,
        cfg=_UNIVERSE_CONFIG["ALPHA"],
    ) is False


def test_select_and_store_replaces_same_day_snapshot():
    """Re-running selection on the same day should not accumulate extra rows."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    companies = [
        Company(
            ticker=f"TEST{i}",
            is_active=True,
            free_float_pct=40.0,
            company_type="OPERATING",
        )
        for i in range(1, 6)
    ]
    session.add_all(companies)
    session.commit()

    scoring_date = date(2026, 3, 16)

    # Stale same-day snapshot with four names.
    for idx, company in enumerate(companies[:4], start=1):
        session.add(
            PortfolioSelection(
                portfolio="ALPHA",
                selection_date=scoring_date,
                company_id=company.id,
                entry_price=100.0,
                composite_score=90.0 - idx,
                target_price=120.0,
                stop_loss_price=82.0,
            )
        )
    session.commit()

    selector = PortfolioSelector(scoring_date=scoring_date)
    selector._get_latest_price = lambda company_id, session: 100.0
    selector.select_all = lambda session: {
        "alpha": [
            {
                "company_id": companies[1].id,
                "ticker": companies[1].ticker,
                "score": 95.0,
                "rank": 1,
                "entry_price": 100.0,
                "target_price": 120.0,
                "stop_loss": 82.0,
            },
            {
                "company_id": companies[2].id,
                "ticker": companies[2].ticker,
                "score": 94.0,
                "rank": 2,
                "entry_price": 100.0,
                "target_price": 120.0,
                "stop_loss": 82.0,
            },
            {
                "company_id": companies[4].id,
                "ticker": companies[4].ticker,
                "score": 93.0,
                "rank": 3,
                "entry_price": 100.0,
                "target_price": 120.0,
                "stop_loss": 82.0,
            },
        ]
    }

    selector.select_and_store(session)

    rows = (
        session.query(PortfolioSelection)
        .filter(
            PortfolioSelection.portfolio == "ALPHA",
            PortfolioSelection.selection_date == scoring_date,
        )
        .order_by(PortfolioSelection.composite_score.desc())
        .all()
    )

    assert len(rows) == 3
    assert {row.company_id for row in rows} == {
        companies[1].id,
        companies[2].id,
        companies[4].id,
    }
