"""FastAPI application exposing read-only endpoints for the Android app."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import or_

from bist_picker import read_service
from bist_picker.api.schemas import (
    AdjustedMetricsResponse,
    CashStateSnapshot,
    CompanyInfoResponse,
    HealthResponse,
    HomePerformance,
    HomeResponse,
    LatestScoresResponse,
    MacroSnapshot,
    OpenPosition,
    PortfolioHistoryItem,
    PricePoint,
    ScoringItem,
    ScoringListResponse,
    ScoringOptionsResponse,
    ScoringSummary,
    ScoringViewMode,
    StockDetailResponse,
    StockPositionResponse,
    StockSearchItem,
)
from bist_picker.db.connection import ensure_runtime_db_ready, get_session
from bist_picker.db.schema import Company

DEFAULT_PAGE_SIZE = 40
MAX_PAGE_SIZE = 100
RESEARCH_BUCKETS = {
    "Quality Shadow",
    "Free-Float Shadow",
    "Non-Core Research",
    "Data-Unscorable",
}

@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_runtime_db_ready()
    yield


app = FastAPI(
    title="BIST Picker Mobile API",
    version="0.1.0",
    lifespan=lifespan,
)


def _normalize_value(value: Any) -> Any:
    """Convert pandas/SQLAlchemy scalars into JSON-friendly Python values."""
    if value is None:
        return None
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


def _frame_records(frame: pd.DataFrame, fields: list[str]) -> list[dict[str, Any]]:
    """Return a normalized record list from a dataframe."""
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for row in frame[fields].to_dict(orient="records"):
        records.append({field: _normalize_value(value) for field, value in row.items()})
    return records


def _to_bool(value: bool | None) -> bool:
    return bool(value) if value is not None else False


def _apply_view_mode(
    frame: pd.DataFrame,
    view_mode: ScoringViewMode,
    min_score: float | None,
) -> pd.DataFrame:
    """Apply the mobile view mode over the full scoring snapshot."""
    result = frame.copy()
    if view_mode == ScoringViewMode.ALPHA_X and min_score is not None:
        result = result[
            result["alpha_x_score"].notna()
            & (result["alpha_x_score"] >= float(min_score))
        ].copy()
    elif min_score is not None:
        result = result[
            result["alpha"].notna()
            & (result["alpha"] >= float(min_score))
        ].copy()

    if view_mode == ScoringViewMode.ALPHA_CORE:
        result = result[result["alpha_core_eligible"]].copy()
    elif view_mode == ScoringViewMode.ALPHA_X:
        result = result[result["alpha_x_eligible"]].copy()
        result = result.sort_values(
            ["alpha_x_score", "alpha_x_confidence", "ticker"],
            ascending=[False, False, True],
            na_position="last",
        )
    elif view_mode == ScoringViewMode.RESEARCH:
        result = result[result["alpha_research_bucket"].isin(RESEARCH_BUCKETS)].copy()
    elif view_mode == ScoringViewMode.MODEL:
        result = result.sort_values(
            ["ranking_score", "model_score", "alpha", "ticker"],
            ascending=[False, False, False, True],
            na_position="last",
        )
    return result.reset_index(drop=True)


def _build_scoring_summary(full_df: pd.DataFrame, current_view_count: int) -> ScoringSummary:
    """Build the summary counters shown above the mobile scoring list."""
    bucket_series = full_df["alpha_research_bucket"].fillna("Excluded") if not full_df.empty else pd.Series(dtype=str)
    return ScoringSummary(
        total=int(len(full_df)),
        alpha_core=int(full_df["alpha_core_eligible"].fillna(False).sum()) if not full_df.empty else 0,
        alpha_x=int(full_df["alpha_x_eligible"].fillna(False).sum()) if not full_df.empty else 0,
        research=int(bucket_series.isin(RESEARCH_BUCKETS).sum()),
        quality_shadow=int((bucket_series == "Quality Shadow").sum()),
        free_float_shadow=int((bucket_series == "Free-Float Shadow").sum()),
        non_core_research=int((bucket_series == "Non-Core Research").sum()),
        data_unscorable=int((bucket_series == "Data-Unscorable").sum()),
        current_view_count=current_view_count,
    )


def _company_info_or_404(ticker: str) -> dict[str, Any]:
    info = read_service.get_company_info(ticker)
    if info is None:
        raise HTTPException(status_code=404, detail=f"{ticker.upper()} bulunamadi")
    return info


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/v1/home", response_model=HomeResponse)
def get_home() -> HomeResponse:
    performance = read_service.get_all_portfolio_performance() or {}
    open_positions = read_service.get_open_positions()
    history = read_service.get_portfolio_history()
    macro = read_service.get_latest_macro()
    cash = read_service.get_latest_cash_state()

    return HomeResponse(
        performance=HomePerformance(
            total_return_avg=_normalize_value(performance.get("total_return_avg")),
            active_return_avg=_normalize_value(performance.get("active_return_avg")),
            win_rate=_normalize_value(performance.get("win_rate")),
            benchmark_ytd=_normalize_value(performance.get("benchmark_ytd")),
        ),
        open_positions=[
            OpenPosition(**record)
            for record in _frame_records(
                open_positions,
                [
                    "portfolio",
                    "ticker",
                    "name",
                    "company_id",
                    "entry_price",
                    "current_price",
                    "pnl_pct",
                    "target_price",
                    "stop_loss_price",
                    "composite_score",
                    "selection_date",
                    "days_held",
                ],
            )
        ],
        portfolio_history=[
            PortfolioHistoryItem(**record)
            for record in _frame_records(
                history,
                [
                    "portfolio",
                    "ticker",
                    "name",
                    "selection_date",
                    "exit_date",
                    "entry_price",
                    "exit_price",
                    "pnl_pct",
                ],
            )
        ],
        macro=MacroSnapshot(**{key: _normalize_value(value) for key, value in (macro or {}).items()}) if macro else None,
        cash=CashStateSnapshot(**{key: _normalize_value(value) for key, value in (cash or {}).items()}) if cash else None,
    )


@app.get("/v1/scoring/options", response_model=ScoringOptionsResponse)
def get_scoring_options() -> ScoringOptionsResponse:
    return ScoringOptionsResponse(
        view_modes=[
            ScoringViewMode.ALPHA_CORE,
            ScoringViewMode.ALPHA_X,
            ScoringViewMode.ALL,
            ScoringViewMode.RESEARCH,
            ScoringViewMode.MODEL,
        ],
        dates=[value.isoformat() for value in read_service.get_scoring_dates()],
        company_types=read_service.get_company_types(),
        sectors=read_service.get_sectors(),
        risk_tiers=["LOW", "MEDIUM", "HIGH"],
    )


@app.get("/v1/scoring/list", response_model=ScoringListResponse)
def get_scoring_list(
    view_mode: ScoringViewMode = Query(default=ScoringViewMode.ALPHA_CORE),
    scoring_date: date | None = Query(default=None),
    company_type: str | None = Query(default=None),
    bist100: bool | None = Query(default=None),
    sector: str | None = Query(default=None),
    risk: str | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0, le=100),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> ScoringListResponse:
    full_df = read_service.get_scoring_results(
        scoring_date=scoring_date,
        company_type=company_type,
        is_bist100=bist100,
        sector_custom=sector,
        risk_tier=risk,
        min_score=None,
        alpha_eligible_only=False,
    )

    if full_df.empty:
        summary = _build_scoring_summary(full_df, current_view_count=0)
        effective_date = scoring_date or read_service.get_latest_scoring_date()
        return ScoringListResponse(
            view_mode=view_mode,
            scoring_date=effective_date.isoformat() if effective_date else None,
            page=page,
            page_size=page_size,
            total=0,
            items=[],
            summary=summary,
        )

    view_df = _apply_view_mode(full_df, view_mode=view_mode, min_score=min_score)
    total = len(view_df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = view_df.iloc[start:end].copy()
    summary = _build_scoring_summary(full_df, current_view_count=total)

    items = [
        ScoringItem(
            ticker=str(record["ticker"]),
            name=record.get("name"),
            type=record.get("type"),
            sector=record.get("sector"),
            ranking_score=record.get("ranking_score"),
            ranking_source=record.get("ranking_source"),
            model_score=record.get("model_score"),
            alpha=record.get("alpha"),
            alpha_x_score=record.get("alpha_x_score"),
            alpha_x_rank=record.get("alpha_x_rank"),
            alpha_core_eligible=_to_bool(record.get("alpha_core_eligible")),
            alpha_x_eligible=_to_bool(record.get("alpha_x_eligible")),
            alpha_reason=record.get("alpha_reason"),
            alpha_primary_blocker=record.get("alpha_primary_blocker"),
            alpha_research_bucket=record.get("alpha_research_bucket"),
            risk=record.get("risk"),
            data_completeness=record.get("data_completeness"),
            free_float_pct=record.get("free_float_pct"),
            avg_volume_try=record.get("avg_volume_try"),
            alpha_snapshot_streak=record.get("alpha_snapshot_streak"),
        )
        for record in _frame_records(
            page_df,
            [
                "ticker",
                "name",
                "type",
                "sector",
                "ranking_score",
                "ranking_source",
                "model_score",
                "alpha",
                "alpha_x_score",
                "alpha_x_rank",
                "alpha_core_eligible",
                "alpha_x_eligible",
                "alpha_reason",
                "alpha_primary_blocker",
                "alpha_research_bucket",
                "risk",
                "data_completeness",
                "free_float_pct",
                "avg_volume_try",
                "alpha_snapshot_streak",
            ],
        )
    ]
    effective_date = scoring_date or read_service.get_latest_scoring_date()

    return ScoringListResponse(
        view_mode=view_mode,
        scoring_date=effective_date.isoformat() if effective_date else None,
        page=page,
        page_size=page_size,
        total=total,
        items=items,
        summary=summary,
    )


@app.get("/v1/stocks/search", response_model=list[StockSearchItem])
def search_stocks(
    q: str = Query(default="", min_length=0),
    limit: int = Query(default=20, ge=1, le=50),
) -> list[StockSearchItem]:
    session = get_session()
    try:
        query = session.query(Company)
        cleaned = q.strip()
        if cleaned:
            like = f"%{cleaned.upper()}%"
            name_like = f"%{cleaned}%"
            query = query.filter(
                or_(
                    Company.ticker.ilike(like),
                    Company.name.ilike(name_like),
                )
            )
        rows = query.order_by(Company.ticker).limit(limit).all()
        return [
            StockSearchItem(
                ticker=row.ticker,
                name=row.name,
                company_type=row.company_type,
                sector=row.sector_custom or row.sector_bist,
                is_active=bool(row.is_active),
            )
            for row in rows
        ]
    finally:
        session.close()


@app.get("/v1/stocks/{ticker}", response_model=StockDetailResponse)
def get_stock_detail(ticker: str) -> StockDetailResponse:
    info = _company_info_or_404(ticker)
    open_position = read_service.get_stock_position(ticker)
    latest_scores = read_service.get_factor_scores(ticker)
    adjusted_metrics = read_service.get_adjusted_metrics(ticker)

    return StockDetailResponse(
        company=CompanyInfoResponse(
            **{key: _normalize_value(value) for key, value in info.items()}
        ),
        open_position=StockPositionResponse(
            **{key: _normalize_value(value) for key, value in open_position.items()}
        ) if open_position else None,
        latest_scores=LatestScoresResponse(
            **{key: _normalize_value(value) for key, value in latest_scores.items()}
        ) if latest_scores else None,
        adjusted_metrics=AdjustedMetricsResponse(
            **{key: _normalize_value(value) for key, value in adjusted_metrics.items()}
        ) if adjusted_metrics else None,
    )


@app.get("/v1/stocks/{ticker}/prices", response_model=list[PricePoint])
def get_stock_prices(
    ticker: str,
    days: int = Query(default=365, ge=1, le=3650),
) -> list[PricePoint]:
    _company_info_or_404(ticker)
    price_df = read_service.get_price_history(ticker, days=days)
    return [PricePoint(**record) for record in _frame_records(
        price_df,
        ["date", "open", "high", "low", "close", "volume", "adjusted_close"],
    )]


def run() -> None:
    """Run the mobile API with Uvicorn."""
    import uvicorn

    uvicorn.run("bist_picker.api.app:app", host="0.0.0.0", port=8000, reload=False)
