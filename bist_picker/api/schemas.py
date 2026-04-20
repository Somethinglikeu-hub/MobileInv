"""Response models for the BIST Picker mobile API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ScoringViewMode(str, Enum):
    ALPHA_CORE = "ALPHA_CORE"
    ALPHA_X = "ALPHA_X"
    ALL = "ALL"
    RESEARCH = "RESEARCH"
    MODEL = "MODEL"


class HealthResponse(BaseModel):
    status: str


class HomePerformance(BaseModel):
    total_return_avg: float | None = None
    active_return_avg: float | None = None
    win_rate: float | None = None
    benchmark_ytd: float | None = None


class DcfBreakdown(BaseModel):
    """Phase 5: DCF valuation inputs + result surfaced to APK."""

    intrinsic_value: float | None = None
    growth_rate_pct: float | None = None
    discount_rate_pct: float | None = None
    terminal_growth_pct: float | None = None
    margin_of_safety_pct: float | None = None


class ReasonFactor(BaseModel):
    """One entry in the 'why selected' top-factors list."""

    factor: str
    label: str
    value: float


class PickDetail(BaseModel):
    """Phase 5: per-pick transparency block attached to each OpenPosition."""

    reason_top_factors: list[ReasonFactor] = []
    red_flags: list[str] = []
    dcf: DcfBreakdown | None = None
    stop_loss_price: float | None = None
    stop_pct_from_entry: float | None = None


class OpenPosition(BaseModel):
    portfolio: str | None = None
    ticker: str
    name: str | None = None
    company_id: int | None = None
    entry_price: float | None = None
    current_price: float | None = None
    pnl_pct: float | None = None
    target_price: float | None = None
    stop_loss_price: float | None = None
    composite_score: float | None = None
    selection_date: str | None = None
    days_held: int | None = None
    detail: PickDetail | None = None


class PortfolioHistoryItem(BaseModel):
    portfolio: str | None = None
    ticker: str
    name: str | None = None
    selection_date: str | None = None
    exit_date: str | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    pnl_pct: float | None = None


class MacroSnapshot(BaseModel):
    date: str | None = None
    policy_rate_pct: float | None = None
    policy_rate_date: str | None = None
    cpi_yoy_pct: float | None = None
    cpi_yoy_date: str | None = None
    usdtry_rate: float | None = None
    usdtry_date: str | None = None
    turkey_cds_5y: float | None = None
    turkey_cds_5y_date: str | None = None
    regime: str | None = None
    regime_date: str | None = None


class CashStateSnapshot(BaseModel):
    """Phase 4: current cash-out state surfaced to the APK."""

    date: str | None = None
    state: str | None = None            # NORMAL / CAUTION / DEFENSIVE / RISK_OFF
    cash_pct: float | None = None       # 0.0, 0.25, 0.50, 0.75
    target_state: str | None = None     # what the signal alone would pick
    market_regime: str | None = None
    macro_regime: str | None = None
    raw_signal: int | None = None
    days_in_state: int | None = None
    last_transition_date: str | None = None
    transitioned_today: bool | None = None
    notes: str | None = None


class HomeResponse(BaseModel):
    performance: HomePerformance
    open_positions: list[OpenPosition]
    portfolio_history: list[PortfolioHistoryItem]
    macro: MacroSnapshot | None = None
    cash: CashStateSnapshot | None = None


class ScoringOptionsResponse(BaseModel):
    view_modes: list[ScoringViewMode]
    dates: list[str]
    company_types: list[str]
    sectors: list[str]
    risk_tiers: list[str]


class ScoringSummary(BaseModel):
    total: int
    alpha_core: int
    alpha_x: int
    research: int
    quality_shadow: int
    free_float_shadow: int
    non_core_research: int
    data_unscorable: int
    current_view_count: int


class ScoringItem(BaseModel):
    ticker: str
    name: str | None = None
    type: str | None = None
    sector: str | None = None
    ranking_score: float | None = None
    ranking_source: str | None = None
    model_score: float | None = None
    alpha: float | None = None
    alpha_x_score: float | None = None
    alpha_x_rank: float | None = None
    alpha_core_eligible: bool = False
    alpha_x_eligible: bool = False
    alpha_reason: str | None = None
    alpha_primary_blocker: str | None = None
    alpha_research_bucket: str | None = None
    risk: str | None = None
    data_completeness: float | None = None
    free_float_pct: float | None = None
    avg_volume_try: float | None = None
    alpha_snapshot_streak: int | None = None


class ScoringListResponse(BaseModel):
    view_mode: ScoringViewMode
    scoring_date: str | None = None
    page: int
    page_size: int
    total: int
    items: list[ScoringItem]
    summary: ScoringSummary


class StockSearchItem(BaseModel):
    ticker: str
    name: str | None = None
    company_type: str | None = None
    sector: str | None = None
    is_active: bool


class CompanyInfoResponse(BaseModel):
    id: int
    ticker: str
    name: str | None = None
    company_type: str | None = None
    sector_bist: str | None = None
    sector_custom: str | None = None
    is_bist100: bool | None = None
    is_ipo: bool | None = None
    free_float_pct: float | None = None
    listing_date: str | None = None
    is_active: bool | None = None


class StockPositionResponse(BaseModel):
    portfolio: str | None = None
    entry_price: float | None = None
    target_price: float | None = None
    stop_loss_price: float | None = None
    selection_date: str | None = None
    composite_score: float | None = None


class LatestScoresResponse(BaseModel):
    scoring_date: str | None = None
    model_used: str | None = None
    data_completeness: float | None = None
    buffett: float | None = None
    graham: float | None = None
    piotroski: float | None = None
    piotroski_raw: int | None = None
    magic_formula: float | None = None
    lynch_peg: float | None = None
    dcf_mos: float | None = None
    momentum: float | None = None
    insider: float | None = None
    technical: float | None = None
    dividend: float | None = None
    alpha: float | None = None
    beta: float | None = None
    delta: float | None = None
    risk_tier: str | None = None
    quality_flags: dict[str, Any] | list[Any] | None = None


class AdjustedMetricsResponse(BaseModel):
    period_end: str | None = None
    reported_net_income: float | None = None
    monetary_gain_loss: float | None = None
    adjusted_net_income: float | None = None
    owner_earnings: float | None = None
    free_cash_flow: float | None = None
    roe_adjusted: float | None = None
    roa_adjusted: float | None = None
    eps_adjusted: float | None = None
    real_eps_growth_pct: float | None = None
    related_party_revenue_pct: float | None = None
    maintenance_capex: float | None = None
    growth_capex: float | None = None


class StockDetailResponse(BaseModel):
    company: CompanyInfoResponse
    open_position: StockPositionResponse | None = None
    latest_scores: LatestScoresResponse | None = None
    adjusted_metrics: AdjustedMetricsResponse | None = None


class PricePoint(BaseModel):
    date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None
    adjusted_close: float | None = None

