"""SQLAlchemy ORM models for the BIST Stock Picker database.

Defines all 9 tables: companies, daily_prices, financial_statements,
adjusted_metrics, corporate_actions, insider_transactions, scoring_results,
portfolio_selections, macro_regime. All tables include created_at and
updated_at auto-timestamps. Backend: SQLite.
"""

from datetime import UTC, date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _utcnow() -> datetime:
    """Return a naive UTC timestamp for SQLite-compatible DateTime columns."""
    return datetime.now(UTC).replace(tzinfo=None)


def _set_updated_at(mapper, connection, target):
    """Auto-update the updated_at timestamp on every flush."""
    target.updated_at = _utcnow()


class Company(Base):
    """BIST-listed company master record."""

    __tablename__ = "companies"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    ticker: str = Column(String(10), unique=True, nullable=False, index=True)
    name: Optional[str] = Column(String(255))
    company_type: Optional[str] = Column(String(20))  # OPERATING / HOLDING / BANK / INSURANCE / REIT
    sector_bist: Optional[str] = Column(String(100))
    sector_custom: Optional[str] = Column(String(100))
    listing_date: Optional[date] = Column(Date)
    delisting_date: Optional[date] = Column(Date)
    free_float_pct: Optional[float] = Column(Float)
    is_bist100: bool = Column(Boolean, default=False)
    is_ipo: bool = Column(Boolean, default=False)
    ipo_age_months: Optional[int] = Column(Integer)
    is_active: bool = Column(Boolean, default=True)
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    # Relationships
    daily_prices = relationship("DailyPrice", back_populates="company")
    financial_statements = relationship("FinancialStatement", back_populates="company")
    adjusted_metrics = relationship("AdjustedMetric", back_populates="company")
    corporate_actions = relationship("CorporateAction", back_populates="company")
    insider_transactions = relationship("InsiderTransaction", back_populates="company")
    scoring_results = relationship("ScoringResult", back_populates="company")
    portfolio_selections = relationship("PortfolioSelection", back_populates="company")


class DailyPrice(Base):
    """Daily OHLCV price data for a company."""

    __tablename__ = "daily_prices"
    __table_args__ = (
        UniqueConstraint("company_id", "date", name="uq_daily_prices_company_date"),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    company_id: int = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    date: date = Column(Date, nullable=False, index=True)
    open: Optional[float] = Column(Float)
    high: Optional[float] = Column(Float)
    low: Optional[float] = Column(Float)
    close: Optional[float] = Column(Float)
    volume: Optional[int] = Column(Integer)
    adjusted_close: Optional[float] = Column(Float)
    source: Optional[str] = Column(String(20))  # ISYATIRIM / YAHOO
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    company = relationship("Company", back_populates="daily_prices")


class FinancialStatement(Base):
    """Raw financial statement data stored as JSON."""

    __tablename__ = "financial_statements"
    __table_args__ = (
        UniqueConstraint(
            "company_id", "period_end", "period_type", "statement_type", "version",
            name="uq_financial_statements_composite",
        ),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    company_id: int = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    period_end: date = Column(Date, nullable=False)
    period_type: str = Column(String(10), nullable=False)  # Q1 / Q2 / Q3 / ANNUAL
    statement_type: str = Column(String(20), nullable=False)  # INCOME / BALANCE / CASHFLOW
    is_consolidated: bool = Column(Boolean, default=True)
    is_inflation_adj: bool = Column(Boolean, default=False)
    publication_date: Optional[date] = Column(Date)
    version: int = Column(Integer, default=1)
    data_json: Optional[str] = Column(Text)  # Full statement as JSON
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    company = relationship("Company", back_populates="financial_statements")


class AdjustedMetric(Base):
    """Calculated clean financial metrics after IAS 29 and inflation adjustments."""

    __tablename__ = "adjusted_metrics"
    __table_args__ = (
        UniqueConstraint(
            "company_id", "period_end",
            name="uq_adjusted_metrics_company_period",
        ),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    company_id: int = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    period_end: date = Column(Date, nullable=False)
    reported_net_income: Optional[float] = Column(Float)
    monetary_gain_loss: Optional[float] = Column(Float)
    adjusted_net_income: Optional[float] = Column(Float)
    owner_earnings: Optional[float] = Column(Float)
    free_cash_flow: Optional[float] = Column(Float)
    roe_adjusted: Optional[float] = Column(Float)
    roa_adjusted: Optional[float] = Column(Float)
    eps_adjusted: Optional[float] = Column(Float)
    real_eps_growth_pct: Optional[float] = Column(Float)
    related_party_revenue_pct: Optional[float] = Column(Float)
    # Greenwald CapEx decomposition (V2.5 improvement)
    maintenance_capex: Optional[float] = Column(Float)
    growth_capex: Optional[float] = Column(Float)
    # IAS 29 adjustments (V2.5 improvement)
    deferred_tax_stripped: Optional[float] = Column(Float)
    excess_depreciation_addback: Optional[float] = Column(Float)
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    company = relationship("Company", back_populates="adjusted_metrics")


class CorporateAction(Base):
    """Corporate actions: splits, bonus shares, rights issues, dividends, mergers."""

    __tablename__ = "corporate_actions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    company_id: int = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    action_date: date = Column(Date, nullable=False)
    action_type: str = Column(String(20), nullable=False)  # SPLIT / BONUS / RIGHTS / DIVIDEND / MERGER
    adjustment_factor: Optional[float] = Column(Float)
    details_json: Optional[str] = Column(Text)
    source: Optional[str] = Column(String(50))
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    company = relationship("Company", back_populates="corporate_actions")


class InsiderTransaction(Base):
    """Insider buying/selling disclosures from KAP."""

    __tablename__ = "insider_transactions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    company_id: int = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    disclosure_date: date = Column(Date, nullable=False)
    person_name: Optional[str] = Column(String(255))
    person_role: Optional[str] = Column(String(50))  # BOARD / CEO / MAJOR_SHAREHOLDER / RELATED
    transaction_type: Optional[str] = Column(String(10))  # BUY / SELL
    shares: Optional[float] = Column(Float)
    price_per_share: Optional[float] = Column(Float)
    total_value_try: Optional[float] = Column(Float)
    source_url: Optional[str] = Column(String(500))
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    company = relationship("Company", back_populates="insider_transactions")


class ScoringResult(Base):
    """Factor scores and composite scores for each company per scoring date."""

    __tablename__ = "scoring_results"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    company_id: int = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    scoring_date: date = Column(Date, nullable=False)
    model_used: Optional[str] = Column(String(20))  # OPERATING / HOLDING / BANKING / IPO
    buffett_score: Optional[float] = Column(Float)
    graham_score: Optional[float] = Column(Float)
    piotroski_fscore: Optional[float] = Column(Float)
    piotroski_fscore_raw: Optional[int] = Column(Integer)  # Raw 0-9 F-Score (not normalized)
    magic_formula_rank: Optional[float] = Column(Float)
    lynch_peg_score: Optional[float] = Column(Float)
    dcf_margin_of_safety_pct: Optional[float] = Column(Float)
    momentum_score: Optional[float] = Column(Float)
    insider_score: Optional[float] = Column(Float)
    technical_score: Optional[float] = Column(Float)
    dividend_score: Optional[float] = Column(Float)  # Dividend yield + consistency (0-100)
    # Sector-specific model composites (pre-weighted by BankingScorer/HoldingScorer/ReitScorer)
    banking_composite: Optional[float] = Column(Float)
    holding_composite: Optional[float] = Column(Float)
    reit_composite: Optional[float] = Column(Float)
    composite_alpha: Optional[float] = Column(Float)
    composite_beta: Optional[float] = Column(Float)
    composite_delta: Optional[float] = Column(Float)
    risk_tier: Optional[str] = Column(String(10))  # HIGH / MEDIUM / LOW
    data_completeness: Optional[float] = Column(Float)
    quality_flags_json: Optional[str] = Column(Text)
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    company = relationship("Company", back_populates="scoring_results")


class PortfolioSelection(Base):
    """Monthly portfolio picks with entry/exit tracking."""

    __tablename__ = "portfolio_selections"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    portfolio: str = Column(String(10), nullable=False)  # ALPHA / BETA / DELTA
    selection_date: date = Column(Date, nullable=False)
    company_id: int = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    entry_price: Optional[float] = Column(Float)
    composite_score: Optional[float] = Column(Float)
    target_price: Optional[float] = Column(Float)
    stop_loss_price: Optional[float] = Column(Float)
    exit_date: Optional[date] = Column(Date)
    exit_price: Optional[float] = Column(Float)
    exit_reason: Optional[str] = Column(String(20))  # REBALANCE / STOP_LOSS / TARGET / THESIS_BREAK
    return_pct: Optional[float] = Column(Float)
    holding_days: Optional[int] = Column(Integer)
    # Phase 4: portfolio weight for this pick, in [0, 1]. At NORMAL cash state
    # every pick has weight = 1/target_count; as the cash state tightens we
    # scale each weight down so (sum of weights) = 1 - cash_pct.
    weight: Optional[float] = Column(Float)
    # Snapshot of the cash state at selection time (handy for UI + audit so we
    # don't have to cross-join cash_allocation_state by date).
    cash_state: Optional[str] = Column(String(20))
    cash_pct: Optional[float] = Column(Float)
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    company = relationship("Company", back_populates="portfolio_selections")


class MacroRegime(Base):
    """Macro regime indicators and classification."""

    __tablename__ = "macro_regime"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    date: date = Column(Date, nullable=False, unique=True)
    policy_rate_pct: Optional[float] = Column(Float)
    cpi_yoy_pct: Optional[float] = Column(Float)
    usdtry_rate: Optional[float] = Column(Float)
    turkey_cds_5y: Optional[float] = Column(Float)
    # Market Participants Survey (TCMB) 24-month-ahead CPI expectation, decimal
    # (e.g., 0.18 = 18%). Feeds DCF terminal growth: g_terminal = expected + real.
    inflation_expectation_24m_pct: Optional[float] = Column(Float)
    regime: Optional[str] = Column(String(20))  # RISK_ON / RISK_OFF / TRANSITION
    weight_adjustments_json: Optional[str] = Column(Text)
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)


class CashAllocationState(Base):
    """Phase 4: daily persisted state of the portfolio cash-out signal.

    One row per business day. The state machine reads prior rows to evaluate
    hysteresis (up/down confirmation windows) and the minimum holding period
    before deciding whether the current day permits a transition.
    """

    __tablename__ = "cash_allocation_state"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    date: date = Column(Date, nullable=False, unique=True, index=True)

    # Inputs at the time of evaluation -- persisted for auditability and so
    # tests / back-diagnosis don't need to re-run the classifiers.
    market_regime: Optional[str] = Column(String(20))   # BULL_LOW_VOL / BULL_HIGH_VOL / BEAR
    macro_regime: Optional[str] = Column(String(20))    # RISK_ON / NEUTRAL / RISK_OFF
    raw_signal: int = Column(Integer, nullable=False)   # 0..4 combined stress score

    # The state the raw_signal ALONE would pick -- useful to show the user
    # "pending" transitions that are blocked by cooldown / confirmation.
    target_state: str = Column(String(20), nullable=False)

    # The state actually applied after hysteresis + cooldown + step-limit.
    state: str = Column(String(20), nullable=False)     # NORMAL / CAUTION / DEFENSIVE / RISK_OFF
    cash_pct: float = Column(Float, nullable=False)

    # Ancillary metadata for the UI and issue-notification layer.
    days_in_state: int = Column(Integer, nullable=False, default=1)
    last_transition_date: Optional[date] = Column(Date)
    transitioned_today: bool = Column(Boolean, nullable=False, default=False)
    notes: Optional[str] = Column(Text)   # e.g. "held by cooldown", "kill-switch disabled"

    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)


# ── Enhanced Pipeline Tables (V2.5) ──────────────────────────────────────────


class KapEvent(Base):
    """LLM-extracted event data from KAP disclosures."""

    __tablename__ = "kap_events"
    __table_args__ = (
        UniqueConstraint(
            "company_id", "raw_text_hash",
            name="uq_kap_events_company_hash",
        ),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    company_id: int = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    disclosure_date: date = Column(Date, nullable=False, index=True)
    event_type: Optional[str] = Column(String(30))  # NEW_CONTRACT / DIVIDEND / SHARE_BUYBACK / etc.
    sentiment_score: Optional[float] = Column(Float)  # -1.0 to 1.0
    monetary_value: Optional[float] = Column(Float)  # Contract/deal size
    currency: Optional[str] = Column(String(5))  # TRY / USD / EUR
    counterparty: Optional[str] = Column(String(255))
    duration_months: Optional[int] = Column(Integer)
    confidence: Optional[float] = Column(Float)  # LLM confidence 0.0-1.0
    raw_text_hash: str = Column(String(64), nullable=False)  # SHA-256 of disclosure text
    raw_text_preview: Optional[str] = Column(Text)  # First 500 chars for debugging
    llm_response_json: Optional[str] = Column(Text)  # Full LLM JSON response
    llm_model: Optional[str] = Column(String(50))  # e.g. gemini-2.5-flash
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    company = relationship("Company")


class MacroNowcast(Base):
    """Macro nowcasting data: BONC, credit card spending, LLM macro signals."""

    __tablename__ = "macro_nowcast"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    date: date = Column(Date, nullable=False, unique=True, index=True)
    # BONC (Composite Leading Indicators)
    bonc_index: Optional[float] = Column(Float)
    bonc_change_mom: Optional[float] = Column(Float)  # Month-over-month change
    bonc_trend: Optional[str] = Column(String(10))  # RISING / FALLING / FLAT
    # Credit card sectoral spending
    credit_card_spending_json: Optional[str] = Column(Text)  # JSON: {sector: amount}
    credit_card_total_change_pct: Optional[float] = Column(Float)  # Total spending MoM %
    # LLM macro headline analysis
    llm_macro_sentiment: Optional[str] = Column(String(20))  # BULLISH / CAUTIOUS / BEARISH
    sector_impacts_json: Optional[str] = Column(Text)  # JSON: {sector: impact_score}
    headline_count: Optional[int] = Column(Integer)
    llm_confidence: Optional[float] = Column(Float)
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)


class EnhancedSignal(Base):
    """Per-company enhanced signal scores combining all forward-looking factors."""

    __tablename__ = "enhanced_signals"
    __table_args__ = (
        UniqueConstraint(
            "company_id", "scoring_date",
            name="uq_enhanced_signals_company_date",
        ),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    company_id: int = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    scoring_date: date = Column(Date, nullable=False, index=True)
    # Individual enhanced factor scores (0-100 scale)
    event_score: Optional[float] = Column(Float)  # KAP event impact
    insider_cluster_score: Optional[float] = Column(Float)  # Insider cluster + drawdown
    macro_nowcast_score: Optional[float] = Column(Float)  # BONC + credit card
    analyst_tone_score: Optional[float] = Column(Float)  # LLM analyst tone
    # Composites
    enhanced_composite: Optional[float] = Column(Float)  # Weighted combo of above
    classic_composite_alpha: Optional[float] = Column(Float)  # Copied from ScoringResult
    blended_alpha: Optional[float] = Column(Float)  # Classic + Enhanced blend
    created_at: datetime = Column(DateTime, default=_utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    company = relationship("Company")


# Register the updated_at auto-setter for all models
for model_class in [
    Company, DailyPrice, FinancialStatement, AdjustedMetric,
    CorporateAction, InsiderTransaction, ScoringResult,
    PortfolioSelection, MacroRegime,
    KapEvent, MacroNowcast, EnhancedSignal,
]:
    event.listen(model_class, "before_update", _set_updated_at)
