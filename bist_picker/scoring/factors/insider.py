"""Insider transaction signal factor for BIST Stock Picker.

Pulls insider buying/selling disclosures from the insider_transactions table
and produces a net-buying signal for 3-month and 6-month windows.

Logic:
  1. Sum role-weighted BUY amounts minus role-weighted SELL amounts in TRY
     over each lookback window (3m and 6m).
  2. Normalise the raw TRY net-buy by approximate market capitalisation so
     that a ₺10M purchase means more for a ₺500M company than for a ₺50B one.
     Market cap is approximated as latest_close × shares_outstanding, where
     shares_outstanding = adjusted_net_income / eps_adjusted (Turkey par = ₺1).
     If market cap cannot be derived, raw TRY amounts are kept and the
     cross-sectional percentile normalisation at score_all level handles it.
  3. Combine 3m (60%) and 6m (40%) into a single insider_raw score.
  4. score_all() percentile-normalises insider_raw to 0–100, where 100 means
     the highest net insider buying in the universe.

Role weights (configurable via thresholds.yaml → insider.role_weights):
  BOARD / CEO                  1.00  — highest conviction
  MAJOR_SHAREHOLDER            0.80  — significant but may hedge
  RELATED                      0.50  — disclosures by close relatives
  OTHER / unknown              0.50  — conservative default

Companies with no transactions in the lookback window return None from
score(); score_all() assigns them a neutral percentile (50.0) so they
neither benefit nor suffer from the signal.

Output keys from score():
  net_buy_3m_try   — role-weighted net TRY buying over 3 months
  net_buy_6m_try   — role-weighted net TRY buying over 6 months
  net_buy_3m_pct   — net_buy_3m_try / market_cap (None if cap unavailable)
  net_buy_6m_pct   — net_buy_6m_try / market_cap (None if cap unavailable)
  transaction_count_3m — number of transactions (any type) in 3m
  transaction_count_6m — number of transactions (any type) in 6m
  market_cap_try   — estimated market cap in TRY (None if unavailable)
  insider_raw      — combined normalised signal; higher = more net buying

Output key added by score_all():
  insider_percentile — 0-100 percentile rank across the active universe
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from bist_picker.db.schema import AdjustedMetric, Company, DailyPrice, InsiderTransaction

logger = logging.getLogger("bist_picker.scoring.factors.insider")

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"
)

# Role weight defaults (overridden by config)
_DEFAULT_ROLE_WEIGHTS: dict[str, float] = {
    "BOARD": 1.0,
    "CEO": 1.0,
    "MAJOR_SHAREHOLDER": 0.8,
    "RELATED": 0.5,
    "OTHER": 0.5,
}

# Combination weights (3-month is more timely)
_DEFAULT_WEIGHT_3M = 0.60
_DEFAULT_WEIGHT_6M = 0.40

# Lookback windows in calendar days
_DEFAULT_LOOKBACK_3M = 91   # ~3 months
_DEFAULT_LOOKBACK_6M = 182  # ~6 months


class InsiderScorer:
    """Calculates net insider buying/selling signal from KAP disclosures.

    Reads config from thresholds.yaml (section 'insider'). Falls back to
    hardcoded defaults if the section is absent.

    Args:
        config_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        path = config_path or _DEFAULT_CONFIG_PATH
        cfg = self._load_config(path)

        insider_cfg = cfg.get("insider", {})
        self._lookback_3m = insider_cfg.get("lookback_3m_days", _DEFAULT_LOOKBACK_3M)
        self._lookback_6m = insider_cfg.get("lookback_6m_days", _DEFAULT_LOOKBACK_6M)
        self._weight_3m = insider_cfg.get("weight_3m", _DEFAULT_WEIGHT_3M)
        self._weight_6m = insider_cfg.get("weight_6m", _DEFAULT_WEIGHT_6M)
        self._min_market_cap = insider_cfg.get("min_market_cap_for_pct", 100_000_000)

        raw_role_weights = insider_cfg.get("role_weights", {})
        self._role_weights: dict[str, float] = {
            k.upper(): float(v) for k, v in raw_role_weights.items()
        }
        if not self._role_weights:
            self._role_weights = dict(_DEFAULT_ROLE_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, company_id: int, session: Session) -> Optional[dict]:
        """Calculate insider signal for a single company.

        Args:
            company_id: Database ID of the company.
            session: Active SQLAlchemy session.

        Returns:
            Dict with signal components, or None if no transactions exist
            within the 6-month lookback window.
        """
        company = session.get(Company, company_id)
        if company is None:
            logger.warning("Company id=%d not found", company_id)
            return None

        as_of = date.today()
        cutoff_3m = as_of - timedelta(days=self._lookback_3m)
        cutoff_6m = as_of - timedelta(days=self._lookback_6m)

        # Fetch all transactions in the 6-month window (covers both periods)
        rows = (
            session.query(InsiderTransaction)
            .filter(
                InsiderTransaction.company_id == company_id,
                InsiderTransaction.disclosure_date >= cutoff_6m,
                InsiderTransaction.disclosure_date <= as_of,
            )
            .all()
        )

        if not rows:
            return None  # No signal — caller gets neutral percentile

        net_3m = self._net_buy(rows, cutoff_3m, as_of)
        net_6m = self._net_buy(rows, cutoff_6m, as_of)
        count_3m = sum(1 for r in rows if r.disclosure_date >= cutoff_3m)
        count_6m = len(rows)

        market_cap = self._estimate_market_cap(company_id, session)

        # Compute pct signals (prefer market-cap normalised)
        if market_cap is not None and market_cap >= self._min_market_cap:
            net_3m_pct = net_3m / market_cap
            net_6m_pct = net_6m / market_cap
            insider_raw = (
                net_3m_pct * self._weight_3m + net_6m_pct * self._weight_6m
            )
        else:
            net_3m_pct = None
            net_6m_pct = None
            # Fall back to raw TRY amounts — percentile normalisation will
            # still make cross-sectional comparisons meaningful
            insider_raw = (
                net_3m * self._weight_3m + net_6m * self._weight_6m
            )

        return {
            "net_buy_3m_try": net_3m,
            "net_buy_6m_try": net_6m,
            "net_buy_3m_pct": net_3m_pct,
            "net_buy_6m_pct": net_6m_pct,
            "transaction_count_3m": count_3m,
            "transaction_count_6m": count_6m,
            "market_cap_try": market_cap,
            "insider_raw": insider_raw,
        }

    def score_all(self, session: Session) -> dict[int, dict]:
        """Score all active companies and add percentile rank (0-100).

        Companies with no insider transactions receive a neutral percentile
        of 50.0 so they are neither rewarded nor penalised.

        Returns:
            Dict mapping company_id -> score dict (including insider_percentile).
        """
        company_ids = [
            cid
            for (cid,) in session.query(Company.id)
            .filter(Company.is_active.is_(True))
            .all()
        ]

        raw_scores: dict[int, dict] = {}
        neutral_ids: list[int] = []

        for cid in company_ids:
            result = self.score(cid, session)
            if result is not None:
                raw_scores[cid] = result
            else:
                neutral_ids.append(cid)

        # Percentile-normalise insider_raw across companies that have data
        if raw_scores:
            sorted_ids = sorted(
                raw_scores.keys(),
                key=lambda cid: raw_scores[cid]["insider_raw"],
            )
            n = len(sorted_ids)
            for i, cid in enumerate(sorted_ids):
                raw_scores[cid]["insider_percentile"] = (
                    (i / (n - 1) * 100.0) if n > 1 else 50.0
                )

        # Give companies without transactions a neutral score
        for cid in neutral_ids:
            raw_scores[cid] = {
                "net_buy_3m_try": 0.0,
                "net_buy_6m_try": 0.0,
                "net_buy_3m_pct": None,
                "net_buy_6m_pct": None,
                "transaction_count_3m": 0,
                "transaction_count_6m": 0,
                "market_cap_try": None,
                "insider_raw": 0.0,
                "insider_percentile": 50.0,
            }

        return raw_scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _net_buy(
        self,
        rows: list[InsiderTransaction],
        since: date,
        until: date,
    ) -> float:
        """Compute role-weighted net buying (BUY minus SELL) in TRY.

        Args:
            rows: All InsiderTransaction rows fetched from DB.
            since: Start of the window (inclusive).
            until: End of the window (inclusive).

        Returns:
            Net TRY amount; positive = net buying, negative = net selling.
        """
        net = 0.0
        for row in rows:
            if row.disclosure_date < since or row.disclosure_date > until:
                continue
            if row.transaction_type is None or row.total_value_try is None:
                continue

            weight = self._role_weight(row.person_role)
            weighted_value = row.total_value_try * weight

            if row.transaction_type.upper() == "BUY":
                net += weighted_value
            elif row.transaction_type.upper() == "SELL":
                net -= weighted_value
            # Any other type (e.g. PLEDGE) is ignored

        return net

    def _role_weight(self, role: Optional[str]) -> float:
        """Return the signal weight for an insider's role.

        Args:
            role: Role string from the DB (e.g. 'BOARD', 'CEO').

        Returns:
            Weight between 0.0 and 1.0.
        """
        if role is None:
            return self._role_weights.get("OTHER", 0.5)
        return self._role_weights.get(role.upper(), self._role_weights.get("OTHER", 0.5))

    def _estimate_market_cap(
        self, company_id: int, session: Session
    ) -> Optional[float]:
        """Estimate market cap as latest_close * shares_outstanding.

        Shares outstanding is approximated as adjusted_net_income / eps_adjusted
        (valid for Turkish companies where par value per share = ₺1).

        Args:
            company_id: Company DB ID.
            session: Active SQLAlchemy session.

        Returns:
            Estimated market cap in TRY, or None if data is insufficient.
        """
        price_row = (
            session.query(DailyPrice.close)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.close.isnot(None),
                DailyPrice.close > 0,
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )
        if price_row is None:
            return None

        latest_close: float = price_row[0]

        metric = (
            session.query(AdjustedMetric)
            .filter(
                AdjustedMetric.company_id == company_id,
                AdjustedMetric.eps_adjusted.isnot(None),
                AdjustedMetric.adjusted_net_income.isnot(None),
            )
            .order_by(AdjustedMetric.period_end.desc())
            .first()
        )
        if metric is None:
            return None

        eps = metric.eps_adjusted
        net_income = metric.adjusted_net_income

        if eps == 0 or net_income is None:
            return None

        # Shares = net_income / eps (both in TRY)
        shares = net_income / eps
        if shares <= 0:
            return None

        return latest_close * shares

    @staticmethod
    def _load_config(path: Path) -> dict:
        """Load YAML config file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed config dict, or empty dict if file is missing/unreadable.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("Config file not found: %s — using defaults", path)
            return {}
        except yaml.YAMLError as exc:
            logger.error("Failed to parse config %s: %s", path, exc)
            return {}

    # ------------------------------------------------------------------
    # Enhanced: Cluster and Drawdown Detection (V2.5)
    # ------------------------------------------------------------------

    _CLUSTER_WINDOW_DAYS = 7
    _CLUSTER_MULTIPLIER_2 = 1.5  # 2 distinct insiders
    _CLUSTER_MULTIPLIER_3 = 2.0  # 3+ distinct insiders
    _DRAWDOWN_THRESHOLD = -0.20  # -20% from 52-week high
    _DRAWDOWN_MULTIPLIER = 1.3

    def detect_clusters(
        self,
        company_id: int,
        session: Session,
        lookback_days: int = 30,
    ) -> Optional[dict]:
        """Detect insider buying clusters — multiple insiders buying close together.

        A cluster is when ≥2 different insider persons buy within a 7-day window.
        This is a much stronger signal than a single insider transaction.

        Args:
            company_id: Database ID of the company.
            session: Active SQLAlchemy session.
            lookback_days: How far back to look for clusters.

        Returns:
            Dict with cluster_detected (bool), cluster_size, cluster_multiplier,
            cluster_persons, cluster_window_start, cluster_window_end.
            None if no BUY transactions in the lookback window.
        """
        as_of = date.today()
        cutoff = as_of - timedelta(days=lookback_days)

        # Fetch only BUY transactions
        buys = (
            session.query(InsiderTransaction)
            .filter(
                InsiderTransaction.company_id == company_id,
                InsiderTransaction.disclosure_date >= cutoff,
                InsiderTransaction.disclosure_date <= as_of,
                InsiderTransaction.transaction_type == "BUY",
            )
            .order_by(InsiderTransaction.disclosure_date.asc())
            .all()
        )

        if not buys:
            return None

        # Group by person to find distinct buyers
        buyer_dates: dict[str, date] = {}
        for buy in buys:
            person = buy.person_name or buy.person_role or "unknown"
            if person not in buyer_dates:
                buyer_dates[person] = buy.disclosure_date

        if len(buyer_dates) < 2:
            return {
                "cluster_detected": False,
                "cluster_size": 1,
                "cluster_multiplier": 1.0,
                "cluster_persons": list(buyer_dates.keys()),
            }

        # Check if ≥2 distinct buyers bought within the cluster window
        all_dates = sorted(buyer_dates.values())
        best_cluster_size = 1
        best_window_start = all_dates[0]
        best_window_end = all_dates[0]

        for i, d1 in enumerate(all_dates):
            window_end = d1 + timedelta(days=self._CLUSTER_WINDOW_DAYS)
            cluster_count = sum(
                1 for d2 in all_dates
                if d1 <= d2 <= window_end
            )
            if cluster_count > best_cluster_size:
                best_cluster_size = cluster_count
                best_window_start = d1
                best_window_end = min(
                    window_end,
                    max(d2 for d2 in all_dates if d2 <= window_end),
                )

        if best_cluster_size >= 3:
            multiplier = self._CLUSTER_MULTIPLIER_3
        elif best_cluster_size >= 2:
            multiplier = self._CLUSTER_MULTIPLIER_2
        else:
            multiplier = 1.0

        # Identify which persons are in the best cluster
        cluster_persons = [
            person for person, d in buyer_dates.items()
            if best_window_start <= d <= best_window_start + timedelta(days=self._CLUSTER_WINDOW_DAYS)
        ]

        return {
            "cluster_detected": best_cluster_size >= 2,
            "cluster_size": best_cluster_size,
            "cluster_multiplier": multiplier,
            "cluster_persons": cluster_persons,
            "cluster_window_start": best_window_start,
            "cluster_window_end": best_window_end,
        }

    def detect_drawdown_buy(
        self,
        company_id: int,
        session: Session,
        lookback_days: int = 30,
    ) -> Optional[dict]:
        """Detect if insiders are buying after a significant stock price drop.

        If the stock has dropped >20% from its 52-week high AND insiders
        are buying, this is a strong contrarian signal.

        Returns:
            Dict with drawdown_buy (bool), drawdown_pct, drawdown_multiplier,
            high_52w, current_price.
            None if insufficient data.
        """
        # Get 52-week high
        as_of = date.today()
        year_ago = as_of - timedelta(days=365)

        high_row = (
            session.query(DailyPrice.high)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.date >= year_ago,
                DailyPrice.date <= as_of,
                DailyPrice.high.isnot(None),
            )
            .order_by(DailyPrice.high.desc())
            .first()
        )

        if high_row is None:
            return None

        high_52w = high_row[0]

        # Get current price
        current_row = (
            session.query(DailyPrice.close)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.close.isnot(None),
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )

        if current_row is None or high_52w <= 0:
            return None

        current_price = current_row[0]
        drawdown_pct = (current_price - high_52w) / high_52w

        # Check if there are recent BUY transactions
        cutoff = as_of - timedelta(days=lookback_days)
        has_recent_buys = (
            session.query(InsiderTransaction.id)
            .filter(
                InsiderTransaction.company_id == company_id,
                InsiderTransaction.disclosure_date >= cutoff,
                InsiderTransaction.transaction_type == "BUY",
            )
            .first()
        ) is not None

        is_drawdown_buy = has_recent_buys and drawdown_pct <= self._DRAWDOWN_THRESHOLD

        return {
            "drawdown_buy": is_drawdown_buy,
            "drawdown_pct": round(drawdown_pct * 100, 1),
            "drawdown_multiplier": self._DRAWDOWN_MULTIPLIER if is_drawdown_buy else 1.0,
            "high_52w": high_52w,
            "current_price": current_price,
        }

    def score_enhanced(
        self,
        company_id: int,
        session: Session,
    ) -> Optional[dict]:
        """Enhanced insider score with cluster and drawdown detection.

        Builds on top of the base score() method, adding:
        - Cluster multiplier (1.5x for 2 insiders, 2.0x for 3+)
        - Drawdown multiplier (1.3x if buying into a >20% drop)

        Returns:
            Base score dict with additional enhanced fields, or None.
        """
        base = self.score(company_id, session)
        if base is None:
            return None

        # Cluster detection
        cluster = self.detect_clusters(company_id, session) or {
            "cluster_detected": False,
            "cluster_multiplier": 1.0,
        }

        # Drawdown detection
        drawdown = self.detect_drawdown_buy(company_id, session) or {
            "drawdown_buy": False,
            "drawdown_multiplier": 1.0,
        }

        # Apply multipliers to insider_raw
        enhanced_raw = base["insider_raw"]
        enhanced_raw *= cluster.get("cluster_multiplier", 1.0)
        enhanced_raw *= drawdown.get("drawdown_multiplier", 1.0)

        result = dict(base)
        result["insider_raw_enhanced"] = enhanced_raw
        result["cluster_detected"] = cluster.get("cluster_detected", False)
        result["cluster_size"] = cluster.get("cluster_size", 0)
        result["cluster_multiplier"] = cluster.get("cluster_multiplier", 1.0)
        result["drawdown_buy"] = drawdown.get("drawdown_buy", False)
        result["drawdown_pct"] = drawdown.get("drawdown_pct")
        result["drawdown_multiplier"] = drawdown.get("drawdown_multiplier", 1.0)

        return result
