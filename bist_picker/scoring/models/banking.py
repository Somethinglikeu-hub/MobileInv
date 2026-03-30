"""Banking model composite scorer for BIST Stock Picker.

Applies ONLY to company_type = 'BANK' or 'INSURANCE'.  Operating companies
use the factor pipeline (buffett, graham, piotroski, …) instead.

Key design rule from CLAUDE.md:
  "Banks are different: Do NOT strip monetary gain/loss for banks — their
   net monetary position IS their business."
  Therefore we use REPORTED net income (not adjusted_net_income) for ROE.

Seven sub-factors, each cross-sectionally percentile-ranked (0-100) within
the bank universe.  Missing data causes that sub-factor's weight to be
redistributed among available sub-factors.

Factor          Weight   Direction   Notes
──────────────  ──────   ─────────   ─────────────────────────────────────
pb_vs_sector    0.20     lower=good  P/B ratio vs bank peers; value signal
nim             0.15     higher      Net Interest Margin = NII / avg assets
npl_ratio       0.15     lower=good  Non-performing loans / total loans
car             0.15     higher      Capital Adequacy Ratio (regulatory)
cost_income     0.15     lower=good  Operating costs / operating income
loan_growth     0.10     higher      Nominal YoY growth in total loans
roe             0.10     higher      Reported net income / avg book equity

Data extraction strategy:
  1. Standard item_codes (codes used by both XI-29 and UFRS):
       3L  net income    2N  equity    1BL  total assets
  2. UFRS-specific: label-based search via Turkish + English keywords on
       desc_tr / desc_eng fields in financial_statements.data_json.
  3. Metrics that can't be extracted (NPL, CAR often in footnotes only)
       return None and their weight is redistributed.

Cross-sectional normalization in score_all():
  - Collect raw ratios from every bank
  - Percentile-rank within the bank universe
  - "lower = better" metrics are inverted (rank 0 → score 100)
  - banking_composite = weighted average of available percentile scores

Weights are read from config/scoring_weights.yaml → 'banking' section.
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from bist_picker.cleaning.inflation import _find_item_by_codes, _find_item_by_labels
from bist_picker.db.schema import AdjustedMetric, Company, DailyPrice, FinancialStatement

logger = logging.getLogger("bist_picker.scoring.models.banking")

_DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "scoring_weights.yaml"
)
_DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"
)

# ── Company types this model handles ─────────────────────────────────────────
_BANK_TYPES = {"BANK", "INSURANCE", "FINANCIAL"}

# ── Standard codes (same across XI-29 and UFRS) ──────────────────────────────
_CODES_NET_INCOME = ["3L"]
_CODES_EQUITY = ["2N"]
_CODES_TOTAL_ASSETS = ["1BL"]

# ── UFRS-specific labels for bank financial statement items ───────────────────
# Net Interest Income
_NII_LABELS_TR = ["net faiz geliri", "net faiz"]
_NII_LABELS_EN = ["net interest income", "net interest"]

# Total loans (balance sheet asset)
_LOANS_LABELS_TR = ["krediler", "toplam krediler", "müşteri kredileri", "kredi ve alacaklar"]
_LOANS_LABELS_EN = ["loans", "total loans", "customer loans", "loans and advances"]

# Total deposits (balance sheet liability)
_DEPOSITS_LABELS_TR = ["mevduat", "toplam mevduat", "müşteri mevduatı"]
_DEPOSITS_LABELS_EN = ["deposits", "total deposits", "customer deposits"]

# Operating expenses (non-interest costs)
_OPEX_LABELS_TR = ["faaliyet giderleri", "genel yönetim giderleri", "personel giderleri ve genel yönetim"]
_OPEX_LABELS_EN = ["operating expense", "operating cost", "personnel and general admin", "non-interest expense"]

# Operating / banking income (denominator for cost/income ratio)
_OPINC_LABELS_TR = ["net faaliyet geliri", "toplam faaliyet geliri", "bankacılık hizmetleri gelirleri"]
_OPINC_LABELS_EN = ["net operating income", "total operating income", "net banking income", "total net revenue"]

# Non-performing loans (often only in supplemental footnotes)
_NPL_LABELS_TR = ["takipteki krediler", "tahsili gecikmiş", "donuk alacak"]
_NPL_LABELS_EN = ["non-performing", "impaired loan", "npl", "stage 3"]

# Capital Adequacy Ratio (regulatory, often only in supplemental tables)
_CAR_LABELS_TR = ["sermaye yeterlilik rasyosu", "sermaye yeterliliği oranı", "yasal sermaye"]
_CAR_LABELS_EN = ["capital adequacy ratio", "car", "tier 1"]

# Factor ordering matches scoring_weights.yaml banking section
_FACTOR_DIRECTION: dict[str, str] = {
    "pb_vs_sector": "lower",   # cheap bank vs peers → better
    "nim":          "higher",
    "npl_ratio":    "lower",   # fewer bad loans → better
    "car":          "higher",
    "cost_income":  "lower",   # lean operation → better
    "loan_growth":  "higher",
    "roe":          "higher",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class BankingScorer:
    """Composite scorer for BANK and INSURANCE companies.

    Usage:
        scorer = BankingScorer()
        # Single company — returns raw ratios (not yet normalized)
        raw = scorer.score(company_id, session)

        # Full universe — returns percentile-ranked sub-scores + composite
        all_scores = scorer.score_all(session)

    Args:
        weights_path: Path to scoring_weights.yaml.
        thresholds_path: Path to thresholds.yaml.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        thresholds_path: Optional[Path] = None,
    ) -> None:
        self._weights = self._load_yaml(
            weights_path or _DEFAULT_WEIGHTS_PATH
        ).get("banking", {})
        self._thresholds = self._load_yaml(
            thresholds_path or _DEFAULT_THRESHOLDS_PATH
        ).get("banking", {})

        if not self._weights:
            logger.warning(
                "No 'banking' section in scoring_weights.yaml — using equal weights"
            )
            self._weights = {k: 1 / len(_FACTOR_DIRECTION) for k in _FACTOR_DIRECTION}

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        company_id: int,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[dict]:
        """Extract raw banking ratios for a single company.

        Returns raw (un-normalised) ratios that score_all() will cross-
        sectionally rank.  No 0-100 normalisation happens here.

        Args:
            company_id: Database ID of the company.
            session: Active SQLAlchemy session.

        Returns:
            Dict with raw metric values, or None if the company is not
            a bank or has insufficient data.
        """
        company = session.get(Company, company_id)
        if company is None:
            logger.warning("Company id=%d not found", company_id)
            return None

        ctype = (company.company_type or "").upper()
        if ctype not in _BANK_TYPES:
            logger.debug(
                "Skipping %s: company_type=%s (not a bank)",
                company.ticker, ctype,
            )
            return None

        # Load most-recent annual INCOME and BALANCE statements
        income_rows = self._load_statements(
            company_id, "INCOME", session, limit=2, scoring_date=scoring_date
        )
        balance_rows = self._load_statements(
            company_id, "BALANCE", session, limit=2, scoring_date=scoring_date
        )

        if not income_rows and not balance_rows:
            logger.debug("No financial statement data for bank %s", company.ticker)
            return None

        # Current-period data
        cur_income = income_rows[0] if income_rows else []
        cur_balance = balance_rows[0] if balance_rows else []

        # Prior-period data (for YoY calculations)
        prev_balance = balance_rows[1] if len(balance_rows) > 1 else None

        # Raw computations
        net_income = self._get_net_income(cur_income)
        equity_cur = self._find_equity(cur_balance)
        equity_prev = self._find_equity(prev_balance) if prev_balance else None
        total_assets_cur = self._find_total_assets(cur_balance)
        total_assets_prev = self._find_total_assets(
            prev_balance) if prev_balance else None

        loans_cur = self._find_loans(cur_balance)
        loans_prev = self._find_loans(prev_balance) if prev_balance else None
        deposits_cur = self._find_deposits(cur_balance)

        nii = self._find_nii(cur_income)
        opex = self._find_opex(cur_income)
        op_income = self._find_op_income(cur_income)
        npl = self._find_npl(cur_balance)
        car_raw = self._find_car(cur_income, cur_balance)

        # ── Compute ratios ────────────────────────────────────────────────
        avg_equity = _avg(equity_cur, equity_prev)
        avg_assets = _avg(total_assets_cur, total_assets_prev)

        roe = _safe_ratio(net_income, avg_equity)
        nim = _safe_ratio(nii, avg_assets)
        cost_income = _safe_ratio(opex, op_income)
        npl_ratio = _safe_ratio(npl, loans_cur)
        loan_growth = _safe_growth(loans_cur, loans_prev)
        loan_to_deposit = _safe_ratio(loans_cur, deposits_cur)

        # P/B ratio: market price × shares / book equity
        pb = self._calc_pb(company_id, equity_cur, session, scoring_date=scoring_date)

        return {
            "pb":              pb,
            "nim":             nim,
            "npl_ratio":       npl_ratio,
            "car":             car_raw,
            "cost_income":     cost_income,
            "loan_growth":     loan_growth,
            "roe":             roe,
            # Extra diagnostics (not directly scored)
            "loan_to_deposit": loan_to_deposit,
            "equity":          equity_cur,
            "total_assets":    total_assets_cur,
            "net_income":      net_income,
        }

    def score_all(
        self,
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> dict[int, dict]:
        """Score all BANK/INSURANCE/FINANCIAL companies with hybrid scoring.

        Hybrid approach:
          1. Extract raw ratios for every active bank/financial.
          2. Compute absolute scores (linear interpolation between floor/ceiling).
          3. Compute peer scores (percentile-rank within the universe).
          4. Blend: 60% absolute + 40% peer (configurable via thresholds.yaml).
          5. banking_composite = weighted average of blended sub-scores.

        Returns:
            Dict mapping company_id -> score dict containing raw ratios,
            blended sub-scores (*_score, 0-100), banking_composite,
            and data_completeness.
        """
        bank_ids = [
            cid
            for (cid,) in session.query(Company.id)
            .filter(
                Company.is_active.is_(True),
                Company.company_type.in_(list(_BANK_TYPES)),
            )
            .all()
        ]

        raw: dict[int, dict] = {}
        for cid in bank_ids:
            if scoring_date is None:
                result = self.score(cid, session)
            else:
                result = self.score(cid, session, scoring_date=scoring_date)
            if result is not None:
                raw[cid] = result

        if not raw:
            return {}

        # Map score key → raw dict key
        factor_to_raw = {
            "pb_vs_sector": "pb",
            "nim":          "nim",
            "npl_ratio":    "npl_ratio",
            "car":          "car",
            "cost_income":  "cost_income",
            "loan_growth":  "loan_growth",
            "roe":          "roe",
        }

        # Read blend weights from thresholds.yaml
        abs_weight = self._thresholds.get("absolute_weight", 0.60)
        peer_weight = self._thresholds.get("peer_weight", 0.40)
        abs_thresholds = self._thresholds.get("absolute_thresholds", {})

        # Compute peer scores (percentile-rank across universe)
        peer_scores: dict[str, dict[int, Optional[float]]] = {}
        for factor, raw_key in factor_to_raw.items():
            direction = _FACTOR_DIRECTION[factor]
            values = {cid: raw[cid].get(raw_key) for cid in raw}
            peer_scores[factor] = _cross_percentile(values, invert=(direction == "lower"))

        # Compute absolute scores and blend
        for factor, raw_key in factor_to_raw.items():
            factor_thresholds = abs_thresholds.get(factor)

            for cid in raw:
                peer_val = peer_scores[factor].get(cid)
                raw_val = raw[cid].get(raw_key)

                # Absolute score from thresholds
                abs_val = None
                if factor_thresholds and raw_val is not None:
                    abs_val = _absolute_score(
                        raw_val,
                        factor_thresholds["floor"],
                        factor_thresholds["ceiling"],
                        factor_thresholds.get("direction", "higher"),
                    )

                # Blend: both available → weighted; one available → use it
                if abs_val is not None and peer_val is not None:
                    blended = abs_weight * abs_val + peer_weight * peer_val
                elif abs_val is not None:
                    blended = abs_val
                elif peer_val is not None:
                    blended = peer_val
                else:
                    blended = None

                raw[cid][f"{factor}_score"] = blended

        # Compute weighted banking_composite and data_completeness per company
        total_factors = len(factor_to_raw)
        for cid in raw:
            raw[cid]["banking_composite"] = self._compute_composite(raw[cid])
            # Data completeness: how many of the 7 factors have a score
            n_available = sum(
                1 for f in factor_to_raw
                if raw[cid].get(f"{f}_score") is not None
            )
            raw[cid]["data_completeness"] = round(
                (n_available / total_factors) * 100.0, 1
            )

        return raw

    # ── Composite calculation ─────────────────────────────────────────────────

    def _compute_composite(self, scores: dict) -> Optional[float]:
        """Weighted average of available factor sub-scores (0-100 each).

        Missing sub-scores have their weight redistributed among available
        factors.  Returns None if no sub-scores are available.

        Args:
            scores: Dict containing <factor>_score keys.

        Returns:
            Weighted composite 0-100, or None.
        """
        parts: list[tuple[float, float]] = []
        for factor, weight in self._weights.items():
            key = f"{factor}_score"
            val = scores.get(key)
            if val is not None:
                parts.append((val, weight))

        if not parts:
            return None

        total_w = sum(w for _, w in parts)
        return round(sum(v * w for v, w in parts) / total_w, 2)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_statements(
        self,
        company_id: int,
        statement_type: str,
        session: Session,
        limit: int = 2,
        scoring_date: Optional[date] = None,
    ) -> list[list[dict]]:
        """Load the most recent `limit` annual statements of a given type.

        Returns a list of data_json arrays (already decoded), most recent first.
        Each inner list is a list of financial statement line items.
        """
        from sqlalchemy import or_

        cutoff_date = scoring_date or date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        rows = (
            session.query(FinancialStatement)
            .filter(
                FinancialStatement.company_id == company_id,
                FinancialStatement.statement_type == statement_type,
                FinancialStatement.period_type == "ANNUAL",
                FinancialStatement.data_json.isnot(None),
                or_(
                    FinancialStatement.publication_date <= cutoff_date,
                    (
                        FinancialStatement.publication_date.is_(None)
                        & (FinancialStatement.period_end <= lagged_cutoff)
                    ),
                ),
            )
            .order_by(FinancialStatement.period_end.desc())
            .limit(limit)
            .all()
        )

        result = []
        for row in rows:
            try:
                data = json.loads(row.data_json)
                if isinstance(data, list):
                    result.append(data)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "Failed to decode data_json for company_id=%d: %s", company_id, exc
                )
        return result

    # ── Raw metric extractors ─────────────────────────────────────────────────

    @staticmethod
    def _get_net_income(data: list[dict]) -> Optional[float]:
        return _find_item_by_codes(data, _CODES_NET_INCOME)

    @staticmethod
    def _find_equity(data: Optional[list[dict]]) -> Optional[float]:
        if not data:
            return None
        return _find_item_by_codes(data, _CODES_EQUITY)

    @staticmethod
    def _find_total_assets(data: Optional[list[dict]]) -> Optional[float]:
        if not data:
            return None
        return _find_item_by_codes(data, _CODES_TOTAL_ASSETS)

    @staticmethod
    def _find_nii(data: list[dict]) -> Optional[float]:
        """Net Interest Income from income statement (label search)."""
        val = _find_item_by_labels(data, _NII_LABELS_TR, lang="tr")
        if val is None:
            val = _find_item_by_labels(data, _NII_LABELS_EN, lang="en")
        return val

    @staticmethod
    def _find_opex(data: list[dict]) -> Optional[float]:
        """Operating expenses from income statement (label search)."""
        val = _find_item_by_labels(data, _OPEX_LABELS_TR, lang="tr")
        if val is None:
            val = _find_item_by_labels(data, _OPEX_LABELS_EN, lang="en")
        return abs(val) if val is not None else None  # ensure positive

    @staticmethod
    def _find_op_income(data: list[dict]) -> Optional[float]:
        """Net operating (banking) income from income statement (label search)."""
        val = _find_item_by_labels(data, _OPINC_LABELS_TR, lang="tr")
        if val is None:
            val = _find_item_by_labels(data, _OPINC_LABELS_EN, lang="en")
        return val

    @staticmethod
    def _find_loans(data: Optional[list[dict]]) -> Optional[float]:
        """Total customer loans from balance sheet (label search)."""
        if not data:
            return None
        val = _find_item_by_labels(data, _LOANS_LABELS_TR, lang="tr")
        if val is None:
            val = _find_item_by_labels(data, _LOANS_LABELS_EN, lang="en")
        return val

    @staticmethod
    def _find_deposits(data: list[dict]) -> Optional[float]:
        """Total customer deposits from balance sheet (label search)."""
        val = _find_item_by_labels(data, _DEPOSITS_LABELS_TR, lang="tr")
        if val is None:
            val = _find_item_by_labels(data, _DEPOSITS_LABELS_EN, lang="en")
        return val

    @staticmethod
    def _find_npl(data: list[dict]) -> Optional[float]:
        """Non-performing loans from balance sheet (label search).

        Often only in footnotes — will frequently return None.
        """
        val = _find_item_by_labels(data, _NPL_LABELS_TR, lang="tr")
        if val is None:
            val = _find_item_by_labels(data, _NPL_LABELS_EN, lang="en")
        return abs(val) if val is not None else None

    @staticmethod
    def _find_car(
        income_data: list[dict], balance_data: list[dict]
    ) -> Optional[float]:
        """Capital Adequacy Ratio (regulatory).

        Searches both income and balance statement data since CAR is
        sometimes disclosed in supplemental tables attached to either.
        Often returns None (disclosed only in footnote PDFs).
        """
        for data in (balance_data, income_data):
            val = _find_item_by_labels(data, _CAR_LABELS_TR, lang="tr")
            if val is not None:
                return val / 100.0 if val > 1.0 else val  # convert % to ratio
            val = _find_item_by_labels(data, _CAR_LABELS_EN, lang="en")
            if val is not None:
                return val / 100.0 if val > 1.0 else val
        return None

    def _calc_pb(
        self,
        company_id: int,
        equity: Optional[float],
        session: Session,
        scoring_date: Optional[date] = None,
    ) -> Optional[float]:
        """P/B = (latest close × shares_outstanding) / book_equity.

        Shares are estimated as adjusted_net_income / eps_adjusted.
        Returns None if any component is unavailable.
        """
        if equity is None or equity <= 0:
            return None

        cutoff_date = scoring_date or date.today()
        lagged_cutoff = cutoff_date - timedelta(days=76)
        price_row = (
            session.query(DailyPrice.close)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.date <= cutoff_date,
                DailyPrice.close.isnot(None),
                DailyPrice.close > 0,
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )
        if price_row is None:
            return None
        close = price_row[0]

        metric = (
            session.query(AdjustedMetric)
            .filter(
                AdjustedMetric.company_id == company_id,
                AdjustedMetric.period_end <= lagged_cutoff,
                AdjustedMetric.eps_adjusted.isnot(None),
                AdjustedMetric.adjusted_net_income.isnot(None),
            )
            .order_by(AdjustedMetric.period_end.desc())
            .first()
        )
        # Fallback: for banks, use reported net income when adjusted might strip too much
        if metric is None or metric.eps_adjusted == 0:
            return None

        shares = metric.adjusted_net_income / metric.eps_adjusted
        if shares <= 0:
            return None

        market_cap = close * shares
        return market_cap / equity

    # ── Config loading ────────────────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("Config file not found: %s — using defaults", path)
            return {}
        except yaml.YAMLError as exc:
            logger.error("Failed to parse %s: %s", path, exc)
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide numerator by denominator, returning None on any issue."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _safe_growth(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    """YoY growth rate = (current - prior) / abs(prior).

    Returns None if either value is missing, or prior is zero.
    Negative prior values are handled: growth = (cur - prior) / abs(prior).
    """
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / abs(prior)


def _avg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Average of two values; falls back to single value if one is missing."""
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a  # may be None if both are None


def _absolute_score(
    value: float, floor: float, ceiling: float, direction: str = "higher"
) -> float:
    """Linearly interpolate a raw value between floor (0) and ceiling (100).

    For 'higher' direction: floor is the worst (score 0), ceiling is best (100).
    For 'lower' direction: floor is the worst (score 0), ceiling is best (100).
    In both cases, floor maps to 0 and ceiling maps to 100.

    The result is clamped to [0, 100].

    Args:
        value: Raw metric value.
        floor: Raw value that maps to score 0.
        ceiling: Raw value that maps to score 100.
        direction: 'higher' (bigger=better) or 'lower' (smaller=better).

    Returns:
        Score in [0.0, 100.0].
    """
    if direction == "lower":
        # For lower-is-better: floor (high raw) → 0, ceiling (low raw) → 100
        # floor > ceiling in config, so we invert
        if floor == ceiling:
            return 50.0
        score = (floor - value) / (floor - ceiling) * 100.0
    else:
        # For higher-is-better: floor (low raw) → 0, ceiling (high raw) → 100
        if floor == ceiling:
            return 50.0
        score = (value - floor) / (ceiling - floor) * 100.0

    return max(0.0, min(100.0, score))


def _cross_percentile(
    values: dict[int, Optional[float]], invert: bool = False
) -> dict[int, Optional[float]]:
    """Cross-sectionally percentile-rank a dict of company_id -> raw value.

    Companies with None values receive None (not ranked).
    All other companies receive a 0-100 percentile score.
    If invert=True, the lowest raw value gets 100 and the highest gets 0.

    Args:
        values: Mapping of company_id -> raw ratio (may contain None).
        invert: If True, lower raw value is better (e.g. NPL, cost/income).

    Returns:
        Mapping of company_id -> percentile score (0-100) or None.
    """
    valid = {cid: v for cid, v in values.items() if v is not None}
    result: dict[int, Optional[float]] = {cid: None for cid in values}

    if not valid:
        return result

    sorted_ids = sorted(valid.keys(), key=lambda c: valid[c])
    if invert:
        sorted_ids = list(reversed(sorted_ids))

    n = len(sorted_ids)
    for i, cid in enumerate(sorted_ids):
        result[cid] = (i / (n - 1) * 100.0) if n > 1 else 50.0

    return result
