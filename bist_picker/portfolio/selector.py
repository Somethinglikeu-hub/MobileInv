"""Portfolio selection module for the BIST Stock Picker.

Implements the PortfolioSelector that picks 3 stocks per portfolio
(ALPHA, BETA, DELTA) from the eligible universe with the following
constraints and rules:

  Constraints
  -----------
  * Max 2 stocks from the same custom sub-sector
  * Max 1 bank per portfolio

  Turnover penalty
  ----------------
  An incumbent holding is retained when:
    (a) It appears in the top-10 candidates by composite score, AND
    (b) The best available new (non-incumbent) candidate does NOT score
        more than 15% higher than the incumbent.

  Target price
  ------------
  Derived from DCF margin-of-safety (dcf_margin_of_safety_pct) when
  available; otherwise falls back to a score-implied upside estimate.

  Stop loss
  ---------
  entry_price * 0.82 (fixed 18% stop).

Results are written to the portfolio_selections table via
select_and_store().
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sqlalchemy import func
from sqlalchemy.orm import Session

from bist_picker.db.schema import Company, DailyPrice, PortfolioSelection, ScoringResult
from bist_picker.portfolio.universes import UniverseBuilder

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"
)

# ── Constants ─────────────────────────────────────────────────────────────────
_PORTFOLIO_SCORE_COL: dict[str, str] = {
    "ALPHA": "composite_alpha",
    "BETA": "composite_beta",
    "DELTA": "composite_delta",
}
_PICKS_PER_PORTFOLIO: int = 3
_MAX_PER_SECTOR: int = 2
_MAX_BANKS: int = 1
_TOP_N_FOR_TURNOVER: int = 10       # Incumbents protected if still in top 10
_TURNOVER_THRESHOLD: float = 1.15   # New candidate must be 15% better to replace
_STOP_LOSS_FACTOR: float = 0.82     # stop_loss = entry_price * 0.82 (fallback)
_MAX_CORRELATION: float = 0.85       # max pairwise correlation between picks
_CORRELATION_LOOKBACK: int = 120     # days of return history for correlation
_ATR_PERIOD: int = 20                # days for ATR calculation
_ATR_MULTIPLIER: float = 2.0         # stop = entry - (ATR × multiplier)
_MIN_STOP_PCT: float = 0.10          # minimum stop distance (10%)
_MAX_STOP_PCT: float = 0.25          # maximum stop distance (25%)

# Fallback upside range for score-implied target price
_MIN_UPSIDE: float = 0.10           # 10% minimum implied upside
_SCORE_UPSIDE_DIVISOR: float = 400  # 100-score stock -> 25% upside (100/400 = 0.25)
_DEFAULT_MAX_TARGET_MULTIPLE: float = 2.5  # cap: target <= entry * 2.5 (150% upside)


def get_selection_target_count(config_path: Optional[Path] = None) -> int:
    """Return the configured target portfolio size.

    Falls back to the module default when the YAML file is unavailable or
    malformed.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        return _PICKS_PER_PORTFOLIO
    try:
        with path.open("r", encoding="utf-8") as fh:
            full = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return _PICKS_PER_PORTFOLIO
    try:
        return int(full.get("selection", {}).get("target_count", _PICKS_PER_PORTFOLIO))
    except (TypeError, ValueError):
        return _PICKS_PER_PORTFOLIO


class PortfolioSelector:
    """Selects the top stocks per portfolio with constraint enforcement.

    Args:
        scoring_date: Reference date for scoring lookups and price queries.
            Defaults to today if omitted.
        config_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
    """

    def __init__(
        self,
        scoring_date: Optional[date] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.scoring_date: date = scoring_date or date.today()
        self._universe = UniverseBuilder(scoring_date=self.scoring_date)
        self._cfg = self._load_selection_config(config_path or _DEFAULT_CONFIG_PATH)

    @staticmethod
    def _load_selection_config(config_path: Path) -> dict:
        """Load the 'selection' section from thresholds.yaml."""
        if not config_path.exists():
            logger.warning("thresholds.yaml not found at %s; using defaults", config_path)
            return {}
        with config_path.open("r", encoding="utf-8") as fh:
            full = yaml.safe_load(fh) or {}
        return full.get("selection", {})

    def _target_count(self) -> int:
        """Return the configured portfolio size for the current selector."""
        try:
            return int(self._cfg.get("target_count", _PICKS_PER_PORTFOLIO))
        except (TypeError, ValueError):
            return _PICKS_PER_PORTFOLIO

    def _max_per_sector(self) -> int:
        """Return the configured sector cap for the current selector."""
        try:
            return int(self._cfg.get("max_per_sector", _MAX_PER_SECTOR))
        except (TypeError, ValueError):
            return _MAX_PER_SECTOR

    def _max_banks(self) -> int:
        """Return the configured bank cap for the current selector."""
        try:
            return int(self._cfg.get("max_banks_per_portfolio", _MAX_BANKS))
        except (TypeError, ValueError):
            return _MAX_BANKS

    # ── Public API ────────────────────────────────────────────────────────────

    def select(
        self,
        portfolio: str,
        session: Session,
        current_holdings: Optional[list[int]] = None,
    ) -> list[dict]:
        """Select the top picks for *portfolio*.

        Args:
            portfolio: One of 'ALPHA', 'BETA', 'DELTA' (case-insensitive).
            session: Active SQLAlchemy session.
            current_holdings: Optional list of company_ids currently in the
                portfolio (used for turnover-penalty calculation).

        Returns:
            List of up to the configured target count, each with keys:
              company_id, ticker, score, rank, target_price, stop_loss, entry_price
        """
        portfolio = portfolio.upper()
        score_col = _PORTFOLIO_SCORE_COL.get(portfolio)
        if score_col is None:
            raise ValueError(
                f"Unknown portfolio {portfolio!r}. Valid: {sorted(_PORTFOLIO_SCORE_COL)}"
            )

        # Step 1: eligible universe
        universe_ids = set(self._universe.get_universe(portfolio, session))
        if not universe_ids:
            logger.warning("Empty universe for %s — returning no picks", portfolio)
            return []

        # Step 2: fetch scored candidates for universe companies
        candidates = self._fetch_candidates(universe_ids, score_col, session)
        if not candidates:
            logger.warning("No scored candidates found for %s", portfolio)
            return []

        # Step 3: sort descending by composite score (None treated as -1)
        candidates.sort(key=lambda c: c["score"] if c["score"] is not None else -1.0, reverse=True)

        # Build lookup structures
        score_by_id = {c["company_id"]: (c["score"] or 0.0) for c in candidates}
        current_holdings_set = set(current_holdings or [])

        turnover_top_n = int(self._cfg.get("turnover_top_n", _TOP_N_FOR_TURNOVER))
        turnover_threshold = 1.0 + float(
            self._cfg.get("turnover_threshold", _TURNOVER_THRESHOLD - 1.0)
        )
        protected_window_ids = {c["company_id"] for c in candidates[:turnover_top_n]}
        target_count = self._target_count()

        # Step 6: determine which incumbents receive turnover protection
        # An incumbent is protected when it is in the configured top-N and
        # the best new candidate does not exceed its score by the turnover
        # threshold.
        best_new_score = max(
            (c["score"] or 0.0)
            for c in candidates
            if c["company_id"] not in current_holdings_set
        ) if candidates else 0.0

        protected_incumbents: set[int] = set()
        for cid in current_holdings_set:
            if cid not in protected_window_ids or cid not in score_by_id:
                continue
            incumbent_score = score_by_id[cid]
            if best_new_score < incumbent_score * turnover_threshold:
                protected_incumbents.add(cid)

        # Split candidates into incumbents-to-keep and new picks
        incumbent_cands = [c for c in candidates if c["company_id"] in protected_incumbents]
        new_cands = [c for c in candidates if c["company_id"] not in protected_incumbents]

        # Steps 4-7: two-pass selection (incumbents first, then new)
        picks: list[dict] = []
        sector_counts: dict[str, int] = {}
        bank_count: int = 0

        for candidate in incumbent_cands:
            if len(picks) >= target_count:
                break
            if self._violates_constraints(candidate, sector_counts, bank_count):
                continue
            pick = self._build_pick(candidate, len(picks) + 1, session)
            if pick is None:
                continue
            picks.append(pick)
            self._update_counters(candidate, sector_counts)
            if candidate["company_type"] == "BANK":
                bank_count += 1

        for candidate in new_cands:
            if len(picks) >= target_count:
                break
            if self._violates_constraints(candidate, sector_counts, bank_count):
                continue
            pick = self._build_pick(candidate, len(picks) + 1, session)
            if pick is None:
                continue
            picks.append(pick)
            self._update_counters(candidate, sector_counts)
            if candidate["company_type"] == "BANK":
                bank_count += 1

        # Step 8: correlation filter — swap correlated picks with next-best
        max_corr = self._cfg.get("max_correlation", _MAX_CORRELATION)
        picks = self._reduce_correlation(
            picks, new_cands, sector_counts, bank_count, max_corr, session
        )

        logger.info(
            "Selected %d picks for %s: %s",
            len(picks),
            portfolio,
            [p["ticker"] for p in picks],
        )
        return picks

    def select_all(self, session: Session) -> dict:
        """Select picks for all three portfolios.

        Returns:
            Dict with lowercase keys 'alpha', 'beta', 'delta', each mapping
            to a list of pick dicts (see select()).
        """
        results: dict = {}
        # NOTE: BETA and DELTA commented out — only ALPHA portfolio is active.
        # To re-enable, uncomment "BETA" and "DELTA" in the tuple below.
        for portfolio in ("ALPHA",):  # "BETA", "DELTA"):
            current_holdings = self._load_current_holdings(portfolio, session)
            try:
                results[portfolio.lower()] = self.select(portfolio, session, current_holdings)
            except Exception as exc:
                logger.error("Portfolio %s selection failed: %s", portfolio, exc, exc_info=True)
                results[portfolio.lower()] = []
        return results

    def select_and_store(self, session: Session) -> dict:
        """Select for all portfolios and persist results to portfolio_selections.

        Existing rows for the same portfolio + selection_date + company_id are
        updated in place; new rows are inserted.
        
        Monthly Rebalancing Logic:
          Before storing new picks, all existing open positions (exit_date IS NULL)
          for the active portfolios are marked as 'exited' with the current close
          price and scoring_date as exit_date. Then, new picks are stored as 
          open positions. This ensures "Open Positions" only reflects the 
          most recent month's selection.

        Returns:
            Same dict as select_all().
        """
        all_picks = self.select_all(session)

        # Safety net: enforce max_target_multiple cap
        max_multiple = self._cfg.get("max_target_multiple", _DEFAULT_MAX_TARGET_MULTIPLE)
        for picks in all_picks.values():
            for pick in picks:
                entry = pick.get("entry_price")
                target = pick.get("target_price")
                if entry and target and target > entry * max_multiple:
                    pick["target_price"] = round(entry * max_multiple, 2)

        for portfolio_key, picks in all_picks.items():
            p_upper = portfolio_key.upper()
            current_ids = {pick["company_id"] for pick in picks}
            
            # --- 1. Exit Previous Month's Positions ---
            # All previously 'open' positions for this portfolio are closed.
            # If they are re-selected today, they will be 're-opened' as new rows.
            # This reflects the user's "buy/sell every month" strategy.
            open_positions = (
                session.query(PortfolioSelection)
                .filter(
                    PortfolioSelection.portfolio == p_upper,
                    PortfolioSelection.exit_date.is_(None),
                    PortfolioSelection.selection_date < self.scoring_date
                )
                .all()
            )
            
            for pos in open_positions:
                # Get current price to record exit
                exit_price = self._get_latest_price(pos.company_id, session)
                pos.exit_date = self.scoring_date
                pos.exit_price = exit_price or pos.entry_price
                logger.info(
                    "Exiting old position: %s at %s",
                    pos.company_id,
                    f"{pos.exit_price:.2f}" if pos.exit_price is not None else "N/A",
                )

            # --- 1b. Re-running on the same day should replace the snapshot ---
            # Same-day rows are working-state artifacts, not historical rebalances.
            # Keep at most one row per selected company and delete stale leftovers.
            same_day_rows = (
                session.query(PortfolioSelection)
                .filter(
                    PortfolioSelection.portfolio == p_upper,
                    PortfolioSelection.selection_date == self.scoring_date,
                )
                .order_by(PortfolioSelection.id)
                .all()
            )
            same_day_by_company: dict[int, list[PortfolioSelection]] = {}
            for row in same_day_rows:
                same_day_by_company.setdefault(row.company_id, []).append(row)

            for company_id, rows in same_day_by_company.items():
                if company_id not in current_ids:
                    for row in rows:
                        session.delete(row)
                    continue
                for duplicate in rows[1:]:
                    session.delete(duplicate)

            # --- 2. Store New Picks ---
            for pick in picks:
                existing = (
                    session.query(PortfolioSelection)
                    .filter_by(
                        portfolio=p_upper,
                        selection_date=self.scoring_date,
                        company_id=pick["company_id"],
                    )
                    .first()
                )
                if existing:
                    existing.composite_score = pick["score"]
                    existing.target_price = pick["target_price"]
                    existing.stop_loss_price = pick["stop_loss"]
                    existing.entry_price = pick["entry_price"]
                else:
                    session.add(
                        PortfolioSelection(
                            portfolio=p_upper,
                            selection_date=self.scoring_date,
                            company_id=pick["company_id"],
                            entry_price=pick["entry_price"],
                            composite_score=pick["score"],
                            target_price=pick["target_price"],
                            stop_loss_price=pick["stop_loss"],
                        )
                    )

        session.commit()
        logger.info("Stored portfolio selections for %s", self.scoring_date)
        return all_picks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_current_holdings(self, portfolio: str, session: Session) -> list[int]:
        """Return company_ids of open positions from the most recent prior selection.

        Used to activate the turnover penalty — incumbents are retained when
        they are still in the top-10 and no challenger beats them by >15%.
        """
        latest_date = (
            session.query(func.max(PortfolioSelection.selection_date))
            .filter(
                PortfolioSelection.portfolio == portfolio,
                PortfolioSelection.selection_date < self.scoring_date,
                PortfolioSelection.exit_date.is_(None),
            )
            .scalar()
        )
        if not latest_date:
            return []
        rows = (
            session.query(PortfolioSelection.company_id)
            .filter(
                PortfolioSelection.portfolio == portfolio,
                PortfolioSelection.selection_date == latest_date,
                PortfolioSelection.exit_date.is_(None),
            )
            .all()
        )
        return [r[0] for r in rows]

    def _fetch_candidates(
        self, universe_ids: set[int], score_col: str, session: Session
    ) -> list[dict]:
        """Fetch latest ScoringResult + Company data for all companies in *universe_ids*.

        Returns a list of candidate dicts with keys:
          company_id, ticker, score, sector_custom, company_type, dcf_mos
        """
        # Strictly use the configured scoring_date
        rows = (
            session.query(ScoringResult, Company)
            .filter(
                ScoringResult.company_id.in_(universe_ids),
                ScoringResult.scoring_date == self.scoring_date,
            )
            .join(Company, Company.id == ScoringResult.company_id)
            .all()
        )

        candidates = []
        for score_row, company in rows:
            composite = getattr(score_row, score_col, None)
            candidates.append(
                {
                    "company_id": company.id,
                    "ticker": company.ticker,
                    "score": composite,
                    "sector_custom": company.sector_custom,
                    "sector_key": company.sector_custom or company.sector_bist,
                    "company_type": company.company_type,
                    "dcf_mos": score_row.dcf_margin_of_safety_pct,
                    "data_completeness": getattr(score_row, "data_completeness", None),
                    "technical_score": getattr(score_row, "technical_score", None),
                }
            )
        return candidates

    def _violates_constraints(
        self,
        candidate: dict,
        sector_counts: dict[str, int],
        bank_count: int,
    ) -> bool:
        """Return True if selecting *candidate* would breach any portfolio constraint."""
        # 1. Falling Knife (Düşen Bıçak) Filter
        # Reject candidates with extremely poor technicals (e.g., well below 200MA + zero momentum)
        # Even if their fundamental score is perfect, we don't catch falling knives.
        min_tech = self._cfg.get("min_technical_score", 35.0)
        tech_score = candidate.get("technical_score")
        if tech_score is not None and tech_score < min_tech:
            logger.debug(
                "Skip %s: 'Falling Knife' filter triggered (technical score %.1f < %.1f)", 
                candidate["ticker"], tech_score, min_tech
            )
            return True

        # 2. Sector Cap
        sector = self._sector_key(candidate)
        max_per_sector = self._max_per_sector()
        if sector is not None:
            if sector_counts.get(sector, 0) >= max_per_sector:
                logger.debug(
                    "Skip %s: sector %r already has %d picks",
                    candidate["ticker"],
                    sector,
                    max_per_sector,
                )
                return True

        # 3. Bank Cap
        if self._is_bank(candidate) and bank_count >= self._max_banks():
            logger.debug("Skip %s: bank limit reached", candidate["ticker"])
            return True

        return False

    def _reduce_correlation(
        self,
        picks: list[dict],
        remaining_candidates: list[dict],
        sector_counts: dict[str, int],
        bank_count: int,
        max_corr: float,
        session: Session,
    ) -> list[dict]:
        """Replace highly correlated picks with the next-best uncorrelated candidate.

        Uses trailing daily return correlation. If any pair exceeds *max_corr*,
        the lower-scored pick is replaced with the next candidate from
        *remaining_candidates* that doesn't violate constraints or correlation.

        Args:
            picks: Current list of selected picks (with company_id, ticker, score).
            remaining_candidates: Candidates not yet selected, sorted by score desc.
            sector_counts: Current sector counter dict.
            bank_count: Current bank count.
            max_corr: Maximum allowed pairwise correlation (e.g. 0.85).
            session: DB session for price lookups.

        Returns:
            Updated picks list with correlated picks replaced.
        """
        if len(picks) < 2:
            return picks

        lookback = self._cfg.get("correlation_lookback_days", _CORRELATION_LOOKBACK)
        cutoff = self.scoring_date - timedelta(days=lookback)

        def _get_returns(company_id: int) -> Optional[np.ndarray]:
            """Fetch daily close prices and compute log returns."""
            rows = (
                session.query(DailyPrice.date, DailyPrice.close)
                .filter(
                    DailyPrice.company_id == company_id,
                    DailyPrice.date >= cutoff,
                    DailyPrice.date <= self.scoring_date,
                    DailyPrice.close.isnot(None),
                )
                .order_by(DailyPrice.date)
                .all()
            )
            if len(rows) < 30:  # Need at least 30 data points
                return None
            prices = np.array([r.close for r in rows], dtype=float)
            returns = np.diff(np.log(prices))  # log returns
            return returns

        # Pre-compute returns for all picks
        pick_returns: dict[int, Optional[np.ndarray]] = {}
        for pick in picks:
            pick_returns[pick["company_id"]] = _get_returns(pick["company_id"])

        # Check all pairs for high correlation
        replaced = True
        max_iterations = 5  # safety cap
        iteration = 0
        while replaced and iteration < max_iterations:
            replaced = False
            iteration += 1
            for i in range(len(picks)):
                for j in range(i + 1, len(picks)):
                    ret_i = pick_returns.get(picks[i]["company_id"])
                    ret_j = pick_returns.get(picks[j]["company_id"])
                    if ret_i is None or ret_j is None:
                        continue

                    # Align lengths (they should be close but might differ slightly)
                    min_len = min(len(ret_i), len(ret_j))
                    if min_len < 20:
                        continue
                    corr = np.corrcoef(ret_i[-min_len:], ret_j[-min_len:])[0, 1]

                    if np.isnan(corr) or corr <= max_corr:
                        continue

                    # High correlation detected — replace the lower-scored pick
                    if (picks[i].get("score") or 0) >= (picks[j].get("score") or 0):
                        drop_idx = j
                    else:
                        drop_idx = i

                    dropped = picks[drop_idx]
                    logger.info(
                        "Correlation %.2f between %s and %s exceeds %.2f — replacing %s",
                        corr,
                        picks[i]["ticker"],
                        picks[j]["ticker"],
                        max_corr,
                        dropped["ticker"],
                    )

                    # Find the next-best replacement from remaining candidates
                    pick_ids = {p["company_id"] for p in picks}
                    replacement = None
                    temp_sector_counts = dict(sector_counts)
                    self._decrement_counters(dropped, temp_sector_counts)
                    temp_bank_count = bank_count - (1 if self._is_bank(dropped) else 0)
                    for cand in remaining_candidates:
                        if cand["company_id"] in pick_ids:
                            continue
                        if self._violates_constraints(cand, temp_sector_counts, temp_bank_count):
                            continue

                        # Check correlation of replacement against other picks
                        cand_ret = _get_returns(cand["company_id"])
                        ok = True
                        if cand_ret is not None:
                            for k, other_pick in enumerate(picks):
                                if k == drop_idx:
                                    continue
                                other_ret = pick_returns.get(other_pick["company_id"])
                                if other_ret is None:
                                    continue
                                ml = min(len(cand_ret), len(other_ret))
                                if ml < 20:
                                    continue
                                c = np.corrcoef(cand_ret[-ml:], other_ret[-ml:])[0, 1]
                                if not np.isnan(c) and c > max_corr:
                                    ok = False
                                    break
                        if ok:
                            replacement = cand
                            pick_returns[cand["company_id"]] = cand_ret
                            break

                    if replacement is not None:
                        new_pick = self._build_pick(
                            replacement, drop_idx + 1, session
                        )
                        if new_pick is None:
                            continue
                        self._decrement_counters(dropped, sector_counts)
                        if self._is_bank(dropped):
                            bank_count = max(0, bank_count - 1)
                        picks[drop_idx] = new_pick
                        self._update_counters(replacement, sector_counts)
                        if self._is_bank(replacement):
                            bank_count += 1
                        replaced = True
                        logger.info(
                            "Replaced with %s (score %.1f)",
                            replacement["ticker"],
                            replacement.get("score") or 0,
                        )
                        break  # restart pair checking
                    else:
                        logger.warning(
                            "No uncorrelated replacement found for %s — keeping it",
                            dropped["ticker"],
                        )
                if replaced:
                    break

        return picks

    def _update_counters(
        self, candidate: dict, sector_counts: dict[str, int]
    ) -> None:
        """Increment the sector pick counter for *candidate*."""
        sector = self._sector_key(candidate)
        if sector is not None:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    def _decrement_counters(
        self, candidate: dict, sector_counts: dict[str, int]
    ) -> None:
        """Decrement the sector pick counter for *candidate* when replacing it."""
        sector = self._sector_key(candidate)
        if sector is None or sector not in sector_counts:
            return
        next_count = sector_counts[sector] - 1
        if next_count > 0:
            sector_counts[sector] = next_count
        else:
            sector_counts.pop(sector, None)

    @staticmethod
    def _is_bank(candidate: dict) -> bool:
        """Apply the bank cap to both banks and banking-routed financials."""
        return (candidate.get("company_type") or "").upper() in {"BANK", "FINANCIAL"}

    @staticmethod
    def _sector_key(candidate: dict) -> Optional[str]:
        """Use the mapped custom sector when available, else fall back to BIST sector."""
        return candidate.get("sector_key") or candidate.get("sector_custom")

    def _build_pick(
        self, candidate: dict, rank: int, session: Session
    ) -> Optional[dict]:
        """Assemble the final pick dict including target price and stop loss."""
        entry = self._get_latest_price(candidate["company_id"], session)
        if entry is None:
            logger.debug(
                "Skip %s: no price available on or before %s",
                candidate["ticker"],
                self.scoring_date,
            )
            return None
        target = self._compute_target_price(candidate, entry)
        stop = self._compute_atr_stop(candidate["company_id"], entry, session)

        return {
            "company_id": candidate["company_id"],
            "ticker": candidate["ticker"],
            "score": candidate["score"],
            "rank": rank,
            "entry_price": entry,
            "target_price": target,
            "stop_loss": stop,
            "dcf_mos": candidate.get("dcf_mos"),
        }

    def _get_latest_price(
        self, company_id: int, session: Session
    ) -> Optional[float]:
        """Return the most recent adjusted_close (or close) for *company_id*."""
        row = (
            session.query(DailyPrice)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.adjusted_close.isnot(None),
                DailyPrice.date <= self.scoring_date,
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )
        if row:
            return row.adjusted_close

        # Fallback: plain close
        row = (
            session.query(DailyPrice)
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.close.isnot(None),
                DailyPrice.date <= self.scoring_date,
            )
            .order_by(DailyPrice.date.desc())
            .first()
        )
        return row.close if row else None

    def _compute_target_price(
        self, candidate: dict, entry_price: Optional[float]
    ) -> Optional[float]:
        """Derive the target price from DCF MoS or a score-implied upside.

        DCF path:
          MoS% represents (intrinsic - price) / intrinsic.
          => intrinsic = price / (1 - MoS/100)

        Score-implied fallback:
          Upside = max(10%, (score/100) * 30%).  A 100-score stock implies ~30% upside.
          Penalized by 30% if data_completeness < 60%.

        The result is capped at entry_price * max_target_multiple (configurable
        in thresholds.yaml selection.max_target_multiple, default 2.5).
        This prevents unrealistically large targets from very high DCF MoS values.
        """
        if entry_price is None:
            return None

        max_multiple = self._cfg.get("max_target_multiple", _DEFAULT_MAX_TARGET_MULTIPLE)
        max_target = entry_price * max_multiple

        mos = candidate.get("dcf_mos")
        if mos is not None and 0.0 < mos < 100.0:
            dcf_target = entry_price / (1.0 - mos / 100.0)
            return round(min(dcf_target, max_target), 2)

        score = candidate.get("score")
        if score is None:
            score = 50.0

        # Improved heuristic: configurable max upside for a perfect 100-score stock.
        # Default is 30% upside for a 100-score stock.
        max_implied_upside = self._cfg.get("max_implied_upside", 0.30)
        upside = max(_MIN_UPSIDE, (score / 100.0) * max_implied_upside)

        # Penalize low data completeness — if we have little data, reduce
        # confidence in the target by cutting upside by 30%.
        # data_completeness is stored as a percentage on the 0-100 scale.
        completeness = candidate.get("data_completeness")
        if completeness is not None and completeness < 60.0:
            upside *= 0.70

        return round(min(entry_price * (1.0 + upside), max_target), 2)

    def _compute_atr_stop(
        self,
        company_id: int,
        entry_price: Optional[float],
        session: Session,
    ) -> Optional[float]:
        """ATR-based stop-loss, clamped between min and max percentage.

        Uses ``ATR_MULTIPLIER × ATR(ATR_PERIOD)`` as the stop distance below
        the entry price. The percentage distance is clamped to
        ``[MIN_STOP_PCT, MAX_STOP_PCT]`` to avoid extreme values.

        Falls back to the fixed ``_STOP_LOSS_FACTOR`` when insufficient
        price data is available for ATR computation.

        Args:
            company_id: Database ID of the company.
            entry_price: Entry price (latest close).
            session: Active DB session.

        Returns:
            Stop-loss price, or None if entry_price is None.
        """
        if entry_price is None or entry_price <= 0:
            return None

        atr_period = self._cfg.get("atr_period", _ATR_PERIOD)
        atr_mult = self._cfg.get("atr_multiplier", _ATR_MULTIPLIER)
        min_stop = self._cfg.get("min_stop_pct", _MIN_STOP_PCT)
        max_stop = self._cfg.get("max_stop_pct", _MAX_STOP_PCT)

        # Fetch recent OHLC data
        cutoff = self.scoring_date - timedelta(days=atr_period * 3)  # extra margin
        rows = (
            session.query(
                DailyPrice.high, DailyPrice.low, DailyPrice.close
            )
            .filter(
                DailyPrice.company_id == company_id,
                DailyPrice.date >= cutoff,
                DailyPrice.date <= self.scoring_date,
                DailyPrice.high.isnot(None),
                DailyPrice.low.isnot(None),
                DailyPrice.close.isnot(None),
            )
            .order_by(DailyPrice.date)
            .all()
        )

        if len(rows) < atr_period + 1:
            # Not enough data — fallback to fixed stop
            return round(entry_price * _STOP_LOSS_FACTOR, 2)

        # Compute True Range for each day
        true_ranges: list[float] = []
        for i in range(1, len(rows)):
            high = rows[i].high
            low = rows[i].low
            prev_close = rows[i - 1].close
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        # ATR = simple moving average of last `atr_period` true ranges
        atr = sum(true_ranges[-atr_period:]) / atr_period

        # Stop distance as percentage of entry
        stop_distance_pct = (atr * atr_mult) / entry_price
        stop_distance_pct = max(min_stop, min(max_stop, stop_distance_pct))

        stop_price = entry_price * (1.0 - stop_distance_pct)

        logger.debug(
            "company_id=%d ATR=%.2f stop_dist=%.1f%% stop=%.2f",
            company_id, atr, stop_distance_pct * 100, stop_price,
        )

        return round(stop_price, 2)
