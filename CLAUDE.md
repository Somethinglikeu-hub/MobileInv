# MobileInv — Claude Handoff Notes

This file is auto-loaded by Claude Code in this repo. Read it before doing any work here.

## What this project is

A **BIST (Borsa Istanbul) stock picker** that runs as a serverless cron pipeline and feeds an Android APK twice a day. There is no backend server — the entire system is GitHub Actions + two state/feed git repos + the APK pulling from a raw GitHub URL.

## Three-repo architecture

| Repo | Purpose | Branch |
|---|---|---|
| `MobileInv` (this one) | Python pipeline code, GH Actions workflow, configs, tests | `main` |
| `MobileInv-state` | Private. Holds runtime SQLite DB (`state/current/bist_picker.db.gz`) between runs so the next cron pickup has prior data. | `main` |
| `MobileInv-feed` | Public. Holds `manifest.json` + `mobile_snapshot.db.gz` consumed by the Android APK. | `gh-pages` |

The user clones all three side-by-side under `~/Desktop/Ai Projects/MobileInvesting/`. Only `MobileInv` has code worth changing.

## How the pipeline runs

`.github/workflows/mobile-feed.yml`. Cron: `17 6 * * 1-5` and `17 16 * * 1-5` (UTC). That's 09:17 and 19:17 Istanbul — pre-open and post-close, weekdays only.

Each run:

1. Restore SQLite DB from `MobileInv-state`.
2. Run `python -m bist_picker fetch` → `clean` → `score` → `pick` → `export-mobile-feed`.
3. Push updated DB back to `MobileInv-state`.
4. Push fresh `manifest.json` + `mobile_snapshot.db.gz` to `MobileInv-feed/gh-pages`.
5. Open/close GitHub issues for stale macro values or cash-state transitions.

Secrets needed: `PUBLISH_REPO_TOKEN`, `TCMB_API_KEY`. Vars: `BIST_STATE_REPOSITORY`, `BIST_FEED_REPOSITORY` (optional, default to `<owner>/MobileInv-state` and `<owner>/MobileInv-feed`).

**The APK depends on the snapshot schema. Don't silently change it.** Add columns; never rename/drop without a migration plan.

## Code layout (`bist_picker/`)

```
bist_picker/
├── cli.py                  # `bist <command>` entry — fetch/clean/score/pick/export-mobile-feed
├── data/                   # fetcher.py (IsYatirim primary, Yahoo fallback, KAP, TCMB EVDS)
├── cleaning/               # financial_prep.py (IAS-29 stripping), inflation.py (real EPS, CPI deflation)
├── classification/         # company_type, sector_mapper, risk_classifier
├── scoring/
│   ├── factors/            # buffett, graham, dcf, piotroski, momentum, technical, dividend, lynch, magic_formula, macro_nowcast_score, event_score
│   ├── models/             # banking, holding, reit (sector-specific scorers)
│   ├── normalizer.py       # winsorize ±3σ → sector z-score → 0–100 percentile
│   ├── composer.py         # weighted blend; re-normalizes weights when factors are missing
│   ├── optimizer.py        # Optuna — currently optimizes a HEURISTIC, not portfolio Sharpe (broken)
│   └── red_flags.py        # MVP: piotroski<4, completeness<60%, MoS<0, technical<40
├── portfolio/
│   ├── selector.py         # top-N with sector caps, correlation filter, turnover protection
│   ├── universes.py        # ALPHA / BETA / DELTA + research universes
│   ├── cash_signal.py      # 4-state regime overlay with asymmetric hysteresis
│   ├── exit_rules.py       # stop-loss / target / insider-selling thesis breaker
│   └── regime_classifier.py
├── backtest/engine.py      # *** DISABLED PLACEHOLDER *** — see audit
├── config/                 # YAML: scoring_weights, thresholds, macro, sectors, settings, exclusions, cash_signal, llm_config, subsidiaries
├── db/                     # SQLAlchemy models
├── output/                 # excel, google_sheets, terminal, performance
├── api/, dashboard/        # optional FastAPI + Streamlit
├── mobile_snapshot.py      # builds the SQLite snapshot consumed by the APK
├── mobile_feed.py          # writes manifest.json + gzips snapshot
├── macro_check.py          # detects stale Damodaran ERP — opens GH issue
└── read_service.py         # large query/aggregation helper
```

Tests live in both `bist_picker/tests/` and top-level `tests/`. `pyproject.toml` lists both.

## Audits — read first when asked "is something wrong with the picks?"

The picking logic has been audited twice. Each audit produced a focused
findings file. Read those before re-deriving anything.

- **[docs/stock_picking_audit.md](docs/stock_picking_audit.md)** — 2026-05-07
  audit. Severity-ranked findings, "what's NOT broken" list, and a fast
  checklist for the next audit. Re-confirms which 2026-04-30 items still
  bite, splits Buffett issue #2 into sub-parts, and adds 8 new findings the
  original audit missed (composite_beta/delta dead, asymmetric data penalty
  for banks, momentum 1-month skip can collapse, Graham value effectively
  dead in TRY rate environment, etc.). Ends with a "Düşünce süreci" section
  capturing how the audit was reasoned through.
- **[docs/mobile_app_v2_plan.md](docs/mobile_app_v2_plan.md)** — 2026-05-07
  plan for APK v2. User wants better-looking, more detailed, more
  explanation-rich mobile app, **without AI/LLM** (deterministic templates
  only). APK source code does NOT exist on disk — only the compiled
  `app-debug.apk`. The plan documents the full reverse-engineering of v1
  (via `androguard`, no Java/jadx needed): package `com.bistpicker.mobile`
  v0.2.0, Compose + M3 + Room + WorkManager + OkHttp + manual DI, 3 tabs
  (Home/Scoring/Detail), `SelectionExplanationBuilder` already produces
  Turkish explanation strings via templates. `data/_bundled_v1_snapshot.db`
  was extracted from the APK's `assets/` for use as a v2 dev fixture.
  Plan: (a) 5-item minimum picker fix set derived from the audit, (b)
  snapshot schema v2 — fill in already-expected `alpha_x_*` columns + add
  `factor_history_quarterly` (sparklines), `home_metrics_history` (equity
  curve), `pick_explanations` (template-generated, no LLM), (c) brand-new
  Compose + M3 APK with 3 tabs (Home/Liste/Makro) + Detail screen,
  properly UTF-8 Turkish in `strings.xml`. Liste tab defaults to
  ALPHA_CORE view mode, with ALPHA_X/MODEL/RESEARCH/ALL as side tabs.
  No Settings tab — manual sync + theme toggle live in Home's overflow
  menu. In-place APK upgrade (same package, versionCode bump). Reverse-
  engineering also revealed v1 already expects more columns than
  `mobile_snapshot.py` currently writes — that mismatch is the first
  thing to fix in Sprint 2. Nothing has been touched yet on the GitHub
  side (workflows, state/feed repos) — this plan is documentation-only;
  first real code change lands in Sprint 1.
- The 2026-04-30 audit table below remains correct as a high-level map.

If the user says "find what's wrong with stock picking" / "audit the picker"
/ similar, **start by reading `docs/stock_picking_audit.md`**, then verify
each item is still present (not silently fixed since 2026-05-07). The file's
"Quick checklist for the next audit" at the bottom gives the fast yes/no
greps. Do that before grepping the whole codebase from scratch.

## 2026-04-30 audit — known issues to fix

Strong parts (don't refactor without reason):
- **DCF** (`scoring/factors/dcf.py`) — dynamic discount = TCMB policy + ERP, terminal growth from 24m inflation expectation, returns `None` for negative-OE.
- **Piotroski** — full 9-signal, divides by available signals (handles missing data fairly).
- **Momentum** — academic 1-month skip, weights 12m=0.4, 6m=0.3, 3m=0.3.
- **Cash signal** — sticky 4-state with asymmetric hysteresis (5d in, 10d out, 20d cooldown).
- **IAS-29 stripping** in `cleaning/financial_prep.py` (banks correctly excluded from monetary-loss strip).
- **Selector** — correlation filter (0.70 / 120d), sector cap (max 2), turnover protection (15%).

Outstanding issues, prioritized by impact on stock picks:

| # | Issue | Where | Fix |
|---|---|---|---|
| 1 | **Backtest disabled** — no empirical validation of any pick | `backtest/engine.py` (placeholder) | Re-implement walk-forward backtest with point-in-time integrity + delisted tickers in universe |
| 2 | **Nominal P/E, ROE, ROA in hyperinflation** distort Buffett/Graham scoring | `scoring/factors/buffett.py`, `graham.py` use raw fields; `cleaning/inflation.py` has real-EPS path that isn't wired in | Add inflation-adjusted ratio variants and wire them through scoring |
| 3 | **Look-ahead bias** — heuristic 76-day lag, no real `publication_date` enforcement | `scoring/context.py:60,85` | Capture KAP filing timestamps, require `publication_date <= scoring_date` strictly |
| 4 | **Survivorship bias** — `is_active=False` applied retroactively | `portfolio/universes.py:301-330` | Time-versioned `company_active_periods` table; backtests must use as-of-date membership |
| 5 | **Buffett thresholds too lax** — D/E ≤ 2.0, ROE floor 5%, gross margin 25% | `config/thresholds.yaml:3-12` | Tighten: ROE floor 12%, full-score 25%; D/E ideal 0.5, max 1.5; gross margin 30%. Single YAML edit. |
| 6 | **Optimizer is fake-quant** — Optuna against a hand-cooked objective, not portfolio Sharpe | `scoring/optimizer.py:113-148` | Replace objective with walk-forward portfolio Sharpe (depends on #1) |
| 7 | **Hardcoded TRY bond yield 30%** in Graham | `config/thresholds.yaml:15` `graham.try_bond_yield` | Pull from `MacroRegime` (TCMB 10y benchmark) at runtime, fall back to YAML |
| 8 | **Bank equity comparability** — GAAP equity used where regulatory capital should be | banks scoring path | Only matters if alpha pool uses raw P/B for banks; verify the banking model already isolates this |

## Working preferences observed

- User iterates on YAML thresholds rather than rewriting code — prefer config edits when both work.
- User runs the full pipeline through GitHub Actions; local smoke tests are `pytest tests/test_mobile_snapshot_export.py`.
- Tests are extensive (`tests/test_banking_model.py`, `test_dcf_scorer.py`, `test_holding_model.py`, `test_portfolio_selector.py`, `test_technical_scorer.py`, etc.) — keep them green.
- Don't break the snapshot schema. The APK is in production for the user.

## Local dev

```bash
python -m pip install ".[phase1-data,dev]"
pytest                                              # full suite
pytest tests/test_mobile_snapshot_export.py         # smoke
python -m bist_picker fetch                         # one stage
python -m bist_picker export-mobile-feed --feed-dir mobile-feed-dist --base-download-url "https://raw.githubusercontent.com/Somethinglikeu-hub/MobileInv-feed/gh-pages"
```

## Changelog (Claude-assisted)

- **2026-04-30** — Audit performed (this file added). Shipped:
  1. Tightened Buffett thresholds in `config/thresholds.yaml`:
     `roe_min_score 0.05 → 0.08`, `debt_equity_max 2.0 → 1.5`. Existing test
     fixtures (perfect/bad stock) still pass under the new bounds; mediocre
     operators (ROE 5–8%, D/E 1.5–2.0) now score zero on those sub-factors.
  2. Made TRY bond yield dynamic in `scoring/factors/graham.py`. Added
     `_resolve_bond_yield(session, scoring_date)` pulling
     `MacroRegime.policy_rate_pct` (cached per scoring run); YAML
     `graham.try_bond_yield` is now a fallback only.
  3. Fixed Buffett OE-trend inflation bias in `scoring/factors/buffett.py`.
     `_score_oe_trend` now subtracts an inflation proxy from the relative
     slope (prefers `MacroRegime.inflation_expectation_24m_pct`, falls back
     to `cpi_yoy_pct`, falls back to no-deflation when macro is empty —
     so unit tests without seeded macro keep their nominal scoring).
     Companies whose nominal "growth" is just CPI pass-through no longer
     score 100/100 on OE trend; they score around 50.
  4. Fixed CPI history storage. Added a `cpi_history` table
     (`db/schema.py:CpiHistory`) holding TCMB TP.FG.J0 monthly index
     levels. `data/fetcher.py:_upsert_cpi_history` now persists an 8-year
     CPI window on every macro fetch (cached by tcmb client, so no
     extra API cost on warm runs). `cleaning/financial_prep.py
     :_get_cpi_series` reads index levels from this table first; the
     legacy `MacroRegime.cpi_yoy_pct` path is kept as a clearly-warned
     fallback only. Result: `calculate_real_growth` finally gets the
     CPI index levels it was designed for, so `real_eps_growth_pct` in
     `adjusted_metrics` becomes meaningful (currently surfaced in the
     mobile snapshot + dashboard, and now also exposed by the
     `web/review.html` reviewer). Tests in `tests/test_cpi_history.py`
     pin the new behavior and document the legacy fallback's known
     wrongness.
  5. Added `web/review.html` — single-file static viewer that fetches
     the public manifest + snapshot, opens the SQLite in-browser via
     sql.js + pako, and renders the 5 picks with full scoring detail.
     Open the file directly in a browser, or host it on
     MobileInv-feed/gh-pages alongside the snapshot.
  - Outstanding (in priority order): re-enable backtest with point-in-time
    integrity & delisted tickers; enforce real `publication_date <=
    scoring_date` (drop the 76-day heuristic); replace optimizer's
    hand-cooked objective with portfolio Sharpe; consider also
    inflation-adjusting Buffett ROE/ROA the way OE-trend now is.

- **2026-05-07** — Second audit run (`docs/stock_picking_audit.md`).
  Re-confirmed audit issues #1, #2, #3 (partial), #4, #6 still present.
  No new code shipped — this audit is documentation-only, capturing 8
  additional findings (`composite_beta/delta` dead, asymmetric
  bank-data-penalty triple-haircut, momentum skip can collapse to 20d,
  Graham factor effectively dead in TRY rates, Piotroski universe-filter
  fallback compares percentile to raw threshold, incumbents re-entered at
  today's price every rebalance, falling-knife filter uses normalized not
  raw signal, source-detection in `_get_avg_volumes` is fragile). See the
  audit file for severity, file:line pointers, and proposed fixes.

- **2026-05-08 — Sprint 1: 6 picker fixes shipped + live-validated.**
  Lifted from the audit's CRITICAL/HIGH/MEDIUM list. 12 new tests, 543/546
  green (pre-existing 3 DCF failures untouched).
  1. **§3.3** Removed BETA/DELTA dead code paths from composer.py +
     selector.py (DB columns retained as legacy NULL for back-compat).
  2. **§3.4** Softened bank/holding/REIT data-penalty in `composer.py`
     (`0.50 + 0.50*coverage` was double-haircutting banks; now they
     compete fairly. Live test: 5 of top-10 are banks/financials —
     previously 0).
  3. **§3.5** Falling-knife filter combines percentile + raw 200MA: a
     stock now needs `technical_score < 35 AND above_200ma=False` to be
     dropped. Persists `above_200ma` boolean on `ScoringResult` (new
     column). 4 new tests in `tests/test_portfolio_selector.py`.
  4. **§3.1** Buffett `_score_roe_level` / `_score_roe_consistency` prefer
     CPI-deflated ROE over nominal. New `AdjustedMetric.roe_real` /
     `roa_real` columns populated by `cleaning/financial_prep.py` via
     Fisher conversion. Falls back to nominal when CPI history empty.
  5. **§3.2** Centralized point-in-time guard
     `_adjusted_metric_pit_filter` in `scoring/context.py` — used by
     ScoringContext, Buffett, Graham, DCF. Prefers
     `AdjustedMetric.publication_date` (new column) when populated, falls
     back to 76-day heuristic for legacy rows. KAP scraper still doesn't
     capture filing dates, so the new path activates once that's wired
     (Sprint 4 task) — for now we're 100% on legacy heuristic but the
     scaffolding is in place. 5 new tests pin both paths.
  6. **§3.7 (new) — Damodaran ERP automated.** Manual `macro.yaml` entry
     replaced by `bist_picker/data/sources/damodaran.py` which scrapes
     Damodaran's `ctryprem.html` once per cron and writes
     `MacroRegime.equity_risk_premium_pct` (auto: `0.0889` on 2026-05-08).
     `DCFScorer._get_erp(session, scoring_date)` now reads DB-first, YAML
     fallback only. 4 tests including a live network test (skippable via
     `BIST_SKIP_NETWORK_TESTS=1`).

  **Live test (2026-05-08):** All 6 fixes ran end-to-end against the
  production DB. Picks list rotated (TCKRC #2 → #5; rest same set:
  PCILT, KIMMR, ASELS, LILAK, TCKRC). Banks now rank visibly in scoring
  but stay out of ALPHA picks because `_ALPHA_CORE_TYPES = {"OPERATING"}`
  excludes them by design — that decision is unchanged. Buffett-real-ROE
  is structurally shipped but a no-op in this test env because TCMB key
  is missing → `cpi_history` empty. Production cron has the key so
  Fisher activates there.

- **2026-05-08 — Sprint 2 §1-5: snapshot v2 wired.**
  `SNAPSHOT_SCHEMA_VERSION = 2`. The APK was already expecting these
  columns; now they're populated.
  - **§1-4:** wired `alpha_x_score`, `alpha_x_rank`, `alpha_x_eligible`,
    `alpha_x_confidence`, `alpha_core_eligible`, `alpha_research_bucket`,
    `alpha_primary_blocker`, `alpha_reason`, `alpha_snapshot_streak`,
    `ranking_score`, `ranking_source`, `model_score` through to
    `mobile_snapshot.scoring_latest`. The data was already being computed
    in `read_service.get_scoring_results` — just had to bump the schema
    version and fix one pre-existing NaN-handling bug in
    `_alpha_x_risk_score`.
  - **§5:** new `factor_history_quarterly` table (594 rows, 75 companies
    × 8 quarter-ends). Stores Buffett/Graham/Piotroski/DCF/Momentum/
    Technical/composite_alpha at each quarter-end so v2 APK can render
    sparklines on the detail screen. Universe = open positions ∪ top-75
    alpha_x_eligible. ~60 KB additional snapshot footprint.

  **What's still in the v2 plan:** factor_history_quarterly's consumer
  (the v2 APK detail screen) doesn't exist yet — that needs the Compose
  rewrite (Sprint 3). `home_metrics_history` and `pick_explanations`
  tables (§6, §7 in the plan) are deferred until v2 APK build starts;
  they're additive and don't change v1 behavior.
