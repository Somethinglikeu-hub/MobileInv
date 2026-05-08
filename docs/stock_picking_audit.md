# Stock Picking — Audit Findings (2026-05-07)

This file captures everything wrong (or merely surprising) with the BIST picker's
selection logic, so the next "audit the stock picks" session does not have to
re-derive the same conclusions. Read this BEFORE re-running the audit.

The 2026-04-30 audit table in [CLAUDE.md](../CLAUDE.md) lists 8 issues. This
audit re-confirms which of those still bite, splits Buffett issue #2 into
sub-parts (only OE-trend was fixed; ROE/ROA/EPS are still nominal), and adds
new findings #5–#16 that were not in the original audit.

Pipeline at a glance: `cli.py score` → for each company runs Buffett, DCF,
Graham, Piotroski, Lynch, Magic Formula, Momentum, Technical, Dividend, plus
sector models (Banking, Holding, REIT). Raw scores are written to
`scoring_results`, then normalized in place (winsorize ±3σ → sector z-score
→ percentile 0–100), then composed into `composite_alpha` per
`scoring_weights.yaml`. `pick` reads `composite_alpha` for the ALPHA universe,
applies sector cap / bank cap / turnover protection / correlation filter.

Severity legend: **CRITICAL** = directly biases picks; **HIGH** = subtle
distortion; **MEDIUM** = robustness / dead code; **LOW** = cosmetic.

---

## CRITICAL — these biases the picks

### 1. Look-ahead bias: 76-day publication-date heuristic still everywhere

**Where:** `scoring/context.py:60, 85, 95`; duplicated in `scoring/factors/buffett.py:136, 484, 531`, `graham.py:119, 473`, `dcf.py:105`.

For `FinancialStatement` we have a `publication_date` column and the query
correctly filters `publication_date <= scoring_date`. But for `AdjustedMetric`
there is no publication_date column at all — the only guard is
`period_end <= scoring_date - 76 days`. Any company that takes >76 days to
file (common for KAP late-filers) leaks future earnings into the scoring
date's view of the past.

For statement queries, the fallback branch `(publication_date IS NULL AND period_end <= scoring_date - 76d)`
also fires for every legacy row that was fetched before the publication-date
column existed — so historical scoring still uses the heuristic for those.

**Why it matters:** when the picker says "company X scored 92 on Buffett as
of 2024-09-01", it may have been silently using Q2-2024 financials that
weren't actually filed until 2024-11-15. Every `score → pick` run is partly
"as of today" rather than "as of the supposed scoring date." Live picks on
the current date are not affected (today's data really is filed). But any
backtest or as-of replay would show fake skill.

**Fix:** either backfill `publication_date` on `AdjustedMetric` (re-scrape KAP
disclosure timestamps) or stop pretending we have point-in-time integrity and
remove the `scoring_date` parameter from these methods.

### 2. Backtest engine is a placeholder stub

**Where:** `backtest/engine.py` is 6 lines of comment.

There is **zero empirical validation** of the picks. Every weight, threshold,
and method has been calibrated by intuition + unit-test fixtures. Sharpe,
hit rate, drawdown, factor attribution — none of it is measured.

`scoring/optimizer.py` exists but optimizes against a hand-cooked heuristic
(audit issue #6), not realized returns, so it doesn't fill this gap.

**Fix:** implement a walk-forward backtest. The recreation spec
([recreation.md](../../recreation.md) §10 fix #1) has the contract.
Prerequisites: fix issue #1 (publication_date) and issue #4 (survivorship)
first, otherwise the backtest will lie.

### 3. Buffett ROE / ROA / earnings_quality still use nominal values

**Where:** `scoring/factors/buffett.py:215` (`_score_roe_level`), `233`
(`_score_roe_consistency`), `324` (`_score_earnings_quality`).

The 2026-04-30 fix only inflation-deflated `_score_oe_trend`. The other
sub-factors still consume `m.roe_adjusted` (still nominal) and
`m.eps_adjusted` (nominal positive-year counter).

In Turkey (CPI 50–85% YoY in 2022–2024), a company with 30% nominal ROE has
strongly negative real ROE. `_score_roe_level` happily caps the score at 100
because 30% > 25% threshold. This biases the picker toward pure
inflation-pass-through names (metals, refineries, importers) over genuine
quality compounders.

**Fix:** either subtract an inflation proxy from `roe_level` like OE-trend
does, OR add `real_roe`, `real_roa`, `real_eps` columns to `AdjustedMetric`
and switch the scorer to those by default. CPI history is already stored
(see [CLAUDE.md](../CLAUDE.md) changelog entry #4) — it just isn't wired
through.

### 4. Survivorship bias at universe-build time

**Where:** `portfolio/universes.py:301, 329` — both filter
`Company.is_active == True`.

`is_active` is mutated in place when a company delists. There is no
`company_active_periods` table, so the universe at any historical date
includes only the *currently* listed names. A 2024-as-of backtest run today
would silently exclude every company that delisted between 2024 and now.

For *live* picks (today's run on today's universe) this is harmless. For
backtests it's a strong upward bias on results.

**Fix:** add `company_active_periods(company_id, active_from, active_to, reason)`
and join on `as_of_date` instead of `is_active`. Recreation spec §9.2
specifies the schema.

---

## HIGH — distorts pick composition

### 5. `composite_beta` and `composite_delta` are dead code

`scoring_weights.yaml` has only `alpha`, `holding`, `banking`, `reit`, `ipo`,
`regime_weights`. No `beta` or `delta`. The `_validate_weights` method warns
once at load and `compose()` returns None for those portfolios — so
`composite_beta` and `composite_delta` are NULL on every row. The harmonize
step iterates over those columns and silently skips. The BETA / DELTA branches
of `select_all` are commented out (`selector.py:339`) so there's no observable
breakage today, but this is a minefield: re-enabling those portfolios will
silently produce empty picks until someone realizes the weights are missing.

**Fix:** either delete the BETA/DELTA columns and code paths entirely, or
populate the YAML sections. Don't leave half-wired plumbing.

### 6. Bank / holding / REIT scores get triple-haircut vs operating

`composer.py:593-595` applies `data_penalty = 0.50 + 0.50 * coverage_ratio`
(50–100%) to bank/holding/REIT composites, while operating gets
`0.90 + 0.10 * coverage_ratio` (90–100%). Then `_harmonize_composites`
multiplies non-OPERATING scores by `peer_factor = 0.60 + 0.40 * min(1.0, n/50)`
— for ~14 BIST banks that's ×0.71. Then percentile-rank across the universe.

Net: a perfectly-scored, fully-data-complete bank starts at 71% before
ranking against ~470 operating companies (where a 50%-data-complete operating
company is multiplied by 0.95). The bank cap of 2 in the selector is rarely
binding because banks don't reach the top.

**Fix:** if banks should compete on equal terms, drop the peer-factor haircut
or apply it symmetrically. If the user *wants* operating-company bias, the
current setup is fine but should be documented so it's not mistaken for a bug
later.

### 7. Momentum 1-month skip can collapse to 20-day skip

**Where:** `scoring/factors/momentum.py:85` (`skip_date = latest_date - 30d`)
+ `_get_price_near` at `:223-235`.

`_get_price_near` first looks `[target − 10d, target]`, then falls back to
`[target, target + 10d]`. So `end_price` for the skip window can be sourced
from `latest_date − 20d` (i.e. only 20 trading days of skip), defeating
the academic 1-month reversal protection on suspended/illiquid names.

**Fix:** make `_get_price_near` strictly look-back-only when called for the
skip endpoint. Or use a longer skip (e.g. 35 days) so that a +10-day fallback
still leaves at least 25 days of skip.

### 8. Graham value factor is effectively dead in the TRY rate environment

`scoring/factors/graham.py:340` computes
`intrinsic_value = EPS × (8.5 + 2g) × (4.4/Y)` where Y is the bond yield in
percent. With Y = TCMB policy rate (~42% currently), `4.4/Y ≈ 0.10`, so
intrinsic values are compressed by ~10× compared to Graham's original 4.4%
benchmark. The score then maps `ratio = intrinsic/price` linearly from 0.5
(score 0) to 1.5 (score 100). At Y=42, almost no BIST stock clears 0.5.

Net: `graham_growth_value` sub-factor returns ~0 for nearly everyone, so the
Graham composite ends up dominated by the other 3 sub-factors (graham_number,
NCAV, P/E×P/B). This isn't strictly a bug — Graham really did mean "the
market is unattractive when bonds yield 42%" — but it means the 30% ALPHA
weight on `value_graham_dcf` is mostly carried by DCF, not Graham. Worth
knowing.

**Fix (optional):** either accept that Graham value is suppressed by design,
or replace `4.4/Y` with a TRY-localized constant (e.g. fixed AAA spread).

---

## MEDIUM — cleanup, latent bugs, robustness

### 9. Piotroski universe filter has a broken fallback path

`portfolio/universes.py:455-465`:
```python
raw_fscore = getattr(score, "piotroski_fscore_raw", None)
if raw_fscore is not None:
    if raw_fscore < cfg["fscore_min"]:
        reasons.append(...)
elif score.piotroski_fscore is None or score.piotroski_fscore < cfg["fscore_min"]:
    ...
```
The `elif` branch compares `piotroski_fscore` (which after normalization is a
0–100 percentile) against `fscore_min = 5`. Any percentile ≥ 5 passes — that's
basically every company. In practice `piotroski_fscore_raw` is populated
whenever Piotroski runs (`cli.py:285`), so this dead path doesn't fire today.
But if the order ever changes (raw written *after* normalization, or raw lost
in a migration), the universe filter silently breaks.

**Fix:** delete the `elif` and make raw F-score required.

### 10. Incumbents are exited and re-entered at today's price every rebalance

`portfolio/selector.py:411-419` (`select_and_store`) exits all open positions
at the current close, then `_build_pick:811` always uses
`_get_latest_price(today)` for the entry price. Even if turnover protection
"keeps" an incumbent, the persisted row records today's price as the new
entry. The original cost basis is lost. This is intentional per the comment
("buy/sell every month strategy"), but it means real-world P&L tracked
externally (broker app) will not match what the picker shows.

### 11. Falling-knife filter (`min_technical_score: 35`) uses normalized score

The filter is "skip if `technical_score < 35` (normalized 0-100)". After
normalization `technical_score` is roughly the cross-sectional percentile, so
"35" means "bottom 35% by sector." In a bull market every sector's bottom
35% is still trending up; in a bear market the filter doesn't get tighter.
A more robust falling-knife guard would use the raw signal (price < 200MA
AND OBV slope < 0 AND RSI < 30) on the *raw* technical output, before
normalization.

### 12. `max_per_sector = 2` × `target_count = 5` ⇒ exactly 3 distinct sectors

If the top scores cluster in two sectors (very common in BIST during a
specific macro regime — e.g. exporters during weak TRY, or financials during
high-rate periods), the portfolio mechanically forces a 3-sector spread. The
cap binds more often than it appears.

### 13. The fake-quant optimizer is still in the repo

`scoring/optimizer.py` runs Optuna against a hand-coded heuristic objective.
The 2026-04-30 audit (issue #6) and recreation.md §10 fix #6 both say "delete
it." It's still here. As long as nobody calls it, it's harmless — but the
existence implies the weights have been "tuned," which they have not.

### 14. `_get_avg_volumes` source detection is fragile

`portfolio/universes.py:337-340` switches between `close × volume` and raw
`volume` based on `source ILIKE 'YAHOO%'`. Any new fetcher whose source name
doesn't match either pattern goes into the else (assumes TRY turnover). One
typo in a future fetcher silently corrupts liquidity rankings.

---

## LOW — interesting but not actionable today

### 15. `correlation_lookback_days = 120` is calendar days, not trading days

`scoring/selector.py:646`: `cutoff = scoring_date - timedelta(days=120)`.
That's ~84 trading days. Naming suggests trading days. Fine in practice;
just inconsistent.

### 16. The 2026-04-30 changelog #4 ("CPI history table") is wired into
`cleaning/financial_prep.py` but the *consumer* (Buffett ROE/ROA scorers) is
not wired through (see CRITICAL #3). The infrastructure exists; the last
mile is missing.

---

## What the audit *did not* find broken

For future reference, these were checked and look correct as written:

- **DCF** (`scoring/factors/dcf.py`): dynamic discount rate (policy + ERP),
  terminal growth from 24m inflation expectation, returns `None` for
  negative OE, log-linear EPS regression with loss-year penalty, intrinsic
  capped via `min_rate_terminal_spread`. Solid.
- **Piotroski** (`scoring/factors/piotroski.py`): full 9-signal, ratio-based
  comparisons are inflation-neutral.
- **Cash signal** (`portfolio/cash_signal.py`): sticky 4-state with
  asymmetric hysteresis (5d up, 10d down, 20d min hold), kill-switch path
  preserves UI signal while forcing NORMAL.
- **Correlation filter** in selector: pre-computes returns once per pick set,
  uses log returns over 120-day window, replaces lower-scored pick with
  next-best uncorrelated candidate. Correct.
- **ATR-based stop loss** (`_compute_atr_stop`): 20-day ATR × 2.0, clamped
  to [10%, 25%] with fixed 18% fallback when not enough OHLC history.
- **Composite weight redistribution** when factors are missing (`composer.py:343-351`):
  redistributes proportionally over present factors, applies a small
  data-coverage penalty.
- **`select_all` ordering**: cash signal computed first, then `select_all`
  picks per portfolio, then exits prior open positions, then stores new ones.
  Order is correct.

---

## Quick checklist for the next audit

When the user asks "audit the stock picking logic" again:

1. Check whether `AdjustedMetric.publication_date` exists yet (issue #1).
2. Check `backtest/engine.py` — still a stub? (issue #2).
3. Grep `scoring/factors/buffett.py` for `roe_adjusted` — still using nominal? (issue #3).
4. Grep `portfolio/universes.py` for `is_active` — still the only filter? (issue #4).
5. Check `scoring_weights.yaml` for `beta:` and `delta:` sections (issue #5).
6. Check `composer.py` data-penalty asymmetry (issue #6).
7. Confirm `scoring/optimizer.py` is still unused (no callers) — `grep -r "from bist_picker.scoring.optimizer"` (issue #13).

Each of these is a fast yes/no that drives where to spend audit time.

---

## Düşünce süreci — ne düşündüm, ne çıktı (2026-05-07)

Bu bölüm, gelecekteki bir oturumun "neden bu sonuçlara vardık?"ı çabucak
anlaması için bilerek bırakıldı. Audit'i tekrar açtığında benim akıl
yürütmemi sıfırdan üretmek zorunda kalmazsın.

### Önce ne düşündüm

Pipeline'ın iddiası net: **fetch → clean → score → pick → export**. Cron
günde iki kez çalışıyor, APK feed'den çekiyor, 5 hisse seçiyor. "Hisse seçim
mantığında bir şey yanlış mı?" sorusuna mantıklı bir cevap vermek için şu
soruları sırayla sormam gerekiyordu:

1. **Geçmiş veri "as-of" doğru mu görülüyor?** — Yani 2024-09-01 için
   hesaplanan skor, gerçekten o tarihte bilinebilecek bilgilere mi
   dayanıyor? Bilgi sızıntısı olmazsa ileri-bakış yanlılığı (look-ahead
   bias) olur, bu da en sinsi yanlılık türü.
2. **Skorlar bir piyasanın gerçeğini yansıtıyor mu, yoksa enflasyonu mu
   ödüllendiriyor?** — Türkiye'nin %50+ TÜFE'si nominal/reel ayrımını yok
   sayan her metriği zehirler.
3. **Faktörler birbiriyle adil yarışıyor mu?** — Banka modeli operasyonel
   şirketlerle aynı listede çıkarken haksız bir handikap alıyorsa, bank
   cap kuralı zaten boşa çalışıyor demektir.
4. **Seçim mekaniği (sektör cap, korelasyon, turnover) niyete uygun
   davranıyor mu?**
5. **Backtest var mı, ki bunların hiçbiri "iyi niyet" değil "ölçülmüş
   etki" olsun?**

İlk başta "audit zaten 2026-04-30'da yapılmış, yamalar uygulanmış"
varsayıyordum. Ama dosyaları açınca çoğu yamanın kısmen uygulandığını veya
hiç uygulanmadığını gördüm.

### Gerçekte ne çıktı

**1. Look-ahead bias hâlâ açık.** `FinancialStatement` için
`publication_date` kolonu var ama `AdjustedMetric` için yok. Bunun yerine
`scoring/context.py:60`'da `period_end + 76 gün` heuristic'i kullanılıyor.
Geç bilanço veren şirketin verisi geleceğe sızıyor — backtest yapsak fake
skill çıkardı. Audit #3 bunu zaten flag'lemişti, kapatılmamış. **CRITICAL.**

**2. Backtest yok.** `backtest/engine.py` 6 satır yorum. Yani sistemin
hiçbir parametresi (ağırlıklar, eşikler, korelasyon limiti, sektör cap'i)
gerçek getiri verisiyle test edilmemiş. Bu, audit'in en büyük tek
zaafiyeti. **CRITICAL.**

**3. Buffett yarı-yamalı.** 2026-04-30 sadece OE-trend faktörünü enflasyonla
deflate etmiş. Ama aynı dosyada `_score_roe_level` (`buffett.py:215`),
`_score_roe_consistency` (`buffett.py:233`) ve `_score_earnings_quality`
(`buffett.py:324`) hâlâ `m.roe_adjusted` (nominal) ve `m.eps_adjusted`
(nominal) kullanıyor. Yani Türkiye'de %30 nominal ROE'su olan şirket
(reel ROE belki -%10) Buffett'ten 100 alıyor. Picker, gerçek kalite
şirketleri yerine enflasyon-pass-through isimleri ödüllendiriyor.
**CRITICAL.** Üstelik altyapı hazır — `cpi_history` tablosu zaten yazılıyor,
sadece son adım (`real_roe`/`real_eps` türetilip skor'a bağlanması) eksik.

**4. Survivorship bias.** `Company.is_active=True` filtresi geçmiş tarihlere
de uygulanıyor. Live skor için zararsız. Backtest açıldığında doğrudan
yanlılık. Audit #4. **CRITICAL.**

**5. Bankalar matematiksel olarak yenilemez halde.** Harmonize adımında
bankalara `peer_factor ≈ 0.71` çarpılıyor (14 banka × 0.40 + 0.60 normu).
Üstüne operasyonel şirketlerin %90-100 data penalty range'ine karşılık
banka/holding/REIT için %50-100 penalty range uygulanıyor (`composer.py:594`).
İki haircut üst üste binince, "bank cap = 2" kuralı zaten aktive
olmuyor — bankalar üst sıralara çıkamıyor ki cap binding olsun. Bu
muhtemelen kasıtlı **değil** (kullanıcının portföyünde bankaları görme
istediğini bilmiyorum, ama bu mekanizma niyeti ne olursa olsun aşırı
sert). **HIGH.**

**6. `composite_beta` ve `composite_delta` ölü kod.** YAML'da `beta:` ve
`delta:` blokları yok. `_validate_weights` "missing section" diye bir kez
warn'lıyor ve sessiz geçiyor. BETA/DELTA portföyleri zaten selector'da
yorum satırı (`selector.py:339`). Yani bu kolonlar veritabanında hep NULL,
APK snapshot'ta hep NULL. Gelecekte BETA/DELTA aktive edilirse "neden 0
hisse seçiyor?" diye saatlerce arayacaksın — sebep budur. **HIGH.**

**7. Graham'ın growth formülü TRY ortamında ölü.** Klasik formül
`V = EPS × (8.5 + 2g) × (4.4/Y)`'de Y nominal bond yield. Y = TCMB politika
faizi ≈ %42 olunca `4.4/42 ≈ 0.10` — yani intrinsic değerler 10x
sıkışıyor. Hiçbir BIST hissesi ratio = intrinsic/price ≥ 0.5'i geçmiyor.
Sub-faktör fiilen sıfır puan veriyor. Graham composite'i `pe_pb_product` +
`graham_number` + `ncav` üçüne yıkılmış oluyor. **HIGH** ama belki kasıtlı
(Graham gerçekten "tahvil %42'ye veriyorken hisse alma" demek istiyor
olabilir).

**8. Momentum 1-ay skip'i 20 güne düşebiliyor.** `_get_price_near` 10
günlük "öncesi yoksa sonrası" fallback'i var, bu yüzden skip endpoint
fiyatı `latest_date - 20d`'den gelebiliyor. Akademik "1-month skip" amacı
kısmen bozuluyor (illikit/askıya alınmış hisselerde). **HIGH** ama pratikte
çok uçlarda ısırıyor.

**9-16.** Daha küçük bulgular: Piotroski universe-fallback'ı normalize-vs-raw
karşılaştırma hatası, incumbent'ların her rebalance'da bugünün fiyatından
yeniden açılması (gerçek maliyet kaybediliyor), falling-knife filtresinin
göreli (percentile) skorda olması, optimizer'ın hâlâ duruyor olması, vb.

### Audit'i nasıl yaptım — pratik notlar

- **CLAUDE.md → recreation.md → kod.** CLAUDE.md zaten bir audit özetiyle
  başlıyor; o özetin hangi maddeleri "çözüldü" diye işaretlediğine bakıp,
  her birini kodda doğruladım. Çözüldü denilen #5 (Buffett threshold) ve
  #7 (Graham bond yield) gerçekten çözülmüş; #2, #3 yarı çözülmüş; #1, #4,
  #6 hiç açılmamış.
- **`scoring/context.py` ana darboğaz.** Look-ahead davranışı tek bir
  yerde değil — context.py + 4 ayrı scorer'da duplicate edilmiş 76-day
  heuristic var. Tek dosyaya bakıp "tamam" demek yetmedi.
- **YAML-kod karşılaştırması.** `scoring_weights.yaml` sadece `alpha` +
  model bölümlerini içeriyor ama composer `_validate_weights` `beta` /
  `delta` da bekliyor. Validate path `warning` log'luyor ama exception
  fırlatmıyor — yani bu tür "yarı-bağlı" bug'lar test paketinde de
  yakalanmıyor.
- **Recreation.md ile çapraz kontrol.** Recreation belgesi v2 spec
  niteliğinde; v1 audit'te flag edilen hangi hataların v2'de "fix" olarak
  numaralandığını gördüm. Bu, hangilerinin gerçekten çözüldüğünü değil —
  hangilerinin çözülmesi gerektiğini gösteriyor. İkisini ayırt etmek
  önemliydi.

### Sonraki audit için öneri

İlk yapılacak iş: yukarıdaki "Quick checklist" 7 grep'ini koşturup hangi
maddenin sessizce kapandığını çıkar. Sonra bu dosyanın CRITICAL/HIGH
bölümünü tarayıp hâlâ açık olanları rapor et. Sıfırdan kod taraması
gereksiz — bu dosya zaten `file:line` adresleriyle dolu.
