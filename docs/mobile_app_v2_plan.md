# Mobile App v2 — Plan & Tasarım

**Tarih:** 2026-05-07 (revize: APK reverse engineering tamamlandı, AI/Gemini katmanı kaldırıldı)
**Hedef:** APK v2'yi sıfırdan yazmak. v1 daha iyi gözükmeyen, sınırlı bilgi veren bir uygulama; v2 daha zengin görsel + daha detaylı veri sunumu yapacak. **AI/LLM yok** — gerek yok, snapshot zaten zengin, açıklamalar template ile çoktan üretiliyor. Finansal seçim mekanizması en kritik parça olduğu için önce o sağlamlaştırılır.

İlgili dosyalar:
- [stock_picking_audit.md](stock_picking_audit.md) — picker'ın hâlâ açık 16 maddesi.
- [mobile_snapshot.py](../bist_picker/mobile_snapshot.py) — snapshot v1 üretici (Python).
- [review.html](../web/review.html) — web reviewer; v2 UI için canlı referans.
- [_bundled_v1_snapshot.db](../data/_bundled_v1_snapshot.db) — APK içinden çıkarılmış 27 MB'lık offline fallback. v2 development için fixture olarak kullan.
- `app-debug.apk` — proje kökünde, reverse-engineering kaynağı.

---

## 1. v1 ne yapıyor — APK reverse engineering bulguları

APK kaynak kodu yok. `androguard` ile çıkarıldı (Java/jadx kurmadan, saf Python).

### 1.1 Kimlik

| Alan | Değer |
|---|---|
| Paket | `com.bistpicker.mobile` |
| App adı | `BIST Picker Mobile` |
| Versiyon | `0.2.0` (versionCode 2) |
| min SDK | 28 (Android 9) |
| target SDK | 36 (Android 16) |
| Main activity | `com.bistpicker.mobile.MainActivity` (tek aktivite) |
| Build | D8 debug, min-api 28 |

### 1.2 İzinler

`INTERNET`, `ACCESS_NETWORK_STATE`, `RECEIVE_BOOT_COMPLETED`, `FOREGROUND_SERVICE`,
`WAKE_LOCK`, `DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`. Yani arkaplan
WorkManager periyodik snapshot çekimi yapıyor — boot sonrası yeniden
uyanıyor.

### 1.3 Mimari (DEX'ten çıkarılmış)

**Manuel DI** (`AppContainer`) — Hilt yok, sade Kotlin.

**Application sınıfı:** `BistPickerApplication`.

**Data katmanı:**
- `BistRepository` (interface) + `LocalBistRepository` (impl)
- Room DB: `SnapshotDatabase` + `SnapshotDao` + entity'ler (`CompanyEntity`,
  `ScoringLatestEntity`, `OpenPositionEntity`, `PortfolioHistoryEntity`,
  `AdjustedMetricsEntity`, `PriceHistoryEntity`, `HomeSummaryEntity`,
  `SnapshotMetadataEntity`)
- API DTO'lar (`data.api.*`): `HomeResponse`, `StockDetailResponse`,
  `ScoringListResponse`, `ScoringOptionsResponse`, `MacroSnapshot`,
  `OpenPosition`, `StockPosition`, `PortfolioHistoryItem`,
  `PortfolioFit`, `RuleCheck`, `SelectionExplanation`,
  `SelectionStatus`, `ExplanationFact`, `ScoringSummary`, `ScoringItem`,
  `StockSearchItem`, `PricePoint`, `CompanyInfo`, `LatestScores`,
  `AdjustedMetrics`, `HomePerformance`
- Sync: `SnapshotSyncManager`, `SnapshotSyncWorker` (WorkManager),
  `OkHttpSnapshotFeedClient`, `SnapshotManifest`, `SnapshotFeedClient`
- State: `DataStoreSnapshotSyncStateStore` (Jetpack DataStore),
  `PersistedSyncState`, `SyncState`, `SyncPhase`
- Snapshot pipeline: `RoomSnapshotStore`, `SnapshotApplier`,
  `SnapshotImportValidator`, `SnapshotInspection`, `SnapshotInfo`
- **Açıklama üreticisi:** `SelectionExplanationBuilder` — verili
  `OpenPosition` + `ScoringResult` üzerinden Türkçe açıklama üretir
  (template-based, LLM yok)
- Filtre state'i: `ScoringFilters`, `ScoringViewMode` (enum: ALL,
  ALPHA_CORE, ALPHA_X, MODEL, RESEARCH)

**UI katmanı:**
- `BistPickerAppKt` — root composable, bottom nav iskeleti
- 3 top-level destination (`TopLevelDestination`)
- 1 detail destination (`DetailDestination` — `createRoute(ticker)`)
- 3 ekran, MVVM ile:
  - **Home** (`ui/home/`): `HomeScreen`, `HomeViewModel`, `HomeUiState`
  - **Scoring** (`ui/scoring/`): `ScoringScreen`, `ScoringViewModel`,
    `ScoringUiState`, `PAGE_SIZE` sabiti — sayfalanmış liste, 5 görüş
    modu, sektör/risk/min-score filtreleri
  - **Detail** (`ui/detail/`): `StockDetailScreen`, `StockDetailViewModel`,
    `StockDetailUiState`
- Bileşenler: `PriceLineChartKt` (özel canvas chart), `OverlayMarker`

### 1.4 Gerçek UI metinleri (DEX literal'lerinden)

**Tab başlıkları (tahmin — string literali):**
- "BIST Picker" (app bar)
- "Hisse Detay" (detail toolbar)
- "Makro Gorunum" — bir ekran/bölüm başlığı

**Faktör etiketleri:** Buffett · Graham · Piotroski · Magic Formula · Lynch
PEG · DCF MoS · Momentum · Quality

**Filtre/sıralama etiketleri:** "Min Skor", "Filter", "Search", "Refresh",
"Risk", "Risk tier", "Free Float", "Owner Earnings", "Free Cash Flow",
"Hedef" (target), "Getiri", "Enflasyon"

**Açıklama metinleri (`SelectionExplanationBuilder` üretir):**
- "Bu hisse guncel snapshot'ta ALPHA Core icin uygun degil; ana engel: ..."
- "Bu hisse guncel snapshot'ta ALPHA Core icin uygun degil; bir veya daha
  fazla temel kural saglanmiyor."
- "Bu hisse mevcut ALPHA portfoyunde hala acik gorunuyor, ancak guncel
  snapshot'ta ALPHA Core uygunlugu zayiflamis olabilir."
- "Hedef fiyat ust sinira kadar kesilmis gorunuyor; yukari potansiyel
  varsayimi agresif olabilir."
- "Risk tier HIGH; modelde yuksek oynaklik ve kirilganlik riski var."
- "Risk tier MEDIUM; secim mantikli olsa da bunu dusuk riskli bir isim gibi
  okumamak gerekir."

**Boş/hata durumları:**
- "Hisse detay yuklenemedi."
- "Offline snapshot bulunamadi. Ana sayfadan guncel snapshot ice aktar."
- "Indirilen snapshot boyutu beklenenden farkli."
- "Desteklenmeyen snapshot sikistirma turu: ..."

**Önemli not:** UI string'leri `res/values/strings.xml`'de DEĞİL, doğrudan
Kotlin literal'lerinde. Çoğu string ASCII (`İ` yerine `I`, `ç` yerine `c`).
v2'de düzgün UTF-8 Türkçe + `strings.xml` lokalizasyonu yapılacak (bkz §4.2).

### 1.5 Veri akışı

1. APK ilk açılışta `assets/mobile_snapshot.db`'den seed eder (27 MB
   gömülü — 2026-04-20 snapshot'ı, 797 şirket, 5 pick).
2. WorkManager `SnapshotSyncWorker` periyodik olarak
   `https://raw.githubusercontent.com/Somethinglikeu-hub/MobileInv-feed/gh-pages/manifest.json`'u
   çeker, sha256 doğrular, gz'i açar, `SnapshotApplier` Room'a aktarır.
3. UI repo üzerinden Flow ile Room'u izler.

Yani APK'nın online bağlantı kurmadığında bile **çalışır halde** kalması
için bütün snapshot logic offline-first.

### 1.6 Bundled snapshot — gerçek pick örneği

`assets/mobile_snapshot.db`'den (2026-04-20):

| Ticker | Composite | DCF MoS | Reason chips | Quality flags |
|---|---|---|---|---|
| PCILT | 99.83 | 19.5% | Buffett 99 · Magic Formula 94 · Graham 86 | — |
| TCKRC | 99.34 | -46.6% | Buffett 97 · Momentum 94 · Technical 79 | DCF_OVERVALUED |
| KIMMR | 99.17 | 45.0% | Piotroski 98 · Momentum 89 · Graham 86 | — |
| ASELS | 98.18 | -100% | Momentum 92 · Piotroski 91 · Buffett 87 | DCF_OVERVALUED |
| ASTOR | 98.01 | -100% | Buffett 99 · Momentum 90 · Technical 82 | DCF_OVERVALUED |

5 pick'in **3'ünde DCF_OVERVALUED**. Picker ALPHA composite skor üzerine
inşa edildi, DCF MoS sadece raporlanıyor. Audit'in HIGH #8 (Graham TRY-fix)
ve CRITICAL #3 (nominal ROE) bulgularıyla tutarlı: Buffett yüksek, Momentum
yüksek, ama içsel-değer açısından çoğu pick aşırı pahalı.

### 1.7 Snapshot şeması — APK'nın gerçekte beklediği

`mobile_snapshot.py`'da yazılı şemadan **fazlası** APK'nın `scoring_latest`
tablosunda bulunuyor. Ekstra kolonlar:

`alpha_x_score`, `alpha_x_rank`, `alpha_x_confidence`, `alpha_x_eligible`,
`alpha_research_bucket`, `alpha_snapshot_streak`, `ranking_score`,
`ranking_source`, `model_score`, `alpha_reason`, `alpha_primary_blocker`

Bu kolonlar bugünkü Python `mobile_snapshot.py`'da değil — yani APK,
şu an yayınlanan snapshot'tan daha gelişmiş bir şema bekliyor. v2
snapshot'ı bu kolonları üretecek (zaten `bist_picker/portfolio/universes.py`
ALPHA bucket diagnostics'i hesaplıyor; sadece `mobile_snapshot.py` bunları
yazmıyor).

---

## 1.8 Sprint 1+2 ne kapandı (2026-05-08 itibarıyla)

| Sprint madde | Durum | Detay |
|---|---|---|
| **§3.1** Buffett enflasyon-aware ROE/ROA | ✓ | `AdjustedMetric.roe_real`/`roa_real` (Fisher); Buffett scorer önce real'e bakıyor |
| **§3.2** publication_date guard | ✓ (yapısal) | `_adjusted_metric_pit_filter` merkezi helper; KAP scraper güncellenince live olur |
| **§3.3** BETA/DELTA ölü kod | ✓ | composer.py + selector.py temizlendi |
| **§3.4** Banka data-penalty hafiflendi | ✓ | Top-10 scoring'de 5 banka/finansal (önce 0) |
| **§3.5** Falling-knife = percentile AND 200MA | ✓ | `ScoringResult.above_200ma` ham sinyal kolonu |
| **§3.7 (yeni)** Damodaran ERP otomatik | ✓ | `macro.yaml` manuel girişi bitti; `MacroRegime.equity_risk_premium_pct` |
| **Snapshot §1-4** alpha_x_* + bucket + reason | ✓ | APK'nın zaten beklediği 11 alan dolu |
| **Snapshot §5** factor_history_quarterly | ✓ | 594 satır × 75 şirket × 8 çeyrek; v2 APK sparkline için hazır |

`SNAPSHOT_SCHEMA_VERSION = 2`. Mevcut v1 APK'lar hâlâ snapshot'tan veri çekebilir (eski kolonlar değişmedi, yeniler eklendi).

**Kalan plan maddeleri (v2 APK build başlayınca):**
- Snapshot §6 `home_metrics_history` (90d portfolio NAV + XU100 equity curve)
- Snapshot §7 `pick_explanations` (template-üretim, Python tarafına taşı)
- Sprint 3: APK v2'yi sıfırdan Compose ile yaz

---

## 2. v2'nin temel yaklaşımı

> "Daha iyi gözüken, daha detaylı bilgi veren, daha detaylı açıklamalar
> yapan" — kullanıcı

| Hedef | Çözüm katmanı |
|---|---|
| Daha iyi gözüken | UI / Compose tema + ekran kompozisyonu |
| Daha detaylı bilgi | Snapshot'ta zaten var olanı UI'da göster + `mobile_snapshot.py`'a `alpha_x_*` ve faktör trendi ekle |
| Daha detaylı açıklamalar | `SelectionExplanationBuilder`'ı genişlet (Kotlin tarafında, Python'a dokunmadan) — daha çok kural, daha bağlamsal cümleler |
| (Örtük) Doğru bilgi | Önce picker'ı düzelt — yanlış pick güzel UI'da daha kötü görünür |

**Sıralama:** picker minimum-fix → snapshot zenginleştirme (Python) →
APK'yı sıfırdan yaz (Kotlin/Compose). AI/LLM bu denkleme dahil değil.

---

## 3. Picker minimum düzeltme seti (v2 öncesi)

`stock_picking_audit.md`'deki 16 maddeden UI v2'nin güvenle dayanabileceği
5'i:

### 3.1 Buffett'i tam enflasyon-aware yap (CRITICAL #3)
- `AdjustedMetric`'e `roe_real`, `roa_real`, `eps_real` kolonları (Alembic).
- `buffett.py:_score_roe_level`, `_score_roe_consistency`,
  `_score_earnings_quality` — yeni kolonları kullan, eski nominal kolon
  fallback olsun.
- Test: 30% nominal + 50% CPI'lı şirket için ROE level skor < 50.

### 3.2 Look-ahead bias minimum kapsama (CRITICAL #1)
- `AdjustedMetric.publication_date` kolonu ekle.
- `scoring/context.py`'de 76-day heuristic'i kaldır; ortak helper'a
  `publication_date <= scoring_date OR (NULL AND period_end <= scoring_date - 120d)`.
- 4 dosyadaki duplicate'ı tek yere taşı.

### 3.3 `composite_beta` / `composite_delta` ölü kodu temizle (HIGH #5)
- YAML'da bölümleri yok. `mobile_snapshot.py`'daki `beta` ve `delta`
  kolonları her zaman NULL. Kolonları **çıkar** veya YAML'a ekle. Tercih:
  çıkar (recreation.md tek-portföy hedefli).

### 3.4 Banka data-penalty asimetrisini hafiflet (HIGH #6)
- `composer.py:594`'teki `0.50 + 0.50 * coverage_ratio` → `0.90 + 0.10 * coverage_ratio`.
- Peer-factor haircut'ı `0.80 + 0.20 * min(1.0, n/30)` yap.
- Test: top 20 ranking'de en az 1 banka.

### 3.5 Falling-knife filtresini ham sinyale çevir (MEDIUM #11)
- `selector.py:_violates_constraints`: `min_technical_score < 35` yerine
  `above_200ma=False AND rsi_14<35 AND obv_slope<0`. Mevcut filtre
  sektörel-göreli; mutlak risk yakalanmıyor.

Diğer 11 audit maddesi v2'ye engel değil — Sprint 4'e ertelendi.

---

## 4. Snapshot v2 — Python tarafı zenginleştirme

`SNAPSHOT_SCHEMA_VERSION = 2`. v1 APK ile geriye uyumlu kal (v1 kolonlar
kalsın, yenileri ek olsun).

### 4.1 `scoring_latest` — eksik kolonları doldur

APK zaten `alpha_x_score`, `alpha_x_rank`, `alpha_x_confidence`,
`alpha_x_eligible`, `alpha_research_bucket`, `alpha_snapshot_streak`,
`ranking_score`, `ranking_source`, `model_score`, `alpha_reason`,
`alpha_primary_blocker` kolonlarını bekliyor. Şu an `mobile_snapshot.py`
bunları yazmıyor.

`bist_picker/portfolio/universes.py` zaten ALPHA bucket diagnostics
hesaplıyor (`alpha_diagnostics()` — `_ALPHA_QUALITY_SHADOW_BUCKET` vb.).
Sadece o dict'ten kolon türetip snapshot'a eklemek gerekiyor.

### 4.2 Yeni tablo: `factor_history_quarterly`

```sql
CREATE TABLE factor_history_quarterly (
  company_id INTEGER NOT NULL,
  scoring_date TEXT NOT NULL,
  buffett REAL, graham REAL, piotroski REAL,
  magic_formula REAL, lynch_peg REAL, dcf_mos REAL,
  momentum REAL, technical REAL, dividend REAL,
  composite_alpha REAL, data_completeness REAL,
  PRIMARY KEY (company_id, scoring_date)
);
```

Sadece çeyrek sonu (4 nokta/yıl × 2 yıl = 8 nokta), pick'lenmiş veya
pick-yakını ~50 şirket için. **APK'da sparkline çizmek için.**

### 4.3 Yeni tablo: `home_metrics_history`

```sql
CREATE TABLE home_metrics_history (
  date TEXT PRIMARY KEY,
  portfolio_nav_pct REAL,    -- 1.05 = +%5
  benchmark_nav_pct REAL,    -- XU100 aynı pencerede
  cash_pct REAL,
  state TEXT
);
```

Son 90 gün. Equity curve (Apple Stocks tarzı) için.

### 4.4 `open_positions` — ek kolonlar

- `data_completeness REAL` — şeffaflık
- `inflation_adjusted_buffett INTEGER` — flag
- `model_used TEXT` — UI banka pick'inde farklı şablon
- `peer_rank_pct REAL` — "Sektörde en iyi %X"
- `confidence_score REAL` — `data_completeness × min(coverage, 1.0)`

### 4.5 `pick_explanations` (template, LLM yok)

```sql
CREATE TABLE pick_explanations (
  company_id INTEGER PRIMARY KEY,
  ticker TEXT NOT NULL,
  buffett_text TEXT,
  graham_text TEXT,
  dcf_text TEXT,
  piotroski_text TEXT,
  momentum_text TEXT,
  technical_text TEXT,
  thesis_summary_tr TEXT,    -- 2-3 cümle özet
  risk_notes_json TEXT,      -- ["...", "..."]
  generated_at TEXT
);
```

**Üretim:** `cli.py pick` komutunun sonunda saf-Python template ile.
Faktör skoru + finansaller + flag'ler input. Cümle örnekleri:
- buffett_text: `f"Son 5 yılda %{roe_real*100:.0f} ortalama reel ROE — {peer_label}."`
- dcf_text: `f"İçsel değer ₺{intrinsic:.2f}, hisse ₺{price:.2f} ({mos_pct:+.0f}%)."`
- thesis_summary_tr: kural tabanlı, faktör skorlarına göre 3-4 şablon

v1'in `SelectionExplanationBuilder`'ı ZATEN bunu yapıyor (Kotlin tarafında).
v2'de bunu Python'a taşıyıp snapshot'a yazarsak APK her açılışta yeniden
hesaplamaz, doğrudan render eder.

**Avantaj:** Determinist, hızlı, ücretsiz, internet gerektirmez. AI gerek
yok — finansal alanda template tabanlı açıklama daha öngörülebilir.

### 4.6 Boyut

v1: ~12 MB. Eklemeler:
- `scoring_latest` ek kolonlar: ~50 KB
- `factor_history_quarterly`: ~32 KB
- `home_metrics_history`: ~5 KB
- `open_positions` ek kolonlar: ihmal
- `pick_explanations`: 5 satır × 2 KB = 10 KB

Toplam ek: ~100 KB. Snapshot ~12.1 MB. APK içinde gömülü asset'in tek
sefer büyümesi (yeni APK release).

---

## 5. APK v2 — sıfırdan Compose proje

Mevcut APK kaynak kodu yok → yeniden yazıyoruz. v1'in yaptığını referans
al, ama mimaride basitleştir + UI'da iyileştir.

### 5.1 Modül yapısı

```
app/
├── build.gradle.kts
├── src/main/
│   ├── AndroidManifest.xml
│   ├── kotlin/com/bistpicker/mobile/
│   │   ├── BistPickerApplication.kt
│   │   ├── MainActivity.kt
│   │   ├── di/
│   │   │   └── AppContainer.kt          # Manuel DI (Hilt overkill)
│   │   ├── ui/
│   │   │   ├── theme/                   # M3 tema, dark+light
│   │   │   ├── BistPickerApp.kt         # Root + bottom nav
│   │   │   ├── TopLevelDestination.kt   # 3 sealed class (Home/Liste/Makro)
│   │   │   ├── screens/
│   │   │   │   ├── home/                # Home: portföy + cash + 3-nokta menü (sync/tema)
│   │   │   │   ├── scoring/             # 797 şirket liste + 5 görüş modu, varsayılan ALPHA_CORE
│   │   │   │   ├── macro/               # Makro Görünüm (yeni — ayrı tab)
│   │   │   │   └── detail/              # Hisse detayı (push, tab değil)
│   │   │   └── components/
│   │   │       ├── PriceLineChart.kt    # 200MA overlay'lı
│   │   │       ├── Sparkline.kt         # YENİ — faktör trendi
│   │   │       ├── EquityCurve.kt       # YENİ — portfolio NAV
│   │   │       ├── FactorCard.kt        # YENİ — akordeon, Buffett vb.
│   │   │       ├── RiskChip.kt
│   │   │       ├── ChipRow.kt
│   │   │       └── EmptyState.kt
│   │   ├── data/
│   │   │   ├── BistRepository.kt        # interface
│   │   │   ├── LocalBistRepository.kt   # Room-backed impl
│   │   │   ├── ScoringFilters.kt
│   │   │   ├── ScoringViewMode.kt       # ALL/ALPHA_CORE/ALPHA_X/MODEL/RESEARCH
│   │   │   ├── local/
│   │   │   │   ├── SnapshotDatabase.kt  # Room
│   │   │   │   ├── SnapshotDao.kt
│   │   │   │   ├── entities/            # Kotlin data classes
│   │   │   │   └── migrations/          # v1 → v2 schema migration
│   │   │   └── sync/
│   │   │       ├── SnapshotSyncWorker.kt
│   │   │       ├── ManifestFetcher.kt   # OkHttp + ETag
│   │   │       └── SnapshotApplier.kt
│   │   └── util/
│   │       ├── Formatters.kt            # ₺, %, sayı, tarih formatları
│   │       └── ExplanationBuilder.kt    # Backup template (snapshot'ta yoksa)
│   └── res/
│       ├── values/strings.xml           # Tüm UI metni TÜRKÇE
│       ├── values-en/strings.xml        # EN ikinci dil (opsiyonel)
│       ├── drawable/                    # Vector iconlar
│       └── mipmap-*/                    # Launcher ikon
└── proguard-rules.pro
```

### 5.2 Bottom navigation — 3 sekme (v1 ile aynı sayı)

| Tab | İkon | Ekran |
|---|---|---|
| Hisseler | `Icons.Filled.Home` | `HomeScreen` — 5 pick + portföy NAV + cash state. Üst çubukta 3-nokta menü: "Manuel yenile", son sync zamanı, tema toggle. |
| Liste | `Icons.AutoMirrored.Filled.List` | `ScoringScreen` — 797 şirket. 5 görüş modu sekmesi yukarıda: **ALPHA_CORE (varsayılan)** · ALPHA_X · MODEL · RESEARCH · ALL. Sektör/risk/min-skor filtreleri. |
| Makro | `Icons.Filled.QueryStats` | `MacroScreen` — TÜFE/faiz/USD-TRY/regime trend |

**Settings tab YOK** — kullanıcı "ne işe yarayacaktı?" diye sordu, gerçekten
gereksiz. Ayar gerekçesi olan 3 küçük şey (manuel sync, son sync zamanı,
tema toggle) Home ekranının üst çubuğundaki 3-nokta menüye katlanır.

**Detay** ayrı route — herhangi bir tab'tan ticker tıklanınca push.

### 5.3 Detay ekranı — v2'nin merkezi

Yukarıdan aşağı:

1. **Üst hero** — Ticker · İsim · Sektör chip · Composite skor (büyük).
   Yanında 4 mini sparkline (Buffett, DCF, Momentum, Technical) —
   `factor_history_quarterly` 8 nokta.

2. **Tez kartı (3 cümle)** — `pick_explanations.thesis_summary_tr`'den.
   AI değil, template — orijin etiketi yok. Sadece "Snapshot 2026-05-07
   itibarıyla" gri rozeti.

3. **Faktör akordeonları** — 8 kart, kapalı başlar, başlık + skor:
   - Buffett 92/100 [reel] — açılınca: ROE/ROA/owner-earnings trend +
     1 cümle açıklama (`buffett_text`).
   - Graham 76/100 — Graham Number, NCAV, P/E×P/B, growth value.
   - DCF 81/100 (MoS %38) — intrinsic value, growth, discount, terminal.
   - Piotroski 8/9 — 9 sinyal sırasıyla ✓/✗.
   - Magic Formula 14/482 — earnings yield + ROIC + sıralama.
   - Lynch PEG 0.42 — PEG ratio + büyüme.
   - Momentum 71/100 — 3m/6m/12m getiri (1-ay skip).
   - Technical 64/100 — 200MA, RSI, hacim oranı, MACD/BB/ADX (varsa).

4. **Risk uyarıları** — `risk_notes_json` chip listesi (kırmızı/sarı).

5. **Finansal özet** — `adjusted_metrics_latest`'tan: ROE/ROA/EPS/reel EPS
   büyümesi/owner earnings/FCF/ilişkili-taraf %.

6. **Fiyat grafiği** — `price_history_730d` 2 yıl, 200MA overlay,
   selection_date dikey çizgi.

7. **Pick parametreleri** — Entry/current/P&L/target (DCF MoS'tan)/stop
   (ATR-based)/days held/portföy ağırlığı (cash-scaled).

### 5.4 Görsel dil

- **Tema:** Dark default + Light. M3 dynamic colors **kapalı** — markalı
  trading-app hissi için sabit palet. Yeşil/kırmızı SADECE P&L ve risk
  flag'lerinde.
- **Tipografi:** Sayılar için tabular figures (SF Mono benzeri). Gövde için
  Inter veya sistem fontu. Metnin okunabilirliği için font scale Material
  default + 5%.
- **Sparkline'lar:** Compose Canvas, 60 satır kod. Vico kütüphanesi gerek
  yok — bu kadar küçük grafikler için overkill.
- **Boş durumlar:** "Pick yok" yerine son cron'un zamanı + sonraki cron'a
  kalan süre. Dakika-üstü güncellik beklentisi yaratma.
- **Türkçe doğru:** v1'de string'ler ASCII'ye dönüştürülmüş ("İ" → "I",
  "ç" → "c"). v2'de tüm UI string'i `res/values/strings.xml`'de UTF-8.
  Bu hem kalite hem ileride localizasyon için.

### 5.5 Erişilebilirlik

- TalkBack: skorlar "Buffett kalite faktörü 92, 100 üzerinden" diye okunsun.
- Dark mode kontrastı: gövde için en az AA, P&L için tek başına renk
  kullanma — ikon + işaret de göster.
- Offline: snapshot zaten offline. Manifest fetch hata verirse "son
  güncelleme: 2026-05-07 09:18" tarihi göster, uygulama tıkanmasın.

---

## 6. Yol haritası — sıralı

### Sprint 1 (1 hafta) — Picker düzeltmeleri
- §3.1–§3.5: 5 picker fix.
- Çıktı: bir cron periyodu çalıştırıp pick listesinin nasıl değiştiğini
  doğrula.

### Sprint 2 (1 hafta) — Snapshot v2 (Python)
- `scoring_latest` eksik kolonları doldur (alpha_x_*, ranking_source vb.)
- `factor_history_quarterly` üret
- `home_metrics_history` üret
- `open_positions` ek kolonlar
- `pick_explanations` template generator (Python; AI yok)
- `SNAPSHOT_SCHEMA_VERSION = 2`
- Test: `tests/test_snapshot_v2_export.py` — hem v1 hem v2 kolonlar var

### Sprint 3 (2-3 hafta) — APK v2
- Yeni Android Studio projesi: `bist-picker-mobile-v2`. Compose + M3 +
  Room + WorkManager + DataStore + OkHttp.
- `assets/mobile_snapshot.db` olarak `data/_bundled_v1_snapshot.db`'yi
  fixture olarak kullan.
- 4-tab iskelet, navigation-compose
- Snapshot reader (Room) + manifest sync (WorkManager)
- HomeScreen, ScoringScreen, MacroScreen, SettingsScreen, DetailScreen
- Sparkline + EquityCurve + PriceLineChart + FactorCard
- Türkçe `strings.xml` (tüm metin)
- Gerçek telefonda test
- Release build + signed APK

### Sprint 4 (sonraki dönem)
- Audit'in kapatılmamış maddeleri: backtest engine, survivorship,
  optimizer silme, Graham TRY-fix, momentum skip katılaştır.
- v2 APK'ya "snapshot manuel yenile" butonu (zaten WorkManager periyodik).
- Push notification (FCM) — v2 APK release sonrası, kullanıcı isteğine göre.

---

## 7. Kararlar (kullanıcı onayıyla, 2026-05-07)

1. **Gemini / AI yok.** Template-based açıklama.
2. **Settings tab yok.** Manuel sync + son sync + tema, Home üst çubuğunda
   3-nokta menü olarak.
3. **Liste varsayılan: ALPHA_CORE.** Diğer 4 mod (ALPHA_X / MODEL /
   RESEARCH / ALL) üst sekme olarak yan yana.
4. **Backtest UI yok** (engine yok).
5. **Yayın: in-place upgrade.** Aynı paket adı, aynı imza, versionCode 3+.
   Test edildikten sonra APK release.
6. **GitHub tarafına henüz dokunulmadı.** Cron / workflow / state-feed
   repoları aynen çalışmaya devam ediyor; bu plan sadece dokümantasyon.
   İlk gerçek değişiklik Sprint 1'de (`bist_picker/scoring/factors/buffett.py`
   ve birkaç kardeş dosyası).

---

## 8. Düşünce süreci — ne düşündüm, ne çıktı

### Önce ne düşündüm

Kullanıcı "sıfırdan APK yapacağız, içine bak" dediğinde üç şeyi aynı anda
çözmem gerektiğini gördüm:
1. Mevcut APK ne yapıyor? (Reverse engineering — Java/jadx olmadan,
   Python ile.)
2. v1'in eksiği nerede? (UI tasarımı + veri kaybı.)
3. v2'yi nasıl yapısal olarak kurmalıyız? (Kotlin/Compose mimarisi.)

İlk düşüncem: APK'yı decompile et, Kotlin kodunu oku, UI'ı yeniden yaz.
Ama Java yok, jadx yok — sıfırdan kurmak yerine `androguard` ile çıkartım
yaptım. Bu, source tree'yi göstermez ama **mimari iskeletini** verir
(class adları, alan tipleri, string literalleri, manifest, asset'ler).

İkinci düşüncem: v1'in *hangi metni* nereye koyduğunu görmek için
string'leri çıkartmak. UI text'i `strings.xml`'de değil, Kotlin literal'i
olarak DEX içinde. Bu ortaya çıktı. Üstelik string'ler ASCII'ye
dönüştürülmüş ("Hisse Detay" — "ı" yok, "İ" yok). v2'de bu düzeltmeli.

Üçüncü düşüncem: v1'in `SelectionExplanationBuilder`'ı zaten bir AI
değil — template ile cümle üretiyor. Yani kullanıcı "AI ile rapor
gereksiz" derken doğru bir şey söylemiş: v1 bile AI kullanmıyor, snapshot
+ kural tabanlı template yetiyor. v2 planındaki Gemini kısmını çıkardım.

Dördüncü düşüncem: APK'nın `assets/`'inde 27 MB'lık bir SQLite var. Bu
inanılmaz — uygulama internet olmadan da çalışır. v2 development için
bu DB'yi `MobileInv/data/_bundled_v1_snapshot.db` olarak kopyaladım,
fixture olarak kullanılabilir.

### Sonra ne çıktı

**Mimari netliği:**
- v1 = 3 tab (Home/Scoring/Detail) + manuel DI + Room + WorkManager + OkHttp
  + Compose + M3
- v1'in yaptığı `SelectionExplanationBuilder` template-based; AI gerek yok
- v1 snapshot şeması bugünkü `mobile_snapshot.py`'dan **fazlasını**
  içeriyor (alpha_x_* kolonları). Yani APK ileri-uyumlu, snapshot generator
  geride.

**v2'nin somut farkları:**
- 3 tab → 4 tab (Macro ayrı tab + Settings eklendi)
- Sparkline + EquityCurve yeni
- 8 faktör akordeonu (v1'de muhtemelen 3 chip'le sınırlı)
- Türkçe doğru kodlama (UTF-8) + lokalize edilmiş `strings.xml`
- Snapshot boyutu ~100 KB artar; aynı offline-first design

**AI çıkarıldı çünkü:**
1. v1 zaten AI kullanmıyor, template ile çalışıyor.
2. Finansal alan için deterministik açıklama daha güvenilir.
3. API key, token bütçesi, fail-mode gibi karmaşalar ortadan kalkıyor.
4. Snapshot 5 pick için 5 satır × ~2KB = 10KB metni Python tarafında
   üretmek 1 dakika sürer; LLM çağrısı dakikada $0.05'e mal olur.
5. Kullanıcı dümdüz "AI ile rapor gereksiz" dedi.

**Reverse engineering gücü:**
APK'yı `androguard` ile çözmek (Java kurmadan), `mobile_snapshot.py`'da
yazılı şemadan **fazlası** olduğunu ortaya çıkardı. Kullanıcı APK'yı en
son ne zaman build ettiğini bilmiyor olabilir; ama APK gömülü snapshot ve
DEX class'ları sayesinde ileride hangi şemayı destekleyeceğini biliyoruz.
Yani v2 snapshot generator'ın doldurması gereken kolonlar **zaten APK
tarafından beklenen** kolonlar — bu eşleştirme şansa kalmadı.

**v2 yapım sıralaması neden bu?**
- Picker fix önce: yanlış pick güzel UI'da daha kötü görünür.
- Snapshot zenginleştirme ikincil: hem v1 hem v2 APK aynı snapshot'tan
  beslenir, eski APK kırılmaz.
- APK v2 son: önceki ikisi tamamlanmadan UI yapmak demek, sonradan
  "şu kolon yokmuş, şu hesaplama yanlışmış" diyerek yeniden yazmak demek.

**Bilinçli olarak yapılmadı:**
- Modülarizasyon (`:core-data`, `:core-ui`, `:feature-home`...) — overkill,
  tek-modül yeterli.
- Hilt — manuel DI hâlâ yeterli, kod basit.
- Multiplatform (KMP) — sadece Android hedefi var.
- Backtest UI — engine yok, dummy data göstermek zaman israfı.
- Live API endpoint — feed URL yeterli; recreation.md'deki Tailscale
  vizyonu farklı bir track.
- AI tezi — yukarıda detaylıca tartıştım.

### Audit ile bağlantı

Sprint 1'in 5 maddesi `stock_picking_audit.md`'deki:
- CRITICAL #3 → Sprint 1 §3.1 (Buffett enflasyon)
- CRITICAL #1 → Sprint 1 §3.2 (look-ahead)
- HIGH #5 → Sprint 1 §3.3 (BETA/DELTA temizle)
- HIGH #6 → Sprint 1 §3.4 (banka asimetrisi)
- MEDIUM #11 → Sprint 1 §3.5 (falling-knife)

Audit'teki diğer maddeler (backtest, survivorship, optimizer, Graham
TRY-fix, momentum skip katılaştır, vb.) Sprint 4'e ertelendi. v2 APK'sını
bunlar için bekletmek gereksiz.