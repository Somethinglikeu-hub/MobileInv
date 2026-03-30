## Mobile Cloud Sync

This repo powers the zero-PC mobile flow:

1. GitHub Actions restores the latest runtime DB from `MobileInv-state`.
2. The pipeline runs `fetch -> clean -> score -> pick`.
3. A compact mobile snapshot is exported and published to `MobileInv-feed`.
4. The Android app checks the raw GitHub manifest URL, downloads newer snapshots, and swaps its local DB atomically.

### Required GitHub Configuration

Repository variables on `MobileInv`:

- `BIST_STATE_REPOSITORY`
- `BIST_FEED_REPOSITORY`
- `BIST_STATE_BRANCH`
  Optional. Defaults to `main`.
- `BIST_FEED_BRANCH`
  Optional. Defaults to `gh-pages`.

Repository secrets on `MobileInv`:

- `PUBLISH_REPO_TOKEN`
- `TCMB_API_KEY`

### Published Feed URLs

- `https://raw.githubusercontent.com/Somethinglikeu-hub/MobileInv-feed/gh-pages/manifest.json`
- `https://raw.githubusercontent.com/Somethinglikeu-hub/MobileInv-feed/gh-pages/mobile_snapshot.db.gz`

### Manual Local Export

```bash
python -m bist_picker export-mobile-feed \
  --feed-dir mobile-feed-dist \
  --base-download-url "https://raw.githubusercontent.com/Somethinglikeu-hub/MobileInv-feed/gh-pages"
```

### Android Release Build

Use the raw manifest URL in the release build:

```bash
./gradlew assembleRelease -PmobileFeedManifestUrl=https://raw.githubusercontent.com/Somethinglikeu-hub/MobileInv-feed/gh-pages/manifest.json
```
