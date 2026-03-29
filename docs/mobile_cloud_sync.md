## Mobile Cloud Sync

This repo now supports a zero-PC mobile feed workflow:

1. GitHub Actions runs `fetch -> clean -> score -> pick`.
2. The updated runtime database is written to a private state repo as `state/current/bist_picker.db.gz`.
3. A compact Android feed is published to a public GitHub Pages repo as:
   - `manifest.json`
   - `mobile_snapshot.db.gz`

### Required GitHub Configuration

Repository variables on the public code repo:

- `BIST_STATE_REPOSITORY`
  Example: `your-org/bist-picker-mobile-state`
- `BIST_FEED_REPOSITORY`
  Example: `your-org/bist-picker-mobile-feed`
- `BIST_FEED_BASE_URL`
  Example: `https://your-org.github.io/bist-picker-mobile-feed`
- `BIST_STATE_BRANCH`
  Optional. Defaults to `main`.
- `BIST_FEED_BRANCH`
  Optional. Defaults to `gh-pages`.

Repository secrets on the public code repo:

- `PUBLISH_REPO_TOKEN`
  A token with write access to the private state repo and the public feed repo.
- `TCMB_API_KEY`
  Required when TCMB-backed macro fetches are enabled.

### Manual Run

```bash
python -m bist_picker export-mobile-feed --feed-dir mobile-feed-dist --base-download-url "https://your-org.github.io/bist-picker-mobile-feed"
```

### Android Release Build

The Android app reads the feed from `BuildConfig.MOBILE_FEED_MANIFEST_URL`.
Provide the final GitHub Pages manifest URL at build time, for example:

```bash
./gradlew assembleRelease -PmobileFeedManifestUrl=https://your-org.github.io/bist-picker-mobile-feed/manifest.json
```
