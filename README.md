# MobileInv

Cloud pipeline and public mobile feed for the BIST Picker Android app.

## What This Repo Does

- Runs the daily market pipeline on GitHub Actions
- Restores and updates the runtime SQLite database from `MobileInv-state`
- Publishes `manifest.json` and `mobile_snapshot.db.gz` to `MobileInv-feed`
- Feeds the Android app from a raw GitHub URL without requiring a PC

## Repository Variables

- `BIST_STATE_REPOSITORY`
  Example: `Somethinglikeu-hub/MobileInv-state`
- `BIST_FEED_REPOSITORY`
  Example: `Somethinglikeu-hub/MobileInv-feed`
- `BIST_STATE_BRANCH`
  Optional. Defaults to `main`.
- `BIST_FEED_BRANCH`
  Optional. Defaults to `gh-pages`.

`BIST_STATE_REPOSITORY` and `BIST_FEED_REPOSITORY` are optional. When they are not set,
the workflow falls back to `Somethinglikeu-hub/MobileInv-state` and
`Somethinglikeu-hub/MobileInv-feed` under the current repository owner.

## Required Repository Secrets

- `PUBLISH_REPO_TOKEN`
- `TCMB_API_KEY`

## Feed Contract

- Manifest:
  `https://raw.githubusercontent.com/Somethinglikeu-hub/MobileInv-feed/gh-pages/manifest.json`
- Snapshot:
  `https://raw.githubusercontent.com/Somethinglikeu-hub/MobileInv-feed/gh-pages/mobile_snapshot.db.gz`

## Local Smoke Test

```bash
python -m pip install ".[phase1-data,dev]"
pytest tests/test_mobile_snapshot_export.py
python -m bist_picker export-mobile-feed --feed-dir mobile-feed-dist --base-download-url "https://raw.githubusercontent.com/Somethinglikeu-hub/MobileInv-feed/gh-pages"
```
