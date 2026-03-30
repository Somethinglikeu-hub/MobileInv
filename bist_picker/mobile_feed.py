"""Package the offline mobile snapshot into a cloud-friendly feed."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from urllib.parse import urljoin

from bist_picker.mobile_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    export_mobile_snapshot,
    validate_mobile_snapshot,
)

MOBILE_FEED_VERSION = 1
DEFAULT_FEED_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "mobile_feed"
DEFAULT_FEED_MANIFEST_FILENAME = "manifest.json"
DEFAULT_FEED_SNAPSHOT_FILENAME = "mobile_snapshot.db.gz"


@dataclass(frozen=True)
class MobileFeedExportResult:
    """Paths and metadata produced by a mobile feed export."""

    manifest_path: Path
    snapshot_path: Path
    manifest: dict[str, object]


def export_mobile_feed(
    feed_dir: str | Path = DEFAULT_FEED_DIRECTORY,
    *,
    base_download_url: str | None = None,
    manifest_filename: str = DEFAULT_FEED_MANIFEST_FILENAME,
    snapshot_filename: str = DEFAULT_FEED_SNAPSHOT_FILENAME,
) -> MobileFeedExportResult:
    """Create a published mobile feed directory with manifest + gzip snapshot."""
    feed_dir = Path(feed_dir).resolve()
    feed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = feed_dir / manifest_filename
    snapshot_path = feed_dir / snapshot_filename

    temp_fd, temp_name = mkstemp(prefix="mobile_feed_", suffix=".db")
    os.close(temp_fd)
    temp_snapshot_path = Path(temp_name)
    try:
        export_mobile_snapshot(temp_snapshot_path)
        metadata = validate_mobile_snapshot(temp_snapshot_path)

        with temp_snapshot_path.open("rb") as source, gzip.open(snapshot_path, "wb") as target:
            shutil.copyfileobj(source, target)
    finally:
        if temp_snapshot_path.exists():
            try:
                temp_snapshot_path.unlink()
            except PermissionError:
                pass

    snapshot_sha256 = _sha256_for_file(snapshot_path)
    manifest = {
        "feedVersion": MOBILE_FEED_VERSION,
        "snapshotSchemaVersion": SNAPSHOT_SCHEMA_VERSION,
        "snapshotDate": metadata.get("snapshot_date"),
        "exportedAt": metadata.get("exported_at"),
        "compression": "gzip",
        "sha256": snapshot_sha256,
        "sizeBytes": snapshot_path.stat().st_size,
        "downloadUrl": _build_download_url(
            base_download_url=base_download_url,
            snapshot_filename=snapshot_filename,
        ),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return MobileFeedExportResult(
        manifest_path=manifest_path,
        snapshot_path=snapshot_path,
        manifest=manifest,
    )


def _build_download_url(base_download_url: str | None, snapshot_filename: str) -> str:
    """Build a feed download URL; relative paths remain relative when no base URL is given."""
    if not base_download_url:
        return snapshot_filename
    base = base_download_url.rstrip("/") + "/"
    return urljoin(base, snapshot_filename)


def _sha256_for_file(path: Path) -> str:
    """Compute a hex sha256 for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
