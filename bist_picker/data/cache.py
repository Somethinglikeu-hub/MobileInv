"""File-based JSON cache for raw API responses.

Stores raw API responses as JSON files in data/cache/ with TTL-based expiry.
This provides a debug trail and avoids re-fetching recently downloaded data.
"""

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("bist_picker.data.cache")

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


class FileCache:
    """File-based JSON cache with TTL expiry.

    Args:
        cache_dir: Directory to store cache files. Defaults to data/cache/.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_response(
        self, source: str, ticker: str, data_type: str, data: Any
    ) -> Path:
        """Save a raw API response as a JSON file.

        Args:
            source: Data source name (e.g., "isyatirim", "kap").
            ticker: Ticker code (e.g., "THYAO").
            data_type: Type of data (e.g., "financials", "prices").
            data: JSON-serializable data to cache.

        Returns:
            Path to the saved cache file.
        """
        filename = f"{source}_{ticker}_{data_type}_{date.today().isoformat()}.json"
        filepath = self._cache_dir / filename

        try:
            filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.debug("Cached %s/%s/%s to %s", source, ticker, data_type, filepath.name)
        except (OSError, TypeError) as e:
            logger.warning("Failed to cache %s/%s/%s: %s", source, ticker, data_type, e)

        return filepath

    def load_raw_response(
        self,
        source: str,
        ticker: str,
        data_type: str,
        max_age_hours: float = 24.0,
    ) -> Optional[Any]:
        """Load a cached response if it exists and is not expired.

        Args:
            source: Data source name.
            ticker: Ticker code.
            data_type: Type of data.
            max_age_hours: Maximum age in hours before cache is considered stale.

        Returns:
            Cached data, or None if not found or expired.
        """
        # Look for matching files (most recent first)
        pattern = f"{source}_{ticker}_{data_type}_*.json"
        matches = sorted(self._cache_dir.glob(pattern), reverse=True)

        if not matches:
            return None

        filepath = matches[0]
        age_hours = (time.time() - filepath.stat().st_mtime) / 3600.0

        if age_hours > max_age_hours:
            logger.debug("Cache expired for %s/%s/%s (%.1fh old)", source, ticker, data_type, age_hours)
            return None

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            logger.debug("Cache hit for %s/%s/%s (%.1fh old)", source, ticker, data_type, age_hours)
            return data
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read cache file %s: %s", filepath, e)
            return None

    def clear(self, source: str | None = None, max_age_days: int = 30) -> int:
        """Delete old cache files.

        Args:
            source: If set, only clear files for this source. Otherwise clear all.
            max_age_days: Delete files older than this many days.

        Returns:
            Number of files deleted.
        """
        pattern = f"{source}_*.json" if source else "*.json"
        deleted = 0
        cutoff = time.time() - (max_age_days * 86400)

        for filepath in self._cache_dir.glob(pattern):
            if filepath.stat().st_mtime < cutoff:
                filepath.unlink()
                deleted += 1

        if deleted:
            logger.info("Cleared %d cache files (older than %d days)", deleted, max_age_days)
        return deleted
