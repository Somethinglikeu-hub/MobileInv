"""Custom sub-sector mapper for BIST Stock Picker.

Maps companies from BIST's coarse official sectors to our custom sub-sectors
(30+ categories defined in config/sectors.yaml). Fine-grained sub-sectors
enable meaningful sector-neutral z-score normalization in the scoring stage.

Mapping priority:
1. Manual ticker mapping from sectors.yaml (highest priority)
2. BIST sector name fallback rules from sectors.yaml
3. Default to "other"
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from bist_picker.db.schema import Company

logger = logging.getLogger("bist_picker.classification.sector_mapper")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sectors.yaml"


class SectorMapper:
    """Maps BIST companies to custom sub-sectors using config-driven rules.

    Loads mapping rules from sectors.yaml:
    - ticker_mapping: exact ticker -> sub-sector overrides
    - bist_sector_fallback: BIST sector name -> sub-sector defaults
    - sub_sectors: list of valid sub-sector names

    Args:
        config_path: Path to sectors.yaml. Defaults to config/sectors.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._ticker_mapping: dict[str, str] = {}
        self._sector_fallback: dict[str, str] = {}
        self._valid_sectors: set[str] = set()
        self._load_config()

    def _load_config(self) -> None:
        """Load and validate sectors.yaml configuration."""
        if not self._config_path.exists():
            logger.warning("Sectors config not found: %s", self._config_path)
            return

        with open(self._config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            logger.warning("Empty sectors config: %s", self._config_path)
            return

        self._valid_sectors = set(config.get("sub_sectors", []))
        self._ticker_mapping = config.get("ticker_mapping", {})
        self._sector_fallback = config.get("bist_sector_fallback", {})

        logger.debug(
            "Loaded sectors config: %d sub-sectors, %d ticker mappings, %d fallback rules",
            len(self._valid_sectors),
            len(self._ticker_mapping),
            len(self._sector_fallback),
        )

    def map_sector(self, ticker: str, bist_sector: str) -> str:
        """Map a single company to a custom sub-sector.

        Args:
            ticker: BIST ticker code (e.g., "THYAO").
            bist_sector: Official BIST sector name.

        Returns:
            Custom sub-sector string (e.g., "airlines", "banking_private").
        """
        ticker_upper = ticker.strip().upper()

        # 1. Manual ticker mapping (highest priority)
        if ticker_upper in self._ticker_mapping:
            mapped = self._ticker_mapping[ticker_upper]
            if mapped in self._valid_sectors:
                return mapped
            logger.warning(
                "Ticker %s mapped to invalid sub-sector '%s', falling back",
                ticker_upper, mapped,
            )

        # 2. BIST sector name fallback
        sector_str = (bist_sector or "").strip()
        if sector_str:
            # Try exact match first
            if sector_str in self._sector_fallback:
                return self._sector_fallback[sector_str]

            # Try substring match (BIST sectors can be verbose)
            sector_lower = sector_str.lower()
            for key, sub_sector in self._sector_fallback.items():
                if key.lower() in sector_lower or sector_lower in key.lower():
                    return sub_sector

        # 3. Default
        return "other"

    def map_all(self, session: Session) -> dict:
        """Map all active companies to custom sub-sectors and update the DB.

        Args:
            session: SQLAlchemy session.

        Returns:
            Stats dict: {total, updated, by_sector: {sector: count}}.
        """
        companies = (
            session.query(Company)
            .filter(Company.is_active.is_(True))
            .all()
        )

        stats = {
            "total": len(companies),
            "updated": 0,
            "by_sector": {},
        }

        for company in companies:
            sub_sector = self.map_sector(
                ticker=company.ticker,
                bist_sector=company.sector_bist or "",
            )

            if company.sector_custom != sub_sector:
                company.sector_custom = sub_sector
                stats["updated"] += 1

            stats["by_sector"][sub_sector] = stats["by_sector"].get(sub_sector, 0) + 1

        if stats["updated"] > 0:
            session.flush()

        logger.info(
            "Mapped %d companies to %d sub-sectors (updated %d)",
            stats["total"],
            len(stats["by_sector"]),
            stats["updated"],
        )
        return stats
