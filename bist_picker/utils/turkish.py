"""Turkish data handling utilities for BIST Stock Picker.

Handles Turkey-specific data formats:
- Number format: dot for thousands, comma for decimal (1.234.567,89)
- Turkish characters: ş, ç, ğ, ı, ö, ü, İ, Ş, Ç, Ğ, Ö, Ü
- Date formats: DD-MM-YYYY, DD.MM.YYYY, DD/MM/YYYY
"""

import re
from datetime import date
from typing import Optional


# Turkish → ASCII character mapping for normalization
_TR_CHAR_MAP = str.maketrans(
    "İıŞşÇçĞğÖöÜü",
    "IiSsCcGgOoUu",
)

# Patterns that mean "no data"
_NULL_PATTERNS = {"n/a", "na", "-", "--", "---", ""}


def convert_turkish_number(text: str) -> Optional[float]:
    """Convert a Turkish-formatted number string to float.

    Turkish format uses dot as thousands separator and comma as decimal:
    1.234.567,89 -> 1234567.89

    Args:
        text: Number string in Turkish format.

    Returns:
        Float value, or None if the input represents missing data.
    """
    if text is None:
        return None

    text = str(text).strip()

    if text.lower() in _NULL_PATTERNS:
        return None

    # Handle parentheses for negatives: (1.234,56) -> -1234.56
    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1].strip()

    # Handle explicit negative sign
    if text.startswith("-"):
        is_negative = True
        text = text[1:].strip()

    # Handle percent sign
    text = text.rstrip("%").strip()

    if not text:
        return None

    # Remove thousands separators (dots) and replace decimal comma
    # Key insight: in Turkish format, dots are thousands separators
    # and comma is the decimal separator.
    # But "1234" (no dots, no commas) is just an integer.
    # And "0,5" is 0.5.
    if "," in text:
        # Has decimal comma — dots before it are thousands separators
        text = text.replace(".", "")
        text = text.replace(",", ".")
    else:
        # No comma — dots could be thousands separators
        # "1.234.567" -> 1234567 (multiple dots = thousands seps)
        # "1234" -> 1234 (no dots = plain integer)
        # Single dot ambiguity: "1.5" could be 1.5 or 1500
        # In Turkish context with no comma, multiple dots = thousands
        dot_count = text.count(".")
        if dot_count > 1:
            # Multiple dots = definitely thousands separators
            text = text.replace(".", "")
        elif dot_count == 1:
            # Single dot: check if it's a thousands separator
            # If 3 digits after dot, it's a thousands separator (e.g., "1.234")
            parts = text.split(".")
            if len(parts[1]) == 3:
                text = text.replace(".", "")
            # Otherwise treat as decimal point (e.g., "1.5")

    try:
        value = float(text)
    except ValueError:
        return None

    return -value if is_negative else value


def normalize_turkish_text(text: str) -> str:
    """Normalize Turkish text by replacing Turkish characters with ASCII equivalents.

    Useful for search/matching where Turkish characters might cause mismatches.
    İ->I, ı->i, ş->s, ç->c, ğ->g, ö->o, ü->u (and uppercase variants).

    Args:
        text: Text that may contain Turkish characters.

    Returns:
        ASCII-normalized text.
    """
    if text is None:
        return ""
    return text.translate(_TR_CHAR_MAP)


def parse_turkish_date(text: str) -> Optional[date]:
    """Parse a Turkish-formatted date string.

    Handles common Turkish date formats:
    - DD-MM-YYYY (e.g., 15-02-2025)
    - DD.MM.YYYY (e.g., 15.02.2025)
    - DD/MM/YYYY (e.g., 15/02/2025)

    Args:
        text: Date string in Turkish format.

    Returns:
        datetime.date object, or None if parsing fails.
    """
    if text is None:
        return None

    text = str(text).strip()
    if not text or text.lower() in _NULL_PATTERNS:
        return None

    # Try all three separator styles
    for sep in ("-", ".", "/"):
        parts = text.split(sep)
        if len(parts) == 3:
            try:
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                # Handle 2-digit year
                if year < 100:
                    year += 2000
                return date(year, month, day)
            except (ValueError, OverflowError):
                continue

    return None
