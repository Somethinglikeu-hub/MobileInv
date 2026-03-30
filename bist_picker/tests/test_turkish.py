"""Tests for Turkish data handling utilities."""

import time
from datetime import date

import pytest

from bist_picker.utils.turkish import (
    convert_turkish_number,
    normalize_turkish_text,
    parse_turkish_date,
)
from bist_picker.utils.rate_limiter import RateLimiter


# --- convert_turkish_number tests ---

class TestConvertTurkishNumber:

    def test_standard_turkish_format(self):
        assert convert_turkish_number("1.234.567,89") == 1234567.89

    def test_negative_with_minus(self):
        assert convert_turkish_number("-1.234,56") == -1234.56

    def test_negative_with_parentheses(self):
        assert convert_turkish_number("(1.234,56)") == -1234.56

    def test_na_returns_none(self):
        assert convert_turkish_number("N/A") is None

    def test_empty_string_returns_none(self):
        assert convert_turkish_number("") is None

    def test_decimal_only(self):
        assert convert_turkish_number("0,5") == 0.5

    def test_plain_integer(self):
        assert convert_turkish_number("1234") == 1234.0

    def test_none_input(self):
        assert convert_turkish_number(None) is None

    def test_dash_returns_none(self):
        assert convert_turkish_number("-") is None

    def test_double_dash_returns_none(self):
        assert convert_turkish_number("--") is None

    def test_large_number(self):
        assert convert_turkish_number("12.345.678.901,23") == 12345678901.23

    def test_small_decimal(self):
        assert convert_turkish_number("0,01") == 0.01

    def test_thousands_no_decimal(self):
        assert convert_turkish_number("1.234") == 1234.0

    def test_percent_stripped(self):
        assert convert_turkish_number("12,5%") == 12.5

    def test_whitespace_handling(self):
        assert convert_turkish_number("  1.234,56  ") == 1234.56


# --- normalize_turkish_text tests ---

class TestNormalizeTurkishText:

    def test_uppercase_i_with_dot(self):
        assert normalize_turkish_text("İstanbul") == "Istanbul"

    def test_lowercase_dotless_i(self):
        assert normalize_turkish_text("ışık") == "isik"

    def test_all_turkish_chars(self):
        result = normalize_turkish_text("şçğıöüŞÇĞİÖÜ")
        assert result == "scgiouSCGIOU"

    def test_mixed_text(self):
        result = normalize_turkish_text("Türk Hava Yolları")
        assert result == "Turk Hava Yollari"

    def test_none_returns_empty(self):
        assert normalize_turkish_text(None) == ""

    def test_ascii_unchanged(self):
        assert normalize_turkish_text("THYAO") == "THYAO"


# --- parse_turkish_date tests ---

class TestParseTurkishDate:

    def test_dot_separator(self):
        assert parse_turkish_date("15.02.2025") == date(2025, 2, 15)

    def test_dash_separator(self):
        assert parse_turkish_date("15-02-2025") == date(2025, 2, 15)

    def test_slash_separator(self):
        assert parse_turkish_date("15/02/2025") == date(2025, 2, 15)

    def test_none_returns_none(self):
        assert parse_turkish_date(None) is None

    def test_empty_returns_none(self):
        assert parse_turkish_date("") is None

    def test_na_returns_none(self):
        assert parse_turkish_date("N/A") is None

    def test_two_digit_year(self):
        assert parse_turkish_date("15-02-25") == date(2025, 2, 15)

    def test_end_of_month(self):
        assert parse_turkish_date("31.12.2024") == date(2024, 12, 31)

    def test_invalid_date_returns_none(self):
        assert parse_turkish_date("32.13.2025") is None

    def test_whitespace_handling(self):
        assert parse_turkish_date("  15.02.2025  ") == date(2025, 2, 15)


# --- RateLimiter tests ---

class TestRateLimiter:

    def test_enforces_delay(self):
        limiter = RateLimiter(min_delay=0.3, name="test")
        limiter.wait()  # first call, no wait
        start = time.monotonic()
        limiter.wait()  # should wait ~0.3s
        elapsed = time.monotonic() - start
        assert elapsed >= 0.25, f"Expected >=0.25s delay, got {elapsed:.3f}s"

    def test_no_delay_on_first_call(self):
        limiter = RateLimiter(min_delay=1.0, name="test")
        start = time.monotonic()
        waited = limiter.wait()
        elapsed = time.monotonic() - start
        # First call should be nearly instant
        assert elapsed < 0.1

    def test_returns_wait_time(self):
        limiter = RateLimiter(min_delay=0.2, name="test")
        limiter.wait()
        waited = limiter.wait()
        assert waited > 0

    def test_repr(self):
        limiter = RateLimiter(min_delay=2.0, name="kap")
        assert "kap" in repr(limiter)
        assert "2.0" in repr(limiter)
