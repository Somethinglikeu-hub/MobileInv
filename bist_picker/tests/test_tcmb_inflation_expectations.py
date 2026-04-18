"""Tests for TCMBClient.fetch_inflation_expectations_24m.

The fetcher tries several candidate EVDS series codes for robustness
(TCMB renames codes occasionally). These tests mock ``_fetch_evds`` to
validate each path without hitting the network.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bist_picker.data.sources.tcmb import TCMBClient


@pytest.fixture
def client():
    # Inject a dummy key so the method doesn't short-circuit.
    return TCMBClient(api_key="dummy-key")


def test_returns_none_without_api_key():
    c = TCMBClient(api_key="")
    c._api_key = None
    assert c.fetch_inflation_expectations_24m() is None


def test_uses_first_successful_series(client):
    """First candidate series returns data → returned, others not tried."""
    def fake_fetch(series, *args, **kwargs):
        if series == client._INFLATION_EXP_24M_SERIES[0]:
            key = series.replace(".", "_")
            return [{"Tarih": "2025-6", key: "18.5"}]
        return []

    with patch.object(client, "_fetch_evds", side_effect=fake_fetch) as mock:
        rate = client.fetch_inflation_expectations_24m()

    assert rate == pytest.approx(0.185)
    # Only the first series should have been tried
    assert mock.call_count == 1


def test_falls_through_to_next_series_on_empty(client):
    """First series returns empty → try the next one."""
    def fake_fetch(series, *args, **kwargs):
        if series == client._INFLATION_EXP_24M_SERIES[1]:
            key = series.replace(".", "_")
            return [{"Tarih": "2025-6", key: "22.0"}]
        return []

    with patch.object(client, "_fetch_evds", side_effect=fake_fetch) as mock:
        rate = client.fetch_inflation_expectations_24m()

    assert rate == pytest.approx(0.22)
    assert mock.call_count == 2


def test_returns_none_when_all_series_empty(client):
    with patch.object(client, "_fetch_evds", return_value=[]):
        rate = client.fetch_inflation_expectations_24m()
    assert rate is None


def test_returns_most_recent_nonnull_value(client):
    """Older nulls are skipped — we want the latest real observation."""
    series = client._INFLATION_EXP_24M_SERIES[0]
    key = series.replace(".", "_")

    items = [
        {"Tarih": "2025-4", key: "20.0"},
        {"Tarih": "2025-5", key: "21.0"},
        {"Tarih": "2025-6", key: ""},        # latest is empty
    ]

    def fake_fetch(s, *args, **kwargs):
        return items if s == series else []

    with patch.object(client, "_fetch_evds", side_effect=fake_fetch):
        rate = client.fetch_inflation_expectations_24m()

    assert rate == pytest.approx(0.21)  # picks 2025-5, skipping empty 2025-6


def test_handles_garbage_value_gracefully(client):
    """Non-numeric value → skipped, falls through to next series."""
    series_first = client._INFLATION_EXP_24M_SERIES[0]
    key_first = series_first.replace(".", "_")
    series_second = client._INFLATION_EXP_24M_SERIES[1]
    key_second = series_second.replace(".", "_")

    def fake_fetch(s, *args, **kwargs):
        if s == series_first:
            return [{"Tarih": "2025-6", key_first: "NOT-A-NUMBER"}]
        if s == series_second:
            return [{"Tarih": "2025-6", key_second: "19.0"}]
        return []

    with patch.object(client, "_fetch_evds", side_effect=fake_fetch):
        rate = client.fetch_inflation_expectations_24m()

    assert rate == pytest.approx(0.19)
