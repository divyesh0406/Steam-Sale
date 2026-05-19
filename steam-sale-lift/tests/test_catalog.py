"""
Smoke tests for the catalog fetch and single-game metadata flow.

These tests make real network calls — they are integration smoke tests,
not unit tests. Run them manually to verify the pipeline still works:

    uv run pytest tests/test_catalog.py -v -s

They are excluded from CI (which only runs the no-network unit tests).
"""

import json
import time
from pathlib import Path

import pytest

# Mark all tests in this file as network-dependent
pytestmark = pytest.mark.network


def test_fetch_app_catalog_shape(tmp_path, monkeypatch):
    """
    Catalog must return ≥ 100,000 entries, each with int appid and str name.
    Cache file must be created in CACHE_DIR.
    """
    # Redirect CACHE_DIR to tmp_path so we don't pollute data/raw/cache/
    import src.scrape.steam_api as mod
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)

    catalog = mod.fetch_app_catalog()

    assert isinstance(catalog, list), "fetch_app_catalog() must return a list"
    assert len(catalog) >= 100_000, (
        f"Expected ≥ 100,000 appids from the catalog mirror, got {len(catalog):,}"
    )

    # Spot-check shape
    for entry in catalog[:10]:
        assert isinstance(entry["appid"], int), f"appid must be int, got {type(entry['appid'])}"
        assert isinstance(entry["name"], str), f"name must be str, got {type(entry['name'])}"

    # Cache file must have been written
    cache_files = list(tmp_path.glob("appid_list_*.json"))
    assert len(cache_files) == 1, f"Expected exactly one cache file, found: {cache_files}"
    print(f"\n  Catalog size: {len(catalog):,} appids")
    print(f"  Cache file:  {cache_files[0].name}")


def test_fetch_app_catalog_cache_hit(tmp_path, monkeypatch):
    """Second call with same CACHE_DIR returns the cached file without re-fetching."""
    import src.scrape.steam_api as mod
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)

    catalog1 = mod.fetch_app_catalog()
    # Write a fake smaller cache file at the same path so we can detect cache hit
    cache_file = list(tmp_path.glob("appid_list_*.json"))[0]
    fake = [{"appid": 999999, "name": "fake game"}]
    cache_file.write_text(json.dumps(fake), encoding="utf-8")

    catalog2 = mod.fetch_app_catalog()
    assert catalog2 == fake, "Second call should return cached content, not re-fetch"


def test_app_details_team_fortress_2(tmp_path, monkeypatch):
    """
    Fetch metadata for appid 440 (Team Fortress 2) via the existing appdetails flow.
    Confirms the Store API path still works end-to-end after the catalog migration.
    """
    import src.scrape.steam_api as mod
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)

    client = mod.SteamClient()
    details = client.get_app_details(440)

    assert details is not None, "appdetails for TF2 (440) returned None — endpoint broken?"
    assert details.get("type") == "game", f"Expected type='game', got {details.get('type')!r}"
    assert "Team Fortress" in details.get("name", ""), (
        f"Unexpected name: {details.get('name')!r}"
    )
    print(f"\n  TF2 appdetails OK: name={details['name']!r}, type={details['type']!r}")
