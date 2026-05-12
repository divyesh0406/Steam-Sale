"""
Unit tests for the SteamClient scraper.
Uses pytest-httpx to mock HTTP responses — no real API calls made.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_app_list_response() -> dict:
    return {
        "applist": {
            "apps": [
                {"appid": 570, "name": "Dota 2"},
                {"appid": 730, "name": "Counter-Strike 2"},
            ]
        }
    }


def make_appdetails_response(appid: int) -> dict:
    return {
        str(appid): {
            "success": True,
            "data": {
                "type": "game",
                "name": "Test Game",
                "steam_appid": appid,
                "price_overview": {"initial": 1999, "final": 1999, "discount_percent": 0},
                "developers": ["Test Dev"],
                "publishers": ["Test Publisher"],
                "genres": [{"id": "23", "description": "Indie"}],
                "categories": [],
                "release_date": {"coming_soon": False, "date": "15 Jan, 2022"},
            },
        }
    }


def make_reviews_response(cursor: str = "next") -> dict:
    return {
        "success": 1,
        "cursor": cursor,
        "reviews": [
            {
                "recommendationid": "12345",
                "voted_up": True,
                "timestamp_created": 1704067200,
                "author": {"playtime_at_review": 300},
                "votes_up": 5,
                "weighted_vote_score": "0.8",
            }
        ],
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSteamClientThrottling:
    def test_cache_key_deterministic(self, tmp_path):
        """Same URL + params always produce the same cache key."""
        import os
        # Patch cache dir to tmp_path so we don't pollute data/raw/cache
        with patch("src.scrape.steam_api.CACHE_DIR", tmp_path):
            from src.scrape.steam_api import SteamClient
            client = SteamClient(api_key="test")
            key1 = client._cache_key("https://example.com", {"a": 1})
            key2 = client._cache_key("https://example.com", {"a": 1})
            assert key1 == key2

    def test_cache_key_differs_by_params(self, tmp_path):
        with patch("src.scrape.steam_api.CACHE_DIR", tmp_path):
            from src.scrape.steam_api import SteamClient
            client = SteamClient(api_key="test")
            key1 = client._cache_key("https://example.com", {"a": 1})
            key2 = client._cache_key("https://example.com", {"a": 2})
            assert key1 != key2

    def test_cache_hit_skips_network(self, tmp_path):
        """If cache file exists, get() returns it without HTTP call."""
        cached = {"cached": True}
        with patch("src.scrape.steam_api.CACHE_DIR", tmp_path):
            from src.scrape.steam_api import SteamClient
            client = SteamClient(api_key="test")
            cache_path = client._cache_key("https://example.com", {})
            cache_path.write_text(json.dumps(cached), encoding="utf-8")

            # If network were called, it would fail (no mock). This must not raise.
            result = client.get("https://example.com", {}, use_cache=True)
            assert result == cached


class TestOwnersParsing:
    def test_parse_owners_normal(self):
        from src.load.ingest import _parse_owners
        assert _parse_owners("200000 .. 500000") == (200000, 500000)

    def test_parse_owners_with_commas(self):
        from src.load.ingest import _parse_owners
        assert _parse_owners("1,000,000 .. 2,000,000") == (1000000, 2000000)

    def test_parse_owners_malformed(self):
        from src.load.ingest import _parse_owners
        assert _parse_owners("bad data") == (0, 0)


class TestDateToSaleEvent:
    def test_summer_2024_start(self):
        from datetime import date
        from src.scrape.steamdb import date_to_sale_event
        # Summer Sale 2024: June 27 – July 11 2024
        assert date_to_sale_event(date(2024, 6, 27)) == "summer_2024"

    def test_summer_2024_end(self):
        from datetime import date
        from src.scrape.steamdb import date_to_sale_event
        assert date_to_sale_event(date(2024, 7, 11)) == "summer_2024"

    def test_non_sale_date(self):
        from datetime import date
        from src.scrape.steamdb import date_to_sale_event
        # Mid-March 2024 — no sale
        assert date_to_sale_event(date(2024, 3, 1)) is None


class TestSaleEventCoverage:
    def test_all_events_have_required_fields(self):
        from src.scrape.steam_api import KNOWN_SALE_EVENTS
        for e in KNOWN_SALE_EVENTS:
            assert "id" in e
            assert "name" in e
            assert "start" in e
            assert "end" in e
            assert "type" in e
            assert e["type"] in ("major", "minor")

    def test_event_start_before_end(self):
        from datetime import datetime
        from src.scrape.steam_api import KNOWN_SALE_EVENTS
        for e in KNOWN_SALE_EVENTS:
            start = datetime.strptime(e["start"], "%Y-%m-%d")
            end = datetime.strptime(e["end"], "%Y-%m-%d")
            assert start < end, f"Sale {e['id']} has start >= end"
