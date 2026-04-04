from pathlib import Path

from src.utils.cache import ToolCache


def test_tool_cache_search_and_content_roundtrip(tmp_path: Path):
    cache = ToolCache(cache_dir=tmp_path / "cache", ttl_days=7, max_entries=10)

    cache.set_search("duckduckgo", "Test Query", 3, [{"title": "x", "url": "u"}])
    got = cache.get_search("duckduckgo", "test   query", 3)
    assert got == [{"title": "x", "url": "u"}]

    cache.set_content("https://example.com/a#frag", "hello")
    assert cache.get_content("https://example.com/a") == "hello"
