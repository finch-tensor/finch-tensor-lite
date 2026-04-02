# AI modified: 2026-04-02T20:46:24Z parent=154b5aeaa66d01a2373296ba9af9705a3db73ed9
from finchlite.util import cache


def test_ensure_cache_fresh_clears_cache_when_code_changes(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached_file = cache_root / "c" / "artifact.txt"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_text("cached")
    timestamp_file = cache_root / cache.cache_timestamp_filename
    timestamp_file.write_text("10")

    monkeypatch.setattr(cache, "_latest_finch_code_mtime_ns", lambda: 20)
    cache._checked_cache_roots.clear()

    cache._ensure_cache_fresh(cache_root)

    assert not cached_file.exists()
    assert timestamp_file.read_text() == "20"


def test_ensure_cache_fresh_keeps_cache_when_code_unchanged(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached_file = cache_root / "c" / "artifact.txt"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_text("cached")
    timestamp_file = cache_root / cache.cache_timestamp_filename
    timestamp_file.write_text("20")

    monkeypatch.setattr(cache, "_latest_finch_code_mtime_ns", lambda: 20)
    cache._checked_cache_roots.clear()

    cache._ensure_cache_fresh(cache_root)

    assert cached_file.exists()
    assert timestamp_file.read_text() == "20"
