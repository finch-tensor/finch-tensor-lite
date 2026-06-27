# AI modified: 2024-12-31T23:58:00Z parent=154b5aeaa66d01a2373296ba9af9705a3db73ed9
# AI modified: 2024-12-31T23:59:00Z parent=06953a764918de34b3a35c1b698198c3b74c5890
# AI modified: 2025-01-01T00:00:00Z parent=4f5a2e5021678965ce8d830bb9edecac1dd3fea9
# AI modified: 2025-01-01T00:01:00Z parent=4f5a2e5021678965ce8d830bb9edecac1dd3fea9
# AI modified: 2026-04-02T22:59:00Z parent=197d5a907823d2a53fcd3b68b674f3f4d4f50b5d
# AI modified: 2026-04-03T15:30:00Z parent=36276c257318d74488f81fa8107d2f2d0a8b804c
import atexit
import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import finchlite

from .config import config, get_version

finch_uuid = UUID("ef66f312-ff6e-4b8a-bb8c-9a843f3ecdf4")
cache_timestamp_filename = ".finch_code_mtime_ns"
_finch_source_root = Path(finchlite.__path__[0])


def _latest_finch_code_mtime_ns() -> int:
    latest_mtime = 0
    for path in _finch_source_root.rglob("*"):
        if (
            "__pycache__" not in path.parts
            and path.is_file()
            and path.suffix not in {".pyc", ".pyo"}
        ):
            latest_mtime = max(latest_mtime, path.stat().st_mtime_ns)
    return latest_mtime


_session_finch_code_mtime_ns = _latest_finch_code_mtime_ns()
_cache_checked = False


def _clear_cache_root(cache_root: Path, *, keep_timestamp: bool = True) -> None:
    for path in cache_root.iterdir():
        if keep_timestamp and path.name == cache_timestamp_filename:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def clear_cache() -> None:
    """Clear Finch's persistent cache for the current Finch version.

    This removes all cached files for the active Finch version under
    ``<data_path>/cache/<version>``. If the cache directory does not exist,
    this function does nothing.
    """

    global _cache_checked
    cache_root = Path(config.get("data_path")) / "cache" / get_version()
    if cache_root.exists():
        _clear_cache_root(cache_root, keep_timestamp=False)
    _cache_checked = False


def _ensure_cache_fresh(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    timestamp_file = cache_root / cache_timestamp_filename
    current_mtime = _session_finch_code_mtime_ns
    should_clear = False

    if timestamp_file.exists():
        try:
            cached_mtime = int(timestamp_file.read_text().strip())
        except ValueError:
            should_clear = True
        else:
            should_clear = current_mtime > cached_mtime
    else:
        should_clear = True

    if should_clear:
        _clear_cache_root(cache_root)

    timestamp_file.write_text(str(current_mtime))


def file_cache(*, ext: str, domain: str) -> Callable:
    """Caches the result of a function to a file.

    Args:
        f: The function to cache.
        ext: The file extension for the cache file.
        domain: The domain name for the cache file.

    Returns:
        A wrapper function that caches the result of the original function.
    """

    def decorator(f: Callable) -> Callable:
        nonlocal domain
        nonlocal ext
        ext = ext.lstrip(".")
        if config.get("cache_enable"):
            global _cache_checked
            cache_root = Path(config.get("data_path")) / "cache" / get_version()
            if not _cache_checked:
                _ensure_cache_fresh(cache_root)
                _cache_checked = True
            cache_dir = cache_root / domain
        else:
            cache_dir = Path(
                tempfile.mkdtemp(
                    prefix=str(Path(config.get("data_path")) / "tmp" / domain)
                )
            )
            atexit.register(
                lambda: shutil.rmtree(cache_dir) if cache_dir.exists() else None
            )

        cache_dir.mkdir(parents=True, exist_ok=True)

        def inner(*args):
            id = uuid.uuid5(finch_uuid, str((f.__name__, f.__module__, args)))
            filename = cache_dir / f"{f.__name__}_{id}.{ext}"
            if not config.get("cache_enable") or not filename.exists():
                f(str(filename), *args)
            return filename

        return inner

    return decorator
