# AI modified: 2026-04-02T20:46:24Z parent=154b5aeaa66d01a2373296ba9af9705a3db73ed9
import atexit
import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from .config import config, get_version

finch_uuid = UUID("ef66f312-ff6e-4b8a-bb8c-9a843f3ecdf4")
cache_timestamp_filename = ".finch_code_mtime_ns"
_checked_cache_roots: set[Path] = set()


def _latest_finch_code_mtime_ns() -> int:
    finch_root = Path(__file__).resolve().parents[1]
    latest_mtime = 0
    for path in finch_root.rglob("*"):
        if path.is_file():
            latest_mtime = max(latest_mtime, path.stat().st_mtime_ns)
    return latest_mtime


def _clear_cache_root(cache_root: Path) -> None:
    for path in cache_root.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _ensure_cache_fresh(cache_root: Path) -> None:
    if cache_root in _checked_cache_roots:
        return
    _checked_cache_roots.add(cache_root)

    cache_root.mkdir(parents=True, exist_ok=True)
    timestamp_file = cache_root / cache_timestamp_filename
    current_mtime = _latest_finch_code_mtime_ns()

    if timestamp_file.exists():
        try:
            cached_mtime = int(timestamp_file.read_text().strip())
        except ValueError:
            cached_mtime = -1
        if current_mtime > cached_mtime:
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
            cache_root = Path(config.get("data_path")) / "cache" / get_version()
            _ensure_cache_fresh(cache_root)
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
