import atexit
import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from .config import get_config, get_version

finch_uuid = UUID("ef66f312-ff6e-4b8a-bb8c-9a843f3ecdf4")


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
        if get_config("FINCH_CACHE_ENABLE"):
            cache_dir = Path(get_config("FINCH_CACHE_PATH")) / get_version() / domain
        else:
            cache_dir = Path(
                tempfile.mkdtemp(prefix=str(Path(get_config("FINCH_TMP")) / domain))
            )
            atexit.register(
                lambda: shutil.rmtree(cache_dir) if cache_dir.exists() else None
            )

        cache_dir.mkdir(parents=True, exist_ok=True)

        def inner(*args):
            id = uuid.uuid5(finch_uuid, str((f.__name__, f.__module__, args)))
            filename = cache_dir / f"{f.__name__}_{id}.{ext}"
            if not get_config("FINCH_CACHE_ENABLE") or not filename.exists():
                f(str(filename), *args)
            return filename

        return inner

    return decorator
