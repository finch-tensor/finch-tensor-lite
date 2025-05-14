from .config import get_config
from typing import Callable, Any
import os
import uuid
import tempfile
import atexit
import shutil
from uuid import UUID
finch_uuid = UUID('ef66f312-ff6e-4b8a-bb8c-9a843f3ecdf4')

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
            cache_dir = os.path.join(get_config("FINCH_CACHE_PATH"), domain)
        else:
            cache_dir = tempfile.mkdtemp(prefix=os.path.join(get_config("FINCH_TMP"), domain))
            atexit.register(lambda: shutil.rmtree(cache_dir) if os.path.exists(cache_dir) else None)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        def inner(*args):
            filename = os.path.join(cache_dir, f"{f.__name__}_{uuid.uuid5(finch_uuid, str((f.__name__, args)))}.{ext}")
            if not get_config("FINCH_CACHE_ENABLE") or not os.path.exists(filename):
                f(filename, *args)
            return filename
        return inner
    return decorator