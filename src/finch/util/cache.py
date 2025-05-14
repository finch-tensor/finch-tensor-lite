from .config import get_config
from typing import Callable, Any
import os
import uuid
import tempfile
import atexit
from uuid import UUID
finch_uuid = UUID('ef66f312-ff6e-4b8a-bb8c-9a843f3ecdf4')

def file_cache(ext: str, /, *, cache_dir=None) -> Callable:
    """Caches the result of a function to a file.

    Args:
        f: The function to cache.
        ext: The file extension for the cache file.
        cache_dir: The directory to store the cache files. Defaults to the value from get_config.

    Returns:
        A wrapper function that caches the result of the original function.
    """
    def decorator(f: Callable) -> Callable:
        nonlocal cache_dir
        nonlocal ext
        if get_config("FINCH_CACHE_ENABLE"):
            if cache_dir is None:
                cache_dir = get_config("FINCH_CACHE_PATH")
        else:
            cache_dir = tempfile.mkdtempdir(prefix=get_config("FINCH_TMP"))
            atexit.register(lambda: os.remove(cache_dir) if os.path.exists(cache_dir) else None)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        def inner(*args):
            filename = os.path.join(cache_dir, f"{f.__name__}_{uuid.uuid5(finch_uuid, str((f.__name__, args)))}.{ext}")
            if not get_config("FINCH_CACHE_ENABLE") or not os.path.exists(filename):
                result = f(filename, *args)
                with open(filename, 'w') as cache_file:
                    cache_file.write(str(result))
            return filename
        return inner
    return decorator