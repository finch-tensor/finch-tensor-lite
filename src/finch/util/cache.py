from .config import get_config
from typing import Callable, Any
import os
import uuid
import tempfile
from uuid import UUID
finch_uuid = UUID('ef66f312-ff6e-4b8a-bb8c-9a843f3ecdf4')

def file_cache(f: Callable, ext: str, /, *, cache_dir=None) -> Callable:
    """Caches the result of a function to a file.

    Args:
        f: The function to cache.
        ext: The file extension for the cache file.
        cache_dir: The directory to store the cache files. Defaults to the value from get_config.

    Returns:
        A wrapper function that caches the result of the original function.
    """
    if get_config("FINCH_CACHE_ENABLED")
        if cache_dir is None:
            cache_dir = get_config("FINCH_CACHE_PATH")
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "finch_cache")
        tempfile.mkd

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    def inner(*args):
        filename = os.path.join(cache_dir, f"{f.__name__}_{uuid.uuid5(finch_uuid, str((f.__name__, args)))}.{ext}")
        if not get_config("FINCH_CACHE_ENABLED") or not os.path.exists(filename):
            result = f(*args)
            with open(filename, 'w') as cache_file:
                cache_file.write(str(result))
        return filename

    return inner