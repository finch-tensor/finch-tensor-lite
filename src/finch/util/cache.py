from .config import get_config
from typing import Callable, Any
import os
import uuid
from uuid import UUID
finch_uuid = UUID('ef66f312-ff6e-4b8a-bb8c-9a843f3ecdf4')

def cache(f: Callable, ext: str) -> Callable:
    """Caches the result of a function to a file.

    Args:
        f: The function to cache.
        cache_dir: The directory to store the cache files.

    Returns:
        A wrapper function that caches the result of the original function.
    """
    cache_path = get_config("FINCH_CACHE_PATH")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    def inner(*args):
        filename = os.path.join(cache_path, f"{f.__name__}_{uuid.uuid5(finch_uuid, str((f.__name__, args)))}.{ext}")
        if not os.path.exists(filename):
            result = f(*args)
            with open(filename, 'w') as cache_file:
                cache_file.write(str(result))
        f(filename, *args)
        assert os.path.exists(filename), f"Couldn't create cache file {f.__name__}({", ".join(*args)})"
    return inner