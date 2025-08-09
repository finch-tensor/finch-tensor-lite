from inspect import isbuiltin, isclass, isfunction
from typing import Any


def qual_str(val: Any) -> str:
    if hasattr(val, "__qual_str__"):
        return val.__qual_str__()
    if isbuiltin(val) or isclass(val) or isfunction(val):
        return str(val.__qualname__)
    return str(val)
