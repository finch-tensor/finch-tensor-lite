from inspect import isbuiltin, isclass, isfunction
from typing import Any


class SPrinter:
    def __call__(self, val: Any) -> str:
        return str(val)


class Printer:
    def __call__(self, val: Any):
        print(val)
        return


def qual_str(val: Any) -> str:
    if hasattr(val, "__qual_str__"):
        return val.__qual_str__()
    if isbuiltin(val) or isclass(val) or isfunction(val):
        return str(val.__qualname__)
    return str(val)
