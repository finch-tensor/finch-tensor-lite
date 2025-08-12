from inspect import isbuiltin, isclass, isfunction
from typing import Any, Optional

class SQPrinter:
    def __init__(self, normalize:bool=False, heap:Optional[dict]=None):
        if normalize:
            heap = {}
        self.heap = heap
    
    def __call__(self, val: Any) -> str:
        return qstr(val, heap=self.heap)

def qstr(val: Any, normalize:bool = False, heap: Optional[dict] = None) -> str:
    if normalize:
        heap = {}
    if hasattr(val, "qstr"):
        return val.qstr(heap=heap)
    if isbuiltin(val) or isclass(val) or isfunction(val):
        return str(val.__qualname__)
    if val.__str__ == object.__str__:
        if heap is not None:
            count = heap.setdefault(id(val), len(heap))
            return f"<{val.__qualname__} {count}>"
    return str(val)