from typing import Any, Optional, List
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class AnnotatedQuery:
    ST: type
    output_name: Any
    reduce_idxs: List[str]
    idx_lowest_root: OrderedDict[str, int]
    idx_op: OrderedDict[str, Any]
    idx_init: OrderedDict[str, Any]
    parent_idxs: OrderedDict[str, List[str]]
    original_idx: OrderedDict[str, str]
    connected_components: List[List[str]]
    connected_idxs: OrderedDict[str, set[str]]
