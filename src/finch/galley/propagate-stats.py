import math
import warnings
from collections import OrderedDict, Counter
from typing import Any, List, Set, Tuple, Callable


# from tensor_stats import TensorDef, NaiveStats, DCStats, TensorStats, DC
from algebra import is_identity, is_annihilator

def merge_tensor_stats_join(op: Callable, *all_stats: TensorStats) -> TensorStats:
    if hasattr(type(all_stats[0]), 'merge_join'):
        return type(all_stats[0]).merge_join(op, *all_stats)
    raise NotImplementedError(f"merge_tensor_stats_join not implemented for {type(all_stats[0])}")


