from typing import Any, Iterable, Callable, Type, TypeVar
from .tensor_stats import TensorStats
from .tensor_def import TensorDef

T = TypeVar("T", bound="DenseStats")

class DenseStats(TensorStats):
    @classmethod
    def from_tensor(self, tensor: Any, fields: Iterable[str]) -> None:
        self.tensordef = TensorDef.from_tensor(tensor, fields)

    @classmethod
    def from_def(cls: Type[T], d: TensorDef) -> T:
        stats = object.__new__(cls)
        stats.tensordef = d.copy()
        return stats

    @staticmethod
    def estimate_non_fill_values(arg: TensorStats) -> float:
        total = 1.0
        for size in arg.get_dim_sizes().values():
            total *= size
        return total

    @staticmethod
    def mapjoin(op: Callable, *args: TensorStats) -> TensorStats:
        new_axes = set().union(*(s.get_index_set() for s in args))

        new_dims = {}
        for i in new_axes:
            for j in args:
                if i in j.get_index_set():
                    new_dims[i] = j.get_dim_size(i)
                    break

        new_fill = op(*(m.get_fill_value() for m in args))

        new_def = TensorDef(new_axes, new_dims, new_fill)
        return DenseStats.from_def(new_def)


    @staticmethod
    def aggregate(op: Callable, fields: Iterable[str], arg: TensorStats) -> "DenseStats":
        new_axes = set(arg.get_index_set()) - set(fields)
        new_dims = {m: arg.get_dim_size(m) for m in new_axes}
        new_fill = arg.get_fill_value()

        new_def = TensorDef(new_axes, new_dims, new_fill)
        return DenseStats.from_def(new_def)

    @staticmethod
    def issimilar(a: TensorStats, b: TensorStats) -> bool:
        return (isinstance(a, DenseStats) and isinstance(b, DenseStats)
            and a.tensordef.get_dim_sizes() == b.tensordef.get_dim_sizes()
            and a.tensordef.get_fill_value() == b.tensordef.get_fill_value()
        )