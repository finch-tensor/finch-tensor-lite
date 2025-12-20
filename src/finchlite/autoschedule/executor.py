from typing import Any, overload

from finchlite.algebra.tensor import Tensor, TensorFType
from finchlite.finch_logic.nodes import TableValue

from .. import finch_logic as lgc
from ..finch_logic import LogicEvaluator, LogicLoader, LogicNode
from ..symbolic import Namespace, PostWalk, Rewrite, ftype
from .formatter import LogicFormatter


def extract_tensors(
    root: lgc.LogicStatement,
    bindings: dict[lgc.Alias, Tensor],
) -> tuple[lgc.LogicStatement, dict[lgc.Alias, Tensor]]:
    """
    Extracts tensors from logic plan, replacing them with aliases.
    """
    bindings = bindings.copy()
    ids: dict[int, lgc.Alias] = {id(val): key for key, val in bindings.items()}
    spc = Namespace(root)
    for alias in bindings:
        spc.freshen(alias.name)

    def rule_0(node):
        match node:
            case lgc.Table(lgc.Literal(tns), idxs):
                if id(tns) in ids:
                    var = ids[id(tns)]
                    if bindings[var].idxs == idxs:
                        return var
                    return lgc.Table(var, idxs)
                var = lgc.Alias(spc.freshen("A"))
                ids[id(tns)] = var
                bindings[var] = tns
                return lgc.Table(var, idxs)

    root = Rewrite(PostWalk(rule_0))(root)
    return root, bindings


class ProvisionTensorsContext:
    def __init__(
        self,
        bindings: dict[lgc.Alias, Tensor],
        types: dict[lgc.Alias, TensorFType],
    ):
        self.bindings: dict[lgc.Alias, Tensor] = bindings.copy()
        self.shapes: dict[lgc.Alias, tuple[Any, ...]] = {
            var: tns.shape for var, tns in bindings.items()
        }
        self.types: dict[lgc.Alias, TensorFType] = types

    def __call__(self, node: lgc.LogicStatement) -> dict[lgc.Alias, Tensor]:
        match node:
            case lgc.Plan(bodies):
                for body in bodies:
                    self(body)
            case lgc.Query(lhs, rhs):
                if lhs not in self.bindings:
                    if lhs not in self.types:
                        raise ValueError(
                            f"Type information missing for {lhs}, did you run"
                            f" tensor formatter?"
                        )
                    shape = rhs.shape(self.shapes)
                    self.bindings[lhs] = self.types[lhs](
                        tuple(dim if dim is not None else 1 for dim in shape)
                    )
                    self.shapes[lhs] = shape
            case lgc.Produces(_):
                pass
            case _:
                raise ValueError(f"Unknown LogicStatement: {type(node)}")
        return self.bindings


class LogicExecutor(LogicEvaluator):
    def __init__(self, ctx: LogicLoader | None = None, verbose: bool = False):
        if ctx is None:
            ctx = LogicFormatter()
        self.ctx: LogicLoader = ctx
        self.verbose: bool = verbose

    def __call__(
        self, prgm: LogicNode, bindings: dict[lgc.Alias, Tensor] | None = None
    ):
        if bindings is None:
            bindings = {}
        
        if isinstance(prgm, lgc.LogicExpression):
            var = lgc.Alias("result")
            stmt: lgc.LogicStatement = lgc.Plan(
                (lgc.Query(var, prgm), lgc.Produces((var,)))
            )
        elif isinstance(prgm, lgc.LogicStatement):
            stmt = prgm
        else:
            raise ValueError(f"Invalid prgm type: {type(prgm)}")
        stmt, bindings = extract_tensors(stmt, bindings)
        binding_ftypes = {var: ftype(val) for var, val in bindings.items()}

        mod, stmt, binding_ftypes = self.ctx(stmt, binding_ftypes)

        bindings = ProvisionTensorsContext(bindings, binding_ftypes)(stmt)
        args = [tns for tns in bindings.values()]

        res = mod.main(*args)

        if isinstance(prgm, lgc.LogicExpression):
            return TableValue(res[0], prgm.fields())
        return tuple(res)

