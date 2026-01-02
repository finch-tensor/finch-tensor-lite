from finchlite.algebra.tensor import Tensor
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
                    return lgc.Table(var, idxs)
                var = lgc.Alias(spc.freshen("A"))
                ids[id(tns)] = var
                bindings[var] = tns
                return lgc.Table(var, idxs)

    root = Rewrite(PostWalk(rule_0))(root)
    return root, bindings


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

        mod, binding_ftypes, binding_idxs = self.ctx(stmt, binding_ftypes)

        binding_shapes = dict[lgc.Field | None, int]()
        for var, tns in bindings.items():
            for idx, dim in zip(binding_idxs[var], tns.shape, strict=True):
                if idx is not None:
                    binding_shapes[idx] = dim

        bindings = bindings.copy()
        for var, tns_ftype in binding_ftypes.items():
            if var not in bindings:
                shape = tuple(binding_shapes.get(idx, 1) for idx in binding_idxs[var])
                bindings[var] = tns_ftype(shape)

        args = list(bindings.values())

        res = mod.main(*args)

        if isinstance(prgm, lgc.LogicExpression):
            return TableValue(res[0], prgm.fields())
        return tuple(res)
