from functools import cache

from .. import finch_logic as lgc
from ..finch_logic import (
    LogicEvaluator,
    LogicLoader,
    LogicNode,
)
from ..symbolic import Namespace, PostWalk, Rewrite, ftype
from .compiler import LogicCompiler


def extract_tables(
    root: LogicNode,
    bindings: dict[lgc.Alias, lgc.TableValue],
) -> tuple[lgc.LogicNode, dict[lgc.Alias, lgc.TableValue]]:
    """
    Extracts tables from logic plan, replacing them with aliases.
    """
    bindings = bindings.copy()
    ids: dict[int, lgc.Alias] = {id(val.tns): key for key, val in bindings.items()}
    spc = Namespace(root)
    for alias in bindings:
        spc.freshen(alias.name)

    def rule_0(node):
        match node:
            case lgc.Table(tns, idxs):
                if not isinstance(tns, lgc.Literal):
                    raise ValueError(f"Table tns must be Literal, got {tns}")
                tns = tns.val
                if id(tns) in ids:
                    var = ids[id(tns)]
                    if bindings[var].idxs == idxs:
                        return var
                    return lgc.Relabel(var, idxs)
                var = lgc.Alias(spc.freshen("A"))
                ids[id(tns)] = var
                bindings[var] = lgc.TableValue(tns, idxs)
                return var

    root = Rewrite(PostWalk(rule_0))(root)
    return root, bindings


class LogicFieldsContext:
    def __init__(self, bindings: dict[lgc.Alias, tuple[lgc.Field, ...]] = None):
        if bindings is None:
            bindings = {}
        self.bindings = bindings

    def __call__(
        self, node: lgc.LogicNode
    ) -> tuple[lgc.Field, ...] | tuple[tuple[lgc.Field, ...], ...]:
        match node:
            case lgc.Alias(_):
                if node not in self.bindings:
                    raise ValueError(f"undefined tensor alias {node}")
                return self.bindings[node]
            case lgc.Table(_, idxs):
                return idxs
            case lgc.MapJoin(_, args):
                args_idxs = [self(a) for a in args]
                return tuple(dict.fromkeys([f for fs in args_idxs for f in fs]))
            case lgc.Aggregate(_, _, arg, idxs):
                arg_idxs = self(arg)
                return tuple(idx for idx in arg_idxs if idx not in idxs)
            case lgc.Relabel(arg, idxs):
                return idxs
            case lgc.Reorder(arg, idxs):
                return idxs
            case lgc.Subquery(lhs, arg):
                res = self.bindings.get(lhs)
                if res is None:
                    res = self(arg)
                    self.bindings[lhs] = res
                return res
            case lgc.Query(lhs, rhs):
                rhs = self(rhs)
                self.bindings[lhs] = rhs
                return (rhs,)
            case lgc.Plan(bodies):
                res = ()
                for body in bodies:
                    res = self(body)
                return res
            case lgc.Produces(args):
                return tuple(self(arg) for arg in args)
            case _:
                raise ValueError(f"Unknown expression type: {type(node)}")


@cache
def get_return_fields(
    prgm: lgc.LogicNode, bindings: dict[lgc.Alias, lgc.TableValueFType]
):
    ctx = LogicFieldsContext({var: tbl.idxs for var, tbl in bindings.items()})
    return ctx(prgm)


class LogicExecutor(LogicEvaluator):
    def __init__(self, ctx: LogicLoader | None = None, verbose: bool = False):
        if ctx is None:
            ctx = LogicCompiler()
        self.ctx: LogicLoader = ctx
        self.verbose: bool = verbose

    def __call__(self, prgm, bindings: dict[lgc.Alias, lgc.TableValue] | None = None):
        if bindings is None:
            bindings = {}
        prgm, bindings = extract_tables(prgm, bindings)
        binding_ftypes = {var: ftype(val) for var, val in bindings.items()}

        sizes = {
            idx: dim
            for tbl in bindings.values()
            for (idx, dim) in zip(tbl.idxs, tbl.tns.shape, strict=True)
        }

        mod, workspace_ftypes = self.ctx(prgm, binding_ftypes)

        inputs = [tbl.tns for tbl in bindings.values()]
        workspaces = [
            tbl.tns(sizes[idx] for idx in tbl.idxs) for tbl in workspace_ftypes
        ]

        res = mod.main(*inputs, *workspaces)

        res_ftype = get_return_fields(
            prgm, {var: tbl.idxs for var, tbl in bindings.items()}
        )

        if isinstance(res, tuple):
            return (t(tns) for t, tns in zip(res_ftype, res, strict=True))
        return res_ftype(res)
