from typing import Any, overload

from finchlite.finch_assembly import AssemblyKernel, AssemblyLibrary
from finchlite.finch_logic.nodes import TableValue

from .. import finch_logic as lgc
from ..finch_logic import LogicEvaluator, LogicInterpreter, LogicLoader, LogicNode
from ..symbolic import Namespace, PostWalk, Rewrite, fisinstance, ftype


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
    def __init__(self, bindings: dict[lgc.Alias, tuple[lgc.Field, ...]] | None = None):
        if bindings is None:
            bindings = {}
        self.bindings: dict[lgc.Alias, tuple[lgc.Field, ...]] = bindings

    @overload
    def __call__(self, node: lgc.LogicExpression) -> tuple[lgc.Field, ...]: ...
    @overload
    def __call__(
        self, node: lgc.LogicStatement
    ) -> tuple[tuple[lgc.Field, ...], ...]: ...
    @overload
    def __call__(
        self, node: lgc.LogicNode
    ) -> tuple[lgc.Field, ...] | tuple[tuple[lgc.Field, ...], ...]: ...
    def __call__(self, node):
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
            case lgc.Relabel(_, idxs):
                return idxs
            case lgc.Reorder(_, idxs):
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


def get_return_fields(
    prgm: lgc.LogicNode, bindings: dict[lgc.Alias, tuple[lgc.Field, ...]]
):
    ctx = LogicFieldsContext(bindings)
    return ctx(prgm)


class LogicInterpreterKernel(AssemblyKernel):
    def __init__(self, prgm, bindings: dict[lgc.Alias, lgc.TableValueFType]):
        self.prgm = prgm
        self.bindings = bindings

    def __call__(self, *args):
        bindings = {
            var: lgc.TableValue(tns, self.bindings[var].idxs)
            for var, tns in zip(self.bindings.keys(), args, strict=True)
        }
        for key in bindings:
            assert fisinstance(bindings[key], self.bindings[key])
        ctx = LogicInterpreter()
        res = ctx(self.prgm, bindings)
        if isinstance(res, tuple):
            return tuple(tbl.tns for tbl in res)
        return res.tns


class LogicInterpreterLibrary(AssemblyLibrary):
    def __init__(self, prgm, bindings: dict[lgc.Alias, lgc.TableValueFType]):
        self.prgm = prgm
        self.bindings = bindings

    def __getattr__(self, name):
        if name == "main":
            return LogicInterpreterKernel(self.prgm, self.bindings)
        if name == "prgm":
            return self.prgm
        raise AttributeError(f"Unknown attribute {name} for InterpreterLibrary")


class FakeLogicCompiler(LogicLoader):
    def __init__(self):
        pass

    def __call__(
        self, prgm: lgc.LogicStatement, bindings: dict[lgc.Alias, lgc.TableValueFType]
    ) -> tuple[LogicInterpreterLibrary, dict[lgc.Alias, lgc.TableValueFType]]:
        return (LogicInterpreterLibrary(prgm, bindings), bindings)


class ProvisionTensorsContext:
    def __init__(
        self,
        bindings: dict[lgc.Alias, lgc.TableValue],
        types: dict[lgc.Alias, lgc.TableValueFType],
    ):
        self.bindings: dict[lgc.Alias, lgc.TableValue] = bindings.copy()
        self.shapes: dict[lgc.Alias, tuple[Any, ...]] = {
            var: tbl.tns.shape for var, tbl in bindings.items()
        }
        self.fields: dict[lgc.Alias, tuple[lgc.Field, ...]] = {
            var: tbl.idxs for var, tbl in types.items()
        }
        self.types: dict[lgc.Alias, lgc.TableValueFType] = types

    def __call__(self, node: lgc.LogicStatement) -> dict[lgc.Alias, lgc.TableValue]:
        match node:
            case lgc.Plan(bodies):
                for body in bodies:
                    self(body)
            case lgc.Query(lhs, rhs):
                if lhs not in self.bindings:
                    shape = rhs.shape(self.shapes, self.fields)
                    tns = self.types[lhs].tns(shape)
                    self.bindings[lhs] = lgc.TableValue(tns, self.types[lhs].idxs)
                    self.shapes[lhs] = shape
            case lgc.Produces(_):
                pass
            case _:
                raise ValueError(f"Unknown LogicStatement: {type(node)}")
        return self.bindings


class LogicExecutor(LogicEvaluator):
    def __init__(self, ctx: LogicLoader | None = None, verbose: bool = False):
        if ctx is None:
            ctx = FakeLogicCompiler()
        self.ctx: LogicLoader = ctx
        self.verbose: bool = verbose

    def __call__(self, prgm, bindings: dict[lgc.Alias, lgc.TableValue] | None = None):
        if bindings is None:
            bindings = {}
        if isinstance(prgm, lgc.LogicExpression):
            prgm = lgc.Produces((prgm,))
        prgm, bindings = extract_tables(prgm, bindings)
        binding_ftypes = {var: ftype(val) for var, val in bindings.items()}

        mod, binding_ftypes = self.ctx(prgm, binding_ftypes)

        bindings = ProvisionTensorsContext(bindings, binding_ftypes)(prgm)
        args = [tbl.tns for tbl in bindings.values()]

        res = mod.main(*args)

        res_idxs = get_return_fields(
            prgm, {var: tbl.idxs for var, tbl in bindings.items()}
        )

        if isinstance(res, tuple):
            return tuple(
                TableValue(tns, idxs) for idxs, tns in zip(res_idxs, res, strict=True)
            )
        return TableValue(res, res_idxs)
