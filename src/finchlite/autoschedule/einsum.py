import finchlite.finch_einsum as ein
import finchlite.finch_logic as lgc
from finchlite.algebra import init_value, overwrite
from finchlite.algebra.tensor import TensorFType
from finchlite.finch_assembly.stages import AssemblyLibrary
from finchlite.finch_einsum import EinsumLoader, MockEinsumLoader
from finchlite.finch_logic import LogicStatement
from finchlite.finch_logic.stages import LogicLoader

from .stages import LogicEinsumLowerer


def generate_einsum_stmt(node: LogicStatement) -> ein.EinsumStatement:
    match node:
        case lgc.Plan(bodies):
            return ein.Plan(tuple(generate_einsum_stmt(body) for body in bodies))
        case lgc.Query(
            lgc.Alias(name),
            lgc.Aggregate(lgc.Literal(operation), lgc.Literal(init), arg, _) as agg,
        ):
            einidxs = tuple(ein.Index(field.name) for field in agg.fields())
            body = ein.Einsum(
                op=ein.Literal(operation),
                tns=ein.Alias(name),
                idxs=einidxs,
                arg=generate_einsum_expr(arg),
            )
            if operation == overwrite or init != init_value(operation, type(init)):
                return ein.Plan(
                    (
                        ein.Einsum(
                            op=ein.Literal(overwrite),
                            tns=ein.Alias(name),
                            idxs=einidxs,
                            arg=ein.Literal(init),
                        ),
                        body,
                    )
                )
            return body
        case lgc.Query(lgc.Alias(name), rhs):
            assert isinstance(rhs, lgc.LogicExpression)
            einarg = generate_einsum_expr(rhs)
            return ein.Einsum(
                op=ein.Literal(overwrite),
                tns=ein.Alias(name),
                idxs=tuple(ein.Index(field.name) for field in rhs.fields()),
                arg=einarg,
            )
        case lgc.Produces(args):
            return ein.Produces(tuple(ein.Alias(ret_arg.name) for ret_arg in args))
        case _:
            raise Exception(f"Unrecognized logic: {node}")


def generate_einsum_expr(
    ex: lgc.LogicExpression,
) -> ein.EinsumExpression:
    match ex:
        case lgc.Reorder(arg, idxs):
            return generate_einsum_expr(arg)
        case lgc.MapJoin(lgc.Literal(operation), lgcargs):
            args = tuple([generate_einsum_expr(arg) for arg in lgcargs])
            return ein.Call(ein.Literal(operation), args)
        case lgc.Table(lgc.Alias(name), idxs):
            return ein.Access(
                tns=ein.Alias(name),
                idxs=tuple(ein.Index(idx.name) for idx in idxs),
            )
        case lgc.Literal(value):
            return ein.Literal(val=value)
        case _:
            raise Exception(f"Unrecognized logic: {ex}")


class EinsumGenerator(LogicEinsumLowerer):
    def __call__(
        self, prgm: LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        bindings_2 = {ein.Alias(var.name): val for var, val in bindings.items()}
        return (generate_einsum_stmt(prgm), bindings_2)


class LogicEinsumLoader(LogicLoader):
    def __init__(
        self,
        ctx_lower: LogicEinsumLowerer | None = None,
        ctx_load: EinsumLoader | None = None,
    ):
        if ctx_lower is None:
            ctx_lower = EinsumGenerator()
        self.ctx_lower: LogicEinsumLowerer = ctx_lower
        if ctx_load is None:
            ctx_load = MockEinsumLoader()
        self.ctx_load: EinsumLoader = ctx_load

    def __call__(
        self, prgm: lgc.LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
    ]:
        ein_prgm, ein_bindings = self.ctx_lower(prgm, bindings)
        mod, ein_bindings, ein_shape_vars = self.ctx_load(ein_prgm, ein_bindings)
        lgc_bindings = {lgc.Alias(var.name): val for var, val in ein_bindings.items()}
        lgc_shape_vars: dict[lgc.Alias, tuple[lgc.Field | None, ...]] = {
            lgc.Alias(var.name): tuple(
                lgc.Field(idx.name) if idx is not None else None for idx in idxs
            )
            for var, idxs in ein_shape_vars.items()
        }
        return mod, lgc_bindings, lgc_shape_vars
