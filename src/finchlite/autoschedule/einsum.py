from typing import Any

import finchlite.finch_einsum as ein
import finchlite.finch_logic as lgc
from finchlite.algebra import init_value, overwrite


class EinsumLowerer:
    def __call__(self, prgm: lgc.Plan) -> tuple[ein.Plan, dict[str, Any]]:
        bindings: dict[str, Any] = {}
        definitions: dict[str, ein.Einsum] = {}
        return self.compile_plan(prgm, bindings, definitions), bindings

    def compile_plan(
        self,
        node: lgc.LogicNode,
        bindings: dict[str, Any],
        definitions: dict[str, ein.Einsum],
    ) -> ein.EinsumNode:
        match node:
            case lgc.Plan(bodies):
                ein_bodies = [self.compile_plan(body, bindings, definitions) for body in bodies]
                not_none_bodies = [body for body in ein_bodies if body is not None]
                return ein.Plan(tuple(not_none_bodies))
            case lgc.Query(lgc.Alias(name), lgc.Table(lgc.Literal(val), _)):
                bindings[name] = val
                return None
            case lgc.Query(
                lgc.Alias(name),
                lgc.Aggregate(lgc.Literal(operation), lgc.Literal(init), arg, _),
            ):
                einidxs = tuple(ein.Index(field.name) for field in node.rhs.fields)
                my_bodies = []
                if init != init_value(operation, type(init)):
                    my_bodies.append(
                        ein.Einsum(
                            op=ein.Literal(overwrite),
                            tns=ein.Alias(name),
                            idxs=einidxs,
                            arg=ein.Literal(init),
                        )
                    )
                my_bodies.append(
                    ein.Einsum(
                        op=ein.Literal(operation),
                        tns=ein.Alias(name),
                        idxs=einidxs,
                        arg=self.compile_operand(arg),
                    )
                )
                return ein.Plan(tuple(my_bodies))
            case lgc.Query(lgc.Alias(name), rhs):
                einarg = self.compile_operand(rhs)
                return ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias(name),
                    idxs=tuple(
                        ein.Index(field.name) for field in node.rhs.fields
                    ),
                    arg=einarg,
                )
                
            case lgc.Produces(args):
                returnValues = []
                for ret_arg in args:
                    if not isinstance(ret_arg, lgc.Alias):
                        raise Exception(f"Unrecognized logic: {ret_arg}")
                    returnValues.append(ein.Alias(ret_arg.name))

                return ein.Produces(tuple(returnValues))
            case _:
                raise Exception(f"Unrecognized logic: {node}")

    # lowers nested mapjoin logic IR nodes into a single pointwise expression
    def compile_operand(
        self,
        ex: lgc.LogicNode,
    ) -> ein.EinsumExpr:
        match ex:
            case lgc.Reformat(_, rhs):
                return self.compile_operand(rhs)
            case lgc.Reorder(arg, idxs):
                return self.compile_operand(arg)
            case lgc.MapJoin(lgc.Literal(operation), lgcargs):
                args = tuple([self.compile_operand(arg) for arg in lgcargs])
                return ein.Call(ein.Literal(operation), args)
            case lgc.Relabel(
                lgc.Alias(name), idxs
            ):  # relable is really just a glorified pointwise access
                return ein.Access(
                    tns=ein.Alias(name),
                    idxs=tuple(ein.Index(idx.name) for idx in idxs),
                )
            case lgc.Literal(value):
                return ein.Literal(val=value)
            case _:
                raise Exception(f"Unrecognized logic: {ex}")
