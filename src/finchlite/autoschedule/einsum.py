from collections.abc import Callable
from typing import Any

import finchlite.finch_einsum as ein
from finchlite.algebra import init_value, is_commutative, overwrite
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Literal,
    LogicNode,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reformat,
    Relabel,
    Reorder,
    Table,
)
from finchlite.symbolic import gensym


class EinsumLowerer:
    def __call__(self, prgm: Plan) -> tuple[ein.Plan, dict[str, Any]]:
        bindings: dict[str, Any] = {}
        definitions: dict[str, ein.Einsum] = {}
        return self.compile_plan(prgm, bindings, definitions), bindings

    def get_next_alias(self) -> ein.Alias:
        return ein.Alias(gensym("einsum"))

    def reorder_einsum(
        self, einsum: ein.Einsum, idxs: tuple[ein.Index, ...]
    ) -> ein.Einsum:
        return ein.Einsum(einsum.op, einsum.tns, idxs, einsum.arg)

    def compile_plan(
        self, plan: Plan, bindings: dict[str, Any], definitions: dict[str, ein.Einsum]
    ) -> ein.Plan:
        bodies: list[ein.EinsumNode] = []

        for body in plan.bodies:
            match body:
                case Plan(_):
                    bodies.append(self.compile_plan(body, bindings, definitions))
                case Query(Alias(name), Table(Literal(val), _)):
                    bindings[name] = val
                case Query(Alias(name), MapJoin(Literal(operation), args)) | Query(
                    Alias(name), Reformat(_, MapJoin(Literal(operation), args))
                ):
                    args_list = [
                        self.compile_operand(arg, bodies, bindings, definitions)
                        for arg in args
                    ]
                    bodies.append(
                        ein.Einsum(
                            op=ein.Literal(overwrite),
                            tns=ein.Alias(name),
                            idxs=tuple(
                                ein.Index(field.name) for field in body.rhs.fields
                            ),
                            arg=self.compile_expr(operation, tuple(args_list)),
                        )
                    )
                case Query(
                    Alias(name), Aggregate(Literal(operation), Literal(init), arg, _)
                ) | Query(
                    Alias(name),
                    Reformat(_, Aggregate(Literal(operation), Literal(init), arg, _)),
                ):
                    einidxs = tuple(ein.Index(field.name) for field in body.rhs.fields)
                    if init != init_value(operation, type(init)):
                        bodies.append(
                            ein.Einsum(
                                op=ein.Literal(overwrite),
                                tns=ein.Alias(name),
                                idxs=einidxs,
                                arg=ein.Literal(init),
                            )
                        )
                    bodies.append(
                        ein.Einsum(
                            op=ein.Literal(operation),
                            tns=ein.Alias(name),
                            idxs=einidxs,
                            arg=self.compile_operand(
                                arg, bodies, bindings, definitions
                            ),
                        )
                    )
                case Produces(args):
                    returnValues = []
                    for arg in args:
                        if not isinstance(arg, Alias):
                            raise Exception(f"Unrecognized logic: {arg}")
                        returnValues.append(ein.Alias(arg.name))

                    bodies.append(ein.Produces(tuple(returnValues)))
                case _:
                    raise Exception(f"Unrecognized logic: {body}")

        return ein.Plan(tuple(bodies))

    def compile_expr(
        self, operation: Callable, args: tuple[ein.EinsumExpr, ...]
    ) -> ein.EinsumExpr:
        # if operation is commutative, we simply pass
        # all the args to the pointwise op since
        # order of args does not matter
        if is_commutative(operation):

            def flatten_args(
                m_args: tuple[ein.EinsumExpr, ...],
            ) -> tuple[ein.EinsumExpr, ...]:
                ret_args: list[ein.EinsumExpr] = []
                for arg in m_args:
                    match arg:
                        case ein.Call(ein.Literal(op2), _) if op2 == operation:
                            ret_args.extend(flatten_args(arg.args))
                        case _:
                            ret_args.append(arg)
                return tuple(ret_args)

            return ein.Call(ein.Literal(operation), flatten_args(args))

        # combine args from left to right (i.e a / b / c -> (a / b) / c)
        return ein.Call(ein.Literal(operation), args)

    # lowers nested mapjoin logic IR nodes into a single pointwise expression
    def compile_operand(
        self,
        ex: LogicNode,
        bodies: list[ein.EinsumNode],
        bindings: dict[str, Any],
        definitions: dict[str, ein.Einsum],
    ) -> ein.EinsumExpr:
        match ex:
            case Reorder(arg, idxs):
                return self.compile_operand(arg, bodies, bindings, definitions)
            case MapJoin(Literal(operation), args):
                args_list = [
                    self.compile_operand(arg, bodies, bindings, definitions)
                    for arg in args
                ]
                return self.compile_expr(operation, tuple(args_list))
            case Relabel(
                Alias(name), idxs
            ):  # relable is really just a glorified pointwise access
                return ein.Access(
                    tns=ein.Alias(name),
                    idxs=tuple(ein.Index(idx.name) for idx in idxs),
                )
            case Literal(value):
                return ein.Literal(val=value)
            case Aggregate(Literal(operation), Literal(init), arg, _):
                alias = self.get_next_alias()
                remaining_idxs = tuple(ein.Index(field.name) for field in ex.fields)
                if init != init_value(operation, type(init)):
                    bodies.append(
                        ein.Einsum(
                            op=ein.Literal(overwrite),
                            tns=ein.Alias(alias),
                            idxs=remaining_idxs,
                            arg=ein.Literal(init),
                        )
                    )
                bodies.append(
                    ein.Einsum(
                        op=ein.Literal(operation),
                        tns=ein.Alias(alias),
                        idxs=remaining_idxs,
                        arg=self.compile_operand(arg, bodies, bindings, definitions),
                    )
                )
                return ein.Access(
                    tns=alias,
                    idxs=remaining_idxs,
                )
            case _:
                raise Exception(f"Unrecognized logic: {ex}")
