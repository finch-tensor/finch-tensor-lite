from collections.abc import Callable
from typing import Any

import numpy as np

import finchlite.finch_einsum as ein
from finchlite.algebra import init_value, is_commutative, overwrite
from finchlite.algebra.tensor import Tensor
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Literal,
    LogicNode,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Table,
)
from finchlite.interface import Scalar

from finchlite.symbolic import gensym

class EinsumLowerer:
    def __call__(self, prgm: Plan) -> tuple[ein.Plan, dict[str, Any]]:
        parameters: dict[str, Any] = {}
        definitions: dict[str, ein.Einsum] = {}
        return self.compile_plan(prgm, parameters, definitions), parameters

    def get_next_alias(self) -> ein.Alias:
        return ein.Alias(gensym("einsum"))

    def rename_einsum(
        self,
        einsum: ein.Einsum,
        new_alias: ein.Alias,
        definitions: dict[str, ein.Einsum],
    ) -> ein.Einsum:
        definitions[new_alias.name] = einsum
        return ein.Einsum(einsum.op, new_alias, einsum.idxs, einsum.arg)

    def reorder_einsum(
        self, einsum: ein.Einsum, idxs: tuple[ein.Index, ...]
    ) -> ein.Einsum:
        return ein.Einsum(einsum.op, einsum.tns, idxs, einsum.arg)

    def compile_plan(
        self, plan: Plan, parameters: dict[str, Any], definitions: dict[str, ein.Einsum]
    ) -> ein.Plan:
        einsums: list[ein.Einsum] = []
        returnValue: list[ein.EinsumExpr] = []

        for body in plan.bodies:
            match body:
                case Plan(_):
                    inner_plan = self.compile_plan(body, parameters, definitions)
                    einsums.extend(inner_plan.bodies)
                    returnValue.extend(inner_plan.returnValues)
                    break
                case Query(Alias(name), Table(Literal(val), _)) if isinstance(
                    val, Scalar
                ):
                    parameters[name] = val.val
                case Query(Alias(name), Table(Literal(tns), _)) if isinstance(
                    tns, Tensor
                ):
                    parameters[name] = np.asarray(tns)
                case Query(Alias(name), rhs):
                    einsums.append(
                        self.rename_einsum(
                            self.lower_to_einsum(rhs, einsums, parameters, definitions),
                            ein.Alias(name),
                            definitions,
                        )
                    )
                case Produces(args):
                    returnValue = []
                    for arg in args:
                        if isinstance(arg, Alias):
                            returnValue.append(ein.Alias(arg.name))
                        else:
                            einsum = self.rename_einsum(
                                self.lower_to_einsum(arg, einsums, parameters, definitions),
                                self.get_next_alias(),
                                definitions,
                            )
                            einsums.append(einsum)
                            returnValue.append(einsum.tns)
                    break
                case _:
                    einsums.append(
                        self.rename_einsum(
                            self.lower_to_einsum(
                                body, einsums, parameters, definitions
                            ),
                            self.get_next_alias(),
                            definitions,
                        )
                    )

        return ein.Plan(tuple(einsums), tuple(returnValue))

    def lower_to_einsum(
        self,
        ex: LogicNode,
        einsums: list[ein.Einsum],
        parameters: dict[str, Any],
        definitions: dict[str, ein.Einsum],
    ) -> ein.Einsum:
        match ex:
            case Plan(_):
                raise Exception("Plans within plans are not supported.")
            case MapJoin(Literal(operation), args):
                args_list = [
                    self.lower_to_pointwise(arg, einsums, parameters, definitions)
                    for arg in args
                ]
                pointwise_expr = self.lower_to_pointwise_op(operation, tuple(args_list))
                return ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=self.get_next_alias(),
                    idxs=tuple(ein.Index(field.name) for field in ex.fields),
                    arg=pointwise_expr,
                )
            case Reorder(arg, idxs):
                return self.reorder_einsum(
                    self.lower_to_einsum(arg, einsums, parameters, definitions),
                    tuple(ein.Index(field.name) for field in idxs),
                )
            case Aggregate(Literal(operation), Literal(init), arg, idxs):
                if init != init_value(operation, type(init)):
                    raise Exception(f"""
                    Init value {init} is not the default value
                    for operation {operation} of type {type(init)}.
                    Non standard init values are not supported.
                    """)
                aggregate_expr = self.lower_to_pointwise(
                    arg, einsums, parameters, definitions
                )
                return ein.Einsum(
                    op=ein.Literal(operation),
                    tns=self.get_next_alias(),
                    idxs=tuple(ein.Index(field.name) for field in ex.fields),
                    arg=aggregate_expr,
                )
            case _:
                raise Exception(f"Unrecognized logic: {ex}")

    def lower_to_pointwise_op(
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
    def lower_to_pointwise(
        self,
        ex: LogicNode,
        einsums: list[ein.Einsum],
        parameters: dict[str, Any],
        definitions: dict[str, ein.Einsum],
    ) -> ein.EinsumExpr:
        match ex:
            case Reorder(arg, idxs):
                return self.lower_to_pointwise(arg, einsums, parameters, definitions)
            case MapJoin(Literal(operation), args):
                args_list = [
                    self.lower_to_pointwise(arg, einsums, parameters, definitions)
                    for arg in args
                ]
                return self.lower_to_pointwise_op(operation, tuple(args_list))
            case Relabel(
                Alias(name), idxs
            ):  # relable is really just a glorified pointwise access
                return ein.Access(
                    tns=ein.Alias(name),
                    idxs=tuple(ein.Index(idx.name) for idx in idxs),
                )
            case Literal(value):
                return ein.Literal(val=value)
            case Aggregate(
                _, _, _, _
            ):  # aggregate has to be computed seperatley as it's own einsum
                aggregate_einsum_alias = self.get_next_alias()
                einsums.append(
                    self.rename_einsum(
                        self.lower_to_einsum(ex, einsums, parameters, definitions),
                        aggregate_einsum_alias,
                        definitions,
                    )
                )
                return ein.Access(
                    tns=aggregate_einsum_alias,
                    idxs=tuple(ein.Index(field.name) for field in ex.fields),
                )
            case _:
                raise Exception(f"Unrecognized logic: {ex}")
