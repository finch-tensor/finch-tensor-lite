from collections.abc import Callable
from typing import Any

import finchlite.finch_einsum as ein
from finchlite.algebra import init_value, is_commutative, overwrite
import finchlite.finch_logic as lgc


class EinsumLowerer:
    def __call__(self, prgm: lgc.Plan) -> tuple[ein.Plan, dict[str, Any]]:
        bindings: dict[str, Any] = {}
        definitions: dict[str, ein.Einsum] = {}
        return self.compile_plan(prgm, bindings, definitions), bindings

    def compile_plan(
        self, plan: lgc.Plan, bindings: dict[str, Any], definitions: dict[str, ein.Einsum]
    ) -> ein.Plan:
        bodies: list[ein.EinsumNode] = []

        for body in plan.bodies:
            match body:
                case lgc.Plan(_):
                    bodies.append(self.compile_plan(body, bindings, definitions))
                case lgc.Query(lgc.Alias(name), lgc.Table(lgc.Literal(val), _)):
                    bindings[name] = val
                case lgc.Query(
                    lgc.Alias(name), lgc.Aggregate(lgc.Literal(operation), lgc.Literal(init), arg, _)
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
                            arg=self.compile_operand(arg),
                        )
                    )
                case lgc.Query(lgc.Alias(name), rhs):
                    einarg = self.compile_operand(rhs)
                    bodies.append(
                        ein.Einsum(
                            op=ein.Literal(overwrite),
                            tns=ein.Alias(name),
                            idxs=tuple(
                                ein.Index(field.name) for field in body.rhs.fields
                            ),
                            arg=einarg,
                        )
                    )
                case lgc.Produces(args):
                    returnValues = []
                    for ret_arg in args:
                        if not isinstance(ret_arg, lgc.Alias):
                            raise Exception(f"Unrecognized logic: {ret_arg}")
                        returnValues.append(ein.Alias(ret_arg.name))

                    bodies.append(ein.Produces(tuple(returnValues)))
                case _:
                    raise Exception(f"Unrecognized logic: {body}")

        return ein.Plan(tuple(bodies))

    # lowers nested mapjoin logic IR nodes into a single pointwise expression
    def compile_operand(
        self,
        ex: lgc.LogicNode,
    ) -> ein.EinsumExpr:
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
        
        match ex:
            case lgc.Reformat(_, rhs):
                return self.compile_operand(rhs)
            case lgc.Reorder(arg, idxs):
                return self.compile_operand(arg)
            case lgc.MapJoin(lgc.Literal(operation), args):
                args = tuple([
                    self.compile_operand(arg)
                    for arg in args
                ])
                return ein.Call(ein.Literal(operation), args 
                    if is_commutative(operation) else flatten_args(args))
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
