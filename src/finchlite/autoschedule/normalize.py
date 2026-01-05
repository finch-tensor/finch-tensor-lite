from typing import TypeVar, overload

from finchlite.algebra.tensor import Tensor
from finchlite.finch_logic.nodes import LogicExpression, LogicStatement

from ..finch_logic import (
    Alias,
    Field,
    LogicEvaluator,
    LogicNode,
    TableValue,
)
from ..symbolic import Namespace, PostWalk, Rewrite

T = TypeVar("T")


@overload
def normalize_names(
    prgm: LogicStatement, bindings: dict[Alias, T]
) -> tuple[LogicStatement, dict[Alias, T]]: ...
@overload
def normalize_names(
    prgm: LogicExpression, bindings: dict[Alias, T]
) -> tuple[LogicExpression, dict[Alias, T]]: ...
@overload
def normalize_names(
    prgm: LogicNode, bindings: dict[Alias, T]
) -> tuple[LogicNode, dict[Alias, T]]: ...
def normalize_names(
    prgm: LogicNode, bindings: dict[Alias, T]
) -> tuple[LogicNode, dict[Alias, T]]:
    """
    Normalizes names of aliases and fields in the logic program to avoid conflicts.
    """
    if bindings is None:
        bindings = {}
    spc = Namespace()
    renames: dict[str, str] = {}

    def rule_0(node: LogicNode) -> LogicNode | None:
        match node:
            case Alias(name):
                if name in renames:
                    return Alias(renames[name])
                new_name = spc.freshen("A")
                renames[name] = new_name
                return Alias(new_name)
            case Field(name):
                if name in renames:
                    return Field(renames[name])
                new_name = spc.freshen("i")
                renames[name] = new_name
                return Field(new_name)
            case _:
                return None

    bindings = {Rewrite(rule_0)(var): tns for var, tns in bindings.items()}
    root = Rewrite(PostWalk(rule_0))(prgm)

    return root, bindings


class LogicNormalizer(LogicEvaluator):
    def __init__(self, ctx: LogicEvaluator):
        self.ctx: LogicEvaluator = ctx

    def __call__(
        self, prgm: LogicNode, bindings: dict[Alias, Tensor] | None = None
    ) -> TableValue | tuple[Tensor, ...]:
        root, bindings = normalize_names(prgm, bindings or {})
        return self.ctx(root, bindings)
