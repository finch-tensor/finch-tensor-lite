from ..finch_logic import (
    Alias,
    Field,
    LogicEvaluator,
    LogicNode,
    TableValue,
)
from ..symbolic import Namespace, PostWalk, Rewrite


class LogicNormalizer(LogicEvaluator):
    def __init__(self, ctx: LogicEvaluator):
        self.ctx: LogicEvaluator = ctx

    def __call__(
        self, prgm: LogicNode, bindings: dict[Alias, TableValue] | None = None
    ) -> TableValue | tuple[TableValue, ...]:
        if bindings is None:
            bindings = {}
        spc = Namespace(prgm)
        for var in bindings:
            spc.freshen(var.name)
        renames: dict[str, str] = {}
        unrenames: dict[str, str] = {}

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
                    unrenames[new_name] = name
                    return Field(new_name)
                case _:
                    return None

        root = Rewrite(PostWalk(rule_0))(prgm)

        def reidx(tbl: TableValue, names):
            return TableValue(
                tbl.tns, tuple(Field(names[idx.name]) for idx in tbl.idxs)
            )

        bindings = {
            Rewrite(rule_0)(var): reidx(tbl, renames) for var, tbl in bindings.items()
        }
        res = self.ctx(root, bindings)

        if isinstance(res, tuple):
            return tuple(reidx(tbl, unrenames) for tbl in res)
        return reidx(res, unrenames)