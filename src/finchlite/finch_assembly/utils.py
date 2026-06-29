from finchlite.symbolic import PostOrderDFS

from .nodes import AssemblyNode, Variable


def get_vars_in_expr(expr: AssemblyNode) -> dict[str, Variable]:
    vars = {}
    for node in PostOrderDFS(expr):
        match node:
            case Variable(name, _) as var:
                vars[name] = var
    return vars
