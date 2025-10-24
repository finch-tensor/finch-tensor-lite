from abc import abstractmethod

from ..finch_assembly import (
    AssemblyNode,
    Assert,
    Assign,
    Call,
    Literal,
    TaggedVariable,
    Variable,
    assembly_build_cfg,
)
from ..symbolic import DataFlowAnalysis
from ..util import qual_str


def assembly_copy_propagation(node: AssemblyNode):
    ctx = AssemblyCopyPropagation(assembly_build_cfg(node))
    ctx.analyze()
    return ctx


class AbstractAssemblyDataflow(DataFlowAnalysis):
    def stmt_str(self, stmt, state: dict) -> str:
        """Annotate a statement with lattice values."""
        match stmt:
            case Assign(lhs, rhs):
                rhs_str = self._expr_str(rhs, state)
                return f"{str(lhs)} = {rhs_str}"
            case Assert(exp):
                exp_str = self._expr_str(exp, state)
                return f"assert({exp_str})"
            case _:
                return str(stmt)

    def _expr_str(self, expr, state: dict) -> str:
        """Annotate an expression with lattice values."""
        match expr:
            case Literal(value):
                return qual_str(value)
            case TaggedVariable(Variable(name, _), id):
                var_str = f"{name}_{id}"
                if name in state:
                    return f"{var_str}: {self.print_lattice_value(name, state)}"
                return var_str
            case Variable(name, _):
                if name in state:
                    return f"{name}: {self.print_lattice_value(name, state)}"
                return str(name)
            case Call(Literal(_) as lit, args):
                annotated_args = [self._expr_str(arg, state) for arg in args]
                return f"{qual_str(lit.val)}({', '.join(annotated_args)})"
            case _:
                return str(expr)

    @abstractmethod
    def print_lattice_value(self, name, state: dict) -> str:
        """Format the lattice value associated with a variable for annotation."""
        ...


class AssemblyCopyPropagation(AbstractAssemblyDataflow):
    def direction(self) -> str:
        """
        Copy propagation is a forward analysis.
        """
        return "forward"

    def transfer(self, stmts, state: dict) -> dict:
        new_state = state.copy()

        for stmt in stmts:
            match stmt:
                case Assign(lhs, rhs):
                    # Get the variable name being assigned to
                    var_name = self._get_variable_name(lhs)

                    if var_name is not None:
                        new_state[var_name] = rhs

                        # invalidate any copies that point
                        # to the variable being assigned
                        to_remove = []
                        for var, val in new_state.items():
                            if val == lhs:
                                to_remove.append(var)

                        for var in to_remove:
                            new_state.pop(var)

        return new_state

    def join(self, state_1: dict, state_2: dict) -> dict:
        result = {}

        # only keep copy relationships that exist in both states with the same value
        for var_name in state_1:
            if var_name in state_2 and self._values_equal(
                state_1[var_name], state_2[var_name]
            ):
                result[var_name] = state_1[var_name]

        return result

    def print_lattice_value(self, name, state: dict) -> str:
        visited: set[str] = set()
        value = state.get(name)

        # iterate through states until we reach a Literal or a variable
        # without a mapping (root value)
        while True:
            match value:
                case Literal(v):
                    return qual_str(v)
                case TaggedVariable(Variable(next_name, _), _) | Variable(next_name, _):
                    # lattice was was already visited -> return
                    if next_name in visited:
                        return str(value)

                    visited.add(next_name)
                    next_val = state.get(next_name)

                    # reached root value -> return
                    if next_val is None:
                        return str(value)

                    value = next_val
                case _:
                    return str(value)

    def _get_variable_name(self, var) -> str | None:
        match var:
            case Variable(name, _):
                return name
            case TaggedVariable(Variable(name, _), _):
                return name
            case _:
                return None

    def _values_equal(self, val1, val2) -> bool:
        """Check if two values are equal."""
        if isinstance(val1, (Variable, TaggedVariable)) and isinstance(
            val2, (Variable, TaggedVariable)
        ):
            return val1 == val2

        if isinstance(val1, Literal) and isinstance(val2, Literal):
            return val1.val == val2.val

        return False
