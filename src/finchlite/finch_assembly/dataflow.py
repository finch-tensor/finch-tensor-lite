import copy
from abc import abstractmethod

from ..symbolic import DataFlowAnalysis, Namespace, PostOrderDFS, PostWalk, Rewrite
from .cfg_builder import (
    NumberedStatement,
    assembly_build_cfg,
    assembly_desugar,
    assembly_number_statements,
)
from .nodes import (
    AssemblyNode,
    Assign,
    Variable,
)


def assembly_dataflow_preprocess(node: AssemblyNode) -> AssemblyNode:
    """
    Preprocess a FinchAssembly node for dataflow analysis (desugar + number statements).
    Args:
        node: Root FinchAssembly node to preprocess.
    Returns:
        AssemblyNode: The preprocessed FinchAssembly node.
    """
    namespace = Namespace(node)
    return assembly_number_statements(assembly_desugar(node, namespace=namespace))


def assembly_dataflow_postprocess(node: AssemblyNode) -> AssemblyNode:
    """
    Postprocess a FinchAssembly node after dataflow analysis (remove numbering).
    Args:
        node: Root FinchAssembly node to postprocess.
    Returns:
        AssemblyNode: The postprocessed FinchAssembly node.
    """

    # TODO: implement resugaring
    # (removing function epilogue + empty Else branches in IfElse)

    # Remove numbering from numbered statements.
    def rw(x: AssemblyNode):
        match x:
            case NumberedStatement(stmt, _):
                return stmt

        return None

    return Rewrite(PostWalk(rw))(node)


def assembly_copy_propagation_debug(node: AssemblyNode):
    """
    Run copy-propagation on a FinchAssembly node (debug/testing only).
    Args:
        node: Root FinchAssembly node to analyze.
    Returns:
        AssemblyCopyPropagation: The completed analysis context.
    """
    ctx = AssemblyCopyPropagation(assembly_build_cfg(node))
    ctx.analyze()
    return ctx


def assembly_copy_propagation(node: AssemblyNode) -> AssemblyNode:
    """
    Apply copy-propagation to a FinchAssembly node.
    Args:
        node: Root FinchAssembly node to optimize.
    Returns:
        AssemblyNode: The optimized FinchAssembly node.
    """

    # Run copy-propagation analysis to collect replacements.
    ctx = AssemblyCopyPropagation(assembly_build_cfg(node))
    ctx.analyze()
    lattice: dict[tuple[int, str], str] = ctx.collect_copy_replacements()

    # Replace variables in a statement according to the collected replacements.
    def replace_vars(target: AssemblyNode, sid: int):
        def rw_var(n: AssemblyNode):
            match n:
                case Variable(name, vtype):
                    key = (sid, name)
                    if key in lattice:
                        return Variable(lattice[key], vtype)

            return None

        return Rewrite(PostWalk(rw_var))(target)

    # Rewrite each numbered statement to replace variables.
    def rw(x: AssemblyNode):
        match x:
            case NumberedStatement(stmt, sid):
                match stmt:
                    # if Assign, replace vars only on rhs to avoid replacing lhs
                    case Assign(lhs, rhs):
                        rhs = replace_vars(rhs, sid)
                        new_stmt = Assign(lhs, rhs)
                        return NumberedStatement(new_stmt, sid)
                    case _:
                        new_stmt = replace_vars(stmt, sid)
                        return NumberedStatement(new_stmt, sid)

        return None

    return Rewrite(PostWalk(rw))(node)


class AbstractAssemblyDataflow(DataFlowAnalysis):
    """Assembly-specific base for dataflow analyses."""

    def stmt_str(self, stmt, state: dict) -> str:
        """Annotate a statement with lattice values.

        Delegates expression traversal and collection of (name,value) pairs to
        ``get_lattice_value`` which now returns the annotation list directly.
        """
        annotations = self.get_lattice_value(state, stmt)
        if annotations:
            annostr = ", ".join(f"{name} = {str(val)}" for name, val in annotations)
            return f"{stmt} \t# {annostr}"
        return str(stmt)

    @abstractmethod
    def get_lattice_value(self, state, stmt) -> list[tuple[str, object]]:
        """Return list of (var_instance_name, lattice_value) pairs for a stmt/expr."""
        ...


class AssemblyCopyPropagation(AbstractAssemblyDataflow):
    """Copy propagation for FinchAssembly.

    Lattice:
    - defs: mapping ``{ var_name: sid | None }`` describing a unique reaching
        definition id for each variable (None means "not uniquely defined").
    - copies: mapping ``{ dst_var: (src_var, src_def_id) }`` describing a direct
        copy ``dst_var = src_var`` that is valid only if ``src_var`` still has the
        same unique reaching definition ``src_def_id``.
    """

    def direction(self) -> str:
        """Copy propagation is a forward analysis."""
        return "forward"

    def collect_copy_replacements(self) -> dict[tuple[int, str], str]:
        """Collect per-statement copy replacements.

        Returns:
            dict: Mapping ``(stmt_id, old_var_name) -> new_var_name`` indicating
                where ``old_var_name`` can be replaced by ``new_var_name`` at
                statement ``stmt_id`` based on copy-propagation facts.
        """
        replacements: dict[tuple[int, str], str] = {}

        for block in self.cfg.blocks.values():
            input_state = self.input_states.get(block.id, {})
            state = copy.deepcopy(input_state)

            for stmt in block.statements:
                sid = getattr(stmt, "sid", None)
                if sid is not None:
                    for name, value in self.get_lattice_value(state, stmt):
                        if isinstance(value, tuple) and value:
                            replacements[(sid, name)] = value[0]

                state = self.transfer([stmt], state)

        return replacements

    def get_lattice_value(self, state, stmt) -> list[tuple[str, object]]:
        """Collect lattice annotations for variables used in a stmt or expr."""
        annotated: list[tuple[str, object]] = []
        target = stmt

        if isinstance(target, NumberedStatement):
            target = target.stmt

        match target:
            case Assign(_, rhs):
                target = rhs

        copies = state.get("copies", {}) if isinstance(state, dict) else {}

        for node in PostOrderDFS(target):
            match node:
                case Variable(name, _):
                    if name in copies:
                        annotated.append((name, copies[name]))
                case _:
                    continue
        return annotated

    def transfer(self, stmts, state: dict) -> dict:
        """Transfer function over a sequence of statements.

        Applies copy-propagation effects of each statement in order, returning
        the updated lattice mapping. Only copies of variables are recorded; any
        existing mappings that point to a variable being reassigned are
        invalidated first.

        Args:
            stmts: Iterable of Assembly statements in a basic block.
            state: Incoming lattice mapping.

        Returns:
            dict: The outgoing lattice mapping after processing ``stmts``.
        """
        state = self._normalize_state(state)
        new_state = {"defs": state["defs"].copy(), "copies": state["copies"].copy()}

        defs: dict[str, int | None] = new_state["defs"]
        copies: dict[str, tuple[str, int | None]] = new_state["copies"]

        for wrapped in stmts:
            sid, stmt = self._unpack_stmt(wrapped)
            match stmt:
                case Assign(Variable(lhs_name, _), rhs):
                    # Any assignment kills previous copy info involving lhs.
                    copies.pop(lhs_name, None)

                    # If some other variable was known to be a copy of lhs, kill it
                    # because lhs's value changed.
                    to_remove = [
                        dst for dst, (src, _) in copies.items() if src == lhs_name
                    ]
                    for dst in to_remove:
                        copies.pop(dst, None)

                    # Update reaching definition for lhs.
                    defs[lhs_name] = sid

                    # If rhs is a variable with a unique reaching def, record a copy.
                    if isinstance(rhs, Variable):
                        rhs_name = rhs.name
                        rhs_def = defs.get(rhs_name)
                        if rhs_def is not None:
                            copies[lhs_name] = (rhs_name, rhs_def)

                    # Ensure all recorded copies remain consistent with current defs.
                    new_state["copies"] = self._prune_inconsistent_copies(defs, copies)
                    copies = new_state["copies"]
                case _:
                    continue

        return new_state

    def join(self, state_1: dict, state_2: dict) -> dict:
        """Meet operator for must copy-propagation.

        - defs join: keep a def id only if both agree, else None.
        - copies join: keep a copy only if both agree exactly, and it remains
          consistent with the joined defs.
        """

        s1 = self._normalize_state(state_1)
        s2 = self._normalize_state(state_2)

        defs_1: dict[str, int | None] = s1["defs"]
        defs_2: dict[str, int | None] = s2["defs"]
        copies_1: dict[str, tuple[str, int | None]] = s1["copies"]
        copies_2: dict[str, tuple[str, int | None]] = s2["copies"]

        joined_defs: dict[str, int | None] = {}
        for name in set(defs_1) | set(defs_2):
            v1 = defs_1.get(name)
            v2 = defs_2.get(name)
            joined_defs[name] = v1 if v1 == v2 else None

        joined_copies: dict[str, tuple[str, int | None]] = {
            dst: val
            for dst, val in copies_1.items()
            if dst in copies_2 and copies_2[dst] == val
        }

        joined_copies = self._prune_inconsistent_copies(joined_defs, joined_copies)

        return {"defs": joined_defs, "copies": joined_copies}

    def _normalize_state(self, state: dict) -> dict:
        if not state:
            return {"defs": {}, "copies": {}}

        if "defs" not in state or "copies" not in state:
            # allow old/empty shapes; upgrade in place
            return {"defs": state.get("defs", {}), "copies": state.get("copies", {})}

        return state

    def _unpack_stmt(self, stmt):
        if isinstance(stmt, NumberedStatement):
            return stmt.sid, stmt.stmt
        return None, stmt

    def _prune_inconsistent_copies(self, defs: dict, copies: dict) -> dict:
        pruned: dict[str, tuple[str, int | None]] = {}
        for dst, (src, src_def) in copies.items():
            if src_def is None:
                continue
            if defs.get(src) != src_def:
                continue
            pruned[dst] = (src, src_def)
        return pruned
