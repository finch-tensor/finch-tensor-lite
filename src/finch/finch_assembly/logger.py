from collections.abc import Mapping, Sequence
from dataclasses import fields as dc_fields
from dataclasses import is_dataclass
from typing import Any


def debug(message: str, var: Any):
    """
    Print message and extract the expressions with their values of the input variable.

    Example:
        TODO

    Output:
        Extract the details of input variable and print out name and value of
        each member within the whole data structure

    """
    visit: set[int] = set()
    variables: list[dict[str, Any]] = []

    def _is_node(obj: Any) -> bool:
        if obj is None:
            return False

        # Cheap duck-typing for node-ish things
        if hasattr(obj, "children") or hasattr(obj, "_children"):
            return True

        # If it’s a custom class from this package, also treat it as a node
        mod = getattr(obj.__class__, "__module__", "")
        if "finch_assembly" in mod:
            return True

        # Dataclasses representing nodes
        if is_dataclass(obj):
            return True

        # A fallback: objects with __dict__ and at least one attribute that is node-like
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                if isinstance(v, (list | tuple | set)) and any(_is_node(x) for x in v):
                    return True
                if _is_node(v):
                    return True

        return False

    def _collect_childish(v: Any, out: list[Any]) -> None:
        if _is_node(v):
            out.append(v)
        elif isinstance(v, (list | tuple | set)):
            out.extend([x for x in v if _is_node(x)])
        elif isinstance(v, Mapping):
            out.extend([x for x in v.values() if _is_node(x)])

    def _iter_children(node: Any) -> list[Any]:
        kids: list[Any] = []

        # Common containers
        for attr in ("children", "_children"):
            if hasattr(node, attr):
                v = getattr(node, attr)
                if isinstance(v, Sequence) and not isinstance(
                    v, (str | bytes | bytearray)
                ):
                    kids.extend([x for x in v if _is_node(x)])
                elif _is_node(v):
                    kids.append(v)

        # Dataclass fields
        if is_dataclass(node):
            for f in dc_fields(node):
                v = getattr(node, f.name)
                _collect_childish(v, kids)

        # __dict__ fields
        if hasattr(node, "__dict__"):
            for k, v in vars(node).items():
                if k in {"children", "_children"}:
                    continue
                _collect_childish(v, kids)

        # De-duplicate while preserving order
        uniq = []
        seen_ids = set()
        for k in kids:
            if id(k) not in seen_ids:
                uniq.append(k)
                seen_ids.add(id(k))

        return uniq

    def _extract_name(node: Any) -> str:
        for key in ("name", "id", "symbol", "label"):
            if hasattr(node, key):
                n = getattr(node, key)
                if isinstance(n, str) and n:
                    return n

        return "None"

    def _extract_value(node: Any) -> str:
        for key in ("value", "val", "data", "tensor", "array"):
            if hasattr(node, key):
                v = getattr(node, key)
                if not _is_node(v):
                    return str(v)

        return "None"

    def _traverse(node: Any, path: str) -> dict[str, Any]:
        if id(node) in visit:
            return {"$ref": path}
        visit.add(id(node))

        node_type = type(node).__name__
        name = _extract_name(node)
        val = _extract_value(node)

        if name is not None and val is not None:
            variables.append(
                {
                    "node_type": node_type,
                    "name": name,
                    "value": val,
                }
            )

        children = _iter_children(node)
        return {
            "type": node_type,
            **({"name": name} if name is not None else {}),
            "children": [
                _traverse(ch, f"{path}.{i}:{type(ch).__name__}")
                for i, ch in enumerate(children)
            ],
        }

    tree = _traverse(var, "root")
    print(f"{message}" + " " + str(tree))
