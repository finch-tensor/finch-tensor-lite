"""
Finch logging module.

This file contains all available filters for customizing your Finch program.
The `compilation_stage` log field is designed in a hierarchical way, to allow
quick filtering of generated IRs. Each segment consists of sub-segments which
build a path to a log record.

```
──> root
    ├──> logic
         ├──> pre-opt
         ├──> TODO: Add logging after each opt stage
         ├──> post-opt
    ├──> notation
    ├──> assembly
    └──> codegen
         ├──> c-backend
         └──> numba-backend
```

For example `root.logic` will print two Logic IRs - pre and post scheduler
optimizations. `root.codegen.numba-backend` will provide only Numba
generated kernel.

Fiters can be combined with `,` when passing to `get_logger_handler`.

Instead of full names, you can use abbrevations, listed in `_abbr_mapping`
dictionary. We also encourage to use Finch's custom logging `FORMAT` for
your logger setup.

Here's an example of logger setup:

```py
import logging
from finchlite.util.logging import get_logger_handler, FORMAT

handler = get_logger_handler(filter_pattern="r.l.pre,r.c.nb")
logging.basicConfig(level=logging.DEBUG, handlers=[handler], format=FORMAT)

# ...Finch code...
```

"""

from logging import Filter, StreamHandler


def _make_dict(val: str):
    return {"compilation_stage": val}


LOG_LOGIC_PRE_OPT = _make_dict("root.logic.pre-opt")
LOG_LOGIC_POST_OPT = _make_dict("root.logic.post-opt")
LOG_NOTATION = _make_dict("root.notation")
LOG_ASSEMBLY = _make_dict("root.assembly")
LOG_BACKEND_C = _make_dict("root.codegen.c-backend")
LOG_BACKEND_NUMBA = _make_dict("root.codegen.numba-backend")

FORMAT = "\x1b[7;30;42m%(asctime)s [%(compilation_stage)s]:\x1b[0m\n\n%(message)s"

_abbr_mapping = {
    "r": "root",
    "l": "logic",
    "n": "notation",
    "a": "assembly",
    "c": "codegen",
    "cb": "c-backend",
    "nb": "numba-backend",
    "pre": "pre-opt",
    "pos": "post-opt",
    "post": "post-opt",
}


def get_logger_handler(filter_pattern: str = "root") -> StreamHandler:
    handler = StreamHandler()
    patterns = filter_pattern.split(",")
    patterns = [
        ".".join([_abbr_mapping.get(s, s) for s in p.split(".")]) for p in patterns
    ]

    class _FinchFilter(Filter):
        def filter(self, record):
            if record.name.startswith("finchlite"):
                return any(record.compilation_stage.startswith(p) for p in patterns)
            return False

    handler.addFilter(_FinchFilter())
    return handler
