from finchlite.finch_assembly import AssemblyKernel, AssemblyLibrary

from .. import finch_logic as lgc
from ..finch_logic import LogicInterpreter, LogicLoader
from ..symbolic import fisinstance


class LogicInterpreterKernel(AssemblyKernel):
    def __init__(self, prgm, bindings: dict[lgc.Alias, lgc.TableValueFType]):
        self.prgm = prgm
        self.bindings = bindings

    def __call__(self, *args):
        bindings = {
            var: lgc.TableValue(tns, self.bindings[var].idxs)
            for var, tns in zip(self.bindings.keys(), args, strict=True)
        }
        for key in bindings:
            assert fisinstance(bindings[key], self.bindings[key])
        ctx = LogicInterpreter()
        res = ctx(self.prgm, bindings)
        if isinstance(res, tuple):
            return tuple(tbl.tns for tbl in res)
        return res.tns


class LogicInterpreterLibrary(AssemblyLibrary):
    def __init__(self, prgm, bindings: dict[lgc.Alias, lgc.TableValueFType]):
        self.prgm = prgm
        self.bindings = bindings

    def __getattr__(self, name):
        if name == "main":
            return LogicInterpreterKernel(self.prgm, self.bindings)
        if name == "prgm":
            return self.prgm
        raise AttributeError(f"Unknown attribute {name} for InterpreterLibrary")


class FakeLogicCompiler(LogicLoader):
    def __init__(self):
        pass

    def __call__(
        self, prgm: lgc.LogicStatement, bindings: dict[lgc.Alias, lgc.TableValueFType]
    ) -> tuple[LogicInterpreterLibrary, dict[lgc.Alias, lgc.TableValueFType]]:
        return (LogicInterpreterLibrary(prgm, bindings), bindings)
