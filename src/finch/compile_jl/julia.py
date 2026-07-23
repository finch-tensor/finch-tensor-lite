from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_jc: Any | None = None
_jl: Any | None = None
_packages_loaded = False


def _julia_exe_from_libjulia(libjulia: str) -> Path | None:
    libpath = Path(libjulia)
    exe_name = "julia.exe" if os.name == "nt" else "julia"
    for bindir in (libpath.parent, libpath.parent.parent / "bin"):
        exe = bindir / exe_name
        if exe.is_file():
            return exe
    return None


def init_julia() -> tuple[Any, Any]:
    global _jc, _jl, _packages_loaded

    if _jl is None:
        os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
        import juliapkg

        libjulia = juliapkg.libjulia()
        if libjulia is not None and not os.path.exists(libjulia):
            juliapkg.resolve(force=True)
            libjulia = juliapkg.libjulia()

        julia_exe = _julia_exe_from_libjulia(libjulia)
        if julia_exe is not None:
            os.environ.setdefault("PYTHON_JULIACALL_EXE", str(julia_exe))
            os.environ.setdefault("PYTHON_JULIACALL_PROJECT", juliapkg.project())
            os.environ.setdefault("PYTHON_JULIACALL_LIB", libjulia)
            os.environ.setdefault("PYTHON_JULIACALL_BINDIR", str(julia_exe.parent))

        import juliacall as juliacall_module
        from juliacall import Main

        _jc = juliacall_module
        _jl = Main

    if not _packages_loaded:
        # To change the version of Finch used, see pyjuliapkg and
        # `juliapkg.json` in this package.
        for pkg in ("Finch", "HDF5", "NPZ", "TensorMarket", "Random", "Statistics"):
            _jl.seval(f"using {pkg}")
        _packages_loaded = True

    return _jc, _jl


def get_jc() -> Any:
    return init_julia()[0]


def get_jl() -> Any:
    return init_julia()[1]


class _LazyJuliaCall:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_jc(), name)


class _LazyJuliaMain:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_jl(), name)


jc = _LazyJuliaCall()
jl = _LazyJuliaMain()
