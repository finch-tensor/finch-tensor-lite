import os
import shutil
import sys
import sysconfig
from importlib.metadata import version
from pathlib import Path

import donfig  # type: ignore[import]

"""
Finch Configuration Module

This module manages configuration settings for the Finch application.
Finch stores its settings and data in the `FINCH_PATH` directory, which
defaults to `~/.finch` but can be customized using the `FINCH_PATH`
environment variable.

Configuration details:
- Settings are stored in a `config.json` file within the `FINCH_PATH` directory.
- Values can be set via environment variables, the `config.json` file,
    or the `set_config` function.
- Configuration values are loaded automatically when the module is imported
    and can be accessed using the `get_config` function.

Use this module to easily manage and retrieve Finch-specific settings.
"""

is_windows = os.name == "nt"
is_apple = sys.platform == "darwin"

default = {
    "data_path": str(Path(sysconfig.get_path("data")) / "finch"),
    "cache_size": 10_000,
    "cache_enable": True,
    "cc": str(os.getenv("CC") or shutil.which("gcc") or ("cl" if is_windows else "cc")),
    "cflags": os.getenv("CFLAGS") or "-O3",
    "shared_cflags": os.getenv(
        "SHARED_CFLAGS",
        "-shared -fPIC",
    ),
    "shared_library_suffix": (
        os.getenv("SHARED_LIBRARY_SUFFIX", (".dll" if is_windows else ".so"))
    ),
}

config = donfig.Config("finch", defaults=[default])


def get_version():
    """
    Get the version of Finch.
    """

    return version("finch-tensor-lite")
