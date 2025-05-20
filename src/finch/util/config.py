import json
import os
import shutil
import sys
import sysconfig
import tomllib
from pathlib import Path

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

depot_dir = Path(os.getenv("FINCH_PATH", Path.home() / ".finch"))

is_windows = os.name == "nt"
is_apple = sys.platform == "darwin"

default_config = {
    "FINCH_CACHE_PATH": str(depot_dir / "cache"),
    "FINCH_CACHE_SIZE": 10_000,
    "FINCH_CACHE_ENABLE": True,
    "FINCH_TMP": str(depot_dir / "tmp"),
    "FINCH_LOG_PATH": str(depot_dir / "log.txt"),
    "FINCH_CC": (
        os.getenv("CC")
        or sysconfig.get_config_var("CC")
        or str(shutil.which("gcc") or "cl" if is_windows else "cc")
    ),
    "FINCH_CFLAGS": os.getenv(
        "CFLAGS",
        [
            "-shared",
            "-fPIC",
            "-O3",
        ],
    ),
    "FINCH_SHLIB_SUFFIX": (
        sysconfig.get_config_var("SHLIB_SUFFIX") or (".dll" if is_windows else ".so")
    ),
}

depot_dir.mkdir(parents=True, exist_ok=True)

config_path = depot_dir / "config.json"
if not config_path.exists():
    with config_path.open("w") as f:
        json.dump(default_config, f)

with config_path.open("r") as f:
    custom_config = json.load(f)


def get_config(var):
    """
    Get the configuration value for a given variable.
    """
    val = os.getenv(var)
    if val is not None:
        try:
            return json.loads(val)
        except json.decoder.JSONDecodeError:
            raise ValueError(
                f"Environment variable {var} is not a valid JSON value."
            ) from None
    else:
        return custom_config.get(var, default_config[var])


def set_config(var, val):
    """
    Set the configuration value for a given variable.
    """
    custom_config[var] = val
    with config_path.open("w") as f:
        json.dump(custom_config, f)


def reset_config():
    """
    Reset the configuration to the default values.
    """
    global custom_config
    custom_config = default_config.copy()
    with config_path.open("w") as f:
        json.dump(custom_config, f)


def get_version():
    """
    Get the version of Finch.
    """
    pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found.")

    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    try:
        return pyproject_data["project"]["version"]
    except KeyError:
        raise ValueError("Version not found in pyproject.toml.") from None
