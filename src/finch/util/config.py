import os
import json

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

depot_dir = os.path.realpath(os.path.expanduser(os.getenv('FINCH_PATH', os.path.joinpath("~", ".finch"))))

default_config = {
    "FINCH_CACHE_PATH": os.path.join(depot_dir, "cache"),
    "FINCH_CACHE_SIZE": 10000,
    "FINCH_CACHE_ENABLE": True,
    "FINCH_TMP": os.path.join(depot_dir, "tmp"),
    "FINCH_LOG_PATH": os.path.join(depot_dir, "log.txt"),
    "FINCH_CC": "gcc",
    "FINCH_CFLAGS": ["-shared", "-fPIC", "-O3"],
}

if not os.path.exists(depot_dir):
    os.mkdir(depot_dir)

if not os.path.exists(os.path.joinpath(depot_dir, "config.json")):
    json.dump(default_config, open(os.path.joinpath(depot_dir, "config.json"), "w"))

custom_config = json.load(open(os.path.joinpath(depot_dir, "config.json"), "r"))

def get_config(var):
    """
    Get the configuration value for a given variable.
    """
    return os.getenv(var, custom_config.get(var, default_config[var]))

def set_config(var, val):
    """
    Get the configuration value for a given variable.
    """
    custom_config[var] = val
    json.dump(custom_config, open(os.path.joinpath(depot_dir, "config.json"), "w"))

def reset_config():
    """
    Reset the configuration to the default values.
    """
    global custom_config
    custom_config = default_config.copy()
    json.dump(custom_config, open(os.path.join(depot_dir, "config.json"), "w"))