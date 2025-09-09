"""
This module contains the C and numba error handling code for FinchLite. Defines the
description of each error code. It also contains the code to throw errors in C and
check for them before the CKernel returns.
"""

error_codes = {
    1: "Index out of bounds",
    2: "Negative size allocation",
}

error_var = "__error_code__"
error_func_name = "get_ecode"


def c_throw(code: int) -> str:
    """
    Returns the string of C code to throw the given error code.
    """
    if code not in error_codes:
        raise ValueError(f"Error code {code} not defined.")
    return f"{error_var} = {code}; /* {error_codes[code]} */"


def check_error_code(module) -> int:
    return module[error_func_name]()


def numba_throw(code: int):
    """
    Returns the string of Python code to throw the given error code.
    """
    if code not in error_codes:
        raise ValueError(f"Error code {code} not defined.")
    return f'raise RuntimeError("Error {code}: {error_codes[code]}")'
