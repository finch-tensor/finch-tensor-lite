import ctypes
import os
import tempfile
import subprocess
import sysconfig
from ..util.config import get_config
from ..util.cache import file_cache
from functools import lru_cache

@file_cache(ext=sysconfig.get_config_var('SHLIB_SUFFIX'), domain="c")
def create_shared_lib(filename, c_code, cc, cflags):
    """
    Compiles a C function into a shared library and returns the path.

    :param c_code: The C code as a string.
    :return: The result of the function call.
    """
    tmp_dir = get_config("FINCH_TMP")
    os.makedirs(tmp_dir, exist_ok=True)
    # Create a temporary directory to store the C file and shared library
    with tempfile.TemporaryDirectory(prefix=tmp_dir) as staging_dir:
        c_file_path = os.path.join(staging_dir, "temp.c")
        shared_lib_path = filename

        # Write the C code to a file
        with open(c_file_path, "w") as c_file:
            c_file.write(c_code)

        # Compile the C code into a shared library
        compile_command = [
            cc,
            *cflags,
            "-o",
            shared_lib_path,
            c_file_path
        ]
        subprocess.run(compile_command, check=True)
        assert os.path.exists(shared_lib_path), f"Compilation failed: {compile_command}"



@lru_cache(maxsize=10_000)
def get_c_function(function_name, c_code):
    """
    :param function_name: The name of the function to call.
    :param c_code: The code to compile
    """
    shared_lib_path = create_shared_lib(c_code, get_config("FINCH_CC"), get_config("FINCH_CFLAGS"))

    # Load the shared library using ctypes
    shared_lib = ctypes.CDLL(shared_lib_path)

    # Get the function from the shared library
    c_function = getattr(shared_lib, function_name)

    return c_function
