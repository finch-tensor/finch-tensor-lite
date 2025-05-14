import ctypes
import os
import tempfile
import subprocess
from .config import get_config

def compile_to_callable(c_code, function_name, *args):
    """
    Compiles a C function into a shared library, loads it using ctypes, and calls the function.

    :param c_code: The C code as a string.
    :param function_name: The name of the function to call.
    :param args: The arguments to pass to the function.
    :return: The result of the function call.
    """
    # Create a temporary directory to store the C file and shared library
    with tempfile.TemporaryDirectory() as temp_dir:
        c_file_path = os.path.join(temp_dir, "temp.c")
        shared_lib_path = os.path.join(temp_dir, "libtemp.so")

        # Write the C code to a file
        with open(c_file_path, "w") as c_file:
            c_file.write(c_code)

        # Compile the C code into a shared library
        compile_command = [
            get_config("FINCH_CC"),
            *get_config("FINCH_CFLAGS"),
            "-o"
            shared_lib_path,
            c_file_path
        ]
        subprocess.run(compile_command, check=True)

        # Load the shared library using ctypes
        shared_lib = ctypes.CDLL(shared_lib_path)

        # Get the function from the shared library
        c_function = getattr(shared_lib, function_name)

        # Call the function with the provided arguments
        result = c_function(*args)

        return result

# Example usage
if __name__ == "__main__":
    c_code = """
    #include <stdio.h>

    int add(int a, int b) {
        return a + b;
    }
    """
    result = compile_and_run_c_function(c_code, "add", 5, 3)
    print("Result:", result)