import ctypes
import shutil
import subprocess
import tempfile
from functools import lru_cache
from operator import methodcaller
from pathlib import Path

from ..util.cache import file_cache
from ..util.config import get_config


@file_cache(ext=get_config("FINCH_SHLIB_SUFFIX"), domain="c")
def create_shared_lib(filename, c_code, cc, cflags):
    """
    Compiles a C function into a shared library and returns the path.

    :param c_code: The C code as a string.
    :return: The result of the function call.
    """
    tmp_dir = Path(get_config("FINCH_TMP"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Create a temporary directory to store the C file and shared library
    with tempfile.TemporaryDirectory(prefix=str(tmp_dir)) as staging_dir:
        staging_dir = Path(staging_dir)
        c_file_path = staging_dir / "temp.c"
        shared_lib_path = Path(filename)

        # Write the C code to a file
        c_file_path.write_text(c_code)

        # Compile the C code into a shared library
        compile_command = [
            str(cc),
            *cflags,
            "-o",
            str(shared_lib_path),
            str(c_file_path),
        ]
        if not shutil.which(cc):
            raise FileNotFoundError(
                f"Compiler '{cc}' not found. Ensure it is installed and in your PATH."
            )
        subprocess.run(compile_command, check=True)
        assert shared_lib_path.exists(), f"Compilation failed: {compile_command}"


@lru_cache(maxsize=10_000)
def get_c_function(function_name, c_code):
    """
    :param function_name: The name of the function to call.
    :param c_code: The code to compile
    """
    shared_lib_path = create_shared_lib(
        c_code, get_config("FINCH_CC"), get_config("FINCH_CFLAGS")
    )

    # Load the shared library using ctypes
    shared_lib = ctypes.CDLL(str(shared_lib_path))

    # Get the function from the shared library
    return getattr(shared_lib, function_name)


class CBufferFormat(ABC):
    @abstractmethod
    def c_load(self, name, index_name, index_type):
        """
        Return C code which loads a named buffer at the given index.
        """

    @abstractmethod
    def c_store(self, name, value_name, value_type, index_name, index_type):
        """
        Return C code which stores a named buffer to the given index.
        """

    @abstractmethod
    def c_resize(self, name, new_length_name, new_length_type):
        """
        Return C code which resizes a named buffer to the given length.
        """


class CKernel:
    """
    A class to represent a C kernel.
    """

    def __init__(self, function_name, c_code):
        self.function_name = function_name
        self.c_code = c_code
        self.c_function = get_c_function(function_name, c_code)

    def __call__(self, *args):
        """
        Calls the C function with the given arguments.
        """
        return self.c_function(*map(methodcaller("to_c"), args))


def to_c_literal(val):
    if hasattr(val, "to_c_literal"):
        return val.to_c_literal()
    return query_property(val, "__self__", "to_c_literal")


register_property(int, "__self__", "to_c_literal", lambda x: str(x))


class CContext:
    """
    A class to represent a C context.
    """

    def __init__(self, level=0):
        self.level = level

    def indent(self):
        return CContext(self.level + 1)

    def prefix(self):
        return "    " * self.level

    def __call__(self, node):
        match node:
            case asm.Immediate(val):
                return to_c_literal(val)
            case asm.Variable(name):
                return name
            case asm.Call(name, args):
                return f"{name}({', '.join(self(arg) for arg in args)})"
            case asm.Assignment(var, value):
                return f"""
{self.prefix()}{var} = {self(value)};
"""
            case asm.ForLoop(var, start, end, body):
                return f"""
{self.prefix()}for (int {var} = {self(start)}; {var} < {self(end)}; {var}++) {{
{self.indent()(body)}
{self.prefix()}}}
"""
            case asm.Block(bodies):
                return f"""
{self.prefix()}{{
{"\n".join(self.indent()(body) for body in bodies)}
{self.prefix()}}}
"""
            case asm.Return(value):
                return f"""
{self.prefix()}return {self(value)};
"""
            case asm.Function(name, args, body):
                return f"""
{self.prefix()}void {name}({", ".join(self(arg) for arg in args)}) {{
{self.indent()(body)}
{self.prefix()}}}
"""
            case asm.Module(funcs, main):
                return "\n".join(map(self, list(*funcs, main)))
