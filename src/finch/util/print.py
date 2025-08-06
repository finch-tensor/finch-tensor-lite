"""
This module contains the general function to print finch_logic, finch_notation,
and finch_assembly programs
"""

from finch.finch_assembly import Module as AssemblyModule
from finch.finch_assembly.printer import PrinterCompiler as asm_printer
from finch.finch_logic import LogicNode
from finch.finch_logic.printer import PrinterCompiler as log_printer
from finch.finch_notation import Module as NotationModule
from finch.finch_notation.printer import PrinterCompiler as ntn_printer


def print_finch_program(program: LogicNode | NotationModule | AssemblyModule) -> str:
    """
    Print a Finch program in a human-readable format.
    Args:
        program: The program to print.
    Returns:
        str: The printed program.
    """
    if isinstance(program, LogicNode):
        return log_printer()(program)
    if isinstance(program, NotationModule):
        return ntn_printer()(program)
    if isinstance(program, AssemblyModule):
        return asm_printer()(program)

    raise TypeError("Unsupported program type for printing.")
