from .nodes import (
    create_asm_comprehensive_node,
    create_asm_dot_node,
    create_asm_if_node,
    create_log_simple_node,
    create_ntn_simple_node,
)


def test_log_printer(file_regression):
    prgm = create_log_simple_node()
    file_regression.check(str(prgm), extension=".txt")


def test_ntn_printer(file_regression):
    prgm = create_ntn_simple_node()
    file_regression.check(str(prgm), extension=".txt")


def test_asm_printer_if(file_regression):
    prgm = create_asm_if_node()
    file_regression.check(str(prgm), extension=".txt")


def test_asm_printer_dot(file_regression):
    prgm = create_asm_dot_node()
    file_regression.check(str(prgm), extension=".txt")


def test_asm_printer_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    file_regression.check(str(prgm), extension=".txt")
