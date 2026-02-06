from finchlite.finch_assembly.cfg_builder import assembly_build_cfg
from finchlite.finch_assembly.dataflow import (
    assembly_copy_propagation_debug,
    assembly_dataflow_preprocess,
)

from .util.nodes import (
    create_asm_comprehensive_node,
    create_asm_dot_node,
    create_asm_if_node,
)


def test_asm_cfg_printer_if(file_regression):
    prgm = create_asm_if_node()
    prgm = assembly_dataflow_preprocess(prgm)
    cfg = assembly_build_cfg(prgm)
    file_regression.check(str(cfg), extension=".txt")


def test_asm_cfg_printer_dot(file_regression):
    prgm = create_asm_dot_node()
    prgm = assembly_dataflow_preprocess(prgm)
    cfg = assembly_build_cfg(prgm)
    file_regression.check(str(cfg), extension=".txt")


def test_asm_copy_propagation_if(file_regression):
    prgm = create_asm_if_node()
    prgm = assembly_dataflow_preprocess(prgm)
    copy_propagation = assembly_copy_propagation_debug(prgm)
    file_regression.check(str(copy_propagation), extension=".txt")


def test_asm_copy_propagation_dot(file_regression):
    prgm = create_asm_dot_node()
    prgm = assembly_dataflow_preprocess(prgm)
    copy_propagation = assembly_copy_propagation_debug(prgm)
    file_regression.check(str(copy_propagation), extension=".txt")


def test_asm_cfg_printer_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    prgm = assembly_dataflow_preprocess(prgm)
    cfg = assembly_build_cfg(prgm)
    file_regression.check(str(cfg), extension=".txt")


def test_asm_copy_propagation_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    prgm = assembly_dataflow_preprocess(prgm)
    copy_propagation = assembly_copy_propagation_debug(prgm)
    file_regression.check(str(copy_propagation), extension=".txt")
