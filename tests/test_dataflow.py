from finchlite.finch_assembly import (
    AssemblyCopyPropagation,
    assembly_build_cfg,
    assembly_copy_propagation,
    assembly_dataflow_analyze,
    assembly_dataflow_preprocess,
)

from .scripts.nodes import (
    create_asm_comprehensive_node,
    create_asm_dot_node,
    create_asm_if_node,
)


def test_asm_cfg_printer_if(file_regression):
    prgm = create_asm_if_node()
    prgm = assembly_dataflow_preprocess(prgm)
    cfg = assembly_build_cfg(prgm)
    file_regression.check(str(cfg), extension=".txt")


def test_asm_copy_propagation_debug_if(file_regression):
    prgm = create_asm_if_node()
    (ctx, _) = assembly_dataflow_analyze(prgm, AssemblyCopyPropagation)
    file_regression.check(str(ctx), extension=".txt")


def test_asm_copy_propagation_if(file_regression):
    prgm = create_asm_if_node()
    prgm = assembly_copy_propagation(prgm)
    file_regression.check(str(prgm), extension=".txt")


def test_asm_cfg_printer_dot(file_regression):
    prgm = create_asm_dot_node()
    prgm = assembly_dataflow_preprocess(prgm)
    cfg = assembly_build_cfg(prgm)
    file_regression.check(str(cfg), extension=".txt")


def test_asm_copy_propagation_debug_dot(file_regression):
    prgm = create_asm_dot_node()
    (ctx, _) = assembly_dataflow_analyze(prgm, AssemblyCopyPropagation)
    file_regression.check(str(ctx), extension=".txt")


def test_asm_copy_propagation_dot(file_regression):
    prgm = create_asm_dot_node()
    prgm = assembly_copy_propagation(prgm)
    file_regression.check(str(prgm), extension=".txt")


def test_asm_cfg_printer_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    prgm = assembly_dataflow_preprocess(prgm)
    cfg = assembly_build_cfg(prgm)
    file_regression.check(str(cfg), extension=".txt")


def test_asm_copy_propagation_debug_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    (ctx, _) = assembly_dataflow_analyze(prgm, AssemblyCopyPropagation)
    file_regression.check(str(ctx), extension=".txt")


def test_asm_copy_propagation_comprehensive(file_regression):
    prgm = create_asm_comprehensive_node()
    prgm = assembly_copy_propagation(prgm)
    file_regression.check(str(prgm), extension=".txt")
