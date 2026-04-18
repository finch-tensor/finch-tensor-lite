from .scripts.nodes import create_ntn_simple_node
from finchlite.finch_notation import NotationCFGBuilder

def test_ntn_cfg_printer_simple(file_regression):
    prgm = create_ntn_simple_node()
    cfg = NotationCFGBuilder().build(prgm)
    file_regression.check(str(cfg), extension=".txt")