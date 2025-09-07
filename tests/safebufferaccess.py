#!/usr/bin/env python3

"""
safebufferaccess.py: A safe buffer testing script for finch

This script is required because when the C Kernel hits an array out of bounds
for a safe buffer, it will panic and bring down the interpreter with it. So
this script needs to be invoked so that the testing code continues running.
"""

import ctypes

import numpy as np

import finch.finch_assembly as asm
from finch.util import qual_str
from finch.codegen.c import CBufferFType, CCompiler, CContext, CStackFType, c_type
from finch.codegen.safe_buffer import SafeNumpyBuffer, SafeNumpyBufferFType
import argparse

parser = argparse.ArgumentParser(
    prog="safebufferaccess.py",
)
parser.add_argument(
    "--size", "-s", type=int, help="the size of the array to initialize", default=3
)
subparser = parser.add_subparsers(required=True, dest="subparser_name")

load = subparser.add_parser("load", help="attempt to load some element")
load.add_argument("index", type=int, help="the index to load")

store = subparser.add_parser("store", help="attempt to store into some element")
store.add_argument("index", type=int, help="the index to load")
store.add_argument("value", type=int, help="the value to store")

args = parser.parse_args()


a = np.array(range(args.size), dtype=ctypes.c_int64)
ab = SafeNumpyBuffer(a)
ab_v = asm.Variable("a", ab.ftype)
ab_slt = asm.Slot("a_", ab.ftype)
idx = asm.Variable("idx", ctypes.c_size_t)
val = asm.Variable("val", ctypes.c_int64)

res_var = asm.Variable("val", ab.ftype.element_type)

mod = CCompiler()(
    asm.Module(
        (
            asm.Function(
                asm.Variable("finch_access", ab.ftype.element_type),
                (ab_v, idx),
                asm.Block(
                    (
                        asm.Unpack(ab_slt, ab_v),
                        asm.Assign(
                            res_var,
                            asm.Load(ab_slt, idx),
                        ),
                        asm.Return(res_var),
                    )
                ),
            ),
            asm.Function(
                asm.Variable("finch_change", ab.ftype.element_type),
                (ab_v, idx, val),
                asm.Block(
                    (
                        asm.Unpack(ab_slt, ab_v),
                        asm.Store(
                            ab_slt,
                            idx,
                            val,
                        ),
                    )
                ),
            ),
        )
    )
)
access = mod.finch_access
change = mod.finch_change

match args.subparser_name:
    case "load":
        print(access(ab, ctypes.c_size_t(args.index)).value)
    case "store":
        change(ab, ctypes.c_size_t(args.index), ctypes.c_int64(args.value))
        print(str(ab.arr).replace("\n", ","))
