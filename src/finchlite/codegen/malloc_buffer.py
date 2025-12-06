import ctypes
from collections.abc import Hashable
from dataclasses import dataclass
from textwrap import dedent
from typing import TypedDict

import numpy as np

from finchlite.codegen.c import (
    CBufferFType,
    CContext,
    CStackFType,
    c_type,
    construct_from_c,
    load_shared_lib,
    serialize_to_c,
)
from finchlite.codegen.numpy_buffer import CBufferFields
from finchlite.finch_assembly import Buffer
from finchlite.finch_assembly.nodes import AssemblyExpression, Stack
from finchlite.util import qual_str


class CMallocBufferStruct(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
    ]


class CMallocBufferMethods(TypedDict):
    init: str
    resize: str
    free: str


@dataclass
class CMallocBufferLibrary:
    library: ctypes.CDLL
    methods: CMallocBufferMethods

    def init(self, *args):
        return getattr(self.library, self.methods["init"])(*args)

    def resize(self, *args):
        return getattr(self.library, self.methods["resize"])(*args)

    def free(self, *args):
        return getattr(self.library, self.methods["free"])(*args)


class MallocBufferBackend:
    _library: dict[Hashable, CMallocBufferLibrary] = {}

    @classmethod
    def gen_code(
        cls,
        ctx: "CContext",
        ftype: "MallocBufferFType",
        inline: bool = False,
    ):
        ctx.add_header("#include <string.h>")
        ctx.add_header("#include <stdlib.h>")
        ctx.add_header("#include <stdio.h>")

        methods: CMallocBufferMethods = {
            "init": ctx.freshen("mallocbuffer_init"),
            "free": ctx.freshen("mallocbuffer_free"),
            "resize": ctx.freshen("mallocbuffer_resize"),
        }
        ctx.datastructures[ftype] = methods

        buffer_type = ctx.ctype_name(CMallocBufferStruct)
        elt_type = ctx.ctype_name(c_type(ftype.element_type))

        inline_s = "static inline " if inline else ""
        libcode = dedent(
            f"""
            {inline_s}{elt_type}* {methods["resize"]}({elt_type}* data, size_t len_old, size_t len_new) {{
                data = realloc(data, sizeof({elt_type}) * len_new);
                if (data == 0) {{
                    printf("Malloc Failed!\\n");
                    exit(1);
                }}
                if (len_new > len_old) {{
                    memset(&data[len_old], 0, (len_new - len_old) * sizeof({elt_type}));
                }}
                return data;
            }}
            // methods below are not used by the kernel.
            {inline_s}void {methods["free"]}({buffer_type} *m) {{
                free(m->data);
                m->data = 0;
                m->length = 0;
            }}
            {inline_s}void {methods["init"]}(
                {buffer_type} *m,
                size_t datasize,
                size_t length
            ) {{
                m->length = length;
                m->data = malloc(length * datasize);
                memset(m->data, 0, length * datasize);
            }}
            """
        )
        ctx.add_header(libcode)
        return methods

    @classmethod
    def library(cls, ftype: "MallocBufferFType") -> CMallocBufferLibrary:
        # lazy compile the library.
        if ftype in cls._library:
            return cls._library[ftype]
        ctx = CContext()
        methods = cls.gen_code(ctx, ftype)
        lib = load_shared_lib(ctx.emit_global())

        resize_func = getattr(lib, methods["resize"])
        resize_func.argtypes = [
            ctypes.POINTER(c_type(ftype.element_type)),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        resize_func.restype = ctypes.POINTER(c_type(ftype.element_type))

        init_func = getattr(lib, methods["init"])
        init_func.argtypes = [
            ctypes.POINTER(CMallocBufferStruct),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        init_func.restype = None

        free_func = getattr(lib, methods["free"])
        free_func.argtypes = [
            ctypes.POINTER(CMallocBufferStruct),
        ]
        free_func.restype = None

        cls._library[ftype] = CMallocBufferLibrary(lib, methods)
        return cls._library[ftype]


class MallocBuffer(Buffer):
    """
    A buffer that uses Malloc buffers to store data.

    To check out the corresponding C code, you should reference
    ./malloc_buffer_backend.c in the same directory as the malloc_buffer.py
    file
    """

    def __init__(self, length: int, dtype, data=None):
        """
        Constructor for the MallocBuffer class.

        length (int): the length of the malloc array.
        dtype (type[ctypes._CData]): the ctype that the buffer will be based on.
        data (optional): a list of data to initialize the buffer with.
        """
        self._dtype = dtype
        self._c_dtype = c_type(dtype)
        self.buffer = ctypes.pointer(CMallocBufferStruct())

        MallocBufferBackend.library(self.ftype).init(
            self.buffer,
            ctypes.c_size_t(ctypes.sizeof(self._c_dtype)),
            ctypes.c_size_t(length),
        )
        if data is None:
            return
        if len(data) > length:
            raise IndexError

        for idx, elt in enumerate(data):
            self.castbuffer[idx] = serialize_to_c(self._dtype, elt)

    def __del__(self):
        """
        Frees the mallocbuffer stored inside.
        """
        if hasattr(self, "buffer"):
            MallocBufferBackend.library(self.ftype).free(self.buffer)

    @property
    def castbuffer(self):
        return ctypes.cast(self.buffer.contents.data, ctypes.POINTER(self._c_dtype))

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a MallocBufferFType.
        """
        return MallocBufferFType(self._dtype)

    # TODO should be property
    def length(self):
        return np.intp(self.buffer.contents.length)

    def load(self, index):
        value = self.castbuffer[index]
        return construct_from_c(self.ftype.element_type, value)

    def store(self, index, value):
        value = serialize_to_c(self.ftype.element_type, value)
        self.castbuffer[index] = value

    def resize(self, new_length):
        new_length = serialize_to_c(self.ftype.length_type, new_length)
        MallocBufferBackend.library(self.ftype).resize(self.buffer, new_length)

    def __str__(self):
        array = self.castbuffer[: self.length()]
        return f"malloc_buf({array})"


class MallocBufferFType(CBufferFType, CStackFType):
    """
    A ftype for buffers that uses libc-provided malloc functions. This is a
    concrete implementation of the BufferFType class.

    This does not support the numba backend.
    """

    def __init__(self, dtype):
        self._dtype = dtype
        self._c_dtype = c_type(dtype)

    def __eq__(self, other):
        if not isinstance(other, MallocBufferFType):
            return False
        return self._dtype == other._dtype

    def __str__(self):
        return f"malloc_buf_t({qual_str(self._dtype)})"

    def __repr__(self):
        return f"MallocBufferFType({qual_str(self._dtype)})"

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return np.intp

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the buffer. This will be a ctypes array.
        """
        return self._dtype

    def __hash__(self):
        return hash(("MallocBufferFType", self._dtype))

    def __call__(self, len: int = 0, dtype: type | None = None):
        if dtype is None:
            dtype = self._dtype
        return MallocBuffer(len, dtype)

    def c_type(self):
        return ctypes.POINTER(CMallocBufferStruct)

    def c_length(self, ctx: "CContext", buf: "Stack"):
        assert isinstance(buf.obj, CBufferFields)
        return buf.obj.length

    def c_data(self, ctx: "CContext", buf: "Stack"):
        assert isinstance(buf.obj, CBufferFields)
        return buf.obj.data

    def c_load(self, ctx: "CContext", buf: "Stack", idx: "AssemblyExpression"):
        assert isinstance(buf.obj, CBufferFields)
        return f"({buf.obj.data})[{ctx(idx)}]"

    def c_store(
        self,
        ctx: "CContext",
        buf: "Stack",
        idx: "AssemblyExpression",
        value: "AssemblyExpression",
    ):
        assert isinstance(buf.obj, CBufferFields)
        ctx.exec(f"{ctx.feed}({buf.obj.data})[{ctx(idx)}] = {ctx(value)};")

    def c_resize(self, ctx: "CContext", buf: "Stack", new_len: "AssemblyExpression"):
        assert isinstance(buf.obj, CBufferFields)
        methods: CMallocBufferMethods | None = ctx.datastructures.get(self)
        assert methods is not None, (
            "A Mallocbuffer must be unpacked before being operated on!"
        )

        new_len = ctx(ctx.cache("len", new_len))
        data = buf.obj.data
        length = buf.obj.length

        ctx.exec(
            f"{ctx.feed}{data} = {methods['resize']}({data}, {length}, {new_len});\n"
            f"{ctx.feed}{length} = {new_len};"
        )
        return

    def c_unpack(self, ctx: "CContext", var_n, val):
        """
        Unpack the malloc buffer into C context.
        """
        data = ctx.freshen(var_n, "data")
        length = ctx.freshen(var_n, "length")
        t = ctx.ctype_name(c_type(self.element_type))
        ctx.add_header("#include <stddef.h>")

        ctx.add_datastructure(
            self,
            lambda ctx: MallocBufferBackend.gen_code(ctx, self, inline=True),
        )

        ctx.exec(
            f"{ctx.feed}{t}* {data} = ({t}*){ctx(val)}->data;\n"
            f"{ctx.feed}size_t {length} = {ctx(val)}->length;"
        )

        return CBufferFields(data, length, var_n)

    def c_repack(self, ctx, var_n: str, obj: "CBufferFields"):
        """
        Repack the buffer from C context.
        """
        ctx.exec(
            f"{ctx.feed}{var_n}->data = (void*){obj.data};\n"
            f"{ctx.feed}{var_n}->length = {obj.length};"
        )
        return

    def serialize_to_c(self, obj: MallocBuffer):
        """
        Serialize the Malloc buffer to a C-compatible structure.
        This is trivial.
        """
        return obj.buffer

    def deserialize_from_c(self, obj, c_buffer):
        pass
        # this is handled by the resize callback

    def construct_from_c(self, c_buffer):
        """
        Construct a MallocBuffer from a C-compatible structure.
        """
        return c_buffer.contents
