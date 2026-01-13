# Q&A about Finch codegen

## What is serialize/construct/unpack/... ?

When a complex data structure, such as a hash table or array, gets passed into a
kernel (C or Numba) as an argument, there are five different stages that are
performed:

1. The data structure is serialized to a format that can be passed into the
   kernel (a cstruct type for C, a python object for numba). This format is
   almost never the data structure itself, but a _wrapper_ that encapsulates it.
2. Inside the kernel, the data structure is unpacked to reveal the underlying
   data structure.
3. The kernel modifies this data structure. The modifications it should do are
   provided by assembly instructions. See the hashtable implementation for
   reference on how these mutations work.
4. The kernel has finished making its modifications. It's time to _repack_ the
   data structure.
5. The kernel has finished running. If we have mutated the serialized data
   structure, we now _deserialize_ to sync up our original data structure.

In particular, deserialize is extremely badly named. It does not return an
actual object, but instead takes a finch type, the original object, and its
serialized equivalent (modified by the kernel). If the object passed is a scalar
or immutable (ints, floats, tuples, ...) deserialization *should do nothing*.

The function that turns an object in the kernel back into a python object is
called `construct_from_c(fmt, obj)` or `construct_from_numba(fmt, obj)`. It is
only called for the object being returned by the kernel.

Unpacking is only implemented for arrays and hash tables. It should not be
implemented for scalar types.

### C Example

In the case of an array containing a contiguous C numpy array, here's how it might
work:

1. The NumpyBuffer (denoted `pybuf`) gets turned into a CNumpyBuffer(buffer, obj)
   where buffer is a raw pointer, obj is a PyObject\* pointing back to our
   NumpyBuffer, and CNumpyBuffer is a ctypes struct. Denote this serialized
   buffer `cnp`. `cnp` gets passed into the kernel.
2. (In kernel) The CNumpyBuffer gets unpacked into variables "buffer1" and
   "obj1" (actual names of these variables will depend on what the context has).
   These variable _names_ get stored in the CContext's slots as a tuple
   `("buffer1", "cnp")`. The following code gets emitted in the process:
   ```c
   int* buffer1 = cnp->buffer;
   ```
3. (In kernel) Assembly instructions for load and store are called on the slot
   which contains that tuple of variable names. The context emits code that
   operates on the variable `buffer1`.
4. (In kernel) Repack those variables back into the original CBuffer. Notice
   that we store `"cnp"` in our tuple for this exact situation:
   ```c
   cnp->buffer = buffer1;
   ```
5. Update the original NumpyBuffer. The deserialization code looks something
   like
   ```python
   pybuf.arr = cnp.arr
   ```

The following functions need to be either registered via `register_property`
(see examples littered in c.py) or added as methods to any data structure:

1. `serialize_to_c(self_fmt, obj)`
2. `deserialize_from_c(self_fmt, obj, ser_obj)`
3. `c_unpack(self, ctx: "CContext", var_n: str, val: AssemblyExpression) -> Variables`
4. `c_repack(self, ctx: "CContext", lhs: str, unpacked_vars: Variables)`

Here, `Variables` is a placeholder for the tuple type that holds all variable
names that were used in the unpacking.

Unpack and Repack require c codegen contexts to freshen variables and emit
initialization code.

### Numba Example

Here's how it works for numba:

1. The NumpyBuffer (denoted `pybuf`) gets turned into a one-element list of the
   form `nnp = list([ndarray])`. `nnp` gets passed into the kernel.
2. (In kernel) The CNumpyBuffer gets unpacked into variables "buffer1" and
   "nnp" (actual names of these variables will depend on what the context has).
   These variable _names_ get stored in the CContext's slots as a tuple
   `("buffer1", "nnp")`. The following code gets emitted in the process:
   ```python
   buffer1 = nnp[0]
   ```
3. (In kernel) Assembly instructions for load and store are called on the slot
   which contains that tuple of variable names. The context emits code that
   operates on the variable `buffer1`.
4. (In kernel) Repack those variables back into the original CBuffer. Notice
   that we store `nnp` in our tuple for this exact situation:
   ```python
    nnp[0] = buffer1
   ```
5. Update the original NumpyBuffer. The deserialization code looks something
   like
   ```python
   pybuf.arr = cnp[0]
   ```

The following functions need to be either registered via `register_property`
(see examples littered in numba_backend.py) or added as methods to any data structure:

1. `serialize_to_numba(self_fmt, obj)`
2. `deserialize_from_numba(self_fmt, obj, ser_obj)`
3. `numba_unpack(self, ctx: "NumbaContext", var_n: str, val: AssemblyExpression) -> Variables`
4. `numba_repack(self, ctx: "NumbaContext", lhs: str, unpacked_vars: Variables)`

Here, `Variables` is a placeholder for the tuple type that holds all variable
names that were used in the unpacking.

Unpack and Repack require numba codegen contexts to freshen variables and emit
initialization code.

## Hash Table Data Types

Hash Tables may have keys and values that have the following finch types:
1. A scalar (as defined in numpy or python)
2. An FType inheriting from `ImmutableStructFType`

Nothing else is currently supported nor is intended to be supported.

All `ImmutableStructFType`'s will get serialized to typed tuples for numba.

The C Context requires two properties, `c_hash` and `c_eq` for a type that wants
to be hashed. Each of these returns the *name* of a macro that will get expanded
for hashing or equality. See some examples below for how these functions work
(they are currently in the codebase):

```python
# For trivial scalars.
def c_hash_default(fmt, ctx: "CContext"):
    ctx.add_header(f'#include "{common_h}"')
    return "c_default_hash"

def c_eq_default(fmt, ctx: "CContext"):
    ctx.add_header(f'#include "{common_h}"')
    return "c_default_eq"

register_property(
    np.generic,
    "c_hash",
    "__attr__",
    c_hash_default,
)
register_property(
    np.generic,
    "c_eq",
    "__attr__",
    c_eq_default,
)

# For immutable structs with fields.
def c_hash_struct(fmt: ImmutableStructFType, ctx: "CContext"):
    # this should be true in whatever structs we have.
    assert isinstance(fmt, Hashable)
    if fmt in ctx.datastructures:
        properties: CHashableProperties = ctx.datastructures[fmt]
        if properties.get("hash") is not None:
            return properties["hash"]
    else:
        ctx.datastructures[fmt] = {}

    macros = [c_hash(fmt, ctx) for fmt in fmt.struct_fieldformats]
    name = ctx.freshen("hash")
    ctx.datastructures[fmt]["hash"] = name

    # implement recursion with &{var_n}->{struct_field}
    var_n = ctx.freshen("var")
    args = ",".join(
        f"{macro}(&({var_n})->{field})"
        for macro, field in zip(macros, fmt.struct_fieldnames, strict=False)
    )
    ctx.add_header(f"#define {name}({var_n}) c_hash_mix({args})")
    return name


register_property(
    ImmutableStructFType,
    "c_hash",
    "__attr__",
    c_hash_struct,
)


def c_eq_struct(fmt: ImmutableStructFType, ctx: "CContext"):
    # this should be true in whatever structs we have.
    assert isinstance(fmt, Hashable)
    if fmt in ctx.datastructures:
        properties: CHashableProperties = ctx.datastructures[fmt]
        if properties.get("eq") is not None:
            return properties["eq"]
    else:
        ctx.datastructures[fmt] = {}

    macros = [c_eq(fmt, ctx) for fmt in fmt.struct_fieldformats]
    name = ctx.freshen("eq")
    ctx.datastructures[fmt]["eq"] = name

    # implement recursion with &{var_n}->{struct_field}
    var1_n = ctx.freshen("var")
    var2_n = ctx.freshen("var")
    args = " && ".join(
        f"{macro}(&({var1_n})->{field}, &({var2_n})->{field})"
        for macro, field in zip(macros, fmt.struct_fieldnames, strict=False)
    )
    ctx.add_header(f"#define {name}({var1_n}, {var2_n}) ({args})")
    return name


register_property(
    ImmutableStructFType,
    "c_eq",
    "__attr__",
    c_eq_struct,
)
```
