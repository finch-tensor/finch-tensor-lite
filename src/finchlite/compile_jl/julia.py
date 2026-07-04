import os  # noqa: I001, F401

os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
import juliapkg  # noqa: I001, F401

# To change the version of Finch used, see the documentation for pyjuliapkg here: https://github.com/JuliaPy/pyjuliapkg
# Use pyjuliapkg to modify the `juliapkg.json` file in the root of this repo.
# You can also run `develop.py` to quickly use a local copy of Finch.jl.
# An example development json is found in `juliapkg_dev.json`
import juliacall as jc  # noqa: F401
from juliacall import Main as jl  # noqa: E402, F401

jl.seval("using Finch")
jl.seval("using HDF5")
jl.seval("using NPZ")
jl.seval("using TensorMarket")
jl.seval("using Random")
jl.seval("using Statistics")
