"""In-memory, CUDA-only, torch.autograd-differentiable binding for SeisCL.

Requires the optional `torch` install extra (`pip install -e .[torch]`),
which builds this subpackage's compiled extension (bindings.cpp) against
seiscl_core (CMakeLists.txt's BUILD_TORCH_CORE target). Importing this
subpackage is optional -- SeisCL.SeisCL (the subprocess/HDF5 workflow)
does not depend on it.
"""

import atexit

from . import _C
from .op import Config, seiscl_forward

# Built engines (CUDA context + compiled kernels + device buffers) are
# reused across calls with the same problem shape, which is worth ~2s per
# call. These control that cache; the defaults are fine for a normal
# training loop.
set_engine_cache_size = _C.set_engine_cache_size
engine_cache_size = _C.engine_cache_size
clear_engine_cache = _C.clear_engine_cache

# Free cached CUDA contexts/kernels/buffers while the driver is still up,
# rather than relying on C++ static destruction order at interpreter exit.
atexit.register(_C._shutdown_engine_cache)

__all__ = ["Config", "seiscl_forward", "set_engine_cache_size",
           "engine_cache_size", "clear_engine_cache"]
