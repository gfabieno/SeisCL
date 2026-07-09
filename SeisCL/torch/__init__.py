"""In-memory, CUDA-only, torch.autograd-differentiable binding for SeisCL.

Requires the optional `torch` install extra (`pip install -e .[torch]`),
which builds this subpackage's compiled extension (bindings.cpp) against
seiscl_core (CMakeLists.txt's BUILD_TORCH_CORE target). Importing this
subpackage is optional -- SeisCL.SeisCL (the subprocess/HDF5 workflow)
does not depend on it.
"""

from .op import Config, seiscl_forward

__all__ = ["Config", "seiscl_forward"]
