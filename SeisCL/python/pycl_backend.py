
import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
from SeisCL.python.seis2D import Grid, Function,  Propagator, Sequence, ReversibleFunction
import re
from SeisCL.python.FDstencils import get_pos_header, FDCoefficients, CUDACL_header, grid_stop_header


options_def = ["-D LCOMM=0",
               "-D ABS_TYPE=0",
               "-D FREESURF=0",
               "-D DEVID=0",
               "-D MYLOCALID=0",
               "-D NUM_DEVICES=0",
               "-D NLOCALP=0"]


# TODO review how dimensions names are assigned
# TODO migration to pycuda, there is no match_dtype_to_c_struct
class GridCL(Grid):

    backend = cl.array.Array

    def __init__(self, queue, shape=(10, 10), nfddim=None,
                 pad=2, dh=1, dt=1, nt=1, dtype=np.float32, zero_boundary=False,
                 **kwargs):
        super().__init__(shape=shape, pad=pad, dh=dh, dt=dt, nt=nt,
                         nfddim=nfddim, zero_boundary=zero_boundary, dtype=dtype)
        self.queue = queue
        self._struct_np, self.struct_c = self.define_struct()
        self.copy_array = CopyArrayCL(queue.context)


    def zero(self):
        return cl.array.zeros(self.queue, self.shape, self.dtype, order="F")

    def random(self):
        if self.zero_boundary:
            state = np.zeros(self.shape, dtype=self.dtype)
            state[self.valid] = np.random.rand(*state[self.valid].shape)
        else:
            state = np.random.rand(*self.shape).astype(dtype=self.dtype)
        state = cl.array.to_device(self.queue,
                                   np.require(state, requirements='F'))
        return state

    def assign_data(self, data):
        return cl.array.to_device(self.queue,
                                  np.require(data.astype(dtype=self.dtype),
                                             requirements='F'))

    def create_cache(self, cache_size=1, regions=None):
        if regions:
            cache = []
            for region in regions:
                shape = [0 for _ in range(len(self.shape))]
                for ii in range(len(self.shape)):
                    if region[ii] is Ellipsis:
                        shape[ii] = self.shape[ii]
                    else:
                        indices = region[ii].indices(self.shape[ii])
                        shape[ii] = int((indices[1]-indices[0])/indices[2])
                cache.append(cl.array.empty(self.queue,
                                            tuple(shape + [cache_size]),
                                            self.dtype, order="F"))
            return cache
        else:
            return cl.array.empty(self.queue,
                                  tuple(list(self.shape) + [cache_size]),
                                  self.dtype, order="F")

    #TODO create such a structure is not available in pycuda
    #Workwaround: define as macro or as seperate kenerl inputs
    def define_struct(self):
        dtype = np.dtype([
            ("NX", np.int32),
            ("NY", np.int32),
            ("NZ", np.int32),
            ("nt", np.int32),
            ("dh", np.float32),
            ("dt", np.float32),
            ("offset", np.int32),
            ("lsizez", np.int32),
            ("lsizex", np.int32),
            ("lidx", np.int32),
            ("lidz", np.int32),
            ("gidx", np.int32),
            ("gidz", np.int32),
        ])
        name = "grid"
        dtype, c_decl = match_dtype_to_c_struct(self.queue.device, name, dtype)
        dtype = get_or_register_dtype(name, dtype)

        return dtype, c_decl

    @property
    def headers(self):
        return self.struct_c + get_pos_header + grid_stop_header

    @property
    def options(self):
        return ["-D __FDOH__=%d" % self.pad,
                "-D __ND__=%d" % self.nfddim]

    @property
    def struct_np(self):
        struct = np.zeros(1, self._struct_np)
        if self.nfddim == 1:
            struct[0]["NZ"] = self.shape[0]
        elif self.nfddim == 2:
            struct[0]["NZ"] = self.shape[0]
            struct[0]["NX"] = self.shape[1]
        elif self.nfddim == 3:
            struct[0]["NZ"] = self.shape[0]
            struct[0]["NY"] = self.shape[1]
            struct[0]["NX"] = self.shape[2]
        struct[0]["nt"] = self.nt
        struct[0]["dh"] = self.dh
        struct[0]["dt"] = self.dt
        return struct

    @staticmethod
    def np(array):
        return array.get()


class ComputeRessource:

    def __init__(self, device_type=cl.device_type.GPU, allowed_device_ids=None):
        platforms = cl.get_platforms()
        self.platform = None
        self.devices = None
        for platform in platforms:
            devs = platform.get_devices(device_type=device_type)
            if devs:
                self.devices = devs
                self.platform = platform
                break
        if self.platform is None:
            raise ValueError("Could not find any allowable devices")
        if allowed_device_ids is not None:
            self.devices = [d for ii, d in enumerate(self.devices)
                            if ii in allowed_device_ids]
        self.context = cl.Context(devices=self.devices)
        self.queues = []
        self.queuecomms = []
        for dev in self.devices:
            self.queues.append(cl.CommandQueue(self.context, device=dev))
            self.queuecomms.append(cl.CommandQueue(self.context, device=dev))


class Kernel:

    def __init__(self, queue, grid, global_size, local_size, src, name,
                 options, default_args, args_renaming):

        self.queue = queue
        self.grid = grid
        self.global_size = global_size
        self.local_size = local_size
        self.src = src
        self.name = name
        self.options = options
        self.args = self.extract_args(self.src)
        self.names = [m.group(1)
                      for m in re.finditer("FUNDEF void (\w*)\s*\(", src)]
        self._prog = None
        self._kernel = None
        self.event = None
        self.default_args = default_args
        self.args_renaming = args_renaming

    def __call__(self, *args, **kwargs):
        if self.default_args:
            for key in self.default_args:
                if key not in kwargs:
                    kwargs[key] = self.default_args[key]
        arg_list = self.assign_args(*args, **kwargs)
        if "backpropagate" in kwargs:
            if kwargs["backpropagate"] ==1:
                kernels = self.kernel[::-1]
        else:
            kernels = self.kernel
        for kernel in kernels:
            self.event = kernel(self.queue,
                                self.global_size(),
                                self.local_size,
                                *arg_list)

        return args[0]

    @property
    def kernel(self):
        if not self._kernel:
            if not self._prog:
                options = options_def + self.options + self.grid.options
                options += ["-D __EPS__=%d" % self.grid.smallest]
                if self.local_size is not None:
                    total = np.prod([el+self.grid.pad*2
                                     for el in self.local_size])
                    options += ["-D __LSIZE__=%d" % total]
                self._prog = cl.Program(self.queue.context,
                                        self.src).build(options)
            self._kernel = [getattr(self._prog, name) for name in self.names]
        return self._kernel

    @staticmethod
    def extract_args(src):

        m = re.search("FUNDEF void \w*\s*\((.*?)\)", src, re.DOTALL)
        if not m:
            raise ValueError("Could not find arguments to kernel")

        arglist = m.group(1).split(",")
        arglist = [a.split() for a in arglist]
        args = [{} for _ in range(len(arglist))]
        for ii, a in enumerate(arglist):
            args[ii]["name"] = re.search('[\w]+', a[-1]).group(0)
            if "_lin" in args[ii]["name"]:
                args[ii]["name"] = args[ii]["name"].split("_lin")[0]
                args[ii]["bp"] = "linear"
            elif "_adj" in args[ii]["name"]:
                args[ii]["name"] = args[ii]["name"].split("_adj")[0]
                args[ii]["bp"] = "adjoint"
            else:
                args[ii]["bp"] = "forward"

            args[ii]["type"] = a[-2]
            if "__global" in a[0] or "GLOBARG" in a[0]:
                args[ii]["scope"] = "global"
            elif "__local" in a[0]:
                args[ii]["scope"] = "local"
            else:
                args[ii]["scope"] = None
        return args

    def assign_args(self, *args, **kwargs):

        if len(args) == 1:
            indict = {**args[0], **kwargs}
        else:
            indict = {**args[1], **kwargs}
        arg_list = [None for _ in self.args]
        for ii, a in enumerate(self.args):
            name = a["name"]
            if a["name"] in self.args_renaming:
                name = self.args_renaming[name]
            if a["type"] == "grid":
                arg_list[ii] = self.grid.struct_np
            elif a["scope"] == "local":
                if self.local_size is None:
                    raise ValueError("local_size should be defined if using "
                                     "local memory in your GPU kernel")
                else:
                    size = np.prod(self.local_size) * self.grid.dtype(1).itemsize
                arg_list[ii] = cl.LocalMemory(size)
            elif a["bp"] == "linear" or a["bp"] == "adjoint":
                arg_list[ii] = args[0][name]
            elif name in indict:
                arg_list[ii] = indict[name]
            else:
                arg_list[ii] = None
            if type(arg_list[ii]) == cl.array.Array:
                arg_list[ii] = arg_list[ii].data
            elif type(arg_list[ii]) == int:
                arg_list[ii] = np.int32(arg_list[ii])
            elif type(arg_list[ii]) == float:
                arg_list[ii] = np.float32(arg_list[ii])
            elif type(arg_list[ii]) == bool:
                arg_list[ii] = np.int32(arg_list[ii])
            elif type(arg_list[ii]) in (list, tuple, dict):
                raise TypeError("GPU kernels arguments must be a buffer or "
                                "a scalar, got %s for arg %s"
                                % (type(arg_list[ii]), a["name"]))
        return arg_list


class CopyArrayCL:

    def __init__(self, ctx):
        self.ctx = ctx
        self._kernel = None

    @property
    def kernel(self):
        if self._kernel is None:
            self._kernel = cl.Program(self.ctx, """__kernel void copy(
                __global char *dest,      
                const int offsetd, 
                const int stridexd, 
                const int strideyd,
                __global const char *src, 
                const int offsets, 
                const int stridexs, 
                const int strideys,
                const int word_size) {
    
                int write_idx = offsetd 
                                + get_global_id(0) 
                                + get_global_id(1) * stridexd 
                                + get_global_id(2) * strideyd ;
                int read_idx  = offsets 
                                + get_global_id(0) 
                                + get_global_id(1) * stridexs 
                                + get_global_id(2) * strideys;
                dest[write_idx] =  src[read_idx];
    
                }""").build()
        return self._kernel

    def __call__(self, dest, src):
        assert dest.dtype == src.dtype
        assert dest.shape == src.shape
        assert len(dest.shape) <= 2
        if len(dest.shape) == 1:
            dest.shape = (dest.shape[0], 1)
            src.shape = (src.shape[0], 1)
            dest.strides = (dest.strides[0], 0)
            src.strides = (src.strides[0], 0)

        self.kernel.copy(dest.queue,
                    (src.dtype.itemsize, src.shape[0], src.shape[1]),
                    None,
                    dest.base_data,
                    np.uint32(dest.offset),
                    np.uint32(dest.strides[0]),
                    np.uint32(dest.strides[1]),
                    src.base_data,
                    np.uint32(src.offset),
                    np.uint32(src.strides[0]),
                    np.uint32(src.strides[1]),
                    np.uint32(src.dtype.itemsize))


class FunctionGPU(Function):

    forward_src = ""
    linear_src = ""
    adjoint_src = ""

    def __init__(self, grids=None, computegrid=None, local_size=None,
                 options=None, default_args=None, default_identity=False,
                 **kwargs):
        super().__init__(grids, **kwargs)

        self.default_identity = default_identity
        if not hasattr(self, "headers"):
            self.headers = ""
        if not hasattr(self, "args_renaming"):
            self.args_renaming = {}

        self.computegrid = computegrid
        self.local_size = local_size
        if computegrid is None:
            if self.updated_states:
                gridname = self.default_grids[self.updated_states[0]]
            else:
                gridname = self.default_grids[self.required_states[0]]
            self.computegrid = self.grids[gridname]
        if options is None:
            options = []
        if default_args is not None:
            default_args = self.make_kwargs_compatible(**default_args)
        name = self.__class__.__name__
        headers = CUDACL_header + self.computegrid.headers + self.headers

        self.forward = Kernel(self.computegrid.queue,
                              self.computegrid,
                              self.global_size_fw,
                              self.local_size,
                              headers + self.forward_src,
                              name,
                              options,
                              default_args,
                              self.args_renaming)
        if self.linear_src:
            self.linear = Kernel(self.computegrid.queue,
                                 self.computegrid,
                                 self.global_size_fw,
                                 self.local_size,
                                 headers + self.linear_src,
                                 name + "_lin",
                                 options,
                                 default_args,
                                 self.args_renaming)
        else:
            if not self.default_identity:
                self.linear = Kernel(self.computegrid.queue,
                                     self.computegrid,
                                     self.global_size_fw,
                                     self.local_size,
                                     headers + self.forward_src,
                                     name,
                                     options,
                                     default_args,
                                     self.args_renaming)
                for ii in range(len(self.linear.args)):
                    aname = self.linear.args[ii]["name"]
                    if (aname in self.required_states
                        or aname in self.updated_states
                        or aname in self.args_renaming):
                        self.linear.args[ii]["bp"] = "linear"
            else:
                self.linear = lambda x: x

        if self.adjoint_src:
            self.adjoint = Kernel(self.computegrid.queue,
                                  self.computegrid,
                                  self.global_size_adj,
                                  self.local_size,
                                  headers + self.adjoint_src,
                                  name + "_adj",
                                  options,
                                  default_args,
                                  self.args_renaming)
        elif default_identity:
            self.adjoint = lambda x: x

    def cache_states(self, states):
        for el in self.updated_states:
            if el in self.updated_regions:
                for ii, region in enumerate(self.updated_regions[el]):
                    self.grids[el].copy_array(self._forward_states[el][ii][..., self.ncall],
                                              states[el][region])
            else:
                self._forward_states[el][..., self.ncall] = states[el]
        self.ncall += 1

    def make_kwargs_compatible(self, **kwargs):
        for el in kwargs:
            if type(kwargs[el]) in [np.ndarray]:
                kwargs[el] = self.computegrid.assign_data(kwargs[el])
        return kwargs

    def global_size_fw(self):
        if self.local_size is None:
            return [np.int32(s - 2 * self.computegrid.pad)
                    for s in self.computegrid.shape]
        else:
            return [np.int32(np.ceil((s - 2 * self.computegrid.pad) / float(l)) * l)
                    for s, l in zip(self.computegrid.shape, self.local_size)]

    def global_size_adj(self):
        return self.global_size_fw()


class PropagatorCL(Propagator):
    def __init__(self, kernel, nt, **kwargs):

        self.computegrid = kernel.computegrid
        super().__init__(kernel, nt, **kwargs)

    def make_kwargs_compatible(self, **kwargs):
        for el in kwargs:
            if type(kwargs[el]) in [np.ndarray]:
                kwargs[el] = cl.array.to_device(self.computegrid.queue,
                                                kwargs[el])
        return kwargs


class SequenceCL(Sequence):

    def __init__(self, kernels, grids=None, **kwargs):

        for kernel in kernels:
            try:
                self.computegrid = kernel.computegrid
                break
            except AttributeError:
                pass
        super().__init__(kernels, grids=grids, **kwargs)

    def make_kwargs_compatible(self, **kwargs):
        for el in kwargs:
            if type(kwargs[el]) in [np.ndarray]:
                kwargs[el] = cl.array.to_device(self.computegrid.queue,
                                                kwargs[el])
        return kwargs


class ReversibleFunctionCL(ReversibleFunction, FunctionGPU):
    def make_kwargs_compatible(self, **kwargs):
        for el in kwargs:
            if type(kwargs[el]) in [np.ndarray]:
                kwargs[el] = cl.array.to_device(self.computegrid.queue,
                                                kwargs[el])
        return kwargs


class Sum(FunctionGPU):

    forward_src = """
FUNDEF void Sum
(
    __global const float *a, __global const float *b, __global float *res)
{
  int gid = get_global_id(0);
  res[gid] = a[gid] + b[gid];
}
"""

    adjoint_src = """
FUNDEF void Sum_adj(__global float *a_adj, 
                      __global float *b_adj, 
                      __global float *res_adj)
{
  int gid = get_global_id(0);
  a_adj[gid] += res_adj[gid];
  b_adj[gid] += res_adj[gid];
  res_adj[gid] = 0;
}
"""

    def __init__(self, grids=None, computegrid=None, **kwargs):
        self.required_states = ["a", "b", "res"]
        self.updated_states = ["res"]
        super().__init__(grids=grids, computegrid=computegrid, **kwargs)


class Gridtester(FunctionGPU):
    forward_src = """
FUNDEF void Gridtester(__global float *a, grid g)
{
    get_pos(&g);
    a[indg(g, 0, 0, 0)] = __ND__;
}
"""

    def __init__(self, grids=None, computegrid=None, **kwargs):
        self.required_states = ["a"]
        self.updated_states = ["a"]
        self.default_grids = {"a": "a"}
        super().__init__(grids=grids, computegrid=computegrid, **kwargs)


class DerivativeTester(FunctionGPU):
    forward_src = """
FUNDEF void DerivativeTester(__global float *a, grid g)
{
    get_pos(&g);
    LOCID float la[__LSIZE__];
    #if LOCAL_OFF==0
        load_local_in(g, a, la);
        load_local_haloz(g, a, la);
        load_local_halox(g, a, la);
        BARRIER
    #endif
    float ax = Dxm(g, la);
    gridstop(g);
    a[indg(g, 0, 0, 0)] = ax;

}
"""

    def __init__(self, grids=None, computegrid=None, fdcoefs=FDCoefficients(),
                 local_size=(16, 16), **kwargs):
        self.required_states = ["a"]
        self.updated_states = ["a"]
        self.default_grids = {"a": "a"}
        self.headers = fdcoefs.header()
        super().__init__(grids=grids, computegrid=computegrid,
                         local_size=local_size, **kwargs)


if __name__ == '__main__':

    # resc = ComputeRessource()
    # nel = 50000
    # a_np = np.random.rand(nel).astype(np.float32)
    # b_np = np.random.rand(nel).astype(np.float32)
    # res_np = np.zeros_like(a_np)
    #
    # grid = GridCL(resc.queues[0], shape=(nel,), pad=0, dtype=np.float32)
    # grids = {"a": grid,
    #               "b": grid,
    #               "res": grid}
    # sumknl = Sum(grids=grids)
    #
    # sumknl.linear_test()
    # sumknl.backward_test()
    # sumknl.dot_test()
    # states = sumknl({"a": a_np, "b": b_np, "res": res_np})
    # res_np = states["res"].get()
    #
    # # Check on CPU with Numpy:
    # print(res_np - (a_np + b_np))
    # print(np.linalg.norm(res_np - (a_np + b_np)))
    # assert np.allclose(res_np, a_np + b_np)

    resc = ComputeRessource()
    nx = 4
    nz = 8

    grid = GridCL(resc.queues[0], shape=(nz, nx), pad=0, dtype=np.float32)
    grids = {"a": grid}
    gridtest = Gridtester(grids=grids)
    states = gridtest()
    print(states["a"].get())
    print(states["a"].flags)

    resc = ComputeRessource()
    nx = 24
    nz = 12

    grid = GridCL(resc.queues[0], shape=(nz, nx), pad=2, dtype=np.float32, zero_boundary=True)
    a = np.tile(np.arange(nx), [nz, 1])
    grids = {"a": grid}
    dertest = DerivativeTester(grids=grids)
    states = dertest({"a": a})
    print(states["a"].get())
    print(states["a"].flags)
