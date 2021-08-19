
import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
from SeisCL.python.seis2D import Grid, StateKernel, State, Propagator, Sequence, ReversibleKernel
import re
from SeisCL.python.FDstencils import get_pos_header, FDCoefficients, CUDACL_header, grid_stop_header


options_def = ["-D LCOMM=0",
               "-D ABS_TYPE=0",
               "-D FREESURF=0",
               "-D DEVID=0",
               "-D MYLOCALID=0",
               "-D NUM_DEVICES=0",
               "-D NLOCALP=0"]


class GridCL(Grid):

    backend = cl.array.Array

    def __init__(self, queue, shape=(10, 10), pad=2, dtype=np.float32,
                 zero_boundary=False, **kwargs):
        super().__init__(shape=shape, pad=pad, zero_boundary=zero_boundary,
                         dtype=dtype)
        self.queue = queue
        self._struct_np, self.struct_c = self.define_struct()

    def zero(self):
        return cl.array.zeros(self.queue, self.shape, self.dtype, order="F")

    def random(self):
        if self.zero_boundary:
            state = np.zeros(self.shape, dtype=self.dtype)
            state[self.valid] = np.random.rand(*state[self.valid].shape)*10e6
            state = cl.array.to_device(self.queue, np.require(state, requirements='F'))
        else:
            state = cl.clrandom.rand(self.queue, self.shape, self.dtype)
        return state

    def assign_data(self, data):
        return cl.array.to_device(self.queue,
                                  np.require(data, requirements='F'))

    def define_struct(self):
        dtype = np.dtype([
            ("NX", np.int32),
            ("NY", np.int32),
            ("NZ", np.int32),
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
                "-D __ND__=%d" % len(self.shape),
                "-D __LOCAL_OFF__=%d" % 0]

    @property
    def struct_np(self):
        struct = np.zeros(1, self._struct_np)
        struct[0]["NX"] = self.shape[-1]
        struct[0]["NZ"] = self.shape[0]
        if len(self.shape) == 3:
            struct[0]["NY"] = self.shape[1]
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

    def __init__(self, queue, grid, global_size, local_size,  src, name, options,
                 default_args):

        self.queue = queue
        self.grid = grid
        self.global_size = global_size
        self.local_size = local_size
        self.src = src
        self.name = name
        self.options = options
        self.args = self.extract_args(self.src)
        self._prog = None
        self._kernel = None
        self.event = None
        self.default_args = default_args

    def __call__(self, *args, **kwargs):
        if self.default_args:
            for key in self.default_args:
                if key not in kwargs:
                    kwargs[key] = self.default_args[key]
        arg_list = self.assign_args(*args, **kwargs)
        self.event = self.kernel(self.queue,
                                 self.global_size(),
                                 self.local_size,
                                 *arg_list)

        return args[0]

    @property
    def kernel(self):
        if not self._kernel:
            if not self._prog:
                options = options_def + self.options + self.grid.options
                if self.local_size is not None:
                    options += ["-D __LSIZE__=%d" % np.prod(self.local_size)]
                self._prog = cl.Program(self.queue.context,
                                        self.src).build(options)
            self._kernel = getattr(self._prog, self.name)
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
                arg_list[ii] = args[0][a["name"]]
            elif a["name"] in indict:
                arg_list[ii] = indict[a["name"]]
            else:
                arg_list[ii] = None
            if type(arg_list[ii]) == cl.array.Array:
                arg_list[ii] = arg_list[ii].data
            elif type(arg_list[ii]) == int:
                arg_list[ii] = np.int32(arg_list[ii])
            elif type(arg_list[ii]) == float:
                arg_list[ii] = np.float32(arg_list[ii])
        return arg_list


class StateKernelGPU(StateKernel):

    forward_src = ""
    linear_src = ""
    adjoint_src = ""

    def __init__(self, grids=None, grid=None, local_size=None,
                 options=None, default_args=None, **kwargs):
        super().__init__(grids, **kwargs)

        if not hasattr(self, "headers"):
            self.headers = ""
        self.grid = grid
        self.local_size = local_size
        if grid is None:
            if self.updated_states:
                self.grid = self.grids[self.updated_states[0]]
            else:
                self.grid = self.grids[self.required_states[0]]
        if options is None:
            options = []

        name = self.__class__.__name__
        headers = CUDACL_header + self.grid.headers + self.headers
        self.forward = Kernel(self.grid.queue,
                              self.grid,
                              self.global_size_fw,
                              self.local_size,
                              headers + self.forward_src,
                              name,
                              options,
                              default_args)
        if self.linear_src:
            self.linear = Kernel(self.grid.queue,
                                 self.grid,
                                 self.global_size_fw,
                                 self.local_size,
                                 headers + self.linear_src,
                                 name + "_lin",
                                 options,
                                 default_args)
        else:
            self.linear = Kernel(self.grid.queue,
                                 self.grid,
                                 self.global_size_fw,
                                 self.local_size,
                                 headers + self.forward_src,
                                 name,
                                 options,
                                 default_args)
            for ii in range(len(self.linear.args)):
                if self.linear.args[ii]["scope"] == "global":
                    self.linear.args[ii]["bp"] = "linear"
        if self.adjoint_src:
            self.adjoint = Kernel(self.grid.queue,
                                  self.grid,
                                  self.global_size_adj,
                                  self.local_size,
                                  headers + self.adjoint_src,
                                  name + "_adj",
                                  options,
                                  default_args)

    def global_size_fw(self):
        if self.local_size is None:
            return [np.int32(s-2*self.grid.pad) for s in self.grid.shape]
        else:
            return [np.int32(np.ceil((s-2*self.grid.pad)/float(l))*l)
                    for s, l in zip(self.grid.shape, self.local_size)]

    def global_size_adj(self):
        return self.global_size_fw()


class PropagatorCL(Propagator, StateKernelGPU):
    pass


class SequenceCL(Sequence, StateKernelGPU):
    pass


class ReversibleKernelCL(ReversibleKernel, StateKernelGPU):
    pass


class Sum(StateKernelGPU):

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

    def __init__(self, grids=None, grid=None, **kwargs):
        self.required_states = ["a", "b", "res"]
        self.updated_states = ["res"]
        super().__init__(grids=grids, grid=grid, **kwargs)


class Gridtester(StateKernelGPU):
    forward_src = """
FUNDEF void Gridtester(__global float *a, grid g)
{
    get_pos(&g);
    a[indg(g, 0, 0, 0)] = __ND__;
}
"""

    def __init__(self, grids=None, grid=None, **kwargs):
        self.required_states = ["a"]
        self.updated_states = ["a"]
        super().__init__(grids=grids, grid=grid, **kwargs)


class DerivativeTester(StateKernelGPU):
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
    a[indg(g, 0, 0, 0)] = ax;
    a[indg(g, 0, 0, 0)] = Dxm(g,la);
    gridstop(g);
}
"""

    def __init__(self, grids=None, grid=None, fdcoefs=FDCoefficients(),
                 local_size=(16, 16), **kwargs):
        self.required_states = ["a"]
        self.updated_states = ["a"]
        self.headers = fdcoefs.header()
        super().__init__(grids=grids, grid=grid,
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
    nz = 24

    grid = GridCL(resc.queues[0], shape=(nz, nx), pad=4, dtype=np.float32, zero_boundary=True)
    a = np.tile(np.arange(nx), [12, 1])
    grids = {"a": grid}
    dertest = DerivativeTester(grids=grids)
    states = dertest({"a": a})
    print(states["a"].get())
    print(states["a"].flags)
