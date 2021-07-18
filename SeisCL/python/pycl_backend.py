
import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from SeisCL.python.seis2D import Grid, StateKernel, State
import re


class GridCL(Grid):

    backend = cl.array.Array

    def __init__(self, queue, shape=(10, 10), pad=2, dtype=np.float32, **kwargs):
        self.shape = shape
        self.pad = pad
        self.valid = tuple([slice(self.pad, -self.pad)] * len(shape))
        self.dtype = dtype
        self.eps = np.finfo(dtype).eps
        self.queue = queue
        self.local_size = None

    @property
    def global_size(self):
        if self.local_size is None:
            return [s-2*self.pad for s in self.shape]
        else:
            return [np.ceil((s-2*self.pad)/float(l))*l
                    for s, l in zip(self.shape, self.local_size)]

    def zero(self):
        return cl.array.zeros(self.queue, self.shape, self.dtype)

    def random(self):
        return cl.clrandom.rand(self.queue, self.shape, self.dtype)

    def assign_data(self, data):
        return cl.array.to_device(self.queue, data)

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

    def __init__(self, grid, src, name):

        self.grid = grid
        self.src = src
        self.name = name
        self.args = self.extract_args(self.src)
        self._prog = None
        self._kernel = None
        self.event = None

    def __call__(self, *args, **kwargs):
        arg_list = self.assign_args(*args, **kwargs)
        self.event = self.kernel(self.grid.queue,
                                 self.grid.global_size,
                                 self.grid.local_size,
                                 *arg_list)

        return args[0]

    @property
    def kernel(self):
        if not self._kernel:
            if not self._prog:
                self._prog = cl.Program(self.grid.queue.context, self.src).build()
            self._kernel = getattr(self._prog, self.name)
        return self._kernel

    @staticmethod
    def extract_args(src):

        m = re.search("__kernel void \w*\s*\((.*?)\)", src, re.DOTALL)
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
            if "__global" in a[0]:
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
            if a["bp"] == "linear" or a["bp"] == "adjoint":
                arg_list[ii] = args[0][a["name"]]
            else:
                arg_list[ii] = indict[a["name"]]
            if type(arg_list[ii]) == cl.array.Array:
                arg_list[ii] = arg_list[ii].data
        return arg_list


class StateKernelGPU(StateKernel):

    forward_src = ""
    linear_src = ""
    adjoint_src = ""

    def __init__(self, state_defs=None, grid=None, **kwargs):
        super().__init__(state_defs, **kwargs)

        if grid is None:
            grid = self.state_defs[self.updated_states[0]].grid
        self.grid = grid

        name = self.__class__.__name__
        self.forward = Kernel(grid, self.forward_src, name)
        if self.linear_src:
            self.linear = Kernel(grid, self.linear_src, name + "_lin")
        else:
            self.linear = Kernel(grid, self.forward_src, name)
            for ii in range(len(self.linear.args)):
                self.linear.args[ii]["bp"] = "linear"
        self.adjoint = Kernel(grid, self.adjoint_src, name + "_adj")


class Sum(StateKernelGPU):

    forward_src = """
__kernel void Sum
(
    __global const float *a, __global const float *b, __global float *res)
{
  int gid = get_global_id(0);
  res[gid] = a[gid] + b[gid];
}
"""

    adjoint_src = """
__kernel void Sum_adj(__global float *a_adj, 
                      __global float *b_adj, 
                      __global float *res_adj)
{
  int gid = get_global_id(0);
  a_adj[gid] += res_adj[gid];
  b_adj[gid] += res_adj[gid];
  res_adj[gid] = 0;
}
"""

    def __init__(self, state_defs=None, grid=None, **kwargs):
        self.required_states = ["a", "b", "res"]
        self.updated_states = ["res"]
        super().__init__(state_defs=state_defs, grid=grid, **kwargs)


if __name__ == '__main__':

    resc = ComputeRessource()
    nel = 50000
    a_np = np.random.rand(nel).astype(np.float32)
    b_np = np.random.rand(nel).astype(np.float32)
    res_np = np.zeros_like(a_np)

    grid = GridCL(resc.queues[0], shape=(nel,), pad=0, dtype=np.float32)
    state_defs = {"a": State("a", grid=grid),
                  "b": State("b", grid=grid),
                  "res": State("res", grid=grid)}
    sumknl = Sum(state_defs=state_defs)

    sumknl.linear_test()
    sumknl.backward_test()
    sumknl.dot_test()
    states = sumknl({"a": a_np, "b": b_np, "res": res_np})
    res_np = states["res"].get()

    # Check on CPU with Numpy:
    print(res_np - (a_np + b_np))
    print(np.linalg.norm(res_np - (a_np + b_np)))
    assert np.allclose(res_np, a_np + b_np)

