
import numpy as np
import unittest
import pyopencl as cl
import pyopencl.array
from SeisCL.python.tape.tape import Variable, Function, ReversibleFunction
from SeisCL.python.FDstencils import FDCoefficients

options_def = ["-D LCOMM=0",
               "-D ABS_TYPE=0",
               "-D FREESURF=0",
               "-D DEVID=0",
               "-D MYLOCALID=0",
               "-D NUM_DEVICES=0",
               "-D NLOCALP=0"]

cudacl = {"FUNDEF": {"opencl": "__kernel", "cuda": "__global__"},
          "LFUNDEF": {"opencl": "", "cuda": "__device__ __inline__"},
          "GLOBARG": {"opencl": "__global", "cuda": ""},
          "LOCID": {"opencl": "__local", "cuda": "__shared__"},
          }

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

# TODO review how dimensions names are assigned
# TODO migration to pycuda, there is no match_dtype_to_c_struct
class VariableCL(Variable):

    backend = cl.array.Array

    def __init__(self, queue, data=None, shape=None, lin=None, grad=None,
                 initialize_method="zero", dtype=np.float32,
                 pad=None, differentiable=True):
        self.queue = queue
        self.copy_array = CopyArrayCL(queue.context)
        super().__init__(data=data, shape=shape, lin=lin, grad=grad,
                         initialize_method=initialize_method, dtype=dtype,
                         pad=pad, differentiable=differentiable)

    def empty(self):
        return cl.array.empty(self.queue, self.shape, self.dtype, order="F")

    def ones(self):
        return cl.array.zeros(self.queue, self.shape, self.dtype, order="F") + 1

    def zero(self):
        return cl.array.zeros(self.queue, self.shape, self.dtype, order="F")

    def todevice(self, data):
        if type(data) is np.ndarray:
            data = cl.array.to_device(self.queue, data)
        elif type(data) is not cl.array.Array:
            raise ValueError("Data type not supported: should be np.ndarray "
                             "or cl.array.Array")
        return data


class Kernel:

    def __init__(self, queue, name, signature, src, header, options,
                 mode, local_size=None, check_shape=False, platform="opencl"):

        self.queue = queue
        self.name = name
        self.signature = signature
        self.src = src
        self.header = header
        self.options = options
        self.mode = mode
        self.local_size = local_size
        self.check_shape = check_shape
        self.platform = platform
        self.event = None
        self._prog = None
        self._kernel = None
        self._kernal_variables_shape = None

    def __call__(self, grid_size, *args, **kwargs):
        a = self.signature.bind(*args, **kwargs)
        a.apply_defaults()
        kernels = self.kernel(a.arguments, grid_size)

        #TODO remove the list of kernel capibility if applicable
        if "backpropagate" in kwargs:
            if kwargs["backpropagate"] == 1:
                kernels = kernels[::-1]

        for kernel in kernels:
            self.event = kernel(self.queue,
                                self.global_size(grid_size),
                                self.local_size,
                                *self.arg_list(a.arguments))
        return self.event

    def arg_list(self, arguments):
        arg_list = []
        for el, var in arguments.items():
            if isinstance(var, VariableCL):
                arg_list.append(var.data.data)
                if self.mode == "linear":
                    arg_list.append(var.lin.data)
                if self.mode == "adjoint":
                    arg_list.append(var.grad.data)
            elif isinstance(var, int) or isinstance(var, np.int32):
                arg_list.append(np.int32(var))
            elif isinstance(var, float) or isinstance(var, np.float32):
                arg_list.append(np.float32(var))
            elif isinstance(var, bool) or isinstance(var, np.bool):
                arg_list.append(np.int32(var))
            else:
                raise TypeError("Type not supported: %s for arg %s"
                                % (type(var), el))
        return arg_list

    def kernel(self, arguments, grid_size):

        if self._kernel and self.check_shape:
            variables = {name: var for name, var in arguments.items()
                         if isinstance(var, Variable)}
            for name, var in variables.items():
                if self._kernel_variables_shape[name] != var.shape:
                    Warning("Shape of variable %s has changed from %s to %s, "
                            "recompiling kernel"
                            % (name, self._kernel_variables_shape[name],
                               var.shape))
                    self._kernel = None
                    break
        #TODO add grid stopper if applicable
        if not self._kernel:
            argdef = argument_definition(arguments, self.mode, self.platform)
            fundef = cudacl["FUNDEF"][self.platform]
            grid_struct, grid_filler = get_positional_headers(grid_size,
                                                              self.local_size)
            variables = {name: var for name, var in arguments.items()
                         if isinstance(var, Variable)}
            variables_headers = get_variable_headers(variables, self.mode)
            if isinstance(self.src, str):
                src = [self.src]
            else:
                src = self.src
            names = ["%s" % self.name + str(ii)
                     for ii in range(len(src))]

            for name, s in zip(names, src):
                srci = "%s void %s(" % (fundef, name)
                indent = len(srci)*" "
                for ii, el in enumerate(argdef):
                    if ii > 0:
                        argdef[ii] = indent + el
                srci += "".join(argdef)
                srci += "){\n"
                srci += grid_filler
                srci += "\n".join(["    " + el.strip() for el in s.split("\n")])
                srci += "\n}"
                src[src.index(s)] = srci
            src = "\n".join(src)
            src = self.header + grid_struct + variables_headers + src
            if not self._prog:
                options = options_def + self.options

                #TODO redefine the EPS macro
                #options += ["-D __EPS__=%d" % self.grid.smallest]
                if self.local_size is not None:
                    total = np.prod([el+self.grid.pad*2
                                     for el in self.local_size])
                    options += ["-D __LSIZE__=%d" % total]
                self._prog = cl.Program(self.queue.context, src).build(options)
            self._kernel = [getattr(self._prog, name) for name in names]
        return self._kernel

    def global_size(self, grid_size):
        if self.local_size is None:
            return grid_size
        else:
            return [np.int32(np.ceil(s/float(l)) * l)
                    for s, l in zip(grid_size, self.local_size)]


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
    """
    Base class for all GPU functions. As for `Function`, override the methods
    `forward`, `linear`  and `adjoint`. You should define the input and output
    signatures, write the body of the GPU kernel as a string and pass it to
    the method `gpukernel` to launch the kernel on GPU.
    """

    def __init__(self, queue, local_size=None, options=None):

        self.queue = queue
        self.local_size = local_size
        if options is None:
            options = []
        self.options = options
        #TODO clean up header creation
        self.header = ""
        self._gpukernel = {"forward": None, "linear": None, "adjoint": None}
        super().__init__()

    def gpukernel(self, src, mode, grid_size, *args, **kwargs):
        """
        Launch the GPU kernel. The kernel is compiled and cached the first time
        it is called. The kernel is then launched with the arguments passed to
        this method.

        :param src: String containing the body of the kernel. If a List of
                    string is passed, multiple kernels are compiled and
                    launched in order.
        :param mode: Either 'forward', 'linear' or 'adjoint'
        :param grid_size: The size of the compute grid to use for the kernel.
        :param args: The list of arguments to pass to the kernel. Must contain
                     only `Variable` objects, int, float or bool.
        :param kwargs: Same as args, but passed as keyword arguments.
        """
        if self._gpukernel[mode] is None:
            if mode not in ["forward", "linear", "adjoint"]:
                raise ValueError("mode must be 'forward', 'linear' or "
                                 "'adjoint'")
            name = self.__class__.__name__ + "_" + mode

            self._gpukernel[mode] = Kernel(self.queue,
                                           name,
                                           self.signature,
                                           src,
                                           self.header,
                                           self.options,
                                           mode,
                                           self.local_size
                                           )
        self._gpukernel[mode](grid_size, *args, **kwargs)
    # def cache_states(self, states):
    #     for el in self.updated_states:
    #         if el in self.updated_regions:
    #             for ii, region in enumerate(self.updated_regions[el]):
    #                 self.grids[el].copy_array(self._forward_states[el][ii][..., self.ncall],
    #                                           states[el][region])
    #         else:
    #             self._forward_states[el][..., self.ncall] = states[el]
    #     self.ncall += 1


class ReversibleFunctionCL(ReversibleFunction, FunctionGPU):
    pass


def get_positional_headers(global_size, local_size):

    grid_struct_header = """typedef struct grid{\n"""
    if len(global_size) == 3:
        names = ["z", "y", "x"]
    elif len(global_size) == 2:
        names = ["z", "x"]
    elif len(global_size) == 1:
        names = ["x"]
    else:
        raise ValueError("global_size must be a list of length 1, 2 or 3")

    for name in names:
        grid_struct_header += "    int %s;\n" % name
    if local_size is not None:
        for name in names:
            grid_struct_header += "    int l%s{};\n" % name
        for name in names:
            grid_struct_header += "    int lsize%s{}\n;" % name

    grid_struct_header += "} grid;\n"

    grid_filler = "    grid g;\n"
    for ii, name in enumerate(names):
        grid_filler += "    g.%s = get_global_id(%d);\n" % (name, ii)
    if local_size is not None:
        for ii, name in enumerate(names):
            grid_filler += "    g.l%s = get_local_id(%d);\n" % (name, ii)
        for ii, name in enumerate(names):
            grid_filler += "    g.lsize%s = get_local_size(%d);\n" % (name, ii)

    return grid_struct_header, grid_filler


def get_variable_headers(variables, mode):

    shapes = {}
    for name, var in variables.items():
        if var.shape not in shapes:
            shapes[var.shape] = {name: var}
        else:
            shapes[var.shape][name] = var
    header = ""
    for shape, vars in shapes.items():
        varii = "%s(" + ", ".join(["x%d"%ii for ii in range(len(shape))]) + ") "
        varii += "%s[" + " + ".join(["(x%d) * %d" % (len(shape) - 1 - ii,
                                                   np.prod(shape[:-ii-1]))
                                     for ii in range(len(shape)-1)])
        if len(shape) >1:
            varii += "+ (x0)]\n"
        else:
            varii += "(x0)]\n"
        for name, var in vars.items():
            header += "#define " + varii % (name, name)
            if mode == "linear":
                header += "#define " + varii % (name + "_lin", name + "_lin")
            if mode == "adjoint":
                header += "#define " + varii % (name + "_adj", name + "_adj")
    return header


def argument_definition(arguments, mode, platform):
    argdef = []
    for el, var in arguments.items():
        if isinstance(var, VariableCL):
            if var.dtype == np.float32:
                dtype = "float"
            elif var.dtype == np.int32:
                dtype = "int"
            elif var.dtype == np.float64:
                dtype = "double"
            elif var.dtype == np.int64:
                dtype = "long"
            elif var.dtype == np.float16:
                dtype = "half"
            else:
                raise TypeError("Type not supported for %s with type %s"
                                % (el, var.dtype))
            globarg = cudacl["GLOBARG"][platform]
            argdef.append("%s %s * %s, \n" % (globarg, dtype, el))
            if mode == "linear":
                argdef.append("%s %s * %s_lin, \n" % (globarg, dtype, el))
            if mode == "adjoint":
                argdef.append("%s %s * %s_adj, \n" % (globarg, dtype, el))
        elif isinstance(var, int) or isinstance(var, np.int32):
            argdef.append("int %s, \n" % (el))
        elif isinstance(var, float) or isinstance(var, np.float32):
            argdef.append("float %s, \n" % (el))
        elif isinstance(var, bool) or isinstance(var, np.bool):
            argdef.append("int %s, \n" % (el))
        else:
            raise TypeError("Type not supported: %s" % type(var))
    argdef[-1] = argdef[-1][:-3]
    return argdef


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


class ComputeRessourceTester(unittest.TestCase):

    def test_opencl_ressources(self):
        resc = ComputeRessource()
        self.assertIsNotNone(resc.devices)


class VariableTester(unittest.TestCase):

    def test_variable_in_device(self):
        resc = ComputeRessource()
        data = np.random.rand(10, 10)
        lin = np.random.rand(10, 10)
        grad = np.random.rand(10, 10)
        var = VariableCL(resc.queues[0], data=data, lin=lin, grad=grad)
        self.assertIsInstance(var.data, cl.array.Array)
        self.assertIsInstance(var.lin, cl.array.Array)
        self.assertIsInstance(var.grad, cl.array.Array)
        self.assertTrue(np.allclose(var.data.get(), data))
        self.assertTrue(np.allclose(var.lin.get(), lin))
        self.assertTrue(np.allclose(var.grad.get(), grad))

    def test_copy_variable(self):
        from copy import copy, deepcopy
        resc = ComputeRessource()
        data = np.random.rand(10, 10)
        lin = np.random.rand(10, 10)
        grad = np.random.rand(10, 10)
        var = VariableCL(resc.queues[0], data=data, lin=lin, grad=grad)
        var2 = deepcopy(var)
        self.assertIsInstance(var2.data, cl.array.Array)
        self.assertIsInstance(var2.lin, cl.array.Array)
        self.assertIsInstance(var2.grad, cl.array.Array)
        self.assertTrue(np.allclose(var2.data.get(), data))
        self.assertTrue(np.allclose(var2.lin.get(), lin))
        self.assertTrue(np.allclose(var2.grad.get(), grad))
        self.assertIsNot(var2.data, var.data)
        self.assertIsNot(var2.lin, var.lin)
        self.assertIsNot(var2.grad, var.grad)
        var3 = copy(var)
        self.assertIs(var3.data, var.data)
        self.assertIs(var3.lin, var.lin)
        self.assertIs(var3.grad, var.grad)


class FunctionGpuTester(unittest.TestCase):

    def get_fun(self, ndim):
        
        class Sum(FunctionGPU):
            def forward(self, a, b, c):
                src = """
                      c(%s) = a(%s) + b(%s);
                      """
                if len(a.shape) == 1:
                    src = src % (("g.x", )*3)
                elif len(a.shape) == 2:
                    src = src % (("g.z, g.x", )*3)
                elif len(a.shape) == 3:
                    src = src % (("g.z, g.y, g.x", )*3)
                self.gpukernel(src, "forward", a.shape, a, b, c)
                return c

            def linear(self, a, b, c):
                src = """
                      c_lin(%s) = a_lin(%s) + b_lin(%s);
                      """
                if len(a.shape) == 1:
                    src = src % (("g.x", )*3)
                elif len(a.shape) == 2:
                    src = src % (("g.z, g.x", )*3)
                elif len(a.shape) == 3:
                    src = src % (("g.z, g.y, g.x", )*3)
                self.gpukernel(src, "linear", a.shape, a, b, c)
                return c

            def adjoint(self, a, b, c):
                src = """
                      a_adj(%s) += c_adj(%s);
                      b_adj(%s) += c_adj(%s);
                      c_adj(%s) = 0;
                      """
                if len(a.shape) == 1:
                    src = src % (("g.x", )*5)
                elif len(a.shape) == 2:
                    src = src % (("g.z, g.x", )*5)
                elif len(a.shape) == 3:
                    src = src % (("g.z, g.y, g.x", )*5)
                self.gpukernel(src, "adjoint", a.shape, a, b, c)
                return a, b
        resc = ComputeRessource()
        sum = Sum(resc.queues[0])
        shape = (3,)*ndim
        a = VariableCL(resc.queues[0],
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape))
        b = VariableCL(resc.queues[0],
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape))
        c = VariableCL(resc.queues[0],
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape))
        return sum, a, b, c

    def test_forward(self):

        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            c_np = a.data.get() + b.data.get()
            c = sum(a, b, c)
            self.assertTrue(np.allclose(c_np, c.data.get()))

    def test_linear(self):
        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            c_lin_np = a.lin.get() + b.lin.get()
            c_lin = sum(a, b, c, mode="linear")
            self.assertTrue(np.allclose(c_lin_np, c_lin.lin.get()))

    def test_dottest(self):
        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            self.assertLess(sum.dot_test(a, b, c), 1e-6)

    def test_backward_test(self):
        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            self.assertLess(sum.backward_test(a, b, c), 1e-12)

    def test_linear_test(self):
        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            self.assertLess(sum.linear_test(a, b, c), 1e-5)

