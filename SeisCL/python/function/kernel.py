
import numpy as np
from SeisCL.python import VariableCL, Variable
from collections import OrderedDict
try:
    import pyopencl as cl
    from pyopencl.array import Array, to_device
    def compile(ctx, src, options):
        return cl.Program(ctx, src).build(options)
except ImportError:
    def compile(ctx, src, options):
        raise ImportError("pyopencl not installed")


cudacl = {"FUNDEF": {"opencl": "__kernel", "cuda": "__global__"},
          "LFUNDEF": {"opencl": "", "cuda": "__device__ __inline__"},
          "GLOBARG": {"opencl": "__global", "cuda": ""},
          "LOCID": {"opencl": "__local", "cuda": "__shared__"},
          "BARRIER": {"opencl": "barrier(CLK_LOCAL_MEM_FENCE);",
                      "cuda": "__syncthreads();"},
          }

options_def = ["-D LCOMM=0",
               "-D ABS_TYPE=0",
               "-D FREESURF=0",
               "-D DEVID=0",
               "-D MYLOCALID=0",
               "-D NUM_DEVICES=0",
               "-D NLOCALP=0"]

#TODO header creation is scattered all over the place, review
#TODO document the header options


class ComputeGrid:

    def __init__(self, queue, shape, origin, local=None):
        self.queue = queue
        self.shape = shape
        self.origin = origin

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        if len(origin) != len(self.shape):
            raise ValueError("origin and shape must have the same length")
        self._origin = to_device(self.queue, np.array(origin, dtype=np.int32))


class Kernel:

    def __init__(self, queue, name, signature, src, header, options,
                 mode, local_size=None, check_shape=False, platform="opencl",
                 include=None):

        self.queue = queue
        self.name = name
        self.signature = signature
        self.src = src
        self.compiled_src = None
        self.header = header
        self.options = options
        self.mode = mode
        self.local_size = local_size
        self.check_shape = check_shape
        self.platform = platform
        self.include = include
        self.event = None
        self._prog = None
        self._kernel = None
        self._kernal_variables_shape = None

    def __call__(self, grid: ComputeGrid, *args, **kwargs):

        a = self.signature.bind(*args, **kwargs)
        a.apply_defaults()
        arguments = a.arguments
        if "origin" in arguments:
            raise ValueError("origin is a reserved argument name")
        arguments["origin"] = grid.origin
        kernels = self.kernel(a.arguments, grid)

        #TODO remove the list of kernel capibility if applicable
        if "backpropagate" in kwargs:
            if kwargs["backpropagate"] == 1:
                kernels = kernels[::-1]

        for kernel in kernels:
            self.event = kernel(self.queue,
                                self.global_size(grid.shape),
                                self.local_size,
                                *self.arg_list(a.arguments))
        return self.event

    def arg_list(self, arguments):
        arg_list = []
        if "args" in arguments:
            newargs = OrderedDict()
            for el, var in arguments.items():
                if el == "args":
                    for ii, arg in enumerate(var):
                        newargs["arg%d"%ii] = arg
                else:
                    newargs[el] = var
            arguments = newargs
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
            elif isinstance(var, Array):
                arg_list.append(var.data)
            else:
                raise TypeError("Type not supported: %s for arg %s"
                                % (type(var), el))
        return arg_list

    def kernel(self, arguments, grid: ComputeGrid):

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

        if not self._kernel:
            argdef, arguments = argument_definition(arguments, self.mode,
                                                    self.platform)
            fundef = cudacl["FUNDEF"][self.platform]
            grid_struct, grid_filler = get_positional_headers(grid,
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
            grid_stop_header = grid_stopper(grid,
                                            use_local=self.local_size is not None)
            src = "\n".join([self.header,
                             grid_struct,
                             variables_headers,
                             grid_stop_header,
                             src])
            self.compiled_src = src
            if not self._prog:
                options = options_def + self.options
                #TODO redefine the EPS macro
                #options += ["-D __EPS__=%d" % self.grid.smallest]
                try:
                    self._prog = compile(self.queue.context, src, options)
                except Exception as e:
                    print(src)
                    raise e
            self._kernel = [getattr(self._prog, name) for name in names]
        return self._kernel

    def global_size(self, grid_size):
        if self.local_size is None:
            return grid_size
        else:
            return [np.int32(np.ceil(s/float(l)) * l)
                    for s, l in zip(grid_size, self.local_size)]


def get_positional_headers(grid: ComputeGrid, local_size: tuple):

    nd = len(grid.shape)
    grid_struct_header = """typedef struct grid{\n"""
    if nd == 3:
        names = ["z", "y", "x"]
    elif nd == 2:
        names = ["z", "x"]
    elif nd == 1:
        names = ["z"]
    else:
        raise ValueError("global_size must be a list of length 1, 2 or 3")

    for name in names:
        grid_struct_header += "    int %s;\n" % name
    if local_size is not None:
        for name in names:
            grid_struct_header += "    int l%s;\n" % name
        for name in names:
            grid_struct_header += "    int nl%s;\n" % name

    grid_struct_header += "} grid;\n"

    grid_filler = "    grid g;\n"

    for ii, name in enumerate(names):
        grid_filler += "    g.%s = get_global_id(%d) + origin[%d];\n" \
                       % (name, ii, ii)

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
        if len(shape) > 1:
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
    if "args" in arguments:
        newargs = OrderedDict()
        for el, var in arguments.items():
            if el == "args":
                for ii, arg in enumerate(var):
                    newargs["arg%d"%ii] = arg
            else:
                newargs[el] = var
        arguments = newargs

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
        elif isinstance(var, np.ndarray) or isinstance(var, Array):
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

        else:
            raise TypeError("Type not supported: %s" % type(var))
    argdef[-1] = argdef[-1][:-3]
    return argdef, arguments


def grid_stopper(grid: ComputeGrid, use_local=True, with_offset=False):

    shape = grid.shape
    if use_local:
        if with_offset:
            offstr = "- offset "
        else:
            offstr = ""

        src = """
        #define gridstop(p)\\
        do{\\
            if (((p).z - origin[0]) >= %d || ((p).y - origin[1]) >= %d || ((p).x %s- origin[2]) >= %d){\\
            return;}\\
            } while(0)\n
        """.strip() + "\n"
        if len(shape) == 1:
            src = src.replace("|| ((p).y - origin[1]) >= %d || ((p).x %s- origin[2]) >= %d", "")
            src = src % shape[0]
        elif len(shape) == 2:
            src = src.replace("|| ((p).y - origin[1]) >= %d", "")
            src = src.replace("[2]", "[1]")
            src = src % (shape[0], offstr, shape[1])
        elif len(shape) == 3:
            src = src % (shape[0], shape[1], offstr, shape[2])
        else:
            raise ValueError("Shape must be 1, 2 or 3 dimensional")
    else:
        src = "#define gridstop(p) \n"
    src = src.strip() + "\n"
    return src