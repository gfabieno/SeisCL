
import numpy as np
from SeisCL.python import Variable

try:
    import pyopencl as cl
    def compile(ctx, src, options):
        return cl.Program(ctx, src).build(options)
except ImportError:
    def compile(ctx, src, options):
        raise ImportError("pyopencl not installed")


cudacl = {"FUNDEF": {"opencl": "__kernel", "cuda": "__global__"},
          "LFUNDEF": {"opencl": "", "cuda": "__device__ __inline__"},
          "GLOBARG": {"opencl": "__global", "cuda": ""},
          "LOCID": {"opencl": "__local", "cuda": "__shared__"},
          }

options_def = ["-D LCOMM=0",
               "-D ABS_TYPE=0",
               "-D FREESURF=0",
               "-D DEVID=0",
               "-D MYLOCALID=0",
               "-D NUM_DEVICES=0",
               "-D NLOCALP=0"]


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
            if isinstance(var, Variable):
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
                self._prog = compile(self.queue.context, src, options)
            self._kernel = [getattr(self._prog, name) for name in names]
        return self._kernel

    def global_size(self, grid_size):
        if self.local_size is None:
            return grid_size
        else:
            return [np.int32(np.ceil(s/float(l)) * l)
                    for s, l in zip(grid_size, self.local_size)]


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
        if isinstance(var, Variable):
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