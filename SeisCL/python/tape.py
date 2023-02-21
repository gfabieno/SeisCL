
#TODO cache data in Tape in memory pool
#TODO eager and compiled mode
from inspect import signature, Parameter, _empty
import numpy as np
from copy import copy, deepcopy

class TapeHolder:
    tape = None


class Tape:
    """
    Keeps track of function calls as well as variables.
    """
    def __init__(self, cache_size=1):
        self.variables = {} # keep track of all encountered variables.
        self.graph = []
        self.previous_tape = TapeHolder.tape
        self.mode = "forward"
        self.cache_size = cache_size
        TapeHolder.tape = self

    # TODO use memory pooling to improve performance
    def append(self, kernel, *args, **kwargs):
        initial_states = kernel.cache_states(*args, **kwargs)
        self.graph.append((kernel, args, kwargs, initial_states))

    def pop(self):
        kernel, args, kwargs, initial_states = self.graph.pop()
        kernel.recover_states(initial_states, *args, **kwargs)
        return kernel, args, kwargs

    def add_variable(self, var):
        if var.name in self.variables:
            raise NameError("Variable name already exists in tape")
        self.variables[var.name] = var

    def empty(self):
        self.variables = {}
        self.graph = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TapeHolder.tape = self.previous_tape


TapeHolder.tape = Tape()


# class Variable(TapeHolder):
#
#     def __init__(self, name):
#         self.name = name
#         self.tape.add_variable(self)
#         self.data = None # contains the data
#         self.lin = None # contains the small data perturbation
#         self.grad = None # contains the gradient
#         self.last_update = None # The last kernel that updated the state


class Variable(TapeHolder):
    """

    """
    def __init__(self, name=None, data=None, shape=None, lin=None, grad=None,
                 initialize_method="zero", dtype=np.float,
                 pad=None):

        self.name = name
        # TODO do we need to track all variables ?
        #self.tape.add_variable(self)
        self.data = data
        if data is None:
            if shape is None:
                raise ValueError("Either data or shape must be provided")
            else:
                self.shape = shape
        else:
            self.shape = data.shape
        self.lin = lin # contains the small data perturbation
        self.grad = grad # contains the gradient
        self.last_update = None # The last kernel that updated the state
        self.initialize_method = initialize_method
        self.dtype = dtype
        self.smallest = np.nextafter(dtype(0), dtype(1))
        self.eps = np.finfo(dtype).eps
        self.pad = pad

    @property
    def data(self):
        if self._data is None:
            self._data = self.initialize()
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def lin(self):
        if self._lin is None:
            self._lin = self.initialize()
        return self._lin

    @lin.setter
    def lin(self, lin):
        self._lin = lin

    @property
    def grad(self):
        if self._grad is None:
            self._grad = self.initialize(method="ones")
        return self._grad

    @grad.setter
    def grad(self, grad):
        self._grad = grad

    @property
    def valid(self):
        if self.pad is None:
            return Ellipsis
        else:
            return tuple([slice(self.pad, -self.pad)] * len(self.shape))

    def initialize(self, method=None):
        if method is None:
            method = self.initialize_method
        if method == "zero":
            return self.zero()
        elif method == "random":
            return self.random()
        elif method == "ones":
            return self.ones()

    def ones(self):
        return np.ones(self.shape, dtype=self.dtype, order="F")

    def zero(self):
        return np.zeros(self.shape, dtype=self.dtype, order="F")

    def random(self):
        if self.pad:
            state = np.zeros(self.shape, dtype=self.dtype, order="F")
            state[self.valid] = np.random.rand(*state[self.valid].shape)*10e6
        else:
            state = np.random.rand(*self.shape).astype(self.dtype)
        return np.require(state, requirements='F')

    #TODO create cache in tape
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
                cache.append(np.empty(shape + [cache_size], dtype=self.dtype,
                                      order="F"))
            return cache
        else:
            return np.empty(list(self.shape) + [cache_size],
                            dtype=self.dtype, order="F")


#TODO Merge Function from seis2D and tape.py
class Function(TapeHolder):
    """
    Transforms States and its operation to the tape, if required.
    """

    mode = "forward"

    def __init__(self):
        self.signature = signature(self.forward)
        self.required_states = [name for name, par
                                in signature(self.forward).parameters.items()
                                if par.kind == Parameter.POSITIONAL_OR_KEYWORD]
        self.updated_states = []

    def __call__(self, *args, mode=None, **kwargs):
        if mode is None:
            mode = self.tape.mode
        if mode == "forward" or mode == "linear":
            self.tape.append(self, *args, **kwargs)
            if mode == "linear":
                self.linear(*args, **kwargs)
            return self.forward(*args, **kwargs)
        elif mode == "adjoint":
            self.tape.pop()
            return self.adjoint(*args, **kwargs)
        else:
            raise ValueError("Tape.mode should be in [\'forward\', \'linear\',"
                             " \'adjoint\']")

    def cache_states(self, *args, **kwargs):
        if not self.required_states and not self.updated_states:
            self.updated_states = self.arguments(*args, **kwargs).keys()
        vars = self.arguments(*args, **kwargs)
        initial_states = {}
        for el in self.updated_states:
            regions = self.updated_regions(vars[el])
            if regions:
                initial_states[el] = []
                for ii, region in enumerate(regions):
                    initial_states[el].append(vars[el].data[region])
            else:
                initial_states[el] = copy(vars[el].data)
        return initial_states

    def recover_states(self, initial_states, *args, **kwargs):
        vars = self.arguments(*args, **kwargs)
        for el in initial_states:
            regions = self.updated_regions(vars[el])
            if not regions:
                vars[el].data = initial_states[el]
            else:
                for ii, region in enumerate(regions):
                    vars[el].data[region] = initial_states[el][ii]

    def updated_regions(self, var):
        return {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def linear(self, *args, **kwargs):
        raise NotImplementedError

    def adjoint(self,  *args, **kwargs):
        raise NotImplementedError

    def backward_test(self, *args, verbose=True, **kwargs):

        vars = self.arguments(*args, **kwargs)
        vars0 = {name: copy(var) for name, var in vars.items()}
        for name, var in vars.items():
            if type(var) is Variable:
                var.data = var.initialize(method="random")
                vars0[name].data = var.data.copy()

        initial_states = self.cache_states(*args, **kwargs)
        self(*args, **kwargs)
        #self.tape.pop()
        self.recover_states(initial_states, *args, **kwargs)

        err = 0
        scale = 0
        for name, var in vars.items():
            smallest = var.smallest
            snp = vars0[name].data
            bsnp = var.data
            errii = snp - bsnp
            err += np.sum(errii**2)
            scale += np.sum((snp - np.mean(snp))**2) + smallest
        err = err / scale
        if verbose:
            print("Backpropagation test for Kernel %s: %.15e"
                  % (self.__class__.__name__, err))

        return err

    def linear_test(self, *args, **kwargs):

        vars = self.arguments(*args, **kwargs)
        pargs = deepcopy(args)
        pkwargs = deepcopy(kwargs)
        pvars = self.arguments(*pargs, **pkwargs)
        fargs = deepcopy(args)
        fkwargs = deepcopy(kwargs)
        fvars = self.arguments(*fargs, **fkwargs)
        for name, var in vars.items():
            var.data = var.initialize(method="random")
            var.lin = var.initialize(method="random")
            fvars[name].data = var.data.copy()
            fvars[name].lin = var.lin.copy()
        outs = self(*fargs, mode="linear", **fkwargs)
        try:
            iter(outs)
        except TypeError:
            outs = (outs,)

        errs = []
        cond = True if vars else False
        eps = 1.0
        while cond:
            for name, var in vars.items():
                pvars[name].data = var.data + eps * var.lin
            pouts = self(*pargs, mode="forward", **pkwargs)
            try:
                iter(pouts)
            except TypeError:
                pouts = (pouts,)

            err = 0
            scale = 0
            for out, pout in zip(outs, pouts):
                err += np.sum((pout.data - out.data - eps * out.lin)**2)
                scale += np.sum((eps*(out.lin - np.mean(out.lin)))**2)
            errs.append([err/(scale+out.smallest)])

            eps /= 10.0
            for el, var in vars.items():
                if np.max(eps*var.lin / (var.data+var.smallest)) < var.eps:
                    cond = False
                    break
        try:
            errmin = np.min(errs)
            print("Linear test for Kernel %s: %.15e"
                  % (self.__class__.__name__, errmin))
        except ValueError:
            errmin = 0
            print("Linear test for Kernel %s: unable to perform"
                  % (self.__class__.__name__))

        return errmin

    def dot_test(self, *args, **kwargs):
        """
        Dot product test for fstates, outputs = F(states)

        dF = [dfstates/dstates     [dstates
              doutputs/dstates]     dparams ]

        dot = [adj_states  ^T [dfstates/dstates     [states
               adj_outputs]    doutputs/dstates]   params]

        """

        vars = self.arguments(*args, **kwargs)
        fargs = deepcopy(args)
        fkwargs = deepcopy(kwargs)
        fvars = self.arguments(*fargs, **fkwargs)
        for name, var in vars.items():
            var.data = var.initialize(method="random")
            var.lin = var.initialize(method="random")
            var.grad = var.initialize(method="random")
            fvars[name].data = var.data.copy()
            fvars[name].lin = var.lin.copy()
            fvars[name].grad = var.grad.copy()
        self(*fargs, mode="linear", **fkwargs)
        self(*fargs, mode="adjoint", **fkwargs)

        prod1 = np.sum([np.sum(fvars[el].lin * vars[el].grad)
                        for el in vars])
        prod2 = np.sum([np.sum(fvars[el].grad * vars[el].lin)
                        for el in vars])

        print("Dot product test for Kernel %s: %.15e"
              % (self.__class__.__name__, (prod1-prod2)/(prod1+prod2)))

        return (prod1-prod2)/(prod1+prod2)

    def arguments(self, *args, **kwargs):
        a = self.signature.bind(*args, **kwargs)
        a.apply_defaults()

        out = {el: var for el, var in a.arguments.items()
               if type(var) is Variable}
        if "args" in a.arguments:
            for ii, var in enumerate(a.arguments["args"]):
                if type(var) is Variable:
                    out["arg"+str(ii)] = var
        return out


class ReversibleFunction(Function):

    def cache_states(self, *args, **kwargs):
        return {}

    def recover_states(self, initial_states, *args, **kwargs):
        return self.forward(*args, backpropagate=True, **kwargs)

#TODO replace localtape by saving the localtape in the current tape
class TapedFunction(Function):

    def __init__(self, fun):

        self.forward = self.linear = fun
        self.localtape = None
        super().__init__()

    def __call__(self, *args, mode=None, **kwargs):
        if mode is None:
            mode = self.tape.mode
        if mode == "forward" or mode == "linear":
            self.tape.append(self, *args, **kwargs)
            with Tape() as self.localtape:
                if mode == "linear":
                    self.localtape.mode = "linear"
                    outs = self.linear(*args, **kwargs)
                    self.localtape.mode = "forward"
                    return outs
                else:
                    return self.forward(*args, **kwargs)
        elif mode == "adjoint":
            return self.adjoint(*args, **kwargs)
        else:
            raise ValueError("Tape.mode should be in [\'forward\', \'linear\',"
                             " \'adjoint\']")

    def adjoint(self, *args, **kwargs):
        if self.localtape is None or not self.localtape.graph:
            raise ValueError("Empty local tape: forward or linear should be "
                             "called before adjoint")
        for argout, argin in zip(args, self.tape.graph[-1][1]):
            if argout is not argin and type(argout) is Variable:
                argout.grad = argin.grad
        for name, var in kwargs:
            if type(var) is Variable:
                if name in self.tape.graph[-1][2]:
                    self.tape.graph[-1][2][name].grad = var.grad
        while self.localtape.graph:
            fun, funargs, funkwargs = self.localtape.pop()
            fun.adjoint(*funargs, **funkwargs)
        return args

    def cache_states(self, *args, **kwargs):
        return {}

    def recover_states(self, initial_states, *args, **kwargs):
        while self.localtape.graph:
            self.localtape.pop()


class Function1(Function):
    def __init__(self,  **kwargs):
        super().__init__()
        self.updated_states = ["vx"]

    def forward(self, vx, vy, vz, alt=None):
        vx.data = True
        return vx

    def linear(self, vx, vy, vz, alt=None):
        vx.lin = True
        return vx

    def adjoint(self, vx, vy, vz, alt=None):
        vx.grad = True
        return vx


#TODO use unitest module

# Function and Variable should see the active tape, regardless of the Tape
# active at their creation.
# Tapeholder.tape should come back to its initial value after leaving context
tape0 = TapeHolder.tape
with Tape() as tape:
    fun1 = Function1()
    var1 = Variable("var1", shape=(1,))
    assert(tape == fun1.tape)
    assert(tape == var1.tape)
assert(tape0 == fun1.tape)
assert(tape0 == var1.tape)
assert(tape0 == TapeHolder.tape)


# Tape set the behavior of __call__ of function, choosing between forward,
# linear  and adjoint.
with Tape() as tape:
    fun1 = Function1()
    var1 = Variable("var1",
                    data=np.array([False]),
                    lin=np.array([False]),
                    grad=np.array([False]))
    assert(not var1.data)
    assert(not var1.lin)
    assert(not var1.grad)
    fun1(var1, var1, var1)
    assert(var1.data)
    assert(not var1.lin)
    assert(not var1.grad)
    var1.data = None
    tape.mode = "linear"
    fun1(var1, var1, var1)
    assert(var1.data)
    assert(var1.lin)
    assert(not var1.grad)
    var1.data = var1.lin = None
    tape.mode = "adjoint"
    fun1(var1, var1, var1)
    assert(not var1.data)
    assert(not var1.lin)
    assert(var1.grad)
    var1.data = var1.lin = var1.grad = None
    fun1(var1, var1, var1, mode="adjoint")
    assert(not var1.data)
    assert(not var1.lin)
    assert(var1.grad)


# TapedFunction can be used as a decorator to function of multiples Functions
with Tape() as tape:
    @TapedFunction
    def fun(var):
        fun1 = Function1()
        fun2 = Function1()
        for ii in range(3):
            fun1(var, var, var)
            fun2(var, var, var)
        return var
    assert(type(fun) is TapedFunction)
    var1 = Variable("var1",
                    data=np.array([False]),
                    lin=np.array([False]),
                    grad=np.array([False]))
    assert(not var1.data)
    assert(not var1.lin)
    assert(not var1.grad)
    fun(var1)
    assert(var1.data)
    assert(not var1.lin)
    assert(not var1.grad)
    assert(fun is tape.graph[0][0])

# Two TapedFunction can be defined without cross interactions
with Tape() as tape:
    @TapedFunction
    def fun1(var):
        fun1 = Function1()
        for ii in range(3):
            fun1(var, var, var)
        return var
    @TapedFunction
    def fun2(var):
        fun1 = Function1()
        for ii in range(3):
            fun1(var, var, var)
        return var
    var1 = Variable("var1",
                    data=np.array([False]),
                    lin=np.array([False]),
                    grad=np.array([False]))
    var2 = Variable("var1",
                    data=np.array([False]),
                    lin=np.array([False]),
                    grad=np.array([False]))
    fun1(var1)
    assert(fun1.localtape)
    assert(fun2.localtape is None)
    fun2(var1)
    assert(fun1.localtape is not fun2.localtape)
    assert(tape.graph[0][0] is fun1)
    assert(tape.graph[1][0] is fun2)


# A TapedFunction can contain Tapedfunctions
#TODO Beware, if fun1 or fun2 is called before fun3, fun3 will erase their tapes
with Tape() as tape:
    @TapedFunction
    def fun1(var):
        fun1 = Function1()
        for ii in range(3):
            fun1(var, var, var)
        return var
    @TapedFunction
    def fun2(var):
        fun1 = Function1()
        for ii in range(3):
            fun1(var, var, var)
        return var
    @TapedFunction
    def fun3(var):
        fun1(var)
        fun2(var)
        return var

    var1 = Variable("var1",
                    data=np.array([False]),
                    lin=np.array([False]),
                    grad=np.array([False]))
    fun3(var1)
    assert(fun1.localtape)
    assert(fun2.localtape)
    assert(fun3.localtape)
    assert(fun1.localtape is not fun2.localtape)
    assert(fun1.localtape is not fun3.localtape)
    assert(tape.graph[0][0] is fun3)


# Methods Function.cache_states and Function.recover_states should save and
# recover the states
with Tape() as tape:

    var1 = Variable("var1",
                    data=np.array([False]),
                    lin=np.array([False]),
                    grad=np.array([False]))
    fun = Function()
    initial_states = fun.cache_states(var1)
    assert(not var1.data)
    assert(not initial_states["arg0"])
    var1.data = True
    assert (var1.data)
    fun.recover_states(initial_states, var1)
    assert(not var1.data)


# Methods Function.cache_states and Function.recover_states should save and
# recover the states with regions
class FunRegion(Function):

    def __init__(self):
        super().__init__()
        self.updated_states = ["var"]
    def updated_regions(self, var):
        return [(1,), (2,)]

    def forward(self, var):
        for region in self.updated_regions(var):
            var.data[region] = True
        return var


with Tape() as tape:

    var1 = Variable("var1",
                    data=np.array([False, False, False]),
                    lin=np.array([False, False, False]),
                    grad=np.array([False, False, False]))
    fun = FunRegion()
    initial_states = fun.cache_states(var1)
    assert(not np.all(var1.data))
    assert(not np.all(initial_states["var"]))
    fun.forward(var1)
    assert (not var1.data[0] and var1.data[1] and var1.data[2])
    fun.recover_states(initial_states, var1)
    assert(not np.all(var1.data))
    assert(fun.backward_test(var1, verbose=False) < 1e-12)


# Reversible Functions should be able to pass backpropagation tests
class FunBack(ReversibleFunction):

    def __init__(self):
        super().__init__()
        self.updated_states = ["var"]

    def forward(self, var, backpropagate=False):
        if backpropagate:
            var.data += 1
        else:
            var.data -= 1
        return var


with Tape() as tape:
    fun = FunBack()
    var1 = Variable("var1",
                    data=np.array([1]))
    assert(var1.data[0] == 1)
    assert(not fun.cache_states(var1))
    fun(var1)
    assert(var1.data[0] == 0)
    fun.recover_states([], var1)
    assert(var1.data[0] == 1)
    assert(fun.backward_test(var1, verbose=False) < 1e-12)


