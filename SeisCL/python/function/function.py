from inspect import signature
from copy import copy
import numpy as np
from SeisCL.python import Tape, TapeHolder
from SeisCL.python import Variable


class Function(TapeHolder):
    """
    Base class to create a function that modifies the value of the `Variables`
    given as inputs.  `Functions`can be differentiated with reverse mode
    automatic differentiation.

    Override the methods `forward` `linear` and `adjoint` to implement
    your function.

    """

    mode = "forward"

    def __init__(self):
        self.signature = signature(self.forward)
        self.updated_states = []

    def __call__(self, *args, mode=None, **kwargs):
        locked = self.tape.locked
        self.tape.locked = True
        if mode is None:
            mode = self.tape.mode
        if mode == "forward" or mode == "linear":
            self.tape.append(self, *args, **kwargs)
            if mode == "linear":
                self.linear(*args, **kwargs)
            out = self.forward(*args, **kwargs)
        elif mode == "adjoint":
            self.tape.pop()
            out = self.adjoint(*args, **kwargs)
        else:
            raise ValueError("Tape.mode should be in [\'forward\', \'linear\',"
                             " \'adjoint\']")
        self.tape.locked = locked
        return out

    def cache_states(self, *args, **kwargs):
        if not self.updated_states:
            updated_states = self.arguments(*args, **kwargs).keys()
        else:
            updated_states = self.updated_states
        vars = self.arguments(*args, **kwargs)
        initial_states = {}
        for el in updated_states:
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
        """
        Performs y = f(x; z), x is a vector containing all differentiable inputs
        z contains all constant inputs and y is a vector containing all
        outputs. All arguments that are an instance of `Variable` with
        `Variable.differentiable` set to `True` are collected into x.

        The `forward` method cannot create new `Variable` instances.
        All variables either inputs or outputs (x or y) must be provided
        as arguments, either in *args or **kwargs.
        The method modifies the `Variable.data` attribute and returns all
         `Variable` instances that were modified. These are automatically
         collected to form y.

        :param args: Can contain `Variables`, ìnt`, `float` or `bool`.
        :param kwargs: Can contain `Variables`, ìnt`, `float` or `bool`.

        :return: Returns the `Variables` that are modified by the forward.

        """
        raise NotImplementedError

    def linear(self, *args, **kwargs):
        """
        Performs py = J(x0) px, where J(x0) = df/dx|_x=x0 is the Jacobian
        evaluated for the input x0 and px and py are the input and output
        perturbations.

        As for the `forward` method, all arguments that are an instance of
        `Variable` with `Variable.differentiable` set to `True` are collected
        into px and x0. The perturbed values px and py are contained in
        `Variable.lin`. Should return all updated variables, meaning it should
        have the same output signature than `forward`.
        """
        raise NotImplementedError

    def adjoint(self, *args, **kwargs):
        """
        Performs gx = J(x0)^T gy, where J(x0)^T is the transpose (adjoint) of
        the Jacobian evaluated at x0 and gy is the gradient (adjoint source)
        of the the outputs of `forward` y and gx is the gradient gradient of the
        inputs x with respect to the output y.

        The method modifies the `Variable.grad` attribute of arguments in x,
        i.e. all `Variable` instances which affect y with
        `Variable.differentiable` set to `True`. Returns all `Variable` instances
        in `x`.

        """
        raise NotImplementedError

    def backward_test(self, *args, verbose=False, **kwargs):

        vars = self.arguments(*args, **kwargs)
        vars0 = {name: copy(var) for name, var in vars.items()
                 if isinstance(var, Variable)}

        initial_states = self.cache_states(*args, **kwargs)
        self(*args, **kwargs)
        self.recover_states(initial_states, *args, **kwargs)

        err = 0
        scale = 0
        for name, var in vars.items():
            smallest = var.smallest
            snp = vars0[name].data
            bsnp = var.data
            errii = snp - bsnp
            err += np.sum(errii**2)
            mean = np.sum(snp) / snp.size
            scale += np.sum((snp - mean)**2) + smallest
        err = err / scale
        if hasattr(err, "get"):
            err = err.get()[()]
        err = np.sqrt(err)
        if verbose:
            print("Backpropagation test for Kernel %s: %.15e"
                  % (self.__class__.__name__, err))

        return err

    def linear_test(self, *args, verbose=False, **kwargs):

        vars = self.arguments(*args, **kwargs)
        pargs = [copy(a) for a in args]
        pkwargs = {name: copy(val) for name, val in kwargs.items()}
        pvars = self.arguments(*pargs, **pkwargs)
        fargs = [copy(a) for a in args]
        fkwargs = {name: copy(val) for name, val in kwargs.items()}
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
                if var.differentiable:
                    pvars[name].data = var.data + eps * var.lin
            pouts = self(*pargs, mode="forward", **pkwargs)
            try:
                iter(pouts)
            except TypeError:
                pouts = (pouts,)

            err = 0
            for out, pout in zip(outs, pouts):
                if pouts and out:
                    smallest = out.smallest
                    erri = np.abs((pout.data - out.data - eps * out.lin) / (eps * out.lin + 10*smallest))
                    if hasattr(erri, "get"):
                        erri = erri.get()[()]
                    err = np.max([err, np.max(erri)])

            errs.append(err)
            eps /= 10
            for el, var in vars.items():
                p = eps*var.lin / (var.data+var.smallest)
                if hasattr(p, "get"):
                    p = p.get()[()]
                if np.max(p) < var.eps:
                    cond = False
                    break
        errmin = np.min(errs)
        if hasattr(errmin, "get"):
            errmin = errmin.get()[()]
        if verbose:
            print("Linear test for Kernel %s: %.15e"
                  % (self.__class__.__name__, errmin))

        return errmin

    def dot_test(self, *args, verbose=False, **kwargs):
        """
        Dot product test for fstates, outputs = F(states)

        dF = [dfstates/dstates     [dstates
              doutputs/dstates]     dparams ]

        dot = [adj_states  ^T [dfstates/dstates     [states
               adj_outputs]    doutputs/dstates]   params]

        """

        vars = self.arguments(*args, **kwargs)
        fargs = [copy(a) for a in args]
        fkwargs = {name: copy(val) for name, val in kwargs.items()}
        fvars = self.arguments(*fargs, **fkwargs)
        for name, var in vars.items():
            var.data = var.initialize(method="random")
            var.lin = var.initialize(method="random")
            if var.differentiable:
                var.grad = var.initialize(method="random")
            fvars[name].data = var.data.copy()
            fvars[name].lin = var.lin.copy()
            if var.differentiable:
                fvars[name].grad = var.grad.copy()
        self(*fargs, mode="linear", **fkwargs)
        self(*fargs, mode="adjoint", **fkwargs)


        prod1 = np.sum([np.sum(fvars[el].lin * vars[el].grad)
                        for el in vars if vars[el].differentiable])
        prod2 = np.sum([np.sum(fvars[el].grad * vars[el].lin)
                        for el in vars if vars[el].differentiable])

        res = (prod1-prod2)/(prod1+prod2)
        if hasattr(res, "get"):
            res = res.get()[()]
        if verbose:
            print("Dot product test for Kernel %s: %.15e"
                  % (self.__class__.__name__, res))

        return res

    def arguments(self, *args, **kwargs):
        a = self.signature.bind(*args, **kwargs)
        a.apply_defaults()

        out = {el: var for el, var in a.arguments.items()
               if isinstance(var, Variable)}
        if "args" in a.arguments:
            for ii, var in enumerate(a.arguments["args"]):
                if isinstance(var, Variable):
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
            with Tape(locked=True) as self.localtape:
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
        for name, var in kwargs.items():
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


#TODO Should we have a MapReduceFunction and a MapFunction?
class MapGather(Function):
    def __init__(self, fun, reduce, *args, **kwargs):
        self.fun = fun
        self.reduce = reduce
        self.child_tapes = []
        super().__init__(*args, **kwargs)

    def __call__(self, args, mode=None, **kwargs):
        if mode is None:
            mode = self.tape.mode
        #each child tape should have the same initial value as the parent tape
        if mode == "forward" or mode == "linear":
            self.child_tapes = [Tape() for __ in args]
            outputs = []
            for ii, arg in enumerate(args):
                with self.child_tapes[ii] as tape:
                    tape.mode = mode
                    outputs.append([el.copy() for el in self.fun(arg)])
        else:
            for ii, arg in enumerate(args):
                with self.child_tapes[ii] as tape:
                    tape.mode = mode
                    outs = self.fun(arg)
                    for argout, argin in zip(arg, self.tape.graph[-1][1]):
                        if argout is not argin and type(argout) is Variable:
                            argout.grad = argin.grad
                    for name, var in kwargs.items():
                        if type(var) is Variable:
                            if name in self.tape.graph[-1][2]:
                                self.tape.graph[-1][2][name].grad = var.grad



        return outputs

    def forward(self, *arg, **kwargs):
        return self.fun(*arg, **kwargs)

    def linear(self, *args, **kwargs):
        return self.fun(*args, **kwargs, mode="linear")

    def adjoint(self, *args, **kwargs):
        return self.fun(*args, **kwargs, mode="adjoint")


