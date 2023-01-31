#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:13:18 2017

@author: gabrielfabien-ouellet
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
from bigfloat import half_precision, BigFloat
from copy import copy
from inspect import signature, Parameter
#import os.path


def Dpx(var):
    return 1.1382 * (var[2:-2, 3:-1] - var[2:-2, 2:-2]) - 0.046414 * (var[2:-2, 4:] - var[2:-2, 1:-3])


def Dpx_adj(var):

    dvar = var.copy()
    dvar[2:-2, 2:-2] = 0
    dvar[2:-2, 3:-1] += 1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 2:-2] += -1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 4:] += - 0.046414 * var[2:-2, 2:-2]
    dvar[2:-2, 1:-3] += 0.046414 * var[2:-2, 2:-2]

    return dvar


def Dmx(var):
    return 1.1382 * (var[2:-2, 2:-2] - var[2:-2, 1:-3]) - 0.046414 * (var[2:-2, 3:-1] - var[2:-2, 0:-4])


def Dmx_adj(var):

    dvar = var.copy()
    dvar[2:-2, 2:-2] = 0
    dvar[2:-2, 2:-2] += 1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 1:-3] += -1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 3:-1] += - 0.046414 * var[2:-2, 2:-2]
    dvar[2:-2, 0:-4] += 0.046414 * var[2:-2, 2:-2]

    return dvar


def Dpz(var):
    return 1.1382 * (var[3:-1, 2:-2] - var[2:-2, 2:-2]) - 0.046414 * (var[4:, 2:-2] - var[1:-3, 2:-2])


def Dpz_adj(var):

    dvar = var.copy()
    dvar[2:-2, 2:-2] = 0
    dvar[1:-3, 2:-2] += 0.046414 * var[2:-2, 2:-2]
    dvar[2:-2, 2:-2] += -1.1382 * var[2:-2, 2:-2]
    dvar[3:-1, 2:-2] += 1.1382 * var[2:-2, 2:-2]
    dvar[4:, 2:-2] += - 0.046414 * var[2:-2, 2:-2]


    return dvar


def Dmz(var):
    return 1.1382 * (var[2:-2, 2:-2] - var[1:-3, 2:-2]) - 0.046414 * (var[3:-1, 2:-2] - var[0:-4, 2:-2])


def Dmz_adj(var):

    dvar = var.copy()
    dvar[2:-2, 2:-2] = 0
    dvar[0:-4, 2:-2] += 0.046414 * var[2:-2, 2:-2]
    dvar[1:-3, 2:-2] += -1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 2:-2] += 1.1382 * var[2:-2, 2:-2]
    dvar[3:-1, 2:-2] += - 0.046414 * var[2:-2, 2:-2]


    return dvar


class Grid:

    backend = np.ndarray

    def __init__(self, shape=(10, 10), pad=2, dh=1, dt=1, nt=1, nfddim=None,
                 dtype=np.float32, zero_boundary=False, **kwargs):
        self.shape = shape
        self.pad = pad
        self.dh = dh
        self.dt = dt
        self.nt = nt
        if nfddim is None:
            nfddim = len(shape)
        self.nfddim = nfddim
        self.dtype = dtype
        self.eps = np.finfo(dtype).eps
        self.smallest = np.nextafter(dtype(0), dtype(1))
        self.zero_boundary = zero_boundary

    @property
    def valid(self):
        return tuple([slice(self.pad, -self.pad)] * self.nfddim)

    def zero(self):
        return np.zeros(self.shape, dtype=self.dtype, order="F")

    def random(self):
        if self.zero_boundary:
            state = np.zeros(self.shape, dtype=self.dtype, order="F")
            state[self.valid] = np.random.rand(*state[self.valid].shape)*10e6
        else:
            state = np.random.rand(*self.shape).astype(self.dtype)
        return np.require(state, requirements='F')

    def assign_data(self, data):
        return np.require(data.astype(self.dtype), requirements='F')

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

    def initialize(self, data=None, method="zero"):

        if data is not None:
            data = self.assign_data(data)
        elif method == "zero":
            data = self.zero()
        elif method == "random":
            data = self.random()
        return data

    def xyz2lin(self, *args):
        return np.ravel_multi_index([np.array(el)+self.pad for el in args],
                                    self.shape, order="F")

    @staticmethod
    def np(array):
        return array


class Tape:
    """
    Keeps track of function calls as well as variables.
    """
    def __init__(self):
        self.vars = {} #keep track of all encountered variables.
        self.graph = []

    def append(self, kernel):
        self.graph.append(kernel)

    def pop(self):
        return self.graph.pop()

    def add_variable(self, var):
        if var.name in self.vars:
            raise NameError("Variable name already exists in tape")
        self.vars[var.name] = var

    def empty(self):
        self.vars = {} #keep track of all encountered variables.
        self.graph = []


class TapeHolder:
    _tape = Tape()


class Function(TapeHolder):
    """
    Kernel implementing forward, linear and adjoint modes.
    """

    def __init__(self, grids=None, **kwargs):
        self._grids = grids
        self.grids = grids
        self._forward_states = None
        self.ncall = 0
        if not hasattr(self, 'updated_states'):
            self.updated_states = {}
        self.signature = signature(self.forward)
        self.required_states = [name for name, par
                                in signature(self.forward).parameters.items()
                                if par.kind == Parameter.POSITIONAL_OR_KEYWORD]
        if not hasattr(self, 'default_grids'):
            self._default_grids = {}
        if not hasattr(self, 'copy_states'):
            self.copy_states = {}
        if not hasattr(self, 'zeroinit_states'):
            self.zeroinit_states = []

    def __call__(self, *args, initialize=True, cache_states=True, **kwargs):

        kwargs = self.make_kwargs_compatible(**kwargs)
        if initialize:
            self.initialize(*args, empty_cache=cache_states, **kwargs)
        if cache_states:
            self._tape.append(self)
            self.cache_states(*args, **kwargs)
        #TODO How can I prevent users from defining which state they modified ?
        return self.forward(*args, **kwargs)

    def call_linear(self, *args, initialize=True, cache_states=True, **kwargs):

        kwargs = self.make_kwargs_compatible(**kwargs)
        if initialize:
            self.initialize(*args, empty_cache=cache_states, **kwargs)
        if cache_states:
            self.cache_states(*args, **kwargs)
        self.linear(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def gradient(self, *args, initialize=True, **kwargs):

        kwargs = self.make_kwargs_compatible(**kwargs)
        if initialize:
            self.initialize(*args,  adjoint=True, **kwargs)
        self.backward(*args, **kwargs)
        return self.adjoint(*args, **kwargs)

    def cache_states(self, *args, **kwargs):
        vars = self.arguments(*args, **kwargs)
        for el in self.updated_states:
            if el in self.updated_regions:
                for ii, region in enumerate(self.updated_regions[el]):
                    self._forward_states[el][ii][..., self.ncall] = vars[el].data[region]
            else:
                self._forward_states[el][..., self.ncall] = vars[el].data
        self.ncall += 1

    @property
    def updated_regions(self):
        return {}

    @property
    def grids(self):
        return self._grids

    @grids.setter
    def grids(self, val):
        self._grids = val

    @property
    def default_grids(self):
        return self._default_grids

    @default_grids.setter
    def default_grids(self, val):
        self._default_grids = val

    @property
    def required_grids(self):
        return list(set([val for val in self.default_grids.values()]))

    def initialize(self, *args, empty_cache=True, method="zero", adjoint=False,
                   copy_state=True, cache_size=1, **kwargs):

        # if adjoint:
        #     toinit = list(set(self.updated_states + self.required_states))
        # else:
        #     toinit = self.required_states
        #     self.ncall = 0
        #
        # for el in toinit:
        #     if el not in self.grids:
        #         self.grids[el] = self.grids[self.default_grids[el]]
        #
        #     if el not in argins and el not in self._tape.vars:
        #         if el not in self.zeroinit_states:
        #             var = Variable(el,
        #                            data=self.grids[el].initialize(method=method)
        #                            )
        #         else:
        #             var = Variable(el,
        #                            data=self.grids[el].initialize()
        #                            )
        #     elif el not in self._tape.vars:
        #         if type(argins[el]) is Variable:
        #             self._tape.vars[el] = argins[el]
        #         else:
        #             if type(argins[el]) is not self.grids[el].backend:
        #                 data = self.grids[el].initialize(data=argins[el])
        #             else:
        #                 data = argins[el]
        #             self._tape.vars[el] = Variable(el, data=data)
        #     else:
        #         raise NotImplemented("Variable already in tape:"
        #                              " cannot overwrite")

        # if copy_state:
        #     for el in self.copy_states:
        #         if el not in self._tape.vars:
        #             self._tape.vars[el] = self._tape.vars[self.copy_states[el]]

        vars = self.arguments(*args, **kwargs)
        if empty_cache:
            self._forward_states = {}
            for el in self.updated_states:
                if el in self.updated_regions:
                    regions = self.updated_regions[el]
                else:
                    regions = None
                self._forward_states[el] = vars[el].create_cache(regions=regions,
                                                                 cache_size=cache_size)
            self.ncall = 0

    def make_kwargs_compatible(self, **kwargs):
        return kwargs

    def forward(self, *args, **kwargs):
        """
        Applies the forward kernel.

        :param **kwargs:
        :param states: A dict containing the variables describing the state of a
                       dynamical system. The state variables are updated by
                       the forward, but they keep the same dimensions.
        :return:
            states:     A dict containing the updated states.
        """
        raise NotImplemented

    def linear(self, *args, **kwargs):
        """
        Applies the linearized forward J = d forward / d states
        to a state perturbation out = J * dstates
        By default, forward is treated as a linear function
        (self.linear = self.forward).

        :param dstates: State perturbation
        :param states:  State at which the forward is linearized
        :param kwargs:
        :return: J * dstates
        """
        raise NotImplemented

    def adjoint(self, *args, **kwargs):
        """
        Applies the adjoint of the forward

        :param **kwargs:
        :param adj_states: A dict containing the adjoint of the forward states.
                           Each elements has the same dimension as the forward
                           state, as the forward kernel do not change the
                           dimension of the state.
        :param states: The states of the system, before calling forward.

        :return:
            adj_states All Variables modified by adjoint.
        """
        raise NotImplemented

    def backward(self, *args, **kwargs):
        """
        Reconstruct the input states from the output of forward

        :param **kwargs:
        :param states: A dict containing the variables describing the state of a
                      dynamical system. The state variables are updated by
                      the forward, but they keep the same dimensions.
        """

        vars = self.arguments(*args, **kwargs)
        self.ncall -= 1
        for el in self._forward_states:
            if el in self.updated_regions:
                for ii, reg in enumerate(self.updated_regions[el]):
                    vars[el].data[reg] = self._forward_states[el][ii][...,
                                                                 self.ncall]
            else:
                vars[el].data = self._forward_states[el][..., self.ncall]

    def backward_test(self, *args, **kwargs):

        vars = self.arguments(*args, **kwargs)
        vars0 = {name: copy(var) for name, var in vars.items()}
        for name, var in vars.items():
            var.data = var.initialize(method="random")
            vars0[name].data = var.data.copy()

        self(*args, **kwargs)
        self.backward(*args, **kwargs)

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
        print("Backpropagation test for Kernel %s: %.15e"
              % (self.__class__.__name__, err))

        return err

    def linear_test(self, *args, **kwargs):
        #TODO correct linear test
        vars = self.arguments(*args, **kwargs)
        vars0 = {name: copy(var) for name, var in vars.items()}
        for name, var in vars.items():
            var.data = var.initialize(method="random")
            var.lin = var.initialize(method="random")
            vars0[name].data = var.data.copy()
            vars0[name].lin = var.lin.copy()

        errs = []
        cond = True if vars else False
        if not any([el not in self.zeroinit_states for el in vars]):
            cond = False
        while cond:
            dstates = {el: self.grids[el].np(dstates[el]) / 10.0
                       for el in dstates}
            dstates = self.initialize(dstates, copy_state=False)
            for el in states:
                if el not in self.zeroinit_states:
                    dnp = self.grids[el].np(dstates[el])
                    snp = self.grids[el].np(states[el])
                    smallest = self.grids[el].smallest
                    eps = self.grids[el].eps
                    if np.max(dnp / (snp+smallest)) < eps:
                        cond = False
                        break
            if not cond:
                break
            pstates = {el: states[el] + dstates[el] for el in states}

            fpstates = self({el: pstates[el].copy() for el in pstates}, **kwargs)
            fstates = self({el: states[el].copy() for el in states}, **kwargs)

            lstates, _ = self.call_linear({el: dstates[el].copy()
                                           for el in dstates},
                                          {el: states[el].copy()
                                           for el in states},
                                          **kwargs)

            err = 0
            scale = 0
            for el in states:
                smallest = self.grids[el].smallest
                eps = self.grids[el].eps
                ls = self.grids[el].np(lstates[el])
                fdls = self.grids[el].np(fpstates[el] - fstates[el])
                errii = fdls - ls
                err += np.sum(errii**2)
                scale += np.sum((ls - np.mean(ls))**2)
            errs.append([err/(scale+smallest)])

        try:
            errmin = np.min(errs)
            print("Linear test for Kernel %s: %.15e"
                  % (self.__class__.__name__, errmin))
        except ValueError:
            errmin = 0
            print("Linear test for Kernel %s: unable to perform"
                  % (self.__class__.__name__))

        return errmin

    def dot_test(self, **kwargs):
        """
        Dot product test for fstates, outputs = F(states)

        dF = [dfstates/dstates     [dstates
              doutputs/dstates]     dparams ]

        dot = [adj_states  ^T [dfstates/dstates     [states
               adj_outputs]    doutputs/dstates]   params]

        """

        states = self.initialize({}, method="random", copy_state=False)
        dstates = self.initialize({}, empty_cache=False,
                                  method="random", copy_state=False)
        dfstates, fstates = self.call_linear({el: dstates[el].copy()
                                              for el in dstates},
                                             {el: states[el].copy()
                                              for el in states},
                                             initialize=False,
                                             **kwargs)

        adj_states = self.initialize({}, empty_cache=False, method="random",
                                     adjoint=True, copy_state=False)
        fadj_states, _ = self.gradient({el: adj_states[el].copy()
                                        for el in adj_states},
                                       {el: fstates[el].copy()
                                        for el in fstates},
                                       initialize=False,
                                       **kwargs)

        prod1 = np.sum([np.sum(self.grids[el].np(dfstates[el] * adj_states[el]))
                        for el in dfstates])
        prod2 = np.sum([np.sum(self.grids[el].np(dstates[el] * fadj_states[el]))
                        for el in dstates])

        print("Dot product test for Kernel %s: %.15e"
              % (self.__class__.__name__, (prod1-prod2)/(prod1+prod2)))

        return (prod1-prod2)/(prod1+prod2)

    def arguments(self, *args, **kwargs):
        a = self.signature.bind(*args, **kwargs)
        a.apply_defaults()
        return a.arguments


class Variable(TapeHolder):
    """

    """
    def __init__(self, name, data=None, shape=None, lin=None,
                 initialize_method="zero", dtype=np.float, zero_boundary=False):
        self.name = name
        self._tape.add_variable(self)
        self.data = data
        if data is None:
            if shape is None:
                raise ValueError("Either data or shape must be provided")
            else:
                self.shape = shape
        else:
            self.shape = data.shape
        self._lin = lin # contains the small data perturbation
        self.grad = None # contains the gradient
        self.last_update = None # The last kernel that updated the state
        self.initialize_method = initialize_method
        self.dtype = dtype
        self.zero_boundary = zero_boundary
        self.smallest = np.nextafter(dtype(0), dtype(1))

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
        if self.zero_boundary:
            state = np.zeros(self.shape, dtype=self.dtype, order="F")
            state[self.valid] = np.random.rand(*state[self.valid].shape)*10e6
        else:
            state = np.random.rand(*self.shape).astype(self.dtype)
        return np.require(state, requirements='F')

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


class RandKernel(Function):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.updated_states = ["x", "y"]
        self.A1 = np.random.rand(10, 10)
        self.A2 = np.random.rand(10, 10)

    def forward(self, x, b, y):

        x.data = np.matmul(self.A1, x.data)
        y.data = np.matmul(self.A2, x.data) * b.data
        return x, y

    def linear(self, x, b, y):
        x.lin = np.matmul(self.A1, x.lin)
        y.lin = np.matmul(self.A2, x.data) * b.lin \
                   + np.matmul(self.A2, x.lin) * b.data

        return x, y

    def adjoint(self, x, b, y):

        A1t = np.transpose(self.A1)
        A2t = np.transpose(self.A2)

        x.grad = np.matmul(A1t, x.grad) + np.matmul(A2t, b.data * y.grad)
        b.grad = np.matmul(self.A2, x.data) * y.grad + b.grad
        return x, b


class Derivative(Function):

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vx"]
        self.updated_states = ["vx"]

    def forward(self, states, **kwargs):
        valid = self.grids["vx"].valid
        states["vx"][valid] = Dmz(states["vx"])
        return states

    def adjoint(self, adj_states, states, **kwargs):
        valid = self.grids["vx"].valid
        adj_states["vx"] = Dmz_adj(adj_states["vx"])
        return adj_states


class Division(Function):

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vx", "vz"]
        self.updated_states = ["vx"]
        self.smallest = self.grids["vx"].smallest

    def forward(self, states, **kwargs):

        states["vx"] = states["vx"] / (states["vz"]+self.smallest)
        return states

    def linear(self, dstates, states, **kwargs):
        """
        [dvx   =   [1/vz -vx/vz**2] [dvx
         dvz]       0        1    ]  dvz]

        [vx'   =   [1/vz       0] [vx'
         vz']       -vx/vz**2  1]  vz']
        """

        dstates["vx"] = dstates["vx"] / (states["vz"] + self.smallest)
        dstates["vx"] += -states["vx"] / (states["vz"] + self.smallest)**2 * dstates["vz"]

        return dstates

    def adjoint(self, adj_states, states, **kwargs):
        adj_states["vz"] += -states["vx"] / (states["vz"] + self.smallest)**2 * adj_states["vx"]
        adj_states["vx"] = adj_states["vx"] / (states["vz"] + self.smallest)
        return adj_states


class Multiplication(Function):

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vx", "vz"]
        self.updated_states = ["vx"]
        self.smallest = self.grids["vx"].smallest

    def forward(self, states, **kwargs):

        states["vx"] = states["vx"] * states["vz"]
        return states

    def linear(self, dstates, states, **kwargs):
        """
        [dvx   =   [vz vx] [dvx
         dvz]       0   1]  dvz]

        [vx'   =   [vz  0] [vx'
         vz']       vx  1]  vz']
        """
        dstates["vx"] = dstates["vx"] * states["vz"]
        dstates["vx"] += states["vx"] * dstates["vz"]

        return dstates

    def adjoint(self, adj_states, states, **kwargs):
        adj_states["vz"] += states["vx"] * adj_states["vx"]
        adj_states["vx"] = adj_states["vx"] * states["vz"]
        return adj_states


class ReversibleFunction(Function):

    def __call__(self, states, initialize=True, **kwargs):
        if initialize:
            states = self.initialize(states)
        kwargs = self.make_kwargs_compatible(**kwargs)
        return self.forward(states, **kwargs)

    def backward(self, states, **kwargs):
        kwargs = self.make_kwargs_compatible(**kwargs)
        states = self.forward(states, backpropagate=True, **kwargs)
        return states


class UpdateVelocity(ReversibleFunction):

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "cv"]
        self.updated_states = ["vx", "vz"]
        self.default_grids = {"vx": "gridfd",
                              "vz": "gridfd",
                              "sxx": "gridfd",
                              "szz": "gridfd",
                              "sxz": "gridfd",
                              "cv": "gridpar"}

    def forward(self, states, backpropagate=False, **kwargs):
        """
        Update the velocity for 2D P-SV isotropic elastic wave propagation.

        In matrix form:
            [vx     [ 1       0    cv * Dpx     0    cv * Dmz      [vx
             vz       0       1       0    cv * Dpz  cv * Dmx       vz
             sxx  =   0       0       1        0        0      =   sxx
             szz      0       0       0        1        0          szz
             sxz]     0       0       0        0        1   ]      sxz]
             :param **kwargs:
        """
        sxx = states["sxx"]
        szz = states["szz"]
        sxz = states["sxz"]
        vx = states["vx"]
        vz = states["vz"]

        cv = states["cv"]

        sxx_x = Dpx(sxx)
        szz_z = Dpz(szz)
        sxz_x = Dmx(sxz)
        sxz_z = Dmz(sxz)

        valid = self.grids["vx"].valid
        if not backpropagate:
            vx[valid] += (sxx_x + sxz_z) * cv[valid]
            vz[valid] += (szz_z + sxz_x) * cv[valid]
        else:
            vx[valid] -= (sxx_x + sxz_z) * cv[valid]
            vz[valid] -= (szz_z + sxz_x) * cv[valid]

        return states

    def linear(self, dstates, states, **kwargs):

        self.forward({"vx": dstates["vx"],
                      "vz": dstates["vz"],
                      "sxx": dstates["sxx"],
                      "szz": dstates["szz"],
                      "sxz": dstates["sxz"],
                      "cv": states["cv"]})
        sxx = states["sxx"]
        szz = states["szz"]
        sxz = states["sxz"]

        dcv = dstates["cv"]

        sxx_x = Dpx(sxx)
        szz_z = Dpz(szz)
        sxz_x = Dmx(sxz)
        sxz_z = Dmz(sxz)

        valid = self.grids["vx"].valid
        dstates["vx"][valid] += (sxx_x + sxz_z) * dcv[valid]
        dstates["vz"][valid] += (szz_z + sxz_x) * dcv[valid]

        return dstates

    def adjoint(self, adj_states, states, **kwargs):
        """
        Adjoint update the velocity for 2D P-SV isotropic elastic wave
        propagation.

        The transpose of the forward:
            [vx'     [      1        0        0     0    0     [vx'
             vz'            0        1        0     0    0      vz'
             sxx'  =   -Dmx * cv     0        1     0    0  =   sxx'
             szz'           0    -Dmz * cv     0     1    0     szz'
             sxz']     -Dpz * cv -Dpx * cv     0     0    1]    sxz']
        """
        adj_sxx = adj_states["sxx"]
        adj_szz = adj_states["szz"]
        adj_sxz = adj_states["sxz"]
        adj_vx = adj_states["vx"]
        adj_vz = adj_states["vz"]

        sxx = states["sxx"]
        szz = states["szz"]
        sxz = states["sxz"]

        cv = states["cv"]
        valid = self.grids["vx"].valid

        cv0 = np.zeros_like(cv)
        cv0[valid] = cv[valid]
        adj_vx_x = Dpx_adj(cv0 * adj_vx)
        adj_vx_z = Dmz_adj(cv0 * adj_vx)
        adj_vz_z = Dpz_adj(cv0 * adj_vz)
        adj_vz_x = Dmx_adj(cv0 * adj_vz)

        sxx_x = Dpx(sxx)
        szz_z = Dpz(szz)
        sxz_x = Dmx(sxz)
        sxz_z = Dmz(sxz)


        adj_sxx += adj_vx_x
        adj_szz += adj_vz_z
        adj_sxz += adj_vx_z + adj_vz_x

        adj_states["cv"][valid] += (sxx_x + sxz_z) * adj_vx[valid]
        adj_states["cv"][valid] += (szz_z + sxz_x) * adj_vz[valid]

        return adj_states


class UpdateVelocity2(UpdateVelocity):

    def adjoint(self, adj_states, states, **kwargs):
        """
        Adjoint update the velocity for 2D P-SV isotropic elastic wave
        propagation.

        The transpose of the forward:
            [vx'     [      1        0        0     0    0     [vx'
             vz'            0        1        0     0    0      vz'
             sxx'  =   -Dmx * cv     0        1     0    0  =   sxx'
             szz'           0    -Dmz * cv     0     1    0     szz'
             sxz']     -Dpz * cv -Dpx * cv     0     0    1]    sxz']
        """
        adj_sxx = adj_states["sxx"]
        adj_szz = adj_states["szz"]
        adj_sxz = adj_states["sxz"]
        adj_vx = adj_states["vx"]
        adj_vz = adj_states["vz"]

        sxx = states["sxx"]
        szz = states["szz"]
        sxz = states["sxz"]

        cv = states["cv"]
        valid = self.grids["vx"].valid

        cv0 = np.zeros_like(cv)
        cv0[valid] = cv[valid]
        adj_vx_x = -Dmx(cv0 * adj_vx)
        adj_vx_z = -Dpz(cv0 * adj_vx)
        adj_vz_z = -Dmz(cv0 * adj_vz)
        adj_vz_x = -Dpx(cv0 * adj_vz)

        sxx_x = Dpx(sxx)
        szz_z = Dpz(szz)
        sxz_x = Dmx(sxz)
        sxz_z = Dmz(sxz)


        adj_sxx[valid] += adj_vx_x
        adj_szz[valid] += adj_vz_z
        adj_sxz[valid] += adj_vx_z + adj_vz_x

        adj_states["cv"][valid] += (sxx_x + sxz_z) * adj_vx[valid]
        adj_states["cv"][valid] += (szz_z + sxz_x) * adj_vz[valid]

        return adj_states


class UpdateStress(ReversibleFunction):

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "csu", "csM"]
        self.updated_states = ["sxx", "szz", "sxz"]
        self.default_grids = {"vx": "gridfd",
                              "vz": "gridfd",
                              "sxx": "gridfd",
                              "szz": "gridfd",
                              "sxz": "gridfd",
                              "csu": "gridpar",
                              "csM": "gridpar"}

    def forward(self, states, backpropagate=False, **kwargs):
        """
        Update the velocity for 2D P-SV isotropic elastic wave propagation.

        In matrix form:
            [vx     [         1                0        0 0 0   [vx
             vz               0                1        0 0 0    vz
             sxx  =        csM Dmx     (csM - 2csu) Dmz 1 0 0 =  sxx
             szz      (csM - 2csu) Dmx      csM Dmz     0 1 0    szz
             sxz]          csu Dpz          csu Dpx     0 0 1]   sxz]
        """
        sxx = states["sxx"]
        szz = states["szz"]
        sxz = states["sxz"]
        vx = states["vx"]
        vz = states["vz"]

        csu = states["csu"]
        csM = states["csM"]

        vx_x = Dmx(vx)
        vx_z = Dpz(vx)
        vz_x = Dpx(vz)
        vz_z = Dmz(vz)

        valid = self.grids["vx"].valid
        if not backpropagate:
            sxx[valid] += csM[valid] * (vx_x + vz_z) - 2.0 * csu[valid] * vz_z
            szz[valid] += csM[valid] * (vx_x + vz_z) - 2.0 * csu[valid] * vx_x
            sxz[valid] += csu[valid] * (vx_z + vz_x)
        else:
            sxx[valid] -= csM[valid] * (vx_x + vz_z) - 2.0 * csu[valid] * vz_z
            szz[valid] -= csM[valid] * (vx_x + vz_z) - 2.0 * csu[valid] * vx_x
            sxz[valid] -= csu[valid] * (vx_z + vz_x)

        return states

    def linear(self, dstates, states, **kwargs):

        self.forward({"vx": dstates["vx"],
                      "vz": dstates["vz"],
                      "sxx": dstates["sxx"],
                      "szz": dstates["szz"],
                      "sxz": dstates["sxz"],
                      "csu": states["csu"],
                      "csM": states["csM"]})

        vx = states["vx"]
        vz = states["vz"]

        vx_x = Dmx(vx)
        vx_z = Dpz(vx)
        vz_x = Dpx(vz)
        vz_z = Dmz(vz)

        dsxx = dstates["sxx"]
        dszz = dstates["szz"]
        dsxz = dstates["sxz"]
        dcsu = dstates["csu"]
        dcsM = dstates["csM"]

        valid = self.grids["vx"].valid
        dsxx[valid] += dcsM[valid] * (vx_x + vz_z) - 2.0 * dcsu[valid] * vz_z
        dszz[valid] += dcsM[valid] * (vx_x + vz_z) - 2.0 * dcsu[valid] * vx_x
        dsxz[valid] += dcsu[valid] * (vx_z + vz_x)

        return dstates

    def adjoint(self, adj_states, states, **kwargs):
        """
        Adjoint update the velocity for 2D P-SV isotropic elastic wave
        propagation.

        The transpose of the forward:
            [vx'     [ 1    0      -Dpx csM     -Dpx(csM - 2csu) -Dmz csu [vx'
             vz'       0    1  -Dpz (csM - 2csu)    -Dpz csM     -Dmx csu  vz'
             sxx'  =   0    0          1               0           0   =   sxx'
             szz'      0    0          0               1           0       szz'
             sxz']     0    0          0               0           1  ]    sxz']
             :param **kwargs:
        """
        adj_sxx = adj_states["sxx"]
        adj_szz = adj_states["szz"]
        adj_sxz = adj_states["sxz"]
        adj_vx = adj_states["vx"]
        adj_vz = adj_states["vz"]

        sxx = states["sxx"]
        szz = states["szz"]
        sxz = states["sxz"]
        vx = states["vx"]
        vz = states["vz"]

        csu = states["csu"]
        csM = states["csM"]

        valid = self.grids["vx"].valid
        csu0 = np.zeros_like(csu)
        csu0[valid] = csu[valid]
        csM0 = np.zeros_like(csM)
        csM0[valid] = csM[valid]

        adj_sxx_x = Dmx_adj(csM0 * adj_sxx)
        adj_sxx_z = Dmz_adj((csM0 - 2.0 * csu0) * adj_sxx)
        adj_szz_x = Dmx_adj((csM0 - 2.0 * csu0) * adj_szz)
        adj_szz_z = Dmz_adj(csM0 * adj_szz)
        adj_sxz_x = Dpx_adj(csu0 * adj_sxz)
        adj_sxz_z = Dpz_adj(csu0 * adj_sxz)

        vx_x = Dmx(vx)
        vx_z = Dpz(vx)
        vz_x = Dpx(vz)
        vz_z = Dmz(vz)


        adj_vx += adj_sxx_x + adj_szz_x + adj_sxz_z
        adj_vz += adj_sxx_z + adj_szz_z + adj_sxz_x

        adj_states["csM"][valid] += (vx_x + vz_z) * adj_sxx[valid]
        adj_states["csM"][valid] += (vx_x + vz_z) * adj_szz[valid]
        adj_states["csu"][valid] += - 2.0 * vz_z * adj_sxx[valid]
        adj_states["csu"][valid] += - 2.0 * vx_x * adj_szz[valid]
        adj_states["csu"][valid] += (vx_z + vz_x) * adj_sxz[valid]

        return adj_states


class UpdateStress2(UpdateStress):


    def adjoint(self, adj_states, states, **kwargs):
        """
        Adjoint update the velocity for 2D P-SV isotropic elastic wave
        propagation.

        The transpose of the forward:
            [vx'     [ 1    0      -Dpx csM     -Dpx(csM - 2csu) -Dmz csu [vx'
             vz'       0    1  -Dpz (csM - 2csu)    -Dpz csM     -Dmx csu  vz'
             sxx'  =   0    0          1               0           0   =   sxx'
             szz'      0    0          0               1           0       szz'
             sxz']     0    0          0               0           1  ]    sxz']
             :param **kwargs:
        """
        adj_sxx = adj_states["sxx"]
        adj_szz = adj_states["szz"]
        adj_sxz = adj_states["sxz"]
        adj_vx = adj_states["vx"]
        adj_vz = adj_states["vz"]

        vx = states["vx"]
        vz = states["vz"]

        csu = states["csu"]
        csM = states["csM"]

        valid = self.grids["vx"].valid
        csu0 = np.zeros_like(csu)
        csu0[valid] = csu[valid]
        csM0 = np.zeros_like(csM)
        csM0[valid] = csM[valid]

        adj_sxx_x = -Dpx(csM0 * adj_sxx)
        adj_sxx_z = -Dpz((csM0 - 2.0 * csu0) * adj_sxx)
        adj_szz_x = -Dpx((csM0 - 2.0 * csu0) * adj_szz)
        adj_szz_z = -Dpz(csM0 * adj_szz)
        adj_sxz_x = -Dmx(csu0 * adj_sxz)
        adj_sxz_z = -Dmz(csu0 * adj_sxz)

        vx_x = Dmx(vx)
        vx_z = Dpz(vx)
        vz_x = Dpx(vz)
        vz_z = Dmz(vz)

        adj_vx[valid] += adj_sxx_x + adj_szz_x + adj_sxz_z
        adj_vz[valid] += adj_sxx_z + adj_szz_z + adj_sxz_x

        adj_states["csM"][valid] += (vx_x + vz_z) * adj_sxx[valid]
        adj_states["csM"][valid] += (vx_x + vz_z) * adj_szz[valid]
        adj_states["csu"][valid] += - 2.0 * vz_z * adj_sxx[valid]
        adj_states["csu"][valid] += - 2.0 * vx_x * adj_szz[valid]
        adj_states["csu"][valid] += (vx_z + vz_x) * adj_sxz[valid]

        return adj_states


class ZeroBoundary(Function):
    def __init__(self, required_states, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = required_states

    def forward(self, states, **kwargs):

        for el in self.required_states:
            mask = np.ones(self.grids[el].shape, np.bool)
            mask[self.grids[el].valid] = 0
            states[el][mask] = 0

        return states

    def adjoint(self, adj_states, states, rec_pos=(), t=0, **kwargs):

        for el in self.required_states:
            mask = np.ones(self.grids[el].shape, np.bool)
            mask[self.grids[el].valid] = 0
            adj_states[el][mask] = 0

        return adj_states


class Cerjan(Function):

    def __init__(self, grids=None, freesurf=False, abpc=4.0, nab=2,
                 required_states=(), **kwargs):
        super().__init__(grids, **kwargs)
        self.abpc = abpc
        self.nab = nab
        self.required_states = required_states
        self.updated_states = required_states
        self.taper = np.exp(np.log(1.0-abpc/100)/nab**2 * np.arange(nab) **2)
        #self.taper = np.concatenate([self.taper,  self.taper[-pad:][::-1]])
        self.taper = np.expand_dims(self.taper, -1)
        self.freesurf = freesurf

    @property
    def updated_regions(self):
        regions = []
        pad = self.grids[self.updated_states[0]].pad
        ndim = len(self.grids[self.updated_states[0]].shape)
        b = self.nab + pad
        for dim in range(ndim):
            region = [Ellipsis for _ in range(ndim)]
            region[dim] = slice(pad, b)
            if dim != 0 or not self.freesurf:
                regions.append(region)
            region = [Ellipsis for _ in range(ndim)]
            region[dim] = slice(-b, -pad)
            regions.append(tuple(region))
        return {el: regions for el in self.updated_states}

    def forward(self, states, **kwargs):
        pad = self.grids[self.updated_states[0]].pad
        for el in self.required_states:
            if not self.freesurf:
                states[el][pad:self.nab+pad, :] *= self.taper[::-1]
            states[el][-self.nab-pad:-pad, :] *= self.taper

            tapert = np.transpose(self.taper)
            states[el][:, pad:self.nab+pad] *= tapert[:, ::-1]
            states[el][:, -self.nab-pad:-pad] *= tapert

        return states

    def adjoint(self, adj_states, states, **kwargs):

        return self.forward(adj_states, **kwargs)


class Receiver(ReversibleFunction):

    def __init__(self, required_states, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = required_states
        self.updated_states = [el+"out" for el in required_states]
        self.required_states += self.updated_states
        self.default_grids = {el: "gridout" for el in self.updated_states}

    def forward(self, states, rec_pos=(), t=0, **kwargs):

        inds = {}
        for r in rec_pos:
            if r["type"] in inds:
                inds[r["type"]] += 1
            else:
                inds[r["type"]] = 0
            states[r["type"]+"out"][t, inds[r["type"]]] += states[r["type"]][r["z"], r["x"]]
        return states

    def adjoint(self, adj_states, states, rec_pos=(), t=0, **kwargs):

        inds = {}
        for r in rec_pos:
            if r["type"] in inds:
                inds[r["type"]] += 1
            else:
                inds[r["type"]] = 0
            adj_states[r["type"]][r["z"], r["x"]] += adj_states[r["type"]+"out"][t, inds[r["type"]]]
        return adj_states

    def backward(self, states, rec_pos=(), t=0, **kwargs):
        inds = {}
        for r in rec_pos:
            if r["type"] in inds:
                inds[r["type"]] += 1
            else:
                inds[r["type"]] = 0
            states[r["type"]+"out"][t, inds[r["type"]]] -= states[r["type"]][r["z"], r["x"]]
        return states


class Source(ReversibleFunction):

    def __init__(self, required_states, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = required_states

    def forward(self, states, src_pos=(), backpropagate=False, t=0, **kwargs):

        if backpropagate:
            sign = -1.0
        else:
            sign = 1.0
        for ii, s in enumerate(src_pos):
            states[s["type"]][s["pos"]] += sign * s["signal"][t]

        return states

    def linear(self, dstates, states, **kwargs):
        return dstates

    def adjoint(self, adj_states, states, **kwargs):
        return adj_states


class FreeSurface(Function):

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "csu", "csM"]
        self.updated_states = ["sxx", "szz", "sxz"]
        self.default_grids = {"vx": "gridfd",
                              "vz": "gridfd",
                              "sxx": "gridfd",
                              "szz": "gridfd",
                              "sxz": "gridfd",
                              "csu": "gridpar",
                              "csM": "gridpar"}

    def __call__(self, states, initialize=True, **kwargs):

        if initialize:
            states = self.initialize(states)
        kwargs = self.make_kwargs_compatible(**kwargs)
        saved = {}
        for el in self.updated_states:
            pad = self.grids[el].pad
            saved[el] = states[el][:pad+1, :].copy()
        self._forward_states.append(saved)

        return self.forward(states, **kwargs)

    def forward(self, states, **kwargs):

        sxx = states["sxx"]
        szz = states["szz"]
        sxz = states["sxz"]
        vx = states["vx"]
        vz = states["vz"]

        csu = states["csu"]
        csM = states["csM"]

        pad = self.grids["vx"].pad
        szz[pad:pad+1, :] = 0.0
        szz[:pad, :] = -szz[pad+1:2*pad+1, :][::-1, :]
        sxz[:pad, :] = -sxz[pad+1:2*pad+1, :][::-1, :]

        vxx = Dmx(vx)[:1, :]
        vzz = Dmz(vz)[:1, :]
        f = csu[pad:pad+1, pad:-pad] * 2.0
        g = csM[pad:pad+1, pad:-pad]
        h = -((g - f) * (g - f) * vxx / g) - ((g - f) * vzz)
        sxx[pad:pad+1, pad:-pad] += h

        return states

    def linear(self, dstates, states, **kwargs):

        self.forward({"vx": dstates["vx"],
                      "vz": dstates["vz"],
                      "sxx": dstates["sxx"],
                      "szz": dstates["szz"],
                      "sxz": dstates["sxz"],
                      "csu": states["csu"],
                      "csM": states["csM"],
                      "cv": states["cv"]})

        vx = states["vx"]
        vz = states["vz"]
        dcsu = dstates["csu"]
        dcsM = dstates["csM"]
        csu = states["csu"]
        csM = states["csM"]
        pad = self.grids["vx"].pad

        vxx = Dmx(vx)[:1, :]
        vzz = Dmz(vz)[:1, :]
        df = dcsu[pad:pad+1, pad:-pad] * 2.0
        dg = dcsM[pad:pad+1, pad:-pad]
        f = csu[pad:pad+1, pad:-pad] * 2.0
        g = csM[pad:pad+1, pad:-pad]
        dh = (2.0 * (g - f) * vxx / g + vzz) * df
        dh += (-2.0 * (g - f) * vxx / g + (g - f)**2 / g**2 * vxx - vzz) * dg
        # dh = vzz * df - vzz * dg
        dstates["sxx"][pad:pad+1, pad:-pad] += dh

        return dstates

    def adjoint(self, adj_states, states, **kwargs):

        adj_sxx = adj_states["sxx"]
        adj_sxz = adj_states["sxz"]
        adj_szz = adj_states["szz"]
        adj_vx = adj_states["vx"]
        adj_vz = adj_states["vz"]
        csu = states["csu"]
        csM = states["csM"]

        pad = self.grids["vx"].pad
        valid = self.grids["vx"].valid

        adj_szz[pad:pad+1, :] = 0.0
        adj_szz[pad+1:2*pad+1, :] += -adj_szz[:pad, :][::-1, :]
        adj_szz[:pad, :] = 0.0
        adj_sxz[pad+1:2*pad+1, :] += -adj_sxz[:pad, :][::-1, :]
        adj_sxz[:pad, :] = 0

        f = csu * 2.0
        g = csM
        hx = -((g - f) * (g - f) / g)
        hz = -(g - f)
        hx0 = np.zeros_like(csu)
        hx0[pad:pad+1, pad:-pad] = hx[pad:pad+1, pad:-pad]
        hz0 = np.zeros_like(csM)
        hz0[pad:pad+1, pad:-pad] = hz[pad:pad+1, pad:-pad]
        adj_sxx_x = Dmx_adj(hx0 * adj_sxx)
        adj_sxx_z = Dmz_adj(hz0 * adj_sxx)
        adj_vx[:2*pad, :] += adj_sxx_x[:2*pad, :]
        adj_vz[:2*pad, :] += adj_sxx_z[:2*pad, :]

        vx = states["vx"]
        vz = states["vz"]
        vxx = Dmx(vx)[:1, :]
        vzz = Dmz(vz)[:1, :]
        f = f[pad:pad+1, pad:-pad]
        g = g[pad:pad+1, pad:-pad]
        adj_states["csM"][pad:pad+1, pad:-pad] += (-2.0 * (g - f) * vxx / g + (g - f)**2 / g**2 * vxx - vzz) * adj_sxx[pad:pad+1, pad:-pad]
        adj_states["csu"][pad:pad+1, pad:-pad] += 2.0 * (2.0 * (g - f) * vxx / g + vzz) * adj_sxx[pad:pad+1, pad:-pad]
        return adj_states

    def backward(self, states, **kwargs):

        torestore = self._forward_states.pop()
        for el in torestore:
            pad = self.grids[el].pad
            states[el][:pad+1, :] = torestore[el]

        return states


class FreeSurface2(ReversibleFunction):
    #TODO Does not pass linear and dot product tests
    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "csu", "csM", "cv"]
        self.updated_states = ["sxx", "szz", "vx", "vz"]
        self.default_grids = {"vx": "gridfd",
                              "vz": "gridfd",
                              "sxx": "gridfd",
                              "szz": "gridfd",
                              "sxz": "gridfd",
                              "cv": "gridpar",
                              "csu": "gridpar",
                              "csM": "gridpar"}

    def forward(self, states, backpropagate=False, **kwargs):

        sxx = states["sxx"]
        szz = states["szz"]
        sxz = states["sxz"]
        vx = states["vx"]
        vz = states["vz"]

        csu = states["csu"]
        csM = states["csM"]
        cv = states["cv"]

        pad = self.grids["vx"].pad
        shape = self.grids["vx"].shape
        if backpropagate:
            sign = -1
        else:
            sign = 1

        def fun1():
            vxx = Dmx(vx)[:1, :]
            vzz = Dmz(vz)[:1, :]
            f = csu[pad:pad+1, pad:-pad] * 2.0
            g = csM[pad:pad+1, pad:-pad]
            h = -((g - f) * (g - f) * vxx / g) - ((g - f) * vzz)
            sxx[pad:pad+1, pad:-pad] += sign * h

            szz[pad:pad+1, pad:-pad] += sign * -((g*(vxx+vzz))-(f*vxx))

        def fun2():
            szz_z = np.zeros((pad, shape[1]-2*pad))
            hc = [1.1382, -0.046414]
            for i in range(pad):
                for j in range(i+1, pad):
                    szz_z[i, :] += hc[j] * szz[pad+j-i, pad:-pad]

            sxz_z = np.zeros((pad, shape[1]-2*pad))
            for i in range(pad):
                for j in range(i, pad):
                    sxz_z[i, :] += hc[j] * sxz[pad+j-i+1, pad:-pad]
            vx[pad:2*pad, pad:-pad] += sign * sxz_z * cv[pad:2*pad, pad:-pad]
            vz[pad:2*pad, pad:-pad] += sign * szz_z * cv[pad:2*pad, pad:-pad]

        if backpropagate:
            fun2()
            fun1()
        else:
            fun1()
            fun2()

        return states

    def linear(self, dstates, states, **kwargs):

        self.forward({"vx": dstates["vx"],
                      "vz": dstates["vz"],
                      "sxx": dstates["sxx"],
                      "szz": dstates["szz"],
                      "sxz": dstates["sxz"],
                      "csu": states["csu"],
                      "csM": states["csM"],
                      "cv": states["cv"]})


        dvx = dstates["vx"]
        dvz = dstates["vz"]
        dszz = dstates["szz"]
        dsxx = dstates["sxx"]
        dsxz = dstates["sxz"]
        dcsu = dstates["csu"]
        dcsM = dstates["csM"]
        dcv = dstates["cv"]
        vx = states["vx"]
        vz = states["vz"]
        szz = states["szz"]
        sxz = states["sxz"]
        csu = states["csu"]
        csM = states["csM"]
        cv = states["cv"]
        pad = self.grids["vx"].pad
        shape = self.grids["vx"].shape

        vxx = Dmx(vx)[:1, :]
        vzz = Dmz(vz)[:1, :]
        dvxx = Dmx(dvx)[:1, :]
        dvzz = Dmz(dvz)[:1, :]
        df = dcsu[pad:pad+1, pad:-pad] * 2.0
        dg = dcsM[pad:pad+1, pad:-pad]
        f = csu[pad:pad+1, pad:-pad] * 2.0
        g = csM[pad:pad+1, pad:-pad]

        dsxx[pad:pad+1, pad:-pad] += -((g - f) * (g - f) * dvxx / g) - ((g - f) * dvzz)
        dszz[pad:pad+1, pad:-pad] += -((g*(dvxx+dvzz))-(f*dvxx))
        dsxx[pad:pad+1, pad:-pad] += \
                                       (2.0 * (g - f) * vxx / g + vzz) * df \
                                     + (-2.0 * (g - f) * vxx / g
                                     + (g - f)*(g - f) / g / g * vxx - vzz) * dg
        dszz[pad:pad+1, pad:-pad] += -((dg*(vxx+vzz))-(df*vxx))


        dszz_z = np.zeros((pad, shape[1]-2*pad))
        hc = [1.1382, -0.046414]
        for i in range(pad):
            for j in range(i+1, pad):
                dszz_z[i, :] += hc[j] * dszz[pad+j-i, pad:-pad]
        dsxz_z = np.zeros((pad, shape[1]-2*pad))
        for i in range(pad):
            for j in range(i, pad):
                dsxz_z[i, :] += hc[j] * dsxz[pad+j-i+1, pad:-pad]
        dvx[pad:2*pad, pad:-pad] += dsxz_z * cv[pad:2*pad, pad:-pad]
        dvz[pad:2*pad, pad:-pad] += dszz_z * cv[pad:2*pad, pad:-pad]

        szz_z = np.zeros((pad, shape[1]-2*pad))
        for i in range(pad):
            for j in range(i+1, pad):
                szz_z[i, :] += hc[j] * szz[pad+j-i, pad:-pad]
        sxz_z = np.zeros((pad, shape[1]-2*pad))
        for i in range(pad):
            for j in range(i, pad):
                sxz_z[i, :] += hc[j] * sxz[pad+j-i+1, pad:-pad]
        dvx[pad:2*pad, pad:-pad] += sxz_z * dcv[pad:2*pad, pad:-pad]
        dvz[pad:2*pad, pad:-pad] += szz_z * dcv[pad:2*pad, pad:-pad]

        return dstates

    def adjoint(self, adj_states, states, **kwargs):

        adj_sxx = adj_states["sxx"]
        adj_sxz = adj_states["sxz"]
        adj_szz = adj_states["szz"]
        adj_vx = adj_states["vx"]
        adj_vz = adj_states["vz"]
        vx = states["vx"]
        vz = states["vz"]
        szz = states["szz"]
        sxz = states["sxz"]
        csu = states["csu"]
        csM = states["csM"]
        cv = states["cv"]

        pad = self.grids["vx"].pad
        shape = self.grids["vx"].shape

        szz_z = np.zeros((3*pad, shape[1]))
        szz_z[:pad, :] = -szz[pad+1:2*pad+1, :][::-1, :]
        szz_z = Dpz(szz_z)
        sxz_z = np.zeros((3*pad, shape[1]))
        sxz_z[:pad, :] = -sxz[pad+1:2*pad+1, :][::-1, :]
        sxz_z = Dmz(sxz_z)
        adj_states["cv"][pad:2*pad, pad:-pad] += sxz_z * adj_vx[pad:2*pad, pad:-pad]
        adj_states["cv"][pad:2*pad, pad:-pad] += szz_z * adj_vz[pad:2*pad, pad:-pad]

        adj_vx_z = np.zeros((3*pad, shape[1]))
        hc = [0.046414, -1.1382]
        for ii in range(pad):
            for jj in range(ii+1):
                adj_vx_z[ii, :] += hc[jj]*adj_vx[pad+ii-jj, :] * cv[pad+ii-jj, :]
        adj_sxz[pad+1:2*pad+1, pad:-pad] += -adj_vx_z[:pad, :][::-1, pad:-pad]

        adj_vz_z = np.zeros((3*pad, shape[1]))
        hc = [0.046414, -1.1382]
        for ii in range(1, pad):
            for jj in range(ii):
                adj_vz_z[ii, :] += hc[jj]*adj_vz[pad+ii-jj-1, :] * cv[pad+ii-jj-1, :]
        adj_szz[pad+1:2*pad+1, pad:-pad] += -adj_vz_z[:pad, :][::-1, pad:-pad]

        adj_szz[pad:pad+1, :] = 0.0

        f = csu * 2.0
        g = csM
        hx = -((g - f) * (g - f) / g)
        hz = -(g - f)
        hx0 = np.zeros_like(csu)
        hx0[pad:pad+1, pad:-pad] = hx[pad:pad+1, pad:-pad]
        hz0 = np.zeros_like(csM)
        hz0[pad:pad+1, pad:-pad] = hz[pad:pad+1, pad:-pad]
        adj_sxx_x = -Dpx(hx0 * adj_sxx)
        adj_sxx_z = -Dpz(hz0 * adj_sxx)
        adj_vx[pad:2*pad, pad:-pad] += adj_sxx_x[:pad, :]
        adj_vz[pad:2*pad, pad:-pad] += adj_sxx_z[:pad, :]

        vxx = Dmx(vx)[:1, :]
        vzz = Dmz(vz)[:1, :]
        f = f[pad:pad+1, pad:-pad]
        g = g[pad:pad+1, pad:-pad]
        adj_states["csM"][pad:pad+1, pad:-pad] += (-2.0 * (g - f) * vxx / g + (g - f)**2 / g**2 * vxx - vzz) * adj_sxx[pad:pad+1, pad:-pad]
        adj_states["csu"][pad:pad+1, pad:-pad] += 2.0 * (2.0 * (g - f) * vxx / g + vzz) * adj_sxx[pad:pad+1, pad:-pad]

        return adj_states


class ScaledParameters(ReversibleFunction):

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["cv", "csu", "csM"]
        self.updated_states = ["cv", "csu", "csM"]
        self.sc = 1.0
        self.default_grids = {"cv": "gridpar",
                              "csu": "gridpar",
                              "csM": "gridpar"}

    @staticmethod
    def scale(M, dt, dx):
        return int(np.log2(np.max(M) * dt / dx))

    def forward(self, states, dt=0.1, dx=2.0, **kwargs):

        vs = states["csu"]
        vp = states["csM"]
        rho = states["cv"]

        M = (vp**2 * rho)
        mu = (vs**2 * rho)
        self.sc = sc = self.scale(M, dt, dx)
        cv = 2 ** sc * dt / dx / rho
        csM = dt / dx * M * 2 ** -sc
        csu = dt / dx * mu * 2 ** -sc

        states["cv"] = cv
        states["csM"] = csM
        states["csu"] = csu

        return states

    def linear(self, dstates, states, dt=0.1, dx=2.0, **kwargs):

        vs = states["csu"]
        vp = states["csM"]
        rho = states["cv"]

        dvs = dstates["csu"]
        dvp = dstates["csM"]
        drho = dstates["cv"]

        dM = 2.0 * (vp * rho) * dvp + vp**2 * drho
        dmu = 2.0 * (vs * rho) * dvs + vs**2 * drho
        sc = self.sc
        dcv = - 2 ** sc * dt / dx / rho**2 * drho
        dcsM = dt / dx * dM * 2 ** -sc
        dcsu = dt / dx * dmu * 2 ** -sc

        dstates["cv"] = dcv
        dstates["csM"] = dcsM
        dstates["csu"] = dcsu

        return dstates

    def adjoint(self, adj_states, states, dt=0.1, dx=2.0, **kwargs):

        vs = states["csu"]
        vp = states["csM"]
        rho = states["cv"]

        adj_csu = adj_states["csu"]
        adj_csM = adj_states["csM"]
        adj_cv = adj_states["cv"]
        sc = self.sc
        adj_csM = dt / dx * adj_csM * 2 ** -sc
        adj_csu = dt / dx * adj_csu * 2 ** -sc

        adj_vp = 2.0 *(vp * rho) * adj_csM
        adj_vs = 2.0 *(vs * rho) * adj_csu
        adj_rho = - 2 ** sc * dt / dx / rho**2 * adj_cv
        adj_rho += vp**2 * adj_csM + vs**2 * adj_csu

        adj_states["csu"] = adj_vs
        adj_states["csM"] = adj_vp
        adj_states["cv"] = adj_rho

        return adj_states

    def backward(self, states, dt=0.1, dx=2.0, **kwargs):

        cv = states["cv"]
        csM = states["csM"]
        csu = states["csu"]

        sc = self.sc
        rho = 2 ** sc * dt / dx / cv
        M = dx / dt * csM * 2 ** sc
        mu = dx / dt * csu * 2 ** sc

        states["csu"] = np.sqrt(mu / rho)
        states["csM"] = np.sqrt(M / rho)
        states["cv"] = rho

        return states


class PrecisionTester(Function):

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz"]
        self.updated_states = []

    def forward(self, states, **kwargs):
        with half_precision:
            for el in self.required_states:
                for ii in range(states[el].shape[0]):
                    for jj in range(states[el].shape[1]):
                        states[el][ii, jj] = BigFloat(float(states[el][ii, jj]))

        return states


def ricker(f0, dt, NT):
    tmin = -2 / f0
    t = np.zeros((NT, 1))
    t[:, 0] = tmin + np.arange(0, NT * dt, dt)
    pf = math.pow(math.pi, 2) * math.pow(f0, 2)
    ricker = np.multiply((1.0 - 2.0 * pf * np.power(t, 2)), np.exp(-pf * np.power(t, 2)))

    return ricker


def define_psv(grid2D, gridout, nab, nt):

    gridpar = copy(grid2D)
    gridpar.zero_boundary = False
    defs = {"gridfd": grid2D, "gridpar": gridpar, "gridout": gridout}

    stepper = Sequence([Source(required_states=["vx"]),
                        UpdateVelocity2(),
                        Cerjan(required_states=["vx", "vz"], freesurf=1, nab=nab),
                        Receiver(required_states=["vx", "vz"]),
                        UpdateStress2(),
                        FreeSurface2(),
                        Cerjan(required_states=["sxx", "szz", "sxz"],
                               freesurf=1, nab=nab),
                        ])
    prop = Propagator(stepper, nt)
    psv2D = Sequence([ScaledParameters(),
                      prop],
                     grids=defs)
    return psv2D

if __name__ == '__main__':

    grid1D = Grid(shape=(10,))
    defs = {"vx": grid1D,
            "vz": grid1D,}
    x = Variable("x", shape=(10, 1), initialize_method="random")
    b = Variable("b", shape=(10, 1), initialize_method="random")
    y = Variable("y", shape=(10, 1), initialize_method="random")
    matmul = RandKernel()
    # matmul(x, b, y)
    # matmul.call_linear(x, b, y)
    # matmul.gradient(x, b, y)
    # print(x.data)
    # print(x.lin)
    # print(x.grad)
    matmul.backward_test(x, b, y)
    # matmul.linear_test()
    # matmul.dot_test()
    # matmul2 = RandKernel(grid=grid1D)
    # seq = Sequence([matmul, matmul2], grids=matmul.grids)
    # seq.dot_test()
    # prop = Propagator(seq, 5)
    # prop.dot_test()
    # der = Derivative({"vx": Grid(shape=(10, 10), pad=2)})
    # der.dot_test()
    # div = Division(grids=defs)
    # div.linear_test()
    # div.backward_test()

    # nrec = 1
    # nt = 3
    # nab = 2
    # grid2D = Grid(shape=(10, 10), type=np.float64, zero_boundary=True)
    # gridout = Grid(shape=(nt, nrec), type=np.float64)
    # psv2D = define_psv(grid2D, gridout, nab, nt)
    #
    # psv2D.backward_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
    #                     src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])
    # psv2D.linear_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
    #                   src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])
    # psv2D.dot_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
    #                src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])
    #
    # nrec = 1
    # nt = 7500
    # nab = 16
    # rec_pos = [{"type": "vx", "z": 3, "x": x} for x in range(50, 250)]
    # rec_pos += [{"type": "vz", "z": 3, "x": x} for x in range(50, 250)]
    # grid2D = Grid(shape=(160, 300), type=np.float64)
    # gridout = Grid(shape=(nt, len(rec_pos)//2), type=np.float64)
    # psv2D = define_psv(grid2D, gridout, nab, nt)
    # dx = 1.0
    # dt = 0.0001
    #
    # csu = np.full(grid2D.shape, 300.0)
    # cv = np.full(grid2D.shape, 1800.0)
    # csM = np.full(grid2D.shape, 1500.0)
    # csu[80:, :] = 600
    # csM[80:, :] = 2000
    # cv[80:, :] = 2000
    # csu0 = csu.copy()
    # csu[5:10, 145:155] *= 1.05
    #
    # states = psv2D({"cv": cv,
    #                 "csu": csu,
    #                 "csM": csM},
    #                dx=dx,
    #                dt=dt,
    #                rec_pos=rec_pos,
    #                src_pos=[{"type": "vz", "pos": (2, 50), "signal": ricker(10, dt, nt)}])
    # plt.imshow(states["vx"])
    # plt.show()
    # #
    # vxobs = states["vxout"]
    # vzobs = states["vzout"]
    # clip = 0.01
    # vmin = np.min(states["vzout"]) * 0.1
    # vmax=-vmin
    # plt.imshow(states["vzout"], aspect="auto", vmin=vmin, vmax=vmax)
    # plt.show()
    #
    # states = psv2D({"cv": cv,
    #                 "csu": csu0,
    #                 "csM": csM},
    #                dx=dx,
    #                dt=dt,
    #                rec_pos=rec_pos,
    #                src_pos=[{"type": "vz", "pos": (3, 50), "signal": ricker(10, dt, nt)}])
    #
    # vxmod = states["vxout"]
    # vzmod = states["vzout"]
    # clip = 0.01
    # vmin = np.min(vxmod) * 0.1
    # vmax=-vmin
    # plt.imshow(vxmod, aspect="auto", vmin=vmin, vmax=vmax)
    # plt.show()
    # grads, states = psv2D.gradient({"vxout": vxmod-vxobs, "vzout": vzmod-vzobs}, states,
    #                                dx=dx,
    #                                dt=dt,
    #                                rec_pos=rec_pos,
    #                                src_pos=[{"type": "vz", "pos": (3, 50), "signal": ricker(10, dt, nt)}])
    #
    # plt.imshow(grads["csu"])
    # plt.show()

