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
from copy import copy, deepcopy
from inspect import signature, Parameter
from tape import Variable, Function, TapedFunction, ReversibleFunction
#import os.path
from collections import OrderedDict

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

    def __init__(self, shape=(10, 10), pad=2, dh=1, dt=1, nt=1,
                 dtype=np.float32, zero_boundary=False, **kwargs):
        self.shape = shape
        self.pad = pad
        self.dh = dh
        self.dt = dt
        self.nt = nt
        self.dtype = dtype
        self.eps = np.finfo(dtype).eps
        self.smallest = np.nextafter(dtype(0), dtype(1))
        self.zero_boundary = zero_boundary

    @property
    def valid(self):
        return tuple([slice(self.pad, -self.pad)] * len(self.shape))

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

#
# class Function(TapeHolder):
#     """
#     Kernel implementing forward, linear and adjoint modes.
#     """
#
#     def __init__(self):
#
#         self._forward_states = None
#         self.ncall = 0
#         self.signature = signature(self.forward)
#         self.required_states = [name for name, par
#                                 in signature(self.forward).parameters.items()
#                                 if par.kind == Parameter.POSITIONAL_OR_KEYWORD]
#         self.updated_states = []
#         if not hasattr(self, 'copy_states'):
#             self.copy_states = {}
#         if not hasattr(self, 'zeroinit_states'):
#             self.zeroinit_states = []
#
#     def __call__(self, *args, initialize=True, cache_states=True, **kwargs):
#
#         kwargs = self.make_kwargs_compatible(**kwargs)
#         if initialize:
#             self.initialize(*args, empty_cache=cache_states, **kwargs)
#         if cache_states:
#             self.tape.append(self)
#             self.cache_states(*args, **kwargs)
#
#         return self.forward(*args, **kwargs)
#
#     def call_linear(self, *args, initialize=True, cache_states=True, **kwargs):
#
#         kwargs = self.make_kwargs_compatible(**kwargs)
#         if initialize:
#             self.initialize(*args, empty_cache=cache_states, **kwargs)
#         if cache_states:
#             self.cache_states(*args, **kwargs)
#         self.linear(*args, **kwargs)
#         return self.forward(*args, **kwargs)
#
#     def gradient(self, *args, initialize=True, **kwargs):
#
#         kwargs = self.make_kwargs_compatible(**kwargs)
#         if initialize:
#             self.initialize(*args,  adjoint=True, empty_cache=False, **kwargs)
#         self.backward(*args, **kwargs)
#         return self.adjoint(*args, **kwargs)
#
#     def cache_states(self, *args, **kwargs):
#         vars = self.arguments(*args, **kwargs)
#         for el in self.updated_states:
#             regions = self.updated_regions(vars[el])
#             if regions is None:
#                 self._forward_states[el][..., self.ncall] = vars[el].data
#             else:
#                 for ii, region in enumerate(regions):
#                     self._forward_states[el][ii][..., self.ncall] = vars[el].data[region]
#         self.ncall += 1
#
#     def updated_regions(self, var):
#         return {}
#
#     def initialize(self, *args, empty_cache=True, method="zero", adjoint=False,
#                    copy_state=True, cache_size=1, **kwargs):
#
#         # if adjoint:
#         #     toinit = list(set(self.updated_states + self.required_states))
#         # else:
#         #     toinit = self.required_states
#         #     self.ncall = 0
#         #
#         # for el in toinit:
#         #     if el not in self.grids:
#         #         self.grids[el] = self.grids[self.default_grids[el]]
#         #
#         #     if el not in argins and el not in self.tape.vars:
#         #         if el not in self.zeroinit_states:
#         #             var = Variable(el,
#         #                            data=self.grids[el].initialize(method=method)
#         #                            )
#         #         else:
#         #             var = Variable(el,
#         #                            data=self.grids[el].initialize()
#         #                            )
#         #     elif el not in self.tape.vars:
#         #         if type(argins[el]) is Variable:
#         #             self.tape.vars[el] = argins[el]
#         #         else:
#         #             if type(argins[el]) is not self.grids[el].backend:
#         #                 data = self.grids[el].initialize(data=argins[el])
#         #             else:
#         #                 data = argins[el]
#         #             self.tape.vars[el] = Variable(el, data=data)
#         #     else:
#         #         raise NotImplemented("Variable already in tape:"
#         #                              " cannot overwrite")
#
#         # if copy_state:
#         #     for el in self.copy_states:
#         #         if el not in self.tape.vars:
#         #             self.tape.vars[el] = self.tape.vars[self.copy_states[el]]
#
#         vars = self.arguments(*args, **kwargs)
#         if not self.updated_states:
#             self.updated_states = vars.keys()
#         if empty_cache:
#             self._forward_states = {}
#             for el in self.updated_states:
#                 regions = self.updated_regions(vars[el])
#                 self._forward_states[el] = vars[el].create_cache(regions=regions,
#                                                                  cache_size=cache_size)
#             self.ncall = 0
#
#     def make_kwargs_compatible(self, **kwargs):
#         return kwargs
#
#     def forward(self, *args, **kwargs):
#         """
#         Applies the forward kernel.
#
#         :param **kwargs:
#         :param states: A dict containing the variables describing the state of a
#                        dynamical system. The state variables are updated by
#                        the forward, but they keep the same dimensions.
#         :return:
#             states:     A dict containing the updated states.
#         """
#         raise NotImplemented
#
#     def linear(self, *args, **kwargs):
#         """
#         Applies the linearized forward J = d forward / d states
#         to a state perturbation out = J * dstates
#         By default, forward is treated as a linear function
#         (self.linear = self.forward).
#
#         :param dstates: State perturbation
#         :param states:  State at which the forward is linearized
#         :param kwargs:
#         :return: J * dstates
#         """
#         raise NotImplemented
#
#     def adjoint(self, *args, **kwargs):
#         """
#         Applies the adjoint of the forward
#
#         :param **kwargs:
#         :param adj_states: A dict containing the adjoint of the forward states.
#                            Each elements has the same dimension as the forward
#                            state, as the forward kernel do not change the
#                            dimension of the state.
#         :param states: The states of the system, before calling forward.
#
#         :return:
#             adj_states All Variables modified by adjoint.
#         """
#         raise NotImplemented
#
#     def backward(self, *args, **kwargs):
#         """
#         Reconstruct the input states from the output of forward
#
#         :param **kwargs:
#         :param states: A dict containing the variables describing the state of a
#                       dynamical system. The state variables are updated by
#                       the forward, but they keep the same dimensions.
#         """
#
#         vars = self.arguments(*args, **kwargs)
#         self.ncall -= 1
#         for el in self._forward_states:
#             regions = self.updated_regions(vars[el])
#             if regions is None:
#                 vars[el].data = self._forward_states[el][..., self.ncall]
#             else:
#                 for ii, reg in enumerate(regions):
#                     vars[el].data[reg] = self._forward_states[el][ii][...,
#                                                                       self.ncall]
#
#     def backward_test(self, *args, **kwargs):
#
#         vars = self.arguments(*args, **kwargs)
#         vars0 = {name: copy(var) for name, var in vars.items()}
#         for name, var in vars.items():
#             if type(var) is Variable:
#                 var.data = var.initialize(method="random")
#                 vars0[name].data = var.data.copy()
#
#         self(*args, **kwargs)
#         self.backward(*args, **kwargs)
#
#         err = 0
#         scale = 0
#         for name, var in vars.items():
#             smallest = var.smallest
#             snp = vars0[name].data
#             bsnp = var.data
#             errii = snp - bsnp
#             err += np.sum(errii**2)
#             scale += np.sum((snp - np.mean(snp))**2) + smallest
#         err = err / scale
#         print("Backpropagation test for Kernel %s: %.15e"
#               % (self.__class__.__name__, err))
#
#         return err
#
#     def linear_test(self, *args, **kwargs):
#
#         vars = self.arguments(*args, **kwargs)
#         pargs = deepcopy(args)
#         pkwargs = deepcopy(kwargs)
#         pvars = self.arguments(*pargs, **pkwargs)
#         fargs = deepcopy(args)
#         fkwargs = deepcopy(kwargs)
#         fvars = self.arguments(*fargs, **fkwargs)
#         for name, var in vars.items():
#             var.data = var.initialize(method="random")
#             var.lin = var.initialize(method="random")
#             fvars[name].data = var.data.copy()
#             fvars[name].lin = var.lin.copy()
#         outs = self.call_linear(*fargs, **fkwargs)
#         try:
#             iter(outs)
#         except TypeError:
#             outs = (outs,)
#
#         errs = []
#         cond = True if vars else False
#         if not any([el not in self.zeroinit_states for el in vars]):
#             cond = False
#         eps = 1.0
#         while cond:
#             for name, var in vars.items():
#                 pvars[name].data = var.data + eps * var.lin
#             pouts = self(*pargs, **pkwargs)
#             try:
#                 iter(pouts)
#             except TypeError:
#                 pouts = (pouts,)
#
#             err = 0
#             scale = 0
#             for out, pout in zip(outs, pouts):
#                 err += np.sum((pout.data - out.data - eps * out.lin)**2)
#                 scale += np.sum((eps*(out.lin - np.mean(out.lin)))**2)
#             errs.append([err/(scale+out.smallest)])
#
#             eps /= 10.0
#             for el, var in vars.items():
#                 if el not in self.zeroinit_states:
#                     if np.max(eps*var.lin / (var.data+var.smallest)) < var.eps:
#                         cond = False
#                         break
#         try:
#             errmin = np.min(errs)
#             print("Linear test for Kernel %s: %.15e"
#                   % (self.__class__.__name__, errmin))
#         except ValueError:
#             errmin = 0
#             print("Linear test for Kernel %s: unable to perform"
#                   % (self.__class__.__name__))
#
#         return errmin
#
#     def dot_test(self, *args, **kwargs):
#         """
#         Dot product test for fstates, outputs = F(states)
#
#         dF = [dfstates/dstates     [dstates
#               doutputs/dstates]     dparams ]
#
#         dot = [adj_states  ^T [dfstates/dstates     [states
#                adj_outputs]    doutputs/dstates]   params]
#
#         """
#
#         vars = self.arguments(*args, **kwargs)
#         fargs = deepcopy(args)
#         fkwargs = deepcopy(kwargs)
#         fvars = self.arguments(*fargs, **fkwargs)
#         for name, var in vars.items():
#             var.data = var.initialize(method="random")
#             var.lin = var.initialize(method="random")
#             var.grad = var.initialize(method="random")
#             fvars[name].data = var.data.copy()
#             fvars[name].lin = var.lin.copy()
#             fvars[name].grad = var.grad.copy()
#         self.call_linear(*fargs, **fkwargs)
#         self.gradient(*fargs, **fkwargs)
#
#         prod1 = np.sum([np.sum(fvars[el].lin * vars[el].grad)
#                         for el in vars])
#         prod2 = np.sum([np.sum(fvars[el].grad * vars[el].lin)
#                         for el in vars])
#
#         print("Dot product test for Kernel %s: %.15e"
#               % (self.__class__.__name__, (prod1-prod2)/(prod1+prod2)))
#
#         return (prod1-prod2)/(prod1+prod2)
#
#     def arguments(self, *args, **kwargs):
#         a = self.signature.bind(*args, **kwargs)
#         a.apply_defaults()
#
#         out = {el: var for el, var in a.arguments.items()
#                if type(var) is Variable}
#         if "args" in a.arguments:
#             for ii, var in enumerate(a.arguments["args"]):
#                 if type(var) is Variable:
#                     out["arg"+str(ii)] = var
#         return out


class RandKernel(Function):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.updated_states = ["x", "y"]
        self.A1 = np.random.rand(10, 10)
        self.A2 = np.random.rand(10, 10)

    def forward(self, x, b, y):

        x.data = np.matmul(self.A1, x.data)
        y.data = np.matmul(self.A2, x.data) + b.data
        return x, y

    def linear(self, x, b, y):
        x.lin = np.matmul(self.A1, x.lin)
        y.lin = np.matmul(self.A2, x.lin) + b.lin
        return x, y

    def adjoint(self, x, b, y):

        x.grad += np.matmul(self.A2.T, y.grad)
        b.grad += y.grad
        y.grad = y.grad * 0
        x.grad = np.matmul(self.A1.T, x.grad)
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


class UpdateVelocity(ReversibleFunction):

    def __init__(self):
        super().__init__()
        self.updated_states = ["vx", "vz"]

    def forward(self, cv, vx, vz, sxx, szz, sxz, backpropagate=False, **kwargs):
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
        sxx_x = Dpx(sxx.data)
        szz_z = Dpz(szz.data)
        sxz_x = Dmx(sxz.data)
        sxz_z = Dmz(sxz.data)

        if not backpropagate:
            vx.data[vx.valid] += (sxx_x + sxz_z) * cv.data[vx.valid]
            vz.data[vz.valid] += (szz_z + sxz_x) * cv.data[vz.valid]
        else:
            vx.data[vx.valid] -= (sxx_x + sxz_z) * cv.data[vx.valid]
            vz.data[vz.valid] -= (szz_z + sxz_x) * cv.data[vz.valid]

        return vx, vz

    def linear(self, cv, vx, vz, sxx, szz, sxz, **kwargs):

        sxx_x = Dpx(sxx.lin)
        szz_z = Dpz(szz.lin)
        sxz_x = Dmx(sxz.lin)
        sxz_z = Dmz(sxz.lin)
        vx.lin[vx.valid] += (sxx_x + sxz_z) * cv.data[vx.valid]
        vz.lin[vz.valid] += (szz_z + sxz_x) * cv.data[vz.valid]

        sxx_x = Dpx(sxx.data)
        szz_z = Dpz(szz.data)
        sxz_x = Dmx(sxz.data)
        sxz_z = Dmz(sxz.data)

        vx.lin[vx.valid] += (sxx_x + sxz_z) * cv.lin[vx.valid]
        vz.lin[vz.valid] += (szz_z + sxz_x) * cv.lin[vz.valid]

        return vx, vz

    def adjoint(self, cv, vx, vz, sxx, szz, sxz, **kwargs):
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

        cv0 = np.zeros_like(cv.data)
        cv0[cv.valid] = cv[cv.valid]
        adj_vx_x = Dpx_adj(cv0 * vx.grad)
        adj_vx_z = Dmz_adj(cv0 * vx.grad)
        adj_vz_z = Dpz_adj(cv0 * vz.grad)
        adj_vz_x = Dmx_adj(cv0 * vz.grad)

        sxx_x = Dpx(sxx.grad)
        szz_z = Dpz(szz.grad)
        sxz_x = Dmx(sxz.grad)
        sxz_z = Dmz(sxz.grad)

        sxx.grad += adj_vx_x
        szz.grad += adj_vz_z
        sxz.grad += adj_vx_z + adj_vz_x

        cv.grad[cv.valid] += (sxx_x + sxz_z) * vx.grad[vx.valid]
        cv.grad[cv.valid] += (szz_z + sxz_x) * vz.grad[vz.valid]

        return sxx, szz, sxz, cv


class UpdateVelocity2(UpdateVelocity):

    def adjoint(self, cv, vx, vz, sxx, szz, sxz):
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

        cv0 = np.zeros_like(cv.data)
        cv0[cv.valid] = cv.data[cv.valid]
        adj_vx_x = -Dmx(cv0.data * vx.grad)
        adj_vx_z = -Dpz(cv0.data * vx.grad)
        adj_vz_z = -Dmz(cv0.data * vz.grad)
        adj_vz_x = -Dpx(cv0.data * vz.grad)

        sxx_x = Dpx(sxx.data)
        szz_z = Dpz(szz.data)
        sxz_x = Dmx(sxz.data)
        sxz_z = Dmz(sxz.data)

        sxx.grad[sxx.valid] += adj_vx_x
        szz.grad[szz.valid] += adj_vz_z
        sxz.grad[sxz.valid] += adj_vx_z + adj_vz_x

        cv.grad[cv.valid] += (sxx_x + sxz_z) * vx.grad[vx.valid]
        cv.grad[cv.valid] += (szz_z + sxz_x) * vz.grad[vz.valid]

        return sxx, szz, sxz, cv


class UpdateStress(ReversibleFunction):

    def __init__(self):
        super().__init__()
        self.updated_states = ["sxx", "szz", "sxz"]

    def forward(self, csM, csu, vx, vz, sxx, szz, sxz, backpropagate=False):
        """
        Update the velocity for 2D P-SV isotropic elastic wave propagation.

        In matrix form:
            [vx     [         1                0        0 0 0   [vx
             vz               0                1        0 0 0    vz
             sxx  =        csM Dmx     (csM - 2csu) Dmz 1 0 0 =  sxx
             szz      (csM - 2csu) Dmx      csM Dmz     0 1 0    szz
             sxz]          csu Dpz          csu Dpx     0 0 1]   sxz]
        """

        vx_x = Dmx(vx.data)
        vx_z = Dpz(vx.data)
        vz_x = Dpx(vz.data)
        vz_z = Dmz(vz.data)

        valid = vx.valid
        if not backpropagate:
            sxx.data[valid] += csM.data[valid] * (vx_x + vz_z) - 2.0 * csu.data[valid] * vz_z
            szz.data[valid] += csM.data[valid] * (vx_x + vz_z) - 2.0 * csu.data[valid] * vx_x
            sxz.data[valid] += csu.data[valid] * (vx_z + vz_x)
        else:
            sxx.data[valid] -= csM.data[valid] * (vx_x + vz_z) - 2.0 * csu.data[valid] * vz_z
            szz.data[valid] -= csM.data[valid] * (vx_x + vz_z) - 2.0 * csu.data[valid] * vx_x
            sxz.data[valid] -= csu.data[valid] * (vx_z + vz_x)

        return sxx, szz, sxz

    def linear(self, csM, csu, vx, vz, sxx, szz, sxz):

        vx_x = Dmx(vx.lin)
        vx_z = Dpz(vx.lin)
        vz_x = Dpx(vz.lin)
        vz_z = Dmz(vz.lin)

        valid = vx.valid
        sxx.lin[valid] += csM.data[valid] * (vx_x + vz_z) - 2.0 * csu.data[valid] * vz_z
        szz.lin[valid] += csM.data[valid] * (vx_x + vz_z) - 2.0 * csu.data[valid] * vx_x
        sxz.lin[valid] += csu.data[valid] * (vx_z + vz_x)

        vx_x = Dmx(vx.data)
        vx_z = Dpz(vx.data)
        vz_x = Dpx(vz.data)
        vz_z = Dmz(vz.data)

        sxx.lin[valid] += csM.lin[valid] * (vx_x + vz_z) - 2.0 * csu.lin[valid] * vz_z
        szz.lin[valid] += csM.lin[valid] * (vx_x + vz_z) - 2.0 * csu.lin[valid] * vx_x
        sxz.lin[valid] += csu.lin[valid] * (vx_z + vz_x)

        return sxx, szz, sxz

    def adjoint(self, csM, csu, vx, vz, sxx, szz, sxz):
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

        valid = vx.valid
        csu0 = np.zeros_like(csu.data)
        csu0[valid] = csu.data[valid]
        csM0 = np.zeros_like(csM.data)
        csM0[valid] = csM.data[valid]

        adj_sxx_x = Dmx_adj(csM0 * sxx.grad)
        adj_sxx_z = Dmz_adj((csM0 - 2.0 * csu0) * sxx.grad)
        adj_szz_x = Dmx_adj((csM0 - 2.0 * csu0) * szz.grad)
        adj_szz_z = Dmz_adj(csM0 * szz.grad)
        adj_sxz_x = Dpx_adj(csu0 * sxz.grad)
        adj_sxz_z = Dpz_adj(csu0 * sxz.grad)

        vx_x = Dmx(vx.data)
        vx_z = Dpz(vx.data)
        vz_x = Dpx(vz.data)
        vz_z = Dmz(vz.data)

        vx.grad += adj_sxx_x + adj_szz_x + adj_sxz_z
        vz.grad += adj_sxx_z + adj_szz_z + adj_sxz_x

        csM.grad[valid] += (vx_x + vz_z) * sxx.grad[valid]
        csM.grad[valid] += (vx_x + vz_z) * szz.grad[valid]
        csu.grad[valid] += - 2.0 * vz_z * sxx.grad[valid]
        csu.grad[valid] += - 2.0 * vx_x * szz.grad[valid]
        csu.grad[valid] += (vx_z + vz_x) * sxz.grad[valid]

        return vx, vz, csM, csu


class UpdateStress2(UpdateStress):

    def adjoint(self, csM, csu, vx, vz, sxx, szz, sxz):
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

        valid = vx.valid
        csu0 = np.zeros_like(csu.data)
        csu0[valid] = csu.data[valid]
        csM0 = np.zeros_like(csM.data)
        csM0[valid] = csM.data[valid]

        adj_sxx_x = -Dpx(csM0 * sxx.grad)
        adj_sxx_z = -Dpz((csM0 - 2.0 * csu0) * sxx.grad)
        adj_szz_x = -Dpx((csM0 - 2.0 * csu0) * szz.grad)
        adj_szz_z = -Dpz(csM0 * szz.grad)
        adj_sxz_x = -Dmx(csu0 * sxz.grad)
        adj_sxz_z = -Dmz(csu0 * sxz.grad)

        vx_x = Dmx(vx.data)
        vx_z = Dpz(vx.data)
        vz_x = Dpx(vz.data)
        vz_z = Dmz(vz.data)

        vx.grad[valid] += adj_sxx_x + adj_szz_x + adj_sxz_z
        vz.grad[valid] += adj_sxx_z + adj_szz_z + adj_sxz_x

        csM.grad[valid] += (vx_x + vz_z) * sxx.grad[valid] \
                         + (vx_x + vz_z) * szz.grad[valid]
        csu.grad[valid] += - 2.0 * vz_z * sxx.grad[valid] \
                         + - 2.0 * vx_x * szz.grad[valid] \
                         + (vx_z + vz_x) * sxz.grad[valid]

        return vx, vz, csM, csu


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

    def __init__(self, freesurf=False, abpc=4.0, nab=2, pad=2):
        super().__init__()
        self.abpc = abpc
        self.nab = nab
        self.pad = pad
        self.taper = np.exp(np.log(1.0-abpc/100)/nab**2 * np.arange(nab) **2)
        #self.taper = np.concatenate([self.taper,  self.taper[-pad:][::-1]])
        self.taper = np.expand_dims(self.taper, -1)
        self.freesurf = freesurf

    # def updated_regions(self, var):
    #     regions = []
    #     pad = self.pad
    #     ndim = len(var.shape)
    #     b = self.nab + pad
    #     for dim in range(ndim):
    #         region = [Ellipsis for _ in range(ndim)]
    #         region[dim] = slice(pad, b)
    #         if dim != 0 or not self.freesurf:
    #             regions.append(tuple(region))
    #         region = [Ellipsis for _ in range(ndim)]
    #         region[dim] = slice(-b, -pad)
    #         regions.append(tuple(region))
    #     return regions

    def forward(self, *args, direction="data"):
        pad = self.pad
        nab = self.nab
        for arg in args:
            d = getattr(arg, direction)
            if not self.freesurf:
                d[pad:nab+pad, :] *= self.taper[::-1]
            d[-nab-pad:-pad, :] *= self.taper

            tapert = np.transpose(self.taper)
            d[:, pad:nab+pad] *= tapert[:, ::-1]
            d[:, -nab-pad:-pad] *= tapert
        return args

    def linear(self, *args):
        return self.forward(*args, direction="lin")

    def adjoint(self, *args):
        return self.forward(*args, direction="grad")


class Receiver(ReversibleFunction):

    def forward(self, var, varout, rec_pos=(), t=0):
        for ii, r in enumerate(rec_pos):
            varout.data[t, ii] += var.data[r]
        return varout

    def linear(self, var, varout, rec_pos=(), t=0):
        for ii, r in enumerate(rec_pos):
            varout.lin[t, ii] += var.lin[r]
        return varout

    def adjoint(self, var, varout, rec_pos=(), t=0):
        for ii, r in enumerate(rec_pos):
            var.grad[r] += varout.grad[t, ii]
        return var

    def recover_states(self, initial_states, var, varout, rec_pos=(), t=0):
        for ii, r in enumerate(rec_pos):
            varout.data[t, ii] -= var.data[r]
        return varout


class PointForceSource(ReversibleFunction):

    def forward(self, var, src, src_pos=(), backpropagate=False):

        if backpropagate:
            sign = -1.0
        else:
            sign = 1.0
        for ii, s in enumerate(src_pos):
            var.data[s] += sign * src

        return var

    def linear(self, var, src, src_pos=()):
        return var

    def adjoint(self, var, src, src_pos=()):
        return var


class FreeSurface(Function):

    def __init__(self):
        super().__init__()
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "csu", "csM"]
        self.updated_states = ["sxx", "szz", "sxz"]


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
    def __init__(self):
        super().__init__()
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "csu", "csM", "cv"]
        self.updated_states = ["sxx", "szz", "vx", "vz"]

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

    def __init__(self, dt, dx):
        super().__init__()
        self.required_states = ["cv", "csu", "csM"]
        self.updated_states = ["cv", "csu", "csM"]
        self.sc = None
        self.dtdx = dt / dx

    def scale(self, M):
        return int(np.log2(np.max(M) * self.dtdx))

    def forward(self, vp, vs, rho, backpropagate=False):

        if not backpropagate:
            vp.data = (vp.data**2 * rho.data)
            vs.data = (vs.data**2 * rho.data)
            self.sc = sc = self.scale(vp.data)
            rho.data[rho.valid] = 2 ** sc * self.dtdx / rho.data[rho.valid]
            vp.data = self.dtdx * vp.data * 2 ** -sc
            vs.data = self.dtdx * vs.data * 2 ** -sc
        else:
            sc = self.sc
            rho.data[rho.valid] = 2 ** sc * self.dtdx / rho.data[rho.valid]
            vp.data = 1.0 / self.dtdx * vp.data * 2 ** sc
            vs.data = 1.0 / self.dtdx * vs.data * 2 ** sc

            vs.data[rho.valid] = np.sqrt(vs.data[rho.valid] / rho.data[rho.valid])
            vp.data[rho.valid] = np.sqrt(vp.data[rho.valid] / rho.data[rho.valid])

        return vp, vs, rho

    def linear(self, vp, vs, rho):

        vp.lin = 2.0 * (vp.data * rho.data) * vp.lin + vp.data**2 * rho.lin
        vs.lin = 2.0 * (vs.data * rho.data) * vs.lin + vs.data**2 * rho.lin
        self.sc = sc = self.scale(vp.data**2 * rho.data)
        rho.lin[rho.valid] = - 2 ** sc * self.dtdx / rho.data[rho.valid]**2 * rho.lin[rho.valid]
        vp.lin = self.dtdx * vp.lin * 2 ** -sc
        vs.lin = self.dtdx * vs.lin * 2 ** -sc

        return vp, vs, rho

    def adjoint(self, vp, vs, rho):

        sc = self.sc
        vp.grad = self.dtdx * vp.grad * 2 ** -sc
        vs.grad = self.dtdx * vs.grad * 2 ** -sc
        rho.grad[rho.valid] = - 2 ** sc * self.dtdx / \
                              rho.data[rho.valid]**2 * rho.grad[rho.valid]

        rho.grad += vp.data**2 * vp.grad + vs.data**2 * vs.grad
        vp.grad = 2.0 *(vp.data * rho.data) * vp.grad
        vs.grad = 2.0 *(vs.data * rho.data) * vs.grad

        return vp, vs, rho


class PrecisionTester(Function):

    def __init__(self):
        super().__init__()
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


@TapedFunction
def elastic2d(vp, vs, rho, vx, vz, sxx, szz, sxz, rec_pos, src_pos, src, dt, dx):

    vp, vs, rho = ScaledParameters(dt, dx)(vp, vs, rho)
    src_fun = PointForceSource()
    rec_fun = Receiver()
    updatev = UpdateVelocity2()
    updates = UpdateStress2()
    abs = Cerjan(nab=2)
    vzout = Variable("vzout", shape=(len(src), len(rec_pos)))
    for t in range(src.size):
        src_fun(vz, src[t], src_pos=src_pos)
        updatev(rho, vx, vz, sxx, szz, sxz)
        #abs(vx, vz)
        updates(vp, vs, vx, vz, sxx, szz, sxz)
        #abs(sxx, szz, sxz)
        rec_fun(vz, vzout, rec_pos=rec_pos, t=t)

    return vp, vs, rho, vx, vz, sxx, szz, sxz


if __name__ == '__main__':

    grid1D = Grid(shape=(10,))
    defs = {"vx": grid1D,
            "vz": grid1D,}
    x = Variable("x", shape=(10, 1), initialize_method="random")
    b = Variable("b", shape=(10, 1), initialize_method="random")
    y = Variable("y", shape=(10, 1), initialize_method="random")
    matmul = RandKernel()
    # matmul.backward_test(x, b, y)
    # matmul.linear_test(x, b, y)
    # matmul.dot_test(x, b, y)

    nrec = 1
    nt = 3
    nab = 2
    dt = 0.00000000001
    dx = 1
    shape = (10, 10)
    vp = Variable(shape=shape, initialize_method="random", pad=2)
    vs = Variable(shape=vp.shape, initialize_method="random", pad=2)
    rho = Variable(shape=vp.shape, initialize_method="random", pad=2)
    vx = Variable(shape=vp.shape, initialize_method="random", pad=2)
    vz = Variable(shape=vp.shape, initialize_method="random", pad=2)
    sxx = Variable(shape=vp.shape, initialize_method="random", pad=2)
    szz = Variable(shape=vp.shape, initialize_method="random", pad=2)
    sxz = Variable(shape=vp.shape, initialize_method="random", pad=2)
    vxout = Variable(shape=vp.shape, initialize_method="random", pad=2)
    ScaledParameters(dt, dx).backward_test(vp, vs, rho)
    ScaledParameters(dt, dx).linear_test(vp, vs, rho)
    ScaledParameters(dt, dx).dot_test(vp, vs, rho)
    UpdateVelocity2().backward_test(rho, vx, vz, sxx, szz, sxz)
    UpdateVelocity2().linear_test(rho, vx, vz, sxx, szz, sxz)
    UpdateVelocity2().dot_test(rho, vx, vz, sxx, szz, sxz)
    UpdateStress2().backward_test(vp, vs, vx, vz, sxx, szz, sxz)
    UpdateStress2().linear_test(vp, vs, vx, vz, sxx, szz, sxz)
    UpdateStress2().dot_test(vp, vs, vx, vz, sxx, szz, sxz)
    Cerjan(nab=2).backward_test(vx)
    Cerjan(nab=2).linear_test(vx)
    Cerjan(nab=2).dot_test(vx)
    PointForceSource().backward_test(vx, 1, (0, 1))
    PointForceSource().linear_test(vx, 1, (0, 1))
    PointForceSource().dot_test(vx, 1, (0, 1))
    Receiver().backward_test(vx, vxout, ((5, 5), (6,6)), t=0)
    Receiver().linear_test(vx, vxout, ((5, 5), (6,6)),  t=0)
    Receiver().dot_test(vx, vxout, ((5, 5), (6,6)), t=0)

    rec_pos = ((5, 5), (6, 6))
    src_pos = (0, 1)
    src = np.ones((15,))
    elastic2d.backward_test(vp, vs, rho, vx, vz, sxx, szz, sxz,
                            rec_pos, src_pos, src, dt, dx)
    elastic2d.linear_test(vp, vs, rho, vx, vz, sxx, szz, sxz, rec_pos, src_pos, src, dt, dx)
    elastic2d.dot_test(vp, vs, rho, vx, vz, sxx, szz, sxz, rec_pos, src_pos, src, dt, dx)
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

