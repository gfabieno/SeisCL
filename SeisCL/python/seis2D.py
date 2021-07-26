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

    def __init__(self, shape=(10, 10), pad=2, dtype=np.float32,
                 zero_boundary=False,**kwargs):
        self.shape = shape
        self.pad = pad
        self.valid = tuple([slice(self.pad, -self.pad)] * len(shape))
        self.dtype = dtype
        self.eps = np.finfo(dtype).eps
        self.smallest = np.nextafter(dtype(0), dtype(1))
        self.zero_boundary = zero_boundary

    def zero(self):
        return np.zeros(self.shape, dtype=self.dtype)

    def random(self):
        if self.zero_boundary:
            state = np.zeros(self.shape, dtype=self.dtype)
            state[self.valid] = np.random.rand(*state[self.valid].shape)*10e6
        else:
            state = np.random.rand(*self.shape).astype(self.dtype)
        return state

    def assign_data(self, data):
        return data.astype(self.dtype)

    @staticmethod
    def np(array):
        return array


class State:

    def __init__(self, name, grid=Grid(), data=None, **kwargs):
        self.name = name
        self.grid = grid

    def initialize(self, data=None, method="zero"):

        if data is not None:
            data = self.grid.assign_data(data)
        elif method == "zero":
            data = self.grid.zero()
        elif method == "random":
            data = self.grid.random()

        return data


class StateKernel:
    """
    Kernel implementing forward, linear and adjoint modes.
    """

    def __init__(self, state_defs=None, **kwargs):
        self._state_defs = state_defs
        self.state_defs = state_defs
        self._forward_states = []
        if not hasattr(self, 'updated_states'):
            self.updated_states = []
        if not hasattr(self, 'required_states'):
            self.required_states = []

    def __call__(self, states=None, initialize=True, **kwargs):

        if not states:
            states = {}
        if initialize:
            self.initialize(states)

        self._forward_states.append({el: states[el].copy()
                                     for el in self.updated_states})
        return self.forward(states, **kwargs)

    @property
    def state_defs(self):
        return self._state_defs

    @state_defs.setter
    def state_defs(self, val):
        self._state_defs = val

    def initialize(self, states, empty_cache=True, method="zero", **kwargs):
        for el in self.required_states:
            if el not in states:
                states[el] = self.state_defs[el].initialize(method=method)
            elif type(states[el]) is not self.state_defs[el].grid.backend:
                states[el] = self.state_defs[el].initialize(data=states[el])
        if empty_cache:
            self._forward_states = []
        return states

    def call_linear(self, dstates, states, **kwargs):

        dstates = self.linear(dstates, states, **kwargs)
        states = self.forward(states, **kwargs)

        return dstates, states

    def gradient(self, adj_states, states, initialize=True, **kwargs):

        if initialize:
            adj_states = self.initialize(adj_states, empty_cache=False)
        states = self.backward(states, **kwargs)
        adj_states = self.adjoint(adj_states, states, **kwargs)
        return adj_states, states

    def forward(self, states, **kwargs):
        """
        Applies the forward kernel.

        :param **kwargs:
        :param states: A dict containing the variables describing the state of a
                       dynamical system. The state variables are updated by
                       the forward, but they keep the same dimensions.
        :return:
            states:     A dict containing the updated states.
        """
        return states

    def linear(self, dstates, states, **kwargs):
        dstates = self.forward(dstates, **kwargs)
        return dstates

    def adjoint(self, adj_states, states, **kwargs):
        """
        Applies the adjoint of the forward

        :param **kwargs:
        :param adj_states: A dict containing the adjoint of the forward variables.
                           Each elements has the same dimension as the forward
                           state, as the forward kernel do not change the
                           dimension of the state.
        :param states: The states of the system, before calling forward.

        :return:
            adj_states A dict containing the updated adjoint states.
        """
        return adj_states

    def backward(self, states, **kwargs):
        """
        Reconstruct the input states from the output of forward

        :param **kwargs:
        :param states: A dict containing the variables describing the state of a
                      dynamical system. The state variables are updated by
                      the forward, but they keep the same dimensions.
        :return:
           states:     A dict containing the input states.
        """
        torestore = self._forward_states.pop()
        for el in torestore:
            states[el] = torestore[el]

        return states

    def backward_test(self, **kwargs):


        states = self.initialize({}, method="random")
        fstates = self({el: states[el].copy() for el in states},
                       initialize=False, **kwargs)
        bstates = self.backward(fstates, **kwargs)

        err = 0
        scale = 0
        for el in states:
            smallest = self.state_defs[el].grid.smallest
            snp = self.state_defs[el].grid.np(states[el])
            bsnp = self.state_defs[el].grid.np(bstates[el])
            errii = snp - bsnp
            err += np.sum(errii**2)
            scale += np.sum((snp - np.mean(snp))**2) + smallest
        err = err / scale
        print("Backpropagation test for Kernel %s: %.15e"
              % (self.__class__.__name__, err))

        return err

    def linear_test(self, **kwargs):

        states = self.initialize({}, method="random")
        dstates = self.initialize({}, method="random")

        errs = []
        cond = True if states else False
        while cond:
            dstates = {el: self.state_defs[el].grid.np(dstates[el])/10.0
                       for el in dstates}
            dstates = self.initialize(dstates)
            for el in states:
                dnp = self.state_defs[el].grid.np(dstates[el])
                snp = self.state_defs[el].grid.np(states[el])
                smallest = self.state_defs[el].grid.smallest
                eps = self.state_defs[el].grid.eps
                if np.max(dnp / (snp+smallest)) < eps:
                    cond = False
                    break
            if not cond:
                break
            pstates = {el: states[el] + dstates[el] for el in states}

            fpstates = self({el: pstates[el].copy() for el in pstates},
                            initialize=False, **kwargs)
            fstates = self({el: states[el].copy() for el in states},
                           initialize=False, **kwargs)

            lstates, _ = self.call_linear({el: dstates[el].copy()
                                           for el in dstates},
                                          {el: states[el].copy()
                                           for el in states},
                                          **kwargs)

            err = 0
            scale = 0
            for el in states:
                smallest = self.state_defs[el].grid.smallest
                eps = self.state_defs[el].grid.eps
                ls = self.state_defs[el].grid.np(lstates[el])
                fdls = self.state_defs[el].grid.np(fpstates[el] - fstates[el])
                errii = fdls - ls
                err += np.sum(errii**2)
                scale += np.sum((ls - np.mean(ls))**2)
            errs.append([err/(scale+smallest)])

        errmin = np.min(errs)
        print("Linear test for Kernel %s: %.15e"
              % (self.__class__.__name__, errmin))

        return errmin

    def dot_test(self, **kwargs):
        """
        Dot product test for fstates, outputs = F(states)

        dF = [dfstates/dstates     [dstates
              doutputs/dstates]     dparams ]

        dot = [adj_states  ^T [dfstates/dstates     [states
               adj_outputs]    doutputs/dstates]   params]

        """

        states = self.initialize({}, method="random")
        fstates = self({el: states[el].copy() for el in states},
                       initialize=False, **kwargs)

        dstates = self.initialize({}, empty_cache=False, method="random")
        dfstates, _ = self.call_linear({el: dstates[el].copy()
                                        for el in dstates},
                                       {el: states[el].copy()
                                        for el in states},
                                       **kwargs)

        adj_states = self.initialize({}, empty_cache=False, method="random")

        fadj_states, _ = self.gradient({el: adj_states[el].copy()
                                        for el in adj_states},
                                       {el: fstates[el].copy()
                                        for el in fstates},
                                       initialize=False,
                                       **kwargs)

        prod1 = np.sum([np.sum(self.state_defs[el].grid.np(dfstates[el] * adj_states[el]))
                        for el in dfstates])
        prod2 = np.sum([np.sum(self.state_defs[el].grid.np(dstates[el] * fadj_states[el]))
                        for el in dstates])

        print("Dot product test for Kernel %s: %.15e"
              % (self.__class__.__name__, (prod1-prod2)/(prod1+prod2)))

        return (prod1-prod2)/(prod1+prod2)


class RandKernel(StateKernel):

    def __init__(self, **kwargs):
        state_defs = {"x": State("x", Grid((10,))),
                      "b": State("b", Grid((10,)))}
        super().__init__(state_defs, **kwargs)
        self.required_states = ["x", "b"]
        self.updated_states = ["x"]
        self.A1 = np.random.rand(10, 10)
        self.A2 = np.random.rand(10, 10)

    def forward(self, states, **kwargs):
        x = states["x"]
        b = states["b"]
        states["x"] = np.matmul(self.A1, x)
        states["y"] = np.matmul(self.A2, x) * b
        return states

    def linear(self, dstates, states, **kwargs):
        x = states["x"]
        b = states["b"]
        dx = dstates["x"]
        db = dstates["b"]
        dstates["x"] = np.matmul(self.A1, dx)
        dstates["y"] = np.matmul(self.A2, x) * db + np.matmul(self.A2, dx) * b

        return dstates

    def adjoint(self, adj_states, states, **kwargs):

        x_adj = adj_states["x"]
        b_adj = adj_states["b"]
        y_adj = adj_states["y"]
        x = states["x"]
        b = states["b"]
        A1t = np.transpose(self.A1)
        A2t = np.transpose(self.A2)

        adj_states["x"] = np.matmul(A1t, x_adj) + np.matmul(A2t, b * y_adj)
        adj_states["b"] = np.matmul(self.A2, x) * y_adj + b_adj
        return adj_states


class Sequence(StateKernel):

    def __init__(self, kernels, state_defs=None, **kwargs):

        self.kernels = kernels
        super().__init__(state_defs, **kwargs)
        self.required_states = []
        self.updated_states = []
        for kernel in kernels:
            self.required_states += [el for el in kernel.required_states
                                     if el not in self.required_states]
            self.updated_states += [el for el in kernel.updated_states
                                    if el not in self.updated_states]
            # if state_defs is not None:
            #     kernel.state_defs = state_defs
    @property
    def state_defs(self):
        return self._state_defs

    @state_defs.setter
    def state_defs(self, val):
        self._state_defs = val
        for kernel in self.kernels:
            kernel.state_defs = val

    def initialize(self, states, empty_cache=False, method="zero", **kwargs):
        states = super(Sequence, self).initialize(states,
                                                  empty_cache=empty_cache,
                                                  method=method,
                                                  **kwargs)
        for kernel in self.kernels:
            states = kernel.initialize(states,
                                       empty_cache=empty_cache,
                                       method=method,
                                       **kwargs)
        return states

    def __call__(self, states, initialize=True, **kwargs):

        if initialize:
            states = self.initialize(states)
        for ii, kernel in enumerate(self.kernels):
            states = kernel(states, initialize=False, **kwargs)
        return states

    def call_linear(self, dstates, states, **kwargs):

        for ii, kernel in enumerate(self.kernels):
            dstates, states = kernel.call_linear(dstates, states, **kwargs)
        return dstates, states

    def gradient(self, adj_states, states, initialize=True, **kwargs):

        if initialize:
            adj_states = self.initialize(adj_states, empty_cache=False)
        for ii, kernel in enumerate(self.kernels[::-1]):
            adj_states, states = kernel.gradient(adj_states, states,
                                                 initialize=False,
                                                 **kwargs)
        return adj_states, states

    def backward(self, states, **kwargs):
        for ii, kernel in enumerate(self.kernels[::-1]):
            states = kernel.backward(states, **kwargs)
        return states

    def backward_test(self, **kwargs):
        print("Back propagation test for all kernels contained in "
              + self.__class__.__name__ + ":")
        print("    ", end='')
        err = super().backward_test(**kwargs)
        for kernel in self.kernels:
            print("    ", end='')
            kernel.backward_test(**kwargs)

        return err

    def linear_test(self, **kwargs):
        print("Linear test for all kernels contained in "
              + self.__class__.__name__ + ":")
        print("    ", end='')
        err = super().linear_test(**kwargs)
        for kernel in self.kernels:
            print("    ", end='')
            kernel.linear_test(**kwargs)

        return err

    def dot_test(self, **kwargs):
        print("Dot product test for all kernels contained in "
              + self.__class__.__name__ + ":")
        print("    ", end='')
        dot = super().dot_test(**kwargs)
        for kernel in self.kernels:
            print("    ", end='')
            kernel.dot_test(**kwargs)
        return dot


class Propagator(StateKernel):
    """
    Applies a series of kernels in forward and adjoint modes.
    """

    def __init__(self, kernel, nt, **kwargs):

        self.kernel = kernel
        super().__init__(kernel.state_defs, **kwargs)
        self.nt = nt
        self.required_states = kernel.required_states
        self.updated_states = kernel.updated_states

    @property
    def state_defs(self):
        return self._state_defs

    @state_defs.setter
    def state_defs(self, val):
        self._state_defs = val
        self.kernel.state_defs = val

    def initialize(self, states, empty_cache=False, method="zero", **kwargs):
        states = super(Propagator, self).initialize(states,
                                                    empty_cache=empty_cache,
                                                    method=method,
                                                    **kwargs)
        states = self.kernel.initialize(states,
                                        empty_cache=empty_cache,
                                        method=method,
                                        **kwargs)
        return states

    def __call__(self, states, initialize=True, **kwargs):

        if initialize:
            states = self.initialize(states)
        for t in range(self.nt):
            states = self.kernel(states, t=t, initialize=False, **kwargs)

        return states

    def call_linear(self, dstates, states, **kwargs):

        for t in range(self.nt):
            dstates, states = self.kernel.call_linear(dstates, states,
                                                      t=t, **kwargs)

        return dstates, states

    def gradient(self, adj_states, states, initialize=True, **kwargs):

        if initialize:
            adj_states = self.initialize(adj_states, empty_cache=False)
        for t in range(self.nt-1, -1, -1):
            adj_states, states = self.kernel.gradient(adj_states, states, t=t,
                                                      initialize=False,
                                                      **kwargs)

        return adj_states, states

    def backward(self, states, **kwargs):
        for t in range(self.nt-1, -1, -1):
            states = self.kernel.backward(states, t=t, **kwargs)
        return states

    def backward_test(self, **kwargs):
        print("*************************************")
        err = super().backward_test(**kwargs)
        self.kernel.backward_test(**kwargs)
        print("*************************************")
        return err

    def linear_test(self, **kwargs):
        print("*************************************")
        err = super().linear_test(**kwargs)
        self.kernel.linear_test(**kwargs)
        print("*************************************")
        return err

    def dot_test(self, **kwargs):
        print("*************************************")
        err = super().dot_test(**kwargs)
        self.kernel.dot_test(**kwargs)
        print("*************************************")
        return err


class Derivative(StateKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["vx"]
        self.updated_states = ["vx"]

    def forward(self, states, **kwargs):
        valid = self.state_defs["vx"].grid.valid
        states["vx"][valid] = Dmz(states["vx"])
        return states

    def adjoint(self, adj_states, states, **kwargs):
        valid = self.state_defs["vx"].grid.valid
        adj_states["vx"] = Dmz_adj(adj_states["vx"])
        return adj_states


class Division(StateKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["vx", "vz"]
        self.updated_states = ["vx"]
        self.smallest = self.state_defs["vx"].grid.smallest

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


class Multiplication(StateKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["vx", "vz"]
        self.updated_states = ["vx"]
        self.smallest = self.state_defs["vx"].grid.smallest

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


class ReversibleKernel(StateKernel):

    def __call__(self, states, initialize=True, **kwargs):
        if initialize:
            states = self.initialize(states)
        return self.forward(states, **kwargs)

    def backward(self, states, **kwargs):
        states = self.forward(states, backpropagate=True, **kwargs)
        return states


class UpdateVelocity(ReversibleKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "cv"]
        self.updated_states = ["vx", "vz"]

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

        valid = self.state_defs["vx"].grid.valid
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

        valid = self.state_defs["vx"].grid.valid
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
        valid = self.state_defs["vx"].grid.valid

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
        valid = self.state_defs["vx"].grid.valid

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


class UpdateStress(ReversibleKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "csu", "csM"]
        self.updated_states = ["sxx", "szz", "sxz"]

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

        valid = self.state_defs["vx"].grid.valid
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

        valid = self.state_defs["vx"].grid.valid
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

        valid = self.state_defs["vx"].grid.valid
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

        valid = self.state_defs["vx"].grid.valid
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

class ZeroBoundary(StateKernel):
    def __init__(self, required_states, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = required_states

    def forward(self, states, **kwargs):

        for el in self.required_states:
            mask = np.ones(self.state_defs[el].grid.shape, np.bool)
            mask[self.state_defs[el].grid.valid] = 0
            states[el][mask] = 0

        return states

    def adjoint(self, adj_states, states, rec_pos=(), t=0, **kwargs):

        for el in self.required_states:
            mask = np.ones(self.state_defs[el].grid.shape, np.bool)
            mask[self.state_defs[el].grid.valid] = 0
            adj_states[el][mask] = 0

        return adj_states


class Cerjan(StateKernel):

    def __init__(self, state_defs=None, freesurf=False, abpc=4.0, nab=2, pad=2,
                 required_states=(), **kwargs):
        super().__init__(state_defs, **kwargs)
        self.abpc = abpc
        self.nab = nab
        self.pad = pad
        self.required_states = required_states
        self.updated_states = required_states
        self.taper = np.exp(np.log(1.0-abpc/100)/nab**2 * np.arange(nab) **2)
        self.taper = np.concatenate([self.taper,  self.taper[-pad:][::-1]])
        self.taper = np.expand_dims(self.taper, -1)
        self.freesurf = freesurf

    def __call__(self, states, initialize=True, **kwargs):

        if initialize:
            states = self.initialize(states)
        saved = {}
        for el in self.updated_states:
            saved[el] = []
            b = self.nab + self.pad
            if not self.freesurf:
                saved[el].append(states[el][:b, :].copy())
            saved[el].append(states[el][-b:, :].copy())
            saved[el].append(states[el][:, :b].copy())
            saved[el].append(states[el][:, -b:].copy())

        self._forward_states.append(saved)

        return self.forward(states, **kwargs)

    def forward(self, states, **kwargs):

        for el in self.required_states:
            if not self.freesurf:
                states[el][:self.nab+2, :] *= self.taper[::-1]
            states[el][-self.nab-2:, :] *= self.taper

            tapert = np.transpose(self.taper)
            states[el][:, :self.nab+2] *= tapert[:, ::-1]
            states[el][:, -self.nab-2:] *= tapert

        return states

    def adjoint(self, adj_states, states, **kwargs):

        return self.forward(adj_states, **kwargs)

    def backward(self, states, **kwargs):

        torestore = self._forward_states.pop()
        for el in torestore:
            b = self.nab + self.pad
            if not self.freesurf:
                states[el][:b, :] = torestore[el][0]
            states[el][-b:, :] = torestore[el][-3]
            states[el][:, :b] = torestore[el][-2]
            states[el][:, -b:] = torestore[el][-1]

        return states


class Receiver(ReversibleKernel):

    def __init__(self, required_states, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = required_states

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


class Source(ReversibleKernel):

    def __init__(self, required_states, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
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


class FreeSurface(StateKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "csu", "csM"]
        self.updated_states = ["sxx", "szz", "sxz"]

    def __call__(self, states, initialize=True, **kwargs):

        if initialize:
            states = self.initialize(states)
        saved = {}
        for el in self.updated_states:
            pad = self.state_defs[el].grid.pad
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

        pad = self.state_defs["vx"].grid.pad
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
        pad = self.state_defs["vx"].grid.pad

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

        pad = self.state_defs["vx"].grid.pad
        valid = self.state_defs["vx"].grid.valid

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
            pad = self.state_defs[el].grid.pad
            states[el][:pad+1, :] = torestore[el]

        return states


class FreeSurface2(FreeSurface):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "csu", "csM", "cv"]
        self.updated_states = ["sxx", "szz", "vx", "vz"]

    def __call__(self, states, initialize=True, **kwargs):

        if initialize:
            states = self.initialize(states)
        saved = {}
        for el in self.updated_states:
            pad = self.state_defs[el].grid.pad
            saved[el] = states[el][pad:2*pad, pad:-pad].copy()
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
        cv = states["cv"]

        pad = self.state_defs["vx"].grid.pad
        shape = self.state_defs["vx"].grid.shape

        vxx = Dmx(vx)[:1, :]
        vzz = Dmz(vz)[:1, :]
        f = csu[pad:pad+1, pad:-pad] * 2.0
        g = csM[pad:pad+1, pad:-pad]
        h = -((g - f) * (g - f) * vxx / g) - ((g - f) * vzz)
        sxx[pad:pad+1, pad:-pad] += h

        szz[pad:pad+1, :] = 0.0
        szz_z = np.zeros((3*pad, shape[1]))
        szz_z[:pad, :] = -szz[pad+1:2*pad+1, :][::-1, :]
        szz_z = Dpz(szz_z)
        sxz_z = np.zeros((3*pad, shape[1]))
        sxz_z[:pad, :] = -sxz[pad+1:2*pad+1, :][::-1, :]
        sxz_z = Dmz(sxz_z)
        vx[pad:2*pad, pad:-pad] += sxz_z * cv[pad:2*pad, pad:-pad]
        vz[pad:2*pad, pad:-pad] += szz_z * cv[pad:2*pad, pad:-pad]

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
        szz = states["szz"]
        sxz = states["sxz"]
        dcsu = dstates["csu"]
        dcsM = dstates["csM"]
        dcv = dstates["cv"]
        csu = states["csu"]
        csM = states["csM"]
        pad = self.state_defs["vx"].grid.pad
        shape = self.state_defs["vx"].grid.shape

        vxx = Dmx(vx)[:1, :]
        vzz = Dmz(vz)[:1, :]
        df = dcsu[pad:pad+1, pad:-pad] * 2.0
        dg = dcsM[pad:pad+1, pad:-pad]
        f = csu[pad:pad+1, pad:-pad] * 2.0
        g = csM[pad:pad+1, pad:-pad]
        dh = (2.0 * (g - f) * vxx / g + vzz) * df
        dh += (-2.0 * (g - f) * vxx / g + (g - f)**2 / g**2 * vxx - vzz) * dg
        dstates["sxx"][pad:pad+1, pad:-pad] += dh

        szz_z = np.zeros((3*pad, shape[1]))
        szz_z[:pad, :] = -szz[pad+1:2*pad+1, :][::-1, :]
        szz_z = Dpz(szz_z)
        sxz_z = np.zeros((3*pad, shape[1]))
        sxz_z[:pad, :] = -sxz[pad+1:2*pad+1, :][::-1, :]
        sxz_z = Dmz(sxz_z)
        dstates["vx"][pad:2*pad, pad:-pad] += sxz_z * dcv[pad:2*pad, pad:-pad]
        dstates["vz"][pad:2*pad, pad:-pad] += szz_z * dcv[pad:2*pad, pad:-pad]

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

        pad = self.state_defs["vx"].grid.pad
        shape = self.state_defs["vx"].grid.shape

        szz_z = np.zeros((3*pad, shape[1]))
        szz_z[:pad, :] = -szz[pad+1:2*pad+1, :][::-1, :]
        szz_z = Dpz(szz_z)
        sxz_z = np.zeros((3*pad, shape[1]))
        sxz_z[:pad, :] = -sxz[pad+1:2*pad+1, :][::-1, :]
        sxz_z = Dmz(sxz_z)
        adj_states["cv"][pad:2*pad, pad:-pad] += sxz_z * adj_vx[pad:2*pad, pad:-pad]
        adj_states["cv"][pad:2*pad, pad:-pad] += szz_z * adj_vz[pad:2*pad, pad:-pad]

        adj_vx_z = np.zeros((3*pad, shape[1]))
        #adj_vx_z[pad:2*pad, pad:-pad] = adj_vx[pad:2*pad, pad:-pad] * cv[pad:2*pad, pad:-pad]
        # adj_vx_z = Dmz_adj(adj_vx_z)
        hc = [0.046414, -1.1382]
        for ii in range(pad):
            for jj in range(ii+1):
                adj_vx_z[ii, :] += hc[jj]*adj_vx[pad+ii-jj, :] * cv[pad+ii-jj, :]
        adj_sxz[pad+1:2*pad+1, pad:-pad] += -adj_vx_z[:pad, :][::-1, pad:-pad]

        adj_vz_z = np.zeros((3*pad, shape[1]))
        # adj_vz_z[pad:2*pad, pad:-pad] = adj_vz[pad:2*pad, pad:-pad] * cv[pad:2*pad, pad:-pad]
        # adj_vz_z = Dpz_adj(adj_vz_z)
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

    def backward(self, states, **kwargs):

        torestore = self._forward_states.pop()
        for el in torestore:
            pad = self.state_defs[el].grid.pad
            states[el][pad:2*pad, pad:-pad] = torestore[el]

        return states


class ScaledParameters(ReversibleKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["cv", "csu", "csM"]
        self.updated_states = ["cv", "csu", "csM"]
        self.sc = 1.0

    def forward(self, states, dt=0.1, dx=2.0, **kwargs):

        vs = states["csu"]
        vp = states["csM"]
        rho = states["cv"]

        M = (vp**2 * rho)
        mu = (vs**2 * rho)
        self.sc = sc = int(np.log2(np.max(M) * dt / dx))
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

        dM = 2.0 *(vp * rho) * dvp + vp**2 * drho
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


class PrecisionTester(StateKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
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
    defs = {"vx": State("vx", grid=grid2D),
            "vz": State("vz", grid=grid2D),
            "sxx": State("sxx", grid=grid2D),
            "szz": State("szz", grid=grid2D),
            "sxz": State("sxz", grid=grid2D),
            "cv": State("cv", grid=gridpar),
            "csu": State("csu", grid=gridpar),
            "csM": State("csM", grid=gridpar),
            "vxout": State("vxout", grid=gridout),
            "vzout": State("vzout", grid=gridout)
            }

    stepper = Sequence([Source(required_states=["vx"]),
                        UpdateVelocity2(),
                        Cerjan(required_states=["vx", "vz"], freesurf=1, nab=nab),
                        Receiver(required_states=["vx", "vxout", "vz", "vzout"]),
                        UpdateStress2(),
                        FreeSurface2(),
                        Cerjan(required_states=["sxx", "szz", "sxz"], freesurf=1, nab=nab),
                        ])
    prop = Propagator(stepper, nt)
    psv2D = Sequence([ScaledParameters(),
                      #ZeroBoundary(required_states=["vx", "vz", "sxx", "szz", "sxz"]),
                      prop],
                     state_defs=defs)
    return psv2D

if __name__ == '__main__':

    grid1D = Grid(shape=(10,))
    defs = {"vx": State("vx", grid=grid1D),
            "vz": State("vz", grid=grid1D),}
    #matmul = RandKernel(grid=grid1D)
    # matmul.linear_test()
    # matmul.dot_test()
    # matmul2 = RandKernel(grid=grid1D)
    # seq = Sequence([matmul, matmul2], state_defs=matmul.state_defs)
    # seq.dot_test()
    # prop = Propagator(seq, 5)
    # prop.dot_test()
    # der = Derivative({"vx": State("vx", grid=Grid(shape=(10, 10), pad=2))})
    # der.dot_test()
    # div = Division(state_defs=defs)
    # div.linear_test()
    # div.backward_test()

    nrec = 1
    nt = 3
    nab = 2
    grid2D = Grid(shape=(10, 10), type=np.float64, zero_boundary=True)
    gridout = Grid(shape=(nt, nrec), type=np.float64)
    psv2D = define_psv(grid2D, gridout, nab, nt)

    psv2D.backward_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
                        src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])
    psv2D.linear_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
                      src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])
    psv2D.dot_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
                   src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])

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
    # # plt.imshow(states["vx"])
    # # plt.show()
    # #
    # vxobs = states["vxout"]
    # vzobs = states["vzout"]
    # clip = 0.01
    # vmin = np.min(states["vxout"]) * 0.1
    # vmax=-vmin
    # plt.imshow(states["vxout"], aspect="auto", vmin=vmin, vmax=vmax)
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

