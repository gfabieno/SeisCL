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
import copy
#import copy
#import os.path


def Dpx(var):
    return 1.1382 * (var[2:-2, 3:-1] - var[2:-2, 2:-2]) - 0.046414 * (var[2:-2, 4:] - var[2:-2, 1:-3])


def Dpx_adj(var):

    dvar = copy.deepcopy(var)
    dvar[2:-2, 2:-2] = 0
    dvar[2:-2, 3:-1] += 1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 2:-2] += -1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 4:] += - 0.046414 * var[2:-2, 2:-2]
    dvar[2:-2, 1:-3] += 0.046414 * var[2:-2, 2:-2]

    return dvar


def Dmx(var):
    return 1.1382 * (var[2:-2, 2:-2] - var[2:-2, 1:-3]) - 0.046414 * (var[2:-2, 3:-1] - var[2:-2, 0:-4])


def Dmx_adj(var):

    dvar = copy.deepcopy(var)
    dvar[2:-2, 2:-2] = 0
    dvar[2:-2, 2:-2] += 1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 1:-3] += -1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 3:-1] += - 0.046414 * var[2:-2, 2:-2]
    dvar[2:-2, 0:-4] += 0.046414 * var[2:-2, 2:-2]

    return dvar


def Dpz(var):
    return 1.1382 * (var[3:-1, 2:-2] - var[2:-2, 2:-2]) - 0.046414 * (var[4:, 2:-2] - var[1:-3, 2:-2])


def Dpz_adj(var):

    dvar = copy.deepcopy(var)
    dvar[2:-2, 2:-2] = 0
    dvar[3:-1, 2:-2] += 1.1382 * var[2:-2, 2:-2]
    dvar[2:-2, 2:-2] += -1.1382 * var[2:-2, 2:-2]
    dvar[4:, 2:-2] += - 0.046414 * var[2:-2, 2:-2]
    dvar[1:-3, 2:-2] += 0.046414 * var[2:-2, 2:-2]

    return dvar


def Dmz(var):
    return 1.1382 * (var[2:-2, 2:-2] - var[1:-3, 2:-2]) - 0.046414 * (var[3:-1, 2:-2] - var[0:-4, 2:-2])


def Dmz_adj(var):

    dvar = copy.deepcopy(var)
    dvar[2:-2, 2:-2] = 0
    dvar[2:-2, 2:-2] += 1.1382 * var[2:-2, 2:-2]
    dvar[1:-3, 2:-2] += -1.1382 * var[2:-2, 2:-2]
    dvar[3:-1, 2:-2] += - 0.046414 * var[2:-2, 2:-2]
    dvar[0:-4, 2:-2] += 0.046414 * var[2:-2, 2:-2]

    return dvar


class Grid:

    def __init__(self, shape=(10, 10), pad=2, dtype=np.float64, **kwargs):
        self.shape = shape
        self.pad = pad
        self.valid = tuple([slice(self.pad, -self.pad)] * len(shape))
        self.dtype = dtype
        self.eps = np.finfo(dtype).eps

    def empty(self):
        return np.zeros(*self.shape)

    def random(self):
        state = np.zeros(self.shape, dtype=self.dtype)
        state[self.valid] = np.random.rand(*state[self.valid].shape)
        state = np.random.rand(*state.shape)
        return state


class State:

    def __init__(self, name, grid=Grid(), **kwargs):
        self.name = name
        self.grid = grid


class StateKernel:
    """
    Kernel implementing forward, linear and adjoint modes.
    """

    def __init__(self, state_defs=None, **kwargs):
        self.state_defs = state_defs
        self._forward_states = []
        self.updated_states = []
        self.required_states = []

    def __call__(self, states, initialize=True, **kwargs):

        if initialize:
            self.initialize(states)

        self._forward_states.append({el: copy.deepcopy(states[el])
                                     for el in self.updated_states})
        return self.forward(states, **kwargs)

    def initialize(self, states):
        for el in self.required_states:
            if el not in states:
                states[el] = self.state_defs[el].grid.empty()
        self._forward_states = []
        return states

    def call_linear(self, dstates, states, **kwargs):

        dstates = self.linear(dstates, states, **kwargs)
        states = self.forward(states, **kwargs)

        return dstates, states

    def gradient(self, adj_states, states, **kwargs):

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
        raise adj_states

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

        states = {self.state_defs[el].name: self.state_defs[el].grid.random()
                  for el in self.required_states}
        fstates = self(copy.deepcopy(states), **kwargs)
        bstates = self.backward(fstates, **kwargs)

        err = np.sum([np.sum(states[el] - bstates[el]) for el in bstates])

        print("Backpropagation test for Kernel %s: %.15e"
              % (self.__class__.__name__, err))

        return err

    def linear_test(self, **kwargs):

        states = {self.state_defs[el].name: self.state_defs[el].grid.random()
                  for el in self.required_states}
        dstates = {self.state_defs[el].name: self.state_defs[el].grid.random()
                   for el in self.required_states}

        errs = []
        for ii in range(0, 20):
            dstates = {el: dstates[el]/10 for el in dstates}
            pstates = {el: states[el] + dstates[el] for el in states}

            fpstates = self(copy.deepcopy(pstates), **kwargs)
            fstates = self(copy.deepcopy(states), **kwargs)

            lstates, _ = self.call_linear(copy.deepcopy(dstates),
                                          copy.deepcopy(states),
                                          **kwargs)

            err = 0
            for el in states:
                err = np.max([err, np.max(np.abs(fpstates[el] - fstates[el] - lstates[el]))])
            errs.append([err])

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

        states = {el: self.state_defs[el].grid.random()
                  for el in self.required_states}
        fstates = self(copy.deepcopy(states), **kwargs)

        dstates = {el: self.state_defs[el].grid.random()
                   for el in self.required_states}
        dfstates, _ = self.call_linear(copy.deepcopy(dstates),
                                       copy.deepcopy(states),
                                       **kwargs)

        adj_states = {el: np.random.random(fstates[el].shape)
                      for el in fstates}

        fadj_states, _ = self.gradient(copy.deepcopy(adj_states),
                                       copy.deepcopy(fstates), **kwargs)

        prod1 = np.sum([np.sum(dfstates[el] * adj_states[el]) for el in dfstates])
        prod2 = np.sum([np.sum(dstates[el] * fadj_states[el]) for el in fadj_states])

        print("Dot product test for Kernel %s: %.15e"
              % (self.__class__.__name__, prod1-prod2))

        return prod1 - prod2


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

        return {"x": np.matmul(A1t, x_adj) + np.matmul(A2t, b * y_adj),
                "b": np.matmul(self.A2, x) * y_adj + b_adj}


class Sequence(StateKernel):

    def __init__(self, kernels, state_defs=None, **kwargs):

        super().__init__(state_defs, **kwargs)
        self.kernels = kernels
        self.required_states = []
        self.updated_states = []
        for kernel in kernels:
            self.required_states += [el for el in kernel.required_states
                                     if el not in self.required_states]
            self.updated_states += [el for el in kernel.updated_states
                                    if el not in self.updated_states]
            if state_defs is not None:
                kernel.state_defs = state_defs

    def initialize(self, states):
        states = super(Sequence, self).initialize(states)
        for kernel in self.kernels:
            states = kernel.initialize(states)
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

    def gradient(self, adj_states, states, **kwargs):

        n = len(self.kernels)
        for ii, kernel in enumerate(self.kernels[::-1]):
            adj_states, states = kernel.gradient(adj_states, states, **kwargs)
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

        super().__init__(kernel.state_defs, **kwargs)
        self.kernel = kernel
        self.nt = nt
        self.required_states = kernel.required_states
        self.updated_states = kernel.updated_states

    def initialize(self, states):
        states = super(Propagator, self).initialize(states)
        states = self.kernel.initialize(states)
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

    def gradient(self, adj_states, states, **kwargs):

        for t in range(self.nt-1, -1, -1):
            adj_states, states = self.kernel.gradient(adj_states, states, t=t,
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
        self.eps = self.state_defs["vx"].grid.eps

    def forward(self, states, **kwargs):

        states["vx"] = states["vx"] / (states["vz"]+self.eps)
        return states

    def linear(self, dstates, states, **kwargs):
        """
        [dvx   =   [1/vz -vx/vz**2] [dvx
         dvz]       0        1    ]  dvz]

        [vx'   =   [1/vz       0] [vx'
         vz']       -vx/vz**2  1]  vz']
        """

        dstates["vx"] = dstates["vx"] / (states["vz"] + self.eps)
        dstates["vx"] += -states["vx"] / (states["vz"] + self.eps)**2 * dstates["vz"]

        return dstates

    def adjoint(self, adj_states, states, **kwargs):
        adj_states["vz"] += -states["vx"] / (states["vz"] + self.eps)**2 * adj_states["vx"]
        adj_states["vx"] = adj_states["vx"] / (states["vz"] + self.eps)
        return adj_states


class Multiplication(StateKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["vx", "vz"]
        self.updated_states = ["vx"]
        self.eps = self.state_defs["vx"].grid.eps

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


class Cerjan(StateKernel):

    def __init__(self, state_defs=None, freesurf=False, abpc=4.0, nab=2,
                 required_states=(), **kwargs):
        super().__init__(state_defs, **kwargs)
        self.abpc = abpc
        self.nab = nab
        self.required_states = required_states
        self.updated_states = required_states
        self.taper = np.exp(-np.log(1.0-abpc/100)/nab**2 * np.arange(nab) **2)
        self.taper = np.expand_dims(self.taper, -1)
        self.freesurf = freesurf

    def __call__(self, states, initialize=True, **kwargs):

        if initialize:
            states = self.initialize(states)
        saved = {}
        for el in self.updated_states:
            saved[el] = []
            valid = self.state_defs[el].grid.valid
            if not self.freesurf:
                saved[el].append(copy.deepcopy(states[el][valid][:self.nab, :]))
            saved[el].append(copy.deepcopy(states[el][valid][-self.nab:, :]))
            saved[el].append(copy.deepcopy(states[el][valid][:, :self.nab]))
            saved[el].append(copy.deepcopy(states[el][valid][:, -self.nab:]))

        self._forward_states.append(saved)

        return self.forward(states, **kwargs)

    def forward(self, states, **kwargs):

        for el in self.required_states:
            valid = self.state_defs[el].grid.valid
            if not self.freesurf:
                states[el][valid][:self.nab, :] *= self.taper[::-1]
            states[el][valid][-self.nab:, :] *= self.taper

            tapert = np.transpose(self.taper)
            states[el][valid][:, :self.nab] *= tapert[::-1]
            states[el][valid][:, -self.nab:] *= tapert

        return states

    def adjoint(self, adj_states, states, **kwargs):

        return self.forward(adj_states, **kwargs)

    def backward(self, states, **kwargs):

        torestore = self._forward_states.pop()
        for el in torestore:
            valid = self.state_defs[el].grid.valid
            if not self.freesurf:
                states[el][valid][:self.nab, :] = torestore[el][0]
            states[el][valid][-self.nab:, :] = torestore[el][-3]
            states[el][valid][:, :self.nab] = torestore[el][-2]
            states[el][valid][:, -self.nab:] = torestore[el][-1]

        return states


class Receiver(ReversibleKernel):

    def __init__(self, required_states, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = required_states

    def forward(self, states, rec_pos=(), t=0, **kwargs):

        for ii, r in enumerate(rec_pos):
            states[r["type"]+"out"][t, ii] += states[r["type"]][r["z"], r["x"]]
        return states

    def adjoint(self, adj_states, states, rec_pos=(), t=0, **kwargs):

        for ii, r in enumerate(rec_pos):
            adj_states[r["type"]][r["z"], r["x"]] += adj_states[r["type"]+"out"][t, ii]
        return adj_states

    def backward(self, states, rec_pos=(), t=0, **kwargs):
        for ii, r in enumerate(rec_pos):
            states[r["type"]+"out"][t, ii] -= states[r["type"]][r["z"], r["x"]]
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
            saved[el] = copy.deepcopy(states[el][:pad+1, :])
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
                      "csM": states["csM"]})

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




        return states


class Initializer(StateKernel):

    def __init__(self, state_defs=None, **kwargs):
        super().__init__(state_defs, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "cv", "csu", "csM"]
        self.updated_states = []
        self.to_diff = []

    def forward(self, *args, **kwargs):
        states = {}
        for el in self.state_defs:
            if el in kwargs:
                states[el] = kwargs[el]
                self.to_diff.append(el)
            else:
                states[el] = np.zeros(self.state_defs[el].grid.shape)
                self.updated_states.append(el)

        return states

    def adjoint(self, adj_states, states, **kwargs):
        adj_states = {el: adj_states[el] for el in self.to_diff}
        return adj_states




class Seis2D:
    """
    2D seismic propagation in velocity-stress formulation
         
    
    :param vp: P-wave velocity model
    :param rho: Density model
    :param rho: Attenuation levels
    :param taul: relaxation times
    :param dt: Time step
    :param dx: spatial step
    :param nab: CPML layer size
    :param src_pos: Position of sources
    :param rec_pos: Position of receivers
    :param src: Source signal
    """

    def __init__(self, vp, vs, rho, dt, dx, src_pos, rec_pos, src,
                 nab=None, dtype=np.float32, dtypepar=np.float32):

        self.M = (np.square(vp) * rho)
        self.mu = (np.square(vs) * rho)
        self.rho = rho
        self.sc = int(np.log2(np.max(self.M) * dt / dx))
        self.cv = (2 ** self.sc * dt / dx / self.rho).astype(dtypepar)
        self.csM = (dt / dx * self.M * 2 ** -self.sc).astype(dtypepar)
        self.csu = (dt / dx * self.mu * 2 ** -self.sc).astype(dtypepar)

        #        self.tau=tau
        #        self.taul=taul
        #        self.L=taul.size
        self.dt = dt
        self.dx = dx
        #        self.eta=self.dt*(2.0*math.pi*taul)
        self.src_pos = src_pos
        self.rec_pos = rec_pos
        self.src = src.astype(np.float16)
        self.nab = nab

        self.NX = self.M.shape[1]
        self.NZ = self.M.shape[0]
        self.NT = self.src.shape[0]
        self.vx = np.zeros([self.NZ + 4, self.NX + 4], dtype=dtype)
        self.vz = np.zeros([self.NZ + 4, self.NX + 4], dtype=dtype)
        self.sxx = np.zeros([self.NZ + 4, self.NX + 4], dtype=dtype)
        self.szz = np.zeros([self.NZ + 4, self.NX + 4], dtype=dtype)
        self.sxz = np.zeros([self.NZ + 4, self.NX + 4], dtype=dtype)
        #        self.r=np.array([ np.zeros([self.NZ,self.NX]) for i in range(0,self.L) ], dtype=dtype)
        #        self.qsigma=np.zeros(2*self.nab,dtype=dtype)
        #        self.qv=np.zeros(2*self.nab,dtype=dtype)
        self.t = 0
        self.vxout = np.zeros([self.NT, rec_pos.size])
        self.vzout = np.zeros([self.NT, rec_pos.size])

    def CPML(self):
        # CPML variables
        npower = 2
        VPPML = vp.mean()
        FPML = 6
        K_MAX_CPML = 2
        Rcoef = 0.0008
        alpha_max_PML = 2.0 * math.pi * (FPML / 2.0)
        a = 0.25
        b = 0.75
        c = 0.0
        d0 = - (npower + 1) * VPPML * math.log(Rcoef) / (2.0 * self.nab * self.dx)
        position = np.zeros(2 * self.nab)
        position[0:self.nab + 1] = (self.nab - np.linspace(0, self.nab, self.nab + 1)) * self.dx
        position[self.nab:] = [position[i] for i in range(self.nab, 0, -1)]
        position_norm = position / self.nab / self.dx
        d = d0 * (a * position_norm + b * pow(position_norm, npower) + c * pow(position_norm, 4))
        self.K = 1.0 + (K_MAX_CPML - 1.0) * pow(position_norm, npower)
        alpha_prime = alpha_max_PML * (1.0 - position_norm)
        self.b = np.exp(- (d / self.K + alpha_prime) * self.dt)
        self.a = d * (self.b - 1.0) / (self.K * (d + self.K * alpha_prime))
        position = np.zeros(2 * self.nab)
        position[0:self.nab] = (self.nab - 0.5 - np.linspace(0, self.nab - 1, self.nab)) * self.dx
        position[self.nab:] = [position[i] for i in range(self.nab - 1, -1, -1)]
        position_norm = position / self.nab / self.dx
        d = d0 * (a * position_norm + b * pow(position_norm, npower) + c * pow(position_norm, (4)))
        self.Kh = 1.0 + (K_MAX_CPML - 1.0) * pow(position_norm, npower)
        alpha_prime = alpha_max_PML * (1.0 - position_norm)
        self.bh = np.exp(- (d / self.Kh + alpha_prime) * self.dt)
        self.ah = d * (self.bh - 1.0) / (self.Kh * (d + self.Kh * alpha_prime))

    def update_v(self):
        sxx_x = Dpx(self.sxx)
        szz_z = Dpz(self.szz)
        sxz_x = Dmx(self.sxz)
        sxz_z = Dmz(self.sxz)

        self.vx[valid] += (sxx_x + sxz_z) * self.cv
        self.vz[valid] += (szz_z + sxz_x) * self.cv

        self.vz[2 + self.src_pos[0], 2 + self.src_pos[1]] += self.src[self.t]

    #        self.vxout[self.t+1,:]=self.vx[2+self.rec_pos[0], 2+self.rec_pos[1]]
    #        self.vzout[self.t+1,:]=self.vx[2+self.rec_pos[0], 2+self.rec_pos[1]]

    def update_s(self):
        vx_x = Dmx(self.vx)
        vx_z = Dpz(self.vx)
        vz_x = Dpx(self.vz)
        vz_z = Dmz(self.vz)

        self.sxz[self.valid] += self.csu * (vx_z + vz_x)
        self.sxx[self.valid] += self.csM * (vx_x + vz_z) - 2.0 * self.csu * vz_z
        self.szz[self.valid] += self.csM * (vx_x + vz_z) - 2.0 * self.csu * vx_x

    def init_seis(self):
        self.vx *= 0
        self.vz *= 0
        self.sxx *= 0
        self.szz *= 0
        self.sxz *= 0
        self.vxout = np.zeros([self.NT, self.rec_pos.size])
        self.vzout = np.zeros([self.NT, self.rec_pos.size])

    def propagate(self):
        self.init_seis()
        for t in range(0, self.NT - 1):
            self.t = t
            self.update_v()
            self.update_s()
            print(np.max(self.vx))
            print(np.max(self.sxx))

    def movie(self):
        fig = plt.figure(figsize=(12, 12))
        im = plt.imshow(self.vx, animated=True, vmin=np.min(self.src) / 10, vmax=np.max(self.src) / 10)
        self.t = 0
        self.init_seis()

        def init():
            im.set_array(self.vz)
            return im,

        def animate(t):
            self.t = t
            self.update_v()
            self.update_s()
            im.set_array(self.vz)
            print(t)
            print(np.max(self.sxx))
            return [im]

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=self.NT - 1, interval=20, blit=False, repeat=False)
        plt.show()


def ricker(f0, dt, NT):
    tmin = -2 / f0
    t = np.zeros((NT, 1))
    t[:, 0] = tmin + np.arange(0, NT * dt, dt)
    pf = math.pow(math.pi, 2) * math.pow(f0, 2)
    ricker = np.multiply((1.0 - 2.0 * pf * np.power(t, 2)), np.exp(-pf * np.power(t, 2)))

    return ricker


if __name__ == '__main__':

    grid1D = Grid(shape=(10,))
    matmul = RandKernel(grid=grid1D)
    matmul.linear_test()
    matmul.dot_test()
    # matmul2 = RandKernel(grid=grid1D)
    # seq = Sequence([matmul, matmul2], state_defs=matmul.state_defs)
    # seq.dot_test()
    # prop = Propagator(seq, 5)
    # prop.dot_test()
    der = Derivative({"vx": State("vx", grid=Grid(shape=(10, 10), pad=2))})
    der.dot_test()

    nrec = 1
    nt = 3
    grid2D = Grid(shape=(10, 10))
    gridout = Grid(shape=(nt, nrec))
    defs = {"vx": State("vx", grid=grid2D),
            "vz": State("vz", grid=grid2D),
            "sxx": State("sxx", grid=grid2D),
            "szz": State("szz", grid=grid2D),
            "sxz": State("sxz", grid=grid2D),
            "cv": State("cv", grid=grid2D),
            "csu": State("csu", grid=grid2D),
            "csM": State("csM", grid=grid2D),
            "vxout": State("vxout", grid=gridout)
           }

    # scale = ScaledParameters(state_defs=defs)
    # scale.linear_test()
    # scale.backward_test()
    # scale.dot_test()

    # init = Initializer(state_defs=defs)
    # init.linear_test(cv=np.random.random((10, 10)))
    # init.backward_test(cv=np.random.random((10, 10)))
    # init.dot_test(cv=np.random.random((10, 10)))

    stepper = Sequence([Source(required_states=["vx"]),
                        UpdateVelocity(),
                        Cerjan(abs_states=["vx", "vz"], freesurf=1),
                        Receiver(required_states=["vx", "vxout"]),
                        UpdateStress(),
                        FreeSurface(),
                        Cerjan(abs_states=["sxx", "szz", "sxz"], freesurf=1),
                        ],
                       state_defs=defs)
    prop = Propagator(stepper, nt)
    # prop.backward_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
    #                    src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])
    # prop.linear_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
    #                  src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])
    # prop.dot_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
    #               src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])


    psv2D = Sequence([ScaledParameters(),
                      prop],
                     state_defs=defs)
    # psv2D.backward_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
    #                     src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])
    # psv2D.linear_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
    #                   src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])
    psv2D.dot_test(rec_pos=[{"type": "vx", "z": 5, "x": 5}],
                   src_pos=[{"type": "vx", "pos": (5, 5), "signal": [10]*nt}])




    # grid2D = Grid(shape=(10, 10))
    # rho = Parameter(grid=grid2D)
    # vp = Parameter(grid=grid2D)
    # vs = Parameter(grid=grid2D)
    # rho, M, mu = ToModulus()
    # cv, csM, csu = ScaledInput(vp, vs, rho)


    # vp = np.zeros([500, 500]) + 3500
    # vs = np.zeros([500, 500]) + 2000
    # rho = np.zeros([500, 500]) + 2000
    # vp[0:220,:]=4000
    # vs[0:220,:]=1600
    # dt = 0.0003
    # dx = 3
    # src_pos = np.array([250, 250])
    # rec_pos = np.array([])
    # src = ricker(10, 0.001, 1700)
    # src = src
    
    
    
    # file = open('../marmousi/madagascar/vel.asc','r')
    # vel= [float(el) for el in file]
    # vp = np.transpose(np.reshape( np.array(vel), [2301, 751]))
    # vp=vp[::5,::5]
    # rho = vp*0+2000
    # vs = vp*0
    # dt = 0.002
    # dx = 20
    # src_pos = np.array([5, 250])
    # rec_pos = np.array([])
    # src = ricker(7.5, dt, 1400)
    # src = src

#
#     model32 = Seis2D(vp, vs, rho, dt, dx, src_pos, rec_pos, src, dtype=np.float32, dtypepar=np.float32)
#     model16 = Seis2D(vp, vs, rho, dt, dx, src_pos, rec_pos, src, dtype=np.float16, dtypepar=np.float16)
#
# #     model.propagate()
#     model16.movie()

#    model32.init_seis()
#    model16.init_seis()
#
#    for t in range(0, int(4.0/20/dt) - 1):
#        model32.t = t
#        model32.update_v()
#        model32.update_s()
#
#
#    model16.vx= (model32.vx).astype(model16.vx.dtype)
#    model16.vz = (model32.vz).astype(model16.vx.dtype)
#    model16.sxx = (model32.sxx).astype(model16.vx.dtype)
#    model16.szz = (model32.szz).astype(model16.vx.dtype)
#    model16.sxz = (model32.sxz).astype(model16.vx.dtype)
#    model16.src=model16.src*0
#    model32.src = model32.src * 0
#
#    fig = plt.figure(figsize=(12, 12))
#    im = plt.imshow(model16.vx, animated=True, vmin=np.min(src) / 5000, vmax=np.max(src) / 5000)
#    model16.t = 0
#    model32.t = 0
#
#    def init():
#        im.set_array(model32.vz - model16.vz)
#        return im,
#
#
#    def animate(t):
#        model16.t = t
#        model32.t = t
#        model16.update_v()
#        model16.update_s()
#        model32.update_v()
#        model32.update_s()
#        im.set_array(model16.vz -model32.vz)
#        print(np.max(model32.vx - model16.vx) / np.max(model32.vx))
#        return [im]
#
#
#    anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                   frames=model16.NT - 1, interval=20, blit=False, repeat=False)
#    plt.show()
