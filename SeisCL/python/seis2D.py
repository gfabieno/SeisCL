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


def Dmx(var):
    return 1.1382 * (var[2:-2, 2:-2] - var[2:-2, 1:-3]) - 0.046414 * (var[2:-2, 3:-1] - var[2:-2, 0:-4])


def Dpz(var):
    return 1.1382 * (var[3:-1, 2:-2] - var[2:-2, 2:-2]) - 0.046414 * (var[4:, 2:-2] - var[1:-3, 2:-2])


def Dmz(var):
    return 1.1382 * (var[2:-2, 2:-2] - var[1:-3, 2:-2]) - 0.046414 * (var[3:-1, 2:-2] - var[0:-4, 2:-2])


class Grid:

    def __init__(self, shape=(10, 10), pad=2, **kwargs):
        self.shape = shape
        self.pad = pad
        self.valid = tuple([slice(self.pad, -self.pad)] * len(shape))


class Boundary(Grid):

    def forward_boundaries(self, states):
        return states
    def adjoint_boundaries(self, adj_states):
        return adj_states


class ZeroBoundary(Boundary):

    def forward_boundaries(self, states):
        for el in states:
            states[el][:self.pad, :] = 0
            states[el][-self.pad:, :] = 0
            states[el][:, :self.pad] = 0
            states[el][:, -self.pad:] = 0

        return states

    def adjoint_boundaries(self, adj_states):
        for el in adj_states:
            adj_states[el][:self.pad, :] = 0
            adj_states[el][-self.pad:, :] = 0
            adj_states[el][:, :self.pad] = 0
            adj_states[el][:, -self.pad:] = 0

        return adj_states


class Kernel(Boundary):
    """
    Kernel implementing forward and adjoint modes for a linear operator w.r.t
    state variables and outputs, and non-linear w.r.t parameters.

    The forward method must respect the following form.

    forward(states, params) = dforward/dstates (params) * states

    """
    states = {}
    params = {}
    updated_states = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._forward_states = []
        self._linear_states = []

    def call_forward(self, states, params, **kwargs):
        self._forward_states.append({el: copy.deepcopy(states[el])
                                     for el in self.updated_states})
        return self.forward(states, params, **kwargs)

    def call_adjoint(self, adj_states, adj_outputs, states, params, **kwargs):
        states = self.backforward(states, params, **kwargs)
        adj_states, adj_params = self.adjoint(adj_states, adj_outputs, states,
                                              params, **kwargs )
        return adj_states, adj_params, states

    def forward(self, states, params, **kwargs):
        """
        Applies the forward kernel.

        :param **kwargs:
        :param states: A dict containing the variables describing the state of a
                       dynamical system. The state variables are updated by
                       the forward, but they keep the same dimensions.
        :param params: A dict containing the parameters on which the forward
                       depends. The parameters are not changed by the forward.

        :return:
            states:     A dict containing the updated states.
            outputs:    A dict containing the output of the forward, usually
                        some measurements of the states.
        """
        raise NotImplementedError

    def adjoint(self, adj_states, adj_outputs, states, params, **kwargs):
        """
        Applies the adjoint of the forward

        :param **kwargs:
        :param adj_states: A dict containing the adjoint of the forward variables.
                           Each elements has the same dimension as the forward
                           state, as the forward kernel do not change the
                           dimension of the state.
        :param adj_outputs: A dict containing the adjoint of the output variables

        :param states: The states of the system, before calling forward.
        :param params: A dict containing the parameters on which the forward
                       depends. The parameters are not changed by the forward.

        :return:
            adj_states A dict containing the updated adjoint states.
            adj_params A dict containing the updated gradients of the parameters
        """
        raise NotImplementedError

    def backforward(self, states, params, **kwargs):
        """
        Reconstruct the input states from the output of forward

        :param **kwargs:
        :param states: A dict containing the variables describing the state of a
                      dynamical system. The state variables are updated by
                      the forward, but they keep the same dimensions.
        :param params: A dict containing the parameters on which the forward
                      depends. The parameters are not changed by the forward.

        :return:
           states:     A dict containing the input states.
        """
        torestore = self._forward_states.pop()
        for el in torestore:
            states[el] = torestore[el]

        return states

    def dot_test(self):
        """
        Dot product test for fstates, outputs = F(states, params)

        dF = [dfstates/dstates dfstates/dparams    [dstates
              doutputs/dstates doutpouts/dparams]   dparams ]

        Because the forward is linear wrt to states, and linear is the
        linearized forward wrt to parameters.

        dF = [forward linear    [dstates
              forward linear]    dparams ]

        dot = [adj_states  ^T [dfstates/dstates dfstates/dparams    [states
               adj_outputs]    doutputs/dstates doutpouts/dparams]   params]

        dot = [adj_states  ^T [forward  linear    [states
               adj_outputs]    forward  linear]    params]
        """
        states = {el: np.random.rand(self.shape) for el in self.states}
        params = {el: np.random.rand(*self.params[el]) for el in self.params}
        states = self.forward_boundaries(states)

        fstates, outputs = self.call_forward(copy.deepcopy(states),
                                             copy.deepcopy(params))

        adj_states = {el: np.random.rand(*fstates[el].shape) for el in fstates}
        adj_outputs = {el: np.random.rand(*outputs[el].shape) for el in outputs}
        adj_states = self.adjoint_boundaries(adj_states)

        fadj_states, fadj_params, _ = self.call_adjoint(
            copy.deepcopy(adj_states), copy.deepcopy(adj_outputs), states,
            params)


        prod1 = np.sum([fstates[el]*adj_states[el] for el in states])
        prod1 += np.sum([outputs[el]*adj_outputs[el] for el in outputs])
        prod2 = np.sum([states[el]*fadj_states[el] for el in states])
        # prod2 += np.sum([params[el]*fadj_params[el] for el in params])

        print("Dot product test for Kernel %s: %.15e"
              % (self.__class__.__name__, prod1-prod2))

        return prod1 - prod2

    def grad_test(self):
        raise NotImplementedError


class RandKernel(Kernel):

    states = {"x": (10,)}
    params = {"b": (5,)}
    updated_states = ["x"]

    def __init__(self):
        super().__init__()
        self.A1 = np.random.rand(10, 10)
        self.A2 = np.random.rand(5, 10)

    def forward(self, states, params, **kwargs):
        x = states["x"]
        b = params["b"]
        outputs = {"y": np.matmul(self.A2, x) * b}
        states["x"] = np.matmul(self.A1, x)
        return states, outputs

    def adjoint(self, adj_states, adj_outputs, states, params, **kwargs):

        x_adj = adj_states["x"]
        y_adj = adj_outputs["y"]
        x = states["x"]
        b = params["b"]
        A1t = np.transpose(self.A1)
        A2t = np.transpose(self.A2)

        return ({"x": np.matmul(A1t, x_adj) + np.matmul(A2t, b * y_adj)},
                {"b": np.ones(b.shape[0]) * y_adj})


class Sequence(Kernel):

    def __init__(self, kernels):

        self.kernels = kernels
        self.states = {}
        self.params = {}
        self.updated_states = []
        for kernel in kernels:
            self.states.update(kernel.states)
            self.params.update(kernel.params)
        self.updated_states = [el for k in kernels for el in k.updated_states]
        self.updated_states = list(set(self.updated_states))

    def call_forward(self, states, params, **kwargs):
        outputs = {}
        for ii, kernel in enumerate(self.kernels):
            states, outputsk = kernel.call_forward(states, params)
            for o in outputsk:
                outputs[str(ii) + ":" + o] = outputsk[o]

        return states, outputs

    def call_adjoint(self, adj_states, adj_outputs, states, params, **kwargs):

        adj_params = {el: np.zeros_like(params[el]) for el in params}
        n = len(self.kernels)
        for ii, kernel in enumerate(self.kernels[::-1]):
            adj_outputks = {}
            for o in adj_outputs:
                kname, oname = o.split(":")
                if int(kname) == n - ii - 1:
                    adj_outputks[oname] = adj_outputs[o]
            adj_states, adj_paramks, states = kernel.call_adjoint(adj_states,
                                                                  adj_outputks,
                                                                  states,
                                                                  params)
            for el in adj_paramks:
                adj_params[el] += adj_paramks[el]

        return adj_states, adj_params, states

    def dot_test(self):
        print("Dot product test for all kernels contained in "
              + self.__class__.__name__ + ":")
        print("    ", end = '')
        dot = super().dot_test()
        for kernel in self.kernels:
            print("    ", end = '')
            kernel.dot_test()
        return dot


class Propagator(Kernel):
    """
    Applies a series of kernels in forward and adjoint modes.
    """

    def __init__(self, kernel, nt):

        super().__init__()
        self.kernel = kernel
        self.nt = nt
        self.states = kernel.states
        self.params = kernel.params
        self.updated_states = kernel.updated_states

    def call_forward(self, states, params, **kwargs):
        outputs = [None for _ in range(self.nt)]

        for t in range(self.nt):
            states, output = self.kernel.call_forward(states, params, t=t)
            outputs[t] = output

        outputs = {el: np.stack([outputs[t][el] for t in range(self.nt)])
                   for el in outputs[0]}
        return states, outputs

    def call_adjoint(self, adj_states, adj_outputs, states, params, **kwargs):
        adj_params = {el: np.zeros_like(params[el]) for el in params}

        for t in range(self.nt-1, -1, -1):
            adj_output = {el: adj_outputs[el][t, :] for el in adj_outputs}
            (adj_states,
             adj_paramst,
             states) = self.kernel.call_adjoint(adj_states, adj_output, states,
                                                params, t=t)
            adj_params = {el: adj_params[el] + adj_paramst[el]
                          for el in adj_params}
        return adj_states, adj_params, states

    def dot_test(self):
        self.kernel.dot_test()
        return super().dot_test()


class Derivative(ZeroBoundary, Kernel):

    states = {"vx": (10, 10)}
    params = {}
    updated_states = ["vx"]

    def forward(self, states, params, **kwargs):
        states["vx"][self.valid] = Dpx(states["vx"])
        return states, {}

    def adjoint(self, adj_states, adj_outputs, states, params, **kwargs):
        adj_states["vx"][self.valid] = -Dmx(adj_states["vx"])
        return adj_states, {}


class UpdateVelocity(ZeroBoundary, Kernel):

    states = {"vx": (10, 10),
              "vz": (10, 10),
              "sxx": (10, 10),
              "szz": (10, 10),
              "sxz": (10, 10)}
    params = {"cv": (10, 10)}
    updated_states = ["vx", "vz"]

    def forward(self, states, params, **kwargs):
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

        cv = params["cv"]

        sxx_x = Dpx(sxx)
        szz_z = Dpz(szz)
        sxz_x = Dmx(sxz)
        sxz_z = Dmz(sxz)

        vx[self.valid] += (sxx_x + sxz_z) * cv[self.valid]
        vz[self.valid] += (szz_z + sxz_x) * cv[self.valid]

        return states, {}

    def adjoint(self, adj_states, adj_outputs, states, params, **kwargs):
        """
        Adjoint update the velocity for 2D P-SV isotropic elastic wave
        propagation.

        The transpose of the forward:
            [vx'     [      1        0        0     0    0     [vx'
             vz'            0        1        0     0    0      vz'
             sxx'  =   -Dmx * cv     0        1     0    0  =   sxx'
             szz'           0    -Dmz * cv     0     1    0     szz'
             sxz']     -Dpz * cv -Dpx * cv     0     0    1]    sxz']
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

        cv = params["cv"]

        adj_vx_x = Dmx(cv * adj_vx)
        adj_vx_z = Dpz(cv * adj_vx)
        adj_vz_z = Dmz(cv * adj_vz)
        adj_vz_x = Dpx(cv * adj_vz)

        sxx_x = Dpx(sxx)
        szz_z = Dpz(szz)
        sxz_x = Dmx(sxz)
        sxz_z = Dmz(sxz)

        adj_sxx[self.valid] -= adj_vx_x
        adj_szz[self.valid] -= adj_vz_z
        adj_sxz[self.valid] -= adj_vx_z + adj_vz_x

        adj_params = {}
        adj_params["cv"] = np.zeros_like(cv)
        adj_params["cv"][self.valid] += (sxx_x + sxz_z) * adj_vx[self.valid]
        adj_params["cv"][self.valid] += (szz_z + sxz_x) * adj_vz[self.valid]

        return adj_states, adj_params


class UpdateStress(ZeroBoundary, Kernel):

    states = {"vx": (10, 10),
              "vz": (10, 10),
              "sxx": (10, 10),
              "szz": (10, 10),
              "sxz": (10, 10)}
    params = {"csu": (10, 10),
              "csM": (10, 10)}
    updated_states = ["sxx", "szz", "sxz"]

    def forward(self, states, params, **kwargs):
        """
        Update the velocity for 2D P-SV isotropic elastic wave propagation.

        In matrix form:
            [vx     [         1                0        0 0 0   [vx
             vz               0                1        0 0 0    vz
             sxx  =        csM Dmx     (csM - 2csu) Dmz 1 0 0 =  sxx
             szz      (csM - 2csu) Dmx      csM Dmz     0 1 0    szz
             sxz]          csu Dpz          csu Dpx     0 0 1]   sxz]
             :param **kwargs:
        """
        sxx = states["sxx"]
        szz = states["szz"]
        sxz = states["sxz"]
        vx = states["vx"]
        vz = states["vz"]

        csu = params["csu"]
        csM = params["csM"]

        vx_x = Dmx(vx)
        vx_z = Dpz(vx)
        vz_x = Dpx(vz)
        vz_z = Dmz(vz)

        sxx[self.valid] += csM[self.valid] * (vx_x + vz_z) - 2.0 * csu[self.valid] * vz_z
        szz[self.valid] += csM[self.valid] * (vx_x + vz_z) - 2.0 * csu[self.valid] * vx_x
        sxz[self.valid] += csu[self.valid] * (vx_z + vz_x)

        return states, {}

    def adjoint(self, adj_states, adj_outputs, states, params, **kwargs):
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

        csu = params["csu"]
        csM = params["csM"]

        adj_sxx_x = Dpx(csM * adj_sxx)
        adj_sxx_z = Dpz((csM - 2.0 * csu) * adj_sxx)
        adj_szz_x = Dpx((csM - 2.0 * csu) * adj_szz)
        adj_szz_z = Dpz(csM * adj_szz)
        adj_sxz_x = Dmx(csu * adj_sxz)
        adj_sxz_z = Dmz(csu * adj_sxz)

        vx_x = Dmx(vx)
        vx_z = Dpz(vx)
        vz_x = Dpx(vz)
        vz_z = Dmz(vz)

        adj_vx[self.valid] -= adj_sxx_x + adj_szz_x + adj_sxz_z
        adj_vz[self.valid] -= adj_sxx_z + adj_szz_z + adj_sxz_x

        adj_params = {}
        adj_params["csM"] = np.zeros_like(csM)
        adj_params["csM"][self.valid] += (vx_x + vz_z) * adj_sxx[self.valid]
        adj_params["csM"][self.valid] += (vx_x + vz_z) * adj_szz[self.valid]
        adj_params["csu"] = np.zeros_like(csM)
        adj_params["csu"][self.valid] += - 2.0 * vz_z * adj_sxx[self.valid]
        adj_params["csu"][self.valid] += - 2.0 * vx_x * adj_szz[self.valid]
        adj_params["csu"][self.valid] += (vx_z + vz_x) * adj_sxz[self.valid]

        return adj_states, adj_params


class Receiver(Kernel):

    states = {"vx": (10, 10),
              "vz": (10, 10),
              "sxx": (10, 10),
              "szz": (10, 10),
              "sxz": (10, 10)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rec_pos = kwargs["rec_pos"]

    def forward(self, states, params, **kwargs):
        outputs = {}
        for r in self.rec_pos:
            outputs[r["type"]] = states[r["type"]][r["z"], r["x"]]
        return states, outputs

    def adjoint(self, adj_states, adj_outputs, states, params, **kwargs):

        for r in self.rec_pos:
            adj_states[r["type"]][r["z"], r["x"]] += adj_outputs[r["type"]]

        return adj_states, {}


class Source(Kernel):

    states = {"vx": (10, 10),
              "vz": (10, 10),
              "sxx": (10, 10),
              "szz": (10, 10),
              "sxz": (10, 10)}
    params = {}

    def forward(self, states, params, t=0, src_pos=(), **kwargs):

        for ii, s in enumerate(src_pos):
            states[s["type"]][s["pos"]] += s["signal"][t]

        return states, {}

    def backforward(self, states, params, t=0, src_pos=(), **kwargs):
        for ii, s in enumerate(src_pos):
            states[s["type"]][s["pos"]] -= s["signal"][t]

    def adjoint(self, adj_states, adj_outputs, states, params, **kwargs):

        return adj_states, {}


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

        self.vx[self.valid] += (sxx_x + sxz_z) * self.cv
        self.vz[self.valid] += (szz_z + sxz_x) * self.cv

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


    matmul = RandKernel()
    matmul2 = RandKernel()
    seq = Sequence([matmul, matmul2])
    prop = Propagator(seq, 5)
    prop.dot_test()
    der = Derivative()
    der.dot_test()
    updatev = UpdateVelocity()
    updatev.dot_test()
    updates = UpdateStress()
    updates.dot_test()
    receiver = Receiver(rec_pos=[{"type": "vx", "z": 5, "x": 5}])
    receiver.dot_test()
    source = Source(rec_pos=[{"type": "vx", "z": 5, "x": 5}])
    source.dot_test()


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
