#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:13:18 2017

@author: gabrielfabien-ouellet
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from SeisCL.python import Variable, Function, TapedFunction, ReversibleFunction
from SeisCL.python.seismic.common.acquisition import Acquisition, Grid, Source, Shot, Receiver
import unittest
from SeisCL.python.Propagator import FWI, Propagator
from typing import List

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


class UpdateVelocity(ReversibleFunction):

    def __init__(self):
        super().__init__()
        self.updated_states = ["vx", "vz"]

    def forward(self, cv, vx, vz, sxx, szz, sxz, backpropagate=False):
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

    def linear(self, cv, vx, vz, sxx, szz, sxz):

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


class Geophone(ReversibleFunction):

    def forward(self, var, varout, rec_pos=(), trids=(), t=0):
        if not trids:
            trids = np.arange(len(rec_pos))
        for ii, r in enumerate(rec_pos):
            varout.data[t, trids[ii]] += var.data[r]
        return varout

    def linear(self, var, varout, rec_pos=(), trids=(), t=0):
        if not trids:
            trids = np.arange(len(rec_pos))
        for ii, r in enumerate(rec_pos):
            varout.lin[t, trids[ii]] += var.lin[r]
        return varout

    def adjoint(self, var, varout, rec_pos=(), trids=(), t=0):
        if not trids:
            trids = np.arange(len(rec_pos))
        for ii, r in enumerate(rec_pos):
            var.grad[r] += varout.grad[t, trids[ii]]
        return var

    def recover_states(self, initial_states, var, varout, rec_pos=(),
                       trids=(), t=0):
        if not trids:
            trids = np.arange(len(rec_pos))
        for ii, r in enumerate(rec_pos):
            varout.data[t, trids[ii]] -= var.data[r]
        return varout


class ElasticReceivers:

    def __init__(self, acquisition: Acquisition):
        self.acquisition = acquisition
        self.rec_fun = Geophone()

    def __call__(self, shot: List[Shot], t: int,
                 vx, vz, sxx, szz, vy=None, syy=None):

        for type, trids in shot.rectypes.items():
            receivers = [shot.receivers[trid] for trid in trids]
            pos = [tuple([int(np.round(el/self.acquisition.grid.dh))
                         for el in [r.z, r.y, r.x] if el is not None])
                   for r in receivers]
            try:
                self.rec_fun(locals()[type], shot.dmod, pos, trids, t)
            except KeyError:
                if type == 'p':
                    self.rec_fun(sxx, shot.dmod, pos, trids, t)
                    self.rec_fun(szz, shot.dmod, pos, trids, t)
                else:
                    raise ValueError('Receiver type %s not implemented' % type)


class PointForceSource(ReversibleFunction):

    def forward(self, var, src, src_pos=(), backpropagate=False):

        if backpropagate:
            sign = -1.0
        else:
            sign = 1.0
        var.data[src_pos] += sign * src

        return var

    def linear(self, var, src, src_pos=()):
        return var

    def adjoint(self, var, src, src_pos=()):
        return var


class ElasticSources:

        def __init__(self, acquisition: Acquisition):
            self.acquisition = acquisition
            self.src_fun = PointForceSource()

        def __call__(self, sources: List[Source], wavelet: Variable,
                     t: int, vx, vz, sxx, szz, vy=None, syy=None):
            for ii, source in enumerate(sources):
                pos = tuple(int(np.round(el/self.acquisition.grid.dh))
                            for el in [source.z, source.y, source.x]
                            if el is not None)
                #lin_src_pos = vx.xyz2lin(*pos)
                try:
                    self.src_fun(locals()[source.type], wavelet.data[t, ii], pos)
                except KeyError:
                    if source.type == 'p':
                        self.src_fun(sxx, wavelet.data[t, ii], pos)
                        self.src_fun(szz, wavelet.data[t, ii], pos)
                    else:
                        raise ValueError('Source type %s not implemented'
                                         % source.type)


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


def ricker(f0, dt, NT):
    tmin = -2 / f0
    t = np.zeros((NT, 1))
    t[:, 0] = tmin + np.arange(0, NT * dt, dt)
    pf = math.pow(math.pi, 2) * math.pow(f0, 2)
    ricker = np.multiply((1.0 - 2.0 * pf * np.power(t, 2)), np.exp(-pf * np.power(t, 2)))

    return ricker


class Elastic2dPropagator(Propagator):

    def __init__(self, acquisition: Acquisition, fdorder=4):

        self.acquisition = acquisition
        #TODO define a class Stencil to handle the fdorder and type of stencil
        self.fdorder = fdorder
        shape = (self.acquisition.grid.nz, self.acquisition.grid.nx)
        pad = self.fdorder//2
        self.vx = Variable(shape=shape, pad=pad)
        self.vz = Variable(shape=shape, pad=pad)
        self.sxx = Variable(shape=shape, pad=pad)
        self.szz = Variable(shape=shape, pad=pad)
        self.sxz = Variable(shape=shape, pad=pad)
        self.vs = Variable(shape=shape, pad=pad)
        self.vp = Variable(shape=shape, pad=pad)
        self.rho = Variable(shape=shape, pad=pad)
        self.scaledparameters = ScaledParameters(self.acquisition.grid.dt,
                                                 self.acquisition.grid.dh)
        self.src_fun = ElasticSources(acquisition)
        self.rec_fun = ElasticReceivers(acquisition)
        self.updatev = UpdateVelocity()
        self.updates = UpdateStress()
        self.abs = Cerjan(nab=self.acquisition.grid.nab)

    def propagate(self, shot, vp, vs, rho):

        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.vx.initialize()
        self.vz.initialize()
        self.sxx.initialize()
        self.szz.initialize()
        self.sxz.initialize()
        shot.dmod.initialize()
        vp, vs, rho = self.scaledparameters(vp, vs, rho)
        vx, vz, sxx, szz, sxz = (self.vx, self.vz, self.sxx, self.szz, self.sxz)
        for t in range(self.acquisition.grid.nt):
            self.src_fun(shot.sources, shot.wavelet, t, vx, vz, sxx, szz)
            self.updatev(rho, vx, vz, sxx, szz, sxz)
            self.abs(vx, vz)
            self.updates(vp, vs, vx, vz,
                         sxx, szz, sxz)
            self.abs(sxx, szz, sxz)
            self.rec_fun(shot, t, vx, vz, sxx, szz)

        return shot.dmod, vp, vs, rho, vx, vz, sxx, szz, sxz


class ElasticTester(unittest.TestCase):

    def setUp(self):

        nrec = 1
        nt = 3
        nab = 2
        self.dt = 0.00000000001
        self.dx = 1
        shape = (10, 10)
        vp = self.vp = Variable(shape=shape, initialize_method="random", pad=2)
        self.vs = Variable(shape=vp.shape, initialize_method="random", pad=2)
        self.rho = Variable(shape=vp.shape, initialize_method="random", pad=2)
        self.vx = Variable(shape=vp.shape, initialize_method="random", pad=2)
        self.vz = Variable(shape=vp.shape, initialize_method="random", pad=2)
        self.sxx = Variable(shape=vp.shape, initialize_method="random", pad=2)
        self.szz = Variable(shape=vp.shape, initialize_method="random", pad=2)
        self.sxz = Variable(shape=vp.shape, initialize_method="random", pad=2)
        self.vxout = Variable(shape=vp.shape, initialize_method="random", pad=2)

    def test_scaledparameters(self):
        vp = self.vp; vs = self.vs; rho = self.rho
        dt = self.dt; dx = self.dx
        sp = ScaledParameters(dt, dx)
        self.assertLess(sp.backward_test(vp, vs, rho), 1e-06)
        self.assertLess(sp.linear_test(vp, vs, rho), 1e-01)
        self.assertLess(sp.dot_test(vp, vs, rho), 1e-06)

    def test_UpdateVelocity(self):
        vp = self.vp; vs = self.vs; rho = self.rho
        vx = self.vx; vz = self.vz
        sxx = self.sxx; szz = self.szz; sxz = self.sxz
        fun = UpdateVelocity()
        self.assertLess(fun.backward_test(rho, vx, vz, sxx, szz, sxz), 1e-08)
        self.assertLess(fun.linear_test(rho, vx, vz, sxx, szz, sxz), 1e-01)
        self.assertLess(fun.dot_test(rho, vx, vz, sxx, szz, sxz), 1e-7)

    def test_UpdateStress(self):
        vp = self.vp; vs = self.vs; rho = self.rho
        vx = self.vx; vz = self.vz
        sxx = self.sxx; szz = self.szz; sxz = self.sxz
        fun = UpdateStress()
        self.assertLess(fun.backward_test(vp, vs, vx, vz, sxx, szz, sxz), 1e-08)
        self.assertLess(fun.linear_test(vp, vs, vx, vz, sxx, szz, sxz), 1e-01)
        self.assertLess(fun.dot_test(vp, vs, vx, vz, sxx, szz, sxz), 1e-7)

    def test_Cerjan(self):
        vx = self.vx; vz = self.vz
        fun = Cerjan()
        self.assertLess(fun.backward_test(vx, vz), 1e-08)
        self.assertLess(fun.linear_test(vx, vz), 1e-2)
        self.assertLess(fun.dot_test(vx, vz), 1e-08)

    def test_PoinSource(self):
        vx = self.vx
        fun = PointForceSource()
        self.assertLess(fun.backward_test(vx, 1, (0, 1)), 1e-08)
        self.assertLess(fun.linear_test(vx, 1, (0, 1)), 1e-2)
        self.assertLess(fun.dot_test(vx, 1, (0, 1)), 1e-08)

    def test_Geophone(self):
        vx = self.vx; vxout = self.vxout
        fun = Geophone()
        self.assertLess(fun.backward_test(vx, vxout, ((5, 5), (6, 6))), 1e-08)
        self.assertLess(fun.linear_test(vx, vxout, ((5, 5), (6, 6))), 1e-2)
        self.assertLess(fun.dot_test(vx, vxout, ((5, 5), (6, 6))), 1e-08)

    def test_elastic2d_propagator(self):
        grid = Grid(nd=2, nx=10, ny=None, nz=10, nt=3, dt=0.00000000001, dh=1.0,
                    nab=2, freesurf=True)
        shot = Shot([Source()], [Receiver(x=0), Receiver(x=1)], 0,
                    grid.nt, grid.dt)
        self.assertIsNot(shot.wavelet, None)
        acquisition = Acquisition(grid=grid, shots=[shot])
        propagator = Elastic2dPropagator(acquisition)
        shot = propagator.acquisition.shots[0]
        vp = propagator.vp; vs = propagator.vs; rho = propagator.rho
        @TapedFunction
        def prop(shot, vp, vs, rho):
            return propagator.propagate(shot, vp, vs, rho)
        self.assertLess(prop.backward_test(shot, vp, vs, rho), 1e-08)
        self.assertLess(prop.linear_test(shot, vp, vs, rho), 1e-2)
        self.assertLess(prop.dot_test(shot, vp, vs, rho), 1e-7)


if __name__ == '__main__':

    grid = Grid(nd=2, nx=300, ny=None, nz=160, nt=4500, dt=0.0001, dh=1.0,
                nab=16, freesurf=True)
    acquisition = Acquisition(grid=grid)
    acquisition.regular2d(rec_types=["vx", "vz"], gz0=4)
    propagator = Elastic2dPropagator(acquisition)
    vp, vs, rho = (propagator.vp, propagator.vs, propagator.rho)
    vp.data[:, :] = 1500
    vs.data[:, :] = 400
    rho.data[:, :] = 1800
    vs.data[80:, :] = 600
    vp.data[80:, :] = 2000
    rho.data[80:, :] = 2000
    vs0 = vs.data.copy()
    vs.data[5:10, 145:155] *= 1.05

    # dmod, _, _, _, _, _, _, _, _ = propagator.propagate(acquisition.shots[1],
    #                                                     vp, vs, rho)

    fwi = FWI(acquisition, propagator)
    shots = fwi(acquisition.shots[:2], vp, vs, rho)
    dmod = shots[1].dmod

    clip = 0.0001
    vmin = np.min(dmod.data) * clip
    vmax=-vmin
    plt.imshow(dmod.data, aspect="auto", vmin=vmin, vmax=vmax)
    plt.show()



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

