from SeisCL.python.PSV2D.elastic_numpy import (ReversibleFunction)
from SeisCL.python.pycl_backend import ComputeRessource, GridCL
import numpy as np
from pyopencl.array import max


class ScaledParameters(ReversibleFunction):

    def __init__(self, grids=None, dt=1, dh=1, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vp", "vs", "rho"]
        self.updated_states = ["vp", "vs", "rho"]
        self.sc = 1.0
        self.dt = dt
        self.dh = dh
        self.default_grids = {el: "gridpar" for el in self.required_states}

    @staticmethod
    def scale(M, dt, dx):
        return int(np.log2(max(M).get() * dt / dx))

    def forward(self, states, **kwargs):

        vp = states["vp"]
        vs = states["vs"]
        rho = states["rho"]
        dt = self.dt
        dh = self.dh
        self.sc = sc = self.scale(vp, dt, dh)
        states["rho"] = 2 ** sc * dt / dh * rho
        states["vp"] = dt / dh * vp * 2 ** -sc
        states["vs"] = dt / dh * vs * 2 ** -sc

        return states

    def linear(self, dstates, states, **kwargs):

        dvp = dstates["vp"]
        dvs = dstates["vs"]
        drho = dstates["rho"]
        dt = self.dt
        dh = self.dh
        dstates["rho"] = 2 ** self.sc * dt / dh * drho
        dstates["vp"] = dt / dh * dvp * 2 ** -self.sc
        dstates["vs"] = dt / dh * dvs * 2 ** -self.sc

        return dstates

    def adjoint(self, adj_states, states, **kwargs):

        adj_vp = adj_states["vp"]
        adj_vs = adj_states["vs"]
        adj_rho = adj_states["rho"]
        dt = self.dt
        dh = self.dh
        adj_states["rho"] = 2 ** self.sc * dt / dh * adj_rho
        adj_states["vp"] = dt / dh * adj_vp * 2 ** -self.sc
        adj_states["vs"] = dt / dh * adj_vs * 2 ** -self.sc

        return adj_states

    def backward(self, states, **kwargs):

        vs = states["vp"]
        vp = states["vs"]
        rho = states["rho"]
        dt = self.dt
        dh = self.dh
        states["rho"] = 2 ** -self.sc * dh / dt * rho
        states["vp"] = dh / dt * vp * 2 ** self.sc
        states["vs"] = dh / dt * vs * 2 ** self.sc

        return states


if __name__ == '__main__':

    resc = ComputeRessource()
    nx = 24
    nz = 24

    grid = GridCL(resc.queues[0], shape=(nz, nx), pad=4, dtype=np.float32,
                  zero_boundary=False)
    veltrans =ScaledParameters(grids={"gridpar": grid})
    veltrans.linear_test()
    veltrans.dot_test()
