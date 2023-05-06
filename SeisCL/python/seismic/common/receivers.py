from SeisCL.python import (ReversibleFunctionGPU, ReversibleFunction,
                           ComputeRessource, ComputeGrid)
from .acquisition import Acquisition, Shot
from typing import List
import numpy as np
try:
    import pyopencl as cl
    from pyopencl.array import Array, to_device, empty, zeros
except ImportError:
    pass


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

#TODO cannot perform dot product
class GeophoneGPU3D(ReversibleFunctionGPU):

    def rec_type(self, shot):
        types = {"vz": 0, "vy": 1, "vx": 2, "p": 3}
        try:
            rec_type = np.array([types[r.type] for r in shot.receivers],
                                dtype=np.int)
        except KeyError as e:
            raise ValueError('Receiver type %s not implemented' % e)
        return to_device(self.queue, rec_type)

    def rec_pos(self, shot, dh, shape):
        rec_pos = np.array([[el for el in [r.x, r.y, r.z] if el is not None]
                            for r in shot.receivers])
        rec_pos = np.round(rec_pos/dh).astype(np.int)
        postuple = [rec_pos[:, i] for i in range(rec_pos.shape[1])]
        rec_pos = np.ravel_multi_index(postuple, shape, order="F")
        return to_device(self.queue, rec_pos)

    def forward(self, vx, vy, vz, sxx, syy, szz, dout, rec_pos, rec_type, t,
                backpropagate=False):
        src = """
        if (rec_type[g.z] == 0)
            dout(t, g.z) = vz[rec_pos[g.z]];
        else if (rec_type[g.z] == 1)
            dout(t, g.z) = vy[rec_pos[g.z]];
        else if (rec_type[g.z] == 2)
            dout(t, g.z) = vx[rec_pos[g.z]];
        else if (rec_type[g.z] == 3)
            dout(t, g.z) = (sxx[rec_pos[g.z]] 
                              + syy[rec_pos[g.z]] 
                              + szz[rec_pos[g.z]])/3.0;
        """

        grid = ComputeGrid(shape=rec_pos.shape,
                           queue=self.queue,
                           origin=[0])
        self.gpukernel(src, "forward", grid, vx, vy, vz, sxx, syy, szz, dout,
                       rec_pos, rec_type, t, backpropagate=backpropagate)
        return dout

    def linear(self, vx, vy, vz, sxx, syy, szz, dout, rec_pos, rec_type, t):
        src = """
        if (rec_type[g.z] == 0)
            dout_lin(t, g.z) = vz_lin[rec_pos[g.z]];
        else if (rec_type[g.z] == 1)
            dout_lin(t, g.z) = vy_lin[rec_pos[g.z]];
        else if (rec_type[g.z] == 2)
            dout_lin(t, g.z) = vx_lin[rec_pos[g.z]];
        else if (rec_type[g.z] == 3)
            dout_lin(t, g.z) = (sxx_lin[rec_pos[g.z]] 
                                  + syy_lin[rec_pos[g.z]] 
                                  + szz_lin[rec_pos[g.z]])/3.0;
        """

        grid = ComputeGrid(shape=rec_pos.shape,
                           queue=self.queue,
                           origin=[0])
        self.gpukernel(src, "linear", grid, vx, vy, vz, sxx, syy, szz, dout,
                       rec_pos, rec_type, t)
        return dout

    def adjoint(self, vx, vy, vz, sxx, syy, szz, dout, rec_pos, rec_type, t):
        src = """
        if (rec_type[g.z] == 0)
            vz_adj[rec_pos[g.z]] += dout_adj(t, g.z);
        else if (rec_type[g.z] == 1)
            vy_adj[rec_pos[g.z]] += dout_adj(t, g.z);
        else if (rec_type[g.z] == 2)
            vx_adj[rec_pos[g.z]] += dout_adj(t, g.z);
        else if (rec_type[g.z] == 3)
            sxx_adj[rec_pos[g.z]] += dout_adj(t, g.z)/3.0;
            syy_adj[rec_pos[g.z]] += dout_adj(t, g.z)/3.0;
            szz_adj[rec_pos[g.z]] += dout_adj(t, g.z)/3.0;
        """

        grid = ComputeGrid(shape=rec_pos.shape,
                           queue=self.queue,
                           origin=[0])
        self.gpukernel(src, "adjoint", grid, vx, vy, vz, sxx, syy, szz, dout,
                       rec_pos, rec_type, t)
        return vx, vy, vz, sxx, syy, szz


class GeophoneGPU2D(GeophoneGPU3D):

    def forward(self, vx, vz, sxx, szz, dout, rec_pos, rec_type, t,
                backpropagate=False):
        src = """
        if (rec_type[g.z] == 0)
            dout(t, g.z) = vz[rec_pos[g.z]];
        else if (rec_type[g.z] == 2)
            dout(t, g.z) = vx[rec_pos[g.z]];
        else if (rec_type[g.z] == 3)
            dout(t, g.z) = (sxx[rec_pos[g.z]] 
                          + szz[rec_pos[g.z]])/2.0;
        """

        grid = ComputeGrid(shape=rec_pos.shape,
                           queue=self.queue,
                           origin=[0])
        self.gpukernel(src, "forward", grid, vx, vz, sxx, szz, dout,
                       rec_pos, rec_type, t, backpropagate=backpropagate)
        return dout

    def linear(self, vx, vz, sxx, szz, dout, rec_pos, rec_type, t):
        src = """
        if (rec_type[g.z] == 0)
            dout_lin(t, g.z) = vz_lin[rec_pos[g.z]];
        else if (rec_type[g.z] == 2)
            dout_lin(t, g.z) = vx_lin[rec_pos[g.z]];
        else if (rec_type[g.z] == 3)
            dout_lin(t, g.z) = (sxx_lin[rec_pos[g.z]] 
                              + szz_lin[rec_pos[g.z]])/2.0;
        """

        grid = ComputeGrid(shape=rec_pos.shape,
                           queue=self.queue,
                           origin=[0])
        self.gpukernel(src, "linear", grid, vx, vz, sxx, szz, dout,
                       rec_pos, rec_type, t)
        return dout

    def adjoint(self, vx, vz, sxx, szz, dout, rec_pos, rec_type, t):
        src = """
        if (rec_type[g.z] == 0)
            vz_adj[rec_pos[g.z]] += dout_adj(t, g.z);
        else if (rec_type[g.z] == 2)
            vx_adj[rec_pos[g.z]] += dout_adj(t, g.z);
        else if (rec_type[g.z] == 3)
            sxx_adj[rec_pos[g.z]] += dout_adj(t, g.z)/2.0;
            szz_adj[rec_pos[g.z]] += dout_adj(t, g.z)/2.0;
        """

        grid = ComputeGrid(shape=rec_pos.shape,
                           queue=self.queue,
                           origin=[0])
        self.gpukernel(src, "adjoint", grid, vx, vz, sxx, szz, dout,
                       rec_pos, rec_type, t)
        return vx, vz, sxx, szz

