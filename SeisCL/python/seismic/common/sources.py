
from SeisCL.python import ReversibleFunctionGPU, ReversibleFunction, ComputeGrid, VariableCL, Variable
from .acquisition import Acquisition, Shot
import numpy as np
try:
    import pyopencl as cl
    from pyopencl.array import Array, to_device, empty, zeros
except ImportError:
    pass


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

    def __call__(self, shot: Shot, t: int, vx, vz, sxx, szz, vy=None, syy=None):
        for ii, pos in enumerate(shot.src_pos):
            pos = np.round(pos/self.acquisition.grid.dh).astype(np.int)
            #lin_src_pos = vx.xyz2lin(*pos)
            try:
                self.src_fun(locals()[shot.sources[ii].type],
                             shot.wavelet.data[t, ii], pos)
            except KeyError:
                if shot.sources[ii].type == 'p':
                    self.src_fun(sxx, shot.wavelet.data[t, ii], pos)
                    self.src_fun(szz, shot.wavelet.data[t, ii], pos)
                else:
                    raise ValueError('Source type %s not implemented'
                                     % shot.sources[ii].type)


class PointSources3DGPU(ReversibleFunctionGPU):

    def src_type(self, shot):
        types = {"vz": 0, "vy": 1, "vx": 2, "p": 3}
        try:
           src_type = np.array([types[s.type] for s in shot.sources],
                               dtype=np.int)
        except KeyError as e:
            raise ValueError('Source type %s not implemented' % e)
        return to_device(self.queue, src_type)

    def src_pos(self, shot, dh, shape):
        src_pos = np.array([[el for el in [s.x, s.y, s.z] if el is not None]
                            for s in shot.sources])
        src_pos = np.round(src_pos/dh).astype(np.int)
        postuple = [src_pos[:, i] for i in range(src_pos.shape[1])]
        src_pos = np.ravel_multi_index(postuple, shape, order="F")
        return to_device(self.queue, src_pos)

    def forward(self, vx, vy, vz, sxx, syy, szz, wavelet, src_pos, src_type, t,
                backpropagate=False):
        nd = len(vz.shape)
        nt = wavelet.shape[0]

        src = """
        int sign = -2*backpropagate+1;
        if (src_type[g.z] == 0)
            vz[src_pos[g.z]] += sign * wavelet[t + g.z*%d];
        else if (src_type[g.z] == 1)
            vy[src_pos[g.z]] += sign * wavelet[t + g.z*%d];
        else if (src_type[g.z] == 2)
            vx[src_pos[g.z]] += sign * wavelet[t + g.z*%d];
        else if (src_type[g.z] == 3)
            sxx[src_pos[g.z]] += sign * wavelet[t + g.z*%d] / %f;
            syy[src_pos[g.z]] += sign * wavelet[t + g.z*%d] / %f;
            szz[src_pos[g.z]] += sign * wavelet[t + g.z*%d] / %f;
        """ % tuple([nt]*3 + [nt, nd]*3)

        grid = ComputeGrid(shape=src_pos.shape,
                           queue=self.queue,
                           origin=[0])
        self.callgpu(src, "forward", grid, vx, vy, vz, sxx, syy, szz, wavelet,
                     src_pos, src_type, t, backpropagate=backpropagate)
        return vx, vy, vz, sxx, syy, szz

    def linear(self, sxx, szz, src, src_pos):
        pass

    def adjoint(self, sxx, szz, src, src_pos):
        pass


class PointSources2DGPU(PointSources3DGPU):

    def forward(self, vx, vz, sxx, szz, wavelet, src_pos, src_type, t,
                backpropagate=False):

        nt = wavelet.shape[0]
        src = """
            int sign = -2*backpropagate+1;
            if (src_type[g.z] == 0)
                vz[src_pos[g.z]] += sign * wavelet[t + g.z*%d];
            else if (src_type[g.z] == 2)
                vx[src_pos[g.z]] += sign * wavelet[t + g.z*%d];
            else if (src_type[g.z] == 3)
                sxx[src_pos[g.z]] += sign * wavelet[t + g.z*%d] / 2;
                szz[src_pos[g.z]] += sign * wavelet[t + g.z*%d] / 2;
            """ % ((nt,)*4)

        grid = ComputeGrid(shape=src_pos.shape,
                           queue=self.queue,
                           origin=[0])
        self.callgpu(src, "forward", grid, vx, vz, sxx, szz, wavelet,
                     src_pos, src_type, t,
                     backpropagate=backpropagate)
        return vx, vz, sxx, szz