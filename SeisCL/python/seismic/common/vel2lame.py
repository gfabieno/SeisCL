from SeisCL.python import ReversibleFunction, ReversibleFunctionGPU
from SeisCL.python import ComputeRessource, ComputeGrid
import numpy as np
import pyopencl.clmath as math


class Velocity2Lame(ReversibleFunction):

    def forward(self, vp, vs, rho, backpropate=False):

        if backpropate:
            vp.data = math.sqrt(vp.data * rho.data)
            vs.data = math.sqrt(vs.data * rho.data)
        else:
            vp.data = vp.data**2 * rho.data
            vs.data = vs.data**2 * rho.data
            #rho.data = 1.0 / (rho.data + rho.smallest)

        return vp, vs

    def linear(self, vp, vs, rho):

        vp.lin = 2.0 * (vp.data * rho.data) * vp.lin + vp.data**2 * rho.lin
        vs.lin = 2.0 * (vs.data * rho.data) * vs.lin + vs.data**2 * rho.lin
        #dstates["rhoi"] = -1.0 / (rho + self.grids["rho"].smallest)**2 * drho

        return vp, vs

    def adjoint(self, vp, vs, rho):

        rho.grad += vp.data**2 * vp.grad + vs.data**2 * vs.grad
        #- 1.0 / (rho + self.grids["rho"].smallest)**2 * adj_rhoi
        vp.grad += 2.0 * (vp.data * rho.data) * vp.grad
        vs.grad += 2.0 * (vs.data * rho.data) * vs.grad

        return vp, vs, rho


class Velocity2LameGPU(ReversibleFunctionGPU):

    def forward(self, vp, vs, rho, backpropagate=False):

        if len(vp.shape) == 3:
            postr = """g.z, g.y, g.x"""
        elif len(vp.shape) == 2:
            postr = """g.z, g.x"""
        else:
            postr = """g.z"""

        src = """
        if (backpropagate){
            vp(%s) = sqrt(vp(%s) / rho(%s));
            vs(%s) = sqrt(vs(%s) / rho(%s));
        }
        else{
            vp(%s) = pow(vp(%s), 2) * rho(%s);
            vs(%s) = pow(vs(%s), 2) * rho(%s);
        }
        """ % ((postr,) * 12)
        grid = ComputeGrid(shape=[s - 2*vp.pad for s in vp.shape],
                           queue=self.queue,
                           origin=[vp.pad for _ in vp.shape])
        self.gpukernel(src, "forward", grid, vp, vs, rho,
                       backpropagate=backpropagate)
        return vp, vs

    def linear(self, vp, vs, rho):

        if len(vp.shape) == 3:
            postr = """g.z, g.y, g.x"""
        elif len(vp.shape) == 2:
            postr = """g.z, g.x"""
        else:
            postr = """g.z"""
        src = """
        vp_lin(%s) = 2.0 * (vp(%s) * rho(%s)) * vp_lin(%s) \
                      + pow(vp(%s), 2) * rho_lin(%s);
        vs_lin(%s) = 2.0 * (vs(%s) * rho(%s)) * vs_lin(%s) \
                      + pow(vs(%s), 2) * rho_lin(%s);
        """ % ((postr,) * 12)
        grid = ComputeGrid(shape=[s - 2*vp.pad for s in vp.shape],
                           queue=self.queue,
                           origin=[vp.pad for _ in vp.shape])
        self.gpukernel(src, "linear", grid, vp, vs, rho)
        return vp, vs

    def adjoint(self, vp, vs, rho):

        if len(vp.shape) == 3:
            postr = """g.z, g.y, g.x"""
        elif len(vp.shape) == 2:
            postr = """g.z, g.x"""
        else:
            postr = """g.z"""
        src = """
        rho_adj(%s) += pow(vp(%s), 2) * vp_adj(%s) \
                        + pow(vs(%s), 2) * vs_adj(%s) ;
        vp_adj(%s) = 2.0 * (vp(%s) * rho(%s)) * vp_adj(%s);
        vs_adj(%s) = 2.0 * (vs(%s) * rho(%s)) * vs_adj(%s);
        """ % ((postr,) * 13)
        grid = ComputeGrid(shape=[s - 2*vp.pad for s in vp.shape],
                           queue=self.queue,
                           origin=[vp.pad for _ in vp.shape])
        self.gpukernel(src, "adjoint", grid, vp, vs, rho)
        return vp, vs
    



if __name__ == '__main__':

    resc = ComputeRessource()
    nx = 24
    nz = 24

    grid = GridCL(resc.queues[0], shape=(nz, nx), pad=4, dtype=np.float32,
                  zero_boundary=False)
    v = np.tile(np.arange(nx, dtype=np.float32), [nz, 1])

    # veltrans =Velocity2Lame(grids={"gridpar": grid})
    # veltrans.linear_test()
    # veltrans.dot_test()

    veltrans =Velocity2LameCL(grids={"gridpar": grid})
    veltrans.linear_test()
    veltrans.backward_test()
    veltrans.dot_test()