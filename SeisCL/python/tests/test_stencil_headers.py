import unittest
from SeisCL.python import FunctionGPU, ComputeRessource, VariableCL, Grid
from SeisCL.python import get_header_stencil
import numpy as np


class DerivativeLocalMemory(FunctionGPU):

    def __init__(self, queue, ndim, order, local_size=None):
        super().__init__(queue, local_size=local_size)
        self.header = get_header_stencil(order, ndim,
                                         local_memory=True,
                                         with_local_ops=False)

    def forward(self, a, b, c):

        src = """
                      LOCID float lvar[__LSIZE__];
                      load_local_in(a, lvar);
                      load_local_halox(a, lvar);
                      load_local_haloy(a, lvar);
                      load_local_haloz(a, lvar);
                      BARRIER
                      
                      b(%s) += Dxp(lvar) + Dyp(lvar) + Dzp(lvar);
                      c(%s) += Dxm(lvar) + Dym(lvar) + Dzm(lvar);
                      """


        if len(a.shape) == 1:
            src = src.replace("Dxp(lvar) + Dyp(lvar) +", "")
            src = src.replace("Dxm(lvar) + Dym(lvar) +", "")
            src = src.replace("load_local_halox(a, lvar);", "")
            src = src.replace("load_local_haloy(a, lvar);", "")
            src = src % (("g.z", )*2)
        elif len(a.shape) == 2:
            src = src.replace("Dyp(lvar) +", "")
            src = src.replace("Dym(lvar) +", "")
            src = src.replace("load_local_haloy(a, lvar);", "")
            src = src % (("g.z, g.x", )*2)
        elif len(a.shape) == 3:
            src = src % (("g.z, g.y, g.x", )*2)
        grid = Grid(shape=a.shape, pad = a.pad)
        self.gpukernel(src, "forward", grid, a, b, c)
        return c

    def linear(self, a, b, c):

        src = """
                      LOCID float lvar[__LSIZE__];
                      load_local_in(a_lin, lvar);
                      load_local_halox(a_lin, lvar);
                      load_local_haloy(a_lin, lvar);
                      load_local_haloz(a_lin, lvar);
                      BARRIER
                      
                      b_lin(%s) += Dxp(lvar) + Dyp(lvar) + Dzp(lvar);
                      c_lin(%s) += Dxm(lvar) + Dym(lvar) + Dzm(lvar);
                      """

        if len(a.shape) == 1:
            src = src.replace("Dxp(lvar) + Dyp(lvar) +", "")
            src = src.replace("Dxm(lvar) + Dym(lvar) +", "")
            src = src.replace("load_local_halox(a_lin, lvar);", "")
            src = src.replace("load_local_haloy(a_lin, lvar);", "")
            src = src % (("g.z", )*2)
        elif len(a.shape) == 2:
            src = src.replace("Dyp(lvar) +", "")
            src = src.replace("Dym(lvar) +", "")
            src = src.replace("load_local_haloy(a_lin, lvar);", "")
            src = src % (("g.z, g.x", )*2)
        elif len(a.shape) == 3:
            src = src % (("g.z, g.y, g.x", )*2)

        grid = Grid(shape=a.shape, pad=a.pad)
        self.gpukernel(src, "linear", grid, a, b, c)

    def adjoint(self, a, b, c):

        src = """
                      LOCID float lvar[__LSIZE__];
                      load_local_in(c_adj, lvar);
                      load_local_halox(c_adj, lvar);
                      load_local_haloy(c_adj, lvar);
                      load_local_haloz(c_adj, lvar);
                      BARRIER
                      a_adj(%s) -= Dxp(lvar) + Dyp(lvar) + Dzp(lvar);
                      
                      BARRIER
                      load_local_in(b_adj, lvar);
                      load_local_halox(b_adj, lvar);
                      load_local_haloy(b_adj, lvar);
                      load_local_haloz(b_adj, lvar);
                      BARRIER
                      a_adj(%s) -= Dxm(lvar) + Dym(lvar) + Dzm(lvar);
                      """

        if len(a.shape) == 1:
            src = src.replace("Dxp(lvar) + Dyp(lvar) +", "")
            src = src.replace("Dxm(lvar) + Dym(lvar) +", "")
            src = src.replace("load_local_halox(b_adj, lvar);", "")
            src = src.replace("load_local_haloy(b_adj, lvar);", "")
            src = src.replace("load_local_halox(c_adj, lvar);", "")
            src = src.replace("load_local_haloy(c_adj, lvar);", "")
            src = src % (("g.z", )*2)
        elif len(a.shape) == 2:
            src = src.replace("Dyp(lvar) +", "")
            src = src.replace("Dym(lvar) +", "")
            src = src.replace("load_local_haloy(b_adj, lvar);", "")
            src = src.replace("load_local_haloy(c_adj, lvar);", "")
            src = src % (("g.z, g.x", )*2)
        elif len(a.shape) == 3:
            src = src % (("g.z, g.y, g.x", )*2)

        grid = Grid(shape=a.shape, pad = a.pad)
        self.gpukernel(src, "adjoint", grid, a, b, c)
        return a


class DerivativeGlobalMemory(FunctionGPU):

    def __init__(self, queue, ndim, order):
        super().__init__(queue)
        self.header = get_header_stencil(order, ndim, local_memory=False)

    def forward(self, a, b, c):

        src = """
                  b(%s) += Dxp(a) + Dyp(a) + Dzp(a);
                  c(%s) += Dxm(a) + Dym(a) + Dzm(a);
                  """

        if len(a.shape) == 1:
            src = src.replace("Dxp(a) + Dyp(a) +", "")
            src = src.replace("Dxm(a) + Dym(a) +", "")
            src = src % (("g.z", )*2)
        elif len(a.shape) == 2:
            src = src.replace("Dyp(a) +", "")
            src = src.replace("Dym(a) +", "")
            src = src % (("g.z, g.x", )*2)
        elif len(a.shape) == 3:
            src = src % (("g.z, g.y, g.x", )*2)
        grid = Grid(shape=a.shape, pad = a.pad)
        self.gpukernel(src, "forward", grid, a, b, c)
        return c

    def linear(self, a, b, c):

        src = """
                      b_lin(%s) += Dxp(a_lin) + Dyp(a_lin) + Dzp(a_lin);
                      c_lin(%s) += Dxm(a_lin) + Dym(a_lin) + Dzm(a_lin);
                      """

        if len(a.shape) == 1:
            src = src.replace("Dxp(a_lin) + Dyp(a_lin) +", "")
            src = src.replace("Dxm(a_lin) + Dym(a_lin) +", "")
            src = src % (("g.z", )*2)
        elif len(a.shape) == 2:
            src = src.replace("Dyp(a_lin) +", "")
            src = src.replace("Dym(a_lin) +", "")
            src = src % (("g.z, g.x", )*2)
        elif len(a.shape) == 3:
            src = src % (("g.z, g.y, g.x", )*2)
        grid = Grid(shape=a.shape, pad=a.pad)
        self.gpukernel(src, "linear", grid, a, b, c)

    def adjoint(self, a, b, c):

        src = """
                      a_adj(%s) -= Dxp(c_adj) + Dyp(c_adj) + Dzp(c_adj);
                      a_adj(%s) -= Dxm(b_adj) + Dym(b_adj) + Dzm(b_adj);
                      """
        if len(a.shape) == 1:
            src = src.replace("Dxp(c_adj) + Dyp(c_adj) +", "")
            src = src.replace("Dxm(b_adj) + Dym(b_adj) +", "")
            src = src % (("g.z", )*2)
        elif len(a.shape) == 2:
            src = src.replace("Dyp(c_adj) +", "")
            src = src.replace("Dym(b_adj) +", "")
            src = src % (("g.z, g.x", )*2)
        elif len(a.shape) == 3:
            src = src % (("g.z, g.y, g.x", )*2)
        grid = Grid(shape=a.shape, pad = a.pad)
        self.gpukernel(src, "adjoint", grid, a, b, c)
        return a


class FDFunctionGpuTester(unittest.TestCase):

    def get_function(self, ndim, order, local_memory):


        resc = ComputeRessource()
        if local_memory:
            local_size = (4, ) * ndim
            der = DerivativeLocalMemory(resc.queues[0], ndim, order,
                                        local_size=local_size)
        else:
            der = DerivativeGlobalMemory(resc.queues[0], ndim, order)
        shape = (20,)*ndim
        a = VariableCL(resc.queues[0], pad=2,
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape),
                       grad=np.random.rand(*shape))
        b = VariableCL(resc.queues[0], pad=2,
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape),
                       grad=np.random.rand(*shape))
        c = VariableCL(resc.queues[0], pad=2,
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape),
                       grad=np.random.rand(*shape))
        return der, a, b, c

    def test_without_local_memory(self):

        # der, a, b, c = self.get_function(3, 8, False)
        # der.adjoint(a, b, c)

        for local_memory in [False, True]:
            for ndim in range(1, 4):
                for order in range(1, 7):
                    for mode in ["forward", "linear", "adjoint"]:
                        with self.subTest(ndim=ndim, order=2*order, mode=mode,
                                          local_memory=local_memory):
                            der, a, b, c = self.get_function(ndim, 2*order,
                                                             local_memory)
                            der.__getattribute__(mode)(a, b, c)

