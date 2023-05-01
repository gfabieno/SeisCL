import unittest
from SeisCL.python import Function, FunctionGPU, ComputeRessource, VariableCL, Variable, ComputeGrid
from SeisCL.python import get_header_stencil, FDCoefficients
import numpy as np
from copy import deepcopy



class DerivativeLocalMemory(FunctionGPU):

    def __init__(self, queue, ndim, order, local_size=None, load_only=False):
        super().__init__(queue, local_size=local_size)
        self.load_only = load_only
        self.header, self.local_header = get_header_stencil(order, ndim,
                                                            local_size=local_size,
                                                            with_local_ops=False)

    def forward(self, a, b, c):

        src = self.local_header
        if self.load_only:
            src += """
                          load_local_in(a, lvar);
                          load_local_halox(a, lvar);
                          load_local_haloy(a, lvar);
                          load_local_haloz(a, lvar);
                          BARRIER
                          gridstop(g);
                          b(%s) = lvar(%s);
                          c(%s) = lvar(%s);
                          """
        else:
            src += """
                          load_local_in(a, lvar);
                          load_local_halox(a, lvar);
                          load_local_haloy(a, lvar);
                          load_local_haloz(a, lvar);
                          BARRIER
                          gridstop(g);
                          b(%s) += Dxp(lvar) + Dyp(lvar) + Dzp(lvar);
                          c(%s) += Dxm(lvar) + Dym(lvar) + Dzm(lvar);
                          """


        if len(a.shape) == 1:
            src = src.replace("Dxp(lvar) + Dyp(lvar) +", "")
            src = src.replace("Dxm(lvar) + Dym(lvar) +", "")
            src = src.replace("load_local_halox(a, lvar);", "")
            src = src.replace("load_local_haloy(a, lvar);", "")
            if self.load_only:
                src = src % (("g.z", "g.lz")*2)
            else:
                src = src % (("g.z", )*2)
        elif len(a.shape) == 2:
            src = src.replace("Dyp(lvar) +", "")
            src = src.replace("Dym(lvar) +", "")
            src = src.replace("load_local_haloy(a, lvar);", "")
            if self.load_only:
                src = src % (("g.z, g.x", "g.lz, g.lx")*2)
            else:
                src = src % (("g.z, g.x", )*2)
        elif len(a.shape) == 3:
            if self.load_only:
                src = src % (("g.z, g.y, g.x", "g.lz, g.ly, g.lx")*2)
            else:
                src = src % (("g.z, g.y, g.x", )*2)
        grid = ComputeGrid(shape=[s - 2 * a.pad for s in a.shape],
                           queue=self.queue,
                           origin=[a.pad for _ in a.shape])
        self.gpukernel(src, "forward", grid, a, b, c)
        return c

    def linear(self, a, b, c):

        src = self.local_header
        src += """
                      load_local_in(a_lin, lvar);
                      load_local_halox(a_lin, lvar);
                      load_local_haloy(a_lin, lvar);
                      load_local_haloz(a_lin, lvar);
                      BARRIER
                    
                      gridstop(g);  
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

        grid = ComputeGrid(shape=[s - 2 * a.pad for s in a.shape],
                           queue=self.queue,
                           origin=[a.pad for _ in a.shape])
        self.gpukernel(src, "linear", grid, a, b, c)

    def adjoint(self, a, b, c):

        src = self.local_header
        src += """
                      load_local_in(c_adj, lvar);
                      load_local_halox(c_adj, lvar);
                      load_local_haloy(c_adj, lvar);
                      load_local_haloz(c_adj, lvar);
                      BARRIER
                      float a_up = Dxp(lvar) + Dyp(lvar) + Dzp(lvar);
                     
                      BARRIER
                      load_local_in(b_adj, lvar);
                      load_local_halox(b_adj, lvar);
                      load_local_haloy(b_adj, lvar);
                      load_local_haloz(b_adj, lvar);
                      BARRIER
                      a_up += Dxm(lvar) + Dym(lvar) + Dzm(lvar);
                      
                      gridstop(g);
                      a_adj(%s) -= a_up;
                      """

        if len(a.shape) == 1:
            src = src.replace("Dxp(lvar) + Dyp(lvar) +", "")
            src = src.replace("Dxm(lvar) + Dym(lvar) +", "")
            src = src.replace("load_local_halox(b_adj, lvar);", "")
            src = src.replace("load_local_haloy(b_adj, lvar);", "")
            src = src.replace("load_local_halox(c_adj, lvar);", "")
            src = src.replace("load_local_haloy(c_adj, lvar);", "")
            src = src % "g.z"
        elif len(a.shape) == 2:
            src = src.replace("Dyp(lvar) +", "")
            src = src.replace("Dym(lvar) +", "")
            src = src.replace("load_local_haloy(b_adj, lvar);", "")
            src = src.replace("load_local_haloy(c_adj, lvar);", "")
            src = src % "g.z, g.x"
        elif len(a.shape) == 3:
            src = src % "g.z, g.y, g.x"

        grid = ComputeGrid(shape=[s - 2 * a.pad for s in a.shape],
                           queue=self.queue,
                           origin=[a.pad for _ in a.shape])
        self.gpukernel(src, "adjoint", grid, a, b, c)
        return a


class DerivativeGlobalMemory(FunctionGPU):

    def __init__(self, queue, ndim, order):
        super().__init__(queue)
        self.header, _ = get_header_stencil(order, ndim)

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
        grid = ComputeGrid(shape=[s - 2 * a.pad for s in a.shape],
                           queue=self.queue,
                           origin=[a.pad for _ in a.shape])
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
        grid = ComputeGrid(shape=[s - 2 * a.pad for s in a.shape],
                           queue=self.queue,
                           origin=[a.pad for _ in a.shape])
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
        grid = ComputeGrid(shape=[s - 2 * a.pad for s in a.shape],
                           queue=self.queue,
                           origin=[a.pad for _ in a.shape])
        self.gpukernel(src, "adjoint", grid, a, b, c)
        return a


class DerivativeNumpy(Function):

    def __init__(self, ndim, order):
        super().__init__()
        self.ndim = ndim
        self.order = order
        h = FDCoefficients(order).coefs
        pad = order // 2

        def Dzp(a):
            out = np.zeros_like(a)
            nd = len(a.shape)
            for ii in range(order//2):
                i1 = pad+ii+1
                i2 = a.shape[-1] - pad+ii+1
                i3 = pad-ii
                i4 = a.shape[-1] - pad-ii
                if nd == 1:
                    out[pad:-pad] += h[ii]*(a[i1:i2] - a[i3:i4])
                elif nd == 2:
                    out[pad:-pad, pad:-pad] += h[ii]*(a[i1:i2, pad:-pad]
                                                      - a[i3:i4, pad:-pad])
                else:
                    out[pad:-pad, pad:-pad, pad:-pad] += h[ii]*(a[i1:i2, pad:-pad, pad:-pad]
                                                      - a[i3:i4, pad:-pad, pad:-pad])
            return out

        def Dzm(a):
            out = np.zeros_like(a)
            nd = len(a.shape)
            for ii in range(order//2):
                i1 = pad+ii
                i2 = a.shape[-1]-pad+ii
                i3 = pad-ii-1
                i4 = a.shape[-1]-pad-ii-1
                if nd == 1:
                    out[pad:-pad] += h[ii]*(a[i1:i2] - a[i3:i4])
                elif nd == 2:
                    out[pad:-pad, pad:-pad] += h[ii]*(a[i1:i2, pad:-pad]
                                                      - a[i3:i4, pad:-pad])
                else:
                    out[pad:-pad, pad:-pad, pad:-pad] += h[ii]*(a[i1:i2, pad:-pad, pad:-pad]
                                                                - a[i3:i4, pad:-pad, pad:-pad])
            return out

        self.Dzp = Dzp
        self.Dzm = Dzm
        if ndim == 3:
            self.Dyp = lambda a: Dzp(a.swapaxes(0, 1)).swapaxes(0, 1)
            self.Dym = lambda a: Dzm(a.swapaxes(0, 1)).swapaxes(0, 1)
        else:
            self.Dyp = lambda a: 0
            self.Dym = lambda a: 0
        if ndim > 1:
            self.Dxp = lambda a: Dzp(a.swapaxes(0, -1)).swapaxes(0, -1)
            self.Dxm = lambda a: Dzm(a.swapaxes(0, -1)).swapaxes(0, -1)
        else:
            self.Dxp = lambda a: 0
            self.Dxm = lambda a: 0

    def forward(self, a, b, c):
        b.data += self.Dxp(a.data) + self.Dyp(a.data) + self.Dzp(a.data)
        c.data += self.Dxm(a.data) + self.Dym(a.data) + self.Dzm(a.data)
        return b, c

    def linear(self, a, b, c):
        b.lin += self.Dxp(a.lin) + self.Dyp(a.lin) + self.Dzp(a.lin)
        c.lin += self.Dxm(a.lin) + self.Dym(a.lin) + self.Dzm(a.lin)

    def adjoint(self, a, b, c):
        a.grad -= self.Dxp(c.grad) + self.Dyp(c.grad) + self.Dzp(c.grad)
        a.grad -= self.Dxm(b.grad) + self.Dym(b.grad) + self.Dzm(b.grad)
        return a


class FDFunctionGpuTester(unittest.TestCase):

    def get_function(self, resc, ndim, order, local_memory, load_only=False):

        if local_memory:
            local_size = (4, ) * ndim
            der = DerivativeLocalMemory(resc.queues[0], ndim, order,
                                        local_size=local_size,
                                        load_only=load_only)
        else:
            der = DerivativeGlobalMemory(resc.queues[0], ndim, order)
        shape = (20,)*ndim
        a = VariableCL(resc.queues[0], pad=order//2,
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape),
                       grad=np.random.rand(*shape))
        b = VariableCL(resc.queues[0], pad=order//2,
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape),
                       grad=np.random.rand(*shape))
        c = VariableCL(resc.queues[0], pad=order//2,
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape),
                       grad=np.random.rand(*shape))

        return der, a, b, c

    def test_global_memory(self):

        resc = ComputeRessource()

        # der, a, b, c = self.get_function(3, 8, True)
        # der.adjoint(a, b, c)
        for ndim in range(1, 4):
            for order in range(1, 7):
                for mode in ["forward", "linear", "adjoint"]:
                    with self.subTest(ndim=ndim, order=2*order, mode=mode):
                        der = DerivativeGlobalMemory(resc.queues[0], ndim,
                                                     2*order)
                        dernp = DerivativeNumpy(ndim, 2*order)
                        shape = (20,)*ndim
                        pad = order
                        a, b, c = (Variable(pad=pad,
                                            data=np.random.rand(*shape),
                                            lin=np.random.rand(*shape),
                                            grad=np.random.rand(*shape))
                                   for _ in range(3))
                        ag, bg, cg = (VariableCL(queue=resc.queues[0], pad=pad,
                                                 data=el.data,
                                                 lin=el.lin,
                                                 grad=el.grad)
                                      for el in [a, b, c])
                        der.__getattribute__(mode)(ag, bg, cg)
                        dernp.__getattribute__(mode)(a, b, c)

                        if mode == "forward":
                            self.assertTrue(np.allclose(bg.data.get(), b.data,
                                                        atol=1e-6))
                            self.assertTrue(np.allclose(cg.data.get(), c.data,
                                                        atol=1e-6))
                        elif mode == "linear":
                            self.assertTrue(np.allclose(bg.lin.get(), b.lin,
                                                        atol=1e-6))
                            self.assertTrue(np.allclose(cg.lin.get(), c.lin,
                                                        atol=1e-6))
                        elif mode == "adjoint":
                            self.assertTrue(np.allclose(ag.grad.get(), a.grad,
                                                        atol=1e-6))

    def test_load_local(self):

        resc = ComputeRessource()

        for ndim in range(1, 4):
            for order in range(1, 7):
                for mode in ["forward"]:
                    with self.subTest(ndim=ndim, order=2*order, mode=mode):
                        derl, al, bl, cl = self.get_function(resc, ndim,
                                                             2*order,
                                                             local_memory=True,
                                                             load_only=True)
                        derl.__getattribute__(mode)(al, bl, cl)
                        if mode == "forward":
                            self.assertTrue(np.allclose(bl.data.get()[bl.valid],
                                                        al.data.get()[al.valid])
                                            )

    def test_local_memory(self):

        resc = ComputeRessource()

        for ndim in range(1, 4):
            for order in range(1, 7):
                for mode in ["forward", "linear", "adjoint"]:
                    with self.subTest(ndim=ndim, order=2*order, mode=mode):
                        derg, ag, bg, cg = self.get_function(resc, ndim, 2*order, False)
                        derl, al, bl, cl = self.get_function(resc, ndim, 2*order, True)
                        al = deepcopy(ag)
                        bl = deepcopy(bg)
                        cl = deepcopy(cg)

                        derg.__getattribute__(mode)(ag, bg, cg)
                        derl.__getattribute__(mode)(al, bl, cl)

                        if mode == "forward":
                            self.assertTrue(np.allclose(cl.data.get(),
                                                        cg.data.get()))
                        elif mode == "linear":
                            self.assertTrue(np.allclose(cl.lin.get(),
                                                        cg.lin.get()))
                        elif mode == "adjoint":
                            d = (al.grad.get() - ag.grad.get())/(al.grad.get()+1e-12)
                            self.assertTrue(np.allclose(al.grad.get(),
                                                        ag.grad.get(),
                                                        atol=1e-7))
                            self.assertTrue(np.allclose(bl.grad.get(),
                                                        bg.grad.get()))
                            self.assertTrue(np.allclose(cl.grad.get(),
                                                        cg.grad.get()))

