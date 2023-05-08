
import unittest
import numpy as np
from SeisCL.python import FunctionGPU, ComputeRessource, VariableCL
from SeisCL.python import ComputeGrid


class FunctionGpuTester(unittest.TestCase):

    def get_fun(self, ndim):

        class Sum(FunctionGPU):
            def forward(self, a, b, c):
                src = """
                      c(%s) = a(%s) + b(%s);
                      """
                if len(a.shape) == 1:
                    src = src % (("g.z", )*3)
                elif len(a.shape) == 2:
                    src = src % (("g.z, g.x", )*3)
                elif len(a.shape) == 3:
                    src = src % (("g.z, g.y, g.x", )*3)
                grid = ComputeGrid(self.queue, a.shape, [0]*ndim)
                self.callgpu(src, "forward", grid, a, b, c)
                return c

            def linear(self, a, b, c):
                src = """
                      c_lin(%s) = a_lin(%s) + b_lin(%s);
                      """
                if len(a.shape) == 1:
                    src = src % (("g.z", )*3)
                elif len(a.shape) == 2:
                    src = src % (("g.z, g.x", )*3)
                elif len(a.shape) == 3:
                    src = src % (("g.z, g.y, g.x", )*3)
                grid = ComputeGrid(self.queue, a.shape, [0]*ndim)
                self.callgpu(src, "linear", grid, a, b, c)
                return c

            def adjoint(self, a, b, c):
                src = """
                      a_adj(%s) += c_adj(%s);
                      b_adj(%s) += c_adj(%s);
                      c_adj(%s) = 0;
                      """
                if len(a.shape) == 1:
                    src = src % (("g.z", )*5)
                elif len(a.shape) == 2:
                    src = src % (("g.z, g.x", )*5)
                elif len(a.shape) == 3:
                    src = src % (("g.z, g.y, g.x", )*5)
                grid = ComputeGrid(self.queue, a.shape, [0]*ndim)
                self.callgpu(src, "adjoint", grid, a, b, c)
                return a, b
        resc = ComputeRessource()
        sum = Sum(resc.queues[0])
        shape = (3,)*ndim
        a = VariableCL(resc.queues[0],
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape))
        b = VariableCL(resc.queues[0],
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape))
        c = VariableCL(resc.queues[0],
                       data=np.random.rand(*shape),
                       lin=np.random.rand(*shape))
        return sum, a, b, c

    def test_forward(self):

        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            c_np = a.data.get() + b.data.get()
            c = sum(a, b, c)
            self.assertTrue(np.allclose(c_np, c.data.get()))

    def test_linear(self):
        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            c_lin_np = a.lin.get() + b.lin.get()
            c_lin = sum(a, b, c, mode="linear")
            self.assertTrue(np.allclose(c_lin_np, c_lin.lin.get()))

    def test_dottest(self):
        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            self.assertLess(sum.dot_test(a, b, c), 1e-6)

    def test_backward_test(self):
        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            self.assertLess(sum.backward_test(a, b, c), 1e-12)

    def test_linear_test(self):
        for dim in range(1, 4):
            sum, a, b, c = self.get_fun(dim)
            self.assertLess(sum.linear_test(a, b, c), 1e-5)

