
import unittest
import numpy as np
from SeisCL.python import VariableCL, ComputeRessource
from SeisCL.python.variable.opencl import Array


class VariableTester(unittest.TestCase):

    def test_variable_in_device(self):
        resc = ComputeRessource()
        data = np.random.rand(10, 10)
        lin = np.random.rand(10, 10)
        grad = np.random.rand(10, 10)
        var = VariableCL(resc.queues[0], data=data, lin=lin, grad=grad)
        self.assertIsInstance(var.data, Array)
        self.assertIsInstance(var.lin, Array)
        self.assertIsInstance(var.grad, Array)
        self.assertTrue(np.allclose(var.data.get(), data))
        self.assertTrue(np.allclose(var.lin.get(), lin))
        self.assertTrue(np.allclose(var.grad.get(), grad))

    def test_copy_variable(self):
        from copy import copy, deepcopy
        resc = ComputeRessource()
        data = np.random.rand(10, 10)
        lin = np.random.rand(10, 10)
        grad = np.random.rand(10, 10)
        var = VariableCL(resc.queues[0], data=data, lin=lin, grad=grad)
        var2 = deepcopy(var)
        self.assertIsInstance(var2.data, Array)
        self.assertIsInstance(var2.lin, Array)
        self.assertIsInstance(var2.grad, Array)
        self.assertTrue(np.allclose(var2.data.get(), data))
        self.assertTrue(np.allclose(var2.lin.get(), lin))
        self.assertTrue(np.allclose(var2.grad.get(), grad))
        self.assertIsNot(var2.data, var.data)
        self.assertIsNot(var2.lin, var.lin)
        self.assertIsNot(var2.grad, var.grad)
        var3 = copy(var)
        self.assertIs(var3.data, var.data)
        self.assertIs(var3.lin, var.lin)
        self.assertIs(var3.grad, var.grad)
