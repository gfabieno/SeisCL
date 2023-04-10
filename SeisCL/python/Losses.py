import numpy as np
from SeisCL.python.tape import Function, Variable
import unittest


class Loss(Function):

    def __init__(self):
        super().__init__()
        self.updated_states = ["loss"]


class NormalizedL2(Loss):

    def forward(self, loss, dmod, dobs, eps=1e-12):
        norm = np.var(dobs.data) + eps
        loss.data += np.sum((dobs.data - dmod.data)**2) / norm
        return loss

    def linear(self, loss, dmod, dobs, eps=1e-12):
        norm = np.var(dobs.data) + eps
        loss.lin += -np.sum(2*(dobs.data - dmod.data) / norm * dmod.lin)
        if dobs.differentiable:
            loss.lin += np.sum(2*(dobs.data - dmod.data) / norm * dobs.lin)
        return loss

    def adjoint(self, loss, dmod, dobs, eps=1e-12):
        norm = np.var(dobs.data) + eps
        dmod.grad += -2*(dobs.data - dmod.data) / norm * loss.grad
        if dobs.differentiable:
            dobs.grad += 2*(dobs.data - dmod.data) / norm * loss.grad
            return dmod, dobs
        else:
            return dmod


class LossTester(unittest.TestCase):

    def setUp(self):
        self.dmod = Variable(shape=(100, 200), initialize_method="random")
        self.dobs = Variable(shape=(100, 200), differentiable=False)
        self.loss = Variable(shape=(1,), initialize_method="random")

    def test_NormalizedL2(self):

        loss = NormalizedL2()
        self.assertLess(loss.backward_test(self.loss, self.dmod, self.dobs),
                        1e-12)
        self.assertLess(loss.linear_test(self.loss, self.dmod, self.dobs),
                        1e-12)
        self.assertLess(loss.dot_test(self.loss, self.dmod, self.dobs), 1e-12)