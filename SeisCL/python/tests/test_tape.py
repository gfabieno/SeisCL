
import unittest
import numpy as np
from SeisCL.python import Tape, TapeHolder, Function, Variable


class Function1(Function):
    def __init__(self,  **kwargs):
        super().__init__()
        self.updated_states = ["vx"]

    def forward(self, vx, vy, vz, alt=None):
        vx.data = True
        return vx

    def linear(self, vx, vy, vz, alt=None):
        vx.lin = True
        return vx

    def adjoint(self, vx, vy, vz, alt=None):
        vx.grad = True
        return vx


class TapeTester(unittest.TestCase):

    def test_tape_as_context_manager(self):
        """
        Function and Variable should see the active tape, regardless of the Tape
        active at their creation. Tapeholder.tape should come back to its
        initial value after leaving context
        """
        tape0 = TapeHolder.tape
        with Tape() as tape:
            fun1 = Function1()
            self.assertEqual(tape, fun1.tape)
        self.assertEqual(tape0, fun1.tape)
        self.assertEqual(tape0, TapeHolder.tape)
        self.assertNotEqual(tape0, tape)

    def test_tape_to_chose_function_modes(self):
        """
        Tape set the behavior of __call__ of function, choosing between forward,
        linear  and adjoint.
        """

        with Tape() as tape:
            fun1 = Function1()
            var1 = Variable(data=np.array([False]),
                            lin=np.array([False]),
                            grad=np.array([False]))
            self.assertFalse(var1.data)
            self.assertFalse(var1.lin)
            self.assertFalse(var1.grad)
            fun1(var1, var1, var1)
            self.assertTrue(var1.data)
            self.assertFalse(var1.lin)
            self.assertFalse(var1.grad)
            var1.data = None
            tape.mode = "linear"
            fun1(var1, var1, var1)
            self.assertTrue(var1.data)
            self.assertTrue(var1.lin)
            self.assertFalse(var1.grad)
            var1.data = var1.lin = None
            tape.mode = "adjoint"
            fun1(var1, var1, var1)
            self.assertFalse(var1.data)
            self.assertFalse(var1.lin)
            self.assertTrue(var1.grad)
            var1.data = var1.lin = var1.grad = None
            fun1(var1, var1, var1, mode="adjoint")
            self.assertFalse(var1.data)
            self.assertFalse(var1.lin)
            self.assertTrue(var1.grad)
