
import unittest
import numpy as np
from SeisCL.python import Function, TapedFunction,  Tape, Variable, ReversibleFunction


class FunctionTester(unittest.TestCase):

    def test_cache_and_recover(self):
        """
        Methods Function.cache_states and Function.recover_states should save
        and recover the states
        """

        with Tape() as tape:

            var1 = Variable(data=np.array([False]),
                            lin=np.array([False]),
                            grad=np.array([False]))
            fun = Function()
            initial_states = fun.cache_states(var1)
            self.assertFalse(var1.data)
            self.assertFalse(initial_states["arg0"])
            var1.data = True
            fun.recover_states(initial_states, var1)
            self.assertFalse(var1.data)

    def test_cache_with_regions(self):
        """
        Methods Function.cache_states and Function.recover_states should save and
        recover the states with regions
        """

        class FunRegion(Function):

            def __init__(self):
                super().__init__()
                self.updated_states = ["var"]

            def updated_regions(self, var):
                return [(1,), (2,)]

            def forward(self, var):
                for region in self.updated_regions(var):
                    var.data[region] = True
                return var


        with Tape() as tape:

            var1 = Variable(data=np.array([False, False, False]),
                            lin=np.array([False, False, False]),
                            grad=np.array([False, False, False]))
            fun = FunRegion()
            initial_states = fun.cache_states(var1)
            self.assertFalse(np.all(var1.data))
            self.assertFalse(np.all(initial_states["var"]))
            fun.forward(var1)
            self.assertTrue(not var1.data[0] and var1.data[1] and var1.data[2])
            fun.recover_states(initial_states, var1)
            self.assertFalse(np.all(var1.data))
            self.assertTrue(fun.backward_test(var1, verbose=False) < 1e-12)

    def test_reversiblefunctions_backpropagation(self):
        """
        Reversible Functions should be able to pass backpropagation tests
        """

        class FunBack(ReversibleFunction):

            def __init__(self):
                super().__init__()
                self.updated_states = ["var"]

            def forward(self, var, backpropagate=False):
                if backpropagate:
                    var.data += 1
                else:
                    var.data -= 1
                return var

        with Tape() as tape:
            fun = FunBack()
            var1 = Variable(data=np.array([1]))
            self.assertEqual(var1.data[0], 1)
            self.assertFalse(fun.cache_states(var1))
            fun(var1)
            self.assertEqual(var1.data[0], 0)
            fun.recover_states([], var1)
            self.assertEqual(var1.data[0], 1)
            self.assertTrue(fun.backward_test(var1, verbose=False) < 1e-8)


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


class TapedFunctionTester(unittest.TestCase):

    def test_tapedfunction_as_decorator(self):
        """
        TapedFunction can be used as a decorator to function of multiples
        Functions
        """

        with Tape() as tape:
            @TapedFunction
            def fun(var):
                fun1 = Function1()
                fun2 = Function1()
                for ii in range(3):
                    fun1(var, var, var)
                    fun2(var, var, var)
                return var
            self.assertIsInstance(fun, TapedFunction)
            var1 = Variable(data=np.array([False]),
                            lin=np.array([False]),
                            grad=np.array([False]))
            self.assertFalse(var1.data)
            self.assertFalse(var1.lin)
            self.assertFalse(var1.grad)
            fun(var1)
            self.assertTrue(var1.data)
            self.assertFalse(var1.lin)
            self.assertFalse(var1.grad)
            self.assertIs(fun, tape.graph[0][0])

    def test_interaction(self):
        """
        Two TapedFunction can be defined without cross interactions
        """
        with Tape() as tape:
            @TapedFunction
            def fun1(var):
                fun1 = Function1()
                for ii in range(3):
                    fun1(var, var, var)
                return var
            @TapedFunction
            def fun2(var):
                fun1 = Function1()
                for ii in range(3):
                    fun1(var, var, var)
                return var
            var1 = Variable(data=np.array([False]),
                            lin=np.array([False]),
                            grad=np.array([False]))
            fun1(var1)
            self.assertTrue(fun1.localtape)
            self.assertFalse(fun2.localtape)
            fun2(var1)
            self.assertIsNot(fun1.localtape, fun2.localtape)
            self.assertIs(tape.graph[0][0],  fun1)
            self.assertIs(tape.graph[1][0], fun2)

    def test_nested(self):
        """
        A TapedFunction can contain Tapedfunctions
        """
        # A TapedFunction can contain Tapedfunctions
        #TODO Beware, if fun1 or fun2 is called before fun3,
        # fun3 will erase their tapes
        with Tape() as tape:
            @TapedFunction
            def fun1(var):
                fun1 = Function1()
                for ii in range(3):
                    fun1(var, var, var)
                return var
            @TapedFunction
            def fun2(var):
                fun1 = Function1()
                for ii in range(3):
                    fun1(var, var, var)
                return var
            @TapedFunction
            def fun3(var):
                fun1(var)
                fun2(var)
                return var

            var1 = Variable(data=np.array([False]),
                            lin=np.array([False]),
                            grad=np.array([False]))
            fun3(var1)
            self.assertTrue(fun1.localtape)
            self.assertTrue(fun2.localtape)
            self.assertTrue(fun3.localtape)
            self.assertIsNot(fun1.localtape, fun2.localtape)
            self.assertIsNot(fun1.localtape, fun3.localtape)
            self.assertIs(tape.graph[0][0], fun3)