
#TODO cache data in Tape in memory pool
#TODO eager and compiled mode
from inspect import signature, Parameter, _empty


class Tape:
    """
    Keeps track of function calls as well as variables.
    """
    def __init__(self):
        self.variables = {} #keep track of all encountered variables.
        self.graph = []

    def append(self, kernel):
        self.graph.append(kernel)

    def pop(self):
        return self.graph.pop()

    def add_variable(self, var):
        if var.name in self.variables:
            raise NameError("Variable name already exists in tape")
        self.variables[var.name] = var

    def empty(self):
        self.variables = {} #keep track of all encountered variables.
        self.graph = []


class TapeHolder:
    _tape = Tape()


class Variable(TapeHolder):
    """

    """
    def __init__(self, name):
        self.name = name
        self._tape.add_variable(self)
        self.data = None # contains the data
        self.linear = None # contains the small data perturbation
        self.grad = None # contains the gradient
        self.last_update = None # The last kernel that updated the state


class Function(TapeHolder):
    """
    Transforms States and its operation to the tape, if required.
    """

    def __init__(self,  **kwargs):
        self.updated_states = ["vx"]
        self.signature = signature(self.forward)
        self.required_states = [name for name, par
                                in self.signature.parameters.items()
                                if par.kind == Parameter.POSITIONAL_OR_KEYWORD]
        self._cache = {}

    def __call__(self, *args, **kwargs):
        self._tape.append(self)
        a = self.signature.bind(*args, **kwargs)
        a.apply_defaults()
        self.cache_states(a.arguments)

        return self.forward(*args, **kwargs)

    def cache_states(self, args):
        for el in self.updated_states:
            self._cache[el] = args[el]

    def forward(self, vx, vy, vz, alt=None):
        """
        Function that takes as inputs Variables and return Variables
        :param vx:
        :param vy:
        :param vz:
        :return:
        """
        return vx

    def call_linear(self):
        pass

    def gradient(self):
        pass


class Function2(Function):
    def __init__(self,  **kwargs):
        self.updated_states = ["arg2"]


fun1 = Function()
var1 = Variable("var1")
print(fun1.required_states)
fun1(var1, var1, var1)
print(fun1.updated_states)
print(fun1._cache)
# fun2 = Function()
# var1 = Variable()
# var2 = Variable()
# for ii in range(1000):
#     states = fun1(var1)
#     states = fun2(states)
#
# print(len(Function._tape.graph))

