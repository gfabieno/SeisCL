
#TODO cache data in Tape in memory pool
#TODO eager and compiled mode

class TapeHolder:
    tape = None


class Tape:
    """
    Keeps track of function calls as well as variables.
    """
    def __init__(self, cache_size=1, locked=False):
        self.variables = {} # keep track of all encountered variables.
        self.graph = []
        self.previous_tape = TapeHolder.tape
        self.mode = "forward"
        self.cache_size = cache_size
        TapeHolder.tape = self
        self.locked = locked

    # TODO use memory pooling to improve performance
    def append(self, kernel, *args, **kwargs):
        initial_states = kernel.cache_states(*args, **kwargs)
        self.graph.append((kernel, args, kwargs, initial_states))

    def pop(self):
        kernel, args, kwargs, initial_states = self.graph.pop()
        kernel.recover_states(initial_states, *args, **kwargs)
        return kernel, args, kwargs

    def empty(self):
        self.variables = {}
        self.graph = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TapeHolder.tape = self.previous_tape


TapeHolder.tape = Tape()










