import numpy as np
from copy import copy


class Variable:
    """

    """
    def __init__(self, data=None, shape=None, lin=None, grad=None,
                 initialize_method="zero", dtype=np.float32,
                 pad=None, differentiable=True):

        #TODO do we really need to keep track of the tape?
        # if self.tape.locked:
        #     raise PermissionError("Tape locked: "
        #                           "Cannot create a new Variable inside "
        #                           "a TapedFunction ")

        self.initialize_method = initialize_method
        self.dtype = dtype
        self.smallest = np.nextafter(dtype(0), dtype(1))
        self.eps = np.finfo(dtype).eps
        self.pad = pad
        self.data = data
        if data is None:
            if shape is None:
                raise ValueError("Either data or shape must be provided")
            else:
                self.shape = shape
        else:
            self.shape = data.shape
        self.lin = lin # contains the small data perturbation
        self.differentiable = differentiable
        self.grad = grad # contains the gradient
        self.last_update = None # The last kernel that updated the state


    @property
    def data(self):
        if self._data is None:
            self._data = self.initialize()
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
        else:
            if type(data) is np.ndarray:
                data = np.require(data, dtype=self.dtype, requirements='F')
            self._data = self.todevice(data)

    @property
    def lin(self):
        if self._lin is None:
            self._lin = self.initialize()
        return self._lin

    @lin.setter
    def lin(self, lin):
        if lin is None:
            self._lin = None
        else:
            if type(lin) is np.ndarray:
                lin = np.require(lin, dtype=self.dtype, requirements='F')
            self._lin = self.todevice(lin)

    @property
    def grad(self):
        if self._grad is None and self.differentiable:
            self._grad = self.initialize(method="ones")
        return self._grad

    @grad.setter
    def grad(self, grad):
        if not self.differentiable and grad is not None:
            raise ValueError("Variable is not Differentiable,"
                             " cannot compute gradient")
        if grad is None:
            self._grad = None
        else:
            if type(grad) is np.ndarray:
                grad = np.require(grad, dtype=self.dtype, requirements='F')
            self._grad = self.todevice(grad)

    @property
    def valid(self):
        if self.pad is None:
            return Ellipsis
        else:
            return tuple([slice(self.pad, -self.pad)] * len(self.shape))

    def initialize(self, method=None):
        if method is None:
            method = self.initialize_method
        if method == "zero":
            return self.zero()
        elif method == "random":
            return self.random()
        elif method == "ones":
            return self.ones()

    def empty(self):
        return np.empty(self.shape, dtype=self.dtype, order="F")

    def ones(self):
        return np.ones(self.shape, dtype=self.dtype, order="F")

    def zero(self):
        return np.zeros(self.shape, dtype=self.dtype, order="F")

    def random(self):
        if self.pad:
            state = np.zeros(self.shape, dtype=self.dtype, order="F")
            state[self.valid] = np.random.rand(*state[self.valid].shape)*10e6
        else:
            state = np.random.rand(*self.shape).astype(self.dtype)
        return self.todevice(np.require(state, requirements='F'))

    def todevice(self, mem):
        return mem

    #TODO create cache in tape
    def create_cache(self, cache_size=1, regions=None):
        if regions:
            cache = []
            for region in regions:
                shape = [0 for _ in range(len(self.shape))]
                for ii in range(len(self.shape)):
                    if region[ii] is Ellipsis:
                        shape[ii] = self.shape[ii]
                    else:
                        indices = region[ii].indices(self.shape[ii])
                        shape[ii] = int((indices[1]-indices[0])/indices[2])
                cache.append(np.empty(shape + [cache_size], dtype=self.dtype,
                                      order="F"))
            return cache
        else:
            return np.empty(list(self.shape) + [cache_size],
                            dtype=self.dtype, order="F")

    def xyz2lin(self, *args):
        return np.ravel_multi_index([np.array(el)+self.pad for el in args],
                                    self.shape, order="F")

    def __deepcopy__(self, memo):
        var = copy(self)
        if var.data is not None:
            var.data = self.data.copy()
        if var.lin is not None:
            var.lin = self.lin.copy()
        if var.grad is not None:
            var.grad = self.grad.copy()
        return var


