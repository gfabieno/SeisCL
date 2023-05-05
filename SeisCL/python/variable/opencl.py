import numpy as np
from .variable import Variable

try:
    import pyopencl as cl
    from pyopencl.array import Array, to_device, empty, zeros
except ImportError:
    def compile(ctx, src, options):
        raise ImportError("pyopencl not installed")
#TODO add support for cuda


# TODO review how dimensions names are assigned
class VariableCL(Variable):

    def __init__(self, queue=None, data=None, shape=None, lin=None, grad=None,
                 initialize_method="zero", dtype=np.float32,
                 pad=None, differentiable=True):
        self.queue = queue
        #self.copy_array = CopyArrayCL(queue.context)
        super().__init__(data=data, shape=shape, lin=lin, grad=grad,
                         initialize_method=initialize_method, dtype=dtype,
                         pad=pad, differentiable=differentiable)

    def empty(self):
        return empty(self.queue, self.shape, self.dtype, order="F")

    def ones(self):
        return zeros(self.queue, self.shape, self.dtype, order="F") + 1

    def zero(self):
        return zeros(self.queue, self.shape, self.dtype, order="F")

    def todevice(self, data):
        if type(data) is np.ndarray:
            data = to_device(self.queue, data)
        elif type(data) is not Array:
            raise ValueError("Data type not supported: should be np.ndarray "
                             "or cl.array.Array")
        return data


class CopyArrayCL:

    def __init__(self, ctx):
        self.ctx = ctx
        self._kernel = None

    @property
    def kernel(self):
        if self._kernel is None:
            self._kernel = cl.Program(self.ctx, """__kernel void copy(
                __global char *dest,      
                const int offsetd, 
                const int stridexd, 
                const int strideyd,
                __global const char *src, 
                const int offsets, 
                const int stridexs, 
                const int strideys,
                const int word_size) {
    
                int write_idx = offsetd 
                                + get_global_id(0) 
                                + get_global_id(1) * stridexd 
                                + get_global_id(2) * strideyd ;
                int read_idx  = offsets 
                                + get_global_id(0) 
                                + get_global_id(1) * stridexs 
                                + get_global_id(2) * strideys;
                dest[write_idx] =  src[read_idx];
    
                }""").build()
        return self._kernel

    def __call__(self, dest, src):
        assert dest.dtype == src.dtype
        assert dest.shape == src.shape
        assert len(dest.shape) <= 2
        if len(dest.shape) == 1:
            dest.shape = (dest.shape[0], 1)
            src.shape = (src.shape[0], 1)
            dest.strides = (dest.strides[0], 0)
            src.strides = (src.strides[0], 0)

        self.kernel.copy(dest.queue,
                         (src.dtype.itemsize, src.shape[0], src.shape[1]),
                         None,
                         dest.base_data,
                         np.uint32(dest.offset),
                         np.uint32(dest.strides[0]),
                         np.uint32(dest.strides[1]),
                         src.base_data,
                         np.uint32(src.offset),
                         np.uint32(src.strides[0]),
                         np.uint32(src.strides[1]),
                         np.uint32(src.dtype.itemsize))


