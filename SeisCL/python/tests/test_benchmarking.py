import numpy as np
from SeisCL.python import ComputeRessource, FunctionGPU, ComputeGrid, VariableCL
import time


def opencl_kernel_overhead():

    resc = ComputeRessource()
    nd = 3
    narg = 20 #overhead is proportional to narg
    shape = (3,)*nd
    grid = ComputeGrid(resc.queues[0], shape, [0]*nd)
    a = VariableCL(resc.queues[0],
                   data=np.random.rand(*shape),
                   lin=np.random.rand(*shape))

    class Empty(FunctionGPU):
        def forward(self, a, *args):
            src = """
            a(%s) += 1;
                      """
            if len(a.shape) == 1:
                src = src % (("g.z", ))
            elif len(a.shape) == 2:
                src = src % (("g.z, g.x", ))
            elif len(a.shape) == 3:
                src = src % (("g.z, g.y, g.x", ))

            self.callgpu(src, "forward", grid, a)
            return a
    t1 = time.time()
    empty = Empty(resc.queues[0])
    for i in range(5000):
        empty(a, *(a,)*narg)
    resc.queues[0].finish()
    t2 = time.time()
    print(t2-t1)


def cuda_kernel_overhead():

    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    stream = cuda.Stream()
    a = np.random.randn(400).astype(np.float32)
    agpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(agpu, a)
    grid = (1, 1)
    block = (400, 1, 1)
    for narg in [1, 10, 20, 40]:
        mod = SourceModule("""
        __global__ void multiply_them(float *a, %s)
        {
          const int i = threadIdx.x;
          a[i] += 1;
        }
        """ % (", ".join(["float *a%d" % i for i in range(narg)])))

        multiply_them = mod.get_function("multiply_them")

        print("bencmarking the __call__ method narg=%d" % narg)
        t1 = time.time()
        for i in range(5000):
            multiply_them(agpu, *(agpu,)*narg, block=(400, 1, 1), grid=(1, 1))
        t2 = time.time()
        print(t2-t1)
        stream.synchronize()

        multiply_them.prepare(["P"]*(narg+1))
        print("bencmarking the prepared_call method, narg=%d" % narg)
        t1 = time.time()
        for i in range(5000):
            multiply_them.prepared_call(grid, block, agpu, *(agpu,)*narg)
        t2 = time.time()
        print(t2-t1)
        stream.synchronize()

        print("bencmarking the prepared_async_call method, narg=%d" % narg)
        t1 = time.time()
        for i in range(5000):
            multiply_them.prepared_async_call(grid, block, stream, agpu, *(agpu,)*narg)
        t2 = time.time()
        print(t2-t1)
        stream.synchronize()


if __name__ == "__main__":
    #opencl_kernel_overhead()
    cuda_kernel_overhead()