/*Macros for writing kernels compatible with CUDA and OpenCL */

#ifdef __OPENCL_VERSION__
    #define FUNDEF __kernel
    #define LFUNDEF
    #define GLOBARG __global
    #define LOCARG __local float *lvar
    #define LOCDEF
    #define BARRIER barrier(CLK_LOCAL_MEM_FENCE);
#else
    #define FUNDEF extern "C" __global__
    #define LFUNDEF extern "C" __device__
    #define GLOBARG
    #define LOCARG float *nullarg
    #define LOCDEF extern __shared__ float lvar[];
    #define BARRIER __syncthreads();
#endif
