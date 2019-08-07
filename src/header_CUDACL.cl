/*Macros for writing kernels compatible with CUDA and OpenCL */

#ifdef __OPENCL_VERSION__
    #define FUNDEF __kernel
    #define LFUNDEF
    #define GLOBARG __global
    #define LOCARG __local float *lvar
    #define LOCARG2 __local __prec2 *lvar2
    #define LOCDEF
    #define LOCDEF2
    #define BARRIER barrier(CLK_LOCAL_MEM_FENCE);
#else
    #define FUNDEF extern "C" __global__
    #define LFUNDEF extern "C" __device__
    #define GLOBARG
    #define LOCARG float *nullarg
    #define LOCARG2 __prec2 *nullarg
    #define LOCDEF extern __shared__ float lvar[];
    #define LOCDEF2 extern __shared__ __prec2 lvar2[];
    #define BARRIER __syncthreads();
#endif
