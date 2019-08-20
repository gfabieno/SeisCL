/*Macros for writing kernels compatible with CUDA and OpenCL */

#ifdef __OPENCL_VERSION__
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #define FUNDEF __kernel
    #define LFUNDEF
    #define GLOBARG __global
    #define LOCARG __local float *lvar
    #define LOCARG2 __local __prec2 *lvar2
    #define LOCID __local
    #define LOCDEF
    #define BARRIER barrier(CLK_LOCAL_MEM_FENCE);
#else
    #define FUNDEF extern "C" __global__
    #define LFUNDEF __device__ __inline__
    #define GLOBARG
    #define LOCARG float *nullarg
    #define LOCARG2 __prec2 *nullarg
    #define LOCDEF extern __shared__ float lvar[];
    #define LOCID
    #define BARRIER __syncthreads();
#endif
