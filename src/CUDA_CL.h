
#define __SEISCL__

#ifdef __SEISCL__

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#define MEM cl_mem
#define QUEUE cl_command_queue
#define EVENT cl_event
#define KERNEL cl_kernel
#define MODULE int
#define PROGRAM cl_program
#define CONTEXT cl_context
#define DEVICE cl_device_id
#define DEVICE_TYPE cl_device_type
#define CUCL_SUCCESS CL_SUCCESS
#define CL_INT cl_int
#define CL_UINT cl_uint
#define CL_CHAR cl_char
#define MEMFREE clReleaseMemObject
#define QUEUEFREE clReleaseCommandQueue
#define WAITQUEUE clFinish
#define FUNDEF "__kernel "
#define GLOBARG "__global "

#else

#include <cuda.h>
#include <nvrtc.h>


#define MEM CUdeviceptr
#define QUEUE CUstream
#define EVENT CUevent
#define KERNEL CUfunction
#define MODULE CUmodule
#define PROGRAM char *
#define CONTEXT CUcontext
#define DEVICE CUdevice
#define DEVICE_TYPE int
#define CUCL_SUCCESS CUDA_SUCCESS
#define CL_INT int
#define CL_UINT uint
#define CL_CHAR char
#define MEMFREE cuMemFree
#define QUEUEFREE cuStreamDestroy
#define WAITQUEUE cuStreamSynchronize
#define FUNDEF "extern \"C\" __global__ "
#define GLOBARG " "

#endif

