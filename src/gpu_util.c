/*------------------------------------------------------------------------
 * Copyright (C) 2016 For the list of authors, see file AUTHORS.
 *
 * This file is part of SeisCL.
 *
 * SeisCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.0 of the License only.
 *
 * SeisCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SeisCL. See file COPYING and/or
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 --------------------------------------------------------------------------*/

/* This is a collection of utility functions for OpenCL */

#include "F.h"

cl_int GetPlatformID( cl_device_type * pref_device_type, cl_device_type * device_type, cl_platform_id* clsel_plat_id, cl_uint  *outnum_devices, int n_no_use_GPUs, int * no_use_GPUs)
{
/* Find all platforms available, and select the first with the desired device type */
    
    char chBuffer[1024];
    cl_uint num_platforms;
    cl_platform_id* clPlatformIDs=NULL;
    cl_int cl_err=0;
    cl_uint num_devices=0;
    int i,j,k;
    cl_int device_found;
    
    
    if (*pref_device_type!=CL_DEVICE_TYPE_GPU &&
        *pref_device_type!=CL_DEVICE_TYPE_CPU &&
        *pref_device_type!=CL_DEVICE_TYPE_ACCELERATOR ){
        fprintf(stderr," Warning: invalid prefered device type, defaulting to GPU\n");
        *pref_device_type=CL_DEVICE_TYPE_GPU;
        
    }
    
    // Get OpenCL platform count
    cl_err = clGetPlatformIDs (0, NULL, &num_platforms);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    else{
        if(num_platforms == 0){
            fprintf(stderr,"No OpenCL platform found!\n\n");
            return 1;
        }
        else{
            fprintf(stderr,"Found %u OpenCL platforms:\n", num_platforms);
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL){
                fprintf(stderr,"Failed to allocate memory for cl_platform ID's!\n\n");
                return 1;
            }
                
            // get platform info for each platform and the platform containing the prefred device type if found
            cl_err = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
            if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
            if (!cl_err){
                for(i = 0; i < num_platforms; ++i){
                    device_found = clGetDeviceIDs(clPlatformIDs[i], *pref_device_type, 0, NULL, &num_devices);

                    if(device_found == CL_SUCCESS){
                        if(num_devices>0){
                            cl_err = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                            if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
                            fprintf(stderr,"Connection to platform %d: %s\n", i, chBuffer);
                            *clsel_plat_id = clPlatformIDs[i];
                            *device_type=*pref_device_type;
                            
                            for (j=0;j<num_devices;j++){
                                for (k=0;k<n_no_use_GPUs;k++){
                                    if (no_use_GPUs[k]==j)
                                    num_devices-=1;
                                }
                            }
                            
                            if (num_devices<1){
                                printf ("no allowed devices could be found\n");
                                return 1;
                            }
                            *outnum_devices=num_devices;
                        }
                    }
                }
                
                // default to the first platform with a GPU otherwise
                if(*clsel_plat_id == NULL){
                    for(i = 0; i < num_platforms; ++i){
                        device_found = clGetDeviceIDs(clPlatformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
                        if(device_found == CL_SUCCESS){
                            if(num_devices>0){
                                cl_err = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                                fprintf(stderr,"Connection to platform %d: %s\n", i, chBuffer);
                                *clsel_plat_id = clPlatformIDs[i];
                                *device_type=CL_DEVICE_TYPE_GPU;
                                *outnum_devices=num_devices;
                            }
                        }
                    }
                }
                
                // default to the first platform with an accelerator otherwise
                if(*clsel_plat_id == NULL){
                    for(i = 0; i < num_platforms; ++i){
                        device_found = clGetDeviceIDs(clPlatformIDs[i], CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &num_devices);
                        if(device_found == CL_SUCCESS){
                            if(num_devices>0){
                                cl_err = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                                fprintf(stderr,"Connection to platform %d: %s\n", i, chBuffer);
                                *clsel_plat_id = clPlatformIDs[i];
                                *device_type=CL_DEVICE_TYPE_ACCELERATOR;
                                *outnum_devices=num_devices;
                            }
                        }
                    }
                }
                
                // default to the first platform with a CPU otherwise
                if(*clsel_plat_id == NULL){
                    for(i = 0; i < num_platforms; ++i){
                        device_found = clGetDeviceIDs(clPlatformIDs[i], CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices);
                        if(device_found == CL_SUCCESS){
                            if(num_devices>0){
                                cl_err = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                                fprintf(stderr,"Connection to platform %d: %s\n", i, chBuffer);
                                *clsel_plat_id = clPlatformIDs[i];
                                *device_type=CL_DEVICE_TYPE_CPU;
                                *outnum_devices=num_devices;
                            }
                        }
                    }
                }
            }
                
            if(*clsel_plat_id == NULL)
            {
                fprintf(stderr,"Error: a platform containing a supported device could not be found\n\n");
                return 1;
            }

            free(clPlatformIDs);
        }
    }
    
    return cl_err;
}

cl_int connect_allgpus(struct varcl ** vcl, cl_context *incontext, cl_device_type * device_type, cl_platform_id* clsel_plat_id, int n_no_use_GPUs, int * no_use_GPUs, int nmax_dev)
{
    
    /*Routine to connect all found computing devices, create the context and the command queues*/

    cl_int cl_err = 0;
    cl_uint num_devices=0;
    cl_uint num_allowed_devices=0;
    cl_device_id *devices=NULL;
    int *allowed_devices=NULL;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    int i,j;
    if (nmax_dev<1){
        fprintf(stderr,"Warning, maximum number of devices too small, default to 1\n");
        nmax_dev=1;
    }
    
    // Find the number of prefered devices
    cl_err = clGetDeviceIDs(*clsel_plat_id, *device_type, 0, NULL, &num_devices);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));

    if (!cl_err && num_devices>0){
        
        devices=malloc(sizeof(cl_device_id)*num_devices);
        if (*device_type==CL_DEVICE_TYPE_GPU)
            fprintf(stderr,"Found %d GPU, ", num_devices);
        else if (*device_type==CL_DEVICE_TYPE_ACCELERATOR)
            fprintf(stderr,"Found %d Accelerator, ", num_devices);
        else if (*device_type==CL_DEVICE_TYPE_CPU)
            fprintf(stderr,"Found %d CPU, ", num_devices);
        cl_err = clGetDeviceIDs(*clsel_plat_id, *device_type, num_devices, devices, NULL);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
        
        if (!cl_err){
            num_allowed_devices=num_devices;
            for (i=0;i<num_devices;i++){
                for (j=0;j<n_no_use_GPUs;j++){
                    if (no_use_GPUs[j]==i)
                    num_allowed_devices-=1;
                }
            }
            if (num_devices<1){
                printf ("no allowed devices could be found");
                return 1;
            }
            
            allowed_devices=malloc(sizeof(int)*num_allowed_devices);
            if (num_allowed_devices==num_devices){
                for (i=0;i<num_devices;i++){
                    allowed_devices[i]=i;
                }
            }
            else{
                int n=0;
                for (i=0;i<num_devices;i++){
                    for (j=0;j<n_no_use_GPUs;j++){
                        if (no_use_GPUs[j]!=i){
                            allowed_devices[n]=i;
                            n+=1;
                        }
                    }
                }
            }
            
            num_allowed_devices=num_allowed_devices>nmax_dev ? nmax_dev : num_allowed_devices;
            fprintf(stderr,"connecting to  %d devices:\n", num_allowed_devices);
            
            for (i=0;i<num_allowed_devices;i++){
                // Get some information about the returned devices
                cl_err = clGetDeviceInfo(devices[allowed_devices[i]], CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
                cl_err = clGetDeviceInfo(devices[allowed_devices[i]], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
                fprintf(stderr,"-Device %d: %s %s\n", i, vendor_name, device_name);
                
            }
        }
    }
    else{
        fprintf(stderr,"Failed to find the identified devices \n");
        return 1;
    }
    // Now create a context to perform our calculation with the
    // specified devices
    
    if (!cl_err) *incontext = clCreateContext(NULL, num_devices, devices, NULL, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    // And also a command queue for the context
    for (i=0;i<num_allowed_devices;i++){
        if (!cl_err) (*vcl)[i].cmd_queue = clCreateCommandQueue(*incontext, devices[allowed_devices[i]], 0 , &cl_err);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
        if (!cl_err) (*vcl)[i].cmd_queuecomm = clCreateCommandQueue(*incontext, devices[allowed_devices[i]], 0 , &cl_err);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    }

//    for (i=0;i<3;i++){
//        (*vcl)[i].cmd_queue = clCreateCommandQueue(*incontext, devices[0], 0 , &cl_err);
//        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
//        (*vcl)[i].cmd_queuecomm = clCreateCommandQueue(*incontext, devices[0], 0 , &cl_err);
//        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
//    }
    
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    if (devices) free(devices);
    if (allowed_devices) free(allowed_devices);
    
    return cl_err;
    
}

cl_int get_device_num(cl_uint * num_devices){
    
    // Find the GPU CL device
    // If there is no GPU device is CL capable, fall back to CPU
    cl_int cl_err = 0;
    cl_err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 0, NULL, num_devices);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    if (*num_devices==0){
        cl_err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 0, NULL, num_devices);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    }
    return cl_err;
    
}


cl_int create_gpu_kernel(const char * filename, cl_program *program, cl_context *context, cl_kernel *kernel, const char * program_name, const char * build_options)
{
    /* Routine to build a kernel from the source file*/
    
    cl_int cl_err = 0;
    char *program_source;
    
   

    struct stat statbuf;
	FILE *fh;

	
    if (!*program){
        fh = fopen(filename, "r");
        if (fh==NULL){
            
            fprintf(stderr,"Could not open the file: %s \n", filename);
        }
        else{
            stat(filename, &statbuf);
            program_source = (char *) malloc(statbuf.st_size + 1);
            fread(program_source, statbuf.st_size, 1, fh);
            program_source[statbuf.st_size] = '\0';
            fclose(fh);
            
            
            
            *program = clCreateProgramWithSource(*context, 1, (const char**)&program_source,NULL, &cl_err);
            if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
            
            cl_err = clBuildProgram(program[0], 0, NULL, build_options, NULL, NULL);
            if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
            
            free(program_source);
            program_source=NULL;
        }
    }
    
    // Now create the kernel "objects"
    *kernel = clCreateKernel(program[0], program_name, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
        
    
    return cl_err;
    
}


cl_int transfer_gpu_memory( cl_command_queue *inqueue, size_t buffer_size, cl_mem *var_mem, float *var)
{
    /*Routine to allocate memory buffers to the device*/
    
    cl_int cl_err = 0;
        /*Transfer memory from host to the device*/
    cl_err = clEnqueueWriteBuffer(*inqueue, *var_mem, CL_TRUE, 0, buffer_size,
                                  (void*)var, 0, NULL, NULL);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    
    return cl_err;
}

cl_int read_gpu_memory( cl_command_queue *inqueue, size_t buffer_size, cl_mem *var_mem, void *var)
{
    /*Routine to read memory buffers from the device*/
    
    cl_int cl_err = 0;
    /*Read memory from device to the host*/
    cl_err = clEnqueueReadBuffer(*inqueue, *var_mem, CL_FALSE, 0, buffer_size, var, 0, NULL, NULL);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    
    return cl_err;
}

cl_int create_gpu_memory_buffer(cl_context *incontext, size_t buffer_size, cl_mem *var_mem)
{
    /*Create the buffer on the device */
    cl_int cl_err = 0;
    *var_mem = clCreateBuffer(*incontext, CL_MEM_READ_WRITE, buffer_size, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    
    return cl_err;
    
}

cl_int create_pinned_memory_buffer(cl_context *incontext, cl_command_queue *inqueue, size_t buffer_size, cl_mem *var_mem, float **var_buf)
{
    /*Create pinned memory */
    cl_int cl_err = 0;
    *var_mem = clCreateBuffer(*incontext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    *var_buf = (float *)clEnqueueMapBuffer(*inqueue, *var_mem, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, buffer_size, 0, NULL, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    
    return cl_err;
    
}

cl_int create_gpu_memory_buffer_cst(cl_context *incontext, size_t buffer_size, cl_mem *var_mem)
{
    /*Create read only memory */
    cl_int cl_err = 0;
    *var_mem = clCreateBuffer(*incontext, CL_MEM_READ_ONLY, buffer_size, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    
    return cl_err;
    
}

cl_int create_gpu_subbuffer(cl_mem *var_mem, cl_mem *sub_mem, cl_buffer_region * region)
{
    /*Create a subbuffer */
    cl_int cl_err = 0;
    
    *sub_mem =  clCreateSubBuffer (	*var_mem, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, region, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));

    
    return cl_err;
    
}

cl_int launch_gpu_kernel( cl_command_queue *inqueue, cl_kernel *kernel, int ndim, size_t global_work_size[2], size_t local_work_size[2], int numevent, cl_event * waitlist, cl_event * eventout){
    
    /*Launch a kernel a check for errors */
    cl_int cl_err = 0;
    cl_err = clEnqueueNDRangeKernel(*inqueue, *kernel, ndim, NULL, global_work_size, local_work_size, numevent, waitlist, eventout);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(cl_err));
    
    return cl_err;
    
}


char *gpu_error_code(cl_int err)
{
    /* Routine that return the OpenCL error in a string format */
    
    switch (err) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        default: return "Unknown OpenCl error code";
    }
}
