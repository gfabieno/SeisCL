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
    cl_uint NUM_DEVICES=0;
    int i,j,k;
    cl_int device_found;
    
    
    if (*pref_device_type!=CL_DEVICE_TYPE_GPU &&
        *pref_device_type!=CL_DEVICE_TYPE_CPU &&
        *pref_device_type!=CL_DEVICE_TYPE_ACCELERATOR ){
        fprintf(stdout," Warning: invalid prefered device type, defaulting to GPU\n");
        *pref_device_type=CL_DEVICE_TYPE_GPU;
        
    }
    
    // Get OpenCL platform count
    cl_err = clGetPlatformIDs (0, NULL, &num_platforms);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    else{
        if(num_platforms == 0){
            fprintf(stderr,"No OpenCL platform found!\n\n");
            return 1;
        }
        else{
            fprintf(stdout,"Found %u OpenCL platforms:\n", num_platforms);
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL){
                fprintf(stderr,"Failed to allocate memory for cl_platform ID's!\n\n");
                return 1;
            }
                
            // get platform info for each platform and the platform containing the prefred device type if found
            cl_err = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
            if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
            if (!cl_err){
                for(i = 0; i < num_platforms; ++i){
                    device_found = clGetDeviceIDs(clPlatformIDs[i], *pref_device_type, 0, NULL, &NUM_DEVICES);

                    if(device_found == CL_SUCCESS){
                        if(NUM_DEVICES>0){
                            cl_err = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                            if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
                            fprintf(stdout,"Connection to platform %d: %s\n", i, chBuffer);
                            *clsel_plat_id = clPlatformIDs[i];
                            *device_type=*pref_device_type;
                            
                            for (j=0;j<NUM_DEVICES;j++){
                                for (k=0;k<n_no_use_GPUs;k++){
                                    if (no_use_GPUs[k]==j)
                                    NUM_DEVICES-=1;
                                }
                            }
                            
                            if (NUM_DEVICES<1){
                                printf ("no allowed devices could be found\n");
                                return 1;
                            }
                            *outnum_devices=NUM_DEVICES;
                        }
                    }
                }
                
                // default to the first platform with a GPU otherwise
                if(*clsel_plat_id == NULL){
                    for(i = 0; i < num_platforms; ++i){
                        device_found = clGetDeviceIDs(clPlatformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &NUM_DEVICES);
                        if(device_found == CL_SUCCESS){
                            if(NUM_DEVICES>0){
                                cl_err = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                                fprintf(stdout,"Connection to platform %d: %s\n", i, chBuffer);
                                *clsel_plat_id = clPlatformIDs[i];
                                *device_type=CL_DEVICE_TYPE_GPU;
                                *outnum_devices=NUM_DEVICES;
                            }
                        }
                    }
                }
                
                // default to the first platform with an accelerator otherwise
                if(*clsel_plat_id == NULL){
                    for(i = 0; i < num_platforms; ++i){
                        device_found = clGetDeviceIDs(clPlatformIDs[i], CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &NUM_DEVICES);
                        if(device_found == CL_SUCCESS){
                            if(NUM_DEVICES>0){
                                cl_err = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                                fprintf(stdout,"Connection to platform %d: %s\n", i, chBuffer);
                                *clsel_plat_id = clPlatformIDs[i];
                                *device_type=CL_DEVICE_TYPE_ACCELERATOR;
                                *outnum_devices=NUM_DEVICES;
                            }
                        }
                    }
                }
                
                // default to the first platform with a CPU otherwise
                if(*clsel_plat_id == NULL){
                    for(i = 0; i < num_platforms; ++i){
                        device_found = clGetDeviceIDs(clPlatformIDs[i], CL_DEVICE_TYPE_CPU, 0, NULL, &NUM_DEVICES);
                        if(device_found == CL_SUCCESS){
                            if(NUM_DEVICES>0){
                                cl_err = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                                fprintf(stdout,"Connection to platform %d: %s\n", i, chBuffer);
                                *clsel_plat_id = clPlatformIDs[i];
                                *device_type=CL_DEVICE_TYPE_CPU;
                                *outnum_devices=NUM_DEVICES;
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
    cl_uint NUM_DEVICES=0;
    cl_uint num_allowed_devices=0;
    cl_device_id *devices=NULL;
    int *allowed_devices=NULL;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    int i,j;
    if (nmax_dev<1){
        fprintf(stdout,"Warning, maximum number of devices too small, default to 1\n");
        nmax_dev=1;
    }
    
    // Find the number of prefered devices
    cl_err = clGetDeviceIDs(*clsel_plat_id, *device_type, 0, NULL, &NUM_DEVICES);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));

    if (!cl_err && NUM_DEVICES>0){
        
        devices=malloc(sizeof(cl_device_id)*NUM_DEVICES);
        if (*device_type==CL_DEVICE_TYPE_GPU)
            fprintf(stdout,"Found %d GPU, ", NUM_DEVICES);
        else if (*device_type==CL_DEVICE_TYPE_ACCELERATOR)
            fprintf(stdout,"Found %d Accelerator, ", NUM_DEVICES);
        else if (*device_type==CL_DEVICE_TYPE_CPU)
            fprintf(stdout,"Found %d CPU, ", NUM_DEVICES);
        cl_err = clGetDeviceIDs(*clsel_plat_id, *device_type, NUM_DEVICES, devices, NULL);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
        
        if (!cl_err){
            num_allowed_devices=NUM_DEVICES;
            for (i=0;i<NUM_DEVICES;i++){
                for (j=0;j<n_no_use_GPUs;j++){
                    if (no_use_GPUs[j]==i)
                    num_allowed_devices-=1;
                }
            }
            if (NUM_DEVICES<1){
                printf ("no allowed devices could be found");
                return 1;
            }
            
            allowed_devices=malloc(sizeof(int)*num_allowed_devices);
            if (num_allowed_devices==NUM_DEVICES){
                for (i=0;i<NUM_DEVICES;i++){
                    allowed_devices[i]=i;
                }
            }
            else{
                int n=0;
                for (i=0;i<NUM_DEVICES;i++){
                    for (j=0;j<n_no_use_GPUs;j++){
                        if (no_use_GPUs[j]!=i){
                            allowed_devices[n]=i;
                            n+=1;
                        }
                    }
                }
            }
            
            num_allowed_devices=num_allowed_devices>nmax_dev ? nmax_dev : num_allowed_devices;
            fprintf(stdout,"connecting to  %d devices:\n", num_allowed_devices);
            
            for (i=0;i<num_allowed_devices;i++){
                // Get some information about the returned devices
                cl_err = clGetDeviceInfo(devices[allowed_devices[i]], CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
                cl_err = clGetDeviceInfo(devices[allowed_devices[i]], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
                fprintf(stdout,"-Device %d: %s %s\n", i, vendor_name, device_name);
                
            }
        }
    }
    else{
        fprintf(stderr,"Failed to find the identified devices \n");
        return 1;
    }
    // Now create a context to perform our calculation with the
    // specified devices
    
    if (!cl_err) *incontext = clCreateContext(NULL, NUM_DEVICES, devices, NULL, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    // And also a command queue for the context
    for (i=0;i<num_allowed_devices;i++){
        if (!cl_err) (*vcl)[i].queue = clCreateCommandQueue(*incontext, devices[allowed_devices[i]], 0 , &cl_err);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
        if (!cl_err) (*vcl)[i].queuecomm = clCreateCommandQueue(*incontext, devices[allowed_devices[i]], 0 , &cl_err);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    }

//    for (i=0;i<3;i++){
//        (*vcl)[i].queue = clCreateCommandQueue(*incontext, devices[0], 0 , &cl_err);
//        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
//        (*vcl)[i].queuecomm = clCreateCommandQueue(*incontext, devices[0], 0 , &cl_err);
//        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
//    }
    
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    if (devices) free(devices);
    if (allowed_devices) free(allowed_devices);
    
    return cl_err;
    
}

cl_int get_device_num(cl_uint * NUM_DEVICES){
    
    // Find the GPU CL device
    // If there is no GPU device is CL capable, fall back to CPU
    cl_int cl_err = 0;
    cl_err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 0, NULL, NUM_DEVICES);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    if (*NUM_DEVICES==0){
        cl_err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 0, NULL, NUM_DEVICES);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
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
            if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
            
            cl_err = clBuildProgram(program[0], 0, NULL, build_options, NULL, NULL);
            if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
            
            free(program_source);
            program_source=NULL;
        }
    }
    
    // Now create the kernel "objects"
    *kernel = clCreateKernel(program[0], program_name, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
        
    
    return cl_err;
    
}

cl_int create_gpu_kernel_from_string(const char *program_source, cl_program *program, cl_context *context, cl_kernel *kernel, const char * program_name, const char * build_options)
{
    /* Routine to build a kernel from the source file contained in a c string*/
    
    cl_int cl_err = 0;
    
    if (!*program){
        *program = clCreateProgramWithSource(*context, 1, &program_source,NULL, &cl_err);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
        
        cl_err = clBuildProgram(*program, 0, NULL, build_options, NULL, NULL);
        if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    }
    // Now create the kernel "objects"
    *kernel = clCreateKernel(*program, program_name, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    
    
    return cl_err;
    
}

cl_int clbuf_send( cl_command_queue *inqueue, struct clbuf * buf)
{
    /*Routine to allocate memory buffers to the device*/
    
    cl_int cl_err = 0;
    /*Transfer memory from host to the device*/
    cl_err = clEnqueueWriteBuffer(*inqueue, buf->mem,
                                  CL_TRUE,
                                  0,
                                  buf->size,
                                  (void*)buf->host,
                                  0,
                                  NULL,
                                  NULL);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    
    return cl_err;
}

cl_int clbuf_read( cl_command_queue *inqueue, struct clbuf * buf)
{
    /*Routine to read memory buffers from the device*/
    
    cl_int cl_err = 0;
    cl_event * event=NULL;
    if (buf->outevent)
        event=&buf->event;

    /*Read memory from device to the host*/
    cl_err = clEnqueueReadBuffer(*inqueue,
                                 buf->mem,
                                 CL_FALSE,
                                 0,
                                 buf->size,
                                 buf->host,
                                 buf->nwait,
                                 buf->waitlist,
                                 event);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    
    return cl_err;
}

cl_int clbuf_readpin( cl_command_queue *inqueue,
                     struct clbuf * buf,
                     struct clbuf * bufpin)
{
    /*Routine to read memory buffers from the device*/
    
    cl_int cl_err = 0;
    cl_event * event=NULL;
    if (buf->outevent)
        event=&buf->event;
    
    /*Read memory from device to the host*/
    cl_err = clEnqueueReadBuffer(*inqueue,
                                 buf->mem,
                                 CL_FALSE,
                                 0,
                                 buf->size,
                                 bufpin->host,
                                 buf->nwait,
                                 buf->waitlist,
                                 event);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    
    return cl_err;
}

cl_int clbuf_create(cl_context *incontext, struct clbuf * buf)
{
    /*Create the buffer on the device */
    cl_int cl_err = 0;
    (*buf).mem = clCreateBuffer(*incontext, CL_MEM_READ_WRITE, (*buf).size, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    
    return cl_err;
    
}

cl_int clbuf_create_pin(cl_context *incontext, cl_command_queue *inqueue,
                                                             struct clbuf * buf)
{
    /*Create pinned memory */
    cl_int cl_err = 0;
    (*buf).mem = clCreateBuffer(*incontext,
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                (*buf).size,
                                NULL,
                                &cl_err);

    (*buf).host = (float*)clEnqueueMapBuffer(*inqueue,
                                             (*buf).mem,
                                             CL_TRUE,
                                             CL_MAP_WRITE | CL_MAP_READ,
                                             0,
                                             (*buf).size,
                                             0,
                                             NULL,
                                             NULL,
                                             &cl_err);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    
    return cl_err;
    
}

cl_int clbuf_create_cst(cl_context *incontext, struct clbuf * buf)
{
    /*Create read only memory */
    cl_int cl_err = 0;
    (*buf).mem = clCreateBuffer(*incontext, CL_MEM_READ_ONLY, (*buf).size, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    
    return cl_err;
    
}


cl_int prog_launch( cl_command_queue *inqueue, struct clprogram * prog){
    
    /*Launch a kernel and check for errors */
    cl_int cl_err = 0;
    cl_event * event=NULL;
    size_t * lsize;
    if (prog->outevent)
        event=&prog->event;
    if (prog->lsize[0]!=0)
        lsize=prog->lsize;
    
    cl_err = clEnqueueNDRangeKernel(*inqueue, prog->kernel,
                                              prog->NDIM, NULL,
                                              prog->gsize,
                                              lsize,
                                              prog->nwait,
                                              prog->waitlist,
                                              event);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(cl_err));
    
    return cl_err;
    
}


char *cl_err_code(cl_int err)
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
