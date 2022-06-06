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
#ifdef _WIN32
// Needed for gettimeofday impl.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#endif //_WIN32
#include "F.h"


/*
* Naive, partial implementation of getopt based on https://www.man7.org/linux/man-pages/man3/getopt.3.html
* Supports only single char options
* Doesn't support options taking arguments
*/
int optind = 1, optopt=0, opterr=0;
int getopt(int argc, char* const argv[], const char* optstring)
{
    static int nextchar = 1;
    if (strchr(optstring, ':')!=NULL) {
        fprintf(stderr, "Error! Option taking arguments isn't supported!");
        return -1;
    }
    char c;
    for (;;) {
        // Initial checks
        if (optind>=argc)
            return -1;
        if (argv[optind]==NULL)
            return -1;
        if (*argv[optind]!='-')
            return -1;
        if (argv[optind][1]==0)
            return -1;
        if (argv[optind][1]=='-') {
            optind++;
            return -1;
        }

        // get option character and check if in optstring
        c = argv[optind][nextchar];
        if (c=='\0') {
            nextchar = 1;
            optind++;
            continue;
        }
        char* p = strchr(optstring, c);
        // Unknown option
        if (p==NULL) {
            optopt = c;
            return '?';
        }
        nextchar++;
        break;
    }

    return c;
}


/* Windows implementation of gettimeofday
Reference: https://stackoverflow.com/a/26085827
*/
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;


int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970 
    static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    time = ((uint64_t)file_time.dwLowDateTime);
    time += ((uint64_t)file_time.dwHighDateTime)<<32;

    tp->tv_sec = (long)((time-EPOCH)/10000000L);
    tp->tv_usec = (long)(system_time.wMilliseconds*1000);
    return 0;
}
/************************************************************************************************/

double wtime(){
#ifndef __NOMPI__
    return MPI_Wtime();
#else
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double) time.tv_sec + time.tv_usec * 1e-6;
#endif
}


int main(int argc, char **argv) {

    int state = 0;

    model m = {0};
    device *dev = NULL;
    int i;
    double time1 = 0.0, time2 = 0.0, time3 = 0.0, time4 = 0.0, time5 = 0.0, time6 = 0.0;
    struct filenames file;
    const char *filein;
    const char *filedata = NULL;
    int index;
    int c;

    opterr = 0;
    while ((c = getopt(argc, argv, "hp")) != -1)
        switch (c) {
            case 'h':
                fprintf(stdout, "Option help");
                break;
            case 'p':
                m.printkernels = 1;
                break;
            case '?':
                fprintf(stderr,
                        "Unknown option character `\\x%x'.\n",
                        optopt);
                return 1;
            default:
                abort();
        }

    filein = "SeisCL";
    for (index = optind; index < argc; index++)
        switch (index - optind) {
            case 0:
                filein = argv[index];
                break;
            case 1:
                filedata = argv[index];
                break;
            default:
                fprintf(stderr,
                        "Too many default argument, expected 2");
                return 1;
        }

    /* Initialize MPI environment */
#ifndef __NOMPI__

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &m.GNP);
    MPI_Comm_rank(MPI_COMM_WORLD, &m.GID);
    MPI_Initialized(&m.MPI_INIT);
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    MPI_Comm node_comm;
    int color = 0;
    for (i = 0; i < MPI_MAX_PROCESSOR_NAME; i++) {
        color += (int) processor_name[i];
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &node_comm);
    MPI_Comm_size(node_comm, &m.LNP);
    MPI_Comm_rank(node_comm, &m.LID);
    MPI_Comm_free(&node_comm);

    if (m.GID==0){
        fprintf(stdout, "\nInitializing MPI\n");
    }
    fprintf(stdout,
            "    Process %d/%d, processor %s, node process %d/%d, pid %d\n",
            m.GID, m.GNP, processor_name, m.LID, m.LNP, getpid());
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
//    if (m.GID == 0) sleep(30);
#endif

    //Input files name is an argument
    snprintf(file.model, sizeof(file.model), "%s%s", filein, "_model.mat");
    snprintf(file.csts, sizeof(file.csts), "%s%s", filein, "_csts.mat");
    snprintf(file.dout, sizeof(file.dout), "%s%s", filein, "_dout.mat");
    snprintf(file.gout, sizeof(file.gout), "%s%s", filein, "_gout.mat");
    snprintf(file.rmsout, sizeof(file.rmsout), "%s%s", filein, "_rms.mat");
    snprintf(file.movout, sizeof(file.movout), "%s%s", filein, "_movie.mat");
    snprintf(file.res, sizeof(file.res), "%s%s", filein, "_res.mat");
    snprintf(file.checkpoint, sizeof(file.checkpoint), "%s_NP%d%s", filein,
             m.GID, "_checkpoint.mat");
    if (filedata == NULL) {
        snprintf(file.din, sizeof(file.din), "%s%s", filein, "_din.mat");
    } else {
        snprintf(file.din, sizeof(file.din), "%s", filedata);
    }
    if (m.GID == 0) {
        fprintf(stdout, "\nInput files for SeisCL: \n");
        fprintf(stdout, "    model: %s \n", file.model);
        fprintf(stdout, "    constants: %s \n", file.csts);
        fprintf(stdout, "    output data: %s \n", file.dout);
        fprintf(stdout, "    output gradient: %s \n", file.gout);
        fprintf(stdout, "    output rms: %s \n", file.rmsout);
        fprintf(stdout, "    output movie: %s \n", file.movout);
        fprintf(stdout, "    input data: %s \n", file.din);
        fprintf(stdout, "    checkpoint: %s \n\n", file.checkpoint);
    }
    /* Check if cache directory exists and create dir if not */
    
    struct stat info;
    const char *homedir;
    if ((homedir = getenv("HOME"))==NULL) {
      #ifdef _WIN32
        homedir = (char*)malloc(PATH_MAX+1);
        GetCurrentDirectory(PATH_MAX, homedir);
      #else
        homedir = (getuid())->pw_dir;
      #endif // _WIN32
        
    }
    snprintf(m.cache_dir, PATH_MAX, "%s%s", filein, "_cache");

    if (stat( m.cache_dir, &info ) != 0 ){
        #ifdef __linux__
        mkdir(m.cache_dir, 0777);
        #elif defined __APPLE__
        mkdir(m.cache_dir, 0777);
        #else
        _mkdir(m.cache_dir);
        #endif
        fprintf(stdout, "Cache directory created: %s \n\n", m.cache_dir);
    }
    else if(info.st_mode & S_IFDIR){}
    else{
        state = 1;
        fprintf(stdout, "%s already exists and is not a directory\n\n",
                m.cache_dir);
    }
    
    // Root process reads the input files
    time1=wtime();

    if (m.GID==0){
        if (!state) state = readhdf5(file, &m);
    }
    time2=wtime();

    // Initiate and transfer data on all process
    #ifndef __NOMPI__
    if (!state) state = Init_MPI(&m);
    #else
    m.NLOCALP = 1;
    m.GNP = 1;
    m.NGROUP = 1;
    #endif


    if (!state) state = Init_cst(&m);
    if (!state) state = Init_data(&m);
    if (!state) state = Init_model(&m);
    time3=wtime();

    if (m.GID == 0) {
        fprintf(stdout, "\nInitializing GPUs\n");
    }
    if (!state) state = Init_CUDA(&m, &dev);

    time4=wtime();
    // Main part, where seismic modeling occurs
    if (!state) state = time_stepping(&m, &dev, file);

    time5=wtime();
    
    #ifndef __NOMPI__
    //Reduce to process 0 all required outputs
    if (m.GNP > 1){
        if (!state) state = Out_MPI(&m);
    }
    #endif
    
    // Write the ouputs to hdf5 files
    if (m.GID==0){
        if (!state) state = writehdf5(file, &m) ;
    }
    time6=wtime();

    //Output time for each part of the program
    if (!state){
        double * times=NULL;

        if (m.GID==0){
            times=malloc(sizeof(double)*6*m.GNP);
        }

        double t1=time2-time1;
        double t2=time3-time2;
        double t3=time4-time3;
        double t4=(time5-time4);
        double t5=(time6-time5);
        double t6=(time6-time1);
        
        #ifndef __NOMPI__
        if (!state) MPI_Gather(&t1, 1, MPI_DOUBLE , &times[0]     ,
                               1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!state) MPI_Gather(&t2, 1, MPI_DOUBLE , &times[  m.GNP],
                               1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!state) MPI_Gather(&t3, 1, MPI_DOUBLE , &times[2*m.GNP],
                               1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!state) MPI_Gather(&t4, 1, MPI_DOUBLE , &times[3*m.GNP],
                               1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!state) MPI_Gather(&t5, 1, MPI_DOUBLE , &times[4*m.GNP],
                               1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!state) MPI_Gather(&t6, 1, MPI_DOUBLE , &times[5*m.GNP],
                               1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        #else
        times[0]=t1;times[1]=t2;times[2]=t3;times[3]=t4;times[4]=t5;times[5]=t6;
        #endif

        if (m.GID==0){
            fprintf(stdout,"\nRun time for each process:\n\n");
            for (i=0;i<m.GNP;i++){
                fprintf(stdout,"Process: %d\n", i);
                fprintf(stdout,"Read variables: %f\n",times[i]);
                fprintf(stdout,"Intialize model: %f\n", times[i+m.GNP]);
                fprintf(stdout,"Intialize OpenCL: %f\n", times[i+2*m.GNP]);
                fprintf(stdout,"Time for modeling: %f\n", times[i+3*m.GNP]);
                fprintf(stdout,"Outputting files: %f\n", times[i+4*m.GNP]);
                fprintf(stdout,"Total time of process: %f\n\n",times[i+5*m.GNP]);
            }

            fprintf(stdout,"Total real time of the program is: %f s\n",t6) ;
            free(times);
        }
    }

    // Free the memory
    Free_OpenCL(&m, dev);
#ifdef _WIN32
    free(homedir);
#endif // _WIN32


//    if (state){
////        sleep(300000);
//        MPI_Abort(MPI_COMM_WORLD, state);
//    }
    #ifndef __NOMPI__
    if (m.MPI_INIT==1){
        MPI_Finalize();
    }
    #endif
    
    return state;
    
}

