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
#include "F.h"


int main(int argc, char **argv) {
    
    int state=0;

    struct modcsts m={};
    struct varcl *vcl=NULL;

    
    int i;
    double time1=0.0, time2=0.0, time3=0.0, time4=0.0, time5=0.0;

    
    //Input files name is an argument
    struct filenames file={};
    const char * filein;
    const char * filedata;
    if (argc>1){
        filein=argv[1];
    }
    else {
        filein="SeisCL";
    }
   
    snprintf(file.model, sizeof(file.model), "%s%s", filein, "_model.mat");
    snprintf(file.csts, sizeof(file.csts), "%s%s", filein, "_csts.mat");
    snprintf(file.dout, sizeof(file.dout), "%s%s", filein, "_dout.mat");
    snprintf(file.gout, sizeof(file.gout), "%s%s", filein, "_gout.mat");
    snprintf(file.RMSOUT, sizeof(file.RMSOUT), "%s%s", filein, "_rms.mat");
    snprintf(file.MOVOUT, sizeof(file.MOVOUT), "%s%s", filein, "_movie.mat");
    
    if (argc>2){
        filedata=argv[2];
        snprintf(file.din, sizeof(file.din), "%s", filedata);
    }
    else {
        filedata=filein;
        snprintf(file.din, sizeof(file.din), "%s%s", filedata, "_din.mat");
    }
    
    
    
    /* Initialize MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &m.NP);
    MPI_Comm_rank(MPI_COMM_WORLD, &m.MYID);
    MPI_Initialized(&m.MPI_INIT);
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    fprintf(stdout,"MPI connected to processor %s\n",processor_name);
    
    //if (m.MYID== 0 ) sleep(600);
    
    // Root process reads the input files
    time1=MPI_Wtime();
    if (m.MYID==0){
        __GUARD readhdf5(file, &m);
    }
    time2=MPI_Wtime();
    
    // Initiate and transfer data on all process
    __GUARD Init_MPI(&m);
    __GUARD Init_cst(&m);
    __GUARD Init_model(&m);
    
    time3=MPI_Wtime();
    
    __GUARD Init_OpenCL(&m, &vcl);
//
//    
//
//    
//    time4=MPI_Wtime();
//    
//    
//    // Main part, where seismic modeling occurs
//    __GUARD time_stepping(&m, &vcl, &mloc);
//    
//    time5=MPI_Wtime();
//    
//    
//    //Output time for each part of the program
//    {
//        double * times=NULL;
//        
//        if (m.MYID==0){
//            times=malloc(sizeof(double)*5*m.NP);
//        }
//        
//        double t1=time2-time1;
//        double t2=time3-time2;
//        double t3=time4-time3;
//        double t4=(time5-time4)/(m.smax-m.smin)/m.NT;
//        double t5=(time5-time1);
//        
//        if (!state) MPI_Gather(&t1, 1, MPI_DOUBLE , &times[0]     , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//        if (!state) MPI_Gather(&t2, 1, MPI_DOUBLE , &times[  m.NP], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//        if (!state) MPI_Gather(&t3, 1, MPI_DOUBLE , &times[2*m.NP], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//        if (!state) MPI_Gather(&t4, 1, MPI_DOUBLE , &times[3*m.NP], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//        if (!state) MPI_Gather(&t5, 1, MPI_DOUBLE , &times[4*m.NP], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//        
//        if (m.MYID==0){
//            fprintf(stdout,"Run time for each process:\n");
//            fprintf(stdout,"process read_variables init_model init_opencl time_per_shot_per_step total_time_per_device\n");
//            
//            for (i=0;i<m.NP;i++){
//                
//                fprintf(stdout,"%d %f %f %f %f %f\n",i, times[i], times[i+m.NP], times[i+2*m.NP], times[i+3*m.NP], times[i+4*m.NP]) ;
//            }
//            fprintf(stdout,"Total real time of the program is: %f s\n",t5) ;
//            free(times);
//        }
//    }
//    
//    //Write ouput files
//    __GUARD Out_MPI(file, &m);
//    
//    
//    // Free the memory
//    Free_OpenCL(&m, &vcl, &mloc);
//    Free_MPI(&m);
    

    if (m.MPI_INIT==1){
        MPI_Finalize();
    }
    
    return 0;
    
}

