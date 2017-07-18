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



int checkexists(hid_t file_id, const char * invar){
    
    int state=0;
    
    if (1!=H5Lexists( file_id, invar, H5P_DEFAULT))          {state=1;fprintf(stderr, "Variable %s is not defined\n",invar);}
    
    return state;
}


int checkscalar(hid_t file_id, const char * invar){
    
    int state=0;
    hid_t dataset_id=0, dataspace_id=0;
    int rank;
    hsize_t dims2[2];
    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    dataspace_id = H5Dget_space( dataset_id );
    rank=H5Sget_simple_extent_ndims(dataspace_id);
    H5Sget_simple_extent_dims(dataspace_id, dims2, NULL );
    if (rank!=2)                    {state=1;fprintf(stderr, "Variable %s must be of rank 2\n",invar);};
    if (dims2[0]!=1 || dims2[1]!=1) {state=1;fprintf(stderr, "Variable %s must be of size 1x1 (scalar)\n", &invar[1]);};
    
    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}

int read_seis(hid_t file_id, hid_t memtype, const char * invar, float * varptr, float * tracesid, int ntraces, int NT){
    
    int state=0, ii,n;
    hid_t dataset_id=0, dataspace=0, memspace=0;
    hsize_t OFFSET[2], count[2];
    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    
    dataspace = H5Dget_space(dataset_id);

    
    OFFSET[1]=0;
    OFFSET[0]=0;
    count[0]=1;
    count[1]=NT;

    OFFSET[0]=(hsize_t)tracesid[4]-1;
    n=0;
    for (ii=1;ii<ntraces;ii++){
        
        if (((hsize_t)tracesid[4+8*ii]-1)==OFFSET[0]+count[0]){
            count[0]+=1;
        }
        else{

        memspace = H5Screate_simple(2,count,NULL);
        state = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, OFFSET, NULL, count, NULL);
        if (0>H5Dread(dataset_id, memtype, memspace, dataspace, H5P_DEFAULT, &varptr[n*NT])) {state=1;fprintf(stderr, "Cannot read variable %s\n", &invar[1]);};
        
        n=ii;
        count[0]=1;
        OFFSET[0]=(hsize_t)tracesid[4+8*ii]-1;
        }
    }
    memspace = H5Screate_simple(2,count,NULL);
    state = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, OFFSET, NULL, count, NULL);
    if (0>H5Dread(dataset_id, memtype, memspace, dataspace, H5P_DEFAULT, &varptr[n*NT])) {state=1;fprintf(stderr, "Cannot read variable %s\n", &invar[1]);};

    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}



int readvar(hid_t file_id, hid_t memtype, const char * invar, void * varptr){
    
    int state=0;
    hid_t dataset_id=0;
    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    if (0>H5Dread(dataset_id, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, varptr)) {state=1;fprintf(stderr, "Cannot read variable %s\n", &invar[1]);};
    
    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}



int getdimmat(hid_t file_id, const char * invar, int NDIM, hsize_t * dims1){
    
    int state=0;
    hid_t dataset_id=0, dataspace_id=0;
    int rank;

    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    dataspace_id = H5Dget_space( dataset_id );
    rank=H5Sget_simple_extent_ndims(dataspace_id);
    if (rank!=NDIM) {state=1;fprintf(stderr, "Variable %s must have %d dimensions\n",invar, NDIM);};
    if (!state) H5Sget_simple_extent_dims(dataspace_id, dims1, NULL );
    
    if (dataspace_id) H5Sclose(dataspace_id);
    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}

int getNDIM(hid_t file_id, const char * invar, int *NDIM){
    
    int state=0;
    hid_t dataset_id=0, dataspace_id=0;

    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    dataspace_id = H5Dget_space( dataset_id );
    *NDIM=H5Sget_simple_extent_ndims(dataspace_id);
   
    if (dataspace_id) H5Sclose(dataspace_id);
    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}


int checkmatNDIM(hid_t file_id, const char * invar, int NDIM, hsize_t * dims1){
    
    int state=0;
    hid_t dataset_id=0, dataspace_id=0;
    int rank;
    hsize_t *dims2;
    int i;
    
    dims2=(hsize_t*)malloc(sizeof(hsize_t)*NDIM);
    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    dataspace_id = H5Dget_space( dataset_id );
    rank=H5Sget_simple_extent_ndims(dataspace_id);
    if (rank!=NDIM) {state=1;fprintf(stderr, "Variable %s must have %d dimensions\n",invar, NDIM);};
    if (!state) H5Sget_simple_extent_dims(dataspace_id, dims2, NULL );
    
    
    for (i=0;i<NDIM;i++){
        if (dims2[i]!=dims1[i]) {state=1;fprintf(stderr, "Dimension %d of variable %s should be %llu, got %llu\n",i,invar, dims1[i], dims2[i]);};
    }
    
    free(dims2);
    if (dataspace_id) H5Sclose(dataspace_id);
    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}

int checkisempty(hid_t file_id, const char * invar){
    
    int state=0;
    hid_t dataset_id=0;
    hsize_t outsize;
    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    outsize = H5Dget_storage_size( dataset_id );
    if (dataset_id) H5Dclose(dataset_id);
    
    if (outsize==16)
        state=1;
    
    return state;
}

int checkmatNDIM_atleast(hid_t file_id, const char * invar, int NDIM, hsize_t * dims1){
    
    int state=0;
    hid_t dataset_id=0, dataspace_id=0;
    int rank;
    hsize_t *dims2;
    int i;
    
    dims2=(hsize_t*)malloc(sizeof(hsize_t)*NDIM);
    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    dataspace_id = H5Dget_space( dataset_id );
    rank=H5Sget_simple_extent_ndims(dataspace_id);
    if (rank!=NDIM) {state=1;fprintf(stderr, "Variable %s must have %d dimensions\n",invar, NDIM);};
    if (!state) H5Sget_simple_extent_dims(dataspace_id, dims2, NULL );
    
    
    for (i=0;i<NDIM;i++){
        if (dims2[i]<dims1[i]) state=1;
    }
    
    free(dims2);
    if (dataspace_id) H5Sclose(dataspace_id);
    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}



int readhdf5(struct filenames files, struct modcsts * m) {
    
    hid_t       file_id=0;
    hsize_t     dimsND[MAX_DIMS],dims2D[2],dimsfreqs[2];
    int         state =0, maxrecid;
    float thisid=0, tmaxf=0, tminf=0;
    int  i=0,  nsg=0, n=0, p=0;
    float *src0=NULL, *src_pos0=NULL, *rec_pos0=NULL ;

    
    /* Open the input file. */
    file_id = H5Fopen(files.csts, H5F_ACC_RDWR, H5P_DEFAULT);
    if (!state) if (file_id<0) {state=1;fprintf(stderr, "Could not open the input file csts");};
    
    
    /* Basic variables__________________________________
     __________________________________________________________________*/
    
    /* Check that basic variables exist */
    __GUARD checkexists(file_id,"/NT");
    __GUARD checkexists(file_id,"/ND");
    __GUARD checkexists(file_id,"/dt");
    __GUARD checkexists(file_id,"/dh");
    __GUARD checkexists(file_id,"/FDORDER");
    __GUARD checkexists(file_id,"/MAXRELERROR");
    __GUARD checkexists(file_id,"/src");
    __GUARD checkexists(file_id,"/src_pos");
    __GUARD checkexists(file_id,"/rec_pos");
    __GUARD checkexists(file_id,"/freesurf");
    __GUARD checkexists(file_id,"/nab");
    __GUARD checkexists(file_id,"/abs_type");
    __GUARD checkexists(file_id,"/L");
    __GUARD checkexists(file_id,"/gradout");
    __GUARD checkexists(file_id,"/gradsrcout");
    __GUARD checkexists(file_id,"/seisout");
    __GUARD checkexists(file_id,"/resout");
    __GUARD checkexists(file_id,"/rmsout");
    __GUARD checkexists(file_id,"/pref_device_type");
    __GUARD checkexists(file_id,"/nmax_dev");
    __GUARD checkexists(file_id,"/no_use_GPUs");
    __GUARD checkexists(file_id,"/MPI_NPROC_SHOT");
    
    /* Check that basic variables are in the required format */
    __GUARD checkscalar(file_id, "/NT");
    __GUARD checkscalar(file_id, "/ND");
    __GUARD checkscalar(file_id, "/dt");
    __GUARD checkscalar(file_id, "/dh");
    __GUARD checkscalar(file_id, "/FDORDER");
    __GUARD checkscalar(file_id, "/MAXRELERROR");
    __GUARD checkscalar(file_id, "/freesurf");
    __GUARD checkscalar(file_id, "/nab");
    __GUARD checkscalar(file_id, "/abs_type");
    __GUARD checkscalar(file_id, "/L");
    __GUARD checkscalar(file_id, "/gradout");
    __GUARD checkscalar(file_id, "/gradsrcout");
    __GUARD checkscalar(file_id, "/seisout");
    __GUARD checkscalar(file_id, "/resout");
    __GUARD checkscalar(file_id, "/rmsout");
    __GUARD checkscalar(file_id, "/pref_device_type");
    __GUARD checkscalar(file_id, "/nmax_dev");
    __GUARD checkscalar(file_id, "/MPI_NPROC_SHOT");
    
    /* Read basic scalar variables */
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/NT", &m->NT);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/ND", &m->ND);
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/dt", &m->dt);
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/dh", &m->dh);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/FDORDER", &m->FDORDER);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/MAXRELERROR", &m->MAXRELERROR);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/freesurf", &m->FREESURF);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/nab", &m->NAB);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/abs_type", &m->ABS_TYPE);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/L", &m->L);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/gradout", &m->GRADOUT);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/gradsrcout", &m->GRADSRCOUT);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/seisout", &m->VARSOUT);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/resout", &m->RESOUT);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/rmsout", &m->RMSOUT);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/pref_device_type", &m->pref_device_type);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/nmax_dev", &m->nmax_dev);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/MPI_NPROC_SHOT", &m->MPI_NPROC_SHOT);
    
    if (!state) {if (1==H5Lexists( file_id, "/tmax", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/tmax");
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT,   "/tmax", &tmaxf);
        m->tmax=tmaxf/m->dt;
    }
    else{
        m->tmax=m->NT;
    }}
    
    if (!state) {if (1==H5Lexists( file_id, "/tmin", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/tmin");
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT,   "/tmin", &tminf);
        m->tmin=tminf/m->dt;
    }
    else
    {
        m->tmin=0;
    }}
    
    if (m->tmin>m->tmax){
        state=1;
        fprintf(stderr, "Error: tmax<tmin\n");
    }

    
    if (!state) {if (1==H5Lexists( file_id, "/fmax", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/fmax");
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT,   "/fmax", &m->fmax);
    }}
    
    if (!state) {if (1==H5Lexists( file_id, "/fmin", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/fmin");
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT,   "/fmin", &m->fmin);
    }}
    
    if (!state) if (1==H5Lexists( file_id, "/scalerms", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/scalerms");
        __GUARD readvar(file_id, H5T_NATIVE_INT,   "/scalerms", &m->scalerms);
    }
    if (!state) if (1==H5Lexists( file_id, "/scaleshot", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/scaleshot");
        __GUARD readvar(file_id, H5T_NATIVE_INT,   "/scaleshot", &m->scaleshot);
    }
    if (!state) if (1==H5Lexists( file_id, "/scalermsnorm", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/scalermsnorm");
        __GUARD readvar(file_id, H5T_NATIVE_INT,   "/scalermsnorm", &m->scalermsnorm);
    }
    if (!state) if (1==H5Lexists( file_id, "/restype", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/restype");
        __GUARD readvar(file_id, H5T_NATIVE_INT,   "/restype", &m->restype);
    }
    if (!state) if (1==H5Lexists( file_id, "/HOUT", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/HOUT");
        __GUARD readvar(file_id, H5T_NATIVE_INT,   "/HOUT", &m->HOUT);
    }
    if (!state) if (1==H5Lexists( file_id, "/MOVOUT", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/MOVOUT");
        __GUARD readvar(file_id, H5T_NATIVE_INT,   "/MOVOUT", &m->MOVOUT);
    }

    if (H5Lexists( file_id, "/param_type", H5P_DEFAULT) ){
        __GUARD checkscalar(file_id, "/param_type");
        __GUARD readvar(file_id, H5T_NATIVE_INT, "/param_type", &m->par_type);
    }
    
    if (m->ND==3){
        m->NDIM=3;
    }
    else{
        m->NDIM=2;
    }
    
    __GUARD checkexists(file_id,"/N");
    dims2D[0]=1;dims2D[1]=m->NDIM;
    if (m->NDIM>MAX_DIMS) {state=1;fprintf(stderr, "Maximum number of dimensions is %d\n",MAX_DIMS);};
    if (!state) if ((state=checkmatNDIM(file_id, "/N",  2, dims2D)))       {state=1;fprintf(stderr, "Variable N must be 1xnumber of dimensions\n");};

    __GUARD readvar(file_id, H5T_NATIVE_INT, "/N", m->N);

    for (i=0;i<m->NDIM;i++){
        dimsND[i]=m->N[m->NDIM-1-i];
    }
    
    m->FDOH=m->FDORDER/2;
    
    /* Read baned GPUs */
    __GUARD getNDIM(file_id, "/no_use_GPUs", &n);
    
    if (!state){
        if (n>1){
            __GUARD getdimmat(file_id, "/no_use_GPUs", 2, dims2D);
            m->n_no_use_GPUs=(int)dims2D[0]*(int)dims2D[1];
            if (m->n_no_use_GPUs>0){
                GMALLOC(m->no_use_GPUs,sizeof(int)*m->n_no_use_GPUs);
                __GUARD readvar(file_id, H5T_NATIVE_INT,   "/no_use_GPUs", m->no_use_GPUs);
            }
        }
        else
            m->n_no_use_GPUs=0;
    }
    
    __GUARD checkexists(file_id,"/back_prop_type");
    __GUARD checkscalar(file_id, "/back_prop_type");
    __GUARD readvar(file_id, H5T_NATIVE_INT, "/back_prop_type", &m->BACK_PROP_TYPE);
    if (!(m->BACK_PROP_TYPE ==1 | m->BACK_PROP_TYPE ==2)) {state=1;fprintf(stderr, "bac_prop_type must be 1 or 2\n");}
    
    if (m->BACK_PROP_TYPE==2 && m->GRADOUT==1){
        __GUARD checkexists(file_id,"/gradfreqs");
        __GUARD getdimmat(file_id, "/gradfreqs", 2, dimsfreqs);
        if (dimsfreqs[0]<2 & dimsfreqs[1]>0)
            m->NFREQS=(int)dimsfreqs[1];
        else if (dimsfreqs[1]<2 & dimsfreqs[0]>0)
            m->NFREQS=(int)dimsfreqs[0];
    }
    
    if (m->restype==0){
        m->res_calc = &res_raw;
    }
    else if (m->restype==1){
        m->res_calc = &res_amp;
    }
    else{
        state=1;fprintf(stderr, "Unknown restype\n");
    }
    
    /*Absorbing boundary variables*/
    
    if (m->ABS_TYPE==2){
        __GUARD checkexists(file_id,"/abpc");
        __GUARD checkscalar(file_id, "/abpc");
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/abpc", &m->abpc);
    }
    else if (m->ABS_TYPE==1){
        __GUARD checkexists(file_id,"/VPPML");
        __GUARD checkexists(file_id,"/FPML");
        __GUARD checkexists(file_id,"/NPOWER");
        __GUARD checkexists(file_id,"/K_MAX_CPML");
        
        __GUARD checkscalar(file_id, "/VPPML");
        __GUARD checkscalar(file_id, "/FPML");
        __GUARD checkscalar(file_id, "/NPOWER");
        __GUARD checkscalar(file_id, "/K_MAX_CPML");
        
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/VPPML", &m->VPPML);
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/FPML", &m->FPML);
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/NPOWER", &m->NPOWER);
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/K_MAX_CPML", &m->K_MAX_CPML);
    }
    else if (m->ABS_TYPE==0){
    }
    else if (!state){
        state=1;fprintf(stderr, "Variable ABS_TYPE allowed values are 0, 1 and 2\n");
    }
    
    
    /* Visco-elastic modeling*/
    if (m->L>0){
        __GUARD checkexists(file_id,"/f0");
        __GUARD checkscalar(file_id, "/f0");
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/f0", &m->f0);
        
    }

    
    /* Sources and receivers variables__________________________________
     __________________________________________________________________*/
    
    /* Check that variables exist */
    __GUARD checkexists(file_id,"/src");
    __GUARD checkexists(file_id,"/src_pos");
    __GUARD checkexists(file_id,"/rec_pos");
    
    /* Check that the variables are in the required format */
    __GUARD getdimmat(file_id, "/src_pos", 2, dims2D);
    if (!state) if(dims2D[1]!=5) {state=1;fprintf(stderr, "src_pos dimension 1 must be 5\n");};
    m->src_recs.allns=(int)dims2D[0];
    dims2D[1]=m->NT;
    if (!state) if ((state=checkmatNDIM(file_id, "/src",  2, dims2D)))  {state=1;fprintf(stderr, "Variable src must be nt x number of sources\n");};
    
    
    __GUARD getdimmat(file_id, "/rec_pos", 2, dims2D);
    if (!state) if(dims2D[1]!=8) {state=1;fprintf(stderr, "rec_pos dimension 1 must be 8\n");};
    m->src_recs.allng=(int)dims2D[0];
    
    /* Assign the memory */
    GMALLOC(src_pos0,sizeof(float)*m->src_recs.allns*5);
    GMALLOC(src0,sizeof(float)*m->src_recs.allns*m->NT);
    GMALLOC(rec_pos0,sizeof(float)*m->src_recs.allng*8);
    
    /* Read variables */
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/src_pos", src_pos0);
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/src", src0);
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/rec_pos", rec_pos0);
    
    /* Determine the number of shots to simulate */
    if (!state){
        m->ns=0;
        thisid=-9999;
        for (i=0;i<m->src_recs.allns;i++){
            if (thisid<src_pos0[3+i*5]){
                thisid=src_pos0[3+i*5];
                m->ns+=1;
            }
            else if (thisid>src_pos0[3+i*5]){
                state=1;
                printf("Sources ids must be sorted in ascending order");
            }
            
        }
        
        nsg=0;
        thisid=-9999;
        maxrecid=0;
        for (i=0;i<m->src_recs.allng;i++){
            maxrecid=  (maxrecid > rec_pos0[4+i*8]) ? maxrecid : rec_pos0[4+i*8];
            if (thisid<rec_pos0[3+i*8]){
                thisid=rec_pos0[3+i*8];
                nsg+=1;
            }
            else if (thisid>rec_pos0[3+i*8]){
                state=1;
                printf("Sources ids in rec_pos must be sorted in ascending order\n");
                break;
            }
            
        }
        if (!state) if (nsg!=m->ns) {state=1;fprintf(stderr, "Number of sources ids in src_pos and rec_pos are not the same\n");};
    }
    
    /* Assign the 2D arrays in which shot and receivers variables are stored */
    GMALLOC(m->src_recs.src_pos,sizeof(float*)*m->ns);
    GMALLOC(m->src_recs.src,sizeof(float*)*m->ns);
    GMALLOC(m->src_recs.nsrc,sizeof(int)*m->ns);
    GMALLOC(m->src_recs.rec_pos,sizeof(float*)*m->ns);
    GMALLOC(m->src_recs.nrec,sizeof(int)*m->ns);
    
    if (!state){
        
        // Determine the number of sources positions per shot
        thisid=src_pos0[3];
        n=1;
        p=0;
        for (i=1;i<m->src_recs.allns;i++){
            if (thisid==src_pos0[3+i*5]){
                n+=1;
            }
            else{
                m->src_recs.nsrc[p]=n;
                n=1;
                p=p+1;
                thisid=src_pos0[3+i*5];
            }
            
        }
        m->src_recs.nsrc[m->ns-1]=n;
        
        // Determine the number of receiver positions per shot
        thisid=rec_pos0[3];
        n=1;
        p=0;
        for (i=1;i<m->src_recs.allng;i++){
            if (thisid==rec_pos0[3+i*8]){
                n+=1;
            }
            else{
                m->src_recs.nrec[p]=n;
                n=1;
                p=p+1;
                thisid=rec_pos0[3+i*8];
            }
            
        }
        m->src_recs.nrec[m->ns-1]=n;
        
        //Assign the right number of shots and geophones for each shot
        m->src_recs.src_pos[0]=src_pos0;
        m->src_recs.src[0]=src0;
        m->src_recs.rec_pos[0]=rec_pos0;
        for (i=1;i<m->ns;i++){
            m->src_recs.src_pos[i]=m->src_recs.src_pos[i-1]+m->src_recs.nsrc[i-1]*5;
            m->src_recs.src[i]=m->src_recs.src[i-1]+m->src_recs.nsrc[i-1]*m->NT;
            m->src_recs.rec_pos[i]=m->src_recs.rec_pos[i-1]+m->src_recs.nrec[i-1]*8;
        }
        
        //Compute the maximum number of geophones and shots within a source id
        m->src_recs.nsmax=0;
        m->src_recs.ngmax=0;
        for (i=0;i<m->ns; i++){
            m->src_recs.nsmax = fmax(m->src_recs.nsmax, m->src_recs.nsrc[i]);
            m->src_recs.ngmax = fmax(m->src_recs.ngmax, m->src_recs.nrec[i]);
        }
    }

    //Assign parameters list depending on which case of modeling is desired
    __GUARD assign_modeling_case(m);
    
    //Read active constants
    for (i=0;i<m->ncsts;i++){
        if (m->csts[i].to_read  && m->csts[i].active==1){
            __GUARD checkexists(file_id,m->csts[i].to_read);
            GMALLOC(m->csts[i].gl_cst,sizeof(float)*m->csts[i].num_ele);
            __GUARD readvar(file_id, H5T_NATIVE_FLOAT, m->csts[i].to_read, m->csts[i].gl_cst);
        }
    }


    
    
    if (file_id>=0) H5Fclose(file_id);
    file_id=0;
    
    
    /* Model file__________________________________
     __________________________________________________________________*/
    


    

    /* Open the model file. */
    if (!state) file_id = H5Fopen(files.model, H5F_ACC_RDWR, H5P_DEFAULT);
    if (!state) if (file_id<0) {state=1;fprintf(stderr, "Could not open the input file model");};
    
    /* Read parameters. */
    for (i=0;i<m->npars;i++){
        if (m->pars[i].to_read){
            __GUARD checkexists(file_id,m->pars[i].to_read);
            GMALLOC(m->pars[i].gl_par,sizeof(float)*m->pars[i].num_ele);
            if (!state) if ((state=checkmatNDIM(file_id, m->pars[i].to_read, m->NDIM, dimsND)))  {state=1;fprintf(stderr, "Variable %s must have dimensions given by N\n",m->pars[i].to_read);};
            __GUARD readvar(file_id, H5T_NATIVE_FLOAT, m->pars[i].to_read, m->pars[i].gl_par);
        }
    }
    
    /* Close files. */
    if (file_id>=0) H5Fclose(file_id);
    file_id=0;
    

    /* Data in file__________________________________
     __________________________________________________________________*/
    
    /* Open the data file. */
    if (m->RMSOUT==1 || m->RESOUT || m->GRADOUT==1){
        file_id = H5Fopen(files.din, H5F_ACC_RDWR, H5P_DEFAULT);
        if (!state) if (file_id<0 && m->GRADOUT) {state=1;fprintf(stderr, "Could not open the input file for data_in\n");};
        
        int anyout=0;
        if (file_id>=0){
            
            
            dims2D[0]=m->src_recs.allng;dims2D[1]=m->NT;
            for (i=0;i<m->nvars;i++){
                if (1==H5Lexists( file_id, m->vars[i].name, H5P_DEFAULT)){
                    m->vars[i].to_output=1;
                    if (!state) if ((state=checkmatNDIM_atleast(file_id, m->vars[i].name,  2, dims2D)))   {state=1;fprintf(stderr, "Variable %s must be nt x number of recording stations\n",m->vars[i].name );};
                    alloc_seismo(&m->vars[i].gl_varin, m);
                    if (m->RESOUT){
                        alloc_seismo(&m->vars[i].gl_var_res, m);
                    }
                    __GUARD read_seis(file_id, H5T_NATIVE_FLOAT, m->vars[i].name, m->vars[i].gl_varin[0], m->src_recs.rec_pos[0], m->src_recs.allng, m->NT);
                    anyout=1;
                }
            }
            for (i=0;i<m->ntvars;i++){
                if (1==H5Lexists( file_id, m->trans_vars[i].name, H5P_DEFAULT)){
                    m->trans_vars[i].to_output=1;
                    if (!state) if ((state=checkmatNDIM_atleast(file_id, m->trans_vars[i].name,  2, dims2D)))   {state=1;fprintf(stderr, "Variable %s must be nt x number of recording stations\n",m->trans_vars[i].name );};
                    alloc_seismo(&m->trans_vars[i].gl_varin, m);
                    if (m->RESOUT){
                        alloc_seismo(&m->trans_vars[i].gl_var_res, m);
                    }
                    __GUARD read_seis(file_id, H5T_NATIVE_FLOAT, m->trans_vars[i].name, m->trans_vars[i].gl_varin[0], m->src_recs.rec_pos[0], m->src_recs.allng, m->NT);
                    anyout=1;
                }
            }
            
            H5Fclose(file_id);
            
        }

        if (!anyout){
            state=1;
            fprintf(stderr, "Cannot output gradient, rms or residuals without reference data\n");
        }
        
    }
    
    if (state && m->MPI_INIT==1) MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    return state;
    
}




