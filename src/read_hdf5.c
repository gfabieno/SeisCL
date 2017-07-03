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
    hsize_t offset[2], count[2];
    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    
    dataspace = H5Dget_space(dataset_id);

    
    offset[1]=0;
    offset[0]=0;
    count[0]=1;
    count[1]=NT;

    offset[0]=(hsize_t)tracesid[4]-1;
    n=0;
    for (ii=1;ii<ntraces;ii++){
        
        if (((hsize_t)tracesid[4+8*ii]-1)==offset[0]+count[0]){
            count[0]+=1;
        }
        else{

        memspace = H5Screate_simple(2,count,NULL);
        state = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        if (0>H5Dread(dataset_id, memtype, memspace, dataspace, H5P_DEFAULT, &varptr[n*NT])) {state=1;fprintf(stderr, "Cannot read variable %s\n", &invar[1]);};
        
        n=ii;
        count[0]=1;
        offset[0]=(hsize_t)tracesid[4+8*ii]-1;
        }
    }
    memspace = H5Screate_simple(2,count,NULL);
    state = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
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



int getdimmat(hid_t file_id, const char * invar, int ndim, hsize_t * dims1){
    
    int state=0;
    hid_t dataset_id=0, dataspace_id=0;
    int rank;

    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    dataspace_id = H5Dget_space( dataset_id );
    rank=H5Sget_simple_extent_ndims(dataspace_id);
    if (rank!=ndim) {state=1;fprintf(stderr, "Variable %s must have %d dimensions\n",invar, ndim);};
    if (!state) H5Sget_simple_extent_dims(dataspace_id, dims1, NULL );
    
    if (dataspace_id) H5Sclose(dataspace_id);
    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}

int getnumdim(hid_t file_id, const char * invar, int *ndim){
    
    int state=0;
    hid_t dataset_id=0, dataspace_id=0;

    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    dataspace_id = H5Dget_space( dataset_id );
    *ndim=H5Sget_simple_extent_ndims(dataspace_id);
   
    if (dataspace_id) H5Sclose(dataspace_id);
    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}


int checkmatndim(hid_t file_id, const char * invar, int ndim, hsize_t * dims1){
    
    int state=0;
    hid_t dataset_id=0, dataspace_id=0;
    int rank;
    hsize_t *dims2;
    int i;
    
    dims2=(hsize_t*)malloc(sizeof(hsize_t)*ndim);
    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    dataspace_id = H5Dget_space( dataset_id );
    rank=H5Sget_simple_extent_ndims(dataspace_id);
    if (rank!=ndim) {state=1;fprintf(stderr, "Variable %s must have %d dimensions\n",invar, ndim);};
    if (!state) H5Sget_simple_extent_dims(dataspace_id, dims2, NULL );
    
    
    for (i=0;i<ndim;i++){
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

int checkmatndim_atleast(hid_t file_id, const char * invar, int ndim, hsize_t * dims1){
    
    int state=0;
    hid_t dataset_id=0, dataspace_id=0;
    int rank;
    hsize_t *dims2;
    int i;
    
    dims2=(hsize_t*)malloc(sizeof(hsize_t)*ndim);
    
    dataset_id = H5Dopen2(file_id, invar, H5P_DEFAULT);
    dataspace_id = H5Dget_space( dataset_id );
    rank=H5Sget_simple_extent_ndims(dataspace_id);
    if (rank!=ndim) {state=1;fprintf(stderr, "Variable %s must have %d dimensions\n",invar, ndim);};
    if (!state) H5Sget_simple_extent_dims(dataspace_id, dims2, NULL );
    
    
    for (i=0;i<ndim;i++){
        if (dims2[i]<dims1[i]) state=1;
    }
    
    free(dims2);
    if (dataspace_id) H5Sclose(dataspace_id);
    if (dataset_id) H5Dclose(dataset_id);
    
    return state;
}



int readhdf5(struct filenames files, struct modcsts * m) {
    
    hid_t       file_id=0;
    hsize_t     dims3D[3],dims2D[2],dimsfreqs[2], dims0[2];
    int         state =0, maxrecid;
    float thisid=0, tmaxf=0, tminf=0;
    int  i=0, nsg=0, n=0, p=0;
    float *src0=NULL, *src_pos0=NULL, *rec_pos0=NULL, *vx00=NULL, *vy00=NULL, *vz00=NULL, *p00=NULL, *mute0=NULL, *weight0=NULL ;

    
    /* Open the input file. */
    file_id = H5Fopen(files.csts, H5F_ACC_RDWR, H5P_DEFAULT);
    if (!state) if (file_id<0) {state=1;fprintf(stderr, "Could not open the input file csts");};
    
    
    /* Basic variables__________________________________
     __________________________________________________________________*/
    
    /* Check that basic variables exist */
    __GUARD checkexists(file_id,"/NT");
    __GUARD checkexists(file_id,"/ND");
    __GUARD checkexists(file_id,"/NX");
    __GUARD checkexists(file_id,"/NY");
    __GUARD checkexists(file_id,"/NZ");
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
    __GUARD checkscalar(file_id, "/NX");
    __GUARD checkscalar(file_id, "/NY");
    __GUARD checkscalar(file_id, "/NZ");
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
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/NX", &m->NX);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/NY", &m->NY);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/NZ", &m->NZ);
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/dt", &m->dt);
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/dh", &m->dh);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/FDORDER", &m->FDORDER);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/MAXRELERROR", &m->MAXRELERROR);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/freesurf", &m->freesurf);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/nab", &m->nab);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/abs_type", &m->abs_type);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/L", &m->L);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/gradout", &m->gradout);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/gradsrcout", &m->gradsrcout);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/seisout", &m->seisout);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/resout", &m->resout);
    __GUARD readvar(file_id, H5T_NATIVE_INT,   "/rmsout", &m->rmsout);
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
    if (!state) if (1==H5Lexists( file_id, "/Hout", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/Hout");
        __GUARD readvar(file_id, H5T_NATIVE_INT,   "/Hout", &m->Hout);
    }
    if (!state) if (1==H5Lexists( file_id, "/movout", H5P_DEFAULT)){
        __GUARD checkscalar(file_id, "/movout");
        __GUARD readvar(file_id, H5T_NATIVE_INT,   "/movout", &m->movout);
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
    
    
    /* Read baned GPUs */
    __GUARD getnumdim(file_id, "/no_use_GPUs", &n);
    
    if (!state){
        if (n>1){
            __GUARD getdimmat(file_id, "/no_use_GPUs", 2, dims2D);
            m->n_no_use_GPUs=(int)dims2D[0]*(int)dims2D[1];
            if (m->n_no_use_GPUs>0){
                GMALLOC(m->no_use_GPUs,sizeof(int)*m->n_no_use_GPUs)
                __GUARD readvar(file_id, H5T_NATIVE_INT,   "/no_use_GPUs", m->no_use_GPUs);
            }
        }
        else
            m->n_no_use_GPUs=0;
    }
    

    /*Absorbing boundary variables*/
    
    if (m->abs_type==2){
        __GUARD checkexists(file_id,"/abpc");
        
        __GUARD checkscalar(file_id, "/abpc");
        
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/abpc", &m->abpc);
    }
    else if (m->abs_type==1){
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
    else if (m->abs_type==0){
    }
    else if (!state){
        state=1;fprintf(stderr, "Variable abs_type allowed values are 0, 1 and 2\n");
    }
 
    
    /* Visco-elastic modeling*/
    if (m->L>0){
        __GUARD checkexists(file_id,"/f0");
        __GUARD checkexists(file_id,"/FL");
        dims2D[0]=1;dims2D[1]=m->L;
        __GUARD checkscalar(file_id, "/f0");
        if (!state) if ((state=checkmatndim(file_id, "/FL",  2, dims2D)))       {state=1;fprintf(stderr, "Variable FL must be 1xL\n");};
        
        GMALLOC(m->FL,sizeof(float)*m->L)
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/f0", &m->f0);
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/FL", m->FL);
     
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
    m->allns=(int)dims2D[0];
    dims2D[1]=m->NT;
    if (!state) if ((state=checkmatndim(file_id, "/src",  2, dims2D)))  {state=1;fprintf(stderr, "Variable src must be nt x number of sources\n");};
    
    
    __GUARD getdimmat(file_id, "/rec_pos", 2, dims2D);
    if (!state) if(dims2D[1]!=8) {state=1;fprintf(stderr, "rec_pos dimension 1 must be 8\n");};
    m->allng=(int)dims2D[0];
    
    /* Assign the memory */
    GMALLOC(src_pos0,sizeof(float)*m->allns*5)
    GMALLOC(src0,sizeof(float)*m->allns*m->NT)
    GMALLOC(rec_pos0,sizeof(float)*m->allng*8)
    
    /* Read variables */
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/src_pos", src_pos0);
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/src", src0);
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/rec_pos", rec_pos0);
    
    /* Determine the number of shots to simulate */
    if (!state){
        m->ns=0;
        thisid=-9999;
        for (i=0;i<m->allns;i++){
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
        for (i=0;i<m->allng;i++){
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
    GMALLOC(m->src_pos,sizeof(float*)*m->ns)
    GMALLOC(m->src,sizeof(float*)*m->ns)
    GMALLOC(m->nsrc,sizeof(int)*m->ns)
    GMALLOC(m->rec_pos,sizeof(float*)*m->ns)
    GMALLOC(m->nrec,sizeof(int)*m->ns)
    
    if (!state){
        
        // Determine the number of sources positions per shot
        thisid=src_pos0[3];
        n=1;
        p=0;
        for (i=1;i<m->allns;i++){
            if (thisid==src_pos0[3+i*5]){
                n+=1;
            }
            else{
                m->nsrc[p]=n;
                n=1;
                p=p+1;
                thisid=src_pos0[3+i*5];
            }
            
        }
        m->nsrc[m->ns-1]=n;
        
        // Determine the number of receiver positions per shot
        thisid=rec_pos0[3];
        n=1;
        p=0;
        for (i=1;i<m->allng;i++){
            if (thisid==rec_pos0[3+i*8]){
                n+=1;
            }
            else{
                m->nrec[p]=n;
                n=1;
                p=p+1;
                thisid=rec_pos0[3+i*8];
            }
            
        }
        m->nrec[m->ns-1]=n;
        
        //Assign the right number of shots and geophones for each shot
        m->src_pos[0]=src_pos0;
        m->src[0]=src0;
        m->rec_pos[0]=rec_pos0;
        for (i=1;i<m->ns;i++){
            m->src_pos[i]=m->src_pos[i-1]+m->nsrc[i-1]*5;
            m->src[i]=m->src[i-1]+m->nsrc[i-1]*m->NT;
            m->rec_pos[i]=m->rec_pos[i-1]+m->nrec[i-1]*8;
        }
    }
    
    
    if (1==H5Lexists( file_id, "/mute", H5P_DEFAULT)){
        dims2D[0]=m->allng;dims2D[1]=5;
        dims0[0]=2;dims0[1]=0;
        if (checkisempty(file_id, "/mute"))
        {
            
        }
        else if (!checkmatndim(file_id, "/mute",  2, dims2D))
        {
            GMALLOC(mute0,sizeof(float)*dims2D[0]*5)
            __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/mute", mute0);
            GMALLOC(m->mute,sizeof(float*)*m->ns)
            if (!state){
                m->mute[0]=mute0;
                for (i=1;i<m->ns;i++){
                    m->mute[i]=m->mute[i-1]+m->nrec[i-1]*5;
                }
            }
  
        }
        else{
                fprintf(stderr, "Warning: Variable mute must be 5 x number of geophones\n");
        };

    }
    
    if (1==H5Lexists( file_id, "/weight", H5P_DEFAULT)){
        if (!checkisempty(file_id, "/weight")){
            dims2D[0]=m->allng;dims2D[1]=1;
            if (!(checkmatndim(file_id, "/weight",  2, dims2D))){
                m->weightlength=1;
            }
            dims2D[0]=m->allng;dims2D[1]=m->NT;
            if (!(checkmatndim(file_id, "/weight",  2, dims2D))){
                m->weightlength=m->NT;
            }
            
            if (m->weightlength==0){
                fprintf(stderr, "Warning: Variable weight must be NT or 1 x number of geophones\n");
            }
            else{
                GMALLOC(weight0,sizeof(float)*dims2D[0]*m->weightlength )
                __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/weight", weight0);
                GMALLOC(m->weight,sizeof(float*)*m->ns)
                if (!state){
                    m->weight[0]=weight0;
                    for (i=1;i<m->ns;i++){
                        m->weight[i]=m->weight[i-1]+m->nrec[i-1]*m->weightlength;
                    }
                }
            }
        }
    }
    
    if (1==H5Lexists( file_id, "/topo", H5P_DEFAULT)){
        dims2D[0]=m->NX;dims2D[1]=m->NY;
        if (m->NY==1){
            if (!state) if ((state=checkmatndim(file_id, "/topo",  1, dims2D)))  {state=1;fprintf(stderr, "Variable topo must NX long in 2D\n");};
            GMALLOC(m->topo,sizeof(float)*dims2D[0])
            __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/topo", m->topo);
        }
        else{
            if (!state) if ((state=checkmatndim(file_id, "/topo",  2, dims2D)))  {state=1;fprintf(stderr, "Variable topo must NX x NY in 3D\n");};
            GMALLOC(m->topo,sizeof(float)*dims2D[0]*dims2D[1])
            __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/topo", m->topo);
        }
        
        float topomin=m->topo[0];
        float topomax=m->topo[0];
        for (i=0;i<m->NX*m->NY;i++){
            if (m->topo[i]>topomax)
                topomax=m->topo[i];
            if (m->topo[i]<topomin)
            topomin=m->topo[i];
        }
        for (i=0;i<m->NX*m->NY;i++){
            m->topo[i]=topomax-m->topo[i];
        }
        m->topowidth=(topomax-topomin)/m->dh;
    }
    
    

        __GUARD checkexists(file_id,"/back_prop_type");
        __GUARD checkscalar(file_id, "/back_prop_type");
        __GUARD readvar(file_id, H5T_NATIVE_INT, "/back_prop_type", &m->back_prop_type);
    if (!(m->back_prop_type ==1 | m->back_prop_type ==2)) {state=1;fprintf(stderr, "bac_prop_type must be 1 or 2\n");}
    
        if (H5Lexists( file_id, "/param_type", H5P_DEFAULT) ){
            __GUARD checkscalar(file_id, "/param_type");
            __GUARD readvar(file_id, H5T_NATIVE_INT, "/param_type", &m->param_type);
        }
        
        if (m->back_prop_type==2 && m->gradout==1){
            
            
            __GUARD checkexists(file_id,"/gradfreqs");
            __GUARD getdimmat(file_id, "/gradfreqs", 2, dimsfreqs);
            if (dimsfreqs[0]<2 & dimsfreqs[1]>0)
                m->nfreqs=(int)dimsfreqs[1];
            else if (dimsfreqs[1]<2 & dimsfreqs[0]>0)
                m->nfreqs=(int)dimsfreqs[0];
            
            GMALLOC(m->gradfreqs,sizeof(float)*m->nfreqs)
            __GUARD readvar(file_id, H5T_NATIVE_FLOAT, "/gradfreqs", m->gradfreqs);
            
            
        }


    
    if (file_id>=0) H5Fclose(file_id);
    file_id=0;
    
    
    /* Model file__________________________________
     __________________________________________________________________*/
    
    /* Open the model file. */
    if (!state) file_id = H5Fopen(files.model, H5F_ACC_RDWR, H5P_DEFAULT);
    if (!state) if (file_id<0) {state=1;fprintf(stderr, "Could not open the input file model");};
    
    
    const char *var1=NULL, *var2=NULL, *var3=NULL, *var4=NULL, *var5=NULL;
    if (m->param_type==1){
        var1="/rho";
        var2="/M";
        var3="/mu";
        var4="/taup";
        var5="/taus";
    }
    else if (m->param_type==2){
        var1="/rho";
        var2="/Ip";
        var3="/Is";
        var4="/taup";
        var5="/taus";
    }
    else if (m->param_type==3){
        if (m->L==0) {state=1;fprintf(stderr, "Viscoelastic modeling is required for param_type 3\n");};
        var1="/rho";
        var2="/vpR";
        var3="/vsR";
        var4="/vpI";
        var5="/vsI";
    }
    else {
        m->param_type=0;
        var1="/rho";
        var2="/vp";
        var3="/vs";
        var4="/taup";
        var5="/taus";
    }
    
    dims3D[2]=m->NZ;dims3D[1]=m->NY;dims3D[0]=m->NX;
    
    __GUARD checkexists(file_id,var1);
    __GUARD checkexists(file_id,var3);
    
    if (!state) if ((state=checkmatndim(file_id, var1,  3, dims3D)))  {state=1;fprintf(stderr, "Variable %s must be NZxNYxNX\n",var1);};
    if (!state) if ((state=checkmatndim(file_id, var3,   3, dims3D)))  {state=1;fprintf(stderr, "Variable %s must be NZxNYxNX\n",var3);};
    
    GMALLOC(m->rho,sizeof(float)*m->NX*m->NY*m->NZ)
    GMALLOC(m->u,sizeof(float)*m->NX*m->NY*m->NZ)
    
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, var1, m->rho);
    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, var3, m->u);
    
    if (m->ND!=21){//P-wave velocity not required for SH wave propagation in 2D
            __GUARD checkexists(file_id,var2);
            if (!state) if ((state=checkmatndim(file_id, var2,   3, dims3D)))  {state=1;fprintf(stderr, "Variable %s must be NZxNYxNX\n",var2);};
            GMALLOC(m->pi,sizeof(float)*m->NX*m->NY*m->NZ)
            __GUARD readvar(file_id, H5T_NATIVE_FLOAT, var2, m->pi);
    }
    
    /* Visco-elastic modeling*/
    if (m->L>0){

        __GUARD checkexists(file_id,var5);
        if (!state) if ((state=checkmatndim(file_id, var5,   3, dims3D)))    {state=1;fprintf(stderr, "Variable %s must be NZxNYxNX\n",var5);};
        GMALLOC(m->taus,sizeof(float)*m->NX*m->NY*m->NZ)
        __GUARD readvar(file_id, H5T_NATIVE_FLOAT, var5, m->taus);
        
        if (m->ND!=21){//P-wave attenuation not required for SH wave propagation in 2D
                    __GUARD checkexists(file_id,var4);
                    if (!state) if ((state=checkmatndim(file_id, var4,  3, dims3D)))     {state=1;fprintf(stderr, "Variable %s must be NZxNYxNX\n",var4);};
                    GMALLOC(m->taup,sizeof(float)*m->NX*m->NY*m->NZ)
                    __GUARD readvar(file_id, H5T_NATIVE_FLOAT, var4, m->taup);
        }
        
        
    }
    

    if (file_id>=0) H5Fclose(file_id);
    file_id=0;
    
    /* Data in file__________________________________
     __________________________________________________________________*/
    
    /* Open the data file. */
    if (m->rmsout==1 || m->resout || m->gradout==1){
        file_id = H5Fopen(files.din, H5F_ACC_RDWR, H5P_DEFAULT);
        if (!state) if (file_id<0 && m->gradout) {state=1;fprintf(stderr, "Could not open the input file for data_in\n");};
        
        
        if (file_id>=0){

            
            dims2D[0]=m->allng;dims2D[1]=m->NT;
            if ( m->ND!=21 && 1==H5Lexists( file_id, "/vz0", H5P_DEFAULT) ){
                
                m->bcastvz=1;
                
                if (!state) if ((state=checkmatndim_atleast(file_id, "/vz0",  2, dims2D)))   {state=1;fprintf(stderr, "Variable vz0 must be nt x number of geophones\n");};
                GMALLOC(vz00,sizeof(float)*m->allng*m->NT)
                if (!state) memset(vz00,0,sizeof(float)*m->allng*m->NT);
                __GUARD read_seis(file_id, H5T_NATIVE_FLOAT, "/vz0", vz00, m->rec_pos[0], m->allng, m->NT);
                GMALLOC(m->vz0,sizeof(float*)*m->ns)
                if (!state){
                    m->vz0[0]=vz00;
                    
                    for (i=1;i<m->ns;i++){
                        m->vz0[i]=m->vz0[i-1]+m->nrec[i-1]*m->NT;
                    }
                }
            }
            if ( m->ND!=21 && 1==H5Lexists( file_id, "/vx0", H5P_DEFAULT) ){
                
                m->bcastvx=1;
                
                if (!state) if ((state=checkmatndim_atleast(file_id, "/vx0",  2, dims2D)))   {state=1;fprintf(stderr, "Variable vx0 must be nt x number of geophones\n");};
                GMALLOC(vx00,sizeof(float)*m->allng*m->NT)
                if (!state) memset(vx00,0,sizeof(float)*m->allng*m->NT);
                __GUARD read_seis(file_id, H5T_NATIVE_FLOAT, "/vx0", vx00, m->rec_pos[0], m->allng, m->NT);
                GMALLOC(m->vx0,sizeof(float*)*m->ns)
                if (!state){
                    m->vx0[0]=vx00;
                    
                    for (i=1;i<m->ns;i++){
                        m->vx0[i]=m->vx0[i-1]+m->nrec[i-1]*m->NT;
                    }
                }
            }
            if ( m->ND!=2 && 1==H5Lexists( file_id, "/vy0", H5P_DEFAULT) ){
                
                m->bcastvy=1;
                if (!state) if ((state=checkmatndim_atleast(file_id, "/vy0",  2, dims2D)))   {state=1;fprintf(stderr, "Variable vy0 must be nt x number of geophones\n");};
                GMALLOC(vy00,sizeof(float)*dims2D[0]*m->NT)
                if (!state) memset(vy00,0,sizeof(float)*m->allng*m->NT);
                __GUARD read_seis(file_id, H5T_NATIVE_FLOAT, "/vy0", vy00, m->rec_pos[0], m->allng, m->NT);
                GMALLOC(m->vy0,sizeof(float*)*m->ns)
                if (!state){
                    m->vy0[0]=vy00;
                    for (i=1;i<m->ns;i++){
                        m->vy0[i]=m->vy0[i-1]+m->nrec[i-1]*m->NT;
                    }
                }
            }
            if ( m->ND!=21 && 1==H5Lexists( file_id, "/p0", H5P_DEFAULT) ){
                
                m->bcastp=1;
                
                if (!state) if ((state=checkmatndim_atleast(file_id, "/p0",  2, dims2D)))   {state=1;fprintf(stderr, "Variable p0 must be nt x number of hydrophones\n");};
                GMALLOC(p00,sizeof(float)*m->allng*m->NT)
                if (!state) memset(p00,0,sizeof(float)*m->allng*m->NT);
                __GUARD read_seis(file_id, H5T_NATIVE_FLOAT, "/p0", p00, m->rec_pos[0], m->allng, m->NT);
                GMALLOC(m->p0,sizeof(float*)*m->ns)
                if (!state){
                    m->p0[0]=vx00;
                    
                    for (i=1;i<m->ns;i++){
                        m->p0[i]=m->p0[i-1]+m->nrec[i-1]*m->NT;
                    }
                }
            }
          
            
        }
        
        if (file_id>=0) H5Fclose(file_id);
        
    }
    if (   m->bcastvx!=1 && m->bcastvy!=1 && m->bcastvz!=1 && m->bcastp!=1 && (m->gradout || m->rmsout==1 || m->resout==1)){
        state=1;
        fprintf(stderr, "Cannot output rms or residuals without reference data\n");
    }
        
    
    
    if (state && m->MPI_INIT==1) MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    return state;
    
}




