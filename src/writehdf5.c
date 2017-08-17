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

/*Write HDF5 output files*/
#include "F.h"

// Write float matrix compatible with .mat v7.3 format
void writetomat(hid_t* file_id,
                const char *var,
                float * varptr,
                int NDIMs,
                hsize_t dims[] ){
    
    hid_t dataspace_id=0, dataset_id=0, attribute_id=0;
    hid_t    plist_id;
    hsize_t  cdims[MAX_DIMS];
    int ii;
    
    for (ii=0;ii<NDIMs;ii++){
        cdims[ii]=10;
    }
    
    hid_t vls_type_c_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(vls_type_c_id, 6);
    
    
    if (1!=H5Lexists( *file_id, var, H5P_DEFAULT)){
        dataspace_id = H5Screate_simple(NDIMs, dims, NULL);
        
        plist_id  = H5Pcreate (H5P_DATASET_CREATE);
        for (ii=0;ii<NDIMs;ii++){
            cdims[ii]=cdims[ii]<dims[ii]?cdims[ii]:dims[ii];
        }
        cdims[0]=dims[0];
        cdims[1]=1;
        H5Pset_chunk (plist_id, NDIMs, cdims);
//        H5Pset_deflate (plist_id, 6);
        
        dataset_id = H5Dcreate2(*file_id,
                                var,
                                H5T_IEEE_F32LE,
                                dataspace_id,
                                H5P_DEFAULT,
                                plist_id,
                                H5P_DEFAULT);
        H5Dwrite(dataset_id,
                 H5T_NATIVE_FLOAT,
                 H5S_ALL,
                 H5S_ALL,
                 H5P_DEFAULT,
                 varptr);
        H5Sclose(dataspace_id);
        
        dataspace_id = H5Screate(H5S_SCALAR);
        attribute_id = H5Acreate2 (dataset_id,
                                   "MATLAB_class",
                                   vls_type_c_id,
                                   dataspace_id,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);
        H5Awrite(attribute_id, vls_type_c_id, "single");
        H5Aclose(attribute_id);
        H5Sclose(dataspace_id);
        H5Pclose (plist_id);
        H5Dclose(dataset_id);
        
    }
    else{
        dataset_id = H5Dopen2(*file_id, var, H5P_DEFAULT);
        H5Dwrite(dataset_id,
                 H5T_NATIVE_FLOAT,
                 H5S_ALL,
                 H5S_ALL,
                 H5P_DEFAULT,
                 varptr);
        H5Dclose(dataset_id);
        
    }
    
}

//Write double matrix compatible with .mat v7.3 format
void writetomatd(hid_t* file_id,
                 const char *var,
                 double * varptr,
                 int NDIMs,
                 hsize_t dims[] ){
    
    hid_t dataspace_id=0, dataset_id=0, attribute_id=0;
    hid_t    plist_id;
    hsize_t  cdims[MAX_DIMS]={10};
    int ii;
    
    hid_t vls_type_c_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(vls_type_c_id, 6);
    
    
    if (1!=H5Lexists( *file_id, var, H5P_DEFAULT)){
        dataspace_id = H5Screate_simple(NDIMs, dims, NULL);
        
        plist_id  = H5Pcreate (H5P_DATASET_CREATE);
        for (ii=0;ii<NDIMs;ii++){
            cdims[ii]=cdims[ii]<dims[ii]?cdims[ii]:dims[ii];
        }
        H5Pset_chunk (plist_id, NDIMs, cdims);
        H5Pset_deflate (plist_id, 6);
        
        dataset_id = H5Dcreate2(*file_id, var,
                                H5T_IEEE_F64LE,
                                dataspace_id,
                                H5P_DEFAULT,
                                plist_id,
                                H5P_DEFAULT);
        H5Dwrite(dataset_id,
                 H5T_NATIVE_DOUBLE,
                 H5S_ALL,
                 H5S_ALL,
                 H5P_DEFAULT,
                 varptr);
        H5Sclose(dataspace_id);
        
        dataspace_id = H5Screate(H5S_SCALAR);
        attribute_id = H5Acreate2 (dataset_id,
                                   "MATLAB_class",
                                   vls_type_c_id,
                                   dataspace_id,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);
        H5Awrite(attribute_id, vls_type_c_id, "double");
        H5Aclose(attribute_id);
        H5Sclose(dataspace_id);
        H5Pclose (plist_id);
        H5Dclose(dataset_id);
        
    }
    else{
        dataset_id = H5Dopen2(*file_id, var, H5P_DEFAULT);
        H5Dwrite(dataset_id,
                 H5T_NATIVE_DOUBLE,
                 H5S_ALL,
                 H5S_ALL,
                 H5P_DEFAULT,
                 varptr);
        H5Dclose(dataset_id);
        
    }
    
}

// Create HDF5 file, compatible with .mat v7.3 format
hid_t create_file(const char *filename){
    FILE * fp;
    hid_t       file_id=0, fcpl_id=0;
    
    fcpl_id = H5Pcreate (H5P_FILE_CREATE);
    H5Pset_userblock(fcpl_id, 512 );
    file_id = H5Fcreate( filename, H5F_ACC_TRUNC, fcpl_id, H5P_DEFAULT );
    H5Fclose(file_id);
    
    fp = fopen(filename,"r+");
    int matbin[] = {0x00000000, 0x00000000, 0x4D490200};
    const char * mathead = "MATLAB 7.3 MAT-file, "
                           "Platform: GLNXA64, "
                           "Created on: Fri Feb 07 02:29:00 2014 "
                           "HDF5 schema 1.00 .                     ";
    fprintf(fp, "%s",mathead);
    fwrite(matbin,sizeof(int),3,fp);
    fclose(fp);
    
    file_id  = H5Fopen( filename, H5F_ACC_RDWR, H5P_DEFAULT );
    
    
    return file_id;
}

int writehdf5(struct filenames file, model * m) {
    
    hid_t       file_id=0;
    int i;
    int state=0;
    float rms;
    char name[100];
    hsize_t dims[MAX_DIMS+2];

   
    // Write data output file
    if (m->VARSOUT || m->RESOUT){
        
        file_id = create_file(file.dout);
        if (!state && file_id<0){
            state=1;
            fprintf(stderr,"Could not open the input/output file %s",file.dout);
        }
        
        dims[0]=m->src_recs.allng; dims[1]=m->NT;
        
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_output){
                sprintf(name, "%sout",m->vars[i].name);
                writetomat(&file_id,name,m->vars[i].gl_varout[0],2,dims);
            }
        }
        for (i=0;i<m->ntvars;i++){
            if (m->trans_vars[i].to_output){
                sprintf(name, "%sout",m->trans_vars[i].name);
                writetomat(&file_id,name,m->trans_vars[i].gl_varout[0],2,dims);
            }
        }
        if (m->RESOUT){
            for (i=0;i<m->nvars;i++){
                if (m->vars[i].to_output){
                    sprintf(name, "%sres",m->vars[i].name);
                    writetomat(&file_id,name,m->vars[i].gl_var_res[0],2,dims);
                }
            }
            for (i=0;i<m->ntvars;i++){
                if (m->trans_vars[i].to_output){
                    sprintf(name, "%sres",m->trans_vars[i].name);
                    writetomat(&file_id,name,m->trans_vars[i].gl_var_res[0],2,dims);
                }
            }
        }
        
        dims[0]=m->src_recs.allns; dims[1]=5;
        writetomat(&file_id, "/src_pos", m->src_recs.src_pos[0], 2, dims );
        
        dims[0]=m->src_recs.allng; dims[1]=8;
        writetomat(&file_id, "/rec_pos", m->src_recs.rec_pos[0], 2, dims );

        if (file_id) H5Fclose(file_id);

        
    }

    // Write rms output file
    if (m->RMSOUT){
        file_id = create_file(file.rmsout);
        rms=sqrt(m->rms);
        dims[0]=1; dims[1]=1;
        writetomat(&file_id, "/rms", &rms, 2, dims );
        rms=sqrt((m->rmsnorm));
        writetomat(&file_id, "/rms_norm", &rms, 2, dims );
        if (file_id) H5Fclose(file_id);
        
    }
    
    // Write gradient output file
    if (m->GRADOUT){
        
        file_id = create_file(file.gout);
        for (i=0;i<m->NDIM;i++){
            dims[m->NDIM-i-1]=m->N[i];
        }
        
        for (i=0;i<m->npars;i++){
            if (m->pars[i].to_grad){
                sprintf(name, "grad%s",&m->pars[i].to_read[1]);
                writetomat(&file_id,name,m->pars[i].gl_grad,m->NDIM,dims);
            }
        }
        // Write Hessian output file
        if ( m->HOUT){
            for (i=0;i<m->npars;i++){
                if (m->pars[i].to_grad){
                    sprintf(name, "H%s",m->pars[i].to_read);
                    writetomat(&file_id,name,m->pars[i].gl_H,m->NDIM,dims);
                }
            }
        }
        
        if (m->GRADSRCOUT==1){
            dims[0]=m->src_recs.allns; dims[1]=m->NT;
            writetomat(&file_id, "/gradsrc", m->src_recs.gradsrc[0], 2, dims );
            
        }

        if (file_id) H5Fclose(file_id);
    
    }
    
    // Write movie output file
    if (m->MOVOUT>0){
        file_id = create_file(file.movout);

        dims[0]=m->src_recs.ns;
        dims[1]=m->NT/m->MOVOUT;
        for (i=0;i<m->NDIM;i++){
            dims[m->NDIM-i+1]=m->N[i];
        }
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_output){
                sprintf(name, "mov%s",m->vars[i].name);
                writetomat(&file_id,name,m->vars[i].gl_mov,m->NDIM+2,dims);
            }
        }
        
        if (file_id) H5Fclose(file_id);
    }
    
    return state;

}
