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
void writetomat(hid_t* file_id, const char *var, float * varptr, int NDIMs, hsize_t dims[] ){
    
    hid_t dataspace_id=0, dataset_id=0, attribute_id=0;
    hid_t    plist_id;
    hsize_t  cdims[10]={20,20,20,20,20,20,20,20,20,20};
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
        
        dataset_id = H5Dcreate2(*file_id, var, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, varptr);
        H5Sclose(dataspace_id);
        
        dataspace_id = H5Screate(H5S_SCALAR);
        attribute_id = H5Acreate2 (dataset_id, "MATLAB_class", vls_type_c_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute_id, vls_type_c_id, "double");
        H5Aclose(attribute_id);
        H5Sclose(dataspace_id);
        H5Pclose (plist_id);
        H5Dclose(dataset_id);
        
    }
    else{
        dataset_id = H5Dopen2(*file_id, var, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, varptr);
        H5Dclose(dataset_id);
        
    }
    
}

//Write double matrix compatible with .mat v7.3 format
void writetomatd(hid_t* file_id, const char *var, double * varptr, int NDIMs, hsize_t dims[] ){
    
    hid_t dataspace_id=0, dataset_id=0, attribute_id=0;
    hid_t    plist_id;
    hsize_t  cdims[3]={20,20,20};
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
        
        dataset_id = H5Dcreate2(*file_id, var, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, varptr);
        H5Sclose(dataspace_id);
        
        dataspace_id = H5Screate(H5S_SCALAR);
        attribute_id = H5Acreate2 (dataset_id, "MATLAB_class", vls_type_c_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute_id, vls_type_c_id, "double");
        H5Aclose(attribute_id);
        H5Sclose(dataspace_id);
        H5Pclose (plist_id);
        H5Dclose(dataset_id);
        
    }
    else{
        dataset_id = H5Dopen2(*file_id, var, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, varptr);
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
    const char * mathead = "MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Fri Feb 07 02:29:00 2014 HDF5 schema 1.00 .                     ";
    fprintf(fp, "%s",mathead);
    fwrite(matbin,sizeof(int),3,fp);
    fclose(fp);
    
    file_id  = H5Fopen( filename, H5F_ACC_RDWR, H5P_DEFAULT );
    
    
    return file_id;
}

int writehdf5(struct filenames file, struct modcsts * m) {
    
    hid_t       file_id=0;
    hsize_t     dims2D[2], dims3D[3], dims5D[5];
    int state=0;
    float rms;

   
    // Write data output file
    if (m->SEISOUT || m->RESOUT){
        
        file_id = create_file(file.dout);
        if (!state) if (file_id<0) {state=1;fprintf(stderr, "Could not open the input/output file %s", file.dout);};
        
        dims2D[0]=m->allng; dims2D[1]=m->NT;
        
        if (m->bcastvx)  writetomat(&file_id, "/vxout",  m->vxout[0],  2, dims2D );
        if (m->bcastvy)  writetomat(&file_id, "/vyout",  m->vyout[0],  2, dims2D );
        if (m->bcastvz)  writetomat(&file_id, "/vzout",  m->vzout[0],  2, dims2D );
        if (m->bcastsxx) writetomat(&file_id, "/sxxout", m->sxxout[0], 2, dims2D );
        if (m->bcastsyy) writetomat(&file_id, "/syyout", m->syyout[0], 2, dims2D );
        if (m->bcastszz) writetomat(&file_id, "/szzout", m->szzout[0], 2, dims2D );
        if (m->bcastsxy) writetomat(&file_id, "/sxyout", m->sxyout[0], 2, dims2D );
        if (m->bcastsxz) writetomat(&file_id, "/sxzout", m->sxzout[0], 2, dims2D );
        if (m->bcastsyz) writetomat(&file_id, "/syzout", m->syzout[0], 2, dims2D );
        if (m->bcastp)   writetomat(&file_id, "/pout",   m->pout[0],   2, dims2D );

        if (m->RESOUT){
            if (m->bcastvx) writetomat(&file_id, "/rx", m->rx[0], 2, dims2D );
            if (m->bcastvz) writetomat(&file_id, "/rz", m->rz[0], 2, dims2D );
            if (m->bcastvy) writetomat(&file_id, "/ry", m->ry[0], 2, dims2D );
        }
        
        dims2D[0]=m->allns; dims2D[1]=5;
        writetomat(&file_id, "/src_pos", m->src_pos[0], 2, dims2D );
        
        dims2D[0]=m->allng; dims2D[1]=8;
        writetomat(&file_id, "/rec_pos", m->rec_pos[0], 2, dims2D );

        if (file_id) H5Fclose(file_id);

        
    }

    // Write rms output file
    if (m->RMSOUT){
        file_id = create_file(file.RMSOUT);
        rms=sqrt(m->rms);
        dims2D[0]=1; dims2D[1]=1;
        writetomat(&file_id, "/rms", &rms, 2, dims2D );
        rms=sqrt((m->rmsnorm));
        writetomat(&file_id, "/rms_norm", &rms, 2, dims2D );
        if (file_id) H5Fclose(file_id);
        
    }
    
    // Write gradient output file
    if (m->GRADOUT){
        
        file_id = create_file(file.gout);
        
        dims3D[2]=m->NZ; dims3D[1]=m->NY, dims3D[0]=m->NX;
        
        // Output name depends on the parametrization
        const char *var1=NULL, *var2=NULL, *var3=NULL, *var4=NULL, *var5=NULL;
        const char *Hvar1=NULL, *Hvar2=NULL, *Hvar3=NULL, *Hvar4=NULL, *Hvar5=NULL;
        if (m->param_type==1){
            var1="/gradrho";
            var2="/gradM";
            var3="/gradmu";
            var4="/gradtaup";
            var5="/gradtaus";
            Hvar1="/Hrho";
            Hvar2="/HM";
            Hvar3="/Hmu";
            Hvar4="/Htaup";
            Hvar5="/Htaus";
        }
        else if (m->param_type==2){
            var1="/gradrho";
            var2="/gradIp";
            var3="/gradIs";
            var4="/gradtaup";
            var5="/gradtaus";
            Hvar1="/Hrho";
            Hvar2="/HIp";
            Hvar3="/HIs";
            Hvar4="/Htaup";
            Hvar5="/Htaus";
        }
        else if (m->param_type==3){
            var1="/gradrho";
            var2="/gradvpR";
            var3="/gradvsR";
            var4="/gradvpI";
            var5="/gradvsI";
            Hvar1="/Hrho";
            Hvar2="/HvpR";
            Hvar3="/HvsR";
            Hvar4="/HvpI";
            Hvar5="/HvsI";
        }
        else {
            var1="/gradrho";
            var2="/gradvp";
            var3="/gradvs";
            var4="/gradtaup";
            var5="/gradtaus";
            Hvar1="/Hrho";
            Hvar2="/Hvp";
            Hvar3="/Hvs";
            Hvar4="/Htaup";
            Hvar5="/Htaus";
        }
        
        
        writetomatd(&file_id, var1, m->gradrho, 3, dims3D );
        if (m->ND!=21)
            writetomatd(&file_id, var2, m->gradM, 3, dims3D );
        writetomatd(&file_id, var3, m->gradmu, 3, dims3D );
        if (m->L>0){
            if (m->ND!=21)
                writetomatd(&file_id, var4, m->gradtaup, 3, dims3D );
            writetomatd(&file_id, var5, m->gradtaus, 3, dims3D );
        }
        
        if (m->GRADSRCOUT==1){
            dims2D[0]=m->allns; dims2D[1]=m->NT;
            writetomat(&file_id, "/gradsrc", m->gradsrc[0], 2, dims2D );
            
        }

        // Write Hessian output file
        if ( m->HOUT){
            writetomatd(&file_id, Hvar1, m->Hrho, 3, dims3D );
            if (m->ND!=21)
            writetomatd(&file_id, Hvar2, m->HM, 3, dims3D );
            writetomatd(&file_id, Hvar3, m->Hmu, 3, dims3D );
            if (m->L>0){
                if (m->ND!=21)
                writetomatd(&file_id, Hvar4, m->Htaup, 3, dims3D );
                writetomatd(&file_id, Hvar5, m->Htaus, 3, dims3D );
            }
        }
        if (file_id) H5Fclose(file_id);
    
    }
    

    
    // Write movie output file
    if (m->MOVOUT>0){
        file_id = create_file(file.MOVOUT);

        dims5D[0]=m->ns;
        dims5D[1]=m->NT/m->MOVOUT;
        dims5D[2]=m->NX;
        dims5D[3]=m->NY;
        dims5D[4]=m->NZ;
        if (m->ND!=21){
            writetomat(&file_id, "/movvx", m->movvx, 5, dims5D );
            writetomat(&file_id, "/movvz", m->movvz, 5, dims5D );
        }
        if (m->ND==3 || m->ND==21){
            writetomat(&file_id, "/movvy", m->movvy, 5, dims5D );
        }
        
        if (file_id) H5Fclose(file_id);
    }
    
    return state;

}
