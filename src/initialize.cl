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

/*Kernels to intialize buffers to 0*/


__kernel void initialize_seis(__global float *vx,         __global float *vy,       __global float *vz,
                              __global float *sxx,        __global float *syy,      __global float *szz,
                              __global float *sxy,        __global float *syz,      __global float *sxz,
                              __global float *rxx,        __global float *ryy,      __global float *rzz,
                              __global float *rxy,        __global float *ryz,      __global float *rxz,
                              __global float *psi_sxx_x,  __global float *psi_sxy_x,     __global float *psi_sxy_y,
                              __global float *psi_sxz_x,  __global float *psi_sxz_z,     __global float *psi_syy_y,
                              __global float *psi_syz_y,  __global float *psi_syz_z,     __global float *psi_szz_z,
                              __global float *psi_vxx,    __global float *psi_vxy,       __global float *psi_vxz,
                              __global float *psi_vyx,    __global float *psi_vyy,       __global float *psi_vyz,
                              __global float *psi_vzx,    __global float *psi_vzy,       __global float *psi_vzz
                              )
{
    
    
    int gid = get_global_id(0);
    
#if ND!=21
        sxx[gid]=0.0;
        szz[gid]=0.0;
        sxz[gid]=0.0;
        vx[gid]=0.0;
        vz[gid]=0.0;
#endif
#if ND==3 || ND==21
        sxy[gid]=0.0;
        syz[gid]=0.0;
        vy[gid]=0.0;
#endif
#if ND==3
        syy[gid]=0.0;
#endif

    
#if Lve>0
    int gsize=get_global_size(0);
#if ND==3
    for (int l=0;l<Lve;l++){
        
        rxx[gid+l*gsize]=0.0;
        rzz[gid+l*gsize]=0.0;
        rxz[gid+l*gsize]=0.0;
        
        ryy[gid+l*gsize]=0.0;
        rxy[gid+l*gsize]=0.0;
        ryz[gid+l*gsize]=0.0;
        
    }
#endif
#if ND==2
    for (int l=0;l<Lve;l++){
        
        rxx[gid+l*gsize]=0.0;
        rzz[gid+l*gsize]=0.0;
        rxz[gid+l*gsize]=0.0;
        
        
    }
#endif
#if ND==21
    for (int l=0;l<Lve;l++){
        rxy[gid+l*gsize]=0.0;
        ryz[gid+l*gsize]=0.0;
 
    }
#endif
    
#endif
    

#if abs_type==1
    
#if ND==3
    if (gid<(NX-2*fdoh)*(NY-2*fdoh)*2*nab){
        psi_sxz_z[gid]=0.0;
        psi_syz_z[gid]=0.0;
        psi_szz_z[gid]=0.0;
        psi_vxz[gid]=0.0;
        psi_vyz[gid]=0.0;
        psi_vzz[gid]=0.0;
    }
    if (gid<(NX-2*fdoh)*(NZ-2*fdoh)*2*nab){
        psi_sxy_y[gid]=0.0;
        psi_syz_y[gid]=0.0;
        psi_syy_y[gid]=0.0;
        psi_vxy[gid]=0.0;
        psi_vyy[gid]=0.0;
        psi_vzy[gid]=0.0;
    }
    if (gid<(NY-2*fdoh)*(NZ-2*fdoh)*2*nab){
        psi_sxx_x[gid]=0.0;
        psi_sxy_x[gid]=0.0;
        psi_sxz_x[gid]=0.0;
        psi_vxx[gid]=0.0;
        psi_vyx[gid]=0.0;
        psi_vzx[gid]=0.0;
    }
#endif
 
#if ND==2
    if (gid<(NX-2*fdoh)*2*nab){
        psi_sxz_z[gid]=0.0;
        psi_szz_z[gid]=0.0;
        psi_vxz[gid]=0.0;
        psi_vzz[gid]=0.0;
    }

    if (gid<(NZ-2*fdoh)*2*nab){
        psi_sxx_x[gid]=0.0;
        psi_sxz_x[gid]=0.0;
        psi_vxx[gid]=0.0;
        psi_vzx[gid]=0.0;
    }
#endif
    
#if ND==21
    if (gid<(NX-2*fdoh)*2*nab){
        psi_syz_z[gid]=0.0;
        psi_vyz[gid]=0.0;
    }
    if (gid<(NZ-2*fdoh)*2*nab){
        psi_sxy_x[gid]=0.0;
        psi_vyx[gid]=0.0;
    }
#endif
    
    
#endif
    
    
    



}

__kernel void initialize_grad(__global float *gradrho,       __global float *gradM,     __global float *gradmu,
                              __global float *gradtaup,      __global float *gradtaus)
{

    int gid = get_global_id(0);
    
    gradrho[gid]=0.0;
    #if ND!=21
    gradM[gid]=0.0;
    #endif
    gradmu[gid]=0.0;
    
    
}

__kernel void initialize_savefreqs(
                                   __global float2 *fvx,   __global float2 *fvy,       __global float2 *fvz,
                                   __global float2 *fsxx,  __global float2 *fsyy,      __global float2 *fszz,
                                   __global float2 *fsxy,  __global float2 *fsyz,      __global float2 *fsxz,
                                   __global float2 *frxx,  __global float2 *fryy,      __global float2 *frzz,
                                   __global float2 *frxy,  __global float2 *fryz,      __global float2 *frxz)
{
    
    int gid = get_global_id(0);
    int freq,l;
    int gsize=get_global_size(0);
    for (freq=0;freq<nfreqs;freq++){
        
#if ND!=21
        fvx[gid +freq*gsize]=0.0;
        fvz[gid +freq*gsize]=0.0;
        fsxx[gid +freq*gsize]=0.0;
        fszz[gid +freq*gsize]=0.0;
        fsxz[gid +freq*gsize]=0.0;
#endif
#if ND==3 || ND==21
        fvy[gid +freq*gsize]=0.0;
        fsxy[gid +freq*gsize]=0.0;
        fsyz[gid +freq*gsize]=0.0;
        
#endif
#if ND==3
        fsyy[gid +freq*gsize]=0.0;
#endif

#if Lve>0
        for (l=0;l<Lve;l++){
#if ND!=21
            frxx[gid + l*gsize + freq*Lve*gsize]=0.0;
            frzz[gid + l*gsize + freq*Lve*gsize]=0.0;
            frxz[gid + l*gsize + freq*Lve*gsize]=0.0;
#endif
#if ND==3 || ND==21
            frxy[gid + l*gsize + freq*Lve*gsize]=0.0;
            fryz[gid + l*gsize + freq*Lve*gsize]=0.0;
#endif
#if ND==3
            fryy[gid + l*gsize + freq*Lve*gsize]=0.0;
#endif
        }
#endif
        
    }

    
}

__kernel void voutinit(__global float *vxout, __global float *vyout, __global float *vzout)
{
    
    
    int gid = get_global_id(0);
#if ND!=21
    vxout[gid]=0.0;
    vzout[gid]=0.0;
#endif
#if ND==3 || ND==21
    vyout[gid]=0.0;
#endif
}
__kernel void initialize_gradsrc(__global float *gradsrc)
    {

        int gid = get_global_id(0);
        gradsrc[gid]=0.0;

}
