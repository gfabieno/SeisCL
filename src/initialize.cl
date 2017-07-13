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

    
#if LVE>0
    int gsize=get_global_size(0);
#if ND==3
    for (int l=0;l<LVE;l++){
        
        rxx[gid+l*gsize]=0.0;
        rzz[gid+l*gsize]=0.0;
        rxz[gid+l*gsize]=0.0;
        
        ryy[gid+l*gsize]=0.0;
        rxy[gid+l*gsize]=0.0;
        ryz[gid+l*gsize]=0.0;
        
    }
#endif
#if ND==2
    for (int l=0;l<LVE;l++){
        
        rxx[gid+l*gsize]=0.0;
        rzz[gid+l*gsize]=0.0;
        rxz[gid+l*gsize]=0.0;
        
        
    }
#endif
#if ND==21
    for (int l=0;l<LVE;l++){
        rxy[gid+l*gsize]=0.0;
        ryz[gid+l*gsize]=0.0;
 
    }
#endif
    
#endif
    

#if ABS_TYPE==1
    
#if ND==3
    if (gid<(NX-2*FDOH)*(NY-2*FDOH)*2*NAB){
        psi_sxz_z[gid]=0.0;
        psi_syz_z[gid]=0.0;
        psi_szz_z[gid]=0.0;
        psi_vxz[gid]=0.0;
        psi_vyz[gid]=0.0;
        psi_vzz[gid]=0.0;
    }
    if (gid<(NX-2*FDOH)*(NZ-2*FDOH)*2*NAB){
        psi_sxy_y[gid]=0.0;
        psi_syz_y[gid]=0.0;
        psi_syy_y[gid]=0.0;
        psi_vxy[gid]=0.0;
        psi_vyy[gid]=0.0;
        psi_vzy[gid]=0.0;
    }
    if (gid<(NY-2*FDOH)*(NZ-2*FDOH)*2*NAB){
        psi_sxx_x[gid]=0.0;
        psi_sxy_x[gid]=0.0;
        psi_sxz_x[gid]=0.0;
        psi_vxx[gid]=0.0;
        psi_vyx[gid]=0.0;
        psi_vzx[gid]=0.0;
    }
#endif
 
#if ND==2
    if (gid<(NX-2*FDOH)*2*NAB){
        psi_sxz_z[gid]=0.0;
        psi_szz_z[gid]=0.0;
        psi_vxz[gid]=0.0;
        psi_vzz[gid]=0.0;
    }

    if (gid<(NZ-2*FDOH)*2*NAB){
        psi_sxx_x[gid]=0.0;
        psi_sxz_x[gid]=0.0;
        psi_vxx[gid]=0.0;
        psi_vzx[gid]=0.0;
    }
#endif
    
#if ND==21
    if (gid<(NX-2*FDOH)*2*NAB){
        psi_syz_z[gid]=0.0;
        psi_vyz[gid]=0.0;
    }
    if (gid<(NZ-2*FDOH)*2*NAB){
        psi_sxy_x[gid]=0.0;
        psi_vyx[gid]=0.0;
    }
#endif
    
    
#endif
    
    
    



}

__kernel void initialize_seis_r(__global float *vx_r,         __global float *vy_r,       __global float *vz_r,
                              __global float *sxx_r,        __global float *syy_r,      __global float *szz_r,
                              __global float *sxy_r,        __global float *syz_r,      __global float *sxz_r,
                              __global float *rxx_r,        __global float *ryy_r,      __global float *rzz_r,
                              __global float *rxy_r,        __global float *ryz_r,      __global float *rxz_r,
                              __global float *psi_sxx_x_r,  __global float *psi_sxy_x_r,     __global float *psi_sxy_y_r,
                              __global float *psi_sxz_x_r,  __global float *psi_sxz_z_r,     __global float *psi_syy_y_r,
                              __global float *psi_syz_y_r,  __global float *psi_syz_z_r,     __global float *psi_szz_z_r,
                              __global float *psi_vxx_r,    __global float *psi_vxy_r_r,       __global float *psi_vxz_r,
                              __global float *psi_vyx_r,    __global float *psi_vyy_r_r,       __global float *psi_vyz_r,
                              __global float *psi_vzx_r,    __global float *psi_vzy_r_r,       __global float *psi_vzz_r
                              )
{
    
    
    int gid = get_global_id(0);
    
#if ND!=21
    sxx_r[gid]=0.0;
    szz_r[gid]=0.0;
    sxz_r[gid]=0.0;
    vx_r[gid]=0.0;
    vz_r[gid]=0.0;
#endif
#if ND==3 || ND==21
    sxy_r[gid]=0.0;
    syz_r[gid]=0.0;
    vy_r[gid]=0.0;
#endif
#if ND==3
    syy_r[gid]=0.0;
#endif
    
    
#if LVE>0
    int gsize=get_global_size(0);
#if ND==3
    for (int l=0;l<LVE;l++){
        
        rxx_r[gid+l*gsize]=0.0;
        rzz_r[gid+l*gsize]=0.0;
        rxz_r[gid+l*gsize]=0.0;
        
        ryy_r[gid+l*gsize]=0.0;
        rxy_r[gid+l*gsize]=0.0;
        ryz_r[gid+l*gsize]=0.0;
        
    }
#endif
#if ND==2
    for (int l=0;l<LVE;l++){
        
        rxx_r[gid+l*gsize]=0.0;
        rzz_r[gid+l*gsize]=0.0;
        rxz_r[gid+l*gsize]=0.0;
        
        
    }
#endif
#if ND==21
    for (int l=0;l<LVE;l++){
        rxy_r[gid+l*gsize]=0.0;
        ryz_r[gid+l*gsize]=0.0;
        
    }
#endif
    
#endif
    
    
#if ABS_TYPE==1
    
#if ND==3
    if (gid<(NX-2*FDOH)*(NY-2*FDOH)*2*NAB){
        psi_sxz_z_r[gid]=0.0;
        psi_syz_z_r[gid]=0.0;
        psi_szz_z_r[gid]=0.0;
        psi_vxz_r[gid]=0.0;
        psi_vyz_r[gid]=0.0;
        psi_vzz_r[gid]=0.0;
    }
    if (gid<(NX-2*FDOH)*(NZ-2*FDOH)*2*NAB){
        psi_sxy_y_r[gid]=0.0;
        psi_syz_y_r[gid]=0.0;
        psi_syy_y_r[gid]=0.0;
        psi_vxy_r[gid]=0.0;
        psi_vyy_r[gid]=0.0;
        psi_vzy_r[gid]=0.0;
    }
    if (gid<(NY-2*FDOH)*(NZ-2*FDOH)*2*NAB){
        psi_sxx_x_r[gid]=0.0;
        psi_sxy_x_r[gid]=0.0;
        psi_sxz_x_r[gid]=0.0;
        psi_vxx_r[gid]=0.0;
        psi_vyx_r[gid]=0.0;
        psi_vzx_r[gid]=0.0;
    }
#endif
    
#if ND==2
    if (gid<(NX-2*FDOH)*2*NAB){
        psi_sxz_z_r[gid]=0.0;
        psi_szz_z_r[gid]=0.0;
        psi_vxz_r[gid]=0.0;
        psi_vzz_r[gid]=0.0;
    }
    
    if (gid<(NZ-2*FDOH)*2*NAB){
        psi_sxx_x_r[gid]=0.0;
        psi_sxz_x_r[gid]=0.0;
        psi_vxx_r[gid]=0.0;
        psi_vzx_r[gid]=0.0;
    }
#endif
    
#if ND==21
    if (gid<(NX-2*FDOH)*2*NAB){
        psi_syz_z_r[gid]=0.0;
        psi_vyz_r[gid]=0.0;
    }
    if (gid<(NZ-2*FDOH)*2*NAB){
        psi_sxy_x_r[gid]=0.0;
        psi_vyx_r[gid]=0.0;
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
    for (freq=0;freq<NFREQS;freq++){
        
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

#if LVE>0
        for (l=0;l<LVE;l++){
#if ND!=21
            frxx[gid + l*gsize + freq*LVE*gsize]=0.0;
            frzz[gid + l*gsize + freq*LVE*gsize]=0.0;
            frxz[gid + l*gsize + freq*LVE*gsize]=0.0;
#endif
#if ND==3 || ND==21
            frxy[gid + l*gsize + freq*LVE*gsize]=0.0;
            fryz[gid + l*gsize + freq*LVE*gsize]=0.0;
#endif
#if ND==3
            fryy[gid + l*gsize + freq*LVE*gsize]=0.0;
#endif
        }
#endif
        
    }

    
}

__kernel void seisoutinit(__global float *vxout,      __global float *vyout,         __global float *vzout,
                          __global float *sxxout,     __global float *syyout,        __global float *szzout,
                          __global float *sxyout,     __global float *syzout,        __global float *sxzout,
                          __global float *pout)
{
    
    
    int gid = get_global_id(0);

#if bcastvx==1
    vxout[gid]=0.0;
#endif
#if bcastvy==1
    vyout[gid]=0.0;
#endif
#if bcastvz==1
    vzout[gid]=0.0;
#endif
#if bcastsxx==1
    sxxout[gid]=0.0;
#endif
#if bcastsyy==1
    syyout[gid]=0.0;
#endif
#if bcastszz==1
    szzout[gid]=0.0;
#endif
#if bcastsxy==1
    sxyout[gid]=0.0;
#endif
#if bcastsxz==1
    sxzout[gid]=0.0;
#endif
#if bcastsyz==1
    syzout[gid]=0.0;
#endif
#if bcastp==1
    pout[gid]=0.0;
#endif
    
    
    
}
__kernel void initialize_gradsrc(__global float *gradsrc)
    {

        int gid = get_global_id(0);
        gradsrc[gid]=0.0;

}
