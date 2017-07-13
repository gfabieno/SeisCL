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

/*Adjoint sources */

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#if ND==3
#define sxx(z,y,x) sxx[(x)*NY*NZ+(y)*NZ+(z)]
#define syy(z,y,x) syy[(x)*NY*NZ+(y)*NZ+(z)]
#define szz(z,y,x) szz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxy(z,y,x) sxy[(x)*NY*NZ+(y)*NZ+(z)]
#define syz(z,y,x) syz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxz(z,y,x) sxz[(x)*NY*NZ+(y)*NZ+(z)]

#define sxx_buf(z,y,x) sxx_buf[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define syy_buf(z,y,x) syy_buf[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define szz_buf(z,y,x) szz_buf[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define sxy_buf(z,y,x) sxy_buf[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define syz_buf(z,y,x) syz_buf[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define sxz_buf(z,y,x) sxz_buf[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#endif

#if ND==2 || ND==21

#define sxx(z,y,x) sxx[(x)*NZ+(z)]
#define syy(z,y,x) syy[(x)*NZ+(z)]
#define szz(z,y,x) szz[(x)*NZ+(z)]
#define sxy(z,y,x) sxy[(x)*NZ+(z)]
#define syz(z,y,x) syz[(x)*NZ+(z)]
#define sxz(z,y,x) sxz[(x)*NZ+(z)]

#define sxx_buf(z,y,x) sxx_buf[(x)*(NZ-2*FDOH)+(z)]
#define syy_buf(z,y,x) syy_buf[(x)*(NZ-2*FDOH)+(z)]
#define szz_buf(z,y,x) szz_buf[(x)*(NZ-2*FDOH)+(z)]
#define sxy_buf(z,y,x) sxy_buf[(x)*(NZ-2*FDOH)+(z)]
#define syz_buf(z,y,x) syz_buf[(x)*(NZ-2*FDOH)+(z)]
#define sxz_buf(z,y,x) sxz_buf[(x)*(NZ-2*FDOH)+(z)]


#endif

__kernel void fill_transfer_buff1_s_out(int gidx0,
                                       __global float *sxx,        __global float *syy,          __global float *szz,
                                       __global float *sxy,        __global float *syz,          __global float *sxz,
                                       __global float *sxx_buf1,        __global float *syy_buf1,          __global float *szz_buf1,
                                       __global float *sxy_buf1,        __global float *syz_buf1,          __global float *sxz_buf1)
{
#if ND==3
    // If we use local memory
#if LOCAL_OFF==0
    
    int gidz = get_global_id(0)+FDOH;
    int gidy = get_global_id(1)+FDOH;
    int gidx = get_global_id(2);
    
    // If local memory is turned off
#elif LOCAL_OFF==1
    
    int gid = get_global_id(0);
    int gidz = gid%glsizez+FDOH;
    int gidy = (gid/glsizez)%glsizey+FDOH;
    int gidx = gid/(glsizez*glsizey);
    
#endif
    
#else
    
    // If we use local memory
#if LOCAL_OFF==0
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1);
    int gidy = 0;
    
    // If local memory is turned off
#elif LOCAL_OFF==1
    int gid = get_global_id(0);
    int gidz = gid%glsizez+FDOH;
    int gidx = (gid/glsizez);
    int gidy = 0;
#endif
#endif


    sxx_buf1(gidz-FDOH,gidy,gidx)=sxx(gidz,gidy,gidx+gidx0);
    szz_buf1(gidz-FDOH,gidy,gidx)=szz(gidz,gidy,gidx+gidx0);
    sxz_buf1(gidz-FDOH,gidy,gidx)=sxz(gidz,gidy,gidx+gidx0);
    
#if ND==3
    syy_buf1(gidz-FDOH,gidy,gidx)=syy(gidz,gidy,gidx+gidx0);
    sxy_buf1(gidz-FDOH,gidy,gidx)=sxy(gidz,gidy,gidx+gidx0);
    syz_buf1(gidz-FDOH,gidy,gidx)=syz(gidz,gidy,gidx+gidx0);
#endif
    

 
}

__kernel void fill_transfer_buff_s_in(int gidx0,
                                       __global float *sxx,        __global float *syy,          __global float *szz,
                                       __global float *sxy,        __global float *syz,          __global float *sxz,
                                       __global float *sxx_buf,        __global float *syy_buf,          __global float *szz_buf,
                                       __global float *sxy_buf,        __global float *syz_buf,          __global float *sxz_buf)
{
#if ND==3
    // If we use local memory
#if LOCAL_OFF==0
    
    int gidz = get_global_id(0)+FDOH;
    int gidy = get_global_id(1)+FDOH;
    int gidx = get_global_id(2);
    
    // If local memory is turned off
#elif LOCAL_OFF==1
    
    int gid = get_global_id(0);
    int gidz = gid%glsizez+FDOH;
    int gidy = (gid/glsizez)%glsizey+FDOH;
    int gidx = gid/(glsizez*glsizey);
    
#endif
    
#else
    
    // If we use local memory
#if LOCAL_OFF==0
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1);
    int gidy = 0;
    
    // If local memory is turned off
#elif LOCAL_OFF==1
    int gid = get_global_id(0);
    int gidz = gid%glsizez+FDOH;
    int gidx = (gid/glsizez);
    int gidy = 0;
#endif
#endif

    sxx(gidz,gidy,gidx+gidx0)=sxx_buf(gidz-FDOH,gidy,gidx);
    szz(gidz,gidy,gidx+gidx0)=szz_buf(gidz-FDOH,gidy,gidx);
    sxz(gidz,gidy,gidx+gidx0)=sxz_buf(gidz-FDOH,gidy,gidx);

#if ND==3
    syy(gidz,gidy,gidx+gidx0)=syy_buf(gidz-FDOH,gidy,gidx);
    sxy(gidz,gidy,gidx+gidx0)=sxy_buf(gidz-FDOH,gidy,gidx);
    syz(gidz,gidy,gidx+gidx0)=syz_buf(gidz-FDOH,gidy,gidx);
#endif
    
}
