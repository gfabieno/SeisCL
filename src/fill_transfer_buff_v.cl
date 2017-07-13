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
#define vx(z,y,x)   vx[(x)*NY*NZ+(y)*NZ+(z)]
#define vy(z,y,x)   vy[(x)*NY*NZ+(y)*NZ+(z)]
#define vz(z,y,x)   vz[(x)*NY*NZ+(y)*NZ+(z)]

#define vx_buf(z,y,x)   vx_buf[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define vy_buf(z,y,x)   vy_buf[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define vz_buf(z,y,x)   vz_buf[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#endif

#if ND==2 || ND==21
#define vx(z,y,x)  vx[(x)*NZ+(z)]
#define vy(z,y,x)  vy[(x)*NZ+(z)]
#define vz(z,y,x)  vz[(x)*NZ+(z)]

#define vx_buf(z,y,x)  vx_buf[(x)*(NZ-2*FDOH)+(z)]
#define vy_buf(z,y,x)  vy_buf[(x)*(NZ-2*FDOH)+(z)]
#define vz_buf(z,y,x)  vz_buf[(x)*(NZ-2*FDOH)+(z)]


#endif

__kernel void fill_transfer_buff_v_out( int gidx0, __global float *vx, __global float *vy, __global float *vz,
                                       __global float *vx_buf,  __global float *vy_buf, __global float *vz_buf)
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
    //gidx0=FDOH for comm1 and NX-2*FDOH for comm2
    vx_buf(gidz-FDOH,gidy,gidx)=vx(gidz,gidy,gidx+gidx0);
    vz_buf(gidz-FDOH,gidy,gidx)=vz(gidz,gidy,gidx+gidx0);

#if ND==3
    vy_buf(gidz-FDOH,gidy,gidx)=vy(gidz,gidy,gidx+gidx0);
#endif


}

__kernel void fill_transfer_buff_v_in( int gidx0, __global float *vx, __global float *vy, __global float *vz,
                                       __global float *vx_buf,  __global float *vy_buf, __global float *vz_buf)
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
    //gidx0=0 for comm1 and NX-FDOH for comm2
    vx(gidz,gidy,gidx+gidx0)=vx_buf(gidz-FDOH,gidy,gidx);
    vz(gidz,gidy,gidx+gidx0)=vz_buf(gidz-FDOH,gidy,gidx);
    
    
#if ND==3
    vy(gidz,gidy,gidx+gidx0)=vy_buf(gidz-FDOH,gidy,gidx);
#endif
    
}
