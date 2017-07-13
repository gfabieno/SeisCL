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
#define vx(z,y,x)   vx[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vy(z,y,x)   vy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vz(z,y,x)   vz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define rip(z,y,x)     rip[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,y,x)     rjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,y,x)     rkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#endif

#if ND==2 || ND==21
#define vx(z,y,x)  vx[(x)*(NZ)+(z)]
#define vy(z,y,x)  vy[(x)*(NZ)+(z)]
#define vz(z,y,x)  vz[(x)*(NZ)+(z)]
#define rip(z,y,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,y,x)    rjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,y,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#endif

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]

#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define rec_pos(y,x) rec_pos[(y)*8+(x)]





__kernel void residuals( __global float *vx, __global float *vy, __global float *vz,
                        __global float *rx,  __global float *ry, __global float *rz,
                        __global float *rip, __global float *rjp,__global float *rkp,
                        __global float *rec_pos, int nt)
{
 
    int gid = get_global_id(0);
    
    int i=(int)(rec_pos(gid,0)/DH-0.5)+FDOH;
    int j=(int)(rec_pos(gid,1)/DH-0.5)+FDOH;
    int k=(int)(rec_pos(gid,2)/DH-0.5)+FDOH;
    
    if ( (i-OFFSET)>=FDOH && (i-OFFSET)<(NX-FDOH) ){
        
#if bcastvx==1
        vx(k,j,i-OFFSET) += rx(gid, nt)/rip(k,j,i-OFFSET);
#endif
#if bcastvy==1
        vy(k,j,i-OFFSET) += ry(gid, nt)/rjp(k,j,i-OFFSET);
#endif
#if bcastvz==1
        vz(k,j,i-OFFSET) += rz(gid, nt)/rkp(k,j,i-OFFSET);
#endif
        

    }
    
}

