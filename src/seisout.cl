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

/*Output seismograms */

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#if ND==3
#define vx(z,y,x)   vx[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vy(z,y,x)   vy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vz(z,y,x)   vz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxx(z,y,x) sxx[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syy(z,y,x) syy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define szz(z,y,x) szz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxy(z,y,x) sxy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syz(z,y,x) syz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxz(z,y,x) sxz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#endif

#if ND==2 || ND==21
#define vx(z,y,x)  vx[(x)*(NZ)+(z)]
#define vy(z,y,x)  vy[(x)*(NZ)+(z)]
#define vz(z,y,x)  vz[(x)*(NZ)+(z)]
#define sxx(z,y,x) sxx[(x)*(NZ)+(z)]
#define syy(z,y,x) syy[(x)*(NZ)+(z)]
#define szz(z,y,x) szz[(x)*(NZ)+(z)]
#define sxy(z,y,x) sxy[(x)*(NZ)+(z)]
#define syz(z,y,x) syz[(x)*(NZ)+(z)]
#define sxz(z,y,x) sxz[(x)*(NZ)+(z)]
#endif

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define sxxout(y,x) sxxout[(y)*NT+(x)]
#define syyout(y,x) syyout[(y)*NT+(x)]
#define szzout(y,x) szzout[(y)*NT+(x)]
#define sxyout(y,x) sxyout[(y)*NT+(x)]
#define sxzout(y,x) sxzout[(y)*NT+(x)]
#define syzout(y,x) syzout[(y)*NT+(x)]
#define pout(y,x) pout[(y)*NT+(x)]


#define rec_pos(y,x) rec_pos[(y)*8+(x)]


__kernel void seisout(__global float *vx,         __global float *vy,            __global float *vz,
                      __global float *sxx,        __global float *syy,           __global float *szz,
                      __global float *sxy,        __global float *syz,           __global float *sxz,
                      __global float *vxout,      __global float *vyout,         __global float *vzout,
                      __global float *sxxout,     __global float *syyout,        __global float *szzout,
                      __global float *sxyout,     __global float *syzout,        __global float *sxzout,
                      __global float *pout,       __global float *rec_pos,       int nt)
{
    
    int gid = get_global_id(0);
    
    int i=(int)(rec_pos(gid,0)/DH-0.5)+FDOH;
    int j=(int)(rec_pos(gid,1)/DH-0.5)+FDOH;
    int k=(int)(rec_pos(gid,2)/DH-0.5)+FDOH;
    
    if ( (i-OFFSET)>=FDOH && (i-OFFSET)<(NX-FDOH) ){
        
#if bcastvx==1
        vxout(gid, nt)= vx(k,j,i-OFFSET);
#endif
#if bcastvy==1
        vyout(gid, nt)= vy(k,j,i-OFFSET);
#endif
#if bcastvz==1
        vzout(gid, nt)= vz(k,j,i-OFFSET);
#endif
#if bcastsxx==1
        sxxout(gid, nt)= sxx(k,j,i-OFFSET);
#endif
#if bcastsyy==1
        syyout(gid, nt)= syy(k,j,i-OFFSET);
#endif
#if bcastszz==1
        szzout(gid, nt)= szz(k,j,i-OFFSET);
#endif
#if bcastsxy==1
        sxyout(gid, nt)= sxy(k,j,i-OFFSET);
#endif
#if bcastsxz==1
        sxzout(gid, nt)= sxz(k,j,i-OFFSET);
#endif
#if bcastsyz==1
        syzout(gid, nt)= syz(k,j,i-OFFSET);
#endif
#if bcastp==1
#if ND==3
        pout(gid, nt)= (sxx(k,j,i-OFFSET)+syy(k,j,i-OFFSET)+szz(k,j,i-OFFSET) )/3.0;
#endif
#if ND==2
        pout(gid, nt)= (sxx(k,j,i-OFFSET)+szz(k,j,i-OFFSET) )/2.0;
#endif
#endif
  
    }
    
}
