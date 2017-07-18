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
#define rip(z,y,x)     rip[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rjp(z,y,x)     rjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,y,x)     rkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define vx(z,y,x)   vx[(x)*NY*NZ+(y)*NZ+(z)]
#define vy(z,y,x)   vy[(x)*NY*NZ+(y)*NZ+(z)]
#define vz(z,y,x)   vz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxx(z,y,x) sxx[(x)*NY*NZ+(y)*NZ+(z)]
#define syy(z,y,x) syy[(x)*NY*NZ+(y)*NZ+(z)]
#define szz(z,y,x) szz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxy(z,y,x) sxy[(x)*NY*NZ+(y)*NZ+(z)]
#define syz(z,y,x) syz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxz(z,y,x) sxz[(x)*NY*NZ+(y)*NZ+(z)]
#endif

#if ND==2 || ND==21
#define rip(z,y,x)    rip[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rjp(z,y,x)    rjp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,y,x)    rkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define vx(z,y,x)  vx[(x)*NZ+(z)]
#define vy(z,y,x)  vy[(x)*NZ+(z)]
#define vz(z,y,x)  vz[(x)*NZ+(z)]
#define sxx(z,y,x) sxx[(x)*NZ+(z)]
#define syy(z,y,x) syy[(x)*NZ+(z)]
#define szz(z,y,x) szz[(x)*NZ+(z)]
#define sxy(z,y,x) sxy[(x)*NZ+(z)]
#define syz(z,y,x) syz[(x)*NZ+(z)]
#define sxz(z,y,x) sxz[(x)*NZ+(z)]
#endif



#define src_pos(y,x) src_pos[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]


__kernel void sources(__global float *vx,         __global float *vy,            __global float *vz,
                      __global float *sxx,        __global float *syy,           __global float *szz,
                      __global float *sxy,        __global float *syz,           __global float *sxz,
                      __global float *rip,        __global float *rjp,           __global float *rkp,
                      __global float *src_pos,    __global float *signals,        int nt, int signprop)
{
    
    int gid = get_global_id(0);
    int nsrc = get_global_size(0);
    
    int i=(int)(src_pos(0,gid)/DH-0.5)+fdoh;
    int j=(int)(src_pos(1,gid)/DH-0.5)+fdoh;
    int k=(int)(src_pos(2,gid)/DH-0.5)+fdoh;
    
    int SOURCE_TYPE= src_pos(4,gid);
    
    
    float amp=signprop*(DT*signals(gid,nt))/(DH*DH*DH); // scaled force amplitude with F= 1N
    
    
    if(i-offset<fdoh || i-offset>NX-fdoh-1){
        return;
    }
    
#if ND!=21
    if (SOURCE_TYPE==1){
        /* pressure source */
        sxx(k,j,i)+=amp;
#if ND==3
        syy(k,j,i)+=amp;
#endif
        szz(k,j,i)+=amp;
    }
#endif
#if ND!=21
    if (SOURCE_TYPE==2){
        /* single force in x */
        vx(k,j,i)  +=  amp/rip(k,j,i-offset);
    }
#endif
#if ND!=2
    if (SOURCE_TYPE==3){
        /* single force in y */
        
        vy(k,j,i)  +=  amp/rjp(k,j,i-offset);
    }
#endif
#if ND!=21
    if (SOURCE_TYPE==4){
        /* single force in z */
        
        vz(k,j,i)  +=  amp/rkp(k,j,i-offset);
    }
#endif
    
    
}
