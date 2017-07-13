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

/*Calculate the DFT of the wavefields */


__kernel void savefreqs(__global float *vx,         __global float *vy,            __global float *vz,
                       __global float *sxx,        __global float *syy,           __global float *szz,
                       __global float *sxy,        __global float *syz,           __global float *sxz,
                       __global float2 *fvx,        __global float2 *fvy,           __global float2 *fvz,
                       __global float2 *fsxx,       __global float2 *fsyy,          __global float2 *fszz,
                       __global float2 *fsxy,       __global float2 *fsyz,          __global float2 *fsxz,
                       __global float *rxx,        __global float *ryy,           __global float *rzz,
                       __global float *rxy,        __global float *ryz,           __global float *rxz,
                       __global float2 *frxx,       __global float2 *fryy,          __global float2 *frzz,
                       __global float2 *frxy,       __global float2 *fryz,          __global float2 *frxz,
                      __constant float *freqs,  int nt)

{
    
    int freq,l;
    float2 fact=0;
    float lvx, lvy, lvz, lsxx, lsyy, lszz, lsxy, lsyz, lsxz;
    
    int gid = get_global_id(0);
    int gsize=get_global_size(0);


#if ND!=21
    lvx=vx[gid];
    lvz=vz[gid];
    lsxx=sxx[gid];
    lszz=szz[gid];
    lsxz=sxz[gid];
#endif
#if ND==3 || ND==21
    lsxy=sxy[gid];
    lsyz=syz[gid];
    lvy=vx[gid];
#endif
#if ND==3
    lsyy=syy[gid];
#endif
    
#if LVE>0
    float lrxx[LVE], lryy[LVE], lrzz[LVE], lrxy[LVE], lryz[LVE], lrxz[LVE];
#pragma unroll    
    for (l=0;l<LVE;l++){
#if ND!=21
        lrxx[l]=rxx[gid+l*gsize];
        lrzz[l]=rzz[gid+l*gsize];
        lrxz[l]=rxz[gid+l*gsize];
#endif
#if ND==3 || ND==21
        lrxy[l]=rxy[gid+l*gsize];
        lryz[l]=ryz[gid+l*gsize];
#endif
#if ND==3
        lryy[l]=ryy[gid+l*gsize];
#endif

    }
#endif
    
    
#pragma unroll
    for (freq=0;freq<NFREQS;freq++){
        fact.x =  DTNYQ*DT*cospi(2.0*freqs[freq]*nt/NTNYQ);
        fact.y = -DTNYQ*DT*sinpi(2.0*freqs[freq]*nt/NTNYQ);
        
#if ND!=21
        fvx[gid+freq*gsize]+=fact*lvx;
        fvz[gid+freq*gsize]+=fact*lvz;
        fsxx[gid+freq*gsize]+=fact*lsxx;
        fszz[gid+freq*gsize]+=fact*lszz;
        fsxz[gid+freq*gsize]+=fact*lsxz;
#endif
#if ND==3 || ND==21
        fvy[gid+freq*gsize]+=fact*lvy;
        fsxy[gid+freq*gsize]+=fact*lsxy;
        fsyz[gid+freq*gsize]+=fact*lsyz;
#endif
#if ND==3
        fsyy[gid+freq*gsize]+=fact*lsyy;
#endif


        
#if LVE>0
//for crosscorrelation, stresses and memory variables must sampled at the same time step
#if DIRPROP==0
        fact.x = 0.5*( fact.x+ DTNYQ*DT*cospi(2.0*freqs[freq]*(nt+1.0)/NTNYQ) );
        fact.y = 0.5*( fact.y- DTNYQ*DT*sinpi(2.0*freqs[freq]*(nt+1.0)/NTNYQ) );
#endif
#if DIRPROP==1
        fact.x = 0.5*( fact.x+ DTNYQ*DT*cospi(2.0*freqs[freq]*(nt-1.0)/NTNYQ) );
        fact.y = 0.5*( fact.y- DTNYQ*DT*sinpi(2.0*freqs[freq]*(nt-1.0)/NTNYQ) );
#endif
#pragma unroll
        for (l=0;l<LVE;l++){
#if ND!=21
            frxx[gid+l*gsize+freq*gsize*LVE]+=fact*lrxx[l];
            frzz[gid+l*gsize+freq*gsize*LVE]+=fact*lrzz[l];
            frxz[gid+l*gsize+freq*gsize*LVE]+=fact*lrxz[l];
#endif
#if ND==3 || ND==21
            frxy[gid+l*gsize+freq*gsize*LVE]+=fact*lrxy[l];
            fryz[gid+l*gsize+freq*gsize*LVE]+=fact*lryz[l];
#endif
#if ND==3
            fryy[gid+l*gsize+freq*gsize*LVE]+=fact*lryy[l];
#endif
        }
#endif
    

    }
    
    
    
}

