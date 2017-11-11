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

/*Adjoint update of the velocities in 2D SV*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define rip(z,x) rip[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define rkp(z,x) rkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define M(z,x) M[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define mu(z,x) mu[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]

#define sxx(z,x) sxx[(x)*NZ+(z)]
#define sxz(z,x) sxz[(x)*NZ+(z)]
#define szz(z,x) szz[(x)*NZ+(z)]
#define vx(z,x) vx[(x)*NZ+(z)]
#define vz(z,x) vz[(x)*NZ+(z)]

#define sxxr(z,x) sxxr[(x)*NZ+(z)]
#define sxzr(z,x) sxzr[(x)*NZ+(z)]
#define szzr(z,x) szzr[(x)*NZ+(z)]
#define vxr(z,x) vxr[(x)*NZ+(z)]
#define vzr(z,x) vzr[(x)*NZ+(z)]

#if LOCAL_OFF==0
#define lvar(z,x) lvar[(x)*2*lsizez+(z)]
#define lvar2(z,x) lvar2[(x)*lsizez+(z)]
#endif



#if FP16==1 || FP16==2

#define __h2f(x) __half2float((x))
#define __h22f2(x) __half22float2((x))
#define __f22h2(x) __float22half2_rn((x))

#else

#define __h2f(x) (x)
#define __h22f2(x) (x)
#define __f22h2(x) (x)

#endif

#if FP16==0

#define __prec float
#define __prec2 float2

#else

#define __prec half
#define __prec2 half2

#endif


#if FP16<3

#define __cprec float2
#define __f22h2c(x) (x)

extern "C" __device__ float2 add2(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x+b.x;
    output.y = a.y+b.y;
    return output;
}
extern "C" __device__ float2 mul2(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x*b.x;
    output.y = a.y*b.y;
    return output;
}
extern "C" __device__ float2 div2(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x/b.x;
    output.y = a.y/b.y;
    return output;
}
extern "C" __device__ float2 sub2(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x-b.x;
    output.y = a.y-b.y;
    return output;
}
extern "C" __device__ float2 f2h2(float a){
    
    float2 output={a,a};
    return output;
}

#else

#define __cprec half2
#define add2 __hadd2
#define mul2 __hmul2
#define mul2 __h2div
#define sub2 __hsub2
#define f2h2 __float2half2_rn
#define __f22h2c(x) __float22half2_rn((x))

#endif

extern "C" __device__ __prec2 __hp(__prec *a ){
    
    __prec2 output;
    *((__prec *)&output) = *a;
    *((__prec *)&output+1) = *(a+1);
    return output;
}

#if FP16==2 || FP16==4

#define __pprec half2

#else

#define __pprec float2

#endif

#if FP16==2

#define __pconv(x) __half22float2((x))

#elif FP16==3

#define __pconv(x) __float22half2_rn((x))

#else

#define __pconv(x) (x)

#endif


#define lbnd (FDOH+NAB)

#define gradrho(z,x) gradrho[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define gradmu(z,x) gradmu[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define gradM(z,x) gradM[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]


// Find boundary indice for boundary injection in backpropagation
extern "C" __device__ int evarm( int k, int i){
    
    
#if NUM_DEVICES==1 & NLOCALP==1

    int NXbnd = (NX-2*FDOH-2*NAB);
    int NZbnd = (NZ-FDOH-NAB);

    int m=-1;
    i-=lbnd;
    k-=lbnd/2;
    
    if ( (k>FDOH/2-1 && k<NZbnd-FDOH/2)  && (i>FDOH-1 && i<NXbnd-FDOH) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<FDOH){//front
        m=i*NZbnd+k;
    }
    else if (i>NXbnd-1-FDOH){//back
        i=i-NXbnd+FDOH;
        m=NZbnd*FDOH+i*NZbnd+k;
    }
    else if (k<FDOH/2){//up
        i=i-FDOH;
        m=NZbnd*FDOH*2+i+k*(NXbnd-2.0*FDOH);
    }
    else {//down
        i=i-FDOH;
        k=k-NZbnd+FDOH/2;
        m=NZbnd*FDOH*2+(NXbnd-2*FDOH)*FDOH/2+i+k*(NXbnd-2.0*FDOH);
    }
    
    

#elif DEVID==0 & MYGROUPID==0
    
    int NXbnd = (NX-2*FDOH-NAB);
    int NZbnd = (NZ-FDOH-NAB);
    
    int m=-1;
    i-=lbnd;
    k-=lbnd/2;
    
    if ( (k>FDOH/2-1 && k<NZbnd-FDOH/2)  && i>FDOH-1  )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<FDOH){//front
        m=i*NZbnd+k;
    }
    else if (k<FDOH/2){//up
        i=i-FDOH;
        m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
    }
    else {//down
        i=i-FDOH;
        k=k-NZbnd+FDOH/2;
        m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH+i+k*(NXbnd-FDOH);
    }

#elif DEVID==NUM_DEVICES-1 & MYGROUPID==NLOCALP-1
    int NXbnd = (NX-2*FDOH-NAB);
    int NZbnd = (NZ-FDOH-NAB);
    
    int m=-1;
    i-=FDOH;
    k-=lbnd/2;
    
    if ( (k>FDOH/2-1 && k<NZbnd-FDOH/2) && i<NXbnd-FDOH )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i>NXbnd-1 )
        m=-1;
    else if (i>NXbnd-1-FDOH){
        i=i-NXbnd+FDOH;
        m=i*NZbnd+k;
    }
    else if (k<FDOH/2){//up
        m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
    }
    else {//down
        k=k-NZbnd+FDOH/2;
        m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH+i+k*(NXbnd-FDOH);
    }
    
#else
    
    int NXbnd = (NX-2*FDOH);
    int NZbnd = (NZ-FDOH-NAB);
    
    int m=-1;
    i-=FDOH;
    k-=lbnd/2;
    
    if ( (k>FDOH/2-1 && k<NZbnd-FDOH/2) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (k<FDOH/2){//up
        m=i+k*(NXbnd);
    }
    else {//down
        k=k-NZbnd+FDOH/2;
        m=(NXbnd)*FDOH+i+k*(NXbnd);
    }
    
    
#endif
    
    
    return m;
 
}


extern "C" __global__ void update_adjv(int offcomm,
                           __pprec *rip, __pprec *rkp,
                           __prec2 *sxx,__prec2 *sxz,__prec2 *szz,
                           __prec2 *vx,__prec2 *vz,
                           __prec2 *sxxbnd,__prec2 *sxzbnd,__prec2 *szzbnd,
                           __prec2 *vxbnd,__prec2 *vzbnd,
                           __prec2 *sxxr,__prec2 *sxzr,__prec2 *szzr,
                           __prec2 *vxr,__prec2 *vzr,
                          float2 *gradrho)
{

    //Local memory
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;
    
    //Grid position
    int lsizez = blockDim.x+FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH/2;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/2;
    int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH+offcomm;
    
    //Define and load private parameters and variables
    __cprec lvx = __h22f2(vx(gidz,gidx));
    __cprec lvz = __h22f2(vz(gidz,gidx));
    __cprec lvxr = __h22f2(vxr(gidz,gidx));
    __cprec lvzr = __h22f2(vzr(gidz,gidx));
    __cprec lrip = __pconv(rip(gidz,gidx));
    __cprec lrkp = __pconv(rkp(gidz,gidx));
    
    //Define private derivatives
    __cprec sxx_x1;
    __cprec sxz_x2;
    __cprec sxz_z2;
    __cprec szz_z1;
    __cprec sxxr_x1;
    __cprec sxzr_x2;
    __cprec sxzr_z2;
    __cprec szzr_z1;
    
// If we use local memory
#if LOCAL_OFF==0

#define lsxx lvar
#define lszz lvar
#define lsxz lvar
#define lsxx2 lvar2
#define lszz2 lvar2
#define lsxz2 lvar2
    
#define lsxxr lvar
#define lszzr lvar
#define lsxzr lvar
#define lsxxr2 lvar2
#define lszzr2 lvar2
#define lsxzr2 lvar2
 
//// If local memory is turned off
//#elif LOCAL_OFF==1
//
//#define lsxx sxx
//#define lszz szz
//#define lsxz sxz
//
//#define lsxxr sxxr
//#define lszzr szzr
//#define lsxzr sxzr
//
//#define lidx gidx
//#define lidz gidz
//
//#define lsizez NZ
//#define lsizex NX
    
#endif
    
// Calculation of the stress spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
    {
#if LOCAL_OFF==0
        lsxx2(lidz,lidx)=sxx(gidz,gidx);
        if (lidx<2*FDOH)
            lsxx2(lidz,lidx-FDOH)=sxx(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxx2(lidz,lidx+lsizex-3*FDOH)=sxx(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxx2(lidz,lidx+FDOH)=sxx(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxx2(lidz,lidx-lsizex+3*FDOH)=sxx(gidz,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        sxx_x1=mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx))));
#elif FDOH == 2
        sxx_x1=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1)))));
#elif FDOH == 3
        sxx_x1=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidx+3)), __h22f2(lsxx2(lidz,lidx-2)))));
#elif FDOH == 4
        sxx_x1=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidx+3)), __h22f2(lsxx2(lidz,lidx-2))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsxx2(lidz,lidx+4)), __h22f2(lsxx2(lidz,lidx-3)))));
#elif FDOH == 5
        sxx_x1=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidx+3)), __h22f2(lsxx2(lidz,lidx-2))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsxx2(lidz,lidx+4)), __h22f2(lsxx2(lidz,lidx-3))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsxx2(lidz,lidx+5)), __h22f2(lsxx2(lidz,lidx-4)))));
#elif FDOH == 6
        sxx_x1=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidx+3)), __h22f2(lsxx2(lidz,lidx-2))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsxx2(lidz,lidx+4)), __h22f2(lsxx2(lidz,lidx-3))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsxx2(lidz,lidx+5)), __h22f2(lsxx2(lidz,lidx-4))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsxx2(lidz,lidx+6)), __h22f2(lsxx2(lidz,lidx-5)))));
#endif
        
#if LOCAL_OFF==0
        __syncthreads();
        lszz2(lidz,lidx)=szz(gidz,gidx);
        if (lidz<FDOH)
            lszz2(lidz-FDOH/2,lidx)=szz(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lszz2(lidz+FDOH/2,lidx)=szz(gidz+FDOH/2,gidx);
        __syncthreads();
#endif
        
#if   FDOH == 1
        szz_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx)))));
#elif FDOH == 2
        szz_z1=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                    mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx))))));
#elif FDOH == 3
        szz_z1=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                         mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx)))))),
                    mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidx))))));
#elif FDOH == 4
        szz_z1=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                              mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx)))))),
                         mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidx)))))),
                    mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszz(2*lidz+4,lidx))), __h22f2(__hp(&lszz(2*lidz-3,lidx))))));
#elif FDOH == 5
        szz_z1=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx)))))),
                              mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidx)))))),
                         mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszz(2*lidz+4,lidx))), __h22f2(__hp(&lszz(2*lidz-3,lidx)))))),
                    mul2( f2h2(HC5), sub2(__h22f2(__hp(&lszz(2*lidz+5,lidx))), __h22f2(__hp(&lszz(2*lidz-4,lidx))))));
#elif FDOH == 6
        szz_z1=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx)))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidx)))))),
                              mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszz(2*lidz+4,lidx))), __h22f2(__hp(&lszz(2*lidz-3,lidx)))))),
                         mul2( f2h2(HC5), sub2(__h22f2(__hp(&lszz(2*lidz+5,lidx))), __h22f2(__hp(&lszz(2*lidz-4,lidx)))))),
                    mul2( f2h2(HC6), sub2(__h22f2(__hp(&lszz(2*lidz+6,lidx))), __h22f2(__hp(&lszz(2*lidz-5,lidx))))));
#endif
        
#if LOCAL_OFF==0
        __syncthreads();
        lsxz2(lidz,lidx)=sxz(gidz,gidx);
        if (lidz<FDOH)
            lsxz2(lidz-FDOH/2,lidx)=sxz(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lsxz2(lidz+FDOH/2,lidx)=sxz(gidz+FDOH/2,gidx);
        if (lidx<2*FDOH)
            lsxz2(lidz,lidx-FDOH)=sxz(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxz2(lidz,lidx+lsizex-3*FDOH)=sxz(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxz2(lidz,lidx+FDOH)=sxz(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxz2(lidz,lidx-lsizex+3*FDOH)=sxz(gidz,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        sxz_x2=mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1))));
#elif FDOH == 2
        sxz_x2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2)))));
#elif FDOH == 3
        sxz_x2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidx+2)), __h22f2(lsxz2(lidz,lidx-3)))));
#elif FDOH == 4
        sxz_x2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidx+2)), __h22f2(lsxz2(lidz,lidx-3))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsxz2(lidz,lidx+3)), __h22f2(lsxz2(lidz,lidx-4)))));
#elif FDOH == 5
        sxz_x2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidx+2)), __h22f2(lsxz2(lidz,lidx-3))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsxz2(lidz,lidx+3)), __h22f2(lsxz2(lidz,lidx-4))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsxz2(lidz,lidx+4)), __h22f2(lsxz2(lidz,lidx-5)))));
#elif FDOH == 6
        sxz_x2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidx+2)), __h22f2(lsxz2(lidz,lidx-3))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsxz2(lidz,lidx+3)), __h22f2(lsxz2(lidz,lidx-4))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsxz2(lidz,lidx+4)), __h22f2(lsxz2(lidz,lidx-5))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsxz2(lidz,lidx+5)), __h22f2(lsxz2(lidz,lidx-6)))));
#endif
        
#if   FDOH == 1
        sxz_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx)))));
#elif FDOH == 2
        sxz_z2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                    mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx))))));
#elif FDOH == 3
        sxz_z2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                         mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx)))))),
                    mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidx))))));
#elif FDOH == 4
        sxz_z2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                              mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx)))))),
                         mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidx)))))),
                    mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxz(2*lidz+3,lidx))), __h22f2(__hp(&lsxz(2*lidz-4,lidx))))));
#elif FDOH == 5
        sxz_z2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx)))))),
                              mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidx)))))),
                         mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxz(2*lidz+3,lidx))), __h22f2(__hp(&lsxz(2*lidz-4,lidx)))))),
                    mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsxz(2*lidz+4,lidx))), __h22f2(__hp(&lsxz(2*lidz-5,lidx))))));
#elif FDOH == 6
        sxz_z2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx)))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidx)))))),
                              mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxz(2*lidz+3,lidx))), __h22f2(__hp(&lsxz(2*lidz-4,lidx)))))),
                         mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsxz(2*lidz+4,lidx))), __h22f2(__hp(&lsxz(2*lidz-5,lidx)))))),
                    mul2( f2h2(HC6), sub2(__h22f2(__hp(&lsxz(2*lidz+5,lidx))), __h22f2(__hp(&lsxz(2*lidz-6,lidx))))));
#endif
        __syncthreads();
}
#endif

#if LOCAL_OFF==0
    __syncthreads();
    lsxzr2(lidz,lidx)=sxzr(gidz,gidx);
    if (lidz<FDOH)
        lsxzr2(lidz-FDOH/2,lidx)=sxzr(gidz-FDOH/2,gidx);
    if (lidz>(lsizez-FDOH-1))
        lsxzr2(lidz+FDOH/2,lidx)=sxzr(gidz+FDOH/2,gidx);
    if (lidx<2*FDOH)
        lsxzr2(lidz,lidx-FDOH)=sxzr(gidz,gidx-FDOH);
    if (lidx+lsizex-3*FDOH<FDOH)
        lsxzr2(lidz,lidx+lsizex-3*FDOH)=sxzr(gidz,gidx+lsizex-3*FDOH);
    if (lidx>(lsizex-2*FDOH-1))
        lsxzr2(lidz,lidx+FDOH)=sxzr(gidz,gidx+FDOH);
    if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
        lsxzr2(lidz,lidx-lsizex+3*FDOH)=sxzr(gidz,gidx-lsizex+3*FDOH);
    __syncthreads();
#endif
    
#if   FDOH == 1
    sxzr_x2=mul2( f2h2(HC1), sub2(__h22f2(lsxzr2(lidz,lidx)), __h22f2(lsxzr2(lidz,lidx-1))));
#elif FDOH == 2
    sxzr_x2=add2(
                 mul2( f2h2(HC1), sub2(__h22f2(lsxzr2(lidz,lidx)), __h22f2(lsxzr2(lidz,lidx-1)))),
                 mul2( f2h2(HC2), sub2(__h22f2(lsxzr2(lidz,lidx+1)), __h22f2(lsxzr2(lidz,lidx-2)))));
#elif FDOH == 3
    sxzr_x2=add2(add2(
                      mul2( f2h2(HC1), sub2(__h22f2(lsxzr2(lidz,lidx)), __h22f2(lsxzr2(lidz,lidx-1)))),
                      mul2( f2h2(HC2), sub2(__h22f2(lsxzr2(lidz,lidx+1)), __h22f2(lsxzr2(lidz,lidx-2))))),
                 mul2( f2h2(HC3), sub2(__h22f2(lsxzr2(lidz,lidx+2)), __h22f2(lsxzr2(lidz,lidx-3)))));
#elif FDOH == 4
    sxzr_x2=add2(add2(add2(
                           mul2( f2h2(HC1), sub2(__h22f2(lsxzr2(lidz,lidx)), __h22f2(lsxzr2(lidz,lidx-1)))),
                           mul2( f2h2(HC2), sub2(__h22f2(lsxzr2(lidz,lidx+1)), __h22f2(lsxzr2(lidz,lidx-2))))),
                      mul2( f2h2(HC3), sub2(__h22f2(lsxzr2(lidz,lidx+2)), __h22f2(lsxzr2(lidz,lidx-3))))),
                 mul2( f2h2(HC4), sub2(__h22f2(lsxzr2(lidz,lidx+3)), __h22f2(lsxzr2(lidz,lidx-4)))));
#elif FDOH == 5
    sxzr_x2=add2(add2(add2(add2(
                                mul2( f2h2(HC1), sub2(__h22f2(lsxzr2(lidz,lidx)), __h22f2(lsxzr2(lidz,lidx-1)))),
                                mul2( f2h2(HC2), sub2(__h22f2(lsxzr2(lidz,lidx+1)), __h22f2(lsxzr2(lidz,lidx-2))))),
                           mul2( f2h2(HC3), sub2(__h22f2(lsxzr2(lidz,lidx+2)), __h22f2(lsxzr2(lidz,lidx-3))))),
                      mul2( f2h2(HC4), sub2(__h22f2(lsxzr2(lidz,lidx+3)), __h22f2(lsxzr2(lidz,lidx-4))))),
                 mul2( f2h2(HC5), sub2(__h22f2(lsxzr2(lidz,lidx+4)), __h22f2(lsxzr2(lidz,lidx-5)))));
#elif FDOH == 6
    sxzr_x2=add2(add2(add2(add2(add2(
                                     mul2( f2h2(HC1), sub2(__h22f2(lsxzr2(lidz,lidx)), __h22f2(lsxzr2(lidz,lidx-1)))),
                                     mul2( f2h2(HC2), sub2(__h22f2(lsxzr2(lidz,lidx+1)), __h22f2(lsxzr2(lidz,lidx-2))))),
                                mul2( f2h2(HC3), sub2(__h22f2(lsxzr2(lidz,lidx+2)), __h22f2(lsxzr2(lidz,lidx-3))))),
                           mul2( f2h2(HC4), sub2(__h22f2(lsxzr2(lidz,lidx+3)), __h22f2(lsxzr2(lidz,lidx-4))))),
                      mul2( f2h2(HC5), sub2(__h22f2(lsxzr2(lidz,lidx+4)), __h22f2(lsxzr2(lidz,lidx-5))))),
                 mul2( f2h2(HC6), sub2(__h22f2(lsxzr2(lidz,lidx+5)), __h22f2(lsxzr2(lidz,lidx-6)))));
#endif
    
#if   FDOH == 1
    sxzr_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxzr(2*lidz,lidx))), __h22f2(__hp(&lsxzr(2*lidz-1,lidx)))));
#elif FDOH == 2
    sxzr_z2=add2(
                 mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxzr(2*lidz,lidx))), __h22f2(__hp(&lsxzr(2*lidz-1,lidx))))),
                 mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxzr(2*lidz+1,lidx))), __h22f2(__hp(&lsxzr(2*lidz-2,lidx))))));
#elif FDOH == 3
    sxzr_z2=add2(add2(
                      mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxzr(2*lidz,lidx))), __h22f2(__hp(&lsxzr(2*lidz-1,lidx))))),
                      mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxzr(2*lidz+1,lidx))), __h22f2(__hp(&lsxzr(2*lidz-2,lidx)))))),
                 mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxzr(2*lidz+2,lidx))), __h22f2(__hp(&lsxzr(2*lidz-3,lidx))))));
#elif FDOH == 4
    sxzr_z2=add2(add2(add2(
                           mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxzr(2*lidz,lidx))), __h22f2(__hp(&lsxzr(2*lidz-1,lidx))))),
                           mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxzr(2*lidz+1,lidx))), __h22f2(__hp(&lsxzr(2*lidz-2,lidx)))))),
                      mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxzr(2*lidz+2,lidx))), __h22f2(__hp(&lsxzr(2*lidz-3,lidx)))))),
                 mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxzr(2*lidz+3,lidx))), __h22f2(__hp(&lsxzr(2*lidz-4,lidx))))));
#elif FDOH == 5
    sxzr_z2=add2(add2(add2(add2(
                                mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxzr(2*lidz,lidx))), __h22f2(__hp(&lsxzr(2*lidz-1,lidx))))),
                                mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxzr(2*lidz+1,lidx))), __h22f2(__hp(&lsxzr(2*lidz-2,lidx)))))),
                           mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxzr(2*lidz+2,lidx))), __h22f2(__hp(&lsxzr(2*lidz-3,lidx)))))),
                      mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxzr(2*lidz+3,lidx))), __h22f2(__hp(&lsxzr(2*lidz-4,lidx)))))),
                 mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsxzr(2*lidz+4,lidx))), __h22f2(__hp(&lsxzr(2*lidz-5,lidx))))));
#elif FDOH == 6
    sxzr_z2=add2(add2(add2(add2(add2(
                                     mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxzr(2*lidz,lidx))), __h22f2(__hp(&lsxzr(2*lidz-1,lidx))))),
                                     mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxzr(2*lidz+1,lidx))), __h22f2(__hp(&lsxzr(2*lidz-2,lidx)))))),
                                mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxzr(2*lidz+2,lidx))), __h22f2(__hp(&lsxzr(2*lidz-3,lidx)))))),
                           mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxzr(2*lidz+3,lidx))), __h22f2(__hp(&lsxzr(2*lidz-4,lidx)))))),
                      mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsxzr(2*lidz+4,lidx))), __h22f2(__hp(&lsxzr(2*lidz-5,lidx)))))),
                 mul2( f2h2(HC6), sub2(__h22f2(__hp(&lsxzr(2*lidz+5,lidx))), __h22f2(__hp(&lsxzr(2*lidz-6,lidx))))));
#endif
    
#if LOCAL_OFF==0
    __syncthreads();
    lsxxr2(lidz,lidx)=sxxr(gidz,gidx);
    if (lidx<2*FDOH)
        lsxxr2(lidz,lidx-FDOH)=sxxr(gidz,gidx-FDOH);
    if (lidx+lsizex-3*FDOH<FDOH)
        lsxxr2(lidz,lidx+lsizex-3*FDOH)=sxxr(gidz,gidx+lsizex-3*FDOH);
    if (lidx>(lsizex-2*FDOH-1))
        lsxxr2(lidz,lidx+FDOH)=sxxr(gidz,gidx+FDOH);
    if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
        lsxxr2(lidz,lidx-lsizex+3*FDOH)=sxxr(gidz,gidx-lsizex+3*FDOH);
    __syncthreads();
#endif
    
#if   FDOH == 1
    sxxr_x1=mul2( f2h2(HC1), sub2(__h22f2(lsxxr2(lidz,lidx+1)), __h22f2(lsxxr2(lidz,lidx))));
#elif FDOH == 2
    sxxr_x1=add2(
                 mul2( f2h2(HC1), sub2(__h22f2(lsxxr2(lidz,lidx+1)), __h22f2(lsxxr2(lidz,lidx)))),
                 mul2( f2h2(HC2), sub2(__h22f2(lsxxr2(lidz,lidx+2)), __h22f2(lsxxr2(lidz,lidx-1)))));
#elif FDOH == 3
    sxxr_x1=add2(add2(
                      mul2( f2h2(HC1), sub2(__h22f2(lsxxr2(lidz,lidx+1)), __h22f2(lsxxr2(lidz,lidx)))),
                      mul2( f2h2(HC2), sub2(__h22f2(lsxxr2(lidz,lidx+2)), __h22f2(lsxxr2(lidz,lidx-1))))),
                 mul2( f2h2(HC3), sub2(__h22f2(lsxxr2(lidz,lidx+3)), __h22f2(lsxxr2(lidz,lidx-2)))));
#elif FDOH == 4
    sxxr_x1=add2(add2(add2(
                           mul2( f2h2(HC1), sub2(__h22f2(lsxxr2(lidz,lidx+1)), __h22f2(lsxxr2(lidz,lidx)))),
                           mul2( f2h2(HC2), sub2(__h22f2(lsxxr2(lidz,lidx+2)), __h22f2(lsxxr2(lidz,lidx-1))))),
                      mul2( f2h2(HC3), sub2(__h22f2(lsxxr2(lidz,lidx+3)), __h22f2(lsxxr2(lidz,lidx-2))))),
                 mul2( f2h2(HC4), sub2(__h22f2(lsxxr2(lidz,lidx+4)), __h22f2(lsxxr2(lidz,lidx-3)))));
#elif FDOH == 5
    sxxr_x1=add2(add2(add2(add2(
                                mul2( f2h2(HC1), sub2(__h22f2(lsxxr2(lidz,lidx+1)), __h22f2(lsxxr2(lidz,lidx)))),
                                mul2( f2h2(HC2), sub2(__h22f2(lsxxr2(lidz,lidx+2)), __h22f2(lsxxr2(lidz,lidx-1))))),
                           mul2( f2h2(HC3), sub2(__h22f2(lsxxr2(lidz,lidx+3)), __h22f2(lsxxr2(lidz,lidx-2))))),
                      mul2( f2h2(HC4), sub2(__h22f2(lsxxr2(lidz,lidx+4)), __h22f2(lsxxr2(lidz,lidx-3))))),
                 mul2( f2h2(HC5), sub2(__h22f2(lsxxr2(lidz,lidx+5)), __h22f2(lsxxr2(lidz,lidx-4)))));
#elif FDOH == 6
    sxxr_x1=add2(add2(add2(add2(add2(
                                     mul2( f2h2(HC1), sub2(__h22f2(lsxxr2(lidz,lidx+1)), __h22f2(lsxxr2(lidz,lidx)))),
                                     mul2( f2h2(HC2), sub2(__h22f2(lsxxr2(lidz,lidx+2)), __h22f2(lsxxr2(lidz,lidx-1))))),
                                mul2( f2h2(HC3), sub2(__h22f2(lsxxr2(lidz,lidx+3)), __h22f2(lsxxr2(lidz,lidx-2))))),
                           mul2( f2h2(HC4), sub2(__h22f2(lsxxr2(lidz,lidx+4)), __h22f2(lsxxr2(lidz,lidx-3))))),
                      mul2( f2h2(HC5), sub2(__h22f2(lsxxr2(lidz,lidx+5)), __h22f2(lsxxr2(lidz,lidx-4))))),
                 mul2( f2h2(HC6), sub2(__h22f2(lsxxr2(lidz,lidx+6)), __h22f2(lsxxr2(lidz,lidx-5)))));
#endif
    
#if LOCAL_OFF==0
    __syncthreads();
    lszzr2(lidz,lidx)=szzr(gidz,gidx);
    if (lidz<FDOH)
        lszzr2(lidz-FDOH/2,lidx)=szzr(gidz-FDOH/2,gidx);
    if (lidz>(lsizez-FDOH-1))
        lszzr2(lidz+FDOH/2,lidx)=szzr(gidz+FDOH/2,gidx);
    __syncthreads();
#endif
    
#if   FDOH == 1
    szzr_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszzr(2*lidz+1,lidx))), __h22f2(__hp(&lszzr(2*lidz,lidx)))));
#elif FDOH == 2
    szzr_z1=add2(
                 mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszzr(2*lidz+1,lidx))), __h22f2(__hp(&lszzr(2*lidz,lidx))))),
                 mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszzr(2*lidz+2,lidx))), __h22f2(__hp(&lszzr(2*lidz-1,lidx))))));
#elif FDOH == 3
    szzr_z1=add2(add2(
                      mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszzr(2*lidz+1,lidx))), __h22f2(__hp(&lszzr(2*lidz,lidx))))),
                      mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszzr(2*lidz+2,lidx))), __h22f2(__hp(&lszzr(2*lidz-1,lidx)))))),
                 mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszzr(2*lidz+3,lidx))), __h22f2(__hp(&lszzr(2*lidz-2,lidx))))));
#elif FDOH == 4
    szzr_z1=add2(add2(add2(
                           mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszzr(2*lidz+1,lidx))), __h22f2(__hp(&lszzr(2*lidz,lidx))))),
                           mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszzr(2*lidz+2,lidx))), __h22f2(__hp(&lszzr(2*lidz-1,lidx)))))),
                      mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszzr(2*lidz+3,lidx))), __h22f2(__hp(&lszzr(2*lidz-2,lidx)))))),
                 mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszzr(2*lidz+4,lidx))), __h22f2(__hp(&lszzr(2*lidz-3,lidx))))));
#elif FDOH == 5
    szzr_z1=add2(add2(add2(add2(
                                mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszzr(2*lidz+1,lidx))), __h22f2(__hp(&lszzr(2*lidz,lidx))))),
                                mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszzr(2*lidz+2,lidx))), __h22f2(__hp(&lszzr(2*lidz-1,lidx)))))),
                           mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszzr(2*lidz+3,lidx))), __h22f2(__hp(&lszzr(2*lidz-2,lidx)))))),
                      mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszzr(2*lidz+4,lidx))), __h22f2(__hp(&lszzr(2*lidz-3,lidx)))))),
                 mul2( f2h2(HC5), sub2(__h22f2(__hp(&lszzr(2*lidz+5,lidx))), __h22f2(__hp(&lszzr(2*lidz-4,lidx))))));
#elif FDOH == 6
    szzr_z1=add2(add2(add2(add2(add2(
                                     mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszzr(2*lidz+1,lidx))), __h22f2(__hp(&lszzr(2*lidz,lidx))))),
                                     mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszzr(2*lidz+2,lidx))), __h22f2(__hp(&lszzr(2*lidz-1,lidx)))))),
                                mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszzr(2*lidz+3,lidx))), __h22f2(__hp(&lszzr(2*lidz-2,lidx)))))),
                           mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszzr(2*lidz+4,lidx))), __h22f2(__hp(&lszzr(2*lidz-3,lidx)))))),
                      mul2( f2h2(HC5), sub2(__h22f2(__hp(&lszzr(2*lidz+5,lidx))), __h22f2(__hp(&lszzr(2*lidz-4,lidx)))))),
                 mul2( f2h2(HC6), sub2(__h22f2(__hp(&lszzr(2*lidz+6,lidx))), __h22f2(__hp(&lszzr(2*lidz-5,lidx))))));
#endif


    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
    // To stop updating if we are outside the model (global id must be amultiple of local id in OpenCL, hence we stop if we have a global idoutside the grid)
#if  LOCAL_OFF==0
#if COMM12==0
    if ( gidz>(NZ-FDOH/2-1) ||  (gidx-offcomm)>(NX-FDOH-1-LCOMM) )
        return;
#else
    if ( gidz>(NZ-FDOH/2-1)  )
        return;
#endif
#endif


// Backpropagate the forward velocity
#if BACK_PROP_TYPE==1
    {
        
        // Update the variables
        lvx=sub2(lvx,mul2(add2(sxx_x1,sxz_z2),lrip));
        lvz=sub2(lvz,mul2(add2(szz_z1,sxz_x2),lrkp));
        
        // Inject the boundary values
//        int m=evarm(gidz, gidx);
//        if (m!=-1){
//            lvx= __h22f2(vxbnd[m]);
//            lvz= __h22f2(vzbnd[m]);
//        }

        
        //Write updated values to global memory
//        vx(gidz,gidx) = __f22h2(lvx);
//        vz(gidz,gidx) = __f22h2(lvz);
        

    }
#endif

    // Update the variables
    lvxr=add2(lvxr,mul2(add2(sxxr_x1,sxzr_z2),lrip));
    lvzr=add2(lvzr,mul2(add2(szzr_z1,sxzr_x2),lrkp));
    //Write updated values to global memory
    vxr(gidz,gidx) = __f22h2(lvxr);
    vzr(gidz,gidx) = __f22h2(lvzr);
    
    
// Density gradient calculation on the fly
#if BACK_PROP_TYPE==1
    lvxr=mul2(add2(sxxr_x1,sxzr_z2),lrip);
    lvzr=mul2(add2(szzr_z1,sxzr_x2),lrkp);
//    gradrho(gidz,gidx)=sub2(sub2(gradrho(gidz,gidx), mul2(lvx, lvxr)), mul2(lvz,lvzr));
    
    
    gradrho(gidz,gidx)=vxr(gidz,gidx);//sub2(gradrho(gidz,gidx), add2(mul2(lvx, lvx), mul2(lvz,lvz)));
#endif

}

