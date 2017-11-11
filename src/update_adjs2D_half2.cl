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

/*Adjoint update of the stresses in 2D SV*/

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

extern "C" __global__ void update_adjs(int offcomm,
                           __pprec *muipkp, __pprec *M, __pprec *mu,
                           __prec2 *sxx,__prec2 *sxz,__prec2 *szz,
                           __prec2 *vx,__prec2 *vz,
                           __prec2 *sxxbnd,__prec2 *sxzbnd,__prec2 *szzbnd,
                           __prec2 *vxbnd,__prec2 *vzbnd,
                           __prec2 *sxxr,__prec2 *sxzr,__prec2 *szzr,
                           __prec2 *vxr,__prec2 *vzr,
                          float2 *gradrho,    float2 *gradM,     float2 *gradmu)
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
    __cprec lsxx = __h22f2(sxx(gidz,gidx));
    __cprec lsxz = __h22f2(sxz(gidz,gidx));
    __cprec lszz = __h22f2(szz(gidz,gidx));
    __cprec lsxxr = __h22f2(sxxr(gidz,gidx));
    __cprec lsxzr = __h22f2(sxzr(gidz,gidx));
    __cprec lszzr = __h22f2(szzr(gidz,gidx));
    __cprec lM = __pconv(M(gidz,gidx));
    __cprec lmu = __pconv(mu(gidz,gidx));
    __cprec lmuipkp = __pconv(muipkp(gidz,gidx));
    
    //Define private derivatives
    __cprec vx_x2;
    __cprec vx_z1;
    __cprec vz_x1;
    __cprec vz_z2;
    __cprec vxr_x2;
    __cprec vxr_z1;
    __cprec vzr_x1;
    __cprec vzr_z2;

    
    
// If we use local memory
#if LOCAL_OFF==0
#define lvx lvar
#define lvz lvar
#define lvx2 lvar2
#define lvz2 lvar2
    
#define lvxr lvar
#define lvzr lvar
#define lvxr2 lvar2
#define lvzr2 lvar2

//// If local memory is turned off
//#elif LOCAL_OFF==1
//
//#define lvx_r vx_r
//#define lvz_r vz_r
//#define lvx vx
//#define lvz vz
//#define lidx gidx
//#define lidz gidz
//
//#define lsizez NZ
//#define lsizex NX
    
#endif
    
// Calculation of the velocity spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
    {
#if LOCAL_OFF==0
        __syncthreads();
        lvx2(lidz,lidx)=vx(gidz,gidx);
        if (lidz<FDOH)
            lvx2(lidz-FDOH/2,lidx)=vx(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lvx2(lidz+FDOH/2,lidx)=vx(gidz+FDOH/2,gidx);
        if (lidx<2*FDOH)
            lvx2(lidz,lidx-FDOH)=vx(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvx2(lidz,lidx+lsizex-3*FDOH)=vx(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvx2(lidz,lidx+FDOH)=vx(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvx2(lidz,lidx-lsizex+3*FDOH)=vx(gidz,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        vx_x2=mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1))));
#elif FDOH == 2
        vx_x2=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2)))));
#elif FDOH == 3
        vx_x2=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
                   mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3)))));
#elif FDOH == 4
        vx_x2=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3))))),
                   mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidx+3)), __h22f2(lvx2(lidz,lidx-4)))));
#elif FDOH == 5
        vx_x2=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3))))),
                        mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidx+3)), __h22f2(lvx2(lidz,lidx-4))))),
                   mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidx+4)), __h22f2(lvx2(lidz,lidx-5)))));
#elif FDOH == 6
        vx_x2=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidx+3)), __h22f2(lvx2(lidz,lidx-4))))),
                        mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidx+4)), __h22f2(lvx2(lidz,lidx-5))))),
                   mul2( f2h2(HC6), sub2(__h22f2(lvx2(lidz,lidx+5)), __h22f2(lvx2(lidz,lidx-6)))));
#endif
        
#if   FDOH == 1
        vx_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx)))));
#elif FDOH == 2
        vx_z1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx))))));
#elif FDOH == 3
        vx_z1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx))))));
#elif FDOH == 4
        vx_z1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                             mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx)))))),
                   mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidx))))));
#elif FDOH == 5
        vx_z1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                                  mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
                             mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx)))))),
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidx)))))),
                   mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvx(2*lidz+5,lidx))), __h22f2(__hp(&lvx(2*lidz-4,lidx))))));
#elif FDOH == 6
        vx_z1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                                       mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx)))))),
                             mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidx)))))),
                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvx(2*lidz+5,lidx))), __h22f2(__hp(&lvx(2*lidz-4,lidx)))))),
                   mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvx(2*lidz+6,lidx))), __h22f2(__hp(&lvx(2*lidz-5,lidx))))));
#endif
        
#if LOCAL_OFF==0
        __syncthreads();
        lvz2(lidz,lidx)=vz(gidz,gidx);
        if (lidz<FDOH)
            lvz2(lidz-FDOH/2,lidx)=vz(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lvz2(lidz+FDOH/2,lidx)=vz(gidz+FDOH/2,gidx);
        if (lidx<2*FDOH)
            lvz2(lidz,lidx-FDOH)=vz(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvz2(lidz,lidx+lsizex-3*FDOH)=vz(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvz2(lidz,lidx+FDOH)=vz(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvz2(lidz,lidx-lsizex+3*FDOH)=vz(gidz,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        vz_x1=mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx))));
#elif FDOH == 2
        vz_x1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1)))));
#elif FDOH == 3
        vz_x1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
                   mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2)))));
#elif FDOH == 4
        vz_x1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2))))),
                   mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidx+4)), __h22f2(lvz2(lidz,lidx-3)))));
#elif FDOH == 5
        vz_x1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2))))),
                        mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidx+4)), __h22f2(lvz2(lidz,lidx-3))))),
                   mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidx+5)), __h22f2(lvz2(lidz,lidx-4)))));
#elif FDOH == 6
        vz_x1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidx+4)), __h22f2(lvz2(lidz,lidx-3))))),
                        mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidx+5)), __h22f2(lvz2(lidz,lidx-4))))),
                   mul2( f2h2(HC6), sub2(__h22f2(lvz2(lidz,lidx+6)), __h22f2(lvz2(lidz,lidx-5)))));
#endif
        
#if   FDOH == 1
        vz_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx)))));
#elif FDOH == 2
        vz_z2=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx))))));
#elif FDOH == 3
        vz_z2=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx))))));
#elif FDOH == 4
        vz_z2=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                             mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx)))))),
                   mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidx))))));
#elif FDOH == 5
        vz_z2=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                                  mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
                             mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx)))))),
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidx)))))),
                   mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvz(2*lidz+4,lidx))), __h22f2(__hp(&lvz(2*lidz-5,lidx))))));
#elif FDOH == 6
        vz_z2=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                                       mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx)))))),
                             mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidx)))))),
                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvz(2*lidz+4,lidx))), __h22f2(__hp(&lvz(2*lidz-5,lidx)))))),
                   mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvz(2*lidz+5,lidx))), __h22f2(__hp(&lvz(2*lidz-6,lidx))))));
#endif
        
        __syncthreads();
    }
#endif
    
// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
#if LOCAL_OFF==0
        lvxr2(lidz,lidx)=vxr(gidz,gidx);
        if (lidz<FDOH)
            lvxr2(lidz-FDOH/2,lidx)=vxr(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lvxr2(lidz+FDOH/2,lidx)=vxr(gidz+FDOH/2,gidx);
        if (lidx<2*FDOH)
            lvxr2(lidz,lidx-FDOH)=vxr(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvxr2(lidz,lidx+lsizex-3*FDOH)=vxr(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvxr2(lidz,lidx+FDOH)=vxr(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvxr2(lidz,lidx-lsizex+3*FDOH)=vxr(gidz,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        vxr_x2=mul2( f2h2(HC1), sub2(__h22f2(lvxr2(lidz,lidx)), __h22f2(lvxr2(lidz,lidx-1))));
#elif FDOH == 2
        vxr_x2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lvxr2(lidz,lidx)), __h22f2(lvxr2(lidz,lidx-1)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lvxr2(lidz,lidx+1)), __h22f2(lvxr2(lidz,lidx-2)))));
#elif FDOH == 3
        vxr_x2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lvxr2(lidz,lidx)), __h22f2(lvxr2(lidz,lidx-1)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lvxr2(lidz,lidx+1)), __h22f2(lvxr2(lidz,lidx-2))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lvxr2(lidz,lidx+2)), __h22f2(lvxr2(lidz,lidx-3)))));
#elif FDOH == 4
        vxr_x2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lvxr2(lidz,lidx)), __h22f2(lvxr2(lidz,lidx-1)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lvxr2(lidz,lidx+1)), __h22f2(lvxr2(lidz,lidx-2))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lvxr2(lidz,lidx+2)), __h22f2(lvxr2(lidz,lidx-3))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lvxr2(lidz,lidx+3)), __h22f2(lvxr2(lidz,lidx-4)))));
#elif FDOH == 5
        vxr_x2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lvxr2(lidz,lidx)), __h22f2(lvxr2(lidz,lidx-1)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lvxr2(lidz,lidx+1)), __h22f2(lvxr2(lidz,lidx-2))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lvxr2(lidz,lidx+2)), __h22f2(lvxr2(lidz,lidx-3))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lvxr2(lidz,lidx+3)), __h22f2(lvxr2(lidz,lidx-4))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lvxr2(lidz,lidx+4)), __h22f2(lvxr2(lidz,lidx-5)))));
#elif FDOH == 6
        vxr_x2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lvxr2(lidz,lidx)), __h22f2(lvxr2(lidz,lidx-1)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lvxr2(lidz,lidx+1)), __h22f2(lvxr2(lidz,lidx-2))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lvxr2(lidz,lidx+2)), __h22f2(lvxr2(lidz,lidx-3))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lvxr2(lidz,lidx+3)), __h22f2(lvxr2(lidz,lidx-4))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lvxr2(lidz,lidx+4)), __h22f2(lvxr2(lidz,lidx-5))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lvxr2(lidz,lidx+5)), __h22f2(lvxr2(lidz,lidx-6)))));
#endif
        
#if   FDOH == 1
        vxr_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvxr(2*lidz+1,lidx))), __h22f2(__hp(&lvxr(2*lidz,lidx)))));
#elif FDOH == 2
        vxr_z1=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvxr(2*lidz+1,lidx))), __h22f2(__hp(&lvxr(2*lidz,lidx))))),
                    mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvxr(2*lidz+2,lidx))), __h22f2(__hp(&lvxr(2*lidz-1,lidx))))));
#elif FDOH == 3
        vxr_z1=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvxr(2*lidz+1,lidx))), __h22f2(__hp(&lvxr(2*lidz,lidx))))),
                         mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvxr(2*lidz+2,lidx))), __h22f2(__hp(&lvxr(2*lidz-1,lidx)))))),
                    mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvxr(2*lidz+3,lidx))), __h22f2(__hp(&lvxr(2*lidz-2,lidx))))));
#elif FDOH == 4
        vxr_z1=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvxr(2*lidz+1,lidx))), __h22f2(__hp(&lvxr(2*lidz,lidx))))),
                              mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvxr(2*lidz+2,lidx))), __h22f2(__hp(&lvxr(2*lidz-1,lidx)))))),
                         mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvxr(2*lidz+3,lidx))), __h22f2(__hp(&lvxr(2*lidz-2,lidx)))))),
                    mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvxr(2*lidz+4,lidx))), __h22f2(__hp(&lvxr(2*lidz-3,lidx))))));
#elif FDOH == 5
        vxr_z1=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvxr(2*lidz+1,lidx))), __h22f2(__hp(&lvxr(2*lidz,lidx))))),
                                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvxr(2*lidz+2,lidx))), __h22f2(__hp(&lvxr(2*lidz-1,lidx)))))),
                              mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvxr(2*lidz+3,lidx))), __h22f2(__hp(&lvxr(2*lidz-2,lidx)))))),
                         mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvxr(2*lidz+4,lidx))), __h22f2(__hp(&lvxr(2*lidz-3,lidx)))))),
                    mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvxr(2*lidz+5,lidx))), __h22f2(__hp(&lvxr(2*lidz-4,lidx))))));
#elif FDOH == 6
        vxr_z1=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvxr(2*lidz+1,lidx))), __h22f2(__hp(&lvxr(2*lidz,lidx))))),
                                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvxr(2*lidz+2,lidx))), __h22f2(__hp(&lvxr(2*lidz-1,lidx)))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvxr(2*lidz+3,lidx))), __h22f2(__hp(&lvxr(2*lidz-2,lidx)))))),
                              mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvxr(2*lidz+4,lidx))), __h22f2(__hp(&lvxr(2*lidz-3,lidx)))))),
                         mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvxr(2*lidz+5,lidx))), __h22f2(__hp(&lvxr(2*lidz-4,lidx)))))),
                    mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvxr(2*lidz+6,lidx))), __h22f2(__hp(&lvxr(2*lidz-5,lidx))))));
#endif
        
#if LOCAL_OFF==0
        __syncthreads();
        lvzr2(lidz,lidx)=vzr(gidz,gidx);
        if (lidz<FDOH)
            lvzr2(lidz-FDOH/2,lidx)=vzr(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lvzr2(lidz+FDOH/2,lidx)=vzr(gidz+FDOH/2,gidx);
        if (lidx<2*FDOH)
            lvzr2(lidz,lidx-FDOH)=vzr(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvzr2(lidz,lidx+lsizex-3*FDOH)=vzr(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvzr2(lidz,lidx+FDOH)=vzr(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvzr2(lidz,lidx-lsizex+3*FDOH)=vzr(gidz,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        vzr_x1=mul2( f2h2(HC1), sub2(__h22f2(lvzr2(lidz,lidx+1)), __h22f2(lvzr2(lidz,lidx))));
#elif FDOH == 2
        vzr_x1=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lvzr2(lidz,lidx+1)), __h22f2(lvzr2(lidz,lidx)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lvzr2(lidz,lidx+2)), __h22f2(lvzr2(lidz,lidx-1)))));
#elif FDOH == 3
        vzr_x1=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lvzr2(lidz,lidx+1)), __h22f2(lvzr2(lidz,lidx)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lvzr2(lidz,lidx+2)), __h22f2(lvzr2(lidz,lidx-1))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lvzr2(lidz,lidx+3)), __h22f2(lvzr2(lidz,lidx-2)))));
#elif FDOH == 4
        vzr_x1=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lvzr2(lidz,lidx+1)), __h22f2(lvzr2(lidz,lidx)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lvzr2(lidz,lidx+2)), __h22f2(lvzr2(lidz,lidx-1))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lvzr2(lidz,lidx+3)), __h22f2(lvzr2(lidz,lidx-2))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lvzr2(lidz,lidx+4)), __h22f2(lvzr2(lidz,lidx-3)))));
#elif FDOH == 5
        vzr_x1=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lvzr2(lidz,lidx+1)), __h22f2(lvzr2(lidz,lidx)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lvzr2(lidz,lidx+2)), __h22f2(lvzr2(lidz,lidx-1))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lvzr2(lidz,lidx+3)), __h22f2(lvzr2(lidz,lidx-2))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lvzr2(lidz,lidx+4)), __h22f2(lvzr2(lidz,lidx-3))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lvzr2(lidz,lidx+5)), __h22f2(lvzr2(lidz,lidx-4)))));
#elif FDOH == 6
        vzr_x1=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lvzr2(lidz,lidx+1)), __h22f2(lvzr2(lidz,lidx)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lvzr2(lidz,lidx+2)), __h22f2(lvzr2(lidz,lidx-1))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lvzr2(lidz,lidx+3)), __h22f2(lvzr2(lidz,lidx-2))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lvzr2(lidz,lidx+4)), __h22f2(lvzr2(lidz,lidx-3))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lvzr2(lidz,lidx+5)), __h22f2(lvzr2(lidz,lidx-4))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lvzr2(lidz,lidx+6)), __h22f2(lvzr2(lidz,lidx-5)))));
#endif
        
#if   FDOH == 1
        vzr_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvzr(2*lidz,lidx))), __h22f2(__hp(&lvzr(2*lidz-1,lidx)))));
#elif FDOH == 2
        vzr_z2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvzr(2*lidz,lidx))), __h22f2(__hp(&lvzr(2*lidz-1,lidx))))),
                    mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvzr(2*lidz+1,lidx))), __h22f2(__hp(&lvzr(2*lidz-2,lidx))))));
#elif FDOH == 3
        vzr_z2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvzr(2*lidz,lidx))), __h22f2(__hp(&lvzr(2*lidz-1,lidx))))),
                         mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvzr(2*lidz+1,lidx))), __h22f2(__hp(&lvzr(2*lidz-2,lidx)))))),
                    mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvzr(2*lidz+2,lidx))), __h22f2(__hp(&lvzr(2*lidz-3,lidx))))));
#elif FDOH == 4
        vzr_z2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvzr(2*lidz,lidx))), __h22f2(__hp(&lvzr(2*lidz-1,lidx))))),
                              mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvzr(2*lidz+1,lidx))), __h22f2(__hp(&lvzr(2*lidz-2,lidx)))))),
                         mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvzr(2*lidz+2,lidx))), __h22f2(__hp(&lvzr(2*lidz-3,lidx)))))),
                    mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvzr(2*lidz+3,lidx))), __h22f2(__hp(&lvzr(2*lidz-4,lidx))))));
#elif FDOH == 5
        vzr_z2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvzr(2*lidz,lidx))), __h22f2(__hp(&lvzr(2*lidz-1,lidx))))),
                                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvzr(2*lidz+1,lidx))), __h22f2(__hp(&lvzr(2*lidz-2,lidx)))))),
                              mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvzr(2*lidz+2,lidx))), __h22f2(__hp(&lvzr(2*lidz-3,lidx)))))),
                         mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvzr(2*lidz+3,lidx))), __h22f2(__hp(&lvzr(2*lidz-4,lidx)))))),
                    mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvzr(2*lidz+4,lidx))), __h22f2(__hp(&lvzr(2*lidz-5,lidx))))));
#elif FDOH == 6
        vzr_z2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvzr(2*lidz,lidx))), __h22f2(__hp(&lvzr(2*lidz-1,lidx))))),
                                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvzr(2*lidz+1,lidx))), __h22f2(__hp(&lvzr(2*lidz-2,lidx)))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvzr(2*lidz+2,lidx))), __h22f2(__hp(&lvzr(2*lidz-3,lidx)))))),
                              mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvzr(2*lidz+3,lidx))), __h22f2(__hp(&lvzr(2*lidz-4,lidx)))))),
                         mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvzr(2*lidz+4,lidx))), __h22f2(__hp(&lvzr(2*lidz-5,lidx)))))),
                    mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvzr(2*lidz+5,lidx))), __h22f2(__hp(&lvzr(2*lidz-6,lidx))))));
#endif
    }
    
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

    
// Backpropagate the forward stresses
#if BACK_PROP_TYPE==1
    {
        // Update the variables
        lsxz=sub2(lsxz,mul2(lmuipkp,add2(vx_z1,vz_x1)));
        lsxx=add2(sub2(lsxx,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vz_z2));
        lszz=add2(sub2(lszz,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vx_x2));

//        int m=evarm(gidz, gidx);
//        if (m!=-1){
//            lsxx= __h22f2(sxxbnd[m]);
//            lszz= __h22f2(szzbnd[m]);
//            lsxz= __h22f2(sxzbnd[m]);
//        }
        
        //Write updated values to global memory
        sxx(gidz,gidx) = __f22h2(lsxx);
        sxz(gidz,gidx) = __f22h2(lsxz);
        szz(gidz,gidx) = __f22h2(lszz);
        
    }
#endif


    
    
// Update adjoint stresses
    {
        // Update the variables
        lsxzr=add2(lsxzr,mul2(lmuipkp,add2(vxr_z1,vzr_x1)));
        lsxxr=sub2(add2(lsxxr,mul2(lM,add2(vxr_x2,vzr_z2))),mul2(mul2(f2h2(2.0),lmu),vzr_z2));
        lszzr=sub2(add2(lszzr,mul2(lM,add2(vxr_x2,vzr_z2))),mul2(mul2(f2h2(2.0),lmu),vxr_x2));
        //Write updated values to global memory
        sxxr(gidz,gidx) = __f22h2(lsxxr);
        sxzr(gidz,gidx) = __f22h2(lsxzr);
        szzr(gidz,gidx) = __f22h2(lszzr);
    }

    // Shear wave modulus and P-wave modulus gradient calculation on the fly
#if BACK_PROP_TYPE==1
    float2 c1= div2(f2h2(1.0), mul2(mul2(f2h2(2.0), sub2(lM,lmu)),mul2(f2h2(2.0), sub2(lM,lmu))));
    float2 c3=div2(f2h2(1.0), mul2(lmu,lmu));
    float2 c5=mul2(f2h2(0.25), c3);
    
    lsxzr=mul2(lmuipkp,add2(vxr_z1,vzr_x1));
    lsxxr=sub2(mul2(lM,add2(vxr_x2,vzr_z2)),mul2(mul2(f2h2(2.0),lmu),vzr_z2));
    lszzr=sub2(mul2(lM,add2(vxr_x2,vzr_z2)),mul2(mul2(f2h2(2.0),lmu),vxr_x2));

    float2 dM=mul2(c1,mul2(add2(lsxx,lszz), add2(lsxxr,lszzr) ) );

    gradM(gidz,gidx)=sub2(gradM(gidz,gidx), dM);//sub2(gradM(gidz,gidx), dM);
    
    gradmu(gidz,gidx)=sub2(sub2(add2(gradM(gidz,gidx), dM), mul2(c3, mul2(lsxz,lsxzr))), mul2(c5,mul2( sub2(lsxx,lszz), sub2(lsxxr,lszzr))));
    
    
#endif


}

