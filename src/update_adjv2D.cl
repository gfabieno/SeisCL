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
#define lbnd (FDOH+NAB)

#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipkp(z,x) uipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define u(z,x)        u[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define pi(z,x)      pi[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,x)  gradrho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,x)  gradM[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradmu(z,x)  gradmu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,x)  gradtaup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaus(z,x)  gradtaus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define taus(z,x)        taus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipkp(z,x) tausipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define taup(z,x)        taup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define vx(z,x)  vx[(x)*(NZ)+(z)]
#define vz(z,x)  vz[(x)*(NZ)+(z)]
#define sxx(z,x) sxx[(x)*(NZ)+(z)]
#define szz(z,x) szz[(x)*(NZ)+(z)]
#define sxz(z,x) sxz[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]

#define vx_r(z,x)  vx_r[(x)*(NZ)+(z)]
#define vz_r(z,x)  vz_r[(x)*(NZ)+(z)]
#define sxx_r(z,x) sxx_r[(x)*(NZ)+(z)]
#define szz_r(z,x) szz_r[(x)*(NZ)+(z)]
#define sxz_r(z,x) sxz_r[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*NAB)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*NAB)+(z)]


#if LOCAL_OFF==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]


#endif


#define vxout(y,x) vxout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define PI (3.141592653589793238462643383279502884197169)
#define signals(y,x) signals[(y)*NT+(x)]
#define rec_pos(y,x) rec_pos[(y)*8+(x)]
#define gradsrc(y,x) gradsrc[(y)*NT+(x)]


// Find boundary indice for boundary injection in backpropagation
int evarm( int k, int i){
    
    
#if NUM_DEVICES==1 & NLOCALP==1

    int NXbnd = (NX-2*FDOH-2*NAB);
    int NZbnd = (NZ-2*FDOH-2*NAB);

    int m=-1;
    i-=lbnd;
    k-=lbnd;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH)  && (i>FDOH-1 && i<NXbnd-FDOH) )
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
    else if (k<FDOH){//up
        i=i-FDOH;
        m=NZbnd*FDOH*2+i+k*(NXbnd-2.0*FDOH);
    }
    else {//down
        i=i-FDOH;
        k=k-NZbnd+FDOH;
        m=NZbnd*FDOH*2+(NXbnd-2*FDOH)*FDOH+i+k*(NXbnd-2.0*FDOH);
    }
    
    

#elif DEVID==0 & MYGROUPID==0
    
    int NXbnd = (NX-2*FDOH-NAB);
    int NZbnd = (NZ-2*FDOH-2*NAB);
    
    int m=-1;
    i-=lbnd;
    k-=lbnd;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH)  && i>FDOH-1  )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<FDOH){//front
        m=i*NZbnd+k;
    }
    else if (k<FDOH){//up
        i=i-FDOH;
        m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
    }
    else {//down
        i=i-FDOH;
        k=k-NZbnd+FDOH;
        m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH+i+k*(NXbnd-FDOH);
    }

#elif DEVID==NUM_DEVICES-1 & MYGROUPID==NLOCALP-1
    int NXbnd = (NX-2*FDOH-NAB);
    int NZbnd = (NZ-2*FDOH-2*NAB);
    
    int m=-1;
    i-=FDOH;
    k-=lbnd;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) && i<NXbnd-FDOH )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i>NXbnd-1 )
        m=-1;
    else if (i>NXbnd-1-FDOH){
        i=i-NXbnd+FDOH;
        m=i*NZbnd+k;
    }
    else if (k<FDOH){//up
        m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
    }
    else {//down
        k=k-NZbnd+FDOH;
        m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH+i+k*(NXbnd-FDOH);
    }
    
#else
    
    int NXbnd = (NX-2*FDOH);
    int NZbnd = (NZ-2*FDOH-2*NAB);
    
    int m=-1;
    i-=FDOH;
    k-=lbnd;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (k<FDOH){//up
        m=i+k*(NXbnd);
    }
    else {//down
        k=k-NZbnd+FDOH;
        m=(NXbnd)*FDOH+i+k*(NXbnd);
    }
    
    
#endif
    
    
    return m;
 
}


__kernel void update_adjv(int offcomm,
                          __global float *vx,         __global float *vz,
                          __global float *sxx,        __global float *szz,
                          __global float *sxz,
                          __global float *vxbnd,      __global float *vzbnd,
                          __global float *sxxbnd,     __global float *szzbnd,
                          __global float *sxzbnd,
                          __global float *vx_r,       __global float *vz_r,
                          __global float *sxx_r,      __global float *szz_r,
                          __global float *sxz_r,
                          __global float *rip,        __global float *rkp,
                          __global float *taper,
                          __global float *K_x,        __global float *a_x,          __global float *b_x,
                          __global float *K_x_half,   __global float *a_x_half,     __global float *b_x_half,
                          __global float *K_z,        __global float *a_z,          __global float *b_z,
                          __global float *K_z_half,   __global float *a_z_half,     __global float *b_z_half,
                          __global float *psi_sxx_x,  __global float *psi_sxz_x,
                          __global float *psi_sxz_z,  __global float *psi_szz_z,
                          __local  float *lvar,       __global float *gradrho, __global float *gradsrc)
{

    int g,i,j,k,m;
    float sxx_x_r;
    float szz_z_r;
    float sxz_x_r;
    float sxz_z_r;
    float sxx_x;
    float szz_z;
    float sxz_x;
    float sxz_z;
    float2 amp;
    float lvx;
    float lvz;
    
// If we use local memory
#if LOCAL_OFF==0
    
    int lsizez = get_local_size(0)+2*FDOH;
    int lsizex = get_local_size(1)+2*FDOH;
    int lidz = get_local_id(0)+FDOH;
    int lidx = get_local_id(1)+FDOH;
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH+offcomm;

#define lsxx lvar
#define lszz lvar
#define lsxz lvar
    
#define lsxx_r lvar
#define lszz_r lvar
#define lsxz_r lvar
 
// If local memory is turned off
#elif LOCAL_OFF==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*FDOH);
    int gidz = gid%glsizez+FDOH;
    int gidx = (gid/glsizez)+FDOH+offcomm;
    
#define lsxx sxx
#define lszz szz
#define lsxz sxz
    
#define lsxx_r sxx_r
#define lszz_r szz_r
#define lsxz_r sxz_r
    
#define lidx gidx
#define lidz gidz
    
#define lsizez NZ
#define lsizex NX
    
#endif
    
// Calculation of the stress spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
    {
#if LOCAL_OFF==0
        lsxx(lidz,lidx)=sxx(gidz, gidx);
        if (lidx<2*FDOH)
            lsxx(lidz,lidx-FDOH)=sxx(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxx(lidz,lidx+lsizex-3*FDOH)=sxx(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxx(lidz,lidx+FDOH)=sxx(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxx(lidz,lidx-lsizex+3*FDOH)=sxx(gidz,gidx-lsizex+3*FDOH);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH ==1
        sxx_x = DTDH*HC1*(lsxx(lidz,lidx+1) - lsxx(lidz,lidx));
#elif FDOH ==2
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidx+1) - lsxx(lidz,lidx))
                      +HC2*(lsxx(lidz,lidx+2) - lsxx(lidz,lidx-1)));
#elif FDOH ==3
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      HC2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      HC3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2)));
#elif FDOH ==4
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      HC2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      HC3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
                      HC4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3)));
#elif FDOH ==5
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      HC2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      HC3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
                      HC4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3))+
                      HC5*(lsxx(lidz,lidx+5)-lsxx(lidz,lidx-4)));
#elif FDOH ==6
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      HC2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      HC3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
                      HC4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3))+
                      HC5*(lsxx(lidz,lidx+5)-lsxx(lidz,lidx-4))+
                      HC6*(lsxx(lidz,lidx+6)-lsxx(lidz,lidx-5)));
#endif
        
        
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lszz(lidz,lidx)=szz(gidz, gidx);
        if (lidz<2*FDOH)
            lszz(lidz-FDOH,lidx)=szz(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lszz(lidz+FDOH,lidx)=szz(gidz+FDOH,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH ==1
        szz_z = DTDH*HC1*(lszz(lidz+1,lidx) - lszz(lidz,lidx));
#elif FDOH ==2
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidx) - lszz(lidz,lidx))
                      +HC2*(lszz(lidz+2,lidx) - lszz(lidz-1,lidx)));
#elif FDOH ==3
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      HC2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      HC3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx)));
#elif FDOH ==4
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      HC2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      HC3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
                      HC4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx)));
#elif FDOH ==5
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      HC2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      HC3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
                      HC4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx))+
                      HC5*(lszz(lidz+5,lidx)-lszz(lidz-4,lidx)));
#elif FDOH ==6
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      HC2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      HC3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
                      HC4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx))+
                      HC5*(lszz(lidz+5,lidx)-lszz(lidz-4,lidx))+
                      HC6*(lszz(lidz+6,lidx)-lszz(lidz-5,lidx)));
#endif
        
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsxz(lidz,lidx)=sxz(gidz, gidx);
        
        if (lidx<2*FDOH)
            lsxz(lidz,lidx-FDOH)=sxz(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxz(lidz,lidx+lsizex-3*FDOH)=sxz(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxz(lidz,lidx+FDOH)=sxz(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxz(lidz,lidx-lsizex+3*FDOH)=sxz(gidz,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lsxz(lidz-FDOH,lidx)=sxz(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lsxz(lidz+FDOH,lidx)=sxz(gidz+FDOH,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH ==1
        sxz_z = DTDH*HC1*(lsxz(lidz,lidx)   - lsxz(lidz-1,lidx));
        sxz_x = DTDH*HC1*(lsxz(lidz,lidx)   - lsxz(lidz,lidx-1));
#elif FDOH ==2
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidx)   - lsxz(lidz-1,lidx))
                      +HC2*(lsxz(lidz+1,lidx) - lsxz(lidz-2,lidx)));
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidx)   - lsxz(lidz,lidx-1))
                      +HC2*(lsxz(lidz,lidx+1) - lsxz(lidz,lidx-2)));
#elif FDOH ==3
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      HC2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      HC3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx)));
        
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      HC2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      HC3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3)));
#elif FDOH ==4
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      HC2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      HC3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
                      HC4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx)));
        
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      HC2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      HC3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
                      HC4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4)));
#elif FDOH ==5
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      HC2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      HC3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
                      HC4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx))+
                      HC5*(lsxz(lidz+4,lidx)-lsxz(lidz-5,lidx)));
        
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      HC2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      HC3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
                      HC4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4))+
                      HC5*(lsxz(lidz,lidx+4)-lsxz(lidz,lidx-5)));
#elif FDOH ==6
        
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      HC2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      HC3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
                      HC4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx))+
                      HC5*(lsxz(lidz+4,lidx)-lsxz(lidz-5,lidx))+
                      HC6*(lsxz(lidz+5,lidx)-lsxz(lidz-6,lidx)));
        
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      HC2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      HC3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
                      HC4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4))+
                      HC5*(lsxz(lidz,lidx+4)-lsxz(lidz,lidx-5))+
                      HC6*(lsxz(lidz,lidx+5)-lsxz(lidz,lidx-6)));
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

// Calculation of the stress spatial derivatives of the adjoint wavefield
#if LOCAL_OFF==0
    lsxx_r(lidz,lidx)=sxx_r(gidz, gidx);
    if (lidx<2*FDOH)
        lsxx_r(lidz,lidx-FDOH)=sxx_r(gidz,gidx-FDOH);
    if (lidx+lsizex-3*FDOH<FDOH)
        lsxx_r(lidz,lidx+lsizex-3*FDOH)=sxx_r(gidz,gidx+lsizex-3*FDOH);
    if (lidx>(lsizex-2*FDOH-1))
        lsxx_r(lidz,lidx+FDOH)=sxx_r(gidz,gidx+FDOH);
    if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
        lsxx_r(lidz,lidx-lsizex+3*FDOH)=sxx_r(gidz,gidx-lsizex+3*FDOH);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH ==1
    sxx_x_r = DTDH*HC1*(lsxx_r(lidz,lidx+1) - lsxx_r(lidz,lidx));
#elif FDOH ==2
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidx+1) - lsxx_r(lidz,lidx))
                  +HC2*(lsxx_r(lidz,lidx+2) - lsxx_r(lidz,lidx-1)));
#elif FDOH ==3
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidx+1)-lsxx_r(lidz,lidx))+
                  HC2*(lsxx_r(lidz,lidx+2)-lsxx_r(lidz,lidx-1))+
                  HC3*(lsxx_r(lidz,lidx+3)-lsxx_r(lidz,lidx-2)));
#elif FDOH ==4
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidx+1)-lsxx_r(lidz,lidx))+
                  HC2*(lsxx_r(lidz,lidx+2)-lsxx_r(lidz,lidx-1))+
                  HC3*(lsxx_r(lidz,lidx+3)-lsxx_r(lidz,lidx-2))+
                  HC4*(lsxx_r(lidz,lidx+4)-lsxx_r(lidz,lidx-3)));
#elif FDOH ==5
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidx+1)-lsxx_r(lidz,lidx))+
                  HC2*(lsxx_r(lidz,lidx+2)-lsxx_r(lidz,lidx-1))+
                  HC3*(lsxx_r(lidz,lidx+3)-lsxx_r(lidz,lidx-2))+
                  HC4*(lsxx_r(lidz,lidx+4)-lsxx_r(lidz,lidx-3))+
                  HC5*(lsxx_r(lidz,lidx+5)-lsxx_r(lidz,lidx-4)));
#elif FDOH ==6
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidx+1)-lsxx_r(lidz,lidx))+
                  HC2*(lsxx_r(lidz,lidx+2)-lsxx_r(lidz,lidx-1))+
                  HC3*(lsxx_r(lidz,lidx+3)-lsxx_r(lidz,lidx-2))+
                  HC4*(lsxx_r(lidz,lidx+4)-lsxx_r(lidz,lidx-3))+
                  HC5*(lsxx_r(lidz,lidx+5)-lsxx_r(lidz,lidx-4))+
                  HC6*(lsxx_r(lidz,lidx+6)-lsxx_r(lidz,lidx-5)));
#endif
    
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lszz_r(lidz,lidx)=szz_r(gidz, gidx);
    if (lidz<2*FDOH)
        lszz_r(lidz-FDOH,lidx)=szz_r(gidz-FDOH,gidx);
    if (lidz>(lsizez-2*FDOH-1))
        lszz_r(lidz+FDOH,lidx)=szz_r(gidz+FDOH,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH ==1
    szz_z_r = DTDH*HC1*(lszz_r(lidz+1,lidx) - lszz_r(lidz,lidx));
#elif FDOH ==2
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidx) - lszz_r(lidz,lidx))
                  +HC2*(lszz_r(lidz+2,lidx) - lszz_r(lidz-1,lidx)));
#elif FDOH ==3
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidx)-lszz_r(lidz,lidx))+
                  HC2*(lszz_r(lidz+2,lidx)-lszz_r(lidz-1,lidx))+
                  HC3*(lszz_r(lidz+3,lidx)-lszz_r(lidz-2,lidx)));
#elif FDOH ==4
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidx)-lszz_r(lidz,lidx))+
                  HC2*(lszz_r(lidz+2,lidx)-lszz_r(lidz-1,lidx))+
                  HC3*(lszz_r(lidz+3,lidx)-lszz_r(lidz-2,lidx))+
                  HC4*(lszz_r(lidz+4,lidx)-lszz_r(lidz-3,lidx)));
#elif FDOH ==5
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidx)-lszz_r(lidz,lidx))+
                  HC2*(lszz_r(lidz+2,lidx)-lszz_r(lidz-1,lidx))+
                  HC3*(lszz_r(lidz+3,lidx)-lszz_r(lidz-2,lidx))+
                  HC4*(lszz_r(lidz+4,lidx)-lszz_r(lidz-3,lidx))+
                  HC5*(lszz_r(lidz+5,lidx)-lszz_r(lidz-4,lidx)));
#elif FDOH ==6
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidx)-lszz_r(lidz,lidx))+
                  HC2*(lszz_r(lidz+2,lidx)-lszz_r(lidz-1,lidx))+
                  HC3*(lszz_r(lidz+3,lidx)-lszz_r(lidz-2,lidx))+
                  HC4*(lszz_r(lidz+4,lidx)-lszz_r(lidz-3,lidx))+
                  HC5*(lszz_r(lidz+5,lidx)-lszz_r(lidz-4,lidx))+
                  HC6*(lszz_r(lidz+6,lidx)-lszz_r(lidz-5,lidx)));
#endif
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lsxz_r(lidz,lidx)=sxz_r(gidz, gidx);
    
    if (lidx<2*FDOH)
        lsxz_r(lidz,lidx-FDOH)=sxz_r(gidz,gidx-FDOH);
    if (lidx+lsizex-3*FDOH<FDOH)
        lsxz_r(lidz,lidx+lsizex-3*FDOH)=sxz_r(gidz,gidx+lsizex-3*FDOH);
    if (lidx>(lsizex-2*FDOH-1))
        lsxz_r(lidz,lidx+FDOH)=sxz_r(gidz,gidx+FDOH);
    if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
        lsxz_r(lidz,lidx-lsizex+3*FDOH)=sxz_r(gidz,gidx-lsizex+3*FDOH);
    if (lidz<2*FDOH)
        lsxz_r(lidz-FDOH,lidx)=sxz_r(gidz-FDOH,gidx);
    if (lidz>(lsizez-2*FDOH-1))
        lsxz_r(lidz+FDOH,lidx)=sxz_r(gidz+FDOH,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH ==1
    sxz_z_r = DTDH*HC1*(lsxz_r(lidz,lidx)   - lsxz_r(lidz-1,lidx));
    sxz_x_r = DTDH*HC1*(lsxz_r(lidz,lidx)   - lsxz_r(lidz,lidx-1));
#elif FDOH ==2
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidx)   - lsxz_r(lidz-1,lidx))
                  +HC2*(lsxz_r(lidz+1,lidx) - lsxz_r(lidz-2,lidx)));
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidx)   - lsxz_r(lidz,lidx-1))
                  +HC2*(lsxz_r(lidz,lidx+1) - lsxz_r(lidz,lidx-2)));
#elif FDOH ==3
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz-1,lidx))+
                  HC2*(lsxz_r(lidz+1,lidx)-lsxz_r(lidz-2,lidx))+
                  HC3*(lsxz_r(lidz+2,lidx)-lsxz_r(lidz-3,lidx)));
    
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz,lidx-1))+
                  HC2*(lsxz_r(lidz,lidx+1)-lsxz_r(lidz,lidx-2))+
                  HC3*(lsxz_r(lidz,lidx+2)-lsxz_r(lidz,lidx-3)));
#elif FDOH ==4
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz-1,lidx))+
                  HC2*(lsxz_r(lidz+1,lidx)-lsxz_r(lidz-2,lidx))+
                  HC3*(lsxz_r(lidz+2,lidx)-lsxz_r(lidz-3,lidx))+
                  HC4*(lsxz_r(lidz+3,lidx)-lsxz_r(lidz-4,lidx)));
    
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz,lidx-1))+
                  HC2*(lsxz_r(lidz,lidx+1)-lsxz_r(lidz,lidx-2))+
                  HC3*(lsxz_r(lidz,lidx+2)-lsxz_r(lidz,lidx-3))+
                  HC4*(lsxz_r(lidz,lidx+3)-lsxz_r(lidz,lidx-4)));
#elif FDOH ==5
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz-1,lidx))+
                  HC2*(lsxz_r(lidz+1,lidx)-lsxz_r(lidz-2,lidx))+
                  HC3*(lsxz_r(lidz+2,lidx)-lsxz_r(lidz-3,lidx))+
                  HC4*(lsxz_r(lidz+3,lidx)-lsxz_r(lidz-4,lidx))+
                  HC5*(lsxz_r(lidz+4,lidx)-lsxz_r(lidz-5,lidx)));
    
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz,lidx-1))+
                  HC2*(lsxz_r(lidz,lidx+1)-lsxz_r(lidz,lidx-2))+
                  HC3*(lsxz_r(lidz,lidx+2)-lsxz_r(lidz,lidx-3))+
                  HC4*(lsxz_r(lidz,lidx+3)-lsxz_r(lidz,lidx-4))+
                  HC5*(lsxz_r(lidz,lidx+4)-lsxz_r(lidz,lidx-5)));
#elif FDOH ==6
    
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz-1,lidx))+
                  HC2*(lsxz_r(lidz+1,lidx)-lsxz_r(lidz-2,lidx))+
                  HC3*(lsxz_r(lidz+2,lidx)-lsxz_r(lidz-3,lidx))+
                  HC4*(lsxz_r(lidz+3,lidx)-lsxz_r(lidz-4,lidx))+
                  HC5*(lsxz_r(lidz+4,lidx)-lsxz_r(lidz-5,lidx))+
                  HC6*(lsxz_r(lidz+5,lidx)-lsxz_r(lidz-6,lidx)));
    
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz,lidx-1))+
                  HC2*(lsxz_r(lidz,lidx+1)-lsxz_r(lidz,lidx-2))+
                  HC3*(lsxz_r(lidz,lidx+2)-lsxz_r(lidz,lidx-3))+
                  HC4*(lsxz_r(lidz,lidx+3)-lsxz_r(lidz,lidx-4))+
                  HC5*(lsxz_r(lidz,lidx+4)-lsxz_r(lidz,lidx-5))+
                  HC6*(lsxz_r(lidz,lidx+5)-lsxz_r(lidz,lidx-6)));
#endif

    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if LOCAL_OFF==0
#if COMM12==0
    if (gidz>(NZ-FDOH-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
        return;
    }
    
#else
    if (gidz>(NZ-FDOH-1) ){
        return;
    }
#endif
#endif


// Backpropagate the forward velocity
#if BACK_PROP_TYPE==1
    {

        lvx=((sxx_x + sxz_z)/rip(gidz,gidx))+amp.x;
        lvz=((szz_z + sxz_x)/rkp(gidz,gidx))+amp.y;
        vx(gidz,gidx)-= lvx;
        vz(gidz,gidx)-= lvz;
        
        // Inject the boundary values
        m=evarm(gidz,  gidx);
        if (m!=-1){
            vx(gidz, gidx)= vxbnd[m];
            vz(gidz, gidx)= vzbnd[m];
        }
    }
#endif

// Correct adjoint spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
    int ind;
    
    if (gidz>NZ-NAB-FDOH-1){
        
        i =gidx-FDOH;
        k =gidz - NZ+NAB+FDOH+NAB;
        ind=2*NAB-1-k;
        
        psi_sxz_z(k,i) = b_z[ind+1] * psi_sxz_z(k,i) + a_z[ind+1] * sxz_z_r;
        sxz_z_r = sxz_z_r / K_z[ind+1] + psi_sxz_z(k,i);
        psi_szz_z(k,i) = b_z_half[ind] * psi_szz_z(k,i) + a_z_half[ind] * szz_z_r;
        szz_z_r = szz_z_r / K_z_half[ind] + psi_szz_z(k,i);
        
    }
    
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        
        i =gidx-FDOH;
        k =gidz-FDOH;
        
        psi_sxz_z(k,i) = b_z[k] * psi_sxz_z(k,i) + a_z[k] * sxz_z_r;
        sxz_z_r = sxz_z_r / K_z[k] + psi_sxz_z(k,i);
        psi_szz_z(k,i) = b_z_half[k] * psi_szz_z(k,i) + a_z_half[k] * szz_z_r;
        szz_z_r = szz_z_r / K_z_half[k] + psi_szz_z(k,i);
        
    }
#endif
    
#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        
        i =gidx-FDOH;
        k =gidz-FDOH;
        
        psi_sxx_x(k,i) = b_x_half[i] * psi_sxx_x(k,i) + a_x_half[i] * sxx_x_r;
        sxx_x_r = sxx_x_r / K_x_half[i] + psi_sxx_x(k,i);
        psi_sxz_x(k,i) = b_x[i] * psi_sxz_x(k,i) + a_x[i] * sxz_x_r;
        sxz_x_r = sxz_x_r / K_x[i] + psi_sxz_x(k,i);
        
    }
#endif
    
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        
        i =gidx - NX+NAB+FDOH+NAB;
        k =gidz-FDOH;
        ind=2*NAB-1-i;
        
        psi_sxx_x(k,i) = b_x_half[ind] * psi_sxx_x(k,i) + a_x_half[ind] * sxx_x_r;
        sxx_x_r = sxx_x_r / K_x_half[ind] + psi_sxx_x(k,i);
        psi_sxz_x(k,i) = b_x[ind+1] * psi_sxz_x(k,i) + a_x[ind+1] * sxz_x_r;
        sxz_x_r = sxz_x_r / K_x[ind+1] + psi_sxz_x(k,i);
        
    }
#endif
    }
#endif
    
    // Update adjoint velocities
    lvx=((sxx_x_r + sxz_z_r)/rip(gidz,gidx));
    lvz=((szz_z_r + sxz_x_r)/rkp(gidz,gidx));
    vx_r(gidz,gidx)+= lvx;
    vz_r(gidz,gidx)+= lvz;
 
    

// Absorbing boundary
#if ABS_TYPE==2
    {
    if (gidz-FDOH<NAB){
        vx_r(gidz,gidx)*=taper[gidz-FDOH];
        vz_r(gidz,gidx)*=taper[gidz-FDOH];
    }
    
    if (gidz>NZ-NAB-FDOH-1){
        vx_r(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
        vz_r(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
    }
    
#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        vx_r(gidz,gidx)*=taper[gidx-FDOH];
        vz_r(gidz,gidx)*=taper[gidx-FDOH];
    }
#endif
    
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        vx_r(gidz,gidx)*=taper[NX-FDOH-gidx-1];
        vz_r(gidz,gidx)*=taper[NX-FDOH-gidx-1];
    }
#endif
    }
#endif
    
    
// Density gradient calculation on the fly
#if BACK_PROP_TYPE==1
    gradrho(gidz,gidx)+=vx(gidz,gidx)*lvx+vz(gidz,gidx)*lvz;
#endif
    
#if GRADSRCOUT==1
    //TODO
//    if (nsrc>0){
//        
//        
//        for (int srci=0; srci<nsrc; srci++){
//            
//            
//            int i=(int)(srcpos_loc(0,srci)/DH-0.5)+FDOH;
//            int k=(int)(srcpos_loc(2,srci)/DH-0.5)+FDOH;
//            
//            if (i==gidx && k==gidz){
//                
//                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
//                
//                if (SOURCE_TYPE==2){
//                    /* single force in x */
//                    gradsrc(srci,nt)+= vx_r(gidz,gidx)/rip(gidx,gidz)/(DH*DH);
//                }
//                else if (SOURCE_TYPE==4){
//                    /* single force in z */
//                    
//                    gradsrc(srci,nt)+= vz_r(gidz,gidx)/rkp(gidx,gidz)/(DH*DH);
//                }
//                
//            }
//        }
//        
//        
//    }
#endif
    
}

