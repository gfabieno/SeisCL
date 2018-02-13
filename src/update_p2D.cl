
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

/*Update of the velocity in 2D SH*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define lbnd (FDOH+NAB)

#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,x)    rjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipkp(z,x) uipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define ujpkp(z,x) ujpkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipjp(z,x) uipjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define u(z,x)        u[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define pi(z,x)      pi[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,x)  gradrho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,x)  gradM[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradmu(z,x)  gradmu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,x)  gradtaup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaus(z,x)  gradtaus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define taus(z,x)        taus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipkp(z,x) tausipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipjp(z,x) tausipjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausjpkp(z,x) tausjpkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define taup(z,x)        taup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define vx(z,x)  vx[(x)*(NZ)+(z)]
#define vy(z,x)  vy[(x)*(NZ)+(z)]
#define vz(z,x)  vz[(x)*(NZ)+(z)]
#define sxx(z,x) sxx[(x)*(NZ)+(z)]
#define szz(z,x) szz[(x)*(NZ)+(z)]
#define sxz(z,x) sxz[(x)*(NZ)+(z)]
#define sxy(z,x) sxy[(x)*(NZ)+(z)]
#define syz(z,x) syz[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxy(z,x,l) rxy[(l)*NX*NZ+(x)*NZ+(z)]
#define ryz(z,x,l) ryz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*NAB)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*NAB)+(z)]
#define psi_sxy_x(z,x) psi_sxy_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_syz_z(z,x) psi_syz_z[(x)*(2*NAB)+(z)]

#define psi_vyx(z,x) psi_vyx[(x)*(NZ-2*FDOH)+(z)]
#define psi_vyz(z,x) psi_vyz[(x)*(2*NAB)+(z)]

#if LOCAL_OFF==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

#endif


#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define PI (3.141592653589793238462643383279502884197169)

#ifdef __OPENCL_VERSION__
#define FUNDEF __kernel
#define LFUNDEF
#define GLOBARG __global
#define LOCARG __local float *lvar
#define LOCDEF
#define BARRIER barrier(CLK_LOCAL_MEM_FENCE);
#else
#define FUNDEF extern "C" __global__
#define LFUNDEF extern "C" __device__
#define GLOBARG
#define LOCARG float *nullarg
#define LOCDEF extern __shared__ float lvar[];
#define BARRIER __syncthreads();
#endif


void dzp(float out, float in, int lidx, int lidz){
#if   FDOH ==1
    out(lidz,lidx) = HC1*(in(lidz,lidx)   - in(lidz-1,lidx));
#elif FDOH ==2
    out(lidz,lidx) = (HC1*(in(lidz,lidx)   - in(lidz-1,lidx))
                      +HC2*(in(lidz+1,lidx) - in(lidz-2,lidx)));
#elif FDOH ==3
    out(lidz,lidx) = (HC1*(in(lidz,lidx)  -in(lidz-1,lidx))+
                      HC2*(in(lidz+1,lidx)-in(lidz-2,lidx))+
                      HC3*(in(lidz+2,lidx)-in(lidz-3,lidx)));
#elif FDOH ==4
    out(lidz,lidx) = (HC1*(in(lidz,lidx)  -in(lidz-1,lidx))+
                      HC2*(in(lidz+1,lidx)-in(lidz-2,lidx))+
                      HC3*(in(lidz+2,lidx)-in(lidz-3,lidx))+
                      HC4*(in(lidz+3,lidx)-in(lidz-4,lidx)));
#elif FDOH ==5
    out(lidz,lidx) = (HC1*(in(lidz,lidx)  -in(lidz-1,lidx))+
                      HC2*(in(lidz+1,lidx)-in(lidz-2,lidx))+
                      HC3*(in(lidz+2,lidx)-in(lidz-3,lidx))+
                      HC4*(in(lidz+3,lidx)-in(lidz-4,lidx))+
                      HC5*(in(lidz+4,lidx)-in(lidz-5,lidx)));
#elif FDOH ==6
    out(lidz,lidx) = (HC1*(in(lidz,lidx)  -in(lidz-1,lidx))+
                      HC2*(in(lidz+1,lidx)-in(lidz-2,lidx))+
                      HC3*(in(lidz+2,lidx)-in(lidz-3,lidx))+
                      HC4*(in(lidz+3,lidx)-in(lidz-4,lidx))+
                      HC5*(in(lidz+4,lidx)-in(lidz-5,lidx))+
                      HC6*(in(lidz+5,lidx)-in(lidz-6,lidx)));
#endif
}

void dzm(float out, float in, int lidx, int lidz){
    #if   FDOH ==1
    out(lidz,lidx) = HC1*(in(lidz+1,lidx) - in(lidz,lidx));
    #elif FDOH ==2
    out(lidz,lidx) = (HC1*(in(lidz+1,lidx) - in(lidz,lidx))
                      +HC2*(in(lidz+2,lidx) - in(lidz-1,lidx)));
    #elif FDOH ==3
    out(lidz,lidx) = (HC1*(in(lidz+1,lidx)-in(lidz,lidx))+
                      HC2*(in(lidz+2,lidx)-in(lidz-1,lidx))+
                      HC3*(in(lidz+3,lidx)-in(lidz-2,lidx)));
    #elif FDOH ==4
    out(lidz,lidx) = (HC1*(in(lidz+1,lidx)-in(lidz,lidx))+
                      HC2*(in(lidz+2,lidx)-in(lidz-1,lidx))+
                      HC3*(in(lidz+3,lidx)-in(lidz-2,lidx))+
                      HC4*(in(lidz+4,lidx)-in(lidz-3,lidx)));
    #elif FDOH ==5
    out(lidz,lidx) = (HC1*(in(lidz+1,lidx)-in(lidz,lidx))+
                      HC2*(in(lidz+2,lidx)-in(lidz-1,lidx))+
                      HC3*(in(lidz+3,lidx)-in(lidz-2,lidx))+
                      HC4*(in(lidz+4,lidx)-in(lidz-3,lidx))+
                      HC5*(in(lidz+5,lidx)-in(lidz-4,lidx)));
    #elif FDOH ==6
    out(lidz,lidx) = (HC1*(in(lidz+1,lidx)-in(lidz,lidx))+
                      HC2*(in(lidz+2,lidx)-in(lidz-1,lidx))+
                      HC3*(in(lidz+3,lidx)-in(lidz-2,lidx))+
                      HC4*(in(lidz+4,lidx)-in(lidz-3,lidx))+
                      HC5*(in(lidz+5,lidx)-in(lidz-4,lidx))+
                      HC6*(in(lidz+6,lidx)-in(lidz-5,lidx)));
    #endif
}

void dzm(float out, float in, int lidx, int lidz){
    #if   FDOH ==1
    out(lidz,lidx) = HC1*(in(lidz,lidx+1) - in(lidz,lidx));
    #elif FDOH ==2
    out(lidz,lidx) = (HC1*(in(lidz,lidx+1) - in(lidz,lidx))
                      +HC2*(in(lidz,lidx+2) - in(lidz,lidx-1)));
    #elif FDOH ==3
    out(lidz,lidx) = (HC1*(in(lidz,lidx+1)-in(lidz,lidx))+
                      HC2*(in(lidz,lidx+2)-in(lidz,lidx-1))+
                      HC3*(in(lidz,lidx+3)-in(lidz,lidx-2)));
    #elif FDOH ==4
    out(lidz,lidx) = (HC1*(in(lidz,lidx+1)-in(lidz,lidx))+
                      HC2*(in(lidz,lidx+2)-in(lidz,lidx-1))+
                      HC3*(in(lidz,lidx+3)-in(lidz,lidx-2))+
                      HC4*(in(lidz,lidx+4)-in(lidz,lidx-3)));
    #elif FDOH ==5
    out(lidz,lidx) = (HC1*(in(lidz,lidx+1)-in(lidz,lidx))+
                      HC2*(in(lidz,lidx+2)-in(lidz,lidx-1))+
                      HC3*(in(lidz,lidx+3)-in(lidz,lidx-2))+
                      HC4*(in(lidz,lidx+4)-in(lidz,lidx-3))+
                      HC5*(in(lidz,lidx+5)-in(lidz,lidx-4)));
    #elif FDOH ==6
    out(lidz,lidx) = (HC1*(in(lidz,lidx+1)-in(lidz,lidx))+
                      HC2*(in(lidz,lidx+2)-in(lidz,lidx-1))+
                      HC3*(in(lidz,lidx+3)-in(lidz,lidx-2))+
                      HC4*(in(lidz,lidx+4)-in(lidz,lidx-3))+
                      HC5*(in(lidz,lidx+5)-in(lidz,lidx-4))+
                      HC6*(in(lidz,lidx+6)-in(lidz,lidx-5)));
    #endif
}
void dzp(float out, float in, int lidx, int lidz){
    #if   FDOH ==1
    out(lidz,lidx) = HC1*(in(lidz,lidx)   - in(lidz,lidx-1));
    #elif FDOH ==2
    out(lidz,lidx) = (HC1*(in(lidz,lidx)   - in(lidz,lidx-1))
                      +HC2*(in(lidz,lidx+1) - in(lidz,lidx-2)));
    #elif FDOH ==3
    out(lidz,lidx) = (HC1*(in(lidz,lidx)  -in(lidz,lidx-1))+
                      HC2*(in(lidz,lidx+1)-in(lidz,lidx-2))+
                      HC3*(in(lidz,lidx+2)-in(lidz,lidx-3)));
    #elif FDOH ==4
    out(lidz,lidx) = (HC1*(in(lidz,lidx)  -in(lidz,lidx-1))+
                      HC2*(in(lidz,lidx+1)-in(lidz,lidx-2))+
                      HC3*(in(lidz,lidx+2)-in(lidz,lidx-3))+
                      HC4*(in(lidz,lidx+3)-in(lidz,lidx-4)));
    #elif FDOH ==5
    out(lidz,lidx) = (HC1*(in(lidz,lidx)  -in(lidz,lidx-1))+
                      HC2*(in(lidz,lidx+1)-in(lidz,lidx-2))+
                      HC3*(in(lidz,lidx+2)-in(lidz,lidx-3))+
                      HC4*(in(lidz,lidx+3)-in(lidz,lidx-4))+
                      HC5*(in(lidz,lidx+4)-in(lidz,lidx-5)));
    #elif FDOH ==6
    out(lidz,lidx) = (HC1*(in(lidz,lidx)  -in(lidz,lidx-1))+
                      HC2*(in(lidz,lidx+1)-in(lidz,lidx-2))+
                      HC3*(in(lidz,lidx+2)-in(lidz,lidx-3))+
                      HC4*(in(lidz,lidx+3)-in(lidz,lidx-4))+
                      HC5*(in(lidz,lidx+4)-in(lidz,lidx-5))+
                      HC6*(in(lidz,lidx+5)-in(lidz,lidx-6)));
    #endif
}

FUNDEF void update_p(int offcomm, int nt,
                     GLOBARG float *p1,         GLOBARG float *p2,
                     GLOBARG float *rip,     GLOBARG float *rkp,
                     GLOBARG float *M,
                     GLOBARG float *taper,
                     GLOBARG float *K_x,        GLOBARG float *a_x,          GLOBARG float *b_x,
                     GLOBARG float *K_x_half,   GLOBARG float *a_x_half,     GLOBARG float *b_x_half,
                     GLOBARG float *K_z,        GLOBARG float *a_z,          GLOBARG float *b_z,
                     GLOBARG float *K_z_half,   GLOBARG float *a_z_half,     GLOBARG float *b_z_half,
                     GLOBARG float *psi_p_x,  GLOBARG float *psi_p_z,
                     LOCARG)
{
    
    LOCDEF
    float p_xx;
    float p_zz;
    float * pflip;
    
    if (nt % 2 == 1){
        pflip = p1;
        p1 = p2;
        p2 = pflip;
    }
    
    
    // If we use local memory
#ifdef __OPENCL_VERSION__
    int lsizez = get_local_size(0)+4*FDOH;
    int lsizex = get_local_size(1)+4*FDOH;
    int lidz = get_local_id(0)+2*FDOH;
    int lidx = get_local_id(1)+2*FDOH;
    int gidz = get_global_id(0)+2*FDOH;
    int gidx = get_global_id(1)+2*FDOH+offcomm;
#else
    int lsizez = blockDim.x+4*FDOH;
    int lsizex = blockDim.y+4*FDOH;
    int lidz = threadIdx.x+2*FDOH;
    int lidx = threadIdx.y+2*FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+2*FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+2*FDOH+offcomm;
#endif
    
#define lp1 lvar
 
    
    
    // Calculation of the stresses spatial derivatives
    {

        BARRIER
        lp(lidz,lidx)=p1(gidz,gidx);
        if (lidx<4*FDOH){
            lp(lidz,lidx-2*FDOH)=lp1(gidz,gidx-2*FDOH);
        }
        if (lidx+lsizex-6*FDOH<2*FDOH){
            lp(lidz,lidx+lsizex-6*FDOH)=lp1(gidz,gidx+lsizex-6*FDOH);
        }
        if (lidx>(lsizex-4*FDOH-1)){
            lp(lidz,lidx+2*FDOH)=lp1(gidz,gidx+2*FDOH);
        }
        if (lidx-lsizex+6*FDOH>(lsizex-2*FDOH-1)){
            lp(lidz,lidx-lsizex+6*FDOH)=lp1(gidz,gidx-lsizex+6*FDOH);
        }
        BARRIER
        
        dxp(lpx, lp, rip, lidx, lidz);
        lpx(lidz,lidx)*=rip(gidz,gidx);
        if (lidx<2*FDOH){
            dxp(lpx, lp, lidx-FDOH, lidz);
            lpx(lidz,lidx-FDOH)*=rip(gidz,gidx-FDOH);
        }
        if (lidx+lsizex-3*FDOH<FDOH){
            dxp(lpx, lp, lidx-3*FDOH, lidz);
            lpx(lidz,lidx-3*FDOH)*=rip(gidz,gidx-3*FDOH);
        }
        if (lidx>(lsizex-2*FDOH-1)){
            dxp(lpx, lp, lidx+FDOH, lidz);
            lpx(lidz,lidx+FDOH)*=rip(gidz,gidx+FDOH);
        }
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1)){
            dxp(lpx, lp, lidx+3*FDOH, lidz);
            lpx(lidz,lidx+3*FDOH)*=rip(gidz,gidx+3*FDOH);
        }
        
        dzp(lpz, lp, rkp, lidx, lisz);
        lpz(lidz,lidx)*=rkp(gidz,gidx);
        if (lidz<2*FDOH){
            dzp(lpz, lp, lidx, lidz-FDOH);
            lpz(lidz-FDOH,lidx)*=rkp(gidz-FDOH,gidx);
        }
        if (lidz+lsizez-3*FDOH<FDOH){
            dzp(lpz, lp, lidx, lidz-3*FDOH);
            lpz(lidz-3*FDOH,lidx)*=rkp(gidz-3*FDOH,gidx);
        }
        if (lidz>(lsizez-2*FDOH-1)){
            dzp(lpz, lp, lidx, lidz+FDOH);
            lpz(lidz+FDOH,lidx)*=rkp(gidz+FDOH,gidx);
        }
        if (lidz-lsizez+3*FDOH>(lsizez-FDOH-1)){
            dzp(lpz, lp, lidx, lidz+3*FDOH);
            lpz(lidz+3*FDOH,lidx)*=rkp(gidz+3*FDOH,gidx);
        }
        
        BARRIER
        
        dxm(&p_xx, lpx, lidx, lidz);
        dzm(&p_zz, lpz, lidx, lidz);

    }
    
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
    
    
    // Update the pressure
    {
        p2(gidz,gidx) = 2*lp(lidz,lidx) - p2(gidz,gidx) + M(gidz,gidx)*(p_xx + p_zz);
    }
    
    // Absorbing boundary
#if ABS_TYPE==2
    {
#if FREESURF==0
        if (gidz-FDOH<NAB){
            p2(gidz,gidx)*=taper[gidz-FDOH];
        }
#endif
        
        if (gidz>NZ-NAB-FDOH-1){
            p2(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            p2(gidz,gidx)*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            p2(gidz,gidx)*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    
    
}


