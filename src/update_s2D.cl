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

/*Update of the stresses in 2D SV*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define lbnd (fdoh+nab)

#define rho(z,x)    rho[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rip(z,x)    rip[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rjp(z,x)    rjp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,x)    rkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipkp(z,x) uipkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define u(z,x)        u[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define pi(z,x)      pi[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradrho(z,x)  gradrho[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradM(z,x)  gradM[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradmu(z,x)  gradmu[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaup(z,x)  gradtaup[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaus(z,x)  gradtaus[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define taus(z,x)        taus[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipkp(z,x) tausipkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define taup(z,x)        taup[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define vx(z,x)  vx[(x)*NZ+(z)]
#define vz(z,x)  vz[(x)*NZ+(z)]
#define sxx(z,x) sxx[(x)*NZ+(z)]
#define szz(z,x) szz[(x)*NZ+(z)]
#define sxz(z,x) sxz[(x)*NZ+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_vxx(z,x) psi_vxx[(x)*(NZ-2*fdoh)+(z)]
#define psi_vzx(z,x) psi_vzx[(x)*(NZ-2*fdoh)+(z)]

#define psi_vxz(z,x) psi_vxz[(x)*(2*nab)+(z)]
#define psi_vzz(z,x) psi_vzz[(x)*(2*nab)+(z)]


#if local_off==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

#endif


#define vxout(y,x) vxout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define PI (3.141592653589793238462643383279502884197169)
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]



__kernel void update_s(int offcomm, int nsrc,  int nt,
                       __global float *vx,         __global float *vz,
                       __global float *sxx,        __global float *szz,        __global float *sxz,
                       __global float *pi,         __global float *u,          __global float *uipkp,
                       __global float *rxx,        __global float *rzz,        __global float *rxz,
                       __global float *taus,       __global float *tausipkp,   __global float *taup,
                       __global float *eta,        __global float *srcpos_loc, __global float *signals,
                       __global float *taper,
                       __global float *K_x,        __global float *a_x,          __global float *b_x,
                       __global float *K_x_half,   __global float *a_x_half,     __global float *b_x_half,
                       __global float *K_z,        __global float *a_z,          __global float *b_z,
                       __global float *K_z_half,   __global float *a_z_half,     __global float *b_z_half,
                       __global float *psi_vxx,    __global float *psi_vxz,
                       __global float *psi_vzx,    __global float *psi_vzz,
                       __local  float *lvar)
{
    
    float vxx, vzz, vzx, vxz;
    int i,k,l,ind;
    float sumrxz, sumrxx, sumrzz;
    float b,c,e,g,d,f,fipkp,dipkp;
    float leta[Lve];
    float lpi, lu, luipkp, ltaup, ltaus, ltausipkp;
    float lsxx, lszz, lsxz;
    
// If we use local memory
#if local_off==0
    int lsizez = get_local_size(0)+2*fdoh;
    int lsizex = get_local_size(1)+2*fdoh;
    int lidz = get_local_id(0)+fdoh;
    int lidx = get_local_id(1)+fdoh;
    int gidz = get_global_id(0)+fdoh;
    int gidx = get_global_id(1)+fdoh+offcomm;

#define lvx lvar
#define lvz lvar
    
// If local memory is turned off
#elif local_off==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*fdoh);
    int gidz = gid%glsizez+fdoh;
    int gidx = (gid/glsizez)+fdoh+offcomm;
    
    
#define lvx vx
#define lvz vz
#define lidx gidx
#define lidz gidz
    
#endif

// Calculation of the velocity spatial derivatives
    {
#if local_off==0
        lvx(lidz,lidx)=vx(gidz, gidx);
        if (lidx<2*fdoh)
            lvx(lidz,lidx-fdoh)=vx(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvx(lidz,lidx+lsizex-3*fdoh)=vx(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvx(lidz,lidx+fdoh)=vx(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvx(lidz,lidx-lsizex+3*fdoh)=vx(gidz,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvx(lidz-fdoh,lidx)=vx(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvx(lidz+fdoh,lidx)=vx(gidz+fdoh,gidx);
        
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh==1
        vxx = hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))/DH;
        vxz = hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))/DH;
#elif fdoh==2
        vxx = (  hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               )/DH;
#elif fdoh==3
        vxx = (  hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + hc3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + hc3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               )/DH;
#elif fdoh==4
        vxx = (   hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + hc3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + hc4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + hc3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               + hc4*(lvx(lidz+4, lidx)-lvx(lidz-3, lidx))
               )/DH;
#elif fdoh==5
        vxx = (  hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + hc3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + hc4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               + hc5*(lvx(lidz, lidx+4)-lvx(lidz, lidx-5))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + hc3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               + hc4*(lvx(lidz+4, lidx)-lvx(lidz-3, lidx))
               + hc5*(lvx(lidz+5, lidx)-lvx(lidz-4, lidx))
               )/DH;
#elif fdoh==6
        vxx = (  hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + hc3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + hc4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               + hc5*(lvx(lidz, lidx+4)-lvx(lidz, lidx-5))
               + hc6*(lvx(lidz, lidx+5)-lvx(lidz, lidx-6))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + hc3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               + hc4*(lvx(lidz+4, lidx)-lvx(lidz-3, lidx))
               + hc5*(lvx(lidz+5, lidx)-lvx(lidz-4, lidx))
               + hc6*(lvx(lidz+6, lidx)-lvx(lidz-5, lidx))
               )/DH;
#endif
        
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvz(lidz,lidx)=vz(gidz, gidx);
        if (lidx<2*fdoh)
            lvz(lidz,lidx-fdoh)=vz(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvz(lidz,lidx+lsizex-3*fdoh)=vz(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvz(lidz,lidx+fdoh)=vz(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvz(lidz,lidx-lsizex+3*fdoh)=vz(gidz,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvz(lidz-fdoh,lidx)=vz(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvz(lidz+fdoh,lidx)=vz(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh==1
        vzz = hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))/DH;
        vzx = hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))/DH;
#elif fdoh==2
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               )/DH;
#elif fdoh==3
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + hc3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + hc3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               )/DH;
#elif fdoh==4
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + hc3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + hc4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + hc3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               + hc4*(lvz(lidz, lidx+4)-lvz(lidz, lidx-3))
               )/DH;
#elif fdoh==5
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + hc3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + hc4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               + hc5*(lvz(lidz+4, lidx)-lvz(lidz-5, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + hc3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               + hc4*(lvz(lidz, lidx+4)-lvz(lidz, lidx-3))
               + hc5*(lvz(lidz, lidx+5)-lvz(lidz, lidx-4))
               )/DH;
#elif fdoh==6
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + hc3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + hc4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               + hc5*(lvz(lidz+4, lidx)-lvz(lidz-5, lidx))
               + hc6*(lvz(lidz+5, lidx)-lvz(lidz-6, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + hc3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               + hc4*(lvz(lidz, lidx+4)-lvz(lidz, lidx-3))
               + hc5*(lvz(lidz, lidx+5)-lvz(lidz, lidx-4))
               + hc6*(lvz(lidz, lidx+6)-lvz(lidz, lidx-5))
               )/DH;
#endif
    }
    
    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if local_off==0
#if comm12==0
    if (gidz>(NZ-fdoh-1) || (gidx-offcomm)>(NX-fdoh-1-lcomm) ){
        return;
    }
#else
    if (gidz>(NZ-fdoh-1) ){
        return;
    }
#endif
#endif

    
// Correct spatial derivatives to implement CPML
    
#if abs_type==1
        {
        
        if (gidz>NZ-nab-fdoh-1){
            
            i =gidx-fdoh;
            k =gidz - NZ+nab+fdoh+nab;
            ind=2*nab-1-k;
            
            psi_vxz(k,i) = b_z_half[ind] * psi_vxz(k,i) + a_z_half[ind] * vxz;
            vxz = vxz / K_z_half[ind] + psi_vxz(k,i);
            psi_vzz(k,i) = b_z[ind+1] * psi_vzz(k,i) + a_z[ind+1] * vzz;
            vzz = vzz / K_z[ind+1] + psi_vzz(k,i);
            
        }
        
#if freesurf==0
        else if (gidz-fdoh<nab){
            
            i =gidx-fdoh;
            k =gidz-fdoh;
            
            
            psi_vxz(k,i) = b_z_half[k] * psi_vxz(k,i) + a_z_half[k] * vxz;
            vxz = vxz / K_z_half[k] + psi_vxz(k,i);
            psi_vzz(k,i) = b_z[k] * psi_vzz(k,i) + a_z[k] * vzz;
            vzz = vzz / K_z[k] + psi_vzz(k,i);
            
            
        }
#endif
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            
            i =gidx-fdoh;
            k =gidz-fdoh;
            
            psi_vxx(k,i) = b_x[i] * psi_vxx(k,i) + a_x[i] * vxx;
            vxx = vxx / K_x[i] + psi_vxx(k,i);
            psi_vzx(k,i) = b_x_half[i] * psi_vzx(k,i) + a_x_half[i] * vzx;
            vzx = vzx / K_x_half[i] + psi_vzx(k,i);
            
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            
            i =gidx - NX+nab+fdoh+nab;
            k =gidz-fdoh;
            ind=2*nab-1-i;
            
            
            psi_vxx(k,i) = b_x[ind+1] * psi_vxx(k,i) + a_x[ind+1] * vxx;
            vxx = vxx /K_x[ind+1] + psi_vxx(k,i);
            psi_vzx(k,i) = b_x_half[ind] * psi_vzx(k,i) + a_x_half[ind] * vzx;
            vzx = vzx / K_x_half[ind]  +psi_vzx(k,i);
            
            
        }
#endif
       }
#endif
    
    
// Read model parameters into local memory
    {
#if Lve==0
        fipkp=uipkp(gidz, gidx)*DT;
        f=2.0*u(gidz, gidx)*DT;
        g=pi(gidz, gidx)*DT;
        
#else
        
        lpi=pi(gidz,gidx);
        lu=u(gidz,gidx);
        luipkp=uipkp(gidz,gidx);
        ltaup=taup(gidz,gidx);
        ltaus=taus(gidz,gidx);
        ltausipkp=tausipkp(gidz,gidx);
        
        for (l=0;l<Lve;l++){
            leta[l]=eta[l];
        }
        
        fipkp=luipkp*DT*(1.0+ (float)Lve*ltausipkp);
        g=lpi*(1.0+(float)Lve*ltaup)*DT;
        f=2.0*lu*(1.0+(float)Lve*ltaus)*DT;
        dipkp=luipkp*ltausipkp;
        d=2.0*lu*ltaus;
        e=lpi*ltaup;
        
#endif
    }
    
// Update the stresses
    {
#if Lve==0
        
        sxz(gidz, gidx)+=(fipkp*(vxz+vzx));
        sxx(gidz, gidx)+=(g*(vxx+vzz))-(f*vzz) ;
        szz(gidz, gidx)+=(g*(vxx+vzz))-(f*vxx) ;
        
        
#else
        
        
        /* computing sums of the old memory variables */
        sumrxz=sumrxx=sumrzz=0;
        for (l=0;l<Lve;l++){
            sumrxz+=rxz(gidz,gidx,l);
            sumrxx+=rxx(gidz,gidx,l);
            sumrzz+=rzz(gidz,gidx,l);
        }
        
        
        /* updating components of the stress tensor, partially */
        lsxz=(fipkp*(vxz+vzx))+(dt2*sumrxz);
        lsxx=((g*(vxx+vzz))-(f*vzz))+(dt2*sumrxx);
        lszz=((g*(vxx+vzz))-(f*vxx))+(dt2*sumrzz);
        
        
        /* now updating the memory-variables and sum them up*/
        sumrxz=sumrxx=sumrzz=0;
        for (l=0;l<Lve;l++){
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            
            rxz(gidz,gidx,l)=b*(rxz(gidz,gidx,l)*c-leta[l]*(dipkp*(vxz+vzx)));
            rxx(gidz,gidx,l)=b*(rxx(gidz,gidx,l)*c-leta[l]*((e*(vxx+vzz))-(d*vzz)));
            rzz(gidz,gidx,l)=b*(rzz(gidz,gidx,l)*c-leta[l]*((e*(vxx+vzz))-(d*vxx)));
            
            sumrxz+=rxz(gidz,gidx,l);
            sumrxx+=rxx(gidz,gidx,l);
            sumrzz+=rzz(gidz,gidx,l);
        }
        
        
        /* and now the components of the stress tensor are
         completely updated */
        sxz(gidz, gidx)+= lsxz + (dt2*sumrxz);
        sxx(gidz, gidx)+= lsxx + (dt2*sumrxx);
        szz(gidz, gidx)+= lszz + (dt2*sumrzz);
        
#endif
    }
    
// Absorbing boundary
#if abs_type==2
    {
#if freesurf==0
        if (gidz-fdoh<nab){
            sxx(gidz,gidx)*=taper[gidz-fdoh];
            szz(gidz,gidx)*=taper[gidz-fdoh];
            sxz(gidz,gidx)*=taper[gidz-fdoh];
        }
#endif
        
        if (gidz>NZ-nab-fdoh-1){
            sxx(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
            szz(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
            sxz(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
        }
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            sxx(gidz,gidx)*=taper[gidx-fdoh];
            szz(gidz,gidx)*=taper[gidx-fdoh];
            sxz(gidz,gidx)*=taper[gidx-fdoh];
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            sxx(gidz,gidx)*=taper[NX-fdoh-gidx-1];
            szz(gidz,gidx)*=taper[NX-fdoh-gidx-1];
            sxz(gidz,gidx)*=taper[NX-fdoh-gidx-1];
        }
#endif
    }
#endif
    
}

