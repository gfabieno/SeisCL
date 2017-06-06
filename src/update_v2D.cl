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

/*Update of the velocity in 2D SV*/

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

#define vx(z,x)  vx[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vy(z,x)  vy[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz(z,x)  vz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxx(z,x) sxx[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define szz(z,x) szz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxz(z,x) sxz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxy(z,x) sxy[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syz(z,x) syz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxy(z,x,l) rxy[(l)*NX*NZ+(x)*NZ+(z)]
#define ryz(z,x,l) ryz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*fdoh)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*fdoh)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*nab)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*nab)+(z)]
#define psi_sxy_x(z,x) psi_sxy_x[(x)*(NZ-2*fdoh)+(z)]
#define psi_syz_z(z,x) psi_syz_z[(x)*(2*nab)+(z)]

#if local_off==0

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
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]






float2 ssource(int gidz, int gidx,  int nsrc, __global float *srcpos_loc, __global float *signals, int nt, __global float * rip, __global float * rkp){
    
    float2 ampv={0.0,0.0};
    int i,k;
    if (nsrc>0){

        for (int srci=0; srci<nsrc; srci++){

            i=(int)(srcpos_loc(0,srci)/DH-0.5)+fdoh;
            k=(int)(srcpos_loc(2,srci)/DH-0.5)+fdoh;
            
            if (i==gidx && k==gidz){
                
                float amp=(DT*signals(srci,nt))/(DH*DH); // scaled force amplitude with F= 1N
                
                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
                
                if (SOURCE_TYPE==2){
                    /* single force in x */
                    ampv.x  +=  amp/rip(k,i-offset);
                }
                else if (SOURCE_TYPE==4){
                    /* single force in z */
                    
                    ampv.y  +=  amp/rkp(k,i-offset);
                }
                
                if (SOURCE_TYPE==2){
                    /* single force in x */
                    ampv.x  +=  amp;
                }
                else if (SOURCE_TYPE==4){
                    /* single force in z */
                    
                    ampv.y  +=  amp;
                }
                
            }
        }
    
        
    }
    
    return ampv;
    
}


__kernel void update_v(int offcomm,int nsrc,  int nt,
                       __global float *vx,      __global float *vz,
                       __global float *sxx,     __global float *szz,     __global float *sxz,
                       __global float *rip,     __global float *rkp,     __global float *srcpos_loc,
                       __global float *signals, __global float *rec_pos, __global float *taper,
                       __global float *K_x,        __global float *a_x,          __global float *b_x,
                       __global float *K_x_half,   __global float *a_x_half,     __global float *b_x_half,
                       __global float *K_z,        __global float *a_z,          __global float *b_z,
                       __global float *K_z_half,   __global float *a_z_half,     __global float *b_z_half,
                       __global float *psi_sxx_x,  __global float *psi_sxz_x,
                       __global float *psi_sxz_z,  __global float *psi_szz_z,
                       __local float *lvar)
{

    float sxx_x;
    float szz_z;
    float sxz_x;
    float sxz_z;

// If we use local memory
#if local_off==0
    int lsizez = get_local_size(0)+2*fdoh;
    int lsizex = get_local_size(1)+2*fdoh;
    int lidz = get_local_id(0)+fdoh;
    int lidx = get_local_id(1)+fdoh;
    int gidz = get_global_id(0)+fdoh;
    int gidx = get_global_id(1)+fdoh+offcomm;
    
    #define lsxx lvar
    #define lszz lvar
    #define lsxz lvar
    
// If local memory is turned off
#elif local_off==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*fdoh);
    int gidz = gid%glsizez+fdoh;
    int gidx = (gid/glsizez)+fdoh+offcomm;
    
#define lsxx sxx
#define lszz szz
#define lsxz sxz
#define lidx gidx
#define lidz gidz
    
#endif
    
// Calculation of the stresses spatial derivatives
    {
#if local_off==0
        lsxx(lidz,lidx)=sxx(gidz, gidx);
        if (lidx<2*fdoh)
            lsxx(lidz,lidx-fdoh)=sxx(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxx(lidz,lidx+lsizex-3*fdoh)=sxx(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxx(lidz,lidx+fdoh)=sxx(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxx(lidz,lidx-lsizex+3*fdoh)=sxx(gidz,gidx-lsizex+3*fdoh);
        
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        sxx_x = dtdh*hc1*(lsxx(lidz,lidx+1) - lsxx(lidz,lidx));
#elif fdoh ==2
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1) - lsxx(lidz,lidx))
                      +hc2*(lsxx(lidz,lidx+2) - lsxx(lidz,lidx-1)));
#elif fdoh ==3
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      hc2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      hc3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2)));
#elif fdoh ==4
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      hc2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      hc3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
                      hc4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3)));
#elif fdoh ==5
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      hc2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      hc3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
                      hc4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3))+
                      hc5*(lsxx(lidz,lidx+5)-lsxx(lidz,lidx-4)));
#elif fdoh ==6
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      hc2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      hc3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
                      hc4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3))+
                      hc5*(lsxx(lidz,lidx+5)-lsxx(lidz,lidx-4))+
                      hc6*(lsxx(lidz,lidx+6)-lsxx(lidz,lidx-5)));
#endif
        
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lszz(lidz,lidx)=szz(gidz, gidx);
        if (lidz<2*fdoh)
            lszz(lidz-fdoh,lidx)=szz(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lszz(lidz+fdoh,lidx)=szz(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        szz_z = dtdh*hc1*(lszz(lidz+1,lidx) - lszz(lidz,lidx));
#elif fdoh ==2
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx) - lszz(lidz,lidx))
                      +hc2*(lszz(lidz+2,lidx) - lszz(lidz-1,lidx)));
#elif fdoh ==3
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      hc2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      hc3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx)));
#elif fdoh ==4
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      hc2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      hc3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
                      hc4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx)));
#elif fdoh ==5
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      hc2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      hc3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
                      hc4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx))+
                      hc5*(lszz(lidz+5,lidx)-lszz(lidz-4,lidx)));
#elif fdoh ==6
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      hc2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      hc3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
                      hc4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx))+
                      hc5*(lszz(lidz+5,lidx)-lszz(lidz-4,lidx))+
                      hc6*(lszz(lidz+6,lidx)-lszz(lidz-5,lidx)));
#endif
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsxz(lidz,lidx)=sxz(gidz, gidx);
        
        if (lidx<2*fdoh)
            lsxz(lidz,lidx-fdoh)=sxz(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxz(lidz,lidx+lsizex-3*fdoh)=sxz(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxz(lidz,lidx+fdoh)=sxz(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxz(lidz,lidx-lsizex+3*fdoh)=sxz(gidz,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lsxz(lidz-fdoh,lidx)=sxz(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lsxz(lidz+fdoh,lidx)=sxz(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        sxz_z = dtdh*hc1*(lsxz(lidz,lidx)   - lsxz(lidz-1,lidx));
        sxz_x = dtdh*hc1*(lsxz(lidz,lidx)   - lsxz(lidz,lidx-1));
#elif fdoh ==2
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)   - lsxz(lidz-1,lidx))
                      +hc2*(lsxz(lidz+1,lidx) - lsxz(lidz-2,lidx)));
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)   - lsxz(lidz,lidx-1))
                      +hc2*(lsxz(lidz,lidx+1) - lsxz(lidz,lidx-2)));
#elif fdoh ==3
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      hc2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      hc3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      hc2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      hc3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3)));
#elif fdoh ==4
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      hc2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      hc3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
                      hc4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      hc2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      hc3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
                      hc4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4)));
#elif fdoh ==5
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      hc2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      hc3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
                      hc4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx))+
                      hc5*(lsxz(lidz+4,lidx)-lsxz(lidz-5,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      hc2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      hc3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
                      hc4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4))+
                      hc5*(lsxz(lidz,lidx+4)-lsxz(lidz,lidx-5)));
#elif fdoh ==6
        
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      hc2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      hc3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
                      hc4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx))+
                      hc5*(lsxz(lidz+4,lidx)-lsxz(lidz-5,lidx))+
                      hc6*(lsxz(lidz+5,lidx)-lsxz(lidz-6,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      hc2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      hc3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
                      hc4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4))+
                      hc5*(lsxz(lidz,lidx+4)-lsxz(lidz,lidx-5))+
                      hc6*(lsxz(lidz,lidx+5)-lsxz(lidz,lidx-6)));
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
        int i,k,ind;
        
        if (gidz>NZ-nab-fdoh-1){
            
            i =gidx-fdoh;
            k =gidz - NZ+nab+fdoh+nab;
            ind=2*nab-1-k;
            
            psi_sxz_z(k,i) = b_z[ind+1] * psi_sxz_z(k,i) + a_z[ind+1] * sxz_z;
            sxz_z = sxz_z / K_z[ind+1] + psi_sxz_z(k,i);
            psi_szz_z(k,i) = b_z_half[ind] * psi_szz_z(k,i) + a_z_half[ind] * szz_z;
            szz_z = szz_z / K_z_half[ind] + psi_szz_z(k,i);
            
        }
        
#if freesurf==0
        else if (gidz-fdoh<nab){
            
            i =gidx-fdoh;
            k =gidz-fdoh;
            
            psi_sxz_z(k,i) = b_z[k] * psi_sxz_z(k,i) + a_z[k] * sxz_z;
            sxz_z = sxz_z / K_z[k] + psi_sxz_z(k,i);
            psi_szz_z(k,i) = b_z_half[k] * psi_szz_z(k,i) + a_z_half[k] * szz_z;
            szz_z = szz_z / K_z_half[k] + psi_szz_z(k,i);
            
        }
#endif
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            
            i =gidx-fdoh;
            k =gidz-fdoh;
            
            psi_sxx_x(k,i) = b_x_half[i] * psi_sxx_x(k,i) + a_x_half[i] * sxx_x;
            sxx_x = sxx_x / K_x_half[i] + psi_sxx_x(k,i);
            psi_sxz_x(k,i) = b_x[i] * psi_sxz_x(k,i) + a_x[i] * sxz_x;
            sxz_x = sxz_x / K_x[i] + psi_sxz_x(k,i);
            
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            
            i =gidx - NX+nab+fdoh+nab;
            k =gidz-fdoh;
            ind=2*nab-1-i;
            
            psi_sxx_x(k,i) = b_x_half[ind] * psi_sxx_x(k,i) + a_x_half[ind] * sxx_x;
            sxx_x = sxx_x / K_x_half[ind] + psi_sxx_x(k,i);
            psi_sxz_x(k,i) = b_x[ind+1] * psi_sxz_x(k,i) + a_x[ind+1] * sxz_x;
            sxz_x = sxz_x / K_x[ind+1] + psi_sxz_x(k,i);
            
        }
#endif
    }
#endif

// Update the velocities
    {
        float2 amp = ssource(gidz, gidx+offset, nsrc, srcpos_loc, signals, nt, rip, rkp);
        vx(gidz,gidx)+= ((sxx_x + sxz_z)/rip(gidz,gidx))+amp.x;
        vz(gidz,gidx)+= ((szz_z + sxz_x)/rkp(gidz,gidx))+amp.y;
    }
    
// Absorbing boundary
#if abs_type==2
    {
        if (gidz-fdoh<nab){
            vx(gidz,gidx)*=taper[gidz-fdoh];
            vz(gidz,gidx)*=taper[gidz-fdoh];
        }
        
        if (gidz>NZ-nab-fdoh-1){
            vx(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
            vz(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
        }
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            vx(gidz,gidx)*=taper[gidx-fdoh];
            vz(gidz,gidx)*=taper[gidx-fdoh];
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            vx(gidz,gidx)*=taper[NX-fdoh-gidx-1];
            vz(gidz,gidx)*=taper[NX-fdoh-gidx-1];
        }
#endif 
    }
#endif
    
}

