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

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*NAB)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*NAB)+(z)]


FUNDEF void update_adjv(int offcomm,
                          GLOBARG float*vx,         GLOBARG float*vz,
                          GLOBARG float*sxx,        GLOBARG float*szz,
                          GLOBARG float*sxz,
                          GLOBARG float*vxbnd,      GLOBARG float*vzbnd,
                          GLOBARG float*sxxbnd,     GLOBARG float*szzbnd,
                          GLOBARG float*sxzbnd,
                          GLOBARG float*vxr,       GLOBARG float*vzr,
                          GLOBARG float*sxxr,      GLOBARG float*szzr,
                          GLOBARG float*sxzr,
                          GLOBARG float*rip,        GLOBARG float*rkp,
                          GLOBARG float*taper,
                          GLOBARG float*K_x,        GLOBARG float*a_x,          GLOBARG float*b_x,
                          GLOBARG float*K_x_half,   GLOBARG float*a_x_half,     GLOBARG float*b_x_half,
                          GLOBARG float*K_z,        GLOBARG float*a_z,          GLOBARG float*b_z,
                          GLOBARG float*K_z_half,   GLOBARG float*a_z_half,     GLOBARG float*b_z_half,
                          GLOBARG float*psi_sxx_x,  GLOBARG float*psi_sxz_x,
                          GLOBARG float*psi_sxz_z,  GLOBARG float*psi_szz_z,
                          GLOBARG float*gradrho, GLOBARG float*gradsrc,
                          GLOBARG float*Hrho,    GLOBARG float*Hsrc,
                        LOCARG)
{

    LOCDEF
    
    int g,i,j,k,m;
    float sxx_xr;
    float szz_zr;
    float sxz_xr;
    float sxz_zr;
    float sxx_x;
    float szz_z;
    float sxz_x;
    float sxz_z;
    float lvx;
    float lvz;
    
// If we use local memory
#if LOCAL_OFF==0
    
#ifdef __OPENCL_VERSION__
    int lsizez = get_local_size(0)+2*FDOH;
    int lsizex = get_local_size(1)+2*FDOH;
    int lidz = get_local_id(0)+FDOH;
    int lidx = get_local_id(1)+FDOH;
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH+offcomm;
#else
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
#endif

#define lsxx lvar
#define lszz lvar
#define lsxz lvar
    
#define lsxxr lvar
#define lszzr lvar
#define lsxzr lvar
 
// If local memory is turned off
#elif LOCAL_OFF==1
    
#ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    int glsizez = (NZ-2*FDOH);
    int gidz = gid%glsizez+FDOH;
    int gidx = (gid/glsizez)+FDOH+offcomm;
#else
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
#endif
    
#define lsxx sxx
#define lszz szz
#define lsxz sxz
    
#define lsxxr sxxr
#define lszzr szzr
#define lsxzr sxzr
    
#define lidx gidx
#define lidz gidz
    
#define lsizez NZ
#define lsizex NX
    
#endif
    
    int indp = (gidx-FDOH)*(NZ-2*FDOH)+(gidz-FDOH);
    int indv = gidx*NZ+gidz;
    
// Calculation of the stress spatial derivatives of the forward wavefield if backpropagation is used
    #if BACK_PROP_TYPE==1
    {
    #if LOCAL_OFF==0
        load_local_in(sxx);
        load_local_halox(sxx);
        BARRIER
    #endif
        sxx_x = Dxp(lsxx);
        
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(szz);
        load_local_haloz(szz);
        BARRIER
    #endif
        szz_z = Dzp(lszz);
        
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxz);
        load_local_haloz(sxz);
        load_local_halox(sxz);
        BARRIER
    #endif
        sxz_z = Dzm(lsxz);
        sxz_x = Dxm(lsxz);
        BARRIER
    }
    #endif

// Calculation of the stress spatial derivatives of the adjoint wavefield
    #if LOCAL_OFF==0
        load_local_in(sxxr);
        load_local_halox(sxxr);
        BARRIER
    #endif
        sxx_xr = Dxp(lsxxr);
    
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(szzr);
        load_local_haloz(szzr);
        BARRIER
    #endif
        szz_zr = Dzp(lszzr);
    
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxzr);
        load_local_haloz(sxzr);
        load_local_halox(sxzr);
        BARRIER
    #endif
        sxz_zr = Dzm(lsxzr);
        sxz_xr = Dxm(lsxzr);


    
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

        lvx=((sxx_x + sxz_z)*rip[indp]);
        lvz=((szz_z + sxz_x)*rkp[indp]);
        vx[indv]-= lvx;
        vz[indv]-= lvz;
        
        // Inject the boundary values
        m=inject_ind(gidz,  gidx);
        if (m!=-1){
            vx[indv]= vxbnd[m];
            vz[indv]= vzbnd[m];
            
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
        
        psi_sxz_z(k,i) = b_z[ind+1] * psi_sxz_z(k,i) + a_z[ind+1] * sxz_zr;
        sxz_zr = sxz_zr / K_z[ind+1] + psi_sxz_z(k,i);
        psi_szz_z(k,i) = b_z_half[ind] * psi_szz_z(k,i) + a_z_half[ind] * szz_zr;
        szz_zr = szz_zr / K_z_half[ind] + psi_szz_z(k,i);
        
    }
    
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        
        i =gidx-FDOH;
        k =gidz-FDOH;
        
        psi_sxz_z(k,i) = b_z[k] * psi_sxz_z(k,i) + a_z[k] * sxz_zr;
        sxz_zr = sxz_zr / K_z[k] + psi_sxz_z(k,i);
        psi_szz_z(k,i) = b_z_half[k] * psi_szz_z(k,i) + a_z_half[k] * szz_zr;
        szz_zr = szz_zr / K_z_half[k] + psi_szz_z(k,i);
        
    }
#endif
    
#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        
        i =gidx-FDOH;
        k =gidz-FDOH;
        
        psi_sxx_x(k,i) = b_x_half[i] * psi_sxx_x(k,i) + a_x_half[i] * sxx_xr;
        sxx_xr = sxx_xr / K_x_half[i] + psi_sxx_x(k,i);
        psi_sxz_x(k,i) = b_x[i] * psi_sxz_x(k,i) + a_x[i] * sxz_xr;
        sxz_xr = sxz_xr / K_x[i] + psi_sxz_x(k,i);
        
    }
#endif
    
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        
        i =gidx - NX+NAB+FDOH+NAB;
        k =gidz-FDOH;
        ind=2*NAB-1-i;
        
        psi_sxx_x(k,i) = b_x_half[ind] * psi_sxx_x(k,i) + a_x_half[ind] * sxx_xr;
        sxx_xr = sxx_xr / K_x_half[ind] + psi_sxx_x(k,i);
        psi_sxz_x(k,i) = b_x[ind+1] * psi_sxz_x(k,i) + a_x[ind+1] * sxz_xr;
        sxz_xr = sxz_xr / K_x[ind+1] + psi_sxz_x(k,i);
        
    }
#endif
    }
#endif
    
    // Update adjoint velocities
    lvx=((sxx_xr + sxz_zr)*rip[indp]);
    lvz=((szz_zr + sxz_xr)*rkp[indp]);
    vxr[indv]+= lvx;
    vzr[indv]+= lvz;
 
    

// Absorbing boundary
#if ABS_TYPE==2
    {
#if FREESURF==0
    if (gidz-FDOH<NAB){
        vxr[indv]*=taper[gidz-FDOH];
        vzr[indv]*=taper[gidz-FDOH];
    }
#endif
    
    if (gidz>NZ-NAB-FDOH-1){
        vxr[indv]*=taper[NZ-FDOH-gidz-1];
        vzr[indv]*=taper[NZ-FDOH-gidz-1];
    }
    
#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        vxr[indv]*=taper[gidx-FDOH];
        vzr[indv]*=taper[gidx-FDOH];
    }
#endif
    
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        vxr[indv]*=taper[NX-FDOH-gidx-1];
        vzr[indv]*=taper[NX-FDOH-gidx-1];
    }
#endif
    }
#endif
    
    
// Density gradient calculation on the fly
#if BACK_PROP_TYPE==1
    gradrho[indp]+=-vx[indv]*lvx-vz[indv]*lvz;
    
//#if HOUT==1
//    Hrho[indp]+= pown(vx[indv],2)+pown(vz[indv],2);
//#endif

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
//                    gradsrc(srci,nt)+= vxr[indv]/rip(gidx,gidz)/(DH*DH);
//                }
//                else if (SOURCE_TYPE==4){
//                    /* single force in z */
//                    
//                    gradsrc(srci,nt)+= vzr[indv]/rkp(gidx,gidz)/(DH*DH);
//                }
//                
//            }
//        }
//        
//        
//    }
#endif
    
}

