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
                          GLOBARG float * RESTRICT vx,              GLOBARG float * RESTRICT vz,
                          GLOBARG const float * RESTRICT sxx,       GLOBARG const float * RESTRICT szz,
                          GLOBARG const float * RESTRICT sxz,
                          GLOBARG const float * RESTRICT vxbnd,     GLOBARG const float * RESTRICT vzbnd,
                          GLOBARG const float * RESTRICT sxxbnd,    GLOBARG const float * RESTRICT szzbnd,
                          GLOBARG const float * RESTRICT sxzbnd,
                          GLOBARG float * RESTRICT vxr,             GLOBARG float * RESTRICT vzr,
                          GLOBARG const float * RESTRICT sxxr,      GLOBARG const float * RESTRICT szzr,
                          GLOBARG const float * RESTRICT sxzr,
                          GLOBARG const float * RESTRICT rip,       GLOBARG const float * RESTRICT rkp,
                          GLOBARG const float * RESTRICT taper,
                          GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                          GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                          GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                          GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                          GLOBARG float * RESTRICT psi_sxx_x,       GLOBARG float * RESTRICT psi_sxz_x,
                          GLOBARG float * RESTRICT psi_sxz_z,       GLOBARG float * RESTRICT psi_szz_z,
                          GLOBARG float * RESTRICT gradrho,         GLOBARG const float * RESTRICT gradsrc,
                          GLOBARG float * RESTRICT gradrip,         GLOBARG float * RESTRICT gradrkp,
                          GLOBARG float * RESTRICT Hrho,            GLOBARG const float * RESTRICT Hsrc,
                          GLOBARG const float * RESTRICT src,       GLOBARG const float * RESTRICT src_pos,
                          int nsrc,                                 int nt,
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
            vx[indv] = vxbnd[m];
            vz[indv] = vzbnd[m];
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
    gradrip[indp]+=-vx[indv]*lvx;
    gradrkp[indp]+=-vz[indv]*lvz;

    /* Source term of the misfit gradient -- GJI 2017 eq. (26a) -- for a FORCE
     * source. The accumulation just above is the (A phi' + B phi) side of
     * eq. (6) only: eq. (A1a) writes dJ/drho as <v~, dt v>, i.e. with the
     * bracket's "- s" dropped, so a spurious <psi, T dLambda^-1/dm T s>
     * survives in cells holding a source. update_adjs2D.cl carries the same
     * term for a pressure source; this is its velocity-block twin.
     *
     * A pressure source (type 100) enters only the stress block and a force
     * source only the velocity block, so the two are mutually exclusive and
     * neither leaks into the other's parameters.
     *
     * The coefficient is 1: (A1a) has no c-factor, and the `rip` inside lvx is
     * how the adjoint velocity equation computes dt(v~), not a coefficient.
     * What matters for the averaging is WHERE this lands -- vx lives at the
     * rip position and vz at the rkp one, so each goes to its own STAGGERED
     * buoyancy accumulator and average_grad_transpose() applies the averaging
     * Jacobian afterwards. Writing it into a cell-centred gradrho instead
     * would skip that Jacobian and put the sensitivity at the wrong point.
     *
     * Source type is the index into kernel_sources()'s src_names[] =
     * {"vx","vy","vz","p",...}, so 0 = force in x and 2 = force in z here. */
    #if GRADOUT==1
    if (nsrc>0){
        for (int srci=0; srci<nsrc; srci++){
            int st = (int)src_pos[4+5*srci];
            if (st==0 || st==2){
                int si=(int)(src_pos[0+5*srci]/DH)+FDOH;
                int sk=(int)(src_pos[2+5*srci]/DH)+FDOH;
                if (si==gidx && sk==gidz){
                    /* No src_scale here: assign_modeling_case.c compiles
                     * this file only at FP16==0, where src_scale is 0 and
                     * kernel_sources()'s amp is plainly DT*src. The half2
                     * kernels are the FP16>0 path. */
                    float samp = DT*src[srci*NT+nt];
                    if (st==0) gradrip[indp] += vxr[indv]*samp;
                    else       gradrkp[indp] += vzr[indv]*samp;
                }
            }
        }
    }
    #endif

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

