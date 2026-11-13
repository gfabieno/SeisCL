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

/*This is the kernel that implement the free surface condition in 2D*/

//TODO: Adjoint surface for viscoelastic propagation

#define indv(z,y,x)  (x)*(NY*NZ*DIV)+(y)*(NZ*DIV)+(z)
#define indp(z,y,x)  ((x)-FDOH)*(NY-2*FDOH)*(NZ*DIV-2*FDOH)+((y)-FDOH)*(NZ*DIV-2*FDOH)+((z)-FDOH)



FUNDEF void surface_adj(GLOBARG __prec *vxr,   GLOBARG __prec *vyr,
                        GLOBARG __prec *vzr,   GLOBARG __prec *sxxr,
                        GLOBARG __prec *syyr,  GLOBARG __prec *szzr,
                        GLOBARG __prec *sxyr,  GLOBARG __prec *syzr,
                        GLOBARG __prec *sxzr,  GLOBARG __prec *M,
                        GLOBARG __prec *rip,   GLOBARG __prec *rjp,
                        GLOBARG __prec *rkp,   GLOBARG __prec *mu,
                        GLOBARG __prec *rxx,   GLOBARG __prec *rzz,
                        GLOBARG __prec *taus,  GLOBARG __prec *taup,
                        GLOBARG float *eta)
{
    /*Indice definition. gsize[0]=N[1] (Y extent), gsize[1]=N[2] (X extent)
     * (Init_OpenCL.c), and both backends map gsize[i] to hardware axis i
     * identically (clEnqueueNDRangeKernel / cuLaunchKernel in clprogram.c),
     * so axis 0 must always land on gidy and axis 1 on gidx -- matching
     * surface3D.cl's (forward) convention below. The OpenCL branch here used
     * to swap them (get_global_id(0) -> gidx), disagreeing with both its own
     * CUDA branch and the forward kernel; harmless on a cubic, x/y-symmetric
     * grid, but a real backend-dependent transpose of the free-surface
     * adjoint correction otherwise. See notes/3d-gradient-findings.md. */
    #ifdef __OPENCL_VERSION__
    int gidy = get_global_id(0) + FDOH;
    int gidx = get_global_id(1) + FDOH;
    #else
    int gidy = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y + FDOH;
    #endif
    int gidz=FDOH;
    __cprec1 hc[6] = {HC1, HC2, HC3, HC4, HC5, HC6};

    /* Global work size is padded to be a multiple of local work size.
     The padding elements must not be updated */
    if ( gidx>(NX-FDOH-1) || gidy>(NY-FDOH-1) ){
        return;
    }

    __cprec1 h, hz;
    __cprec1 lszz, lsyy, lsxx;
    __cprec1 lmu, lM;
    
    
    //Read variables in register
    lszz = -szzr[indv(gidz,gidy,gidx)];
    lsxx = -sxxr[indv(gidz,gidy,gidx)];
    lsyy = -syyr[indv(gidz,gidy,gidx)];
    lmu = mu[indp(gidz,gidy,gidx)];
    lM = M[indp(gidz,gidy,gidx)];
    
    //Preadjoint transformations
    {
        __cprec1 temp1, temp2, temp3, a;
        temp1 = lsxx;
        temp2 = lsyy;
        temp3 = lszz;
        a = (__cprec1)1.0/( (__cprec1)6.0*lM*lmu - (__cprec1)8.0*lmu*lmu);
        lsxx = a * ( (__cprec1)2.0*(lM-lmu) * temp1
                    + (-lM + (__cprec1)2.0 * lmu) * (temp2 +temp3) );
        lsyy = a * ( (__cprec1)2.0*(lM-lmu) * temp2
                    + (-lM + (__cprec1)2.0 * lmu) * (temp1 +temp3) );
        lszz = a * ( (__cprec1)2.0*(lM-lmu) * temp3
                    + (-lM + (__cprec1)2.0 * lmu) * (temp1 +temp2) );
    }
    /*Adjoint of the mirror method*/
    lszz=0.0;
    
    h=-((lM-lmu*(__cprec1)2.0)*(lM-lmu*(__cprec1)2.0)/lM);
    hz=-(lM-lmu*(__cprec1)2.0);
    
#if FP16!=0
    /* FP16 path (no surface_adj_fold): scatter into vxr, vyr of neighbouring
       columns. Not race-free: other threads add to the same cells. */
    for (int m=0; m<FDOH; m++){
        if ( (gidx-FDOH + m) < (NX - 2*FDOH))
            vxr[indv(gidz,gidy,gidx+m)]+=(hc[m] * h * (lsxx + lsyy) *
                                         (__cprec1)rip[indp(gidz,gidy,gidx+m)]);
        if ( (gidx - (m + 1)) > (FDOH - 1))
            vxr[indv(gidz,gidy,gidx-(m+1))]+=(-hc[m] * h* (lsxx + lsyy) *
                                             (__cprec1)rip[indp(gidz,gidy,gidx-(m+1))]);
        if ( (gidy-FDOH + m) < (NY - 2*FDOH))
            vyr[indv(gidz,gidy+m,gidx)]+=(hc[m] * h * (lsxx + lsyy) *
                                         (__cprec1)rjp[indp(gidz,gidy+m,gidx)]);
        if ( (gidy - (m + 1)) > (FDOH - 1))
            vyr[indv(gidz,gidy-(m+1),gidx)]+=(-hc[m] * h* (lsxx + lsyy) *
                                              (__cprec1)rjp[indp(gidz,gidy-(m+1),gidx)]);
            
        vzr[indv(gidz+m,gidy,gidx)]+=(hc[m] *hz * (lsxx + lsyy)
                                      * (__cprec1)rkp[indp(gidz+m,gidy, gidx)]);
    }
#endif
    
    //Perform the post-adjoint transformation
    sxxr[indv(gidz,gidy,gidx)] = -(lM * lsxx + (lM-(__cprec1)2.0*lmu) * (lsyy+lszz));
    syyr[indv(gidz,gidy,gidx)] = -(lM * lsyy + (lM-(__cprec1)2.0*lmu) * (lsxx+lszz));
    szzr[indv(gidz,gidy,gidx)] = -(lM * lszz + (lM-(__cprec1)2.0*lmu) * (lsxx+lsyy));
    

}





/* Adjoint of the free surface, first part (FREESURF==1, 3D elastic, FP16==0).
   Same role as in surface2D_adj.cl: launched just before surface_adj, between
   the adjoint stress update and the adjoint velocity update; one thread per
   (y, x) column, each writing only its own column (race-free).
   1. Adjoint of the stress images (szz through Dzp in vz, sxz and syz through
      Dzm in vx and vy), folded back into the rows below the surface:
        img_szz(k) = -sum_{m>=k}   hc[m] vzr(FDOH-k+m)
        img_sxz(k) = -sum_{m>=k-1} hc[m] vxr(FDOH-k+m+1)   (syz: vyr)
        szzr += M img_szz, sxxr and syyr += (M-2mu) img_szz at FDOH+k,
        sxzr += muipkp img_sxz, syzr += mujpkp img_syz at FDOH+k-1;
      image rows of the adjoint stresses stay zero.
   2. Adjoint of h = hx*(Dxm vx + Dym vy) + hz*Dzm vz added to sxx and syy: a
      gather from the surface stresses of the neighbouring columns in x
      (into vxr) and y (into vyr), read before surface_adj transforms them.
   Checked against the exact transpose of one time step:
   ~/data/claude/torch-robust/fsadj/ref3d.py (3e-15). */
/* lsxx + lsyy at the surface of column (y, x), after the pre-adjoint transform
   (as in surface_adj), and hx = -(M-2mu)^2/M there. */
LFUNDEF __cprec1 lsum3d(GLOBARG __prec *sxxr, GLOBARG __prec *syyr,
                        GLOBARG __prec *szzr, GLOBARG __prec *M,
                        GLOBARG __prec *mu, int y, int x)
{
    __cprec1 cM = M[indp(FDOH, y, x)], cmu = mu[indp(FDOH, y, x)];
    __cprec1 t1 = -(__cprec1)sxxr[indv(FDOH, y, x)];
    __cprec1 t2 = -(__cprec1)syyr[indv(FDOH, y, x)];
    __cprec1 t3 = -(__cprec1)szzr[indv(FDOH, y, x)];
    __cprec1 a = (__cprec1)1.0/( (__cprec1)6.0*cM*cmu - (__cprec1)8.0*cmu*cmu);
    return a * ( (__cprec1)2.0*(cM-cmu) * (t1 + t2)
                 + (-cM + (__cprec1)2.0*cmu) * (t1 + t2 + (__cprec1)2.0*t3) );
}
LFUNDEF __cprec1 hx3d(GLOBARG __prec *M, GLOBARG __prec *mu, int y, int x)
{
    __cprec1 d = M[indp(FDOH, y, x)] - (__cprec1)2.0*mu[indp(FDOH, y, x)];
    return -(d*d/M[indp(FDOH, y, x)]);
}
FUNDEF void surface_adj_fold(GLOBARG __prec *vxr,   GLOBARG __prec *vyr,
                             GLOBARG __prec *vzr,   GLOBARG __prec *sxxr,
                             GLOBARG __prec *syyr,  GLOBARG __prec *szzr,
                             GLOBARG __prec *syzr,  GLOBARG __prec *sxzr,
                             GLOBARG __prec *M,     GLOBARG __prec *mu,
                             GLOBARG __prec *muipkp, GLOBARG __prec *mujpkp,
                             GLOBARG __prec *rip,   GLOBARG __prec *rjp,
                             GLOBARG __prec *rkp)
{
    #ifdef __OPENCL_VERSION__
    int gidy = get_global_id(0) + FDOH;
    int gidx = get_global_id(1) + FDOH;
    #else
    int gidy = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y + FDOH;
    #endif
    if ( gidx>(NX-FDOH-1) || gidy>(NY-FDOH-1) ){
        return;
    }
    int gidz = FDOH;
    __cprec1 hc[6] = {HC1, HC2, HC3, HC4, HC5, HC6};

    // 1. Adjoint of the stress images (reads vxr, vyr, vzr of this column first)
    for (int k=1; k<=FDOH; k++){
        __cprec1 isz = 0, isx = 0, isy = 0;
        for (int m=k; m<FDOH; m++){
            isz -= hc[m] * (__cprec1)vzr[indv(FDOH-k+m, gidy, gidx)];
        }
        for (int m=k-1; m<FDOH; m++){
            isx -= hc[m] * (__cprec1)vxr[indv(FDOH-k+m+1, gidy, gidx)];
            isy -= hc[m] * (__cprec1)vyr[indv(FDOH-k+m+1, gidy, gidx)];
        }
        int z = FDOH + k;
        __cprec1 lM = M[indp(z, gidy, gidx)];
        __cprec1 lmu = mu[indp(z, gidy, gidx)];
        szzr[indv(z, gidy, gidx)] += lM * isz;
        sxxr[indv(z, gidy, gidx)] += (lM - (__cprec1)2.0 * lmu) * isz;
        syyr[indv(z, gidy, gidx)] += (lM - (__cprec1)2.0 * lmu) * isz;
        sxzr[indv(z-1, gidy, gidx)] += (__cprec1)muipkp[indp(z-1, gidy, gidx)] * isx;
        syzr[indv(z-1, gidy, gidx)] += (__cprec1)mujpkp[indp(z-1, gidy, gidx)] * isy;
    }
    for (int m=1; m<=FDOH; m++){
        szzr[indv(FDOH-m, gidy, gidx)] = 0;
        sxzr[indv(FDOH-m, gidy, gidx)] = 0;
        syzr[indv(FDOH-m, gidy, gidx)] = 0;
    }

    // 2. Adjoint of the sxx, syy correction, as a gather
    __cprec1 dvx = 0, dvy = 0;
    for (int m=0; m<FDOH; m++){
        int c = gidx - m;
        if (c >= FDOH) dvx += hc[m] * hx3d(M, mu, gidy, c) * lsum3d(sxxr, syyr, szzr, M, mu, gidy, c);
        c = gidx + m + 1;
        if (c <= NX-FDOH-1) dvx -= hc[m] * hx3d(M, mu, gidy, c) * lsum3d(sxxr, syyr, szzr, M, mu, gidy, c);
        c = gidy - m;
        if (c >= FDOH) dvy += hc[m] * hx3d(M, mu, c, gidx) * lsum3d(sxxr, syyr, szzr, M, mu, c, gidx);
        c = gidy + m + 1;
        if (c <= NY-FDOH-1) dvy -= hc[m] * hx3d(M, mu, c, gidx) * lsum3d(sxxr, syyr, szzr, M, mu, c, gidx);
    }
    vxr[indv(gidz, gidy, gidx)] += dvx * (__cprec1)rip[indp(gidz, gidy, gidx)];
    vyr[indv(gidz, gidy, gidx)] += dvy * (__cprec1)rjp[indp(gidz, gidy, gidx)];
    {
        __cprec1 ls = lsum3d(sxxr, syyr, szzr, M, mu, gidy, gidx);
        __cprec1 hz = -(M[indp(gidz, gidy, gidx)] - (__cprec1)2.0*mu[indp(gidz, gidy, gidx)]);
        for (int m=0; m<FDOH; m++){
            vzr[indv(gidz+m, gidy, gidx)] += hc[m] * hz * ls * (__cprec1)rkp[indp(gidz+m, gidy, gidx)];
        }
    }
}



#ifdef HAVE_INJECT_IND   /* compiled only with header_injectbnd.cl (the surf_grad program) */
#ifndef SURFROWS
#define SURFROWS (FDOH+1)
#endif
/* Gradient in the top rows under a stress-image free surface, 3D: the twin
   of surface_grad in surface2D_adj.cl (see there for the derivation and the
   timing). Launched at the start of each adjoint step, before update_adjs.
   Adjoint fields = T*eta_t, T = diag(B, -C); stresses nu = -C^-1 a with
   C = lambda 1 1^T + 2 mu I on the normal stresses, lambda = M - 2mu.
   Rows 1..SURFROWS-1: dJ/dM = (sum nu)(sum eps), dJ/dmu = -2 (sum nu)(sum eps)
     + 2 sum nu_ii eps_ii, shear planes nu_ij (v_i,j + v_j,i).
   Surface row (szz = 0, sxx and syy += -(lambda^2/M)(vxx+vyy) - lambda vzz):
     dJ/dM  = 4 mu^2/M^2 (nu_xx+nu_yy)(eps_xx+eps_yy),
     dJ/dmu = 4 lambda/M (nu_xx+nu_yy)(eps_xx+eps_yy) - 2 (nu_xx eps_yy + nu_yy eps_xx).
   Density, rows 0..FDOH-1, from the stored states (minus a force source). */
FUNDEF void surface_grad(GLOBARG __prec *vx,     GLOBARG __prec *vy,
                         GLOBARG __prec *vz,     GLOBARG __prec *vxr,
                         GLOBARG __prec *vyr,    GLOBARG __prec *vzr,
                         GLOBARG __prec *sxxr,   GLOBARG __prec *syyr,
                         GLOBARG __prec *szzr,   GLOBARG __prec *sxyr,
                         GLOBARG __prec *sxzr,   GLOBARG __prec *syzr,
                         GLOBARG __prec *M,      GLOBARG __prec *mu,
                         GLOBARG __prec *muipjp, GLOBARG __prec *muipkp,
                         GLOBARG __prec *mujpkp,
                         GLOBARG __prec *vxbnd,  GLOBARG __prec *vybnd,
                         GLOBARG __prec *vzbnd,  GLOBARG float *taper,
                         GLOBARG float *gradM,   GLOBARG float *gradmu,
                         GLOBARG float *gradmuipjp, GLOBARG float *gradmuipkp,
                         GLOBARG float *gradmujpkp, GLOBARG float *gradrip,
                         GLOBARG float *gradrjp, GLOBARG float *gradrkp,
                         GLOBARG float *src,     GLOBARG float *src_pos,
                         int nsrc,               int nt,
                         GLOBARG float *pout,    GLOBARG float *rec_pos,
                         int nrec)
{
    #ifdef __OPENCL_VERSION__
    int gidy = get_global_id(0) + FDOH;
    int gidx = get_global_id(1) + FDOH;
    #else
    int gidy = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y + FDOH;
    #endif
    if ( gidx>(NX-FDOH-1) || gidy>(NY-FDOH-1) ){
        return;
    }
    __cprec1 hc[6] = {HC1, HC2, HC3, HC4, HC5, HC6};
    for (int k=0; k<SURFROWS && FDOH+k<NZ-FDOH; k++){
        int z = FDOH+k;
        __cprec1 exx=0, eyy=0, ezz=0, vxy=0, vyx=0, vxz=0, vzx=0, vyz=0, vzy=0;
        for (int m=0; m<FDOH; m++){
            exx += hc[m]*((__cprec1)vx[indv(z, gidy, gidx+m)]   - (__cprec1)vx[indv(z, gidy, gidx-m-1)]);
            eyy += hc[m]*((__cprec1)vy[indv(z, gidy+m, gidx)]   - (__cprec1)vy[indv(z, gidy-m-1, gidx)]);
            ezz += hc[m]*((__cprec1)vz[indv(z+m, gidy, gidx)]   - (__cprec1)vz[indv(z-m-1, gidy, gidx)]);
            vxy += hc[m]*((__cprec1)vx[indv(z, gidy+m+1, gidx)] - (__cprec1)vx[indv(z, gidy-m, gidx)]);
            vyx += hc[m]*((__cprec1)vy[indv(z, gidy, gidx+m+1)] - (__cprec1)vy[indv(z, gidy, gidx-m)]);
            vxz += hc[m]*((__cprec1)vx[indv(z+m+1, gidy, gidx)] - (__cprec1)vx[indv(z-m, gidy, gidx)]);
            vzx += hc[m]*((__cprec1)vz[indv(z, gidy, gidx+m+1)] - (__cprec1)vz[indv(z, gidy, gidx-m)]);
            vyz += hc[m]*((__cprec1)vy[indv(z+m+1, gidy, gidx)] - (__cprec1)vy[indv(z-m, gidy, gidx)]);
            vzy += hc[m]*((__cprec1)vz[indv(z, gidy+m+1, gidx)] - (__cprec1)vz[indv(z, gidy-m, gidx)]);
        }
        int ip = indp(z, gidy, gidx), iv = indv(z, gidy, gidx);
        __cprec1 lM = M[ip], lmu = mu[ip], lam = lM-(__cprec1)2.0*lmu;
        __cprec1 axx = sxxr[iv], ayy = syyr[iv], azz = szzr[iv];
        #if PRESOUT==1
        for (int g=0; g<nrec; g++){
            int ri=(int)(rec_pos[0+8*g]/DH)+FDOH;
            int rj=(int)(rec_pos[1+8*g]/DH)+FDOH;
            int rk=(int)(rec_pos[2+8*g]/DH)+FDOH;
            if (ri==gidx && rj==gidy && rk==z){
                axx -= pout[NT*g+nt]/(__cprec1)3.0;
                ayy -= pout[NT*g+nt]/(__cprec1)3.0;
                azz -= pout[NT*g+nt]/(__cprec1)3.0;
            }
        }
        #endif
        __cprec1 nxx=0, nyy=0, nzz=0;
        if (lmu != 0){
            __cprec1 sa = axx+ayy+azz;
            __cprec1 q = lam/((__cprec1)2.0*lmu+(__cprec1)3.0*lam);
            nxx = -(axx - q*sa)/((__cprec1)2.0*lmu);
            nyy = -(ayy - q*sa)/((__cprec1)2.0*lmu);
            nzz = -(azz - q*sa)/((__cprec1)2.0*lmu);
        }
        __cprec1 gM, gmu;
        if (k==0){
            gM  = (__cprec1)4.0*lmu*lmu/(lM*lM)*(nxx+nyy)*(exx+eyy);
            gmu = (__cprec1)4.0*lam/lM*(nxx+nyy)*(exx+eyy) - (__cprec1)2.0*(nxx*eyy + nyy*exx);
        }
        else{
            __cprec1 sn = nxx+nyy+nzz, se = exx+eyy+ezz;
            gM  = sn*se;
            gmu = -(__cprec1)2.0*sn*se + (__cprec1)2.0*(nxx*exx + nyy*eyy + nzz*ezz);
        }
        gradM[ip] += gM;
        gradmu[ip] += gmu;
        if (muipjp[ip] != 0) gradmuipjp[ip] += -(__cprec1)sxyr[iv]/muipjp[ip]*(vxy+vyx);
        if (muipkp[ip] != 0) gradmuipkp[ip] += -(__cprec1)sxzr[iv]/muipkp[ip]*(vxz+vzx);
        if (mujpkp[ip] != 0) gradmujpkp[ip] += -(__cprec1)syzr[iv]/mujpkp[ip]*(vyz+vzy);

        if (k<FDOH){
            int mb = inject_ind(z, gidy, gidx);
            if (mb!=-1){
                __cprec1 gt = 1;
                #if ABS_TYPE==2
                if (z>NZ-NAB-FDOH-1) gt*=taper[NZ-FDOH-z-1];
                if (gidy-FDOH<NAB) gt*=taper[gidy-FDOH];
                if (gidy>NY-NAB-FDOH-1) gt*=taper[NY-FDOH-gidy-1];
                #if DEVID==0 & MYLOCALID==0
                if (gidx-FDOH<NAB) gt*=taper[gidx-FDOH];
                #endif
                #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
                if (gidx>NX-NAB-FDOH-1) gt*=taper[NX-FDOH-gidx-1];
                #endif
                #endif
                __cprec1 sx = 0, sy = 0, sz = 0;
                for (int srci=0; srci<nsrc; srci++){
                    int st = (int)src_pos[4+5*srci];
                    int si = (int)(src_pos[0+5*srci]/DH)+FDOH;
                    int sj = (int)(src_pos[1+5*srci]/DH)+FDOH;
                    int sk = (int)(src_pos[2+5*srci]/DH)+FDOH;
                    if (si==gidx && sj==gidy && sk==z){
                        if (st==0) sx += DT*src[srci*NT+nt];
                        if (st==1) sy += DT*src[srci*NT+nt];
                        if (st==2) sz += DT*src[srci*NT+nt];
                    }
                }
                gradrip[ip] += -(__cprec1)vxr[iv]*((__cprec1)vx[iv] - gt*((__cprec1)vxbnd[mb] + sx));
                gradrjp[ip] += -(__cprec1)vyr[iv]*((__cprec1)vy[iv] - gt*((__cprec1)vybnd[mb] + sy));
                gradrkp[ip] += -(__cprec1)vzr[iv]*((__cprec1)vz[iv] - gt*((__cprec1)vzbnd[mb] + sz));
            }
        }
    }
}
#endif
