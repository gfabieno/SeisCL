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

#include "F.h"
#include "third_party/NVIDIA_FP16/fp16_conversion.h"

/* The DFT gradient (back_prop_type 2) is computed on the device only, by
   grad_dft*.cl. Its host copy, calc_grad() (OpenCL build only, selected by
   SEISCL_DFT_HOST / SEISCL_DFT_CHECK), was removed on 2026-11-14: it had
   missed the scheme's discrete d/dt and the source terms (2026-09-05), and
   SeisCL/tests/dft_reference.py is the independent reference. */

/* ---------------------------------------------------------------------------
   Transpose of the material-parameter averaging.

   The FD physics is evaluated at staggered, averaged parameters (rip/rjp/rkp
   from the buoyancy, muipkp/muipjp/mujpkp from mu, tausipkp/... from taus), so
   that is where the correlation accumulates its gradient. Mapping those back
   onto the cell-centred parameters the user inverts for is the transpose of
   the averaging operator's Jacobian.

   Written as a scatter, mirroring the forward routines in
   assign_modeling_case.c loop for loop -- including the trailing copy region
   each of them ends with, where the Jacobian is 1 rather than the averaging
   formula. Specified and dot-tested independently in
   SeisCL/tests/dot_prod_average.py.

   Must run *before* unscale_par(): the Jacobians are evaluated at the stored
   (internal) parameter values, which unscale_par overwrites. It commutes with
   unscale_grad(), which is a uniform factor on a linear operator.
   --------------------------------------------------------------------------*/

/* Loop order. The arrays are x-slowest, z-fastest (ind = i*NY*NZ + j*NZ + k),
   so every loop below runs x outermost and z innermost: contiguous accesses.
   Until 2026-11-25 they ran z outermost and x innermost, a stride of NY*NZ
   floats per step: on a 3-D 128x128x384 gradient these transposes were ~2 s
   of single-thread cache misses, more than half of the host time of the run.

   The result is bit-identical to that order. A cell receives up to four
   contributions (one per node that averages it), added in float one after the
   other, so their order sets the rounding. The old scatter added them node by
   node in its (z, y, x) order; scatter_order() gives that same order per cell,
   and the 4-point transposes apply it as up to four passes, one contribution
   per cell and pass. The 2-point density transpose needs no pass: the node of
   a cell's second contribution is the cell itself, last in both orders. */

/* The order in which the z-outer scatter added the contributions d = 0..3 of
   the nodes (c - off[d]) to a cell c: the node first in (z, y, x) order, i.e.
   the largest offset in that order, first; equal offsets (a zero direction,
   SH) by d, as the old inner loop over d did. */
static void scatter_order(int (*dir)[3], int order[4]){
    int off[4][3];
    int c, d, e, t, cmp;
    for (c=0;c<3;c++){
        off[0][c] = 0;
        off[1][c] = dir[0][c];
        off[2][c] = dir[1][c];
        off[3][c] = dir[0][c]+dir[1][c];
    }
    for (d=0;d<4;d++) order[d] = d;
    /* insertion sort, stable: larger offset first */
    for (d=1;d<4;d++){
        for (e=d;e>0;e--){
            cmp = 0;
            for (c=0;c<3 && cmp==0;c++)
                cmp = off[order[e]][c] - off[order[e-1]][c];
            if (cmp<=0) break;
            t = order[e]; order[e] = order[e-1]; order[e-1] = t;
        }
    }
}

static void grad_T_arithmetic_rho(float * pin, float * gin, float * gout,
                                  int * N, int ndim, int * dir){
    int i,j,k;
    int NX, NY, NZ;
    int NX0=0, NY0=0, NZ0=0;
    int ind1, ind2;
    if (ndim==3){ NX=N[2]; NY=N[1]; NZ=N[0]; }
    else        { NX=N[1]; NY=1;    NZ=N[0]; }

    for (i=0;i<NX-dir[2];i++){
        for (j=0;j<NY-dir[1];j++){
            for (k=0;k<NZ-dir[0];k++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                ind2 = (i+dir[2])*NY*NZ+(j+dir[1])*NZ+(k+dir[0]);
                /* The kernels accumulate gradrip/gradrkp against the
                   averaged DENSITY (update_adjv: -B^2 dJ/dB), and nothing
                   downstream multiplies by -b^2: the transpose of the
                   arithmetic density average rho_avg = (rho1+rho2)/2 is 1/2
                   for each cell. The buoyancy Jacobian used before,
                   0.5*avg^2/p^2 = 0.5*(rho_j/rho_avg)^2, is that only for a
                   uniform density. With one vacuum neighbour (FREESURF==2,
                   ave_two(): b_avg = 2b, i.e. rho_avg = rho/2) it is still
                   1/2, for the solid cell only; it used to be dropped. */
                if (pin[ind1]==0.0 && pin[ind2]==0.0) continue;
                if (pin[ind1]!=0.0) gout[ind1] += 0.5*gin[ind1];
                if (pin[ind2]!=0.0) gout[ind2] += 0.5*gin[ind1];
            }
        }
    }
    if (dir[2]==1)      NX0=NX-1;
    else if (dir[1]==1) NY0=NY-1;
    if (dir[0]==1)      NZ0=NZ-1;
    for (i=NX0;i<NX;i++){
        for (j=NY0;j<NY;j++){
            for (k=NZ0;k<NZ;k++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                gout[ind1] += gin[ind1];
            }
        }
    }
}

/* The copy region at the far edge of each averaging direction, where the
   forward routines copy the cell instead of averaging (Jacobian 1). */
static void grad_T_edge_copy(float * gin, float * gout,
                             int NX, int NY, int NZ, int (*dir)[3]){
    int i,j,k,d;
    int NX0, NY0, NZ0;
    for (d=0;d<2;d++){
        /* A zero direction (SH: 2-point average) has no edge region; without
           this test the copy below would cover the whole grid. */
        if (dir[d][0]==0 && dir[d][1]==0 && dir[d][2]==0) continue;
        NX0=0; NY0=0; NZ0=0;
        if (dir[d][2]==1)      NX0=NX-1;
        else if (dir[d][1]==1) NY0=NY-1;
        if (dir[d][0]==1)      NZ0=NZ-1;
        for (i=NX0;i<NX;i++){
            for (j=NY0;j<NY;j++){
                for (k=NZ0;k<NZ;k++){
                    gout[i*NY*NZ+j*NZ+k] += gin[i*NY*NZ+j*NZ+k];
                }
            }
        }
    }
}

static int grad_T_harmonic_mu(float * pin, float * gin, float * gout,
                              int * N, int ndim, int (*dir)[3]){
    int state=0;
    int i,j,k,d,p,n,c;
    int NX, NY, NZ, IX, IY, IZ;
    int ind[4], off[4], order[4];
    double avg, s;
    double * w=NULL;
    unsigned char * valid=NULL;
    if (ndim==3){ NX=N[2]; NY=N[1]; NZ=N[0]; }
    else        { NX=N[1]; NY=1;    NZ=N[0]; }
    IX = NX-dir[0][2]-dir[1][2];
    IY = NY-dir[0][1]-dir[1][1];
    IZ = NZ-dir[0][0]-dir[1][0];
    off[0] = 0;
    off[1] = dir[0][2]*NY*NZ+dir[0][1]*NZ+dir[0][0];
    off[2] = dir[1][2]*NY*NZ+dir[1][1]*NZ+dir[1][0];
    off[3] = off[1]+off[2];

    GMALLOC(w, sizeof(double)*NX*NY*NZ);
    GMALLOC(valid, sizeof(unsigned char)*NX*NY*NZ);
    if (state){ GFree(w); GFree(valid); return state; }

    /* pout = 4/sum(1/p_j)  =>  d(pout)/d(p_j) = (pout^2/4)/p_j^2, and all four
       are zero in a vacuum cell, where ave_harmonic_mu forces pout to 0.
       First the node factor gin*pout^2/4, then the four contributions. */
    for (i=0;i<IX;i++){
        for (j=0;j<IY;j++){
            for (k=0;k<IZ;k++){
                n = i*NY*NZ+j*NZ+k;
                for (d=0;d<4;d++) ind[d] = n+off[d];
                if (pin[ind[0]]==0 || pin[ind[1]]==0
                    || pin[ind[2]]==0 || pin[ind[3]]==0){
                    continue;
                }
                s = 1.0/pin[ind[0]] + 1.0/pin[ind[1]]
                  + 1.0/pin[ind[2]] + 1.0/pin[ind[3]];
                if (s==0.0) continue;
                avg = 4.0/s;
                w[n] = gin[n]*0.25*avg*avg;
                valid[n] = 1;
            }
        }
    }
    scatter_order(dir, order);
    for (p=0;p<4;p++){
        d = order[p];
        for (i=0;i<IX;i++){
            for (j=0;j<IY;j++){
                for (k=0;k<IZ;k++){
                    n = i*NY*NZ+j*NZ+k;
                    if (!valid[n]) continue;
                    c = n+off[d];
                    gout[c] += w[n]/(pin[c]*pin[c]);
                }
            }
        }
    }
    GFree(w); GFree(valid);
    grad_T_edge_copy(gin, gout, NX, NY, NZ, dir);
    return state;
}

static void grad_T_arithmetic_tau(float * gin, float * gout,
                                  int * N, int ndim, int (*dir)[3]){
    int i,j,k,d,p,n;
    int NX, NY, NZ, IX, IY, IZ;
    int off[4], order[4];
    if (ndim==3){ NX=N[2]; NY=N[1]; NZ=N[0]; }
    else        { NX=N[1]; NY=1;    NZ=N[0]; }
    IX = NX-dir[0][2]-dir[1][2];
    IY = NY-dir[0][1]-dir[1][1];
    IZ = NZ-dir[0][0]-dir[1][0];
    off[0] = 0;
    off[1] = dir[0][2]*NY*NZ+dir[0][1]*NZ+dir[0][0];
    off[2] = dir[1][2]*NY*NZ+dir[1][1]*NZ+dir[1][0];
    off[3] = off[1]+off[2];

    /* Plain 4-point arithmetic mean: every Jacobian entry is 0.25. */
    scatter_order(dir, order);
    for (p=0;p<4;p++){
        d = order[p];
        for (i=0;i<IX;i++){
            for (j=0;j<IY;j++){
                for (k=0;k<IZ;k++){
                    n = i*NY*NZ+j*NZ+k;
                    gout[n+off[d]] += 0.25*gin[n];
                }
            }
        }
    }
    grad_T_edge_copy(gin, gout, NX, NY, NZ, dir);
}

/* One helper so a missing parameter (SH has no M, elastic has no taus, ...)
   is skipped rather than dereferenced. */
static float * par_grad(model * m, const char * name, float ** value){
    parameter * p = get_par(m->pars, m->npars, name);
    if (!p || !p->gl_grad) return NULL;
    if (value) *value = p->gl_par;
    return p->gl_grad;
}

static double gsls_alpha(model * m);

/* One shear node (muipkp, tausipkp, ...) back to the cells (mu, taus).

   With L > 0 the DFT kernels give, at each node n, g_mu = dJ/dmu_p,n and
   g_tau = dJ/dtau_n at fixed mu_p,n, for the paper's modulus
   mu_p,n = mu_s,n (1 + alpha tau_n) (GJI 2017 Appendix A). The averages
   act on the STORED modulus and on tau: mu_s,n = harmonic(mu_s cells),
   tau_n = arithmetic(tau cells). So, exactly:
     1. node, paper -> node, stored:
          dJ/dmu_s,n = g_mu (1 + alpha tau_n)
          dJ/dtau_n|mu_s = g_tau + g_mu mu_s,n alpha
     2. transpose of the harmonic (mu_s) and arithmetic (tau) averages;
     3. cell, stored -> cell, paper (what chain_rule_par_type expects, as
        for the cell-centred mu that needs no average):
          dJ/dmu_p,c = dJ/dmu_s,c / (1 + alpha tau_c)
          dJ/dtau_c|mu_p = dJ/dtau_c|mu_s - dJ/dmu_s,c mu_s,c alpha/(1+alpha tau_c)
   Until 2026-11-19 the node gradients were transposed as if they were with
   respect to stored values and converted at the cell: exact only where tau
   and mu are uniform across the average (2-D heterogeneous test: taus 0.89).
   mu_s enters steps 1 and 3 in physical units (muphys), the units the DFT
   gradient refers to (unscale_grad_dft is uniform over the parameters).
   Elastic (alpha = 0): plain transposes, as before. */
static int shear_node_transpose(model * m, const char * muname,
                                const char * tauname, int (*dir)[3]){
    int state=0;
    int i;
    float * pmu; float * ptau=NULL;
    float * pmun; float * ptaun=NULL;
    float * gmu = par_grad(m, "mu", &pmu);
    float * gmun = par_grad(m, muname, &pmun);
    float * gtau = par_grad(m, "taus", &ptau);
    float * gtaun = par_grad(m, tauname, &ptaun);
    parameter * p;
    double al = gsls_alpha(m);
    double muphys = m->dh/m->dt*pow(2.0, -m->par_scale);
    int n;
    float * gm=NULL; float * gt=NULL; float * cm=NULL; float * ct=NULL;

    if (!gmu || !gmun) {
        /* taus without mu: plain arithmetic transpose. */
        if (gtau && gtaun) grad_T_arithmetic_tau(gtaun, gtau, m->N, m->NDIM, dir);
        return state;
    }
    if (al==0.0){
        __GUARD grad_T_harmonic_mu(pmu, gmun, gmu, m->N, m->NDIM, dir);
        if (gtau && gtaun) grad_T_arithmetic_tau(gtaun, gtau, m->N, m->NDIM, dir);
        return state;
    }
    /* Node and cell tau values are needed even when taus has no gradient. */
    p = get_par(m->pars, m->npars, "taus");     if (p) ptau = p->gl_par;
    p = get_par(m->pars, m->npars, tauname);    if (p) ptaun = p->gl_par;
    if (!ptau || !ptaun){
        fprintf(stderr, "Error: %s needs taus and %s\n", muname, tauname);
        return 1;
    }
    n = get_par(m->pars, m->npars, "mu")->num_ele;
    GMALLOC(gm, sizeof(float)*n);
    GMALLOC(gt, sizeof(float)*n);
    GMALLOC(cm, sizeof(float)*n);
    GMALLOC(ct, sizeof(float)*n);
    if (!state){
        for (i=0;i<n;i++){
            gm[i] = gmun[i]*(1.0+al*ptaun[i]);
            gt[i] = (gtaun ? gtaun[i] : 0.0f) + gmun[i]*pmun[i]*muphys*al;
            cm[i] = 0; ct[i] = 0;
        }
        state = grad_T_harmonic_mu(pmu, gm, cm, m->N, m->NDIM, dir);
    }
    if (!state){
        grad_T_arithmetic_tau(gt, ct, m->N, m->NDIM, dir);
        for (i=0;i<n;i++){
            double k = 1.0+al*ptau[i];
            gmu[i] += cm[i]/k;
            if (gtau) gtau[i] += ct[i] - cm[i]*pmu[i]*muphys*al/k;
        }
    }
    GFree(gm); GFree(gt); GFree(cm); GFree(ct);
    return state;
}

int average_grad_transpose(model * m) {
    int state=0;
    float * gsrc; float * psrc;
    float * gavg; float * pavg;
    int d1[3];
    int d2[2][3];

    /* buoyancy -> rip / rjp / rkp */
    gsrc = par_grad(m, "rho", &psrc);
    if (gsrc){
        gavg = par_grad(m, "rip", &pavg);
        if (gavg){ d1[0]=0; d1[1]=0; d1[2]=1;
            grad_T_arithmetic_rho(psrc, gavg, gsrc, m->N, m->NDIM, d1); }
        gavg = par_grad(m, "rjp", &pavg);
        if (gavg){ d1[0]=0; d1[1]=1; d1[2]=0;
            grad_T_arithmetic_rho(psrc, gavg, gsrc, m->N, m->NDIM, d1); }
        gavg = par_grad(m, "rkp", &pavg);
        if (gavg){ d1[0]=1; d1[1]=0; d1[2]=0;
            grad_T_arithmetic_rho(psrc, gavg, gsrc, m->N, m->NDIM, d1); }
    }

    /* mu, taus -> the shear nodes muipkp / mujpkp / muipjp and their taus
       twins. SH (ND 21): the 3-D averages with nothing varying along y,
       i.e. along z (mujpkp) or x (muipjp) only, as assign_modeling_case.c
       computes them; the 3-D directions would loop zero times (NY = 1) and
       drop the SH gradient. */
    d2[0][0]=0; d2[0][1]=0; d2[0][2]=1; d2[1][0]=1; d2[1][1]=0; d2[1][2]=0;
    __GUARD shear_node_transpose(m, "muipkp", "tausipkp", d2);
    d2[0][0]=0; d2[0][1]=1; d2[0][2]=0; d2[1][0]=1; d2[1][1]=0; d2[1][2]=0;
    if (m->ND==21){ d2[0][0]=1; d2[0][1]=0; d2[0][2]=0;
                    d2[1][0]=0; d2[1][1]=0; d2[1][2]=0; }
    __GUARD shear_node_transpose(m, "mujpkp", "tausjpkp", d2);
    d2[0][0]=0; d2[0][1]=0; d2[0][2]=1; d2[1][0]=0; d2[1][1]=1; d2[1][2]=0;
    if (m->ND==21){ d2[1][1]=0; }
    __GUARD shear_node_transpose(m, "muipjp", "tausipjp", d2);

    return state;
}

/* unpack_par_fp16, unscale_par_grad and chain_rule_par_type used to be one
   function. They are three unrelated jobs -- storage format, units, and
   parameterization -- and fusing them made it impossible to say which of them
   a given caller wanted. BACK_PROP_TYPE==2 in particular needs the
   parameterization but not the scaling (its correlation kernel already
   converts to physical units on the device), and the forthcoming
   material-averaging transpose has to run *between* the scaling and the
   parameterization. Keeping them separate is what makes those compositions
   expressible; transf_grad() below is simply all three in the original order,
   so existing callers are unaffected. */

/* FP16>1 stores the parameters as half in the same buffer. Expand in place,
   backwards, so the wider floats do not overwrite unread halves. */
int unpack_par_fp16(model * m) {
    int state=0;
    int i, j;
    half * hpar;
    if (m->FP16>1){
        for (i=0;i<m->npars;i++){
            hpar = (half*)m->pars[i].gl_par;
            for (j=m->pars[i].num_ele-1;j>=0;j--){
                m->pars[i].gl_par[j] = half_to_float(hpar[j]);
            }
            
        }
    }
    return state;
}

/* Parameters back to physical units: rho from buoyancy, M and mu from their
   dt/dh scaling, both times 2^-par_scale. Every path needs this before
   chain_rule_par_type(), which evaluates its Jacobians at the physical
   values. */
int unscale_par(model * m) {
    int state=0;
    int i, num_ele=0;

    float * rho = get_par(m->pars, m->npars, "rho")->gl_par;
    num_ele = get_par(m->pars, m->npars, "rho")->num_ele;
    float * M = get_par(m->pars, m->npars, "M")->gl_par;
    float * mu = get_par(m->pars, m->npars, "mu")->gl_par;

    int scaler=m->par_scale;
    /* 2^-scaler once, not one powf per cell in each loop below (gcc does not
       hoist powf, errno): SeisCL.torch runs this after every gradient call
       (2026-11-25). Exact power of two, so the same values as before. */
    const float sc = ldexpf(1.0f, -scaler);

    for (i=0;i<num_ele;i++){
        /* A zero buoyancy marks a vacuum cell (rho() in
           assign_modeling_case.c keeps it exactly 0, not 1/0, for the same
           reason -- a vacuum cell comes from the user's model). Guard the
           inverse here too:
           unguarded, 1.0/0.0 gives +inf, which the sqrt(rho[i]*M[i])-style
           multiplications a few lines below turn into inf*0=NaN even though
           the physical density (and M/mu) are genuinely, correctly zero
           there. Ported from SeisCL-freesurface (notes/back-prop-type1-
           zero-material-nan.md), which found and fixed this in its
           (then-unsplit) transf_grad() -- this is that same site, now in
           unscale_par() after the unscale_par/unscale_grad split. */
        if (rho[i]==0){
            rho[i]=0;
        }
        else{
            rho[i]= 1.0/rho[i]*m->dt/m->dh*sc;
        }
    }
    if (M){
        for (i=0;i<num_ele;i++){
            M[i]*=m->dh/m->dt*sc;
        }
    }
    if (mu){
        for (i=0;i<num_ele;i++){
            mu[i]*=m->dh/m->dt*sc;
        }
    }
    return state;
}

/* Drop the dt the time integration put into the gradient. Separate from
   unscale_par() because it is *not* universal: BACK_PROP_TYPE==2 accumulates
   its gradient from frequency-domain spectra with its own normalization
   (dftnorm/DTNYQ) and must not be given this factor, while still needing the
   parameters unscaled above. */
int unscale_grad(model * m) {
    int state=0;
    int i, num_ele=0;
    /* FP16>0 stores the wavefield in scaled units (par_scale, see
       unscale_par()'s powf(2,-scaler) on M/mu/rho). The raw kernel-
       accumulated gradient inherits that same scaling from the forward and
       adjoint wavefields it correlates, and is otherwise never corrected for
       it (unlike the parameters). Confirmed empirically: at FP16=1 with
       par_scale=-20, the crosswell finite-difference check showed the
       returned gradient uniformly 2^20 too large across vp/vs/rho alike
       (ratio ~1048564/1048515/1050131 vs the expected ~1) -- consistent with
       a single powf(2, par_scale) factor missing here, not a per-parameter
       bug. At FP16=0, par_scale=0 and this is a no-op, which is why it went
       unnoticed until back_prop_type=1 was checked at FP16>0 for the first
       time. */
    float scale2 = powf(2.0f, (float)m->par_scale);

    float * gradrho = get_par(m->pars, m->npars, "rho")->gl_grad;
    num_ele = get_par(m->pars, m->npars, "rho")->num_ele;
    float * M = get_par(m->pars, m->npars, "M")->gl_par;
    float * gradM = get_par(m->pars, m->npars, "M")->gl_grad;
    float * mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float * gradmu = get_par(m->pars, m->npars, "mu")->gl_grad;

    /* gradrho carries an extra dh^2/dt^2 relative to gradM/gradmu's plain
       1/dt: it is accumulated (via average_grad_transpose(), from
       update_adjv2D.cl's gradrip/gradrkp) against the buoyancy parameter,
       whose internal (non-dimensionalized) representation is scaled by
       dt/dh relative to the physical value (see unscale_par()'s
       rho[i]=1/rho[i]*dt/dh), the inverse of M/mu's dh/dt scaling -- so its
       raw kernel-accumulated gradient needs the reciprocal-squared
       correction here rather than the same plain 1/dt gradM/gradmu get.
       Confirmed empirically: crosswell finite-difference check on the
       (back_prop_type=1) rho gradient went from ratio -33 (missing this
       factor entirely, i.e. only /dt) to 0.9999 (this exact dh^2/dt^3). */
    for (i=0;i<num_ele;i++){
        gradrho[i] *= (m->dh*m->dh)/(m->dt*m->dt*m->dt)*scale2;
    }
    if (M){
        for (i=0;i<num_ele;i++){
            gradM[i] *= scale2/m->dt;
        }
    }
    if (mu){
        for (i=0;i<num_ele;i++){
            gradmu[i] *= scale2/m->dt;
        }
    }
    return state;
}

/* The BACK_PROP_TYPE==2 (DFT) analog of unscale_grad(): the frequency-domain
   correlation (grad_dft{2,3}D.cl, calc_grad()'s host DFT branch, and their
   shared numpy reference SeisCL/tests/dft_reference.py) has never applied any
   equivalent of unscale_grad()'s dt/dh unit conversion, on the reasoning that
   it is "a spurious global factor" the old proportional-only finite-
   difference check could not see anyway (time_stepping.c's comment, now
   stale). Derived 2026-09-03 (notes/todo.md item 0h), not fitted: per
   Fabien-Ouellet et al. (2016) eq. A1b/A2a, dJ/dM = -c1^M*P1 with
   P1 = <adjoint_trace, d/dt(forward_trace)>. Parseval applied to that
   *derivative* correlation (pairing DFT bin k with -k via real-signal
   conjugate symmetry, which is exactly what folds the two-sided spectrum
   onto the positive-frequency-only bins the kernel sums) gives
   P1 = (2/dt) * sum_bins(d0), i.e. the kernel's own per-bin sum (no extra
   factor, as coded) equals (dt/2)*P1, not P1 itself. Separately, the
   kernel's `iden`/`i3den`/`i2ndmu2` coefficients are evaluated at the
   *physically rescaled* M_p/mu_p (grad_dft2D.cl's own dh/dt conversion),
   whereas BACK_PROP_TYPE==1's raw kernel-accumulated gradM uses the same
   coefficient evaluated at *raw* (internally non-dimensionalized) M/mu and
   then gets a single extra 1/dt from unscale_grad() above -- dividing the
   raw-M coefficient by the physically-scaled one shows they differ by
   exactly (dh/dt)^2. Combining both factors:
     (2/dt, the DFT correlation's own missing factor)
     x ((dh/dt)^2 / (1/dt), the raw-vs-physical coefficient difference)
     = 2*dh^2/dt^4.
   The same derivation applies unchanged to gradmu (i3den/i2ndmu2 share
   iden's M_p/mu_p scaling) and to gradrho/gradrip/gradrjp/gradrkp (raw,
   unconverted velocity-velocity correlations, matching unscale_grad()'s own
   rho path, which needs dh^2/dt^3 against gradM/gradmu's 1/dt -- the extra
   dh^2/dt factor cancels against the DFT correlation's own missing 2/dt
   piece the same way for every parameter, so one uniform factor applies to
   every to_grad array here, not a per-parameter one).

   Verified against the one fully-trusted reference available
   (BACK_PROP_TYPE==1, velocity output, itself calibrated to <0.1% -- see
   notes/todo.md item 1): isolated single-parameter finite-difference checks
   (2D crosswell) gave ad/fd = 1.005 (vp), 1.022 (vs), 0.945 (rho) after this
   fix, all within this codebase's ordinary FD-check precision. NOT verified
   for 3D's rho: 3D's DFT kernel writes gradrho directly, bypassing the
   gradrip/gradrjp/gradrkp + average_grad_transpose() path 2D uses, which
   is a separate, pre-existing material-averaging gap (notes/todo.md item 1)
   this fix does not address -- 3D vp/vs still calibrate to ~1-3% but 3D rho
   remains ~35% off. HOUT (Hessian) is a different kind of quantity (an
   auto-power spectrum, not a P1-style cross-correlation) and is
   deliberately left uncorrected here -- not re-derived.

   No FP16 (`par_scale`) factor here, unlike unscale_grad(): tried
   multiplying by the same `scale2=2^par_scale` unscale_grad() uses, by
   analogy -- broke test_dft_gradient_every_fp16_level (FP16=1 came back a
   full magnitude off FP16=0, previously exactly matching). The DFT kernel
   already fully accounts for FP16 internally (grad_dft2D.cl's own
   `sc_ss`/`sc_vv`/`s2` factors, see that file's own comment on the
   src_scale/res_scale/par_scale bookkeeping) -- unlike BACK_PROP_TYPE=1's
   raw kernel accumulation, which carries no FP16 correction of its own and
   so needs unscale_grad()'s scale2 to supply one. Adding a second, redundant
   par_scale factor here double-counts it. This function's own derivation
   was verified only at FP16=0 (where scale2 is a no-op either way), so this
   is confirmed by elimination, not re-derived from first principles -- flag
   for anyone revisiting FP16 + BACK_PROP_TYPE=2 together.

   2026-11-14: that conclusion was wrong. test_dft_gradient_every_fp16_level
   compares cosines only, so it cannot see a global factor. Measured against
   FP16=0 (same model and data): FP16=1 was 2^(-par_scale) too large in 2-D
   (2^20, every parameter, cos 1.00000), and in 3-D also 2^(src_scale+
   res_scale) (grad_dft3D.cl had no sc_ss/sc_vv, now added). The factor
   2^par_scale is the same one unscale_grad() applies; it is applied here. */
int unscale_grad_dft(model * m) {
    int state=0;
    int i, j;
    float dftcal = 2.0f*(m->dh*m->dh)/(m->dt*m->dt*m->dt*m->dt);
    float scale2 = powf(2.0f, (float)m->par_scale);   /* 1 at FP16=0 */

    for (j=0;j<m->npars;j++){
        if (m->pars[j].to_grad && m->pars[j].gl_grad){
            float * g = m->pars[j].gl_grad;
            int num_ele = m->pars[j].num_ele;
            for (i=0;i<num_ele;i++){
                g[i] *= dftcal*scale2;
            }
        }
    }
    return state;
}

/* Map the internal (M, mu, rho) gradient onto m->par_type. Expects physical
   units, i.e. unscale_par_grad() already run (and, once it exists, the
   material-averaging transpose too -- that maps the staggered gradients onto
   the cell-centred ones, which is a different chain rule and belongs before
   this one). */
/* d(M)/d(vp) and d(mu)/d(vs) for par_type==0.

   M() and mu() (assign_modeling_case.c) do NOT store rho*vp^2: for L>0 they
   divide by the GSLS phase-velocity normalization, so that the phase
   velocity at f0 is the vp the user supplied (the elastic convention):

       M = rho*vp^2 / (1 + alpha*taup),   alpha = sum_l r^2/(1+r^2), r=f0/FL_l

   Therefore dM/dvp = 2*rho*vp/(1+alpha*taup), whereas the elastic identity
   2*sqrt(rho*M) evaluates to 2*rho*vp/sqrt(1+alpha*taup) -- too large by
   exactly sqrt(1+alpha*taup). This function returns that missing
   1/sqrt(1+alpha*tau), which is identically 1 when L==0.

   UPDATE 2026-11-17: the DFT kernels now give the gradient with respect to the
   paper's M_paper = rho vp^2 (GJI 2017 Appendix A is written for it), so the
   chain rule uses dM_paper/dvp = 2 rho vp = 2 sqrt(rho M) sqrt(1+alpha taup):
   chain_rule_par_type MULTIPLIES by sqrt(1+alpha tau). The measurements above
   were taken with the coefficients evaluated at the stored M, where the two
   errors partly cancelled at FL = f0.

   Measured: a tau-scan at fixed alpha (FL=f0, alpha=0.5) on the 2D
   viscoelastic FD check gives a raw gradvp/FD ratio drifting 0.998 -> 1.019
   over taup = 0.005 -> 0.1, while ratio/sqrt(1+alpha*taup) is flat to 0.22%
   (0.9966 / 0.9964 / 0.9959 / 0.9944) -- i.e. the whole tau dependence of
   gradvp's calibration is this factor. See notes/todo.md item 0k.

   NOT to be applied to the gradrho cross terms: dM/drho|_{vp,taup} =
   vp^2/(1+alpha*taup) = M/rho exactly, so those are already correct. And
   NOT to gradtaup/gradtaus: the published coefficients (eq. A4i-A4p, the
   (L-alpha) group) already carry the fixed-vp dependence -- verified
   numerically, a missing term there would be 16x larger than the observed
   taup discrepancy. */
static double gsls_alpha(model * m){
    int l;
    double al = 0.0;
    if (m->L <= 0)
        return 0.0;
    float * FL = get_cst(m->csts, m->ncsts, "FL")->gl_cst;
    if (!FL)
        return 0.0;
    for (l=0;l<m->L;l++){
        if (FL[l] <= 0.0f)
            continue;
        double r = (double)m->f0/(double)FL[l];
        al += r*r/(1.0 + r*r);
    }
    return al;
}

int chain_rule_par_type(model * m) {
    int state=0;
    int i, num_ele=0;

    /* The phase-velocity normalization factors, 1 when elastic. */
    double al = gsls_alpha(m);
    float * taup_par = (m->L>0)
                     ? get_par(m->pars, m->npars, "taup")->gl_par : NULL;
    float * taus_par = (m->L>0)
                     ? get_par(m->pars, m->npars, "taus")->gl_par : NULL;

    float * rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float * gradrho = get_par(m->pars, m->npars, "rho")->gl_grad;
    float * Hrho = get_par(m->pars, m->npars, "rho")->gl_H;
    num_ele = get_par(m->pars, m->npars, "rho")->num_ele;
    float * M = get_par(m->pars, m->npars, "M")->gl_par;
    float * gradM = get_par(m->pars, m->npars, "M")->gl_grad;
    float * HM = get_par(m->pars, m->npars, "M")->gl_H;
    float * mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float * gradmu = get_par(m->pars, m->npars, "mu")->gl_grad;
    float * Hmu = get_par(m->pars, m->npars, "mu")->gl_H;

    if (m->par_type==0){

        for (i=0;i<num_ele;i++){
            /*rho[i]==0 marks a vacuum cell (M[i]==mu[i]==0 there too) --
              there is no real material to compute a density sensitivity
              for, so leave gradrho[i] as accumulated rather than divide
              0/0.*/
            if (rho[i]==0){
                continue;
            }
            /* With L>0 the DFT kernels give dJ/dM and dJ/dmu for the paper's
               M = rho vp^2, mu = rho vs^2 (GJI 2017 Appendix A), while M[] and
               mu[] hold M/(1+alpha taup), mu/(1+alpha taus). kp, ks = 1 when
               elastic. */
            double kp = (taup_par) ? 1.0+al*(double)taup_par[i] : 1.0;
            double ks = (taus_par) ? 1.0+al*(double)taus_par[i] : 1.0;
            if (M) gradrho[i]= gradrho[i]+M[i]*kp/rho[i]*gradM[i];   /* SH (ND 21) has no M */
            if (mu[i]>0){
                gradrho[i]= gradrho[i]+mu[i]*ks/rho[i]*gradmu[i];
            }
        }
        if (Hrho){
            for (i=0;i<num_ele;i++){
                if (rho[i]==0){
                    continue;
                }
                if (M) Hrho[i]= Hrho[i]+M[i]/rho[i]*HM[i];
                if (mu[i]>0){
                    Hrho[i]= Hrho[i]+mu[i]/rho[i]*Hmu[i];
                }
            }
        }
        /* gradM is dJ/dM_paper, M_paper = rho vp^2 = M (1+alpha taup), so
           dJ/dvp = 2 rho vp dJ/dM_paper = 2 sqrt(rho M) sqrt(1+alpha taup)
           dJ/dM_paper (2026-11-17: the kernels now evaluate Appendix A at
           M_paper; before, they used the stored M and this line divided by
           sqrt(1+alpha taup)). nrmp/nrms are 1 when L==0. */
        if (M){
            for (i=0;i<num_ele;i++){
                double nrmp = taup_par ? sqrt(1.0+al*(double)taup_par[i]) : 1.0;
                gradM[i]  = 2.0*sqrt((double)rho[i]*(double)M[i])*nrmp*gradM[i];   /* 2 sqrt(rho M_paper) */
            }
        }
        if (HM){
            for (i=0;i<num_ele;i++){
                double nrmp = taup_par ? sqrt(1.0+al*(double)taup_par[i]) : 1.0;
                HM[i]  = 2.0*sqrt((double)rho[i]*(double)M[i])/nrmp*HM[i];
            }
        }
        if (mu){
            for (i=0;i<num_ele;i++){
                double nrms = taus_par ? sqrt(1.0+al*(double)taus_par[i]) : 1.0;
                gradmu[i] = 2.0*sqrt((double)rho[i]*(double)mu[i])*nrms*gradmu[i];   /* 2 sqrt(rho mu_paper) */
            }
        }
        if (Hmu){
            for (i=0;i<num_ele;i++){
                double nrms = taus_par ? sqrt(1.0+al*(double)taus_par[i]) : 1.0;
                Hmu[i] = 2.0*sqrt((double)rho[i]*(double)mu[i])/nrms*Hmu[i];
            }
        }
    }
    else if (m->par_type==1){

    }
    else if (m->par_type==2){
        for (i=0;i<num_ele;i++){
            /*See the matching par_type==0 comment above: rho[i]==0 is a
              vacuum cell, skip rather than divide 0/0. par_type==2 divides
              by rho[i] even more than par_type==0 (M[i]/rho[i] appears
              inside the gradM/gradmu sqrt() too, not just gradrho), so
              every one of those sites needs the same guard.*/
            if (rho[i]==0){
                continue;
            }
            if (M) gradrho[i]= gradrho[i]+M[i]/rho[i]*gradM[i];
            if (mu[i]>0){
                gradrho[i]= gradrho[i]+mu[i]/rho[i]*gradmu[i];
            }
        }

        if (Hrho){
            for (i=0;i<num_ele;i++){
                if (rho[i]==0){
                    continue;
                }
                if (M) Hrho[i]= Hrho[i]+M[i]/rho[i]*HM[i];
                if (mu[i]>0){
                    Hrho[i]= Hrho[i]+mu[i]/rho[i]*Hmu[i];
                }
            }
        }
        if (M){
            for (i=0;i<num_ele;i++){
                gradM[i]  = rho[i]==0 ? 0
                          : 2.0*sqrt((double)M[i]/(double)rho[i])*gradM[i];
            }
        }
        if (HM){
            for (i=0;i<num_ele;i++){
                HM[i]  = rho[i]==0 ? 0
                       : 2.0*sqrt((double)M[i]/(double)rho[i])*HM[i];
            }
        }
        if (mu){
            for (i=0;i<num_ele;i++){
                gradmu[i] = rho[i]==0 ? 0
                          : 2.0*sqrt((double)mu[i]/(double)rho[i])*gradmu[i];
            }
        }
        if (Hmu){
            for (i=0;i<num_ele;i++){
                Hmu[i] = rho[i]==0 ? 0
                       : 2.0*sqrt((double)mu[i]/(double)rho[i])*Hmu[i];
            }
        }
    }
    else{
        fprintf(stdout,"Warning: Gradiant transformation not implemented: ");
        fprintf(stdout,"Outputting grad for M,mu,rho parametrization\n");
    }
    return state;
}

/* The original composition: storage, then units, then parameterization. */
int transf_grad(model * m) {
    int state=0;
    __GUARD unpack_par_fp16(m);
    __GUARD average_grad_transpose(m);
    __GUARD unscale_par(m);
    __GUARD unscale_grad(m);
    __GUARD chain_rule_par_type(m);



    return state;

}



