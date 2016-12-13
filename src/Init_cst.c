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

#define rho(z,y,x) rho[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define rip(z,y,x) rip[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define rjp(z,y,x) rjp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define rkp(z,y,x) rkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define uipjp(z,y,x) uipjp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define ujpkp(z,y,x) ujpkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define uipkp(z,y,x) uipkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define u(z,y,x) u[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define pi(z,y,x) pi[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define grad(z,y,x) grad[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define grads(z,y,x) grads[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define amp1(z,y,x) amp1[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define amp2(z,y,x) amp2[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

#define vxout(y,x) vxout[(y)*m->NT+(x)]
#define vyout(y,x) vyout[(y)*m->NT+(x)]
#define vzout(y,x) vzout[(y)*m->NT+(x)]
#define vx0(y,x) vx0[(y)*m->NT+(x)]
#define vy0(y,x) vy0[(y)*m->NT+(x)]
#define vz0(y,x) vz0[(y)*m->NT+(x)]
#define rx(y,x) rx[(y)*m->NT+(x)]
#define ry(y,x) ry[(y)*m->NT+(x)]
#define rz(y,x) rz[(y)*m->NT+(x)]

#define vxcum(y,x) vxcum[(y)*m->NT+(x)]
#define vycum(y,x) vycum[(y)*m->NT+(x)]

#define u_in(z,y,x) u_in[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define pi_in(z,y,x) pi_in[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define uL(z,y,x) uL[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define piL(z,y,x) piL[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define taus(z,y,x) taus[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tausipjp(z,y,x) tausipjp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tausjpkp(z,y,x) tausjpkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tausipkp(z,y,x) tausipkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define taup(z,y,x) taup[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]


#define PI (3.141592653589793238462643383279502884197169)


int holbergcoeff(struct modcsts *inm) {
    /*------------------------------------------------------------------------
     * Copyright (C) 2011 For the list of authors, see file AUTHORS.
     *
     * This file is part of SOFI2D.
     *
     * SOFI2D is free software: you can redistribute it and/or modify
     * it under the terms of the GNU General Public License as published by
     * the Free Software Foundation, version 2.0 of the License only.
     *
     * SOFI2D is distributed in the hope that it will be useful,
     * but WITHOUT ANY WARRANTY; without even the implied warranty of
     * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     * GNU General Public License for more details.
     *
     * You should have received a copy of the GNU General Public License
     * along with SOFI2D. See file COPYING and/or
     * <http://www.gnu.org/licenses/gpl-2.0.html>.
     --------------------------------------------------------------------------*/
    /* -------------------------------------------------------------
     * Holberg coefficients for a certain FD order and a margin of error E
     * (MAXRELERROR)
     *
     * MAXRELERROR = 0 -> Taylor-coeff.
     * MAXRELERROR = 1 -> Holberg-coeff.: E = 0.1 %
     * MAXRELERROR = 2 ->                 E = 0.5 %
     * MAXRELERROR = 3 ->                 E = 1.0 %
     * MAXRELERROR = 4 ->                 E = 3.0 %
     *
     * hc: column 0 = minimum number of grid points per shortest wavelength
     *     columns 1-6 = Holberg coefficients
     *
     * ------------------------------------------------------------- */
    int i;
    int state=0;
    
    
    float hcall[5][6][7] =
    {
        {
            { 23.0, 1.0,                0.0,              0.0,                0.0,              0.0,              0.0            },
            {  8.0, 9.0/8.0,           -1.0/24.0,         0.0,                0.0,              0.0,              0.0            },
            {  6.0, 75.0/64.0,         -25.0/384.0,       3.0/640.0,          0.0,              0.0,              0.0            },
            {  5.0, 1225.0/1024.0,     -245.0/3072.0,     49.0/5120.0,       -5.0/7168.0,       0.0,              0.0            },
            {  5.0, 19845.0/16384.0,   -735.0/8192.0,     567.0/40960.0,     -405.0/229376.0,   35.0/294912.0,    0.0            },
            {  4.0, 160083.0/131072.0, -12705.0/131072.0, 22869.0/1310720.0, -5445.0/1835008.0, 847.0/2359296.0, -63.0/2883584.0 }
        },
        {
            { 49.7 , 1.0010,  0.0,      0.0,        0.0,       0.0,        0.0       },
            {  8.32, 1.1382, -0.046414, 0.0,        0.0,       0.0,        0.0       },
            {  4.77, 1.1965, -0.078804, 0.0081781,  0.0,       0.0,        0.0       },
            {  3.69, 1.2257, -0.099537, 0.018063,  -0.0026274, 0.0,        0.0       },
            {  3.19, 1.2415, -0.11231,  0.026191,  -0.0064682, 0.001191,   0.0       },
            {  2.91, 1.2508, -0.12034,  0.032131,  -0.010142,  0.0029857, -0.00066667}
        },
        {
            { 22.2 , 1.0050,  0.0,      0.0,        0.0,       0.0,        0.0       },
            {  5.65, 1.1534, -0.052806, 0.0,        0.0,       0.0,        0.0       },
            {  3.74, 1.2111, -0.088313, 0.011768,   0.0,       0.0,        0.0       },
            {  3.11, 1.2367, -0.10815,  0.023113,  -0.0046905, 0.0,        0.0       },
            {  2.80, 1.2496, -0.11921,  0.031130,  -0.0093272, 0.0025161,  0.0       },
            {  2.62, 1.2568, -0.12573,  0.036423,  -0.013132,  0.0047484, -0.0015979 }
        },
        {
            { 15.8,  1.0100,  0.0,      0.0,        0.0,       0.0,        0.0       },
            {  4.80, 1.1640, -0.057991, 0.0,        0.0,       0.0,        0.0       },
            {  3.39, 1.2192, -0.094070, 0.014608,   0.0,       0.0,        0.0       },
            {  2.90, 1.2422, -0.11269,  0.026140,  -0.0064054, 0.0,        0.0       },
            {  2.65, 1.2534, -0.12257,  0.033755,  -0.011081,  0.0036784,  0.0       },
            {  2.51, 1.2596, -0.12825,  0.038550,  -0.014763,  0.0058619, -0.0024538 }
        },
        {
            {  9.16, 1.0300,  0.0,      0.0,        0.0,       0.0,        0.0       },
            {  3.47, 1.1876, -0.072518, 0.0,        0.0,       0.0,        0.0       },
            {  2.91, 1.2341, -0.10569,  0.022589,   0.0,       0.0,        0.0       },
            {  2.61, 1.2516, -0.12085,  0.032236,  -0.011459,  0.0,        0.0       },
            {  2.45, 1.2596, -0.12829,  0.038533,  -0.014681,  0.0072580,  0.0       },
            {  2.36, 1.2640, -0.13239,  0.042217,  -0.017803,  0.0081959, -0.0051848 }
        },
    };
    
    
    if (((*inm).FDORDER!=2) && ((*inm).FDORDER!=4) && ((*inm).FDORDER!=6) && ((*inm).FDORDER!=8) && ((*inm).FDORDER!=10) && ((*inm).FDORDER!=12)) {
        state=1;
        fprintf(stderr," Error in selection of FD coefficients: wrong FDORDER! ");
    }
    
    if (((*inm).MAXRELERROR<0) || ((*inm).MAXRELERROR>4)) {
        state =1;
        fprintf(stderr," Error in selection of FD coefficients: wrong choice of maximum relative error! ");
    }
    
    for (i=0; i<=6; i++) {
        (*inm).hc[i] = hcall[(*inm).MAXRELERROR][(*inm).fdoh-1][i];
        //fprintf(stderr,"hc[%i]= %5.5f \n ",i,hc[i]);
    }
    
    return state;
    
}


int Init_cst(struct modcsts * m) {
    
    int state=0;
    int i;
    
    //Calculate some useful constants
    m->nsmax=0;
    m->ngmax=0;
    for (i=0;i<m->ns; i++){
        m->nsmax = fmax(m->nsmax, m->nsrc[i]);
        m->ngmax = fmax(m->ngmax, m->nrec[i]);
    }
    m->dhi = 1.0/m->dh;
    m->fdo=m->FDORDER/2 + 1;
    m->fdoh=m->FDORDER/2;
    
    
    if (!state) if (holbergcoeff(m))                {state=1; fprintf(stderr,"Could not determine holberg coefficients\n");};
    
    //Create the taper zone for absorbing boundary
    if (m->abs_type==2){
        GMALLOC(m->taper ,m->nab*sizeof(float))
        float amp=1-m->abpc/100;
        float a=sqrt(-log(amp)/((m->nab-1)*(m->nab-1)));
        for (i=1; i<=m->nab; i++) {
            m->taper[i-1]=exp(-(a*a*(m->nab-i)*(m->nab-i)));
        }
    }
    else if (m->abs_type==1){
        GMALLOC(m->K_x  ,2*m->nab*sizeof(float))
        GMALLOC(m->a_x  ,2*m->nab*sizeof(float))
        GMALLOC(m->b_x  ,2*m->nab*sizeof(float))
        GMALLOC(m->K_x_half  ,2*m->nab*sizeof(float))
        GMALLOC(m->a_x_half  ,2*m->nab*sizeof(float))
        GMALLOC(m->b_x_half  ,2*m->nab*sizeof(float))
        
        if (m->ND==3){// For 3D only
            GMALLOC(m->K_y  ,2*m->nab*sizeof(float))
            GMALLOC(m->a_y  ,2*m->nab*sizeof(float))
            GMALLOC(m->b_y  ,2*m->nab*sizeof(float))
            GMALLOC(m->K_y_half  ,2*m->nab*sizeof(float))
            GMALLOC(m->a_y_half  ,2*m->nab*sizeof(float))
            GMALLOC(m->b_y_half  ,2*m->nab*sizeof(float))
        }
        
        GMALLOC(m->K_z  ,2*m->nab*sizeof(float))
        GMALLOC(m->a_z  ,2*m->nab*sizeof(float))
        GMALLOC(m->b_z  ,2*m->nab*sizeof(float))
        GMALLOC(m->K_z_half  ,2*m->nab*sizeof(float))
        GMALLOC(m->a_z_half  ,2*m->nab*sizeof(float))
        GMALLOC(m->b_z_half  ,2*m->nab*sizeof(float))
        
        
        CPML_coeff(m);
        
        
    }
    
    if (m->L>0){

        GMALLOC(m->eta ,m->L*sizeof(float))
 
    }
    
    
    //Create averaged properties
    if (m->ND!=21){
        GMALLOC(m->rip  ,m->NX*m->NY*m->NZ*sizeof(float))
        GMALLOC(m->rkp  ,m->NX*m->NY*m->NZ*sizeof(float))
        GMALLOC(m->uipkp,m->NX*m->NY*m->NZ*sizeof(float))
    }
    if (m->ND==3 || m->ND==21){
        GMALLOC(m->rjp  ,m->NX*m->NY*m->NZ*sizeof(float))
        GMALLOC(m->uipjp,m->NX*m->NY*m->NZ*sizeof(float))
        GMALLOC(m->ujpkp,m->NX*m->NY*m->NZ*sizeof(float))
    }
    
    if (m->L){
        if (m->ND!=21){
            GMALLOC(m->tausipkp,m->NX*m->NY*m->NZ*sizeof(float))
        }
        if (m->ND==3 || m->ND==21){// For 3D only
            GMALLOC(m->tausjpkp,m->NX*m->NY*m->NZ*sizeof(float))
            GMALLOC(m->tausipjp,m->NX*m->NY*m->NZ*sizeof(float))
            
        }
    }
    

    //Alocate memory for the gradient
    if (m->gradout==1 ){
        
        GMALLOC(m->gradrho  ,m->NX*m->NY*m->NZ*sizeof(double))
        if (m->ND!=21){
            GMALLOC(m->gradM  ,m->NX*m->NY*m->NZ*sizeof(double))
        }
        GMALLOC(m->gradmu  ,m->NX*m->NY*m->NZ*sizeof(double))
        
        if (m->L>0){
            if (m->ND!=21){
                GMALLOC(m->gradtaup  ,m->NX*m->NY*m->NZ*sizeof(double))
            }
            GMALLOC(m->gradtaus  ,m->NX*m->NY*m->NZ*sizeof(double))
            
        }
        
        
        
        
        
        if (m->back_prop_type==2){
            
            GMALLOC(m->gradfreqsn  ,m->NT*sizeof(float))
            

            if (m->Hout){
                GMALLOC(m->H.pp,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.mp,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.up,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.tpp,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.tsp,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                
                GMALLOC(m->H.mm,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.um,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.tpm,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.tsm,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                
                GMALLOC(m->H.uu,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.tpu,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.tsu,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                
                GMALLOC(m->H.tptp,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                GMALLOC(m->H.tstp,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));
                
                GMALLOC(m->H.tsts,m->nfreqs*m->NX*m->NY*m->NZ*sizeof(double));

                
                
            }
            
            
        }
        
    }
    
    if (m->gradsrcout==1 ){
        GMALLOC(m->gradsrc,sizeof(float*)*m->ns)
        if (!state) memset((void*)m->gradsrc, 0, sizeof(float*)*m->ns);
        GMALLOC(m->gradsrc[0],sizeof(float)*m->allns*m->NT)
        for (i=1;i<m->ns;i++){
            m->gradsrc[i]=m->gradsrc[i-1]+m->nsrc[i-1]*m->NT;
        }    
    }
    
    
    return state;
    
    
    
}