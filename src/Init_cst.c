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


int CPML_coeff(float NPOWER, float k_max_CPML, float FPML, float VPPML, float dh, float dt, int NAB, float * K_i, float * b_i, float * a_i, float * K_i_half, float * b_i_half, float * a_i_half)
{
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
    
    /*------------------------------------------------------------------------
     * Copyright (C) 2011 For the list of authors, see file AUTHORS.
     *
     * This file is part of SOFI3D.
     *
     * SOFI3D is free software: you can redistribute it and/or modify
     * it under the terms of the GNU General Public License as published by
     * the Free Software Foundation, version 2.0 of the License only.
     *
     * SOFI3D is distributed in the hope that it will be useful,
     * but WITHOUT ANY WARRANTY; without even the implied warranty of
     * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     * GNU General Public License for more details.
     *
     * You should have received a copy of the GNU General Public License
     * along with SOFI3D. See file COPYING and/or
     * <http://www.gnu.org/licenses/gpl-2.0.html>.
     --------------------------------------------------------------------------*/
    /* ----------------------------------------------------------------------
     * Define damping profiles for CPML boundary condition
     * This C-PML implementation is adapted from the 2nd order isotropic CPML code by Dimitri Komatitsch and based in part on formulas given in Roden and Gedney (2000).
     * Additionally the code is based on the following references:
     *
     * @ARTICLE{KoMa07,
     * author = {Dimitri Komatitsch and Roland Martin},
     * title = {An unsplit convolutional {P}erfectly {M}atched {L}ayer improved
     *          at grazing incidence for the seismic wave equation},
     * journal = {Geophysics},
     * year = {2007},
     * volume = {72},
     * number = {5},
     * pages = {SM155-SM167},
     * doi = {10.1190/1.2757586}}
     *
     * @ARTICLE{MaKoEz08,
     * author = {Roland Martin and Dimitri Komatitsch and Abdela\^aziz Ezziani},
     * title = {An unsplit convolutional perfectly matched layer improved at grazing
     * incidence for seismic wave equation in poroelastic media},
     * journal = {Geophysics},
     * year = {2008},
     * volume = {73},
     * pages = {T51-T61},
     * number = {4},
     * doi = {10.1190/1.2939484}}
     *
     * @ARTICLE{MaKo09,
     * author = {Roland Martin and Dimitri Komatitsch},
     * title = {An unsplit convolutional perfectly matched layer technique improved
     * at grazing incidence for the viscoelastic wave equation},
     * journal = {Geophysical Journal International},
     * year = {2009},
     * volume = {179},
     * pages = {333-344},
     * number = {1},
     * doi = {10.1111/j.1365-246X.2009.04278.x}}
     *
     * @ARTICLE{MaKoGe08,
     * author = {Roland Martin and Dimitri Komatitsch and Stephen D. Gedney},
     * title = {A variational formulation of a stabilized unsplit convolutional perfectly
     * matched layer for the isotropic or anisotropic seismic wave equation},
     * journal = {Computer Modeling in Engineering and Sciences},
     * year = {2008},
     * volume = {37},
     * pages = {274-304},
     * number = {3}}
     *
     * The original CPML technique for Maxwell's equations is described in:
     *
     * @ARTICLE{RoGe00,
     * author = {J. A. Roden and S. D. Gedney},
     * title = {Convolution {PML} ({CPML}): {A}n Efficient {FDTD} Implementation
     *          of the {CFS}-{PML} for Arbitrary Media},
     * journal = {Microwave and Optical Technology Letters},
     * year = {2000},
     * volume = {27},
     * number = {5},
     * pages = {334-339},
     * doi = {10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A}}
     * extended version of sofi2D
     ----------------------------------------------------------------------*/
    
    int state=0;
    int i, l,i1;
    
    float * d_i=NULL,* d_i_half=NULL;
    float *alpha_prime_i, *alpha_prime_i_half;
    
    /* read from iput file
     npower -> power to compute d0 profile, exponent of damping, the larger the more damping,
     -> default npower=2
     k_max_CPML -> if wave incident angle is less than 90° with respect to CPML interface, k_max_CPML shoudl be increased
     -> default k_max_CMPL =10.0
     */
    
    const float npower = NPOWER;  /*  power to compute d0 profile */
    /* K_MAX_CPML(from Gedney page 8.11) */
    const float alpha_max_PML = 2.0 * PI * (FPML/2.0);   /* from festa and Vilotte 2.0*...*/
    const float Rcoef = 0.0008;       /* reflection coefficient (INRIA report section 6.1) */
    const float a = 0.25, b = 0.75 , c = 0.0;
    float d0_i, position_norm, position_in_PML;
    
    
    d_i = malloc(sizeof(float)*2*NAB);
    d_i_half = malloc(sizeof(float)*2*NAB);
    
    alpha_prime_i = malloc(sizeof(float)*2*NAB);
    alpha_prime_i_half = malloc(sizeof(float)*2*NAB);
    
    
    
    /* compute d0 from INRIA report section 6.1 */
    d0_i = - (npower + 1) * VPPML * log(Rcoef) / (2.0 * NAB*dh);
    
    /* damping in the X direction */
    /* -------------------------- */
    
   	for (i=0;i<=NAB;i++){
        
        K_i[i] = 1.0;
        
        /* define damping profile at the grid points */
        position_in_PML = (NAB-i)*dh; /*distance to boundary in meter */
        position_norm = position_in_PML / (NAB*dh); /*normalised by PML thickness*/
        
        d_i[i] = d0_i *(a*position_norm+b*pow(position_norm,npower)+c*pow(position_norm,(4)));
        
        /* this taken from Gedney page 8.2 */
        K_i[i] = 1.0 + (k_max_CPML - 1.0) * pow(position_norm,npower);
        alpha_prime_i[i] = alpha_max_PML * (1.0 - position_norm);
        
        if(alpha_prime_i[i] < 0.0){state=1; fprintf(stderr,"ERROR:alpha_prime_i[i] < 0.0, i %d", i);}
        
        b_i[i] = exp(- (d_i[i] / K_i[i] + alpha_prime_i[i]) * dt);
        
        /* avoid division by zero outside the PML */
        if(fabsf(d_i[i]) > 1.0e-6){ a_i[i] = d_i[i] * (b_i[i] - 1.0) / (K_i[i] * (d_i[i] + K_i[i] * alpha_prime_i[i]));}
        else a_i[i]=0.0;
        
        if(i<=NAB-1){
            
            /* define damping profile at half the grid points (half a grid point in -x)*/
            position_in_PML = (NAB-i+0.5-1.0) *dh;
            position_norm = position_in_PML / (NAB*dh);
            
            i1=i;
            
            K_i_half[i1] = 1.0;
            d_i_half[i1] = d0_i * (a*position_norm+b*pow(position_norm,npower)+c*pow(position_norm,(4)));
            
            if(position_in_PML < 0.0) {state=1;fprintf(stderr,"ERROR: Position in PML (x-boundary) smaller 0");}
            
            /* this taken from Gedney page 8.2 */
            K_i_half[i1] = 1.0 + (k_max_CPML - 1.0) * pow(position_norm,npower);
            alpha_prime_i_half[i1] = alpha_max_PML * (1.0 - position_norm);
            
            /* just in case, for -5 at the end */
            if(alpha_prime_i_half[i1] < 0.0) {state=1;fprintf(stderr,"ERROR:alpha_prime_i_half[i] < 0.0, i %d", i);}
            
            b_i_half[i1] = exp(- (d_i_half[i1] / K_i_half[i1] + alpha_prime_i_half[i1]) * dt);
            
            if(fabs(d_i_half[i1]) > 1.0e-6){ a_i_half[i1] = d_i_half[i1] * (b_i_half[i1] - 1.0) / (K_i_half[i1] * (d_i_half[i1] + K_i_half[i1] * alpha_prime_i_half[i1]));}
            
            /* right boundary --> mirroring left boundary*/
            
            l = 2* NAB -i-1;
            
            if(i>0){
                K_i[l+1]=K_i[i];
                b_i[l+1] = b_i[i];
                if(fabs(d_i[i]) > 1.0e-6){ a_i[l+1] = a_i[i]; }
            }
            
            K_i_half[l]=K_i_half[i];
            b_i_half[l] = b_i_half[i];  /*half a grid point in +x)*/
            if(fabs(d_i[i]) > 1.0e-6){ a_i_half[l] = a_i_half[i]; }
            
        } 
    }
    
    free(d_i);
    free(d_i_half);
    
    
    free(alpha_prime_i);
    free(alpha_prime_i_half);
    
    return state;
    
}


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
        (*inm).hc[i] = hcall[(*inm).MAXRELERROR][(*inm).FDOH-1][i];
        //fprintf(stderr,"hc[%i]= %5.5f \n ",i,hc[i]);
    }
    
    return state;
    
}


int Init_cst(struct modcsts * m) {
    
    int state=0;
    int i,j;
    float * K_i;
    float * a_i;
    float * b_i;
    float * K_i_half;
    float * a_i_half;
    float * b_i_half;
    
    
    __GUARD holbergcoeff(m);
    
    //Create the taper zone for absorbing boundary
    float * taper= m->csts[0].gl_cst;
    if (m->ABS_TYPE==2){
        float amp=1-m->abpc/100;
        float a=sqrt(-log(amp)/((m->NAB-1)*(m->NAB-1)));
        for (i=1; i<=m->NAB; i++) {
            taper[i-1]=exp(-(a*a*(m->NAB-i)*(m->NAB-i)));
        }
    }
    else if (m->ABS_TYPE==1){
        for (i=0;i<m->NDIM;i++){
            K_i=m->csts[1+i*6].gl_cst;
            a_i=m->csts[2+i*6].gl_cst;
            b_i=m->csts[3+i*6].gl_cst;
            K_i_half=m->csts[4+i*6].gl_cst;
            a_i_half=m->csts[5+i*6].gl_cst;
            b_i_half=m->csts[6+i*6].gl_cst;
            CPML_coeff(m->NPOWER, m->K_MAX_CPML, m->FPML, m->VPPML, m->dh, m->dt, m->NAB, K_i, b_i, a_i, K_i_half, b_i_half, a_i_half);
        }
    }
    
    //Viscoelastic constants initialization
    float * eta=m->csts[20].gl_cst;
    float * FL=m->csts[20].gl_cst;
    if (m->L>0){
        for (int l=0;l<m->L;l++) {
            eta[l]=(2.0*PI*FL[l])/m->dt;
        }
    }
    
    //Initialize the gradient
    if (m->GRADOUT==1 && m->BACK_PROP_TYPE==2){
        float * gradfreqs = m->csts[21].gl_cst;
        float * gradfreqsn = m->csts[22].gl_cst;
        float fmaxout=0;
        for (j=0;j<m->NFREQS;j++){
            if (gradfreqs[j]>fmaxout)
                fmaxout=gradfreqs[j];
        }
        float df;
        m->DTNYQ=ceil(0.0156/fmaxout/m->dt);
        m->NTNYQ=(m->tmax-m->tmin)/m->DTNYQ+1;
        df=1.0/m->NTNYQ/m->dt/m->DTNYQ;
        for (j=0;j<m->NFREQS;j++){
           gradfreqsn[j]=floor(gradfreqs[j]/df);
        }
    }
    
    return state;

}
