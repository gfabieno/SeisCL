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
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <stdint.h>
#define PI (3.141592653589793238462643383279502884197169)

int CPML_coeff(float NPOWER, float k_max_CPML, float FPML, float VPPML, float dh, float dt, int NAB, float * K_i, float * b_i, float * a_i, float * K_i_half, float * b_i_half, float * a_i_half)
{
  

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
        
        if(alpha_prime_i[i] < 0.0){state=1; fprintf(stderr,"ERROR: alpha_prime_i[i] < 0.0, i %d", i);}
        
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
            if(alpha_prime_i_half[i1] < 0.0) {state=1;fprintf(stderr,"ERROR: alpha_prime_i_half[i] < 0.0, i %d", i);}
            
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
