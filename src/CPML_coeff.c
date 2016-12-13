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

#include "F.h"

void CPML_coeff(struct modcsts * m)
{


	
	int i, l,i1;
	      
      float * d_x=NULL,* d_x_half=NULL,* d_y=NULL,* d_y_half=NULL,* d_z=NULL,* d_z_half=NULL;
      float *alpha_prime_x, *alpha_prime_x_half, *alpha_prime_y, *alpha_prime_y_half, *alpha_prime_z, *alpha_prime_z_half;
    
      /* read from iput file
       	  npower -> power to compute d0 profile, exponent of damping, the larger the more damping,
      	  	  	 -> default npower=2
      	  k_max_CPML -> if wave incident angle is less than 90° with respect to CPML interface, k_max_CPML shoudl be increased
      	  	  	 -> default k_max_CMPL =10.0
	  */

      const float npower = m->NPOWER;  /*  power to compute d0 profile */
      const float k_max_CPML = m->K_MAX_CPML;   /* (from Gedney page 8.11) */
      const float alpha_max_PML = 2.0 * PI * (m->FPML/2.0);   /* from festa and Vilotte 2.0*...*/
      const float Rcoef = 0.0008;       /* reflection coefficient (INRIA report section 6.1) */
      const float a = 0.25, b = 0.75 , c = 0.0; 
      float d0_x, d0_y, d0_z, position_norm, position_in_PML;

    
      d_x = malloc(sizeof(float)*2*m->nab);
      d_x_half = malloc(sizeof(float)*2*m->nab);
      d_y = malloc(sizeof(float)*2*m->nab);
      d_y_half = malloc(sizeof(float)*2*m->nab);
      d_z = malloc(sizeof(float)*2*m->nab);
      d_z_half = malloc(sizeof(float)*2*m->nab);
    
      alpha_prime_x = malloc(sizeof(float)*2*m->nab);
      alpha_prime_x_half = malloc(sizeof(float)*2*m->nab);
      alpha_prime_y = malloc(sizeof(float)*2*m->nab);
      alpha_prime_y_half = malloc(sizeof(float)*2*m->nab);
      alpha_prime_z = malloc(sizeof(float)*2*m->nab);
      alpha_prime_z_half = malloc(sizeof(float)*2*m->nab);

     
      /* compute d0 from INRIA report section 6.1 */
      d0_x = - (npower + 1) * m->VPPML * log(Rcoef) / (2.0 * m->nab*m->dh);
      d0_y = - (npower + 1) * m->VPPML * log(Rcoef) / (2.0 * m->nab*m->dh);
      d0_z = - (npower + 1) * m->VPPML * log(Rcoef) / (2.0 * m->nab*m->dh);


	
      /* damping in the X direction */
      /* -------------------------- */

   	for (i=0;i<=m->nab;i++){
	
        m->K_x[i] = 1.0;

        /* define damping profile at the grid points */
        position_in_PML = (m->nab-i)*m->dh; /*distance to boundary in meter */
        position_norm = position_in_PML / (m->nab*m->dh); /*normalised by PML thickness*/

        d_x[i] = d0_x *(a*position_norm+b*pow(position_norm,npower)+c*pow(position_norm,(4)));

        /* this taken from Gedney page 8.2 */
        m->K_x[i] = 1.0 + (k_max_CPML - 1.0) * pow(position_norm,npower);
        alpha_prime_x[i] = alpha_max_PML * (1.0 - position_norm);

	if(alpha_prime_x[i] < 0.0){ fprintf(stderr,"ERROR:alpha_prime_x[i] < 0.0, i %d", i);}
	
	m->b_x[i] = exp(- (d_x[i] / m->K_x[i] + alpha_prime_x[i]) * m->dt);

 	/* avoid division by zero outside the PML */
        if(fabsf(d_x[i]) > 1.0e-6){ m->a_x[i] = d_x[i] * (m->b_x[i] - 1.0) / (m->K_x[i] * (d_x[i] + m->K_x[i] * alpha_prime_x[i]));}
	else m->a_x[i]=0.0;
	
	if(i<=m->nab-1){

        /* define damping profile at half the grid points (half a grid point in -x)*/
        position_in_PML = (m->nab-i+0.5-1.0) *m->dh;
        position_norm = position_in_PML / (m->nab*m->dh);

	i1=i;
	
	m->K_x_half[i1] = 1.0;
        d_x_half[i1] = d0_x * (a*position_norm+b*pow(position_norm,npower)+c*pow(position_norm,(4)));

        if(position_in_PML < 0.0) {fprintf(stderr,"ERROR: Position in PML (x-boundary) smaller 0");}
          
        /* this taken from Gedney page 8.2 */
        m->K_x_half[i1] = 1.0 + (k_max_CPML - 1.0) * pow(position_norm,npower);
        alpha_prime_x_half[i1] = alpha_max_PML * (1.0 - position_norm);
           
        /* just in case, for -5 at the end */
        if(alpha_prime_x_half[i1] < 0.0) {fprintf(stderr,"ERROR:alpha_prime_x_half[i] < 0.0, i %d", i);}

        m->b_x_half[i1] = exp(- (d_x_half[i1] / m->K_x_half[i1] + alpha_prime_x_half[i1]) * m->dt);

        if(fabs(d_x_half[i1]) > 1.0e-6){ m->a_x_half[i1] = d_x_half[i1] * (m->b_x_half[i1] - 1.0) / (m->K_x_half[i1] * (d_x_half[i1] + m->K_x_half[i1] * alpha_prime_x_half[i1]));}

	/* right boundary --> mirroring left boundary*/
	
	l = 2* m->nab -i-1;

	if(i>0){
	m->K_x[l+1]=m->K_x[i];
	m->b_x[l+1] = m->b_x[i];
	if(fabs(d_x[i]) > 1.0e-6){ m->a_x[l+1] = m->a_x[i]; }
	}

	m->K_x_half[l]=m->K_x_half[i];
        m->b_x_half[l] = m->b_x_half[i];  /*half a grid point in +x)*/
        if(fabs(d_x[i]) > 1.0e-6){ m->a_x_half[l] = m->a_x_half[i]; }

        } 
	}


    if (m->ND==3){
      /* damping in the Y direction */
      /* -------------------------- */

        for (i=0;i<=m->nab;i++){
	
        m->K_y[i] = 1.0; 
          
        /* define damping profile at the grid points */
        position_in_PML = (m->nab-i)*m->dh; /*distance to boundary in meter */
        position_norm = position_in_PML / (m->nab*m->dh); /*normalised by PML thickness*/

        d_y[i] = d0_y * (a*position_norm+b*pow(position_norm,npower)+c*pow(position_norm,(4)));

        /* this taken from Gedney page 8.2 */
        m->K_y[i] = 1.0 + (k_max_CPML - 1.0) * pow(position_norm,npower);
        alpha_prime_y[i] = alpha_max_PML * (1.0 - position_norm);

	/* just in case, for -5 at the end */
        if(alpha_prime_y[i] < 0.0){ fprintf(stderr,"ERROR:alpha_prime_y[i] < 0.0, i %d", i);}

	m->b_y[i] = exp(- (d_y[i] / m->K_y[i] + alpha_prime_y[i]) * m->dt);

 	/* avoid division by zero outside the PML */
        if(fabs(d_y[i]) > 1.0e-6){ m->a_y[i] = d_y[i] * (m->b_y[i] - 1.0) / (m->K_y[i] * (d_y[i] + m->K_y[i] * alpha_prime_y[i]));}
      	else m->a_x[i]=0.0;

	if(i<=m->nab-1){

          /* define damping profile at half the grid points (half a grid point in -x)*/
        position_in_PML = (m->nab-i+0.5-1.0) *m->dh;
        position_norm = position_in_PML / (m->nab*m->dh);

	i1=i;
	m->K_y_half[i1] = 1.0;
        d_y_half[i1] = d0_y * (a*position_norm+b*pow(position_norm,npower)+c*pow(position_norm,(4)));

        if(position_in_PML < 0.0) {fprintf(stderr,"ERROR: Position in PML (y-boundary) <0");}
          
        /* this taken from Gedney page 8.2 */
        m->K_y_half[i1] = 1.0 + (k_max_CPML - 1.0) * pow(position_norm,npower);
        alpha_prime_y_half[i1] = alpha_max_PML * (1.0 - position_norm);
      
        if(alpha_prime_y_half[i1] < 0.0) {fprintf(stderr,"ERROR:alpha_prime_y_half[i] < 0.0, i %d", i);}
        m->b_y_half[i1] = exp(- (d_y_half[i1] / m->K_y_half[i1] + alpha_prime_y_half[i1]) * m->dt);
          
      	if(fabs(d_y_half[i1]) > 1.0e-6){ m->a_y_half[i1] = d_y_half[i1] * (m->b_y_half[i1] - 1.0) / (m->K_y_half[i1] * (d_y_half[i1] + m->K_y_half[i1] * alpha_prime_y_half[i1]));}
	
        /* right boundary --> mirroring left boundary*/
        l = 2* m->nab -i-1;
	
	if(i>0){
	m->K_y[l+1] = m->K_y[i];
	m->b_y[l+1] = m->b_y[i];
	if(fabs(d_y[i]) > 1.0e-6){ m->a_y[l+1] = m->a_y[i]; }
	}
  
	
	m->K_y_half[l]=m->K_y_half[i];
        m->b_y_half[l] = m->b_y_half[i];  /*half a grid point in +x)*/ 
        if(fabs(d_y[i]) > 1.0e-6){ m->a_y_half[l] = m->a_y_half[i]; }
	}
        } 
    }

       /* damping in the Z direction */
      /* -------------------------- */

	for (i=0;i<=m->nab;i++){
	
        m->K_z[i] = 1.0;
                    
    	/* define damping profile at the grid points */
      	position_in_PML = (m->nab-i)*m->dh; /*distance to boundary in meter */
      	position_norm = position_in_PML / (m->nab*m->dh); /*normalised by PML thickness*/

      	d_z[i] = d0_z * (a*position_norm+b*pow(position_norm,npower)+c*pow(position_norm,(4)));

  	/* this taken from Gedney page 8.2 */
    	m->K_z[i] = 1.0 + (k_max_CPML - 1.0) * pow(position_norm,npower);
    	alpha_prime_z[i] = alpha_max_PML * (1.0 - position_norm);
           
	/* just in case, for -5 at the end */
    	if(alpha_prime_z[i] < 0.0){ fprintf(stderr,"ERROR:alpha_prime_z[i] < 0.0, i %d", i);}

	m->b_z[i] = exp(- (d_z[i] / m->K_z[i] + alpha_prime_z[i]) * m->dt);

 	/* avoid division by zero outside the PML */
        if(fabs(d_z[i]) > 1.0e-6){ m->a_z[i] = d_z[i] * (m->b_z[i] - 1.0) / (m->K_z[i] * (d_z[i] + m->K_z[i] * alpha_prime_z[i]));}
	
	if(i<=m->nab-1){
    	/* define damping profile at half the grid points (half a grid point in -x)*/
        position_in_PML = (m->nab-i+0.5-1.0) *m->dh;
        position_norm = position_in_PML / (m->nab*m->dh);

	i1=i;

	m->K_z_half[i1] = 1.0;
        d_z_half[i1] = d0_z *(a*position_norm+b* pow(position_norm,npower)+c*pow(position_norm,(4)));

        if(position_in_PML < 0.0) {fprintf(stderr,"ERROR: Position in PML (y-boundary) <0");}
 
        /* this taken from Gedney page 8.2 */
        m->K_z_half[i1] = 1.0 + (k_max_CPML - 1.0) * pow(position_norm,npower);
        alpha_prime_z_half[i1] = alpha_max_PML * (1.0 - position_norm);

        if(alpha_prime_z_half[i1] < 0.0) {fprintf(stderr,"ERROR:alpha_prime_z_half[i] < 0.0, i %d", i);}
 
        m->b_z_half[i1] = exp(- (d_z_half[i1] / m->K_z_half[i1] + alpha_prime_z_half[i1]) * m->dt);

        if(fabs(d_z_half[i1]) > 1.0e-6){ m->a_z_half[i1] = d_z_half[i1] * (m->b_z_half[i1] - 1.0) / (m->K_z_half[i1] * (d_z_half[i1] + m->K_z_half[i1] * alpha_prime_z_half[i1]));}
	
        /* right boundary --> mirroring left boundary*/
        l = 2* m->nab -i-1;

	if(i>0){
	m->K_z[l+1] = m->K_z[i];
	m->b_z[l+1] = m->b_z[i];
	if(fabs(d_z[i]) > 1.0e-6){ m->a_z[l+1] = m->a_z[i]; }
	}
	
	
	m->K_z_half[l]=m->K_z_half[i];
        m->b_z_half[l] = m->b_z_half[i];  /*half a grid point in +x)*/
        if(fabs(d_z[i]) > 1.0e-6){ m->a_z_half[l] = m->a_z_half[i]; }
	}
}
    
    free(d_x);
    free(d_x_half);
    free(d_y);
    free(d_y_half);
    free(d_z);
    free(d_z_half);
    
    free(alpha_prime_x);
    free(alpha_prime_x_half);
    free(alpha_prime_y);
    free(alpha_prime_y_half);
    free(alpha_prime_z);
    free(alpha_prime_z_half);

   

}



