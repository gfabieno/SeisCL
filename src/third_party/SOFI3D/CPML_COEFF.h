
#ifndef CPML_COEFF_h
#define CPML_COEFF_h
#include <stdio.h>
#endif /* CPML_COEFF_h */

int CPML_coeff(float NPOWER,
               float k_max_CPML,
               float FPML,
               float VPPML,
               float dh,
               float dt,
               int NAB,
               float * K_i,
               float * b_i,
               float * a_i,
               float * K_i_half,
               float * b_i_half,
               float * a_i_half);
