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

/*Gradient calculation in the frequency domain */
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


#define gradrho(z,y,x) gradrho[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define gradM(z,y,x) gradM[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define gradmu(z,y,x) gradmu[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define gradtaup(z,y,x) gradtaup[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define gradtaus(z,y,x) gradtaus[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

#define Hrho(z,y,x) Hrho[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define HM(z,y,x) HM[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define Hmu(z,y,x) Hmu[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define Htaup(z,y,x) Htaup[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define Htaus(z,y,x) Htaus[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

#define pp(z,y,x) pp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define mp(z,y,x) mp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define up(z,y,x) up[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tpp(z,y,x) tpp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tsp(z,y,x) tsp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

#define mm(z,y,x) mm[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define um(z,y,x) um[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tpm(z,y,x) tpm[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tsm(z,y,x) tsm[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

#define uu(z,y,x) uu[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tpu(z,y,x) tpu[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tsu(z,y,x) tsu[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

#define tptp(z,y,x) tptp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tstp(z,y,x) tstp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

#define tsts(z,y,x) tsts[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

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

#define f_vx3(z,y,x,f)   f_vx[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_vy3(z,y,x,f)   f_vy[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_vz3(z,y,x,f)   f_vz[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxx3(z,y,x,f) f_sxx[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_syy3(z,y,x,f) f_syy[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_szz3(z,y,x,f) f_szz[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxy3(z,y,x,f) f_sxy[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_syz3(z,y,x,f) f_syz[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxz3(z,y,x,f) f_sxz[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]

#define f_rxx3(z,y,x,l,f) f_rxx[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_ryy3(z,y,x,l,f) f_ryy[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rzz3(z,y,x,l,f) f_rzz[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rxy3(z,y,x,l,f) f_rxy[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_ryz3(z,y,x,l,f) f_ryz[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rxz3(z,y,x,l,f) f_rxz[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]

#define f_vxr3(z,y,x,f)   f_vxr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_vyr3(z,y,x,f)   f_vyr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_vzr3(z,y,x,f)   f_vzr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxxr3(z,y,x,f) f_sxxr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_syyr3(z,y,x,f) f_syyr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_szzr3(z,y,x,f) f_szzr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxyr3(z,y,x,f) f_sxyr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_syzr3(z,y,x,f) f_syzr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxzr3(z,y,x,f) f_sxzr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]

#define f_rxxr3(z,y,x,l,f) f_rxxr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_ryyr3(z,y,x,l,f) f_ryyr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rzzr3(z,y,x,l,f) f_rzzr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rxyr3(z,y,x,l,f) f_rxyr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_ryzr3(z,y,x,l,f) f_ryzr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rxzr3(z,y,x,l,f) f_rxzr[(f)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NY+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((y)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]








#define f_vx2(z,x,f)   f_vx[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_vy2(z,x,f)   f_vy[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_vz2(z,x,f)   f_vz[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxx2(z,x,f) f_sxx[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_syy2(z,x,f) f_syy[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_szz2(z,x,f) f_szz[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxy2(z,x,f) f_sxy[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_syz2(z,x,f) f_syz[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxz2(z,x,f) f_sxz[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]

#define f_rxx2(z,x,l,f) f_rxx[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_ryy2(z,x,l,f) f_ryy[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rzz2(z,x,l,f) f_rzz[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rxy2(z,x,l,f) f_rxy[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_ryz2(z,x,l,f) f_ryz[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rxz2(z,x,l,f) f_rxz[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]

#define f_vxr2(z,x,f)   f_vxr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_vyr2(z,x,f)   f_vyr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_vzr2(z,x,f)   f_vzr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxxr2(z,x,f) f_sxxr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_syyr2(z,x,f) f_syyr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_szzr2(z,x,f) f_szzr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxyr2(z,x,f) f_sxyr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_syzr2(z,x,f) f_syzr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_sxzr2(z,x,f) f_sxzr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]

#define f_rxxr2(z,x,l,f) f_rxxr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_ryyr2(z,x,l,f) f_ryyr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rzzr2(z,x,l,f) f_rzzr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rxyr2(z,x,l,f) f_rxyr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_ryzr2(z,x,l,f) f_ryzr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]
#define f_rxzr2(z,x,l,f) f_rxzr[(f)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)*mglob->L+(l)*(m->NX+mglob->FDORDER)*(m->NZ+mglob->FDORDER)+((x)+mglob->fdoh)*(m->NZ+mglob->FDORDER)+((z)+mglob->fdoh)]

//Some functions to perfom complex operations with OpenCl vectors

static inline float
cl_itreal(cl_float2 a, cl_float2 b)
{
    float output =(a.y*b.x-a.x*b.y);
    return output;
}

static inline cl_float2
cl_add(cl_float2 a, cl_float2 b, cl_float2 c)
{
    cl_float2 output;
    output.x=a.x+b.x+c.x;
    output.y=a.y+b.y+c.y;
    return output;
}
static inline cl_float2
cl_diff(cl_float2 a, cl_float2 b, cl_float2 c)
{
    cl_float2 output;
    output.x=a.x-b.x-c.x;
    output.y=a.y-b.y-c.y;
    return output;
}

static inline cl_float2
cl_add2(cl_float2 a, cl_float2 b)
{
    cl_float2 output;
    output.x=a.x+b.x;
    output.y=a.y+b.y;
    return output;
}
static inline cl_float2
cl_diff2(cl_float2 a, cl_float2 b)
{
    cl_float2 output;
    output.x=a.x-b.x;
    output.y=a.y-b.y;
    return output;
}

static inline float
cl_rm(cl_float2 a,cl_float2 b, float tausig, float w)
{

    return tausig*(a.x*b.x+a.y*b.y)+(a.x*b.y-a.y*b.x)/w;
}

static inline cl_float2
cl_stat(cl_float2 a, float dt, float nf, float Nt)
{
    float fcos=cosf(2*PI*dt*nf/Nt);
    float fsin=sinf(2*PI*dt*nf/Nt);
    cl_float2 output;
    output.x=a.x*fcos-a.y*fsin;
    output.y=a.x*fsin+a.y*fcos;
    return output;
}
static inline cl_float2
cl_integral(cl_float2 a, float w)
{
    cl_float2 output;
    output.x=a.y/w;
    output.y=-a.x/w;
    return output;
}
static inline cl_float2
cl_derivative(cl_float2 a, float w)
{
    cl_float2 output;
    output.x=-a.y*w;
    output.y=a.x*w;
    return output;
}
static inline float
cl_norm(cl_float2 a)
{
    return pow(a.x,2)+pow(a.y,2);
}

// Coefficient of the scalar products
int grad_coefvisc_0(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    
    
    double fact1=pow(ND*M*(1.0+L*taup)*(1.0+al*taus)-2.0*(ND-1.0)*mu*(1.0+L*taus)*(1.0+al*taup),2.0);
    double fact2=pow(ND*M*taup*(1.0+al*taus)-2.0*(ND-1.0)*mu*taus*(1.0+al*taup),2.0);

    (*c)[0]= 2.0*sqrtf(rho*M)*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[1]= 2.0*sqrtf(rho*M)*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[2]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[3]= 2.0*sqrtf(rho*mu)*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[4]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[5]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( mu*mu*taus );
    (*c)[6]= 2.0*sqrtf(rho*mu)*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[7]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    (*c)[8]= M*(L-al)*pow(1+al*taus,2)/fact1;
    (*c)[9]= M*pow(1+al*taus,2)/fact2;
    (*c)[10]= (L-al)/( mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[11]= (ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2)/fact1;
    (*c)[12]= (L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[13]= 1.0/( mu*taus*taus );
    (*c)[14]= (ND+1.0)/3.0*mu*pow(1+al*taup,2)/fact2;
    (*c)[15]= 1.0/( 2*ND*mu*taus*taus );
    
    (*c)[16]= M/rho*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[17]= M/rho*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[18]= mu/rho*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[19]= mu/rho*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[20]= mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[21]= mu/rho*(1.0+al*taus)/( mu*mu*taus );
    (*c)[22]= mu/rho*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[23]= mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    


    return 1;
}
int grad_coefelast_0(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    (*c)[0]= 2.0*sqrtf(rho*M)*1.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    
    (*c)[2]= 2.0*sqrtf(rho*mu)*1.0/( mu*mu);
    (*c)[3]= 2.0*sqrtf(rho*mu)*(ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[4]= 2.0*sqrtf(rho*mu)*1.0/( 2*ND*mu*mu );
    
    (*c)[16]= 2.0*sqrtf(rho*M)*1.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    
    (*c)[18]= M/rho*1.0/( mu*mu);
    (*c)[19]= mu/rho*(ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[20]= mu/rho*1.0/( 2*ND*mu*mu );
    
    return 1;
}
int grad_coefvisc_1(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    
    
    double fact1=pow(ND*M*(1.0+L*taup)*(1.0+al*taus)-2.0*(ND-1.0)*mu*(1.0+L*taus)*(1.0+al*taup),2.0);
    double fact2=pow(ND*M*taup*(1.0+al*taus)-2.0*(ND-1.0)*mu*taus*(1.0+al*taup),2.0);
    
    (*c)[0]= (1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[1]= taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[2]= (1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[3]= (ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[4]= (1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[5]= (1.0+al*taus)/( mu*mu*taus );
    (*c)[6]= (ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[7]= (1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    (*c)[8]= M*(L-al)*pow(1+al*taus,2)/fact1;
    (*c)[9]= M*pow(1+al*taus,2)/fact2;
    (*c)[10]= (L-al)/( mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[11]= (ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2)/fact1;
    (*c)[12]= (L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[13]= 1.0/( mu*taus*taus );
    (*c)[14]= (ND+1.0)/3.0*mu*pow(1+al*taup,2)/fact2;
    (*c)[15]= 1.0/( 2*ND*mu*taus*taus );
    
    return 1;
}
int grad_coefelast_1(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    (*c)[0]= 1.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    
    (*c)[2]= 1.0/( mu*mu);
    (*c)[3]= (ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[4]= 1.0/( 2*ND*mu*mu );
    
    
    return 1;
}
int grad_coefvisc_2(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    
    
    double fact1=pow(ND*M*(1.0+L*taup)*(1.0+al*taus)-2.0*(ND-1.0)*mu*(1.0+L*taus)*(1.0+al*taup),2.0);
    double fact2=pow(ND*M*taup*(1.0+al*taus)-2.0*(ND-1.0)*mu*taus*(1.0+al*taup),2.0);
    
    (*c)[0]= 2.0*sqrtf(M/rho)*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[1]= 2.0*sqrtf(M/rho)*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[2]= 2.0*sqrtf(mu/rho)*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[3]= 2.0*sqrtf(mu/rho)*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[4]= 2.0*sqrtf(mu/rho)*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[5]= 2.0*sqrtf(mu/rho)*(1.0+al*taus)/( mu*mu*taus );
    (*c)[6]= 2.0*sqrtf(mu/rho)*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[7]= 2.0*sqrtf(mu/rho)*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    (*c)[8]= M*(L-al)*pow(1+al*taus,2)/fact1;
    (*c)[9]= M*pow(1+al*taus,2)/fact2;
    (*c)[10]= (L-al)/( mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[11]= (ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2)/fact1;
    (*c)[12]= (L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[13]= 1.0/( mu*taus*taus );
    (*c)[14]= (ND+1.0)/3.0*mu*pow(1+al*taup,2)/fact2;
    (*c)[15]= 1.0/( 2*ND*mu*taus*taus );
    
    (*c)[16]= -M/rho*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[17]= -M/rho*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[18]= -mu/rho*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[19]= -mu/rho*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[20]= -mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[21]= -mu/rho*(1.0+al*taus)/( mu*mu*taus );
    (*c)[22]= -mu/rho*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[23]= -mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    
    return 1;
}
int grad_coefelast_2(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    (*c)[0]= 2.0*sqrtf(M/rho)*1.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    
    (*c)[2]= 2.0*sqrtf(mu/rho)*1.0/( mu*mu);
    (*c)[3]= 2.0*sqrtf(mu/rho)*(ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[4]= 2.0*sqrtf(mu/rho)*1.0/( 2*ND*mu*mu );
    
    (*c)[18]= -M/rho*1.0/( mu*mu);
    (*c)[19]= -mu/rho*(ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[20]= -mu/rho*1.0/( 2*ND*mu*mu );
    
    return 1;
}
int grad_coefvisc_3(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    
    
    double fact1=pow(ND*M*(1.0+L*taup)*(1.0+al*taus)-2.0*(ND-1.0)*mu*(1.0+L*taus)*(1.0+al*taup),2.0);
    double fact2=pow(ND*M*taup*(1.0+al*taus)-2.0*(ND-1.0)*mu*taus*(1.0+al*taup),2.0);
    
    (*c)[0]= (2.0*sqrtf(rho*M)*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2) -taup/sqrtf(M/rho)*M*(L-al)*pow(1+al*taus,2))/fact1;
    (*c)[1]= (2.0*sqrtf(rho*M)*taup*(1.0+al*taup)*pow(1.0+al*taus,2) - taup/sqrtf(M/rho)*M*pow(1+al*taus,2) ) /fact2;
    (*c)[2]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/(mu*mu*(1.0+L*taus))  -  taup/sqrtf(M/rho)*(L-al)/( mu*(1.0+L*taus)*(1.0+L*taus))   ;
    (*c)[3]= (2.0*sqrtf(rho*mu)*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)-  taup/sqrtf(M/rho)*(ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2))/fact1 ;
    (*c)[4]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) )-taup/sqrtf(M/rho)*(L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[5]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( mu*mu*taus ) - taup/sqrtf(M/rho)*1.0/( mu*taus*taus ) ;
    (*c)[6]= (2.0*sqrtf(rho*mu)*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)  - taup/sqrtf(M/rho)*(ND+1.0)/3.0*mu*pow(1+al*taup,2) )/fact2;
    (*c)[7]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*taus ) - taup/sqrtf(M/rho)*1.0/( 2*ND*mu*taus*taus );
    (*c)[8]= (-2.0*sqrtf(rho*M)*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2) + (1.0+taup)/sqrtf(M/rho)*M*(L-al)*pow(1+al*taus,2))/fact1;
    (*c)[9]= (-2.0*sqrtf(rho*M)*taup*(1.0+al*taup)*pow(1.0+al*taus,2) + (1.0+taup)/sqrtf(M/rho)*M*pow(1+al*taus,2) ) /fact2;
    (*c)[10]= -2.0*sqrtf(rho*mu)*(1.0+al*taus)/(mu*mu*(1.0+L*taus))  + (1.0+taup)/sqrtf(M/rho)*(L-al)/( mu*(1.0+L*taus)*(1.0+L*taus))   ;
    (*c)[11]= (-2.0*sqrtf(rho*mu)*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)+ (1.0+taup)/sqrtf(M/rho)*(ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2))/fact1 ;
    (*c)[12]= -2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) )+ (1.0+taup)/sqrtf(M/rho)*(L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[13]= -2.0*sqrtf(rho*mu)*(1.0+al*taus)/( mu*mu*taus ) + (1.0+taup)/sqrtf(M/rho)*1.0/( mu*taus*taus ) ;
    (*c)[14]= (-2.0*sqrtf(rho*mu)*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)  + (1.0+taup)/sqrtf(M/rho)*(ND+1.0)/3.0*mu*pow(1+al*taup,2) )/fact2;
    (*c)[15]= -2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*taus ) + (1.0+taup)/sqrtf(M/rho)*1.0/( 2*ND*mu*taus*taus );
    
    (*c)[16]= M/rho*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[17]= M/rho*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[18]= mu/rho*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[19]= mu/rho*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[20]= mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[21]= mu/rho*(1.0+al*taus)/( mu*mu*taus );
    (*c)[22]= mu/rho*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[23]= mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    
    return 1;
}
int grad_coefvisc_0_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 2.0*sqrtf(rho*mu)*(1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[1]= 2.0*sqrtf(rho*mu)*(1+al*taus)/taus/pow(mu,2);
    (*c)[2]= (L-al)/pow(1+L*taus,2)/mu;
    (*c)[3]= 1/pow(taus,2)/mu;
    
    (*c)[4]= mu/rho*(1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[5]= mu/rho*(1+al*taus)/taus/pow(mu,2);


    
    return 1;
}
int grad_coefelast_0_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 2.0*sqrtf(rho*mu)/pow(mu,2);
    
    (*c)[4]= mu/rho/pow(mu,2);

    
    
    
    return 1;
}
int grad_coefvisc_1_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){

    
    (*c)[0]= (1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[1]= (1+al*taus)/taus/pow(mu,2);
    (*c)[2]= (L-al)/pow(1+L*taus,2)/mu;
    (*c)[3]= 1/pow(taus,2)/mu;

    return 1;
}
int grad_coefelast_1_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 1.0/pow(mu,2);

    
    return 1;
}
int grad_coefvisc_2_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 2.0*sqrtf(mu/rho)*(1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[1]= 2.0*sqrtf(mu/rho)*(1+al*taus)/taus/pow(mu,2);
    (*c)[2]= (L-al)/pow(1+L*taus,2)/mu;
    (*c)[3]= 1/pow(taus,2)/mu;
    
    (*c)[4]= -mu/rho*(1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[5]= -mu/rho*(1+al*taus)/taus/pow(mu,2);
    
    return 1;
}
int grad_coefelast_2_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 2.0*sqrtf(mu/rho)/pow(mu,2);
    
    (*c)[4]= -mu/rho*(1+al*taus)/(1+L*taus)/pow(mu,2);
    
    return 1;
}
int grad_coefvisc_3_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    //A faire

    
    return 1;
}
int grad_coefelast_3_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    //A faire
    
    
    return 1;
}


int calc_grad(struct modcsts* mglob, struct modcstsloc * m)  {
    
    int i,j,k,f,l;
    float df,freq,ND, al,w0;
    double c[24]={0}, dot[17]={0};
    float * tausigl=NULL;
    cl_float2 sxxyyzz, sxxyyzzr, sxx_myyzz, syy_mxxzz, szz_mxxyy;
    cl_float2 rxxyyzz, rxxyyzzr, rxx_myyzz, ryy_mxxzz, rzz_mxxyy;
    
    cl_float2 sxxzz, sxxzzr, sxx_mzz, szz_mxx;
    cl_float2 rxxzz, rxxzzr, rxx_mzz, rzz_mxx;
    cl_float2 one={1,1};
    
    int (*c_calc)(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al)=NULL;
    
    ND=(float)mglob->ND;
    df=1.0/mglob->NTnyq/mglob->dt/mglob->dtnyq;
    
    w0=2.0*PI*mglob->f0;
    al=0;
    
    if (mglob->L>0){
        tausigl=malloc(sizeof(float)*mglob->L);
        for (l=0;l<mglob->L;l++){
            tausigl[l]=1.0/(2.0*PI*mglob->FL[l]);
            al+=pow(w0/(2.0*PI*mglob->FL[l]),2)/(1.0+pow(w0/(2.0*PI*mglob->FL[l]),2));
        }
    }
    
    // Choose the right parameters depending on the dimensions
    if (mglob->ND!=21){
        if (mglob->param_type==0){
            if (mglob->L>0)
                c_calc=&grad_coefvisc_0;
            else
                c_calc=&grad_coefelast_0;
        }
        else if (mglob->param_type==1){
            if (mglob->L>0)
                c_calc=&grad_coefvisc_1;
            else
                c_calc=&grad_coefelast_1;
        }
        else if (mglob->param_type==2){
            if (mglob->L>0)
                c_calc=&grad_coefvisc_2;
            else
                c_calc=&grad_coefelast_2;
        }
        else if (mglob->param_type==3){
            c_calc=&grad_coefvisc_3;
            
        }
    }
    else if (mglob->ND==21){
        if (mglob->param_type==0){
            if (mglob->L>0)
                c_calc=&grad_coefvisc_0_SH;
            else
                c_calc=&grad_coefelast_0_SH;
        }
        else if (mglob->param_type==1){
            if (mglob->L>0)
                c_calc=&grad_coefvisc_1_SH;
            else
                c_calc=&grad_coefelast_1_SH;
        }
        else if (mglob->param_type==2){
            if (mglob->L>0)
                c_calc=&grad_coefvisc_2_SH;
            else
                c_calc=&grad_coefelast_2_SH;
        }
        else if (mglob->param_type==3){
            c_calc=&grad_coefvisc_3_SH;
            
        }
        
        
    }
    
    
    
    
    if (ND==3){
        
        for (i=0;i<m->NX;i++){
            for (j=0;j<m->NY;j++){
                for (k=0;k<m->NZ;k++){
                    for (f=0;f<mglob->nfreqs;f++){
                        
                        freq=2.0*PI*df* (float)mglob->gradfreqsn[f];
                        
                        if (mglob->L>0)
                            c_calc(&c,m->pi(k,j,i), m->u(k,j,i), m->taup(k,j,i), m->taus(k,j,i), m->rho(k,j,i), ND,mglob->L,al);
                        else
                            c_calc(&c,m->pi(k,j,i), m->u(k,j,i), 0, 0, m->rho(k,j,i), ND,mglob->L,al);

                        dot[1]=0;dot[5]=0;dot[6]=0;dot[7]=0;
                        for (l=0;l<mglob->L;l++){
                            m->f_sxx3(k,j,i,f)=cl_diff2(m->f_sxx3(k,j,i,f), cl_integral(m->f_rxx3(k,j,i,f,l),freq));
                            m->f_szz3(k,j,i,f)=cl_diff2(m->f_szz3(k,j,i,f), cl_integral(m->f_rzz3(k,j,i,f,l),freq));
                            m->f_syy3(k,j,i,f)=cl_diff2(m->f_syy3(k,j,i,f), cl_integral(m->f_ryy3(k,j,i,f,l),freq));
                            m->f_sxz3(k,j,i,f)=cl_diff2(m->f_sxz3(k,j,i,f), cl_integral(m->f_rxz3(k,j,i,f,l),freq));
                            m->f_sxy3(k,j,i,f)=cl_diff2(m->f_sxy3(k,j,i,f), cl_integral(m->f_rxy3(k,j,i,f,l),freq));
                            m->f_syz3(k,j,i,f)=cl_diff2(m->f_syz3(k,j,i,f), cl_integral(m->f_ryz3(k,j,i,f,l),freq));
                            
                            m->f_sxxr3(k,j,i,f)=cl_diff2(m->f_sxxr3(k,j,i,f), cl_integral(m->f_rxxr3(k,j,i,f,l),freq));
                            m->f_szzr3(k,j,i,f)=cl_diff2(m->f_szzr3(k,j,i,f), cl_integral(m->f_rzzr3(k,j,i,f,l),freq));
                            m->f_syyr3(k,j,i,f)=cl_diff2(m->f_syyr3(k,j,i,f), cl_integral(m->f_ryyr3(k,j,i,f,l),freq));
                            m->f_sxzr3(k,j,i,f)=cl_diff2(m->f_sxzr3(k,j,i,f), cl_integral(m->f_rxzr3(k,j,i,f,l),freq));
                            m->f_sxyr3(k,j,i,f)=cl_diff2(m->f_sxyr3(k,j,i,f), cl_integral(m->f_rxyr3(k,j,i,f,l),freq));
                            m->f_syzr3(k,j,i,f)=cl_diff2(m->f_syzr3(k,j,i,f), cl_integral(m->f_ryz3(k,j,i,f,l),freq));
                            
                            
                            rxxyyzz=    cl_add(m->f_rxx3(k,j,i,f,l), m->f_ryy3(k,j,i,f,l), m->f_rzz3(k,j,i,f,l));
                            rxxyyzzr=   cl_add(m->f_rxxr3(k,j,i,f,l), m->f_ryyr3(k,j,i,f,l), m->f_rzzr3(k,j,i,f,l));
                            rxx_myyzz= cl_diff(m->f_rxx3(k,j,i,f,l), m->f_ryy3(k,j,i,f,l), m->f_rzz3(k,j,i,f,l));
                            ryy_mxxzz= cl_diff(m->f_rxx3(k,j,i,f,l), m->f_ryy3(k,j,i,f,l), m->f_rzz3(k,j,i,f,l));
                            rzz_mxxyy= cl_diff(m->f_rxx3(k,j,i,f,l), m->f_ryy3(k,j,i,f,l), m->f_rzz3(k,j,i,f,l));
                            dot[1]+=cl_rm( rxxyyzzr, rxxyyzz, tausigl[l],freq )/mglob->NTnyq;
                            
                            dot[5]+=(+cl_rm( m->f_rxyr3(k,j,i,f,l), m->f_rxy3(k,j,i,f,l) , tausigl[l],freq)
                                     +cl_rm( m->f_rxzr3(k,j,i,f,l), m->f_rxz3(k,j,i,f,l) , tausigl[l],freq)
                                     +cl_rm( m->f_ryzr3(k,j,i,f,l), m->f_ryz3(k,j,i,f,l) , tausigl[l],freq))/mglob->NTnyq;
                            dot[6]=dot[1];
                            dot[7]+=(+cl_rm( m->f_rxxr3(k,j,i,f,l), rxx_myyzz , tausigl[l],freq)
                                     +cl_rm( m->f_ryyr3(k,j,i,f,l), ryy_mxxzz , tausigl[l],freq)
                                     +cl_rm( m->f_rzzr3(k,j,i,f,l), rzz_mxxyy , tausigl[l],freq))/mglob->NTnyq;
                        }
                        
                        sxxyyzz=    cl_add(m->f_sxx3(k,j,i,f), m->f_syy3(k,j,i,f), m->f_szz3(k,j,i,f));
                        sxxyyzzr=   cl_add(m->f_sxxr3(k,j,i,f),m->f_syyr3(k,j,i,f),m->f_szzr3(k,j,i,f));
                        sxx_myyzz= cl_diff(m->f_sxx3(k,j,i,f), m->f_syy3(k,j,i,f), m->f_szz3(k,j,i,f));
                        syy_mxxzz= cl_diff(m->f_syy3(k,j,i,f), m->f_sxx3(k,j,i,f), m->f_szz3(k,j,i,f));
                        szz_mxxyy= cl_diff(m->f_szz3(k,j,i,f), m->f_sxx3(k,j,i,f), m->f_syy3(k,j,i,f));

                        dot[0]=freq*cl_itreal( sxxyyzzr, sxxyyzz )/mglob->NTnyq;
                        dot[2]=freq*(+cl_itreal( m->f_sxyr3(k,j,i,f), m->f_sxy3(k,j,i,f) )
                                     +cl_itreal( m->f_sxzr3(k,j,i,f), m->f_sxz3(k,j,i,f) )
                                     +cl_itreal( m->f_syzr3(k,j,i,f), m->f_syz3(k,j,i,f) ))/mglob->NTnyq;
                        dot[3]=dot[0];
                        dot[4]=freq*(+cl_itreal( m->f_sxxr3(k,j,i,f), sxx_myyzz )
                                     +cl_itreal( m->f_syyr3(k,j,i,f), syy_mxxzz )
                                     +cl_itreal( m->f_szzr3(k,j,i,f), szz_mxxyy ))/mglob->NTnyq;

                        
                        dot[8]=freq*(
                                     cl_itreal( m->f_vxr3(k,j,i,f), m->f_vx3(k,j,i,f) ) +
                                     cl_itreal( m->f_vyr3(k,j,i,f), m->f_vy3(k,j,i,f) ) +
                                     cl_itreal( m->f_vzr3(k,j,i,f), m->f_vz3(k,j,i,f) )
                                     )/mglob->NTnyq;
                        
                        m->gradM(k,j,i)+=c[0]*dot[0]-c[1]*dot[1];
                        m->gradmu(k,j,i)+=c[2]*dot[2]-c[3]*dot[3]+c[4]*dot[4]-c[5]*dot[5]+c[6]*dot[6]-c[7]*dot[7];
                        
                        if (mglob->L>0){
                            m->gradtaup(k,j,i)+=c[8]*dot[0]-c[9]*dot[1];
                            m->gradtaus(k,j,i)+=c[10]*dot[2]-c[11]*dot[3]+c[12]*dot[4]-c[13]*dot[5]+c[14]*dot[6]-c[15]*dot[7];
                        }
                        
                        m->gradrho(k,j,i)+=dot[8] +c[16]*dot[0]-c[17]*dot[1]  +c[18]*dot[2]-c[19]*dot[3]+c[20]*dot[4]-c[21]*dot[5]+c[22]*dot[6]-c[23]*dot[7];

                    }
                    
                }
            }
        }
        
    }
    else if (ND==2){
        
        for (i=0;i<m->NX;i++){
            for (k=0;k<m->NZ;k++){
                for (f=0;f<mglob->nfreqs;f++){
                    
                    freq=2.0*PI*df* (float)mglob->gradfreqsn[f];
                    
                    if (mglob->L>0)
                        c_calc(&c,m->pi(k,0,i), m->u(k,0,i), m->taup(k,0,i), m->taus(k,0,i), m->rho(k,0,i), ND,mglob->L,al);
                    else
                        c_calc(&c,m->pi(k,0,i), m->u(k,0,i), 0, 0, m->rho(k,0,i), ND,mglob->L,al);
                    
                    dot[1]=0;dot[5]=0;dot[6]=0;dot[7]=0;
                    for (l=0;l<mglob->L;l++){
                        m->f_sxx2(k,i,f)=cl_diff2(m->f_sxx2(k,i,f), cl_integral(m->f_rxx2(k,i,f,l),freq) );
                        m->f_szz2(k,i,f)=cl_diff2(m->f_szz2(k,i,f), cl_integral(m->f_rzz2(k,i,f,l),freq) );
                        m->f_sxz2(k,i,f)=cl_diff2(m->f_sxz2(k,i,f), cl_integral(m->f_rxz2(k,i,f,l),freq) );
                        m->f_sxxr2(k,i,f)=cl_diff2(m->f_sxxr2(k,i,f), cl_integral(m->f_rxxr2(k,i,f,l),freq) );
                        m->f_szzr2(k,i,f)=cl_diff2(m->f_szzr2(k,i,f), cl_integral(m->f_rzzr2(k,i,f,l),freq) );
                        m->f_sxzr2(k,i,f)=cl_diff2(m->f_sxzr2(k,i,f), cl_integral(m->f_rxzr2(k,i,f,l),freq) );
                        
                        rxxzz=    cl_add2(m->f_rxx2(k,i,f,l), m->f_rzz2(k,i,f,l));
                        rxxzzr=   cl_add2(m->f_rxxr2(k,i,f,l),m->f_rzzr2(k,i,f,l));
                        rxx_mzz= cl_diff2(m->f_rxx2(k,i,f,l), m->f_rzz2(k,i,f,l));
                        rzz_mxx= cl_diff2(m->f_rzz2(k,i,f,l), m->f_rxx2(k,i,f,l));
                        
                        dot[1]+=cl_rm( rxxzzr, rxxzz, tausigl[l],freq )/mglob->NTnyq;
                        
                        dot[5]+=(cl_rm( m->f_rxzr2(k,i,f,l), m->f_rxz2(k,i,f,l) , tausigl[l],freq) )/mglob->NTnyq;
                        dot[6]=dot[1];
                        dot[7]+=(+cl_rm( m->f_rxxr2(k,i,f,l), rxx_mzz , tausigl[l],freq)
                                 +cl_rm( m->f_rzzr2(k,i,f,l), rzz_mxx , tausigl[l],freq))/mglob->NTnyq;
                        
                    }
                    sxxzz=    cl_add2(m->f_sxx2(k,i,f), m->f_szz2(k,i,f));
                    sxxzzr=   cl_add2(m->f_sxxr2(k,i,f),m->f_szzr2(k,i,f));
                    sxx_mzz= cl_diff2(m->f_sxx2(k,i,f), m->f_szz2(k,i,f));
                    szz_mxx= cl_diff2(m->f_szz2(k,i,f), m->f_sxx2(k,i,f));
                    

                    
                    dot[0]=freq*cl_itreal( sxxzzr, sxxzz )/mglob->NTnyq;
                    dot[2]=freq* ( cl_itreal( m->f_sxzr2(k,i,f), m->f_sxz2(k,i,f))  )/mglob->NTnyq;
                    dot[3]=dot[0];
                    dot[4]=freq*(+cl_itreal( m->f_sxxr2(k,i,f), sxx_mzz )
                                 +cl_itreal( m->f_szzr2(k,i,f), szz_mxx ))/mglob->NTnyq;

                    dot[8]=freq*(cl_itreal( m->f_vxr2(k,i,f), m->f_vx2(k,i,f) ) + cl_itreal( m->f_vzr2(k,i,f), m->f_vz2(k,i,f) ))/mglob->NTnyq;
                    
                    
                    m->gradM(k,0,i)+=c[0]*dot[0]-c[1]*dot[1];
                    m->gradmu(k,0,i)+=c[2]*dot[2]-c[3]*dot[3]+c[4]*dot[4]-c[5]*dot[5]+c[6]*dot[6]-c[7]*dot[7];
                    
                    if (mglob->L>0){
                        m->gradtaup(k,0,i)+=c[8]*dot[0]-c[9]*dot[1];
                        m->gradtaus(k,0,i)+=c[10]*dot[2]-c[11]*dot[3]+c[12]*dot[4]-c[13]*dot[5]+c[14]*dot[6]-c[15]*dot[7];
                    }
                    
                    m->gradrho(k,0,i)+=dot[8] +c[16]*dot[0]-c[17]*dot[1]+c[18]*dot[2]-c[19]*dot[3]+c[20]*dot[4]-c[21]*dot[5]+c[22]*dot[6]-c[23]*dot[7];
                    
                    if(mglob->Hout){
                        dot[1]=0;dot[5]=0;dot[6]=0;dot[7]=0;
                        for (l=0;l<mglob->L;l++){
                            rxxzz=    cl_add2(m->f_rxx2(k,i,f,l), m->f_rzz2(k,i,f,l));
                            rxx_mzz= cl_diff2(m->f_rxx2(k,i,f,l), m->f_rzz2(k,i,f,l));
                            rzz_mxx= cl_diff2(m->f_rzz2(k,i,f,l), m->f_rxx2(k,i,f,l));
                            
                            dot[1]+=cl_norm(cl_add2( rxxzz, cl_derivative(rxxzz, freq*tausigl[l])) )/mglob->NTnyq;
                            dot[5]+=cl_norm(cl_add2( m->f_rxz2(k,i,f,l), cl_derivative(m->f_rxz2(k,i,f,l), freq*tausigl[l])) )/mglob->NTnyq;
                            dot[6]=dot[1];
                            dot[7]+=(cl_norm(cl_add2( rxx_mzz, cl_derivative(rxx_mzz, freq*tausigl[l])) )
                                    +cl_norm(cl_add2( rzz_mxx, cl_derivative(rzz_mxx, freq*tausigl[l])) ))/mglob->NTnyq;
                            
                        }
                        sxxzz=    cl_add2(m->f_sxx2(k,i,f), m->f_szz2(k,i,f));
                        sxx_mzz= cl_diff2(m->f_sxx2(k,i,f), m->f_szz2(k,i,f));
                        szz_mxx= cl_diff2(m->f_szz2(k,i,f), m->f_sxx2(k,i,f));
                        
                        
                        dot[0]=cl_norm(cl_derivative(sxxzz, freq))/mglob->NTnyq;
                        dot[2]=cl_norm(cl_derivative(m->f_sxz2(k,i,f), freq))/mglob->NTnyq;
                        dot[3]=dot[0];
                        dot[4]=(cl_norm(cl_derivative(sxx_mzz, freq))
                                    +cl_norm(cl_derivative(szz_mxx, freq)))/mglob->NTnyq;
                        dot[8]=(cl_norm(cl_derivative(m->f_vx2(k,i,f), freq))
                                +cl_norm(cl_derivative(m->f_vz2(k,i,f), freq)))/mglob->NTnyq;
                        
                        m->HM(k,0,i)+=c[0]*dot[0]-c[1]*dot[1];
                        m->Hmu(k,0,i)+=c[2]*dot[2]-c[3]*dot[3]+c[4]*dot[4]-c[5]*dot[5]+c[6]*dot[6]-c[7]*dot[7];
                        
                        if (mglob->L>0){
                            m->Htaup(k,0,i)+=c[8]*dot[0]-c[9]*dot[1];
                            m->Htaus(k,0,i)+=c[10]*dot[2]-c[11]*dot[3]+c[12]*dot[4]-c[13]*dot[5]+c[14]*dot[6]-c[15]*dot[7];
                        }
                        
                        m->Hrho(k,0,i)+=dot[8] +c[16]*dot[0]-c[17]*dot[1]+c[18]*dot[2]-c[19]*dot[3]+c[20]*dot[4]-c[21]*dot[5]+c[22]*dot[6]-c[23]*dot[7];
                        
                    }
                    
                    
                }
            }
        }
        
    }
    else if (ND==21){
        
        for (i=0;i<m->NX;i++){
            for (k=0;k<m->NZ;k++){
                for (f=0;f<mglob->nfreqs;f++){
                    
                    freq=2.0*PI*df* (float)mglob->gradfreqsn[f];
                    
                    if (mglob->L>0)
                        c_calc(&c,0, m->u(k,0,i), 0, m->taus(k,0,i), m->rho(k,0,i), ND, mglob->L, al);
                    else
                        c_calc(&c,0, m->u(k,0,i), 0, 0, m->rho(k,0,i), ND, mglob->L, al);
                    
                    
                    dot[0]=freq*(cl_itreal(m->f_sxyr2(k,i,f),m->f_sxy2(k,i,f))+ cl_itreal(m->f_syzr2(k,i,f),m->f_syz2(k,i,f)) )/mglob->NTnyq;

                    for (l=0;l<mglob->L;l++){
                        dot[1]=(cl_rm( m->f_rxyr2(k,i,l,f), m->f_rxy2(k,i,l,f),tausigl[l],freq )+cl_rm( m->f_ryzr2(k,i,l,f), m->f_ryz2(k,i,l,f),tausigl[l],freq ))/mglob->NTnyq;
                    }
                    
                    dot[2]=freq*(cl_itreal( m->f_vyr2(k,i,f), m->f_vy2(k,i,f) ))/mglob->NTnyq;
                    

                    m->gradmu(k,0,i)+=c[0]*dot[0]-c[1]*dot[1];
                    
                    if (mglob->L>0){
                        m->gradtaus(k,0,i)+=c[2]*dot[0]-c[3]*dot[1];
                    }
                    
                    m->gradrho(k,0,i)+=dot[2] +c[4]*dot[0]-c[5]*dot[1]  ;
                    
                }
            }
        }
        
    }
    
        
    if (tausigl) free(tausigl);
    return 0;
    
}

