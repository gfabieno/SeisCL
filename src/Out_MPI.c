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


/* Write output files */

#include "F.h"

#define gradrho(z,y,x) gradrho[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define gradM(z,y,x) gradM[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define gradmu(z,y,x) gradmu[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define gradtaup(z,y,x) gradtaup[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define gradtaus(z,y,x) gradtaus[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]


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

#define rho(z,y,x) rho[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define rip(z,y,x) rip[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define rjp(z,y,x) rjp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define rkp(z,y,x) rkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define uipjp(z,y,x) uipjp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define ujpkp(z,y,x) ujpkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define uipkp(z,y,x) uipkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define u(z,y,x) u[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define pi(z,y,x) pi[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

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

int cmpfunc (const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}



int Out_MPI(struct filenames file, struct modcsts * m)  {

    int state=0;

    // Gather the seismograms from all processing elements
    if (!state && m->SEISOUT){
        if (m->MYID==0){
            if (m->bcastvx)  MPI_Reduce(MPI_IN_PLACE, m->vxout[0],  m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastvy)  MPI_Reduce(MPI_IN_PLACE, m->vyout[0],  m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastvz)  MPI_Reduce(MPI_IN_PLACE, m->vzout[0],  m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsxx) MPI_Reduce(MPI_IN_PLACE, m->sxxout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsyy) MPI_Reduce(MPI_IN_PLACE, m->syyout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastszz) MPI_Reduce(MPI_IN_PLACE, m->szzout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsxy) MPI_Reduce(MPI_IN_PLACE, m->sxyout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsxz) MPI_Reduce(MPI_IN_PLACE, m->sxzout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsyz) MPI_Reduce(MPI_IN_PLACE, m->syzout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastp)   MPI_Reduce(MPI_IN_PLACE, m->pout[0],   m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            
        }
        else{
            if (m->bcastvx)  MPI_Reduce(m->vxout[0],  m->vxout[0],  m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastvy)  MPI_Reduce(m->vyout[0],  m->vyout[0],  m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastvz)  MPI_Reduce(m->vzout[0],  m->vzout[0],  m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsxx) MPI_Reduce(m->sxxout[0], m->sxxout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsyy) MPI_Reduce(m->syyout[0], m->syyout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastszz) MPI_Reduce(m->szzout[0], m->szzout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsxy) MPI_Reduce(m->sxyout[0], m->sxyout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsxz) MPI_Reduce(m->sxzout[0], m->sxzout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastsyz) MPI_Reduce(m->syzout[0], m->syzout[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastp)   MPI_Reduce(m->pout[0],   m->pout[0],   m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }
    // Add the rms value of all processing elements
    if (m->RMSOUT){
        if (m->MYID==0){
            if (!state) MPI_Reduce(MPI_IN_PLACE, &m->rms, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!state) MPI_Reduce(MPI_IN_PLACE, &m->rmsnorm, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else{
            if (!state) MPI_Reduce(&m->rms, &m->rms, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!state) MPI_Reduce(&m->rmsnorm, &m->rmsnorm, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }
    
    // Gather the residuals from all processing elements
    if (m->RESOUT){
        
        if (m->MYID==0){
            if (m->bcastvx) if (!state) MPI_Reduce(MPI_IN_PLACE, m->rx[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastvy) if (!state) MPI_Reduce(MPI_IN_PLACE, m->ry[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastvz) if (!state) MPI_Reduce(MPI_IN_PLACE, m->rz[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else{
            if (m->bcastvx) if (!state) MPI_Reduce(m->rx[0], m->rx[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastvy) if (!state) MPI_Reduce(m->ry[0], m->ry[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->bcastvz) if (!state) MPI_Reduce(m->rz[0], m->rz[0], m->allng*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        }

    }

    // Gather the movie from all processing elements
    if (m->MOVOUT){
        
        if (m->MYID==0){
            if (m->ND!=21){
                if (!state) MPI_Reduce(MPI_IN_PLACE, m->movvx, m->ns*m->NT/m->MOVOUT*m->NX*m->NY*m->NZ, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
                if (!state) MPI_Reduce(MPI_IN_PLACE, m->movvz, m->ns*m->NT/m->MOVOUT*m->NX*m->NY*m->NZ, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
            if (m->ND==3 || m->ND==21){
                if (!state) MPI_Reduce(MPI_IN_PLACE, m->movvy, m->ns*m->NT/m->MOVOUT*m->NX*m->NY*m->NZ, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }
        else{
            if (m->ND!=21){
                if (!state) MPI_Reduce(m->movvx, m->movvx, m->ns*m->NT/m->MOVOUT*m->NX*m->NY*m->NZ, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
                if (!state) MPI_Reduce(m->movvz, m->movvz, m->ns*m->NT/m->MOVOUT*m->NX*m->NY*m->NZ, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
            if (m->ND==3 || m->ND==21){
                if (!state) MPI_Reduce(m->movvy, m->movvy, m->ns*m->NT/m->MOVOUT*m->NX*m->NY*m->NZ, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }
        
    }
    
    // Gather the gradient from all processing elements
    if (m->GRADOUT==1){
        

        if (m->MYID==0){
            if (!state) MPI_Reduce(MPI_IN_PLACE, m->gradrho, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->ND!=21)
                if (!state) MPI_Reduce(MPI_IN_PLACE, m->gradM,   m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!state) MPI_Reduce(MPI_IN_PLACE, m->gradmu,  m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->L>0){
                if (m->ND!=21)
                    if (!state) MPI_Reduce(MPI_IN_PLACE, m->gradtaup, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (!state) MPI_Reduce(MPI_IN_PLACE, m->gradtaus, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }
        else{
            if (!state) MPI_Reduce(m->gradrho, m->gradrho, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->ND!=21)
                if (!state) MPI_Reduce(m->gradM,   m->gradM, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!state) MPI_Reduce(m->gradmu,  m->gradmu, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (m->L>0){
                if (m->ND!=21)
                    if (!state) MPI_Reduce(m->gradtaup, m->gradtaup, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (!state) MPI_Reduce(m->gradtaus, m->gradtaus, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }
        
        if (m->GRADSRCOUT==1){
            if (m->MYID==0){
                if (!state) MPI_Reduce(MPI_IN_PLACE, m->gradsrc[0], m->allns*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
            else{
                if (!state) MPI_Reduce(m->gradsrc, m->gradsrc[0], m->allns*m->NT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }
        
        if (m->HOUT==1){
            
            
            if (m->MYID==0){
                if (!state) MPI_Reduce(MPI_IN_PLACE, m->Hrho, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (m->ND!=21)
                if (!state) MPI_Reduce(MPI_IN_PLACE, m->HM,   m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (!state) MPI_Reduce(MPI_IN_PLACE, m->Hmu,  m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (m->L>0){
                    if (m->ND!=21)
                    if (!state) MPI_Reduce(MPI_IN_PLACE, m->Htaup, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    if (!state) MPI_Reduce(MPI_IN_PLACE, m->Htaus, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                }
            }
            else{
                if (!state) MPI_Reduce(m->gradrho, m->Hrho, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (m->ND!=21)
                if (!state) MPI_Reduce(m->gradM,   m->HM, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (!state) MPI_Reduce(m->gradmu,  m->Hmu, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (m->L>0){
                    if (m->ND!=21)
                    if (!state) MPI_Reduce(m->gradtaup, m->Htaup, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    if (!state) MPI_Reduce(m->gradtaus, m->Htaus, m->NX*m->NY*m->NZ, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                }
            }
            
        }
        
    }

    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Write the ouput to hdf5 files
    if (m->MYID==0){
        if (!state) state=writehdf5(file, m) ;
    }

    
    return state;

}
