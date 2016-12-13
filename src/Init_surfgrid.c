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

// Interpolation subroutines, NOT WORKING
#include "F.h"
#define var(z,y,x) var[(x)*NY*NZ+(y)*NZ+(z)]


float sinc(const float x)
{
    if (x==0)
        return 1;
    return sin(x)/x;
}

void intertrilin(int NXs, int NYs, int NZs, int Nr,int NX, int NY, int NZ, float * var, float * vars){
    int i,j,k, x,y,z;
    int ind;
    float c00,c01,c10,c11,c0,c1;
    float dx,dy,dz;
    
    for (i=0;i<NXs;i++){
        for (j=0;j<NYs;j++){
            for (k=1;NZs;k++){
                x=j/pow(2,Nr)-1;
                y=j/pow(2,Nr)-1;
                z=(k-1)/pow(2,Nr)-1;
                
                dx= (i % (int)(pow(2,Nr))) / (pow(2,Nr));
                dy= (j % (int)(pow(2,Nr))) / (pow(2,Nr));
                dz= ( (k-1) % (int)(pow(2,Nr)) )/ (pow(2,Nr)) ;
                
                if (x<0) {x=0;dx=0;}
                if (x>NX-1) {x=NX-1;dx=0;}
                if (y<0) {y=0;dy=0;}
                if (y>NY-1) {y=NY-1;dy=0;}
                if (z<0) {z=0;dz=0;}
                if (z>NZ-1) {z=NZ-1;dz=0;}
                
                ind=i*NYs*NZs+j*NZs+k;
                
                c00=var(z,y,x)*(1-dx)+var(z,y,x+1)*dx;
                c01=var(z+1,y,x)*(1-dx)+var(z+1,y,x+1)*dx;
                c10=var(z,y+1,x)*(1-dx)+var(z,y+1,x+1)*dx;
                c11=var(z+1,y+1,x)*(1-dx)+var(z+1,y+1,x+1)*dx;
                
                c0=c00*(1-dy)+c10*dy;
                c1=c01*(1-dy)+c11*dy;
                
                vars[ind]=c0*(1-dz)+c1*dz;
                
                
            }
        }
    }
}

void initprop(int NXs, int NYs, int NZs, float * var, float  value){
    int i,j, ind;

    
    for (i=0;i<NXs;i++){
        for (j=0;j<NYs;j++){
                ind=i*NYs*NZs+j*NZs;
                var[ind]=value;
        }
    }
}

int Init_surfgrid(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, struct modcsts * m_s, struct varcl ** vcl_s, struct modcstsloc ** mloc_s)  {
    
    int state=0;
    int d,i,j,t;
    int indmax, indmin, ind;
    float m0, m1, p0, p1, h00, h10, h01, h11, dt;
    
    m_s->NX=pow(2,m->Nr)*(m->NX+1)-1;
    if (m->NY>1){
    m_s->NY=pow(2,m->Nr)*(m->NY+1)-1;
    }
    else{
        m_s->NY=1;
    }
    m_s->NZ=(m->topowidth+m->fdoh+2)*pow(2,m->Nr)+1;
    m_s->NT=m->NT*pow(2,m->Nr);
    m_s->FDORDER=2;
    m_s->nab=m->nab*pow(2,m->Nr);
    m_s->MAXRELERROR=m->MAXRELERROR;
    m_s->gradout=m->gradout;
    m_s->ns=m->ns;
    m_s->L=m->L;
    m_s->TAU=m->TAU;
    m_s->dt=m->dt/pow(2,m->Nr);
    m_s->dh=m->dh/pow(2,m->Nr);
    m_s->abpc=m_s->abpc;
    m_s->pref_device_type=m->pref_device_type;
    m_s->nmax_dev=m->nmax_dev;
    m_s->n_no_use_GPUs=m->n_no_use_GPUs;
    m_s->MPI_NPROC_SHOT=m->MPI_NPROC_SHOT;

    
    m_s->freesurf=m->freesurf;
    m_s->ND=m->ND;
    m_s->abs_type=m->abs_type;
    m_s->VPPML=m->VPPML;
    m_s->FPML=m->FPML;
    m_s->NPOWER=m->NPOWER;
    m_s->K_MAX_CPML=m->K_MAX_CPML;
    m_s->f0=m->f0;
    m_s->allng=m->allng;
    m_s->allns=m->allns;
    m_s->bcastvx=m->bcastvx;
    m_s->bcastvy=m->bcastvy;
    m_s->bcastvz=m->bcastvz;
    m_s->Nr=m->Nr;
    
    m_s->MPI_NPROC_SHOT=m->MPI_NPROC_SHOT;
    m_s->MYGROUPID=m->MYGROUPID;
    m_s->NLOCALP=m->NLOCALP;
    m_s->MYLOCALID=m->MYLOCALID;
    
    m_s->no_use_GPUs=m->no_use_GPUs;
    m_s->nsrc=m->nsrc;
    m_s->src_pos=m->src_pos;
    m_s->nrec=m->nrec;
    m_s->rec_pos=m->rec_pos;
    
    GMALLOC(m_s->src, sizeof(float*)*m_s->ns)
    GMALLOC(m_s->src[0], sizeof(float)*m_s->allns*m_s->NT)
    for (i=1;i<m_s->ns;i++){
        m_s->src[i]=m_s->src[i-1]+m_s->nsrc[i-1]*m_s->NT;
    }
//    for (i=0;i<m_s->ns;i++){
//        for (j=0;j<m_s->nsrc[i];j++){
//            for (t=0;t<m_s->NT;t++){
//                dt=(t%(int)pow(2,m->Nr))/pow(2,m->Nr);
//                m_s->src[i][j*m_s->NT+t]=m->src[i][j*m->NT+t/(int)pow(2,m->Nr)]*(1.0-dt)+m->src[i][j*m->NT+t/(int)pow(2,m->Nr)+1]*dt;
////                indmin=t/(int)pow(2,m->Nr)-20;
////                if (indmin<0)
////                    indmin=0;
////                indmax=t/(int)pow(2,m->Nr)+20;
////                if (indmax>m->NT-1)
////                    indmin=m->NT-1;
////                for (ind=indmin;ind<indmax;ind++){
////                    m_s->src[i][j*m_s->NT+t]+=m->src[i][j*m->NT+ind]*sinc(PI*((float)t*m_s->dt/m->dt-(float)ind));
//////                    printf("%d %f\n",ind, sinc(PI*((float)t*m_s->dt/m->dt-(float)ind)));
////
////                }
//            }
//        }
//    }
    
    // Cubic interpolation of the source
    for (i=0;i<m_s->ns;i++){
        for (j=0;j<m_s->nsrc[i];j++){
            for (t=0;t<m->NT;t++){
                p0=m->src[i][j*m->NT+t];
                p1=0.0;
                if (t<m->NT-1)
                    p1=m->src[i][j*m->NT+t+1];
                m0=0;
                if (t<m->NT-1 && t>0)
                    m0=(m->src[i][j*m->NT+t+1]-m->src[i][j*m->NT+t-1])/2.0;
                m1=0;
                if (t<m->NT-2)
                    m1=(m->src[i][j*m->NT+t+2]-m->src[i][j*m->NT+t])/2.0;
                for (ind=0;ind<pow(2,m->Nr);ind++){
                    dt=1.0/pow(2,m->Nr)*ind;
                    h00=2*pow(dt,3)-3*pow(dt,2)+1;
                    h10=pow(dt,3)-2*pow(dt,2)+dt;
                    h01=-2*pow(dt,3)+3*pow(dt,2);
                    h11=pow(dt,3)-pow(dt,2);
                    m_s->src[i][j*m_s->NT+t*(int)pow(2,m->Nr)+ind]=h00*p0+h10*m0+h01*p1+h11*m1;
                }
            }
        }
    }

    
    m_s->FL=m->FL;
    m_s->topo=m->topo;
    m_s->vxout=m->vxout;
    m_s->vyout=m->vyout;
    m_s->vzout=m->vzout;
    m_s->movvx=m->movvx;
    m_s->movvy=m->movvy;
    m_s->movvz=m->movvz;
    m_s->rx=m->rx;
    m_s->ry=m->ry;
    m_s->rz=m->rz;
    m_s->vx0=m->vx0;
    m_s->vy0=m->vy0;
    m_s->vz0=m->vz0;
    
    m_s->back_prop_type=m->back_prop_type;
    m_s->nfreqs=m->nfreqs;
    m_s->gradfreqs=m->gradfreqs;
    
    m_s->seisout=m->seisout;
    m_s->movout=m->movout;
    m_s->resout=m->resout;
    m_s->rmsout=m->rmsout;
    m_s->tmax=m->tmax;
    m_s->tmin=m->tmin;
    m_s->param_type=m->param_type;
    m_s->fmin=m->fmin;
    m_s->fmax=m->fmax;
    m_s->restype=m->restype;
    m_s->vpmax=m->vpmax;
    m_s->vsmin=m->vsmin;
    m_s->topowidth=m->topowidth;
    m_s->scalerms=m->scalerms;
    m_s->scaleshot=m->scaleshot;
    m_s->scalermsnorm=m->scalermsnorm;
   
    
    GMALLOC(m_s->rho, sizeof(float)*m_s->NX*m_s->NY*m_s->NZ)
    GMALLOC(m_s->u, sizeof(float)*m_s->NX*m_s->NY*m_s->NZ)
    if (m->ND!=21)
        GMALLOC(m_s->pi, sizeof(float)*m_s->NX*m_s->NY*m_s->NZ)
    __GUARD Init_cst(m_s);
    
    if (m_s->rho) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->rho, m_s->rho);
    if (m_s->rip) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->rip, m_s->rip);
    if (m_s->rjp) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->rjp, m_s->rjp);
    if (m_s->rkp) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->rkp, m_s->rkp);
    if (m_s->u) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->u, m_s->u);
    if (m_s->uipjp) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->uipjp, m_s->uipjp);
    if (m_s->ujpkp) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->ujpkp, m_s->ujpkp);
    if (m_s->uipkp) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->uipkp, m_s->uipkp);
    if (m_s->pi) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->pi, m_s->pi);
    if (m_s->taus) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->taus, m_s->taus);
    if (m_s->tausipjp) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->tausipjp, m_s->tausipjp);
    if (m_s->tausjpkp) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->tausjpkp, m_s->tausjpkp);
    if (m_s->tausipkp) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->tausipkp, m_s->tausipkp);
    if (m_s->taup) intertrilin(m_s->NX, m_s->NY, m_s->NZ, m_s->Nr, m->NX, m->NY, m->NZ, m->taup, m_s->taup);
    
    if (m_s->rho) initprop(m_s->NX, m_s->NY, m_s->NZ, m->rho, 1.2);
    if (m_s->rip) initprop(m_s->NX, m_s->NY, m_s->NZ, m->rip, 1.2);
    if (m_s->rjp) initprop(m_s->NX, m_s->NY, m_s->NZ, m->rjp, 1.2);
    if (m_s->rkp) initprop(m_s->NX, m_s->NY, m_s->NZ, m->rkp, 1.2);
    if (m_s->u) initprop(m_s->NX, m_s->NY, m_s->NZ, m->u, 0.01);
    if (m_s->uipjp) initprop(m_s->NX, m_s->NY, m_s->NZ, m->uipjp, 0.01);
    if (m_s->ujpkp) initprop(m_s->NX, m_s->NY, m_s->NZ, m->ujpkp, 0.01);
    if (m_s->uipkp) initprop(m_s->NX, m_s->NY, m_s->NZ, m->uipkp, 0.01);
    if (m_s->pi) initprop(m_s->NX, m_s->NY, m_s->NZ, m->pi, 1.2*pow(340,2));
    if (m_s->taus) initprop(m_s->NX, m_s->NY, m_s->NZ, m->taus, 0);
    if (m_s->tausipjp) initprop(m_s->NX, m_s->NY, m_s->NZ, m->tausipjp, 0);
    if (m_s->tausjpkp) initprop(m_s->NX, m_s->NY, m_s->NZ, m->tausjpkp, 0);
    if (m_s->tausipkp) initprop(m_s->NX, m_s->NY, m_s->NZ, m->tausipkp, 0);
    if (m_s->taup) initprop(m_s->NX, m_s->NY, m_s->NZ, m->taup, 0);
    
    
    m_s->device_type=m->device_type;
    m_s->num_devices=m->num_devices;
    m_s->context=m->context;
    GMALLOC(vcl_s,sizeof(struct varcl)*m_s->num_devices)
    if (!state) memset ((void*)(vcl_s), 0, sizeof(struct varcl)*m_s->num_devices);
    GMALLOC(mloc_s,sizeof(struct modcstsloc)*m_s->num_devices)
    if (!state) memset ((void*)(mloc_s), 0, sizeof(struct modcstsloc)*m_s->num_devices);
    for (d=0;d<m_s->num_devices;d++){
        if (!state) vcl_s[d]->cmd_queue = vcl[d]->cmd_queue;
        if (!state) vcl_s[d]->cmd_queuecomm= vcl[d]->cmd_queuecomm;
        if (!state) mloc_s[d]->required_global_mem_size= mloc[d]->required_global_mem_size;
    }
    
    GMALLOC(m_s->sinccoef,sizeof(float)* pow(m->FDORDER,2) )
    
    for (i=0;i<m->FDORDER;i++){
        for (j=0;j<pow(m->Nr,2);j++){
            
            m_s->sinccoef[i*m->FDORDER+j]=sinc(PI*(float)(  j*m_s->dt/m->dt  +(m->fdoh-1)-i));
            
        }
    }
    m_s->maingrid=0;
    
    int cubic_wt3d[1000]={1,1,-3,3,-2,-1,2,-2,1,1,1,1,-3,3,-2,-1,2,-2,1,1,-3,3,-2,-1,-3,3,-2,-1,9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1,-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1,2,-2,1,1,2,-2,1,1,-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1,4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1,1,1,-3,3,-2,-1,2,-2,1,1,1,1,-3,3,-2,-1,2,-2,1,1,-3,3,-2,-1,-3,3,-2,-1,9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1,-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1,2,-2,1,1,2,-2,1,1,-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1,4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1,-3,3,-2,-1,-3,3,-2,-1,9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1,-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1,-3,3,-2,-1,-3,3,-2,-1,9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1,-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1,9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1,9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1,-27,27,27,-27,27,-27,-27,27,-18,-9,18,9,18,9,-18,-9,-18,18,-9,9,18,-18,9,-9,-18,18,18,-18,-9,9,9,-9,-12,-6,-6,-3,12,6,6,3,-12,-6,12,6,-6,-3,6,3,-12,12,-6,6,-6,6,-3,3,-8,-4,-4,-2,-4,-2,-2,-1,18,-18,-18,18,-18,18,18,-18,9,9,-9,-9,-9,-9,9,9,12,-12,6,-6,-12,12,-6,6,12,-12,-12,12,6,-6,-6,6,6,6,3,3,-6,-6,-3,-3,6,6,-6,-6,3,3,-3,-3,8,-8,4,-4,4,-4,2,-2,4,4,2,2,2,2,1,1,-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1,-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1,18,-18,-18,18,-18,18,18,-18,12,6,-12,-6,-12,-6,12,6,9,-9,9,-9,-9,9,-9,9,12,-12,-12,12,6,-6,-6,6,6,3,6,3,-6,-3,-6,-3,8,4,-8,-4,4,2,-4,-2,6,-6,6,-6,3,-3,3,-3,4,2,4,2,2,1,2,1,-12,12,12,-12,12,-12,-12,12,-6,-6,6,6,6,6,-6,-6,-6,6,-6,6,6,-6,6,-6,-8,8,8,-8,-4,4,4,-4,-3,-3,-3,-3,3,3,3,3,-4,-4,4,4,-2,-2,2,2,-4,4,-4,4,-2,2,-2,2,-2,-2,-2,-2,-1,-1,-1,-1,2,-2,1,1,2,-2,1,1,-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1,4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1,2,-2,1,1,2,-2,1,1,-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1,4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1,-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1,-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1,18,-18,-18,18,-18,18,18,-18,12,6,-12,-6,-12,-6,12,6,12,-12,6,-6,-12,12,-6,6,9,-9,-9,9,9,-9,-9,9,8,4,4,2,-8,-4,-4,-2,6,3,-6,-3,6,3,-6,-3,6,-6,3,-3,6,-6,3,-3,4,2,2,1,4,2,2,1,-12,12,12,-12,12,-12,-12,12,-6,-6,6,6,6,6,-6,-6,-8,8,-4,4,8,-8,4,-4,-6,6,6,-6,-6,6,6,-6,-4,-4,-2,-2,4,4,2,2,-3,-3,3,3,-3,-3,3,3,-4,4,-2,2,-4,4,-2,2,-2,-2,-1,-1,-2,-2,-1,-1,4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1,4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1,-12,12,12,-12,12,-12,-12,12,-8,-4,8,4,8,4,-8,-4,-6,6,-6,6,6,-6,6,-6,-6,6,6,-6,-6,6,6,-6,-4,-2,-4,-2,4,2,4,2,-4,-2,4,2,-4,-2,4,2,-3,3,-3,3,-3,3,-3,3,-2,-1,-2,-1,-2,-1,-2,-1,8,-8,-8,8,-8,8,8,-8,4,4,-4,-4,-4,-4,4,4,4,-4,4,-4,-4,4,-4,4,4,-4,-4,4,4,-4,-4,4,2,2,2,2,-2,-2,-2,-2,2,2,-2,-2,2,2,-2,-2,2,-2,2,-2,2,-2,2,-2,1,1,1,1,1,1,1,1};
    
    int cubic_ind3d[1000]={0,8,0,1,8,9,0,1,8,9,16,32,16,17,32,33,16,17,32,33,0,2,16,18,8,10,32,34,0,1,2,3,8,9,10,11,16,17,18,19,32,33,34,35,0,1,2,3,8,9,10,11,16,17,18,19,32,33,34,35,0,2,16,18,8,10,32,34,0,1,2,3,8,9,10,11,16,17,18,19,32,33,34,35,0,1,2,3,8,9,10,11,16,17,18,19,32,33,34,35,24,40,24,25,40,41,24,25,40,41,48,56,48,49,56,57,48,49,56,57,24,26,48,50,40,42,56,58,24,25,26,27,40,41,42,43,48,49,50,51,56,57,58,59,24,25,26,27,40,41,42,43,48,49,50,51,56,57,58,59,24,26,48,50,40,42,56,58,24,25,26,27,40,41,42,43,48,49,50,51,56,57,58,59,24,25,26,27,40,41,42,43,48,49,50,51,56,57,58,59,0,4,24,28,8,12,40,44,0,1,4,5,8,9,12,13,24,25,28,29,40,41,44,45,0,1,4,5,8,9,12,13,24,25,28,29,40,41,44,45,16,20,48,52,32,36,56,60,16,17,20,21,32,33,36,37,48,49,52,53,56,57,60,61,16,17,20,21,32,33,36,37,48,49,52,53,56,57,60,61,0,2,4,6,16,18,20,22,24,26,28,30,48,50,52,54,8,10,12,14,32,34,36,38,40,42,44,46,56,58,60,62,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,0,2,4,6,16,18,20,22,24,26,28,30,48,50,52,54,8,10,12,14,32,34,36,38,40,42,44,46,56,58,60,62,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,0,4,24,28,8,12,40,44,0,1,4,5,8,9,12,13,24,25,28,29,40,41,44,45,0,1,4,5,8,9,12,13,24,25,28,29,40,41,44,45,16,20,48,52,32,36,56,60,16,17,20,21,32,33,36,37,48,49,52,53,56,57,60,61,16,17,20,21,32,33,36,37,48,49,52,53,56,57,60,61,0,2,4,6,16,18,20,22,24,26,28,30,48,50,52,54,8,10,12,14,32,34,36,38,40,42,44,46,56,58,60,62,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,0,2,4,6,16,18,20,22,24,26,28,30,48,50,52,54,8,10,12,14,32,34,36,38,40,42,44,46,56,58,60,62,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63};
    
    int cubic_pos3d[65]={0,1,2,6,10,11,12,16,20,24,28,44,60,64,68,84,100,101,102,106,110,111,112,116,120,124,128,144,160,164,168,184,200,204,208,224,240,244,248,264,280,296,312,376,440,456,472,536,600,604,608,624,640,644,648,664,680,696,712,776,840,856,872,936,1000};
    
    int cubic_wt2d[100]={1,1,-3,3,-2,-1,2,-2,1,1,1,1,-3,3,-2,-1,2,-2,1,1,-3,3,-2,-1,-3,3,-2,-1,9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1,-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1,2,-2,1,1,2,-2,1,1,-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1,4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1};

    int cubic_ind2d[100]={0,4,0,1,4,5,0,1,4,5,8,12,8,9,12,13,8,9,12,13,0,2,8,10,4,6,12,14,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,2,8,10,4,6,12,14,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        
    int cubic_pos2d[17]={0,1,2,6,10,11,12,16,20,24,28,44,60,64,68,84,100};
    
    
    if (m_s->ND==3){
        m_s->cubic_wt=cubic_wt3d;
        m_s->cubic_ind=cubic_ind3d;
        m_s->cubic_pos=cubic_pos3d;
    }
    
    else {
        m_s->cubic_wt=cubic_wt2d;
        m_s->cubic_ind=cubic_ind2d;
        m_s->cubic_pos=cubic_pos2d;
    }
    
    return state;
}
