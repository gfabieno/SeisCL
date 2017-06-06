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

/*Adjoint update of the stresses in 3D*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define lbnd (fdoh+nab)

#define rho(z,y,x)             rho[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rip(z,y,x)             rip[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rjp(z,y,x)             rjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,y,x)             rkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipjp(z,y,x)         uipjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define ujpkp(z,y,x)         ujpkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipkp(z,y,x)         uipkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define u(z,y,x)                 u[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define pi(z,y,x)               pi[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradrho(z,y,x)     gradrho[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradM(z,y,x)         gradM[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradmu(z,y,x)       gradmu[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaup(z,y,x)   gradtaup[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaus(z,y,x)   gradtaus[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define taus(z,y,x)         taus[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipjp(z,y,x) tausipjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausjpkp(z,y,x) tausjpkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipkp(z,y,x) tausipkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define taup(z,y,x)         taup[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define vx(z,y,x)   vx[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vy(z,y,x)   vy[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz(z,y,x)   vz[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxx(z,y,x) sxx[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syy(z,y,x) syy[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define szz(z,y,x) szz[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxy(z,y,x) sxy[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syz(z,y,x) syz[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxz(z,y,x) sxz[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]

#define rxx(z,y,x,l) rxx[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryy(z,y,x,l) ryy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rzz(z,y,x,l) rzz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxy(z,y,x,l) rxy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryz(z,y,x,l) ryz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxz(z,y,x,l) rxz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]

#define vx_r(z,y,x)   vx_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vy_r(z,y,x)   vy_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz_r(z,y,x)   vz_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxx_r(z,y,x) sxx_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syy_r(z,y,x) syy_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define szz_r(z,y,x) szz_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxy_r(z,y,x) sxy_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syz_r(z,y,x) syz_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxz_r(z,y,x) sxz_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]

#define rxx_r(z,y,x,l) rxx_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryy_r(z,y,x,l) ryy_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rzz_r(z,y,x,l) rzz_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxy_r(z,y,x,l) rxy_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryz_r(z,y,x,l) ryz_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxz_r(z,y,x,l) rxz_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]

#define psi_vxx(z,y,x) psi_vxx[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vyx(z,y,x) psi_vyx[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vzx(z,y,x) psi_vzx[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]

#define psi_vxy(z,y,x) psi_vxy[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vyy(z,y,x) psi_vyy[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vzy(z,y,x) psi_vzy[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]

#define psi_vxz(z,y,x) psi_vxz[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_vyz(z,y,x) psi_vyz[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_vzz(z,y,x) psi_vzz[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#if local_off==0

#define lvar(z,y,x)   lvar[(x)*lsizey*lsizez+(y)*lsizez+(z)]


#endif



#define PI (3.141592653589793238462643383279502884197169)
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]
#define rec_pos(y,x) rec_pos[(y)*8+(x)]

#define gradsrc(y,x) gradsrc[(y)*NT+(x)]

float psource(int gidz, int gidy, int gidx,  int nsrc, __global float *srcpos_loc, __global float *signals, int nt){
    
    float amp=0.0;
    if (nsrc>0){
        
        for (int srci=0; srci<nsrc; srci++){
            
            int SOURCE_TYPE= (int)srcpos_loc(4,srci);
            
            if (SOURCE_TYPE==1){
                int i=(int)(srcpos_loc(0,srci)/DH-0.5)+fdoh;
                int j=(int)(srcpos_loc(1,srci)/DH-0.5)+fdoh;
                int k=(int)(srcpos_loc(2,srci)/DH-0.5)+fdoh;
                
                
                if (i==gidx && j==gidy && k==gidz){
                    
                    //amp+=signals(srci,nt)/(DH*DH*DH);
                    
                    //                    if ( (nt>0) && (nt< NT ) ){
                    amp+=(signals(srci,nt+1)-signals(srci,nt-1) )/(2.0*DH*DH*DH);
                    //                    }
                    //                    else if (nt==0)
                    //                        amp+=signals(srci,nt+1) /(2.0*DH*DH*DH);
                    //                    else if (nt==NT)
                    //                        amp+=signals(srci,nt-1) /(2.0*DH*DH*DH);
                }
            }
        }
    }
    
    return amp;
    
}

// Find boundary indice for boundary injection in backpropagation
int evarm( int k, int j, int i){
    
    
#if num_devices==1 & NLOCALP==1
    
    int NXbnd = (NX-2*fdoh-2*nab);
    int NYbnd = (NY-2*fdoh-2*nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    
    int m=-1;
    i-=lbnd;
    j-=lbnd;
    k-=lbnd;
    
    if ( (k>fdoh-1 && k<NZbnd-fdoh) && (j>fdoh-1 && j<NYbnd-fdoh) && (i>fdoh-1 && i<NXbnd-fdoh) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<fdoh){//front
        m=i*NYbnd*NZbnd+j*NZbnd+k;
    }
    else if (i>NXbnd-1-fdoh){//back
        i=i-NXbnd+fdoh;
        m=NYbnd*NZbnd*fdoh+i*NYbnd*NZbnd+j*NZbnd+k;
    }
    else if (j<fdoh){//left
        i=i-fdoh;
        m=NYbnd*NZbnd*fdoh*2+i*fdoh*NZbnd+j*NZbnd+k;
    }
    else if (j>NYbnd-1-fdoh){//right
        i=i-fdoh;
        j=j-NYbnd+fdoh;
        m=NYbnd*NZbnd*fdoh*2+(NXbnd-2*fdoh)*NZbnd*fdoh+i*fdoh*NZbnd+j*NZbnd+k;
    }
    else if (k<fdoh){//up
        i=i-fdoh;
        j=j-fdoh;
        m=NYbnd*NZbnd*fdoh*2+(NXbnd-2*fdoh)*NZbnd*fdoh*2+i*(NYbnd-2*fdoh)*fdoh+j*fdoh+k;
    }
    else {//down
        i=i-fdoh;
        j=j-fdoh;
        k=k-NZbnd+fdoh;
        m=NYbnd*NZbnd*fdoh*2+(NXbnd-2*fdoh)*NZbnd*fdoh*2+(NXbnd-2*fdoh)*(NYbnd-2*fdoh)*fdoh+i*(NYbnd-2*fdoh)*fdoh+j*fdoh+k;
    }

    
#elif dev==0 & MYGROUPID==0
    int NXbnd = (NX-2*fdoh-nab);
    int NYbnd = (NY-2*fdoh-2*nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    
    int m=-1;
    i-=lbnd;
    j-=lbnd;
    k-=lbnd;
    
    if ( (k>fdoh-1 && k<NZbnd-fdoh) && (j>fdoh-1 && j<NYbnd-fdoh) && i>fdoh-1  )
        m=-1;
    else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<fdoh){//front
        m=i*NYbnd*NZbnd+j*NZbnd+k;
    }
    else if (j<fdoh){//left
        i=i-fdoh;
        m=NYbnd*NZbnd*fdoh+i*fdoh*NZbnd+j*NZbnd+k;
    }
    else if (j>NYbnd-1-fdoh){//right
        i=i-fdoh;
        j=j-NYbnd+fdoh;
        m=NYbnd*NZbnd*fdoh+(NXbnd-fdoh)*NZbnd*fdoh+i*fdoh*NZbnd+j*NZbnd+k;
    }
    else if (k<fdoh){//up
        i=i-fdoh;
        j=j-fdoh;
        m=NYbnd*NZbnd*fdoh+(NXbnd-fdoh)*NZbnd*fdoh*2+i*(NYbnd-2*fdoh)*fdoh+j*fdoh+k;
    }
    else {//down
        i=i-fdoh;
        j=j-fdoh;
        k=k-NZbnd+fdoh;
        m=NYbnd*NZbnd*fdoh+(NXbnd-fdoh)*NZbnd*fdoh*2+(NXbnd-fdoh)*(NYbnd-2*fdoh)*fdoh+i*(NYbnd-2*fdoh)*fdoh+j*fdoh+k;
    }
#elif dev==num_devices-1 & MYGROUPID==NLOCALP-1
    int NXbnd = (NX-2*fdoh-nab);
    int NYbnd = (NY-2*fdoh-2*nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    
    int m=-1;
    i-=fdoh;
    j-=lbnd;
    k-=lbnd;
    
    if ( (k>fdoh-1 && k<NZbnd-fdoh) && (j>fdoh-1 && j<NYbnd-fdoh) && i<NXbnd-fdoh )
        m=-1;
    else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1  || i>NXbnd-1 )
        m=-1;
    else if (i>NXbnd-1-fdoh){//back
        i=i-NXbnd+fdoh;
        m=i*NYbnd*NZbnd+j*NZbnd+k;
    }
    else if (j<fdoh){//left
        m=NYbnd*NZbnd*fdoh+i*fdoh*NZbnd+j*NZbnd+k;
    }
    else if (j>NYbnd-1-fdoh){//right
        j=j-NYbnd+fdoh;
        m=NYbnd*NZbnd*fdoh+(NXbnd-fdoh)*NZbnd*fdoh+i*fdoh*NZbnd+j*NZbnd+k;
    }
    else if (k<fdoh){//up
        j=j-fdoh;
        m=NYbnd*NZbnd*fdoh+(NXbnd-fdoh)*NZbnd*fdoh*2+i*(NYbnd-2*fdoh)*fdoh+j*fdoh+k;
    }
    else {//down
        j=j-fdoh;
        k=k-NZbnd+fdoh;
        m=NYbnd*NZbnd*fdoh+(NXbnd-fdoh)*NZbnd*fdoh*2+(NXbnd-fdoh)*(NYbnd-2*fdoh)*fdoh+i*(NYbnd-2*fdoh)*fdoh+j*fdoh+k;
    }
    
#else
    int NXbnd = (NX-2*fdoh);
    int NYbnd = (NY-2*fdoh-2*nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    
    int m=-1;
    i-=fdoh;
    j-=lbnd;
    k-=lbnd;
    
    if ( (k>fdoh-1 && k<NZbnd-fdoh) && (j>fdoh-1 && j<NYbnd-fdoh) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1  || i<0 || i>NXbnd-1 )
        m=-1;
    else if (j<fdoh){//left
        m=i*fdoh*NZbnd+j*NZbnd+k;
    }
    else if (j>NYbnd-1-fdoh){//right
        j=j-NYbnd+fdoh;
        m=NXbnd*NZbnd*fdoh+i*fdoh*NZbnd+j*NZbnd+k;
    }
    else if (k<fdoh){//up
        j=j-fdoh;
        m=NXbnd*NZbnd*fdoh*2+i*(NYbnd-2*fdoh)*fdoh+j*fdoh+k;
    }
    else {//down
        j=j-fdoh;
        k=k-NZbnd+fdoh;
        m=NXbnd*NZbnd*fdoh*2+NXbnd*(NYbnd-2*fdoh)*fdoh+i*(NYbnd-2*fdoh)*fdoh+j*fdoh+k;
    }
    
#endif
    
    
    return m;
    
}

__kernel void update_adjs(int offcomm, int nsrc,  int nt,
                          __global float *vx,         __global float *vy,       __global float *vz,
                          __global float *sxx,        __global float *syy,      __global float *szz,
                          __global float *sxy,        __global float *syz,      __global float *sxz,
                          __global float *vxbnd,      __global float *vybnd,    __global float *vzbnd,
                          __global float *sxxbnd,     __global float *syybnd,   __global float *szzbnd,
                          __global float *sxybnd,     __global float *syzbnd,   __global float *sxzbnd,
                          __global float *vx_r,       __global float *vy_r,     __global float *vz_r,
                          __global float *sxx_r,      __global float *syy_r,    __global float *szz_r,
                          __global float *sxy_r,      __global float *syz_r,    __global float *sxz_r,
                          __global float *rxx,        __global float *ryy,      __global float *rzz,
                          __global float *rxy,        __global float *ryz,      __global float *rxz,
                          __global float *rxx_r,      __global float *ryy_r,    __global float *rzz_r,
                          __global float *rxy_r,      __global float *ryz_r,    __global float *rxz_r,
                          __global float *pi,         __global float *u,        __global float *uipjp,
                          __global float *ujpkp,      __global float *uipkp,
                          __global float *taus,       __global float *tausipjp, __global float *tausjpkp,
                          __global float *tausipkp,   __global float *taup,     __global float *eta,
                          __global float *srcpos_loc, __global float *signals,  __global float *taper,
                          __global float *K_x,        __global float *a_x,      __global float *b_x,
                          __global float *K_x_half,   __global float *a_x_half, __global float *b_x_half,
                          __global float *K_y,        __global float *a_y,      __global float *b_y,
                          __global float *K_y_half,   __global float *a_y_half, __global float *b_y_half,
                          __global float *K_z,        __global float *a_z,      __global float *b_z,
                          __global float *K_z_half,   __global float *a_z_half, __global float *b_z_half,
                          __global float *psi_vxx,    __global float *psi_vxy,  __global float *psi_vxz,
                          __global float *psi_vyx,    __global float *psi_vyy,  __global float *psi_vyz,
                          __global float *psi_vzx,    __global float *psi_vzy,  __global float *psi_vzz,
                          __global float *gradrho,    __global float *gradM,    __global float *gradmu,
                          __global float *gradtaup,   __global float *gradtaus, __global float *gradsrc,
                          __local  float *lvar)
{

    int i,j,k,m;
    float vxx,vxy,vxz,vyx,vyy,vyz,vzx,vzy,vzz;
    float vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
    float vxx_r,vxy_r,vxz_r,vyx_r,vyy_r,vyz_r,vzx_r,vzy_r,vzz_r;
    float vxyyx_r,vyzzy_r,vxzzx_r,vxxyyzz_r,vyyzz_r,vxxzz_r,vxxyy_r;
    float fipjp, fjpkp, fipkp, f, g;
    float sumrxy,sumryz,sumrxz,sumrxx,sumryy,sumrzz;
    float b,c,e,d,dipjp,djpkp,dipkp;
    int l;
    float lpi, lu, luipjp, luipkp, lujpkp,ltaup, ltaus, ltausipjp, ltausipkp, ltausjpkp;
    float leta[Lve];
    float lsxx, lsyy, lszz, lsxy, lsxz, lsyz;
    
    
    
    
// If we use local memory
#if local_off==0
    int lsizez = get_local_size(0)+2*fdoh;
    int lsizey = get_local_size(1)+2*fdoh;
    int lsizex = get_local_size(2)+2*fdoh;
    int lidz = get_local_id(0)+fdoh;
    int lidy = get_local_id(1)+fdoh;
    int lidx = get_local_id(2)+fdoh;
    int gidz = get_global_id(0)+fdoh;
    int gidy = get_global_id(1)+fdoh;
    int gidx = get_global_id(2)+fdoh+offcomm;

#define lvx lvar
#define lvy lvar
#define lvz lvar
#define lvx_r lvar
#define lvy_r lvar
#define lvz_r lvar

// If local memory is turned off
#elif local_off==1
    int gid = get_global_id(0);
    int glsizez = (NZ-2*fdoh);
    int glsizey = (NY-2*fdoh);
    int gidz = gid%glsizez+fdoh;
    int gidy = (gid/glsizez)%glsizey+fdoh;
    int gidx = gid/(glsizez*glsizey)+fdoh+offcomm;
#define lvx_r vx_r
#define lvy_r vy_r
#define lvz_r vz_r
#define lvx vx
#define lvy vy
#define lvz vz
#define lidx gidx
#define lidy gidy
#define lidz gidz
    
#define lsizez NZ
#define lsizey NY
#define lsizex NX
    
#endif
    
    
// Calculation of the velocity spatial derivatives of the forward wavefield if backpropagation is used
#if back_prop_type==1
    {
#if local_off==0
        lvx(lidz,lidy,lidx)=vx(gidz, gidy, gidx);
        if (lidy<2*fdoh)
            lvx(lidz,lidy-fdoh,lidx)=vx(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lvx(lidz,lidy+lsizey-3*fdoh,lidx)=vx(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lvx(lidz,lidy+fdoh,lidx)=vx(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lvx(lidz,lidy-lsizey+3*fdoh,lidx)=vx(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lvx(lidz,lidy,lidx-fdoh)=vx(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvx(lidz,lidy,lidx+lsizex-3*fdoh)=vx(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvx(lidz,lidy,lidx+fdoh)=vx(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvx(lidz,lidy,lidx-lsizex+3*fdoh)=vx(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvx(lidz-fdoh,lidy,lidx)=vx(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvx(lidz+fdoh,lidy,lidx)=vx(gidz+fdoh,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh==1
    vxx = (lvx(lidz,lidy,lidx)-lvx(lidz,lidy,lidx-1))/DH;
    vxy = (lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))/DH;
    vxz = (lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))/DH;
#elif fdoh==2
    vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2)))/DH;
    
    vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx)))/DH;
    
    vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx)))/DH;
#elif fdoh==3
    vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
           hc3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3)))/DH;
    
    vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
           hc3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx)))/DH;
    
    vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
           hc3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx)))/DH;
#elif fdoh==4
    vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
           hc3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
           hc4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4)))/DH;
    
    vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
           hc3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
           hc4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx)))/DH;
    
    vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
           hc3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
           hc4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx)))/DH;
#elif fdoh==5
    vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
           hc3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
           hc4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4))+
           hc5*(lvx(lidz,lidy,lidx+4)-lvx(lidz,lidy,lidx-5)))/DH;
    
    vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
           hc3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
           hc4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx))+
           hc5*(lvx(lidz,lidy+5,lidx)-lvx(lidz,lidy-4,lidx)))/DH;
    
    vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
           hc3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
           hc4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx))+
           hc5*(lvx(lidz+5,lidy,lidx)-lvx(lidz-4,lidy,lidx)))/DH;
#elif fdoh==6
    vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
           hc3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
           hc4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4))+
           hc5*(lvx(lidz,lidy,lidx+4)-lvx(lidz,lidy,lidx-5))+
           hc6*(lvx(lidz,lidy,lidx+5)-lvx(lidz,lidy,lidx-6)))/DH;
    
    vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
           hc3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
           hc4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx))+
           hc5*(lvx(lidz,lidy+5,lidx)-lvx(lidz,lidy-4,lidx))+
           hc6*(lvx(lidz,lidy+6,lidx)-lvx(lidz,lidy-5,lidx)))/DH;
    
    vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
           hc3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
           hc4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx))+
           hc5*(lvx(lidz+5,lidy,lidx)-lvx(lidz-4,lidy,lidx))+
           hc6*(lvx(lidz+6,lidy,lidx)-lvx(lidz-5,lidy,lidx)))/DH;
#endif
    
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
        lvy(lidz,lidy,lidx)=vy(gidz, gidy, gidx);
        if (lidy<2*fdoh)
            lvy(lidz,lidy-fdoh,lidx)=vy(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lvy(lidz,lidy+lsizey-3*fdoh,lidx)=vy(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lvy(lidz,lidy+fdoh,lidx)=vy(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lvy(lidz,lidy-lsizey+3*fdoh,lidx)=vy(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lvy(lidz,lidy,lidx-fdoh)=vy(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvy(lidz,lidy,lidx+lsizex-3*fdoh)=vy(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvy(lidz,lidy,lidx+fdoh)=vy(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvy(lidz,lidy,lidx-lsizex+3*fdoh)=vy(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvy(lidz-fdoh,lidy,lidx)=vy(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvy(lidz+fdoh,lidy,lidx)=vy(gidz+fdoh,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh==1
    vyx = (lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))/DH;
    vyy = (lvy(lidz,lidy,lidx)-lvy(lidz,lidy-1,lidx))/DH;
    vyz = (lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))/DH;
#elif fdoh==2
    vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1)))/DH;
    
    vyy = (hc1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
           hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx)))/DH;
    
    vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx)))/DH;
#elif fdoh==3
    vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
           hc3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2)))/DH;
    
    vyy = (hc1*(lvy(lidz,lidy,lidx)-lvy(lidz,lidy-1,lidx))+
           hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
           hc3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx)))/DH;
    
    vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
           hc3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx)))/DH;
#elif fdoh==4
    vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
           hc3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
           hc4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3)))/DH;
    
    vyy = (hc1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
           hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
           hc3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
           hc4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx)))/DH;
    
    vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
           hc3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
           hc4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx)))/DH;
#elif fdoh==5
    vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
           hc3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
           hc4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3))+
           hc5*(lvy(lidz,lidy,lidx+5)-lvy(lidz,lidy,lidx-4)))/DH;
    
    vyy = (hc1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
           hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
           hc3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
           hc4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx))+
           hc5*(lvy(lidz,lidy+4,lidx)-lvy(lidz,lidy-5,lidx)))/DH;
    
    vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
           hc3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
           hc4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx))+
           hc5*(lvy(lidz+5,lidy,lidx)-lvy(lidz-4,lidy,lidx)))/DH;
#elif fdoh==6
    vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
           hc3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
           hc4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3))+
           hc5*(lvy(lidz,lidy,lidx+5)-lvy(lidz,lidy,lidx-4))+
           hc6*(lvy(lidz,lidy,lidx+6)-lvy(lidz,lidy,lidx-5)))/DH;
    
    vyy = (hc1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
           hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
           hc3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
           hc4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx))+
           hc5*(lvy(lidz,lidy+4,lidx)-lvy(lidz,lidy-5,lidx))+
           hc6*(lvy(lidz,lidy+5,lidx)-lvy(lidz,lidy-6,lidx)))/DH;
    
    vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
           hc3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
           hc4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx))+
           hc5*(lvy(lidz+5,lidy,lidx)-lvy(lidz-4,lidy,lidx))+
           hc6*(lvy(lidz+6,lidy,lidx)-lvy(lidz-5,lidy,lidx)))/DH;
#endif
    
    
    
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
        lvz(lidz,lidy,lidx)=vz(gidz, gidy, gidx);
        if (lidy<2*fdoh)
            lvz(lidz,lidy-fdoh,lidx)=vz(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lvz(lidz,lidy+lsizey-3*fdoh,lidx)=vz(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lvz(lidz,lidy+fdoh,lidx)=vz(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lvz(lidz,lidy-lsizey+3*fdoh,lidx)=vz(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lvz(lidz,lidy,lidx-fdoh)=vz(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvz(lidz,lidy,lidx+lsizex-3*fdoh)=vz(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvz(lidz,lidy,lidx+fdoh)=vz(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvz(lidz,lidy,lidx-lsizex+3*fdoh)=vz(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvz(lidz-fdoh,lidy,lidx)=vz(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvz(lidz+fdoh,lidy,lidx)=vz(gidz+fdoh,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
    
#if   fdoh==1
    vzx = (lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))/DH;
    vzy = (lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))/DH;
    vzz = (lvz(lidz,lidy,lidx)-lvz(lidz-1,lidy,lidx))/DH;
#elif fdoh==2
    
    vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1)))/DH;
    
    vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx)))/DH;
    
    vzz = (hc1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
           hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx)))/DH;
#elif fdoh==3
    vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
           hc3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2)))/DH;
    
    vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
           hc3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx)))/DH;
    
    vzz = (hc1*(lvz(lidz,lidy,lidx)-lvz(lidz-1,lidy,lidx))+
           hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
           hc3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx)))/DH;
#elif fdoh==4
    vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
           hc3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
           hc4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3)))/DH;
    
    vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
           hc3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
           hc4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx)))/DH;
    
    vzz = (hc1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
           hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
           hc3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
           hc4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx)))/DH;
#elif fdoh==5
    vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
           hc3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
           hc4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3))+
           hc5*(lvz(lidz,lidy,lidx+5)-lvz(lidz,lidy,lidx-4)))/DH;
    
    vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
           hc3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
           hc4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx))+
           hc5*(lvz(lidz,lidy+5,lidx)-lvz(lidz,lidy-4,lidx)))/DH;
    
    vzz = (hc1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
           hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
           hc3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
           hc4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx))+
           hc5*(lvz(lidz+4,lidy,lidx)-lvz(lidz-5,lidy,lidx)))/DH;
#elif fdoh==6
    vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
           hc3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
           hc4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3))+
           hc5*(lvz(lidz,lidy,lidx+5)-lvz(lidz,lidy,lidx-4))+
           hc6*(lvz(lidz,lidy,lidx+6)-lvz(lidz,lidy,lidx-5)))/DH;
    
    vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
           hc3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
           hc4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx))+
           hc5*(lvz(lidz,lidy+5,lidx)-lvz(lidz,lidy-4,lidx))+
           hc6*(lvz(lidz,lidy+6,lidx)-lvz(lidz,lidy-5,lidx)))/DH;
    
    vzz = (hc1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
           hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
           hc3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
           hc4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx))+
           hc5*(lvz(lidz+4,lidy,lidx)-lvz(lidz-5,lidy,lidx))+
           hc6*(lvz(lidz+5,lidy,lidx)-lvz(lidz-6,lidy,lidx)))/DH;
#endif

        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif

// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
#if local_off==0
        lvx_r(lidz,lidy,lidx)=vx_r(gidz, gidy, gidx);
        if (lidy<2*fdoh)
            lvx_r(lidz,lidy-fdoh,lidx)=vx_r(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lvx_r(lidz,lidy+lsizey-3*fdoh,lidx)=vx_r(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lvx_r(lidz,lidy+fdoh,lidx)=vx_r(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lvx_r(lidz,lidy-lsizey+3*fdoh,lidx)=vx_r(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lvx_r(lidz,lidy,lidx-fdoh)=vx_r(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvx_r(lidz,lidy,lidx+lsizex-3*fdoh)=vx_r(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvx_r(lidz,lidy,lidx+fdoh)=vx_r(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvx_r(lidz,lidy,lidx-lsizex+3*fdoh)=vx_r(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvx_r(lidz-fdoh,lidy,lidx)=vx_r(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvx_r(lidz+fdoh,lidy,lidx)=vx_r(gidz+fdoh,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh==1
    vxx_r = (lvx_r(lidz,lidy,lidx)-lvx_r(lidz,lidy,lidx-1))/DH;
    vxy_r = (lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))/DH;
    vxz_r = (lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))/DH;
#elif fdoh==2
    vxx_r = (hc1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           hc2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2)))/DH;
    
    vxy_r = (hc1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx)))/DH;
    
    vxz_r = (hc1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx)))/DH;
#elif fdoh==3
    vxx_r = (hc1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           hc2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2))+
           hc3*(lvx_r(lidz,lidy,lidx+2)-lvx_r(lidz,lidy,lidx-3)))/DH;
    
    vxy_r = (hc1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx))+
           hc3*(lvx_r(lidz,lidy+3,lidx)-lvx_r(lidz,lidy-2,lidx)))/DH;
    
    vxz_r = (hc1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx))+
           hc3*(lvx_r(lidz+3,lidy,lidx)-lvx_r(lidz-2,lidy,lidx)))/DH;
#elif fdoh==4
    vxx_r = (hc1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           hc2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2))+
           hc3*(lvx_r(lidz,lidy,lidx+2)-lvx_r(lidz,lidy,lidx-3))+
           hc4*(lvx_r(lidz,lidy,lidx+3)-lvx_r(lidz,lidy,lidx-4)))/DH;
    
    vxy_r = (hc1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx))+
           hc3*(lvx_r(lidz,lidy+3,lidx)-lvx_r(lidz,lidy-2,lidx))+
           hc4*(lvx_r(lidz,lidy+4,lidx)-lvx_r(lidz,lidy-3,lidx)))/DH;
    
    vxz_r = (hc1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx))+
           hc3*(lvx_r(lidz+3,lidy,lidx)-lvx_r(lidz-2,lidy,lidx))+
           hc4*(lvx_r(lidz+4,lidy,lidx)-lvx_r(lidz-3,lidy,lidx)))/DH;
#elif fdoh==5
    vxx_r = (hc1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           hc2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2))+
           hc3*(lvx_r(lidz,lidy,lidx+2)-lvx_r(lidz,lidy,lidx-3))+
           hc4*(lvx_r(lidz,lidy,lidx+3)-lvx_r(lidz,lidy,lidx-4))+
           hc5*(lvx_r(lidz,lidy,lidx+4)-lvx_r(lidz,lidy,lidx-5)))/DH;
    
    vxy_r = (hc1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx))+
           hc3*(lvx_r(lidz,lidy+3,lidx)-lvx_r(lidz,lidy-2,lidx))+
           hc4*(lvx_r(lidz,lidy+4,lidx)-lvx_r(lidz,lidy-3,lidx))+
           hc5*(lvx_r(lidz,lidy+5,lidx)-lvx_r(lidz,lidy-4,lidx)))/DH;
    
    vxz_r = (hc1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx))+
           hc3*(lvx_r(lidz+3,lidy,lidx)-lvx_r(lidz-2,lidy,lidx))+
           hc4*(lvx_r(lidz+4,lidy,lidx)-lvx_r(lidz-3,lidy,lidx))+
           hc5*(lvx_r(lidz+5,lidy,lidx)-lvx_r(lidz-4,lidy,lidx)))/DH;
#elif fdoh==6
    vxx_r = (hc1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           hc2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2))+
           hc3*(lvx_r(lidz,lidy,lidx+2)-lvx_r(lidz,lidy,lidx-3))+
           hc4*(lvx_r(lidz,lidy,lidx+3)-lvx_r(lidz,lidy,lidx-4))+
           hc5*(lvx_r(lidz,lidy,lidx+4)-lvx_r(lidz,lidy,lidx-5))+
           hc6*(lvx_r(lidz,lidy,lidx+5)-lvx_r(lidz,lidy,lidx-6)))/DH;
    
    vxy_r = (hc1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx))+
           hc3*(lvx_r(lidz,lidy+3,lidx)-lvx_r(lidz,lidy-2,lidx))+
           hc4*(lvx_r(lidz,lidy+4,lidx)-lvx_r(lidz,lidy-3,lidx))+
           hc5*(lvx_r(lidz,lidy+5,lidx)-lvx_r(lidz,lidy-4,lidx))+
           hc6*(lvx_r(lidz,lidy+6,lidx)-lvx_r(lidz,lidy-5,lidx)))/DH;
    
    vxz_r = (hc1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           hc2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx))+
           hc3*(lvx_r(lidz+3,lidy,lidx)-lvx_r(lidz-2,lidy,lidx))+
           hc4*(lvx_r(lidz+4,lidy,lidx)-lvx_r(lidz-3,lidy,lidx))+
           hc5*(lvx_r(lidz+5,lidy,lidx)-lvx_r(lidz-4,lidy,lidx))+
           hc6*(lvx_r(lidz+6,lidy,lidx)-lvx_r(lidz-5,lidy,lidx)))/DH;
#endif
    
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
        lvy_r(lidz,lidy,lidx)=vy_r(gidz, gidy, gidx);
        if (lidy<2*fdoh)
            lvy_r(lidz,lidy-fdoh,lidx)=vy_r(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lvy_r(lidz,lidy+lsizey-3*fdoh,lidx)=vy_r(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lvy_r(lidz,lidy+fdoh,lidx)=vy_r(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lvy_r(lidz,lidy-lsizey+3*fdoh,lidx)=vy_r(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lvy_r(lidz,lidy,lidx-fdoh)=vy_r(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvy_r(lidz,lidy,lidx+lsizex-3*fdoh)=vy_r(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvy_r(lidz,lidy,lidx+fdoh)=vy_r(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvy_r(lidz,lidy,lidx-lsizex+3*fdoh)=vy_r(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvy_r(lidz-fdoh,lidy,lidx)=vy_r(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvy_r(lidz+fdoh,lidy,lidx)=vy_r(gidz+fdoh,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh==1
    vyx_r = (lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))/DH;
    vyy_r = (lvy_r(lidz,lidy,lidx)-lvy_r(lidz,lidy-1,lidx))/DH;
    vyz_r = (lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))/DH;
#elif fdoh==2
    vyx_r = (hc1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1)))/DH;
    
    vyy_r = (hc1*(lvy_r(lidz,lidy,lidx)  -lvy_r(lidz,lidy-1,lidx))+
           hc2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx)))/DH;
    
    vyz_r = (hc1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx)))/DH;
#elif fdoh==3
    vyx_r = (hc1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1))+
           hc3*(lvy_r(lidz,lidy,lidx+3)-lvy_r(lidz,lidy,lidx-2)))/DH;
    
    vyy_r = (hc1*(lvy_r(lidz,lidy,lidx)-lvy_r(lidz,lidy-1,lidx))+
           hc2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx))+
           hc3*(lvy_r(lidz,lidy+2,lidx)-lvy_r(lidz,lidy-3,lidx)))/DH;
    
    vyz_r = (hc1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx))+
           hc3*(lvy_r(lidz+3,lidy,lidx)-lvy_r(lidz-2,lidy,lidx)))/DH;
#elif fdoh==4
    vyx_r = (hc1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1))+
           hc3*(lvy_r(lidz,lidy,lidx+3)-lvy_r(lidz,lidy,lidx-2))+
           hc4*(lvy_r(lidz,lidy,lidx+4)-lvy_r(lidz,lidy,lidx-3)))/DH;
    
    vyy_r = (hc1*(lvy_r(lidz,lidy,lidx)  -lvy_r(lidz,lidy-1,lidx))+
           hc2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx))+
           hc3*(lvy_r(lidz,lidy+2,lidx)-lvy_r(lidz,lidy-3,lidx))+
           hc4*(lvy_r(lidz,lidy+3,lidx)-lvy_r(lidz,lidy-4,lidx)))/DH;
    
    vyz_r = (hc1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx))+
           hc3*(lvy_r(lidz+3,lidy,lidx)-lvy_r(lidz-2,lidy,lidx))+
           hc4*(lvy_r(lidz+4,lidy,lidx)-lvy_r(lidz-3,lidy,lidx)))/DH;
#elif fdoh==5
    vyx_r = (hc1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1))+
           hc3*(lvy_r(lidz,lidy,lidx+3)-lvy_r(lidz,lidy,lidx-2))+
           hc4*(lvy_r(lidz,lidy,lidx+4)-lvy_r(lidz,lidy,lidx-3))+
           hc5*(lvy_r(lidz,lidy,lidx+5)-lvy_r(lidz,lidy,lidx-4)))/DH;
    
    vyy_r = (hc1*(lvy_r(lidz,lidy,lidx)  -lvy_r(lidz,lidy-1,lidx))+
           hc2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx))+
           hc3*(lvy_r(lidz,lidy+2,lidx)-lvy_r(lidz,lidy-3,lidx))+
           hc4*(lvy_r(lidz,lidy+3,lidx)-lvy_r(lidz,lidy-4,lidx))+
           hc5*(lvy_r(lidz,lidy+4,lidx)-lvy_r(lidz,lidy-5,lidx)))/DH;
    
    vyz_r = (hc1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx))+
           hc3*(lvy_r(lidz+3,lidy,lidx)-lvy_r(lidz-2,lidy,lidx))+
           hc4*(lvy_r(lidz+4,lidy,lidx)-lvy_r(lidz-3,lidy,lidx))+
           hc5*(lvy_r(lidz+5,lidy,lidx)-lvy_r(lidz-4,lidy,lidx)))/DH;
#elif fdoh==6
    vyx_r = (hc1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1))+
           hc3*(lvy_r(lidz,lidy,lidx+3)-lvy_r(lidz,lidy,lidx-2))+
           hc4*(lvy_r(lidz,lidy,lidx+4)-lvy_r(lidz,lidy,lidx-3))+
           hc5*(lvy_r(lidz,lidy,lidx+5)-lvy_r(lidz,lidy,lidx-4))+
           hc6*(lvy_r(lidz,lidy,lidx+6)-lvy_r(lidz,lidy,lidx-5)))/DH;
    
    vyy_r = (hc1*(lvy_r(lidz,lidy,lidx)  -lvy_r(lidz,lidy-1,lidx))+
           hc2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx))+
           hc3*(lvy_r(lidz,lidy+2,lidx)-lvy_r(lidz,lidy-3,lidx))+
           hc4*(lvy_r(lidz,lidy+3,lidx)-lvy_r(lidz,lidy-4,lidx))+
           hc5*(lvy_r(lidz,lidy+4,lidx)-lvy_r(lidz,lidy-5,lidx))+
           hc6*(lvy_r(lidz,lidy+5,lidx)-lvy_r(lidz,lidy-6,lidx)))/DH;
    
    vyz_r = (hc1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           hc2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx))+
           hc3*(lvy_r(lidz+3,lidy,lidx)-lvy_r(lidz-2,lidy,lidx))+
           hc4*(lvy_r(lidz+4,lidy,lidx)-lvy_r(lidz-3,lidy,lidx))+
           hc5*(lvy_r(lidz+5,lidy,lidx)-lvy_r(lidz-4,lidy,lidx))+
           hc6*(lvy_r(lidz+6,lidy,lidx)-lvy_r(lidz-5,lidy,lidx)))/DH;
#endif
    
    
    
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
        lvz_r(lidz,lidy,lidx)=vz_r(gidz, gidy, gidx);
        if (lidy<2*fdoh)
            lvz_r(lidz,lidy-fdoh,lidx)=vz_r(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lvz_r(lidz,lidy+lsizey-3*fdoh,lidx)=vz_r(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lvz_r(lidz,lidy+fdoh,lidx)=vz_r(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lvz_r(lidz,lidy-lsizey+3*fdoh,lidx)=vz_r(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lvz_r(lidz,lidy,lidx-fdoh)=vz_r(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvz_r(lidz,lidy,lidx+lsizex-3*fdoh)=vz_r(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvz_r(lidz,lidy,lidx+fdoh)=vz_r(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvz_r(lidz,lidy,lidx-lsizex+3*fdoh)=vz_r(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvz_r(lidz-fdoh,lidy,lidx)=vz_r(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvz_r(lidz+fdoh,lidy,lidx)=vz_r(gidz+fdoh,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
    
#if   fdoh==1
    vzx_r = (lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))/DH;
    vzy_r = (lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))/DH;
    vzz_r = (lvz_r(lidz,lidy,lidx)-lvz_r(lidz-1,lidy,lidx))/DH;
#elif fdoh==2
    
    vzx_r = (hc1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1)))/DH;
    
    vzy_r = (hc1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx)))/DH;
    
    vzz_r = (hc1*(lvz_r(lidz,lidy,lidx)  -lvz_r(lidz-1,lidy,lidx))+
           hc2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx)))/DH;
#elif fdoh==3
    vzx_r = (hc1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1))+
           hc3*(lvz_r(lidz,lidy,lidx+3)-lvz_r(lidz,lidy,lidx-2)))/DH;
    
    vzy_r = (hc1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx))+
           hc3*(lvz_r(lidz,lidy+3,lidx)-lvz_r(lidz,lidy-2,lidx)))/DH;
    
    vzz_r = (hc1*(lvz_r(lidz,lidy,lidx)-lvz_r(lidz-1,lidy,lidx))+
           hc2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx))+
           hc3*(lvz_r(lidz+2,lidy,lidx)-lvz_r(lidz-3,lidy,lidx)))/DH;
#elif fdoh==4
    vzx_r = (hc1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1))+
           hc3*(lvz_r(lidz,lidy,lidx+3)-lvz_r(lidz,lidy,lidx-2))+
           hc4*(lvz_r(lidz,lidy,lidx+4)-lvz_r(lidz,lidy,lidx-3)))/DH;
    
    vzy_r = (hc1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx))+
           hc3*(lvz_r(lidz,lidy+3,lidx)-lvz_r(lidz,lidy-2,lidx))+
           hc4*(lvz_r(lidz,lidy+4,lidx)-lvz_r(lidz,lidy-3,lidx)))/DH;
    
    vzz_r = (hc1*(lvz_r(lidz,lidy,lidx)  -lvz_r(lidz-1,lidy,lidx))+
           hc2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx))+
           hc3*(lvz_r(lidz+2,lidy,lidx)-lvz_r(lidz-3,lidy,lidx))+
           hc4*(lvz_r(lidz+3,lidy,lidx)-lvz_r(lidz-4,lidy,lidx)))/DH;
#elif fdoh==5
    vzx_r = (hc1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1))+
           hc3*(lvz_r(lidz,lidy,lidx+3)-lvz_r(lidz,lidy,lidx-2))+
           hc4*(lvz_r(lidz,lidy,lidx+4)-lvz_r(lidz,lidy,lidx-3))+
           hc5*(lvz_r(lidz,lidy,lidx+5)-lvz_r(lidz,lidy,lidx-4)))/DH;
    
    vzy_r = (hc1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx))+
           hc3*(lvz_r(lidz,lidy+3,lidx)-lvz_r(lidz,lidy-2,lidx))+
           hc4*(lvz_r(lidz,lidy+4,lidx)-lvz_r(lidz,lidy-3,lidx))+
           hc5*(lvz_r(lidz,lidy+5,lidx)-lvz_r(lidz,lidy-4,lidx)))/DH;
    
    vzz_r = (hc1*(lvz_r(lidz,lidy,lidx)  -lvz_r(lidz-1,lidy,lidx))+
           hc2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx))+
           hc3*(lvz_r(lidz+2,lidy,lidx)-lvz_r(lidz-3,lidy,lidx))+
           hc4*(lvz_r(lidz+3,lidy,lidx)-lvz_r(lidz-4,lidy,lidx))+
           hc5*(lvz_r(lidz+4,lidy,lidx)-lvz_r(lidz-5,lidy,lidx)))/DH;
#elif fdoh==6
    vzx_r = (hc1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1))+
           hc3*(lvz_r(lidz,lidy,lidx+3)-lvz_r(lidz,lidy,lidx-2))+
           hc4*(lvz_r(lidz,lidy,lidx+4)-lvz_r(lidz,lidy,lidx-3))+
           hc5*(lvz_r(lidz,lidy,lidx+5)-lvz_r(lidz,lidy,lidx-4))+
           hc6*(lvz_r(lidz,lidy,lidx+6)-lvz_r(lidz,lidy,lidx-5)))/DH;
    
    vzy_r = (hc1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           hc2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx))+
           hc3*(lvz_r(lidz,lidy+3,lidx)-lvz_r(lidz,lidy-2,lidx))+
           hc4*(lvz_r(lidz,lidy+4,lidx)-lvz_r(lidz,lidy-3,lidx))+
           hc5*(lvz_r(lidz,lidy+5,lidx)-lvz_r(lidz,lidy-4,lidx))+
           hc6*(lvz_r(lidz,lidy+6,lidx)-lvz_r(lidz,lidy-5,lidx)))/DH;
    
    vzz_r = (hc1*(lvz_r(lidz,lidy,lidx)  -lvz_r(lidz-1,lidy,lidx))+
           hc2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx))+
           hc3*(lvz_r(lidz+2,lidy,lidx)-lvz_r(lidz-3,lidy,lidx))+
           hc4*(lvz_r(lidz+3,lidy,lidx)-lvz_r(lidz-4,lidy,lidx))+
           hc5*(lvz_r(lidz+4,lidy,lidx)-lvz_r(lidz-5,lidy,lidx))+
           hc6*(lvz_r(lidz+5,lidy,lidx)-lvz_r(lidz-6,lidy,lidx)))/DH;
#endif
    }
    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if local_off==0
#if comm12==0
    if (gidy>(NY-fdoh-1) || gidz>(NZ-fdoh-1) || (gidx-offcomm)>(NX-fdoh-1-lcomm) ){
        return;
    }
    
#else
    if (gidy>(NY-fdoh-1) || gidz>(NZ-fdoh-1) ){
        return;
    }
#endif
#endif
    
 
    
// Read model parameters into local memory
#if Lve==0
    lpi=pi(gidz,gidy,gidx);
    lu=u(gidz,gidy,gidx);
    fipjp=uipjp(gidz,gidy,gidx)*DT;
    fjpkp=ujpkp(gidz,gidy,gidx)*DT;
    fipkp=uipkp(gidz,gidy,gidx)*DT;
    g=lpi*DT;
    f=2.0*lu*DT;

    
#else
    
    lpi=pi(gidz,gidy,gidx);
    lu=u(gidz,gidy,gidx);
    luipkp=uipkp(gidz,gidy,gidx);
    luipjp=uipjp(gidz,gidy,gidx);
    lujpkp=ujpkp(gidz,gidy,gidx);
    ltaup=taup(gidz,gidy,gidx);
    ltaus=taus(gidz,gidy,gidx);
    ltausipkp=tausipkp(gidz,gidy,gidx);
    ltausipjp=tausipjp(gidz,gidy,gidx);
    ltausjpkp=tausjpkp(gidz,gidy,gidx);
    
    for (l=0;l<Lve;l++){
        leta[l]=eta[l];
    }
    
    fipjp=luipjp*DT*(1.0+ (float)Lve*ltausipjp);
    fjpkp=lujpkp*DT*(1.0+ (float)Lve*ltausjpkp);
    fipkp=luipkp*DT*(1.0+ (float)Lve*ltausipkp);
    g=lpi*(1.0+(float)Lve*ltaup)*DT;
    f=2.0*lu*(1.0+(float)Lve*ltaus)*DT;
    dipjp=luipjp*ltausipjp;
    djpkp=lujpkp*ltausjpkp;
    dipkp=luipkp*ltausipkp;
    d=2.0*lu*ltaus;
    e=lpi*ltaup;
    
    
#endif
    
    
// Backpropagate the forward stresses
#if back_prop_type==1
    {
#if Lve==0

        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;
        
        float amp = psource(gidz, gidy, gidx+offset, nsrc, srcpos_loc, signals, nt);
        
        sxy(gidz,gidy,gidx)-=(fipjp*vxyyx);
        syz(gidz,gidy,gidx)-=(fjpkp*vyzzy);
        sxz(gidz,gidy,gidx)-=(fipkp*vxzzx);
        sxx(gidz,gidy,gidx)-=((g*vxxyyzz)-(f*vyyzz)) + amp;
        syy(gidz,gidy,gidx)-=((g*vxxyyzz)-(f*vxxzz)) + amp;
        szz(gidz,gidy,gidx)-=((g*vxxyyzz)-(f*vxxyy)) + amp;
        
// Backpropagation is not stable for viscoelastic wave equation
#else
        
    
        /* computing sums of the old memory variables */
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<Lve;l++){
            sumrxy+=rxy(gidz,gidy,gidx,l);
            sumryz+=ryz(gidz,gidy,gidx,l);
            sumrxz+=rxz(gidz,gidy,gidx,l);
            sumrxx+=rxx(gidz,gidy,gidx,l);
            sumryy+=ryy(gidz,gidy,gidx,l);
            sumrzz+=rzz(gidz,gidy,gidx,l);
        }
        
        /* updating components of the stress tensor, partially */
        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;
        
        lsxy=(fipjp*vxyyx)+(dt2*sumrxy);
        lsyz=(fjpkp*vyzzy)+(dt2*sumryz);
        lsxz=(fipkp*vxzzx)+(dt2*sumrxz);
        lsxx=((g*vxxyyzz)-(f*vyyzz))+(dt2*sumrxx);
        lsyy=((g*vxxyyzz)-(f*vxxzz))+(dt2*sumryy);
        lszz=((g*vxxyyzz)-(f*vxxyy))+(dt2*sumrzz);
        
        
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<Lve;l++){
            //Those variables change sign for reverse time
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            
            rxy(gidz,gidy,gidx,l)=b*(rxy(gidz,gidy,gidx,l)*c-leta[l]*(dipjp*vxyyx));
            ryz(gidz,gidy,gidx,l)=b*(ryz(gidz,gidy,gidx,l)*c-leta[l]*(djpkp*vyzzy));
            rxz(gidz,gidy,gidx,l)=b*(rxz(gidz,gidy,gidx,l)*c-leta[l]*(dipkp*vxzzx));
            rxx(gidz,gidy,gidx,l)=b*(rxx(gidz,gidy,gidx,l)*c-leta[l]*((e*vxxyyzz)-(d*vyyzz)));
            ryy(gidz,gidy,gidx,l)=b*(ryy(gidz,gidy,gidx,l)*c-leta[l]*((e*vxxyyzz)-(d*vxxzz)));
            rzz(gidz,gidy,gidx,l)=b*(rzz(gidz,gidy,gidx,l)*c-leta[l]*((e*vxxyyzz)-(d*vxxyy)));
            
            sumrxy+=rxy(gidz,gidy,gidx,l);
            sumryz+=ryz(gidz,gidy,gidx,l);
            sumrxz+=rxz(gidz,gidy,gidx,l);
            sumrxx+=rxx(gidz,gidy,gidx,l);
            sumryy+=ryy(gidz,gidy,gidx,l);
            sumrzz+=rzz(gidz,gidy,gidx,l);
        }
        
        float amp = psource(gidz, gidy, gidx+offset, nsrc, srcpos_loc, signals, nt);
        /* and now the components of the stress tensor are
         completely updated */
        sxy(gidz,gidy,gidx)-=lsxy+(dt2*sumrxy);
        syz(gidz,gidy,gidx)-=lsyz+(dt2*sumryz);
        sxz(gidz,gidy,gidx)-=lsxz+(dt2*sumrxz);
        sxx(gidz,gidy,gidx)-=lsxx+(dt2*sumrxx)+amp;
        syy(gidz,gidy,gidx)-=lsyy+(dt2*sumryy)+amp;
        szz(gidz,gidy,gidx)-=lszz+(dt2*sumrzz)+amp;
        
#endif
        
        m=evarm(gidz, gidy, gidx);
        if (m!=-1){
            sxx(gidz,gidy, gidx)= sxxbnd[m];
            syy(gidz,gidy, gidx)= syybnd[m];
            szz(gidz,gidy, gidx)= szzbnd[m];
            sxy(gidz,gidy, gidx)= sxybnd[m];
            syz(gidz,gidy, gidx)= syzbnd[m];
            sxz(gidz,gidy, gidx)= sxzbnd[m];
        }
        
    }
#endif
    
// Correct adjoint spatial derivatives to implement CPML
#if abs_type==1
    {
        int ind;
        
        if (gidz>NZ-nab-fdoh-1){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz - NZ+nab+fdoh+nab;
            ind=2*nab-1-k;
            
            psi_vxz(k,j,i) = b_z_half[ind] * psi_vxz(k,j,i) + a_z_half[ind] * vxz_r;
            vxz_r = vxz_r / K_z_half[ind] + psi_vxz(k,j,i);
            psi_vyz(k,j,i) = b_z_half[ind] * psi_vyz(k,j,i) + a_z_half[ind] * vyz_r;
            vyz_r = vyz_r / K_z_half[ind] + psi_vyz(k,j,i);
            psi_vzz(k,j,i) = b_z[ind+1] * psi_vzz(k,j,i) + a_z[ind+1] * vzz_r;
            vzz_r = vzz_r / K_z[ind+1] + psi_vzz(k,j,i);
            
        }
        
#if freesurf==0
        else if (gidz-fdoh<nab){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            
            psi_vxz(k,j,i) = b_z_half[k] * psi_vxz(k,j,i) + a_z_half[k] * vxz_r;
            vxz_r = vxz_r / K_z_half[k] + psi_vxz(k,j,i);
            psi_vyz(k,j,i) = b_z_half[k] * psi_vyz(k,j,i) + a_z_half[k] * vyz_r;
            vyz_r = vyz_r / K_z_half[k] + psi_vyz(k,j,i);
            psi_vzz(k,j,i) = b_z[k] * psi_vzz(k,j,i) + a_z[k] * vzz_r;
            vzz_r = vzz_r / K_z[k] + psi_vzz(k,j,i);
            
            
        }
#endif
        
        if (gidy-fdoh<nab){
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_vxy(k,j,i) = b_y_half[j] * psi_vxy(k,j,i) + a_y_half[j] * vxy_r;
            vxy_r = vxy_r / K_y_half[j] + psi_vxy(k,j,i);
            psi_vyy(k,j,i) = b_y[j] * psi_vyy(k,j,i) + a_y[j] * vyy_r;
            vyy_r = vyy_r / K_y[j] + psi_vyy(k,j,i);
            psi_vzy(k,j,i) = b_y_half[j] * psi_vzy(k,j,i) + a_y_half[j] * vzy_r;
            vzy_r = vzy_r / K_y_half[j] + psi_vzy(k,j,i);
            
        }
        
        else if (gidy>NY-nab-fdoh-1){
            
            i =gidx-fdoh;
            j =gidy - NY+nab+fdoh+nab;
            k =gidz-fdoh;
            ind=2*nab-1-j;
            
            
            psi_vxy(k,j,i) = b_y_half[ind] * psi_vxy(k,j,i) + a_y_half[ind] * vxy_r;
            vxy_r = vxy_r / K_y_half[ind] + psi_vxy(k,j,i);
            psi_vyy(k,j,i) = b_y[ind+1] * psi_vyy(k,j,i) + a_y[ind+1] * vyy_r;
            vyy_r = vyy_r / K_y[ind+1] + psi_vyy(k,j,i);
            psi_vzy(k,j,i) = b_y_half[ind] * psi_vzy(k,j,i) + a_y_half[ind] * vzy_r;
            vzy_r = vzy_r / K_y_half[ind] + psi_vzy(k,j,i);
            
            
        }
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_vxx(k,j,i) = b_x[i] * psi_vxx(k,j,i) + a_x[i] * vxx_r;
            vxx_r = vxx_r / K_x[i] + psi_vxx(k,j,i);
            psi_vyx(k,j,i) = b_x_half[i] * psi_vyx(k,j,i) + a_x_half[i] * vyx_r;
            vyx_r = vyx_r / K_x_half[i] + psi_vyx(k,j,i);
            psi_vzx(k,j,i) = b_x_half[i] * psi_vzx(k,j,i) + a_x_half[i] * vzx_r;
            vzx_r = vzx_r / K_x_half[i] + psi_vzx(k,j,i);
            
            
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            
            i =gidx - NX+nab+fdoh+nab;
            j =gidy-fdoh;
            k =gidz-fdoh;
            ind=2*nab-1-i;
            
            
            psi_vxx(k,j,i) = b_x[ind+1] * psi_vxx(k,j,i) + a_x[ind+1] * vxx_r;
            vxx_r = vxx_r /K_x[ind+1] + psi_vxx(k,j,i);
            psi_vyx(k,j,i) = b_x_half[ind] * psi_vyx(k,j,i) + a_x_half[ind] * vyx_r;
            vyx_r = vyx_r  /K_x_half[ind] + psi_vyx(k,j,i);
            psi_vzx(k,j,i) = b_x_half[ind] * psi_vzx(k,j,i) + a_x_half[ind] * vzx_r;
            vzx_r = vzx_r / K_x_half[ind]  +psi_vzx(k,j,i);
            
            
        }
#endif
        
    }
#endif

// Update adjoint stresses
    {
#if Lve==0
    
    vxyyx_r=vxy_r+vyx_r;
    vyzzy_r=vyz_r+vzy_r;
    vxzzx_r=vxz_r+vzx_r;
    vxxyyzz_r=vxx_r+vyy_r+vzz_r;
    vyyzz_r=vyy_r+vzz_r;
    vxxzz_r=vxx_r+vzz_r;
    vxxyy_r=vxx_r+vyy_r;
    
        lsxy=(fipjp*vxyyx_r);
        lsyz=(fjpkp*vyzzy_r);
        lsxz=(fipkp*vxzzx_r);
        lsxx=DT*((g*vxxyyzz_r)-(f*vyyzz_r));
        lsyy=DT*((g*vxxyyzz_r)-(f*vxxzz_r));
        lszz=DT*((g*vxxyyzz_r)-(f*vxxyy_r));
    
    sxy_r(gidz,gidy,gidx)+=lsxy;
    syz_r(gidz,gidy,gidx)+=lsyz;
    sxz_r(gidz,gidy,gidx)+=lsxz;
    sxx_r(gidz,gidy,gidx)+=lsxx;
    syy_r(gidz,gidy,gidx)+=lsyy;
    szz_r(gidz,gidy,gidx)+=lszz;
    
    
#else
    
    /* computing sums of the old memory variables */
    sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
    for (l=0;l<Lve;l++){
        sumrxy+=rxy_r(gidz,gidy,gidx,l);
        sumryz+=ryz_r(gidz,gidy,gidx,l);
        sumrxz+=rxz_r(gidz,gidy,gidx,l);
        sumrxx+=rxx_r(gidz,gidy,gidx,l);
        sumryy+=ryy_r(gidz,gidy,gidx,l);
        sumrzz+=rzz_r(gidz,gidy,gidx,l);
    }
    
    vxyyx_r=vxy_r+vyx_r;
    vyzzy_r=vyz_r+vzy_r;
    vxzzx_r=vxz_r+vzx_r;
    vxxyyzz_r=vxx_r+vyy_r+vzz_r;
    vyyzz_r=vyy_r+vzz_r;
    vxxzz_r=vxx_r+vzz_r;
    vxxyy_r=vxx_r+vyy_r;

    lsxy=(fipjp*vxyyx_r)+(dt2*sumrxy);
    lsyz=(fjpkp*vyzzy_r)+(dt2*sumryz);
    lsxz=(fipkp*vxzzx_r)+(dt2*sumrxz);
    lsxx=((g*vxxyyzz_r)-(f*vyyzz_r))+(dt2*sumrxx);
    lsyy=((g*vxxyyzz_r)-(f*vxxzz_r))+(dt2*sumryy);
    lszz=((g*vxxyyzz_r)-(f*vxxyy_r))+(dt2*sumrzz);
    
    sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
    for (l=0;l<Lve;l++){
        b=1.0/(1.0+(leta[l]*0.5));
        c=1.0-(leta[l]*0.5);
        
        rxy(gidz,gidy,gidx,l)=b*(rxy_r(gidz,gidy,gidx,l)*c-leta[l]*(dipjp*vxyyx_r));
        ryz(gidz,gidy,gidx,l)=b*(ryz_r(gidz,gidy,gidx,l)*c-leta[l]*(djpkp*vyzzy_r));
        rxz(gidz,gidy,gidx,l)=b*(rxz_r(gidz,gidy,gidx,l)*c-leta[l]*(dipkp*vxzzx_r));
        rxx(gidz,gidy,gidx,l)=b*(rxx_r(gidz,gidy,gidx,l)*c-leta[l]*((e*vxxyyzz_r)-(d*vyyzz_r)));
        ryy(gidz,gidy,gidx,l)=b*(ryy_r(gidz,gidy,gidx,l)*c-leta[l]*((e*vxxyyzz_r)-(d*vxxzz_r)));
        rzz(gidz,gidy,gidx,l)=b*(rzz_r(gidz,gidy,gidx,l)*c-leta[l]*((e*vxxyyzz_r)-(d*vxxyy_r)));
        
        sumrxy=rxy_r(gidz,gidy,gidx,l);
        sumryz=ryz_r(gidz,gidy,gidx,l);
        sumrxz=rxz_r(gidz,gidy,gidx,l);
        sumrxx=rxx_r(gidz,gidy,gidx,l);
        sumryy=ryy_r(gidz,gidy,gidx,l);
        sumrzz=rzz_r(gidz,gidy,gidx,l);
    }

    /* and now the components of the stress tensor are
     completely updated */
    sxy_r(gidz,gidy,gidx)+=lsxy+(dt2*sumrxy);
    syz_r(gidz,gidy,gidx)+=lsyz+(dt2*sumryz);
    sxz_r(gidz,gidy,gidx)+=lsxz+(dt2*sumrxz);
    sxx_r(gidz,gidy,gidx)+=lsxx+(dt2*sumrxx);
    syy_r(gidz,gidy,gidx)+=lsyy+(dt2*sumryy);
    szz_r(gidz,gidy,gidx)+=lszz+(dt2*sumrzz);
    
#endif
}

    
// Absorbing boundary
#if abs_type==2
    {
        if (gidz-fdoh<nab){
            sxy_r(gidz,gidy,gidx)*=taper[gidz-fdoh];
            syz_r(gidz,gidy,gidx)*=taper[gidz-fdoh];
            sxz_r(gidz,gidy,gidx)*=taper[gidz-fdoh];
            sxx_r(gidz,gidy,gidx)*=taper[gidz-fdoh];
            syy_r(gidz,gidy,gidx)*=taper[gidz-fdoh];
            szz_r(gidz,gidy,gidx)*=taper[gidz-fdoh];
        }
        
        if (gidz>NZ-nab-fdoh-1){
            sxy_r(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            syz_r(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            sxz_r(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            sxx_r(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            syy_r(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            szz_r(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
        }
        if (gidy-fdoh<nab){
            sxy_r(gidz,gidy,gidx)*=taper[gidy-fdoh];
            syz_r(gidz,gidy,gidx)*=taper[gidy-fdoh];
            sxz_r(gidz,gidy,gidx)*=taper[gidy-fdoh];
            sxx_r(gidz,gidy,gidx)*=taper[gidy-fdoh];
            syy_r(gidz,gidy,gidx)*=taper[gidy-fdoh];
            szz_r(gidz,gidy,gidx)*=taper[gidy-fdoh];
        }
        
        if (gidy>NY-nab-fdoh-1){
            sxy_r(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            syz_r(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            sxz_r(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            sxx_r(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            syy_r(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            szz_r(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
        }
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            sxy_r(gidz,gidy,gidx)*=taper[gidx-fdoh];
            syz_r(gidz,gidy,gidx)*=taper[gidx-fdoh];
            sxz_r(gidz,gidy,gidx)*=taper[gidx-fdoh];
            sxx_r(gidz,gidy,gidx)*=taper[gidx-fdoh];
            syy_r(gidz,gidy,gidx)*=taper[gidx-fdoh];
            szz_r(gidz,gidy,gidx)*=taper[gidx-fdoh];
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            sxy_r(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            syz_r(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            sxz_r(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            sxx_r(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            syy_r(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            szz_r(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
        }
#endif
    }
#endif
    
// Shear wave modulus and P-wave modulus gradient calculation on the fly    
#if back_prop_type==1
                 float c1=1.0/pown(3.0*lpi-4.0*lu,2);
                 float c3=1.0/pown(lu,2);
                 float c5=1.0/6.0*c3;
                 
                 float dM=c1*( sxx(gidz,gidy,gidx)+syy(gidz,gidy,gidx)+szz(gidz,gidy,gidx) )*( lsxx+lsyy+lszz );
                 
                 gradM(gidz,gidy,gidx)+=dM;
                 gradmu(gidz,gidy,gidx)+=c3*(sxz(gidz,gidy,gidx)*lsxz +sxy(gidz,gidy,gidx)*lsxy +syz(gidz,gidy,gidx)*lsyz )
                                         - 4.0/3*dM
                                         +c5*(  lsxx*(2.0*sxx(gidz,gidy,gidx)- syy(gidz,gidy,gidx)-szz(gidz,gidy,gidx) )
                                               +lsyy*(2.0*syy(gidz,gidy,gidx)- sxx(gidz,gidy,gidx)-szz(gidz,gidy,gidx) )
                                               +lszz*(2.0*szz(gidz,gidy,gidx)- sxx(gidz,gidy,gidx)-syy(gidz,gidy,gidx) )
                                              );
#endif
    
#if gradsrcout==1
    float pressure;
    if (nsrc>0){
        
        for (int srci=0; srci<nsrc; srci++){
            
            int SOURCE_TYPE= (int)srcpos_loc(4,srci);
            
            if (SOURCE_TYPE==1){
                int i=(int)(srcpos_loc(0,srci)/DH-0.5)+fdoh;
                int j=(int)(srcpos_loc(1,srci)/DH-0.5)+fdoh;
                int k=(int)(srcpos_loc(2,srci)/DH-0.5)+fdoh;
                
                
                if (i==gidx && j==gidy && k==gidz){
                    
                    pressure=(sxx_r(gidz,gidy,gidx)+syy_r(gidz,gidy,gidx)+szz_r(gidz,gidy,gidx) )/(2.0*DH*DH*DH);
                    if ( (nt>0) && (nt< NT ) ){
                        gradsrc(srci,nt+1)+=pressure;
                        gradsrc(srci,nt-1)-=pressure;
                    }
                    else if (nt==0)
                        gradsrc(srci,nt+1)+=pressure;
                    else if (nt==NT)
                        gradsrc(srci,nt-1)-=pressure;

                }
            }
        }
    }
    
#endif


}

