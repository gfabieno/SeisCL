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

#define rho(z,y,x)             rho[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,y,x)             rip[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,y,x)             rjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,y,x)             rkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipjp(z,y,x)         muipjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define mujpkp(z,y,x)         mujpkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipkp(z,y,x)         muipkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define mu(z,y,x)                 mu[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define M(z,y,x)               M[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,y,x)     gradrho[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,y,x)         gradM[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradmu(z,y,x)       gradmu[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,y,x)   gradtaup[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaus(z,y,x)   gradtaus[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define Hrho(z,y,x)     Hrho[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define HM(z,y,x)         HM[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define Hmu(z,y,x)       Hmu[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define Htaup(z,y,x)   Htaup[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define Htaus(z,y,x)   Htaus[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define taus(z,y,x)         taus[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipjp(z,y,x) tausipjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausjpkp(z,y,x) tausjpkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipkp(z,y,x) tausipkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define taup(z,y,x)         taup[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define vx(z,y,x)   vx[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vy(z,y,x)   vy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vz(z,y,x)   vz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxx(z,y,x) sxx[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syy(z,y,x) syy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define szz(z,y,x) szz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxy(z,y,x) sxy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syz(z,y,x) syz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxz(z,y,x) sxz[(x)*NY*(NZ)+(y)*(NZ)+(z)]

#define rxx(z,y,x,l) rxx[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryy(z,y,x,l) ryy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rzz(z,y,x,l) rzz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxy(z,y,x,l) rxy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryz(z,y,x,l) ryz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxz(z,y,x,l) rxz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]

#define vx_r(z,y,x)   vx_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vy_r(z,y,x)   vy_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vz_r(z,y,x)   vz_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxx_r(z,y,x) sxx_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syy_r(z,y,x) syy_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define szz_r(z,y,x) szz_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxy_r(z,y,x) sxy_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syz_r(z,y,x) syz_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxz_r(z,y,x) sxz_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]

#define rxx_r(z,y,x,l) rxx_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryy_r(z,y,x,l) ryy_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rzz_r(z,y,x,l) rzz_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxy_r(z,y,x,l) rxy_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryz_r(z,y,x,l) ryz_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxz_r(z,y,x,l) rxz_r[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]

#define psi_vx_x(z,y,x) psi_vx_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vy_x(z,y,x) psi_vy_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vz_x(z,y,x) psi_vz_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vx_y(z,y,x) psi_vx_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vy_y(z,y,x) psi_vy_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vz_y(z,y,x) psi_vz_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vx_z(z,y,x) psi_vx_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vy_z(z,y,x) psi_vy_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vz_z(z,y,x) psi_vz_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#if LOCAL_OFF==0

#define lvar(z,y,x)   lvar[(x)*lsizey*lsizez+(y)*lsizez+(z)]


#endif



#define PI (3.141592653589793238462643383279502884197169)

#define signals(y,x) signals[(y)*NT+(x)]
#define rec_pos(y,x) rec_pos[(y)*8+(x)]

#define gradsrc(y,x) gradsrc[(y)*NT+(x)]



// Find boundary indice for boundary injection in backpropagation
int evarm( int k, int j, int i){
    
    
#if NUM_DEVICES==1 & NLOCALP==1
    
    int NXbnd = (NX-2*FDOH-2*NAB);
    int NYbnd = (NY-2*FDOH-2*NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH- 2*NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH- NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    
    int m=-1;
    i-=lbnd;
    j-=lbnd;
    k-=lbnds;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) && (j>FDOH-1 && j<NYbnd-FDOH) && (i>FDOH-1 && i<NXbnd-FDOH) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<FDOH){//front
        m=i*NYbnd*NZbnd+j*NZbnd+k;
    }
    else if (i>NXbnd-1-FDOH){//back
        i=i-NXbnd+FDOH;
        m=NYbnd*NZbnd*FDOH+i*NYbnd*NZbnd+j*NZbnd+k;
    }
    else if (j<FDOH){//left
        i=i-FDOH;
        m=NYbnd*NZbnd*FDOH*2+i*FDOH*NZbnd+j*NZbnd+k;
    }
    else if (j>NYbnd-1-FDOH){//right
        i=i-FDOH;
        j=j-NYbnd+FDOH;
        m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
    }
    else if (k<FDOH){//up
        i=i-FDOH;
        j=j-FDOH;
        m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH*2+(NXbnd-2*FDOH)*(NYbnd-2*FDOH)*FDOH+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
        
        
    }
    else {//down
        i=i-FDOH;
        j=j-FDOH;
        k=k-NZbnd+FDOH;
        m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    
    
    
#elif DEVID==0 & MYGROUPID==0
    int NXbnd = (NX-2*FDOH-NAB);
    int NYbnd = (NY-2*FDOH-2*NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH- 2*NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH- NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    
    int m=-1;
    i-=lbnd;
    j-=lbnd;
    k-=lbnds;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) && (j>FDOH-1 && j<NYbnd-FDOH) && i>FDOH-1  )
        m=-1;
    else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<FDOH){//front
        m=i*NYbnd*NZbnd+j*NZbnd+k;
    }
    else if (j<FDOH){//left
        i=i-FDOH;
        m=NYbnd*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
    }
    else if (j>NYbnd-1-FDOH){//right
        i=i-FDOH;
        j=j-NYbnd+FDOH;
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
    }
    else if (k<FDOH){//up
        i=i-FDOH;
        j=j-FDOH;
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+(NXbnd-FDOH)*(NYbnd-2*FDOH)*FDOH+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
        
    }
    else {//down
        i=i-FDOH;
        j=j-FDOH;
        k=k-NZbnd+FDOH;
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
#elif DEVID==NUM_DEVICES-1 & MYGROUPID==NLOCALP-1
    int NXbnd = (NX-2*FDOH-NAB);
    int NYbnd = (NY-2*FDOH-2*NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH- 2*NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH- NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    
    int m=-1;
    i-=FDOH;
    j-=lbnd;
    k-=lbnds;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) && (j>FDOH-1 && j<NYbnd-FDOH) && i<NXbnd-FDOH )
        m=-1;
    else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1  || i>NXbnd-1 )
        m=-1;
    else if (i>NXbnd-1-FDOH){//back
        i=i-NXbnd+FDOH;
        m=i*NYbnd*NZbnd+j*NZbnd+k;
    }
    else if (j<FDOH){//left
        m=NYbnd*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
    }
    else if (j>NYbnd-1-FDOH){//right
        j=j-NYbnd+FDOH;
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
    }
    else if (k<FDOH){//up
        j=j-FDOH;
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+(NXbnd-FDOH)*(NYbnd-2*FDOH)*FDOH+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
        
    }
    else {//down
        j=j-FDOH;
        k=k-NZbnd+FDOH;
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    
#else
    int NXbnd = (NX-2*FDOH);
    int NYbnd = (NY-2*FDOH-2*NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH- 2*NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH- NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    
    int m=-1;
    i-=FDOH;
    j-=lbnd;
    k-=lbnds;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) && (j>FDOH-1 && j<NYbnd-FDOH) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1  || i<0 || i>NXbnd-1 )
        m=-1;
    else if (j<FDOH){//left
        m=i*FDOH*NZbnd+j*NZbnd+k;
    }
    else if (j>NYbnd-1-FDOH){//right
        j=j-NYbnd+FDOH;
        m=NXbnd*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
    }
    else if (k<FDOH){//up
        j=j-FDOH;
        m=NXbnd*NZbnd*FDOH*2+NXbnd*(NYbnd-2*FDOH)*FDOH+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
        
    }
    else {//down
        j=j-FDOH;
        k=k-NZbnd+FDOH;
        m=NXbnd*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    
#endif
    
    
    return m;
    
}

__kernel void update_adjs(int offcomm, 
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
                          __global float *M,         __global float *mu,        __global float *muipjp,
                          __global float *mujpkp,      __global float *muipkp,
                          __global float *taus,       __global float *tausipjp, __global float *tausjpkp,
                          __global float *tausipkp,   __global float *taup,     __global float *eta,
                          __global float *taper,
                          __global float *K_x,        __global float *a_x,      __global float *b_x,
                          __global float *K_x_half,   __global float *a_x_half, __global float *b_x_half,
                          __global float *K_y,        __global float *a_y,      __global float *b_y,
                          __global float *K_y_half,   __global float *a_y_half, __global float *b_y_half,
                          __global float *K_z,        __global float *a_z,      __global float *b_z,
                          __global float *K_z_half,   __global float *a_z_half, __global float *b_z_half,
                          __global float *psi_vx_x,    __global float *psi_vx_y,  __global float *psi_vx_z,
                          __global float *psi_vy_x,    __global float *psi_vy_y,  __global float *psi_vy_z,
                          __global float *psi_vz_x,    __global float *psi_vz_y,  __global float *psi_vz_z,
                          __global float *gradrho,    __global float *gradM,    __global float *gradmu,
                          __global float *gradtaup,   __global float *gradtaus, __global float *gradsrc,
                          __global float *Hrho,    __global float *HM,     __global float *Hmu,
                          __global float *Htaup,   __global float *Htaus,  __global float *Hsrc,
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
    float lM, lmu, lmuipjp, lmuipkp, lmujpkp,ltaup, ltaus, ltausipjp, ltausipkp, ltausjpkp;
    float leta[LVE];
    float lsxx, lsyy, lszz, lsxy, lsxz, lsyz;
    
    
    
    
// If we use local memory
#if LOCAL_OFF==0
    int lsizez = get_local_size(0)+2*FDOH;
    int lsizey = get_local_size(1)+2*FDOH;
    int lsizex = get_local_size(2)+2*FDOH;
    int lidz = get_local_id(0)+FDOH;
    int lidy = get_local_id(1)+FDOH;
    int lidx = get_local_id(2)+FDOH;
    int gidz = get_global_id(0)+FDOH;
    int gidy = get_global_id(1)+FDOH;
    int gidx = get_global_id(2)+FDOH+offcomm;

#define lvx lvar
#define lvy lvar
#define lvz lvar
#define lvx_r lvar
#define lvy_r lvar
#define lvz_r lvar

// If local memory is turned off
#elif LOCAL_OFF==1
    int gid = get_global_id(0);
    int glsizez = (NZ-2*FDOH);
    int glsizey = (NY-2*FDOH);
    int gidz = gid%glsizez+FDOH;
    int gidy = (gid/glsizez)%glsizey+FDOH;
    int gidx = gid/(glsizez*glsizey)+FDOH+offcomm;
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
#if BACK_PROP_TYPE==1
    {
#if LOCAL_OFF==0
        lvx(lidz,lidy,lidx)=vx(gidz, gidy, gidx);
        if (lidy<2*FDOH)
            lvx(lidz,lidy-FDOH,lidx)=vx(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvx(lidz,lidy+lsizey-3*FDOH,lidx)=vx(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvx(lidz,lidy+FDOH,lidx)=vx(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvx(lidz,lidy-lsizey+3*FDOH,lidx)=vx(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvx(lidz,lidy,lidx-FDOH)=vx(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvx(lidz,lidy,lidx+lsizex-3*FDOH)=vx(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvx(lidz,lidy,lidx+FDOH)=vx(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvx(lidz,lidy,lidx-lsizex+3*FDOH)=vx(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvx(lidz-FDOH,lidy,lidx)=vx(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvx(lidz+FDOH,lidy,lidx)=vx(gidz+FDOH,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH==1
    vxx = (lvx(lidz,lidy,lidx)-lvx(lidz,lidy,lidx-1));
    vxy = (lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx));
    vxz = (lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx));
#elif FDOH==2
    vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2)));
    
    vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx)));
    
    vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx)));
#elif FDOH==3
    vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
           HC3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3)));
    
    vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
           HC3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx)));
    
    vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
           HC3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx)));
#elif FDOH==4
    vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
           HC3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
           HC4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4)));
    
    vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
           HC3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
           HC4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx)));
    
    vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
           HC3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
           HC4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx)));
#elif FDOH==5
    vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
           HC3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
           HC4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4))+
           HC5*(lvx(lidz,lidy,lidx+4)-lvx(lidz,lidy,lidx-5)));
    
    vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
           HC3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
           HC4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx))+
           HC5*(lvx(lidz,lidy+5,lidx)-lvx(lidz,lidy-4,lidx)));
    
    vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
           HC3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
           HC4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx))+
           HC5*(lvx(lidz+5,lidy,lidx)-lvx(lidz-4,lidy,lidx)));
#elif FDOH==6
    vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
           HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
           HC3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
           HC4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4))+
           HC5*(lvx(lidz,lidy,lidx+4)-lvx(lidz,lidy,lidx-5))+
           HC6*(lvx(lidz,lidy,lidx+5)-lvx(lidz,lidy,lidx-6)));
    
    vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
           HC3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
           HC4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx))+
           HC5*(lvx(lidz,lidy+5,lidx)-lvx(lidz,lidy-4,lidx))+
           HC6*(lvx(lidz,lidy+6,lidx)-lvx(lidz,lidy-5,lidx)));
    
    vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
           HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
           HC3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
           HC4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx))+
           HC5*(lvx(lidz+5,lidy,lidx)-lvx(lidz-4,lidy,lidx))+
           HC6*(lvx(lidz+6,lidy,lidx)-lvx(lidz-5,lidy,lidx)));
#endif
    
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
        lvy(lidz,lidy,lidx)=vy(gidz, gidy, gidx);
        if (lidy<2*FDOH)
            lvy(lidz,lidy-FDOH,lidx)=vy(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvy(lidz,lidy+lsizey-3*FDOH,lidx)=vy(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvy(lidz,lidy+FDOH,lidx)=vy(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvy(lidz,lidy-lsizey+3*FDOH,lidx)=vy(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvy(lidz,lidy,lidx-FDOH)=vy(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvy(lidz,lidy,lidx+lsizex-3*FDOH)=vy(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvy(lidz,lidy,lidx+FDOH)=vy(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvy(lidz,lidy,lidx-lsizex+3*FDOH)=vy(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvy(lidz-FDOH,lidy,lidx)=vy(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvy(lidz+FDOH,lidy,lidx)=vy(gidz+FDOH,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH==1
    vyx = (lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx));
    vyy = (lvy(lidz,lidy,lidx)-lvy(lidz,lidy-1,lidx));
    vyz = (lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx));
#elif FDOH==2
    vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1)));
    
    vyy = (HC1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
           HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx)));
    
    vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx)));
#elif FDOH==3
    vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
           HC3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2)));
    
    vyy = (HC1*(lvy(lidz,lidy,lidx)-lvy(lidz,lidy-1,lidx))+
           HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
           HC3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx)));
    
    vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
           HC3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx)));
#elif FDOH==4
    vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
           HC3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
           HC4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3)));
    
    vyy = (HC1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
           HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
           HC3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
           HC4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx)));
    
    vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
           HC3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
           HC4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx)));
#elif FDOH==5
    vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
           HC3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
           HC4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3))+
           HC5*(lvy(lidz,lidy,lidx+5)-lvy(lidz,lidy,lidx-4)));
    
    vyy = (HC1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
           HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
           HC3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
           HC4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx))+
           HC5*(lvy(lidz,lidy+4,lidx)-lvy(lidz,lidy-5,lidx)));
    
    vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
           HC3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
           HC4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx))+
           HC5*(lvy(lidz+5,lidy,lidx)-lvy(lidz-4,lidy,lidx)));
#elif FDOH==6
    vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
           HC3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
           HC4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3))+
           HC5*(lvy(lidz,lidy,lidx+5)-lvy(lidz,lidy,lidx-4))+
           HC6*(lvy(lidz,lidy,lidx+6)-lvy(lidz,lidy,lidx-5)));
    
    vyy = (HC1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
           HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
           HC3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
           HC4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx))+
           HC5*(lvy(lidz,lidy+4,lidx)-lvy(lidz,lidy-5,lidx))+
           HC6*(lvy(lidz,lidy+5,lidx)-lvy(lidz,lidy-6,lidx)));
    
    vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
           HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
           HC3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
           HC4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx))+
           HC5*(lvy(lidz+5,lidy,lidx)-lvy(lidz-4,lidy,lidx))+
           HC6*(lvy(lidz+6,lidy,lidx)-lvy(lidz-5,lidy,lidx)));
#endif
    
    
    
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
        lvz(lidz,lidy,lidx)=vz(gidz, gidy, gidx);
        if (lidy<2*FDOH)
            lvz(lidz,lidy-FDOH,lidx)=vz(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvz(lidz,lidy+lsizey-3*FDOH,lidx)=vz(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvz(lidz,lidy+FDOH,lidx)=vz(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvz(lidz,lidy-lsizey+3*FDOH,lidx)=vz(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvz(lidz,lidy,lidx-FDOH)=vz(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvz(lidz,lidy,lidx+lsizex-3*FDOH)=vz(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvz(lidz,lidy,lidx+FDOH)=vz(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvz(lidz,lidy,lidx-lsizex+3*FDOH)=vz(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvz(lidz-FDOH,lidy,lidx)=vz(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvz(lidz+FDOH,lidy,lidx)=vz(gidz+FDOH,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
    
#if   FDOH==1
    vzx = (lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx));
    vzy = (lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx));
    vzz = (lvz(lidz,lidy,lidx)-lvz(lidz-1,lidy,lidx));
#elif FDOH==2
    
    vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1)));
    
    vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx)));
    
    vzz = (HC1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
           HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx)));
#elif FDOH==3
    vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
           HC3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2)));
    
    vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
           HC3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx)));
    
    vzz = (HC1*(lvz(lidz,lidy,lidx)-lvz(lidz-1,lidy,lidx))+
           HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
           HC3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx)));
#elif FDOH==4
    vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
           HC3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
           HC4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3)));
    
    vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
           HC3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
           HC4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx)));
    
    vzz = (HC1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
           HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
           HC3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
           HC4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx)));
#elif FDOH==5
    vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
           HC3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
           HC4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3))+
           HC5*(lvz(lidz,lidy,lidx+5)-lvz(lidz,lidy,lidx-4)));
    
    vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
           HC3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
           HC4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx))+
           HC5*(lvz(lidz,lidy+5,lidx)-lvz(lidz,lidy-4,lidx)));
    
    vzz = (HC1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
           HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
           HC3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
           HC4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx))+
           HC5*(lvz(lidz+4,lidy,lidx)-lvz(lidz-5,lidy,lidx)));
#elif FDOH==6
    vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
           HC3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
           HC4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3))+
           HC5*(lvz(lidz,lidy,lidx+5)-lvz(lidz,lidy,lidx-4))+
           HC6*(lvz(lidz,lidy,lidx+6)-lvz(lidz,lidy,lidx-5)));
    
    vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
           HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
           HC3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
           HC4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx))+
           HC5*(lvz(lidz,lidy+5,lidx)-lvz(lidz,lidy-4,lidx))+
           HC6*(lvz(lidz,lidy+6,lidx)-lvz(lidz,lidy-5,lidx)));
    
    vzz = (HC1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
           HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
           HC3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
           HC4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx))+
           HC5*(lvz(lidz+4,lidy,lidx)-lvz(lidz-5,lidy,lidx))+
           HC6*(lvz(lidz+5,lidy,lidx)-lvz(lidz-6,lidy,lidx)));
#endif

        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif

// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
#if LOCAL_OFF==0
        lvx_r(lidz,lidy,lidx)=vx_r(gidz, gidy, gidx);
        if (lidy<2*FDOH)
            lvx_r(lidz,lidy-FDOH,lidx)=vx_r(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvx_r(lidz,lidy+lsizey-3*FDOH,lidx)=vx_r(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvx_r(lidz,lidy+FDOH,lidx)=vx_r(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvx_r(lidz,lidy-lsizey+3*FDOH,lidx)=vx_r(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvx_r(lidz,lidy,lidx-FDOH)=vx_r(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvx_r(lidz,lidy,lidx+lsizex-3*FDOH)=vx_r(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvx_r(lidz,lidy,lidx+FDOH)=vx_r(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvx_r(lidz,lidy,lidx-lsizex+3*FDOH)=vx_r(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvx_r(lidz-FDOH,lidy,lidx)=vx_r(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvx_r(lidz+FDOH,lidy,lidx)=vx_r(gidz+FDOH,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH==1
    vxx_r = (lvx_r(lidz,lidy,lidx)-lvx_r(lidz,lidy,lidx-1));
    vxy_r = (lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx));
    vxz_r = (lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx));
#elif FDOH==2
    vxx_r = (HC1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           HC2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2)));
    
    vxy_r = (HC1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx)));
    
    vxz_r = (HC1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx)));
#elif FDOH==3
    vxx_r = (HC1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           HC2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2))+
           HC3*(lvx_r(lidz,lidy,lidx+2)-lvx_r(lidz,lidy,lidx-3)));
    
    vxy_r = (HC1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx))+
           HC3*(lvx_r(lidz,lidy+3,lidx)-lvx_r(lidz,lidy-2,lidx)));
    
    vxz_r = (HC1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx))+
           HC3*(lvx_r(lidz+3,lidy,lidx)-lvx_r(lidz-2,lidy,lidx)));
#elif FDOH==4
    vxx_r = (HC1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           HC2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2))+
           HC3*(lvx_r(lidz,lidy,lidx+2)-lvx_r(lidz,lidy,lidx-3))+
           HC4*(lvx_r(lidz,lidy,lidx+3)-lvx_r(lidz,lidy,lidx-4)));
    
    vxy_r = (HC1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx))+
           HC3*(lvx_r(lidz,lidy+3,lidx)-lvx_r(lidz,lidy-2,lidx))+
           HC4*(lvx_r(lidz,lidy+4,lidx)-lvx_r(lidz,lidy-3,lidx)));
    
    vxz_r = (HC1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx))+
           HC3*(lvx_r(lidz+3,lidy,lidx)-lvx_r(lidz-2,lidy,lidx))+
           HC4*(lvx_r(lidz+4,lidy,lidx)-lvx_r(lidz-3,lidy,lidx)));
#elif FDOH==5
    vxx_r = (HC1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           HC2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2))+
           HC3*(lvx_r(lidz,lidy,lidx+2)-lvx_r(lidz,lidy,lidx-3))+
           HC4*(lvx_r(lidz,lidy,lidx+3)-lvx_r(lidz,lidy,lidx-4))+
           HC5*(lvx_r(lidz,lidy,lidx+4)-lvx_r(lidz,lidy,lidx-5)));
    
    vxy_r = (HC1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx))+
           HC3*(lvx_r(lidz,lidy+3,lidx)-lvx_r(lidz,lidy-2,lidx))+
           HC4*(lvx_r(lidz,lidy+4,lidx)-lvx_r(lidz,lidy-3,lidx))+
           HC5*(lvx_r(lidz,lidy+5,lidx)-lvx_r(lidz,lidy-4,lidx)));
    
    vxz_r = (HC1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx))+
           HC3*(lvx_r(lidz+3,lidy,lidx)-lvx_r(lidz-2,lidy,lidx))+
           HC4*(lvx_r(lidz+4,lidy,lidx)-lvx_r(lidz-3,lidy,lidx))+
           HC5*(lvx_r(lidz+5,lidy,lidx)-lvx_r(lidz-4,lidy,lidx)));
#elif FDOH==6
    vxx_r = (HC1*(lvx_r(lidz,lidy,lidx)  -lvx_r(lidz,lidy,lidx-1))+
           HC2*(lvx_r(lidz,lidy,lidx+1)-lvx_r(lidz,lidy,lidx-2))+
           HC3*(lvx_r(lidz,lidy,lidx+2)-lvx_r(lidz,lidy,lidx-3))+
           HC4*(lvx_r(lidz,lidy,lidx+3)-lvx_r(lidz,lidy,lidx-4))+
           HC5*(lvx_r(lidz,lidy,lidx+4)-lvx_r(lidz,lidy,lidx-5))+
           HC6*(lvx_r(lidz,lidy,lidx+5)-lvx_r(lidz,lidy,lidx-6)));
    
    vxy_r = (HC1*(lvx_r(lidz,lidy+1,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz,lidy+2,lidx)-lvx_r(lidz,lidy-1,lidx))+
           HC3*(lvx_r(lidz,lidy+3,lidx)-lvx_r(lidz,lidy-2,lidx))+
           HC4*(lvx_r(lidz,lidy+4,lidx)-lvx_r(lidz,lidy-3,lidx))+
           HC5*(lvx_r(lidz,lidy+5,lidx)-lvx_r(lidz,lidy-4,lidx))+
           HC6*(lvx_r(lidz,lidy+6,lidx)-lvx_r(lidz,lidy-5,lidx)));
    
    vxz_r = (HC1*(lvx_r(lidz+1,lidy,lidx)-lvx_r(lidz,lidy,lidx))+
           HC2*(lvx_r(lidz+2,lidy,lidx)-lvx_r(lidz-1,lidy,lidx))+
           HC3*(lvx_r(lidz+3,lidy,lidx)-lvx_r(lidz-2,lidy,lidx))+
           HC4*(lvx_r(lidz+4,lidy,lidx)-lvx_r(lidz-3,lidy,lidx))+
           HC5*(lvx_r(lidz+5,lidy,lidx)-lvx_r(lidz-4,lidy,lidx))+
           HC6*(lvx_r(lidz+6,lidy,lidx)-lvx_r(lidz-5,lidy,lidx)));
#endif
    
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
        lvy_r(lidz,lidy,lidx)=vy_r(gidz, gidy, gidx);
        if (lidy<2*FDOH)
            lvy_r(lidz,lidy-FDOH,lidx)=vy_r(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvy_r(lidz,lidy+lsizey-3*FDOH,lidx)=vy_r(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvy_r(lidz,lidy+FDOH,lidx)=vy_r(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvy_r(lidz,lidy-lsizey+3*FDOH,lidx)=vy_r(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvy_r(lidz,lidy,lidx-FDOH)=vy_r(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvy_r(lidz,lidy,lidx+lsizex-3*FDOH)=vy_r(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvy_r(lidz,lidy,lidx+FDOH)=vy_r(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvy_r(lidz,lidy,lidx-lsizex+3*FDOH)=vy_r(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvy_r(lidz-FDOH,lidy,lidx)=vy_r(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvy_r(lidz+FDOH,lidy,lidx)=vy_r(gidz+FDOH,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH==1
    vyx_r = (lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx));
    vyy_r = (lvy_r(lidz,lidy,lidx)-lvy_r(lidz,lidy-1,lidx));
    vyz_r = (lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx));
#elif FDOH==2
    vyx_r = (HC1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1)));
    
    vyy_r = (HC1*(lvy_r(lidz,lidy,lidx)  -lvy_r(lidz,lidy-1,lidx))+
           HC2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx)));
    
    vyz_r = (HC1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx)));
#elif FDOH==3
    vyx_r = (HC1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1))+
           HC3*(lvy_r(lidz,lidy,lidx+3)-lvy_r(lidz,lidy,lidx-2)));
    
    vyy_r = (HC1*(lvy_r(lidz,lidy,lidx)-lvy_r(lidz,lidy-1,lidx))+
           HC2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx))+
           HC3*(lvy_r(lidz,lidy+2,lidx)-lvy_r(lidz,lidy-3,lidx)));
    
    vyz_r = (HC1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx))+
           HC3*(lvy_r(lidz+3,lidy,lidx)-lvy_r(lidz-2,lidy,lidx)));
#elif FDOH==4
    vyx_r = (HC1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1))+
           HC3*(lvy_r(lidz,lidy,lidx+3)-lvy_r(lidz,lidy,lidx-2))+
           HC4*(lvy_r(lidz,lidy,lidx+4)-lvy_r(lidz,lidy,lidx-3)));
    
    vyy_r = (HC1*(lvy_r(lidz,lidy,lidx)  -lvy_r(lidz,lidy-1,lidx))+
           HC2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx))+
           HC3*(lvy_r(lidz,lidy+2,lidx)-lvy_r(lidz,lidy-3,lidx))+
           HC4*(lvy_r(lidz,lidy+3,lidx)-lvy_r(lidz,lidy-4,lidx)));
    
    vyz_r = (HC1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx))+
           HC3*(lvy_r(lidz+3,lidy,lidx)-lvy_r(lidz-2,lidy,lidx))+
           HC4*(lvy_r(lidz+4,lidy,lidx)-lvy_r(lidz-3,lidy,lidx)));
#elif FDOH==5
    vyx_r = (HC1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1))+
           HC3*(lvy_r(lidz,lidy,lidx+3)-lvy_r(lidz,lidy,lidx-2))+
           HC4*(lvy_r(lidz,lidy,lidx+4)-lvy_r(lidz,lidy,lidx-3))+
           HC5*(lvy_r(lidz,lidy,lidx+5)-lvy_r(lidz,lidy,lidx-4)));
    
    vyy_r = (HC1*(lvy_r(lidz,lidy,lidx)  -lvy_r(lidz,lidy-1,lidx))+
           HC2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx))+
           HC3*(lvy_r(lidz,lidy+2,lidx)-lvy_r(lidz,lidy-3,lidx))+
           HC4*(lvy_r(lidz,lidy+3,lidx)-lvy_r(lidz,lidy-4,lidx))+
           HC5*(lvy_r(lidz,lidy+4,lidx)-lvy_r(lidz,lidy-5,lidx)));
    
    vyz_r = (HC1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx))+
           HC3*(lvy_r(lidz+3,lidy,lidx)-lvy_r(lidz-2,lidy,lidx))+
           HC4*(lvy_r(lidz+4,lidy,lidx)-lvy_r(lidz-3,lidy,lidx))+
           HC5*(lvy_r(lidz+5,lidy,lidx)-lvy_r(lidz-4,lidy,lidx)));
#elif FDOH==6
    vyx_r = (HC1*(lvy_r(lidz,lidy,lidx+1)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz,lidy,lidx+2)-lvy_r(lidz,lidy,lidx-1))+
           HC3*(lvy_r(lidz,lidy,lidx+3)-lvy_r(lidz,lidy,lidx-2))+
           HC4*(lvy_r(lidz,lidy,lidx+4)-lvy_r(lidz,lidy,lidx-3))+
           HC5*(lvy_r(lidz,lidy,lidx+5)-lvy_r(lidz,lidy,lidx-4))+
           HC6*(lvy_r(lidz,lidy,lidx+6)-lvy_r(lidz,lidy,lidx-5)));
    
    vyy_r = (HC1*(lvy_r(lidz,lidy,lidx)  -lvy_r(lidz,lidy-1,lidx))+
           HC2*(lvy_r(lidz,lidy+1,lidx)-lvy_r(lidz,lidy-2,lidx))+
           HC3*(lvy_r(lidz,lidy+2,lidx)-lvy_r(lidz,lidy-3,lidx))+
           HC4*(lvy_r(lidz,lidy+3,lidx)-lvy_r(lidz,lidy-4,lidx))+
           HC5*(lvy_r(lidz,lidy+4,lidx)-lvy_r(lidz,lidy-5,lidx))+
           HC6*(lvy_r(lidz,lidy+5,lidx)-lvy_r(lidz,lidy-6,lidx)));
    
    vyz_r = (HC1*(lvy_r(lidz+1,lidy,lidx)-lvy_r(lidz,lidy,lidx))+
           HC2*(lvy_r(lidz+2,lidy,lidx)-lvy_r(lidz-1,lidy,lidx))+
           HC3*(lvy_r(lidz+3,lidy,lidx)-lvy_r(lidz-2,lidy,lidx))+
           HC4*(lvy_r(lidz+4,lidy,lidx)-lvy_r(lidz-3,lidy,lidx))+
           HC5*(lvy_r(lidz+5,lidy,lidx)-lvy_r(lidz-4,lidy,lidx))+
           HC6*(lvy_r(lidz+6,lidy,lidx)-lvy_r(lidz-5,lidy,lidx)));
#endif
    
    
    
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
        lvz_r(lidz,lidy,lidx)=vz_r(gidz, gidy, gidx);
        if (lidy<2*FDOH)
            lvz_r(lidz,lidy-FDOH,lidx)=vz_r(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvz_r(lidz,lidy+lsizey-3*FDOH,lidx)=vz_r(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvz_r(lidz,lidy+FDOH,lidx)=vz_r(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvz_r(lidz,lidy-lsizey+3*FDOH,lidx)=vz_r(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvz_r(lidz,lidy,lidx-FDOH)=vz_r(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvz_r(lidz,lidy,lidx+lsizex-3*FDOH)=vz_r(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvz_r(lidz,lidy,lidx+FDOH)=vz_r(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvz_r(lidz,lidy,lidx-lsizex+3*FDOH)=vz_r(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvz_r(lidz-FDOH,lidy,lidx)=vz_r(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvz_r(lidz+FDOH,lidy,lidx)=vz_r(gidz+FDOH,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
    
#if   FDOH==1
    vzx_r = (lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx));
    vzy_r = (lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx));
    vzz_r = (lvz_r(lidz,lidy,lidx)-lvz_r(lidz-1,lidy,lidx));
#elif FDOH==2
    
    vzx_r = (HC1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1)));
    
    vzy_r = (HC1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx)));
    
    vzz_r = (HC1*(lvz_r(lidz,lidy,lidx)  -lvz_r(lidz-1,lidy,lidx))+
           HC2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx)));
#elif FDOH==3
    vzx_r = (HC1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1))+
           HC3*(lvz_r(lidz,lidy,lidx+3)-lvz_r(lidz,lidy,lidx-2)));
    
    vzy_r = (HC1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx))+
           HC3*(lvz_r(lidz,lidy+3,lidx)-lvz_r(lidz,lidy-2,lidx)));
    
    vzz_r = (HC1*(lvz_r(lidz,lidy,lidx)-lvz_r(lidz-1,lidy,lidx))+
           HC2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx))+
           HC3*(lvz_r(lidz+2,lidy,lidx)-lvz_r(lidz-3,lidy,lidx)));
#elif FDOH==4
    vzx_r = (HC1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1))+
           HC3*(lvz_r(lidz,lidy,lidx+3)-lvz_r(lidz,lidy,lidx-2))+
           HC4*(lvz_r(lidz,lidy,lidx+4)-lvz_r(lidz,lidy,lidx-3)));
    
    vzy_r = (HC1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx))+
           HC3*(lvz_r(lidz,lidy+3,lidx)-lvz_r(lidz,lidy-2,lidx))+
           HC4*(lvz_r(lidz,lidy+4,lidx)-lvz_r(lidz,lidy-3,lidx)));
    
    vzz_r = (HC1*(lvz_r(lidz,lidy,lidx)  -lvz_r(lidz-1,lidy,lidx))+
           HC2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx))+
           HC3*(lvz_r(lidz+2,lidy,lidx)-lvz_r(lidz-3,lidy,lidx))+
           HC4*(lvz_r(lidz+3,lidy,lidx)-lvz_r(lidz-4,lidy,lidx)));
#elif FDOH==5
    vzx_r = (HC1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1))+
           HC3*(lvz_r(lidz,lidy,lidx+3)-lvz_r(lidz,lidy,lidx-2))+
           HC4*(lvz_r(lidz,lidy,lidx+4)-lvz_r(lidz,lidy,lidx-3))+
           HC5*(lvz_r(lidz,lidy,lidx+5)-lvz_r(lidz,lidy,lidx-4)));
    
    vzy_r = (HC1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx))+
           HC3*(lvz_r(lidz,lidy+3,lidx)-lvz_r(lidz,lidy-2,lidx))+
           HC4*(lvz_r(lidz,lidy+4,lidx)-lvz_r(lidz,lidy-3,lidx))+
           HC5*(lvz_r(lidz,lidy+5,lidx)-lvz_r(lidz,lidy-4,lidx)));
    
    vzz_r = (HC1*(lvz_r(lidz,lidy,lidx)  -lvz_r(lidz-1,lidy,lidx))+
           HC2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx))+
           HC3*(lvz_r(lidz+2,lidy,lidx)-lvz_r(lidz-3,lidy,lidx))+
           HC4*(lvz_r(lidz+3,lidy,lidx)-lvz_r(lidz-4,lidy,lidx))+
           HC5*(lvz_r(lidz+4,lidy,lidx)-lvz_r(lidz-5,lidy,lidx)));
#elif FDOH==6
    vzx_r = (HC1*(lvz_r(lidz,lidy,lidx+1)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy,lidx+2)-lvz_r(lidz,lidy,lidx-1))+
           HC3*(lvz_r(lidz,lidy,lidx+3)-lvz_r(lidz,lidy,lidx-2))+
           HC4*(lvz_r(lidz,lidy,lidx+4)-lvz_r(lidz,lidy,lidx-3))+
           HC5*(lvz_r(lidz,lidy,lidx+5)-lvz_r(lidz,lidy,lidx-4))+
           HC6*(lvz_r(lidz,lidy,lidx+6)-lvz_r(lidz,lidy,lidx-5)));
    
    vzy_r = (HC1*(lvz_r(lidz,lidy+1,lidx)-lvz_r(lidz,lidy,lidx))+
           HC2*(lvz_r(lidz,lidy+2,lidx)-lvz_r(lidz,lidy-1,lidx))+
           HC3*(lvz_r(lidz,lidy+3,lidx)-lvz_r(lidz,lidy-2,lidx))+
           HC4*(lvz_r(lidz,lidy+4,lidx)-lvz_r(lidz,lidy-3,lidx))+
           HC5*(lvz_r(lidz,lidy+5,lidx)-lvz_r(lidz,lidy-4,lidx))+
           HC6*(lvz_r(lidz,lidy+6,lidx)-lvz_r(lidz,lidy-5,lidx)));
    
    vzz_r = (HC1*(lvz_r(lidz,lidy,lidx)  -lvz_r(lidz-1,lidy,lidx))+
           HC2*(lvz_r(lidz+1,lidy,lidx)-lvz_r(lidz-2,lidy,lidx))+
           HC3*(lvz_r(lidz+2,lidy,lidx)-lvz_r(lidz-3,lidy,lidx))+
           HC4*(lvz_r(lidz+3,lidy,lidx)-lvz_r(lidz-4,lidy,lidx))+
           HC5*(lvz_r(lidz+4,lidy,lidx)-lvz_r(lidz-5,lidy,lidx))+
           HC6*(lvz_r(lidz+5,lidy,lidx)-lvz_r(lidz-6,lidy,lidx)));
#endif
    }
    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if LOCAL_OFF==0
#if COMM12==0
    if (gidy>(NY-FDOH-1) || gidz>(NZ-FDOH-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
        return;
    }
    
#else
    if (gidy>(NY-FDOH-1) || gidz>(NZ-FDOH-1) ){
        return;
    }
#endif
#endif
    
 
    
// Read model parameters into local memory
#if LVE==0
    lM=M(gidz,gidy,gidx);
    lmu=mu(gidz,gidy,gidx);
    fipjp=muipjp(gidz,gidy,gidx);
    fjpkp=mujpkp(gidz,gidy,gidx);
    fipkp=muipkp(gidz,gidy,gidx);
    g=lM;
    f=2.0*lmu;

    
#else
    
    lM=M(gidz,gidy,gidx);
    lmu=u(gidz,gidy,gidx);
    lmuipkp=muipkp(gidz,gidy,gidx);
    lmuipjp=muipjp(gidz,gidy,gidx);
    lmujpkp=mujpkp(gidz,gidy,gidx);
    ltaup=taup(gidz,gidy,gidx);
    ltaus=taus(gidz,gidy,gidx);
    ltausipkp=tausipkp(gidz,gidy,gidx);
    ltausipjp=tausipjp(gidz,gidy,gidx);
    ltausjpkp=tausjpkp(gidz,gidy,gidx);
    
    for (l=0;l<LVE;l++){
        leta[l]=eta[l];
    }
    
    fipjp=lmuipjp*(1.0+ (float)LVE*ltausipjp);
    fjpkp=lmujpkp*(1.0+ (float)LVE*ltausjpkp);
    fipkp=lmuipkp*(1.0+ (float)LVE*ltausipkp);
    g=lM*(1.0+(float)LVE*ltaup);
    f=2.0*lmu*(1.0+(float)LVE*ltaus);
    dipjp=lmuipjp*ltausipjp;
    djpkp=lmujpkp*ltausjpkp;
    dipkp=lmuipkp*ltausipkp;
    d=2.0*lmu*ltaus;
    e=lM*ltaup;
    
    
#endif
    
    
// Backpropagate the forward stresses
#if BACK_PROP_TYPE==1
    {
#if LVE==0

        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;
        
        sxy(gidz,gidy,gidx)-=(fipjp*vxyyx);
        syz(gidz,gidy,gidx)-=(fjpkp*vyzzy);
        sxz(gidz,gidy,gidx)-=(fipkp*vxzzx);
        sxx(gidz,gidy,gidx)-=((g*vxxyyzz)-(f*vyyzz)) ;
        syy(gidz,gidy,gidx)-=((g*vxxyyzz)-(f*vxxzz)) ;
        szz(gidz,gidy,gidx)-=((g*vxxyyzz)-(f*vxxyy)) ;
        
// Backpropagation is not stable for viscoelastic wave equation
#else
        
    
        /* computing sums of the old memory variables */
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<LVE;l++){
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
        
        lsxy=(fipjp*vxyyx)+(DT2*sumrxy);
        lsyz=(fjpkp*vyzzy)+(DT2*sumryz);
        lsxz=(fipkp*vxzzx)+(DT2*sumrxz);
        lsxx=((g*vxxyyzz)-(f*vyyzz))+(DT2*sumrxx);
        lsyy=((g*vxxyyzz)-(f*vxxzz))+(DT2*sumryy);
        lszz=((g*vxxyyzz)-(f*vxxyy))+(DT2*sumrzz);
        
        
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<LVE;l++){
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

        /* and now the components of the stress tensor are
         completely updated */
        sxy(gidz,gidy,gidx)-=lsxy+(DT2*sumrxy);
        syz(gidz,gidy,gidx)-=lsyz+(DT2*sumryz);
        sxz(gidz,gidy,gidx)-=lsxz+(DT2*sumrxz);
        sxx(gidz,gidy,gidx)-=lsxx+(DT2*sumrxx);
        syy(gidz,gidy,gidx)-=lsyy+(DT2*sumryy);
        szz(gidz,gidy,gidx)-=lszz+(DT2*sumrzz);
        
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
#if ABS_TYPE==1
    {
        int ind;
        
        if (gidz>NZ-NAB-FDOH-1){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz - NZ+NAB+FDOH+NAB;
            ind=2*NAB-1-k;
            
            psi_vx_z(k,j,i) = b_z_half[ind] * psi_vx_z(k,j,i) + a_z_half[ind] * vxz_r;
            vxz_r = vxz_r / K_z_half[ind] + psi_vx_z(k,j,i);
            psi_vy_z(k,j,i) = b_z_half[ind] * psi_vy_z(k,j,i) + a_z_half[ind] * vyz_r;
            vyz_r = vyz_r / K_z_half[ind] + psi_vy_z(k,j,i);
            psi_vz_z(k,j,i) = b_z[ind+1] * psi_vz_z(k,j,i) + a_z[ind+1] * vzz_r;
            vzz_r = vzz_r / K_z[ind+1] + psi_vz_z(k,j,i);
            
        }
        
#if FREESURF==0
        else if (gidz-FDOH<NAB){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;
            
            
            psi_vx_z(k,j,i) = b_z_half[k] * psi_vx_z(k,j,i) + a_z_half[k] * vxz_r;
            vxz_r = vxz_r / K_z_half[k] + psi_vx_z(k,j,i);
            psi_vy_z(k,j,i) = b_z_half[k] * psi_vy_z(k,j,i) + a_z_half[k] * vyz_r;
            vyz_r = vyz_r / K_z_half[k] + psi_vy_z(k,j,i);
            psi_vz_z(k,j,i) = b_z[k] * psi_vz_z(k,j,i) + a_z[k] * vzz_r;
            vzz_r = vzz_r / K_z[k] + psi_vz_z(k,j,i);
            
            
        }
#endif
        
        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;
            
            psi_vx_y(k,j,i) = b_y_half[j] * psi_vx_y(k,j,i) + a_y_half[j] * vxy_r;
            vxy_r = vxy_r / K_y_half[j] + psi_vx_y(k,j,i);
            psi_vy_y(k,j,i) = b_y[j] * psi_vy_y(k,j,i) + a_y[j] * vyy_r;
            vyy_r = vyy_r / K_y[j] + psi_vy_y(k,j,i);
            psi_vz_y(k,j,i) = b_y_half[j] * psi_vz_y(k,j,i) + a_y_half[j] * vzy_r;
            vzy_r = vzy_r / K_y_half[j] + psi_vz_y(k,j,i);
            
        }
        
        else if (gidy>NY-NAB-FDOH-1){
            
            i =gidx-FDOH;
            j =gidy - NY+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-j;
            
            
            psi_vx_y(k,j,i) = b_y_half[ind] * psi_vx_y(k,j,i) + a_y_half[ind] * vxy_r;
            vxy_r = vxy_r / K_y_half[ind] + psi_vx_y(k,j,i);
            psi_vy_y(k,j,i) = b_y[ind+1] * psi_vy_y(k,j,i) + a_y[ind+1] * vyy_r;
            vyy_r = vyy_r / K_y[ind+1] + psi_vy_y(k,j,i);
            psi_vz_y(k,j,i) = b_y_half[ind] * psi_vz_y(k,j,i) + a_y_half[ind] * vzy_r;
            vzy_r = vzy_r / K_y_half[ind] + psi_vz_y(k,j,i);
            
            
        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;
            
            psi_vx_x(k,j,i) = b_x[i] * psi_vx_x(k,j,i) + a_x[i] * vxx_r;
            vxx_r = vxx_r / K_x[i] + psi_vx_x(k,j,i);
            psi_vy_x(k,j,i) = b_x_half[i] * psi_vy_x(k,j,i) + a_x_half[i] * vyx_r;
            vyx_r = vyx_r / K_x_half[i] + psi_vy_x(k,j,i);
            psi_vz_x(k,j,i) = b_x_half[i] * psi_vz_x(k,j,i) + a_x_half[i] * vzx_r;
            vzx_r = vzx_r / K_x_half[i] + psi_vz_x(k,j,i);
            
            
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            
            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            
            
            psi_vx_x(k,j,i) = b_x[ind+1] * psi_vx_x(k,j,i) + a_x[ind+1] * vxx_r;
            vxx_r = vxx_r /K_x[ind+1] + psi_vx_x(k,j,i);
            psi_vy_x(k,j,i) = b_x_half[ind] * psi_vy_x(k,j,i) + a_x_half[ind] * vyx_r;
            vyx_r = vyx_r  /K_x_half[ind] + psi_vy_x(k,j,i);
            psi_vz_x(k,j,i) = b_x_half[ind] * psi_vz_x(k,j,i) + a_x_half[ind] * vzx_r;
            vzx_r = vzx_r / K_x_half[ind]  +psi_vz_x(k,j,i);
            
            
        }
#endif
        
    }
#endif

// Update adjoint stresses
    {
#if LVE==0
    
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
    for (l=0;l<LVE;l++){
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

    lsxy=(fipjp*vxyyx_r)+(DT2*sumrxy);
    lsyz=(fjpkp*vyzzy_r)+(DT2*sumryz);
    lsxz=(fipkp*vxzzx_r)+(DT2*sumrxz);
    lsxx=((g*vxxyyzz_r)-(f*vyyzz_r))+(DT2*sumrxx);
    lsyy=((g*vxxyyzz_r)-(f*vxxzz_r))+(DT2*sumryy);
    lszz=((g*vxxyyzz_r)-(f*vxxyy_r))+(DT2*sumrzz);
    
    sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
    for (l=0;l<LVE;l++){
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
    sxy_r(gidz,gidy,gidx)+=lsxy+(DT2*sumrxy);
    syz_r(gidz,gidy,gidx)+=lsyz+(DT2*sumryz);
    sxz_r(gidz,gidy,gidx)+=lsxz+(DT2*sumrxz);
    sxx_r(gidz,gidy,gidx)+=lsxx+(DT2*sumrxx);
    syy_r(gidz,gidy,gidx)+=lsyy+(DT2*sumryy);
    szz_r(gidz,gidy,gidx)+=lszz+(DT2*sumrzz);
    
#endif
}

    
// Absorbing boundary
#if ABS_TYPE==2
    {
        if (gidz-FDOH<NAB){
            sxy_r(gidz,gidy,gidx)*=taper[gidz-FDOH];
            syz_r(gidz,gidy,gidx)*=taper[gidz-FDOH];
            sxz_r(gidz,gidy,gidx)*=taper[gidz-FDOH];
            sxx_r(gidz,gidy,gidx)*=taper[gidz-FDOH];
            syy_r(gidz,gidy,gidx)*=taper[gidz-FDOH];
            szz_r(gidz,gidy,gidx)*=taper[gidz-FDOH];
        }
        
        if (gidz>NZ-NAB-FDOH-1){
            sxy_r(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            syz_r(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            sxz_r(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            sxx_r(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            syy_r(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            szz_r(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
        }
        if (gidy-FDOH<NAB){
            sxy_r(gidz,gidy,gidx)*=taper[gidy-FDOH];
            syz_r(gidz,gidy,gidx)*=taper[gidy-FDOH];
            sxz_r(gidz,gidy,gidx)*=taper[gidy-FDOH];
            sxx_r(gidz,gidy,gidx)*=taper[gidy-FDOH];
            syy_r(gidz,gidy,gidx)*=taper[gidy-FDOH];
            szz_r(gidz,gidy,gidx)*=taper[gidy-FDOH];
        }
        
        if (gidy>NY-NAB-FDOH-1){
            sxy_r(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            syz_r(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            sxz_r(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            sxx_r(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            syy_r(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            szz_r(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            sxy_r(gidz,gidy,gidx)*=taper[gidx-FDOH];
            syz_r(gidz,gidy,gidx)*=taper[gidx-FDOH];
            sxz_r(gidz,gidy,gidx)*=taper[gidx-FDOH];
            sxx_r(gidz,gidy,gidx)*=taper[gidx-FDOH];
            syy_r(gidz,gidy,gidx)*=taper[gidx-FDOH];
            szz_r(gidz,gidy,gidx)*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            sxy_r(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            syz_r(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            sxz_r(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            sxx_r(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            syy_r(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            szz_r(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    
// Shear wave modulus and P-wave modulus gradient calculation on the fly    
#if BACK_PROP_TYPE==1
    float c1=1.0/pown(3.0*lM-4.0*lmu,2);
    float c3=1.0/pown(lmu,2);
    float c5=1.0/6.0*c3;
    
    float dM=-c1*( sxx(gidz,gidy,gidx)+syy(gidz,gidy,gidx)+szz(gidz,gidy,gidx) )*( lsxx+lsyy+lszz );
    
    gradM(gidz,gidy,gidx)+=-dM;
    gradmu(gidz,gidy,gidx)+=-c3*(sxz(gidz,gidy,gidx)*lsxz +sxy(gidz,gidy,gidx)*lsxy +syz(gidz,gidy,gidx)*lsyz )
    + 4.0/3*dM
    -c5*(  lsxx*(2.0*sxx(gidz,gidy,gidx)- syy(gidz,gidy,gidx)-szz(gidz,gidy,gidx) )
         +lsyy*(2.0*syy(gidz,gidy,gidx)- sxx(gidz,gidy,gidx)-szz(gidz,gidy,gidx) )
         +lszz*(2.0*szz(gidz,gidy,gidx)- sxx(gidz,gidy,gidx)-syy(gidz,gidy,gidx) )
         );
#if HOUT==1
    float dMH=c1*pown( sxx(gidz,gidy,gidx)+syy(gidz,gidy,gidx)+szz(gidz,gidy,gidx),2);
    HM(gidz,gidx)+= dMH;
    Hmu(gidz,gidx)+=c3*(pown(sxz(gidz,gidy,gidx),2)+pown(sxy(gidz,gidy,gidx),2)+pown(syz(gidz,gidy,gidx),2))
                    - 4.0/3*dM
                    +c5*(pown((2.0*sxx(gidz,gidy,gidx)- syy(gidz,gidy,gidx)-szz(gidz,gidy,gidx)),2)
                         +pown((2.0*syy(gidz,gidy,gidx)- sxx(gidz,gidy,gidx)-szz(gidz,gidy,gidx)),2)
                         +pown(2.0*szz(gidz,gidy,gidx)- sxx(gidz,gidy,gidx)-syy(gidz,gidy,gidx)),2));
#endif
    
#endif
    
#if GRADSRCOUT==1
    //TODO
//    float pressure;
//    if (nsrc>0){
//
//        for (int srci=0; srci<nsrc; srci++){
//
//            int SOURCE_TYPE= (int)srcpos_loc(4,srci);
//
//            if (SOURCE_TYPE==1){
//                int i=(int)(srcpos_loc(0,srci)-0.5)+FDOH;
//                int j=(int)(srcpos_loc(1,srci)-0.5)+FDOH;
//                int k=(int)(srcpos_loc(2,srci)-0.5)+FDOH;
//                
//                
//                if (i==gidx && j==gidy && k==gidz){
//                    
//                    pressure=(sxx_r(gidz,gidy,gidx)+syy_r(gidz,gidy,gidx)+szz_r(gidz,gidy,gidx) )/(2.0*DH*DH*DH);
//                    if ( (nt>0) && (nt< NT ) ){
//                        gradsrc(srci,nt+1)+=pressure;
//                        gradsrc(srci,nt-1)-=pressure;
//                    }
//                    else if (nt==0)
//                        gradsrc(srci,nt+1)+=pressure;
//                    else if (nt==NT)
//                        gradsrc(srci,nt-1)-=pressure;
//
//                }
//            }
//        }
//    }
    
#endif


}

