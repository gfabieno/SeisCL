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

/*Adjoint update of the velocities in 3D SV*//

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define lbnd (FDOH+NAB)

#define rho(z,y,x)     rho[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,y,x)     rip[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,y,x)     rjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,y,x)     rkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipjp(z,y,x) uipjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define ujpkp(z,y,x) ujpkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipkp(z,y,x) uipkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define u(z,y,x)         u[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define pi(z,y,x)       pi[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,y,x)   gradrho[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,y,x)   gradM[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradmu(z,y,x)   gradmu[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,y,x)   gradtaup[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaus(z,y,x)   gradtaus[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

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

#define vx_r(z,y,x)   vx_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vy_r(z,y,x)   vy_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vz_r(z,y,x)   vz_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxx_r(z,y,x) sxx_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syy_r(z,y,x) syy_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define szz_r(z,y,x) szz_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxy_r(z,y,x) sxy_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syz_r(z,y,x) syz_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxz_r(z,y,x) sxz_r[(x)*NY*(NZ)+(y)*(NZ)+(z)]

#define psi_sxx_x(z,y,x) psi_sxx_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxy_x(z,y,x) psi_sxy_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxz_x(z,y,x) psi_sxz_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxy_y(z,y,x) psi_sxy_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_syy_y(z,y,x) psi_syy_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_syz_y(z,y,x) psi_syz_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxz_z(z,y,x) psi_sxz_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_syz_z(z,y,x) psi_syz_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_szz_z(z,y,x) psi_szz_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]

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
    int NZbnd = (NZ-2*FDOH-2*NAB);

    int m=-1;
    i-=lbnd;
    j-=lbnd;
    k-=lbnd;
    
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
        m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    else {//down
        i=i-FDOH;
        j=j-FDOH;
        k=k-NZbnd+FDOH;
        m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH*2+(NXbnd-2*FDOH)*(NYbnd-2*FDOH)*FDOH+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    
    

#elif DEVID==0 & MYGROUPID==0
    int NXbnd = (NX-2*FDOH-NAB);
    int NYbnd = (NY-2*FDOH-2*NAB);
    int NZbnd = (NZ-2*FDOH-2*NAB);
    
    int m=-1;
    i-=lbnd;
    j-=lbnd;
    k-=lbnd;
    
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
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    else {//down
        i=i-FDOH;
        j=j-FDOH;
        k=k-NZbnd+FDOH;
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+(NXbnd-FDOH)*(NYbnd-2*FDOH)*FDOH+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
#elif DEVID==NUM_DEVICES-1 & MYGROUPID==NLOCALP-1
    int NXbnd = (NX-2*FDOH-NAB);
    int NYbnd = (NY-2*FDOH-2*NAB);
    int NZbnd = (NZ-2*FDOH-2*NAB);
    
    int m=-1;
    i-=FDOH;
    j-=lbnd;
    k-=lbnd;
    
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
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    else {//down
        j=j-FDOH;
        k=k-NZbnd+FDOH;
        m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+(NXbnd-FDOH)*(NYbnd-2*FDOH)*FDOH+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    
#else
    int NXbnd = (NX-2*FDOH);
    int NYbnd = (NY-2*FDOH-2*NAB);
    int NZbnd = (NZ-2*FDOH-2*NAB);
    
    int m=-1;
    i-=FDOH;
    j-=lbnd;
    k-=lbnd;
    
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
        m=NXbnd*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    else {//down
        j=j-FDOH;
        k=k-NZbnd+FDOH;
        m=NXbnd*NZbnd*FDOH*2+NXbnd*(NYbnd-2*FDOH)*FDOH+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
    }
    
#endif
    
    
    return m;
 
}


__kernel void update_adjv(int offcomm, 
                          __global float *vx,         __global float *vy,      __global float *vz,
                          __global float *sxx,        __global float *syy,     __global float *szz,
                          __global float *sxy,        __global float *syz,     __global float *sxz,
                          __global float *vxbnd,      __global float *vybnd,   __global float *vzbnd,
                          __global float *sxxbnd,     __global float *syybnd,  __global float *szzbnd,
                          __global float *sxybnd,     __global float *syzbnd,  __global float *sxzbnd,
                          __global float *vx_r,       __global float *vy_r,    __global float *vz_r,
                          __global float *sxx_r,      __global float *syy_r,   __global float *szz_r,
                          __global float *sxy_r,      __global float *syz_r,   __global float *sxz_r,
                          __global float *rip,        __global float *rjp,     __global float *rkp,
                          __global float *taper,
                          __global float *K_x,        __global float *a_x,          __global float *b_x,
                          __global float *K_x_half,   __global float *a_x_half,     __global float *b_x_half,
                          __global float *K_y,        __global float *a_y,          __global float *b_y,
                          __global float *K_y_half,   __global float *a_y_half,     __global float *b_y_half,
                          __global float *K_z,        __global float *a_z,          __global float *b_z,
                          __global float *K_z_half,   __global float *a_z_half,     __global float *b_z_half,
                          __global float *psi_sxx_x,  __global float *psi_sxy_x,     __global float *psi_sxy_y,
                          __global float *psi_sxz_x,  __global float *psi_sxz_z,     __global float *psi_syy_y,
                          __global float *psi_syz_y,  __global float *psi_syz_z,     __global float *psi_szz_z,
                          __local  float *lvar, __global float *gradrho, __global float *gradsrc)
{

    int g,i,j,k,m;
    float sxx_x, syy_y, szz_z, sxy_y, sxy_x, syz_y, syz_z, sxz_x, sxz_z;
    float sxx_x_r, syy_y_r, szz_z_r, sxy_y_r, sxy_x_r, syz_y_r, syz_z_r, sxz_x_r, sxz_z_r;
    float lvx, lvy, lvz;

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
    
#define lsxx lvar
#define lsyy lvar
#define lszz lvar
#define lsxy lvar
#define lsyz lvar
#define lsxz lvar
    
#define lsxx_r lvar
#define lsyy_r lvar
#define lszz_r lvar
#define lsxy_r lvar
#define lsyz_r lvar
#define lsxz_r lvar

// If local memory is turned off
#elif LOCAL_OFF==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*FDOH);
    int glsizey = (NY-2*FDOH);
    int gidz = gid%glsizez+FDOH;
    int gidy = (gid/glsizez)%glsizey+FDOH;
    int gidx = gid/(glsizez*glsizey)+FDOH+offcomm;
    
#define lsxx sxx
#define lsyy syy
#define lszz szz
#define lsxy sxy
#define lsyz syz
#define lsxz sxz
    
#define lsxx_r sxx_r
#define lsyy_r syy_r
#define lszz_r szz_r
#define lsxy_r sxy_r
#define lsyz_r syz_r
#define lsxz_r sxz_r
    
#define lidx gidx
#define lidy gidy
#define lidz gidz
    
#define lsizez NZ
#define lsizey NY
#define lsizex NX
    
#endif
    
// Calculation of the stress spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
    {

#if LOCAL_OFF==0
        lsxx(lidz,lidy,lidx)=sxx(gidz,gidy,gidx);
        if (lidx<2*FDOH)
            lsxx(lidz,lidy,lidx-FDOH)=sxx(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxx(lidz,lidy,lidx+lsizex-3*FDOH)=sxx(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxx(lidz,lidy,lidx+FDOH)=sxx(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxx(lidz,lidy,lidx-lsizex+3*FDOH)=sxx(gidz,gidy,gidx-lsizex+3*FDOH);
        
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH ==1
        sxx_x = DTDH*HC1*(lsxx(lidz,lidy,lidx+1) - lsxx(lidz,lidy,lidx));
#elif FDOH ==2
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidy,lidx+1) - lsxx(lidz,lidy,lidx))
                      +HC2*(lsxx(lidz,lidy,lidx+2) - lsxx(lidz,lidy,lidx-1)));
#elif FDOH ==3
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      HC2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      HC3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2)));
#elif FDOH ==4
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      HC2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      HC3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2))+
                      HC4*(lsxx(lidz,lidy,lidx+4)-lsxx(lidz,lidy,lidx-3)));
#elif FDOH ==5
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      HC2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      HC3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2))+
                      HC4*(lsxx(lidz,lidy,lidx+4)-lsxx(lidz,lidy,lidx-3))+
                      HC5*(lsxx(lidz,lidy,lidx+5)-lsxx(lidz,lidy,lidx-4)));
#elif FDOH ==6
        sxx_x = DTDH*(HC1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      HC2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      HC3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2))+
                      HC4*(lsxx(lidz,lidy,lidx+4)-lsxx(lidz,lidy,lidx-3))+
                      HC5*(lsxx(lidz,lidy,lidx+5)-lsxx(lidz,lidy,lidx-4))+
                      HC6*(lsxx(lidz,lidy,lidx+6)-lsxx(lidz,lidy,lidx-5)));
#endif
        
        
#if LOCAL_OFF==0
        lsyy(lidz,lidy,lidx)=syy(gidz,gidy,gidx);
        if (lidy<2*FDOH)
            lsyy(lidz,lidy-FDOH,lidx)=syy(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsyy(lidz,lidy+lsizey-3*FDOH,lidx)=syy(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsyy(lidz,lidy+FDOH,lidx)=syy(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsyy(lidz,lidy-lsizey+3*FDOH,lidx)=syy(gidz,gidy-lsizey+3*FDOH,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH ==1
        syy_y = DTDH*HC1*(lsyy(lidz,lidy+1,lidx) - lsyy(lidz,lidy,lidx));
#elif FDOH ==2
        syy_y = DTDH*(HC1*(lsyy(lidz,lidy+1,lidx) - lsyy(lidz,lidy,lidx))
                      +HC2*(lsyy(lidz,lidy+2,lidx) - lsyy(lidz,lidy-1,lidx)));
#elif FDOH ==3
        syy_y = DTDH*(HC1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      HC2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      HC3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx)));
#elif FDOH ==4
        syy_y = DTDH*(HC1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      HC2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      HC3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx))+
                      HC4*(lsyy(lidz,lidy+4,lidx)-lsyy(lidz,lidy-3,lidx)));
#elif FDOH ==5
        syy_y = DTDH*(HC1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      HC2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      HC3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx))+
                      HC4*(lsyy(lidz,lidy+4,lidx)-lsyy(lidz,lidy-3,lidx))+
                      HC5*(lsyy(lidz,lidy+5,lidx)-lsyy(lidz,lidy-4,lidx)));
#elif FDOH ==6
        syy_y = DTDH*(HC1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      HC2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      HC3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx))+
                      HC4*(lsyy(lidz,lidy+4,lidx)-lsyy(lidz,lidy-3,lidx))+
                      HC5*(lsyy(lidz,lidy+5,lidx)-lsyy(lidz,lidy-4,lidx))+
                      HC6*(lsyy(lidz,lidy+6,lidx)-lsyy(lidz,lidy-5,lidx)));
#endif
        
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lszz(lidz,lidy,lidx)=szz(gidz,gidy,gidx);
        if (lidz<2*FDOH)
            lszz(lidz-FDOH,lidy,lidx)=szz(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lszz(lidz+FDOH,lidy,lidx)=szz(gidz+FDOH,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH ==1
        szz_z = DTDH*HC1*(lszz(lidz+1,lidy,lidx) - lszz(lidz,lidy,lidx));
#elif FDOH ==2
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidy,lidx) - lszz(lidz,lidy,lidx))
                      +HC2*(lszz(lidz+2,lidy,lidx) - lszz(lidz-1,lidy,lidx)));
#elif FDOH ==3
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      HC2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      HC3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx)));
#elif FDOH ==4
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      HC2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      HC3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx))+
                      HC4*(lszz(lidz+4,lidy,lidx)-lszz(lidz-3,lidy,lidx)));
#elif FDOH ==5
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      HC2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      HC3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx))+
                      HC4*(lszz(lidz+4,lidy,lidx)-lszz(lidz-3,lidy,lidx))+
                      HC5*(lszz(lidz+5,lidy,lidx)-lszz(lidz-4,lidy,lidx)));
#elif FDOH ==6
        szz_z = DTDH*(HC1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      HC2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      HC3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx))+
                      HC4*(lszz(lidz+4,lidy,lidx)-lszz(lidz-3,lidy,lidx))+
                      HC5*(lszz(lidz+5,lidy,lidx)-lszz(lidz-4,lidy,lidx))+
                      HC6*(lszz(lidz+6,lidy,lidx)-lszz(lidz-5,lidy,lidx)));
#endif
        
        
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsxy(lidz,lidy,lidx)=sxy(gidz,gidy,gidx);
        if (lidy<2*FDOH)
            lsxy(lidz,lidy-FDOH,lidx)=sxy(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsxy(lidz,lidy+lsizey-3*FDOH,lidx)=sxy(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsxy(lidz,lidy+FDOH,lidx)=sxy(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsxy(lidz,lidy-lsizey+3*FDOH,lidx)=sxy(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lsxy(lidz,lidy,lidx-FDOH)=sxy(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxy(lidz,lidy,lidx+lsizex-3*FDOH)=sxy(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxy(lidz,lidy,lidx+FDOH)=sxy(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxy(lidz,lidy,lidx-lsizex+3*FDOH)=sxy(gidz,gidy,gidx-lsizex+3*FDOH);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH ==1
        sxy_y = DTDH*HC1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy-1,lidx));
        sxy_x = DTDH*HC1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy,lidx-1));
#elif FDOH ==2
        sxy_y = DTDH*(HC1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy-1,lidx))
                      +HC2*(lsxy(lidz,lidy+1,lidx) - lsxy(lidz,lidy-2,lidx)));
        sxy_x = DTDH*(HC1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy,lidx-1))
                      +HC2*(lsxy(lidz,lidy,lidx+1) - lsxy(lidz,lidy,lidx-2)));
#elif FDOH ==3
        sxy_y = DTDH*(HC1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      HC2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      HC3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx)));
        
        sxy_x = DTDH*(HC1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      HC2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      HC3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3)));
#elif FDOH ==4
        sxy_y = DTDH*(HC1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      HC2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      HC3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx))+
                      HC4*(lsxy(lidz,lidy+3,lidx)-lsxy(lidz,lidy-4,lidx)));
        
        sxy_x = DTDH*(HC1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      HC2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      HC3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3))+
                      HC4*(lsxy(lidz,lidy,lidx+3)-lsxy(lidz,lidy,lidx-4)));
#elif FDOH ==5
        sxy_y = DTDH*(HC1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      HC2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      HC3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx))+
                      HC4*(lsxy(lidz,lidy+3,lidx)-lsxy(lidz,lidy-4,lidx))+
                      HC5*(lsxy(lidz,lidy+4,lidx)-lsxy(lidz,lidy-5,lidx)));
        
        sxy_x = DTDH*(HC1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      HC2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      HC3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3))+
                      HC4*(lsxy(lidz,lidy,lidx+3)-lsxy(lidz,lidy,lidx-4))+
                      HC5*(lsxy(lidz,lidy,lidx+4)-lsxy(lidz,lidy,lidx-5)));
        
#elif FDOH ==6
        
        sxy_y = DTDH*(HC1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      HC2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      HC3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx))+
                      HC4*(lsxy(lidz,lidy+3,lidx)-lsxy(lidz,lidy-4,lidx))+
                      HC5*(lsxy(lidz,lidy+4,lidx)-lsxy(lidz,lidy-5,lidx))+
                      HC6*(lsxy(lidz,lidy+5,lidx)-lsxy(lidz,lidy-6,lidx)));
        
        sxy_x = DTDH*(HC1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      HC2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      HC3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3))+
                      HC4*(lsxy(lidz,lidy,lidx+3)-lsxy(lidz,lidy,lidx-4))+
                      HC5*(lsxy(lidz,lidy,lidx+4)-lsxy(lidz,lidy,lidx-5))+
                      HC6*(lsxy(lidz,lidy,lidx+5)-lsxy(lidz,lidy,lidx-6)));
#endif
        
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsyz(lidz,lidy,lidx)=syz(gidz,gidy,gidx);
        if (lidy<2*FDOH)
            lsyz(lidz,lidy-FDOH,lidx)=syz(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsyz(lidz,lidy+lsizey-3*FDOH,lidx)=syz(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsyz(lidz,lidy+FDOH,lidx)=syz(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsyz(lidz,lidy-lsizey+3*FDOH,lidx)=syz(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidz<2*FDOH)
            lsyz(lidz-FDOH,lidy,lidx)=syz(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lsyz(lidz+FDOH,lidy,lidx)=syz(gidz+FDOH,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH ==1
        syz_z = DTDH*HC1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz-1,lidy,lidx));
        syz_y = DTDH*HC1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz,lidy-1,lidx));
#elif FDOH ==2
        syz_z = DTDH*(HC1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz-1,lidy,lidx))
                      +HC2*(lsyz(lidz+1,lidy,lidx) - lsyz(lidz-2,lidy,lidx)));
        syz_y = DTDH*(HC1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz,lidy-1,lidx))
                      +HC2*(lsyz(lidz,lidy+1,lidx) - lsyz(lidz,lidy-2,lidx)));
#elif FDOH ==3
        syz_z = DTDH*(HC1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      HC2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      HC3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx)));
        
        syz_y = DTDH*(HC1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz,lidy-1,lidx))+
                      HC2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      HC3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx)));
#elif FDOH ==4
        syz_z = DTDH*(HC1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      HC2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      HC3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx))+
                      HC4*(lsyz(lidz+3,lidy,lidx)-lsyz(lidz-4,lidy,lidx)));
        
        syz_y = DTDH*(HC1*(lsyz(lidz,lidy,lidx)-lsyz(lidz,lidy-1,lidx))+
                      HC2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      HC3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx))+
                      HC4*(lsyz(lidz,lidy+3,lidx)-lsyz(lidz,lidy-4,lidx)));
#elif FDOH ==5
        syz_z = DTDH*(HC1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      HC2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      HC3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx))+
                      HC4*(lsyz(lidz+3,lidy,lidx)-lsyz(lidz-4,lidy,lidx))+
                      HC5*(lsyz(lidz+4,lidy,lidx)-lsyz(lidz-5,lidy,lidx)));
        
        syz_y = DTDH*(HC1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz,lidy-1,lidx))+
                      HC2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      HC3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx))+
                      HC4*(lsyz(lidz,lidy+3,lidx)-lsyz(lidz,lidy-4,lidx))+
                      HC5*(lsyz(lidz,lidy+4,lidx)-lsyz(lidz,lidy-5,lidx)));
#elif FDOH ==6
        syz_z = DTDH*(HC1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      HC2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      HC3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx))+
                      HC4*(lsyz(lidz+3,lidy,lidx)-lsyz(lidz-4,lidy,lidx))+
                      HC5*(lsyz(lidz+4,lidy,lidx)-lsyz(lidz-5,lidy,lidx))+
                      HC6*(lsyz(lidz+5,lidy,lidx)-lsyz(lidz-6,lidy,lidx)));
        
        syz_y = DTDH*(HC1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz,lidy-1,lidx))+
                      HC2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      HC3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx))+
                      HC4*(lsyz(lidz,lidy+3,lidx)-lsyz(lidz,lidy-4,lidx))+
                      HC5*(lsyz(lidz,lidy+4,lidx)-lsyz(lidz,lidy-5,lidx))+
                      HC6*(lsyz(lidz,lidy+5,lidx)-lsyz(lidz,lidy-6,lidx)));
#endif
        
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsxz(lidz,lidy,lidx)=sxz(gidz,gidy,gidx);
        
        if (lidx<2*FDOH)
            lsxz(lidz,lidy,lidx-FDOH)=sxz(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxz(lidz,lidy,lidx+lsizex-3*FDOH)=sxz(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxz(lidz,lidy,lidx+FDOH)=sxz(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxz(lidz,lidy,lidx-lsizex+3*FDOH)=sxz(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lsxz(lidz-FDOH,lidy,lidx)=sxz(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lsxz(lidz+FDOH,lidy,lidx)=sxz(gidz+FDOH,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH ==1
        sxz_z = DTDH*HC1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz-1,lidy,lidx));
        sxz_x = DTDH*HC1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz,lidy,lidx-1));
#elif FDOH ==2
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz-1,lidy,lidx))
                      +HC2*(lsxz(lidz+1,lidy,lidx) - lsxz(lidz-2,lidy,lidx)));
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz,lidy,lidx-1))
                      +HC2*(lsxz(lidz,lidy,lidx+1) - lsxz(lidz,lidy,lidx-2)));
        
#elif FDOH ==3
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      HC2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      HC3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx)));
        
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      HC2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      HC3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3)));
#elif FDOH ==4
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      HC2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      HC3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx))+
                      HC4*(lsxz(lidz+3,lidy,lidx)-lsxz(lidz-4,lidy,lidx)));
        
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      HC2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      HC3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3))+
                      HC4*(lsxz(lidz,lidy,lidx+3)-lsxz(lidz,lidy,lidx-4)));
#elif FDOH ==5
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      HC2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      HC3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx))+
                      HC4*(lsxz(lidz+3,lidy,lidx)-lsxz(lidz-4,lidy,lidx))+
                      HC5*(lsxz(lidz+4,lidy,lidx)-lsxz(lidz-5,lidy,lidx)));
        
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      HC2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      HC3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3))+
                      HC4*(lsxz(lidz,lidy,lidx+3)-lsxz(lidz,lidy,lidx-4))+
                      HC5*(lsxz(lidz,lidy,lidx+4)-lsxz(lidz,lidy,lidx-5)));
#elif FDOH ==6
        sxz_z = DTDH*(HC1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      HC2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      HC3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx))+
                      HC4*(lsxz(lidz+3,lidy,lidx)-lsxz(lidz-4,lidy,lidx))+
                      HC5*(lsxz(lidz+4,lidy,lidx)-lsxz(lidz-5,lidy,lidx))+
                      HC6*(lsxz(lidz+5,lidy,lidx)-lsxz(lidz-6,lidy,lidx)));
        
        
        sxz_x = DTDH*(HC1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      HC2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      HC3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3))+
                      HC4*(lsxz(lidz,lidy,lidx+3)-lsxz(lidz,lidy,lidx-4))+
                      HC5*(lsxz(lidz,lidy,lidx+4)-lsxz(lidz,lidy,lidx-5))+
                      HC6*(lsxz(lidz,lidy,lidx+5)-lsxz(lidz,lidy,lidx-6)));
        
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
    
// Calculation of the stress spatial derivatives of the adjoint wavefield
    {
#if LOCAL_OFF==0
    lsxx_r(lidz,lidy,lidx)=sxx_r(gidz,gidy,gidx);
        if (lidx<2*FDOH)
            lsxx_r(lidz,lidy,lidx-FDOH)=sxx_r(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxx_r(lidz,lidy,lidx+lsizex-3*FDOH)=sxx_r(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxx_r(lidz,lidy,lidx+FDOH)=sxx_r(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxx_r(lidz,lidy,lidx-lsizex+3*FDOH)=sxx_r(gidz,gidy,gidx-lsizex+3*FDOH);
    
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH ==1
    sxx_x_r = DTDH*HC1*(lsxx_r(lidz,lidy,lidx+1) - lsxx_r(lidz,lidy,lidx));
#elif FDOH ==2
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidy,lidx+1) - lsxx_r(lidz,lidy,lidx))
                  +HC2*(lsxx_r(lidz,lidy,lidx+2) - lsxx_r(lidz,lidy,lidx-1)));
#elif FDOH ==3
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidy,lidx+1)-lsxx_r(lidz,lidy,lidx))+
                  HC2*(lsxx_r(lidz,lidy,lidx+2)-lsxx_r(lidz,lidy,lidx-1))+
                  HC3*(lsxx_r(lidz,lidy,lidx+3)-lsxx_r(lidz,lidy,lidx-2)));
#elif FDOH ==4
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidy,lidx+1)-lsxx_r(lidz,lidy,lidx))+
                  HC2*(lsxx_r(lidz,lidy,lidx+2)-lsxx_r(lidz,lidy,lidx-1))+
                  HC3*(lsxx_r(lidz,lidy,lidx+3)-lsxx_r(lidz,lidy,lidx-2))+
                  HC4*(lsxx_r(lidz,lidy,lidx+4)-lsxx_r(lidz,lidy,lidx-3)));
#elif FDOH ==5
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidy,lidx+1)-lsxx_r(lidz,lidy,lidx))+
                  HC2*(lsxx_r(lidz,lidy,lidx+2)-lsxx_r(lidz,lidy,lidx-1))+
                  HC3*(lsxx_r(lidz,lidy,lidx+3)-lsxx_r(lidz,lidy,lidx-2))+
                  HC4*(lsxx_r(lidz,lidy,lidx+4)-lsxx_r(lidz,lidy,lidx-3))+
                  HC5*(lsxx_r(lidz,lidy,lidx+5)-lsxx_r(lidz,lidy,lidx-4)));
#elif FDOH ==6
    sxx_x_r = DTDH*(HC1*(lsxx_r(lidz,lidy,lidx+1)-lsxx_r(lidz,lidy,lidx))+
                  HC2*(lsxx_r(lidz,lidy,lidx+2)-lsxx_r(lidz,lidy,lidx-1))+
                  HC3*(lsxx_r(lidz,lidy,lidx+3)-lsxx_r(lidz,lidy,lidx-2))+
                  HC4*(lsxx_r(lidz,lidy,lidx+4)-lsxx_r(lidz,lidy,lidx-3))+
                  HC5*(lsxx_r(lidz,lidy,lidx+5)-lsxx_r(lidz,lidy,lidx-4))+
                  HC6*(lsxx_r(lidz,lidy,lidx+6)-lsxx_r(lidz,lidy,lidx-5)));
#endif
    
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lsyy_r(lidz,lidy,lidx)=syy_r(gidz,gidy,gidx);
        if (lidy<2*FDOH)
            lsyy_r(lidz,lidy-FDOH,lidx)=syy_r(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsyy_r(lidz,lidy+lsizey-3*FDOH,lidx)=syy_r(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsyy_r(lidz,lidy+FDOH,lidx)=syy_r(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsyy_r(lidz,lidy-lsizey+3*FDOH,lidx)=syy_r(gidz,gidy-lsizey+3*FDOH,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH ==1
    syy_y_r = DTDH*HC1*(lsyy_r(lidz,lidy+1,lidx) - lsyy_r(lidz,lidy,lidx));
#elif FDOH ==2
    syy_y_r = DTDH*(HC1*(lsyy_r(lidz,lidy+1,lidx) - lsyy_r(lidz,lidy,lidx))
                  +HC2*(lsyy_r(lidz,lidy+2,lidx) - lsyy_r(lidz,lidy-1,lidx)));
#elif FDOH ==3
    syy_y_r = DTDH*(HC1*(lsyy_r(lidz,lidy+1,lidx)-lsyy_r(lidz,lidy,lidx))+
                  HC2*(lsyy_r(lidz,lidy+2,lidx)-lsyy_r(lidz,lidy-1,lidx))+
                  HC3*(lsyy_r(lidz,lidy+3,lidx)-lsyy_r(lidz,lidy-2,lidx)));
#elif FDOH ==4
    syy_y_r = DTDH*(HC1*(lsyy_r(lidz,lidy+1,lidx)-lsyy_r(lidz,lidy,lidx))+
                  HC2*(lsyy_r(lidz,lidy+2,lidx)-lsyy_r(lidz,lidy-1,lidx))+
                  HC3*(lsyy_r(lidz,lidy+3,lidx)-lsyy_r(lidz,lidy-2,lidx))+
                  HC4*(lsyy_r(lidz,lidy+4,lidx)-lsyy_r(lidz,lidy-3,lidx)));
#elif FDOH ==5
    syy_y_r = DTDH*(HC1*(lsyy_r(lidz,lidy+1,lidx)-lsyy_r(lidz,lidy,lidx))+
                  HC2*(lsyy_r(lidz,lidy+2,lidx)-lsyy_r(lidz,lidy-1,lidx))+
                  HC3*(lsyy_r(lidz,lidy+3,lidx)-lsyy_r(lidz,lidy-2,lidx))+
                  HC4*(lsyy_r(lidz,lidy+4,lidx)-lsyy_r(lidz,lidy-3,lidx))+
                  HC5*(lsyy_r(lidz,lidy+5,lidx)-lsyy_r(lidz,lidy-4,lidx)));
#elif FDOH ==6
    syy_y_r = DTDH*(HC1*(lsyy_r(lidz,lidy+1,lidx)-lsyy_r(lidz,lidy,lidx))+
                  HC2*(lsyy_r(lidz,lidy+2,lidx)-lsyy_r(lidz,lidy-1,lidx))+
                  HC3*(lsyy_r(lidz,lidy+3,lidx)-lsyy_r(lidz,lidy-2,lidx))+
                  HC4*(lsyy_r(lidz,lidy+4,lidx)-lsyy_r(lidz,lidy-3,lidx))+
                  HC5*(lsyy_r(lidz,lidy+5,lidx)-lsyy_r(lidz,lidy-4,lidx))+
                  HC6*(lsyy_r(lidz,lidy+6,lidx)-lsyy_r(lidz,lidy-5,lidx)));
#endif
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lszz_r(lidz,lidy,lidx)=szz_r(gidz,gidy,gidx);
    if (lidz<2*FDOH)
        lszz_r(lidz-FDOH,lidy,lidx)=szz_r(gidz-FDOH,gidy,gidx);
    if (lidz>(lsizez-2*FDOH-1))
        lszz_r(lidz+FDOH,lidy,lidx)=szz_r(gidz+FDOH,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH ==1
    szz_z_r = DTDH*HC1*(lszz_r(lidz+1,lidy,lidx) - lszz_r(lidz,lidy,lidx));
#elif FDOH ==2
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidy,lidx) - lszz_r(lidz,lidy,lidx))
                  +HC2*(lszz_r(lidz+2,lidy,lidx) - lszz_r(lidz-1,lidy,lidx)));
#elif FDOH ==3
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidy,lidx)-lszz_r(lidz,lidy,lidx))+
                  HC2*(lszz_r(lidz+2,lidy,lidx)-lszz_r(lidz-1,lidy,lidx))+
                  HC3*(lszz_r(lidz+3,lidy,lidx)-lszz_r(lidz-2,lidy,lidx)));
#elif FDOH ==4
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidy,lidx)-lszz_r(lidz,lidy,lidx))+
                  HC2*(lszz_r(lidz+2,lidy,lidx)-lszz_r(lidz-1,lidy,lidx))+
                  HC3*(lszz_r(lidz+3,lidy,lidx)-lszz_r(lidz-2,lidy,lidx))+
                  HC4*(lszz_r(lidz+4,lidy,lidx)-lszz_r(lidz-3,lidy,lidx)));
#elif FDOH ==5
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidy,lidx)-lszz_r(lidz,lidy,lidx))+
                  HC2*(lszz_r(lidz+2,lidy,lidx)-lszz_r(lidz-1,lidy,lidx))+
                  HC3*(lszz_r(lidz+3,lidy,lidx)-lszz_r(lidz-2,lidy,lidx))+
                  HC4*(lszz_r(lidz+4,lidy,lidx)-lszz_r(lidz-3,lidy,lidx))+
                  HC5*(lszz_r(lidz+5,lidy,lidx)-lszz_r(lidz-4,lidy,lidx)));
#elif FDOH ==6
    szz_z_r = DTDH*(HC1*(lszz_r(lidz+1,lidy,lidx)-lszz_r(lidz,lidy,lidx))+
                  HC2*(lszz_r(lidz+2,lidy,lidx)-lszz_r(lidz-1,lidy,lidx))+
                  HC3*(lszz_r(lidz+3,lidy,lidx)-lszz_r(lidz-2,lidy,lidx))+
                  HC4*(lszz_r(lidz+4,lidy,lidx)-lszz_r(lidz-3,lidy,lidx))+
                  HC5*(lszz_r(lidz+5,lidy,lidx)-lszz_r(lidz-4,lidy,lidx))+
                  HC6*(lszz_r(lidz+6,lidy,lidx)-lszz_r(lidz-5,lidy,lidx)));
#endif
    
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lsxy_r(lidz,lidy,lidx)=sxy_r(gidz,gidy,gidx);
        if (lidy<2*FDOH)
            lsxy_r(lidz,lidy-FDOH,lidx)=sxy_r(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsxy_r(lidz,lidy+lsizey-3*FDOH,lidx)=sxy_r(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsxy_r(lidz,lidy+FDOH,lidx)=sxy_r(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsxy_r(lidz,lidy-lsizey+3*FDOH,lidx)=sxy_r(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lsxy_r(lidz,lidy,lidx-FDOH)=sxy_r(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxy_r(lidz,lidy,lidx+lsizex-3*FDOH)=sxy_r(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxy_r(lidz,lidy,lidx+FDOH)=sxy_r(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxy_r(lidz,lidy,lidx-lsizex+3*FDOH)=sxy_r(gidz,gidy,gidx-lsizex+3*FDOH);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH ==1
    sxy_y_r = DTDH*HC1*(lsxy_r(lidz,lidy,lidx)   - lsxy_r(lidz,lidy-1,lidx));
    sxy_x_r = DTDH*HC1*(lsxy_r(lidz,lidy,lidx)   - lsxy_r(lidz,lidy,lidx-1));
#elif FDOH ==2
    sxy_y_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)   - lsxy_r(lidz,lidy-1,lidx))
                  +HC2*(lsxy_r(lidz,lidy+1,lidx) - lsxy_r(lidz,lidy-2,lidx)));
    sxy_x_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)   - lsxy_r(lidz,lidy,lidx-1))
                  +HC2*(lsxy_r(lidz,lidy,lidx+1) - lsxy_r(lidz,lidy,lidx-2)));
#elif FDOH ==3
    sxy_y_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy-1,lidx))+
                  HC2*(lsxy_r(lidz,lidy+1,lidx)-lsxy_r(lidz,lidy-2,lidx))+
                  HC3*(lsxy_r(lidz,lidy+2,lidx)-lsxy_r(lidz,lidy-3,lidx)));
    
    sxy_x_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy,lidx-1))+
                  HC2*(lsxy_r(lidz,lidy,lidx+1)-lsxy_r(lidz,lidy,lidx-2))+
                  HC3*(lsxy_r(lidz,lidy,lidx+2)-lsxy_r(lidz,lidy,lidx-3)));
#elif FDOH ==4
    sxy_y_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy-1,lidx))+
                  HC2*(lsxy_r(lidz,lidy+1,lidx)-lsxy_r(lidz,lidy-2,lidx))+
                  HC3*(lsxy_r(lidz,lidy+2,lidx)-lsxy_r(lidz,lidy-3,lidx))+
                  HC4*(lsxy_r(lidz,lidy+3,lidx)-lsxy_r(lidz,lidy-4,lidx)));
    
    sxy_x_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy,lidx-1))+
                  HC2*(lsxy_r(lidz,lidy,lidx+1)-lsxy_r(lidz,lidy,lidx-2))+
                  HC3*(lsxy_r(lidz,lidy,lidx+2)-lsxy_r(lidz,lidy,lidx-3))+
                  HC4*(lsxy_r(lidz,lidy,lidx+3)-lsxy_r(lidz,lidy,lidx-4)));
#elif FDOH ==5
    sxy_y_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy-1,lidx))+
                  HC2*(lsxy_r(lidz,lidy+1,lidx)-lsxy_r(lidz,lidy-2,lidx))+
                  HC3*(lsxy_r(lidz,lidy+2,lidx)-lsxy_r(lidz,lidy-3,lidx))+
                  HC4*(lsxy_r(lidz,lidy+3,lidx)-lsxy_r(lidz,lidy-4,lidx))+
                  HC5*(lsxy_r(lidz,lidy+4,lidx)-lsxy_r(lidz,lidy-5,lidx)));
    
    sxy_x_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy,lidx-1))+
                  HC2*(lsxy_r(lidz,lidy,lidx+1)-lsxy_r(lidz,lidy,lidx-2))+
                  HC3*(lsxy_r(lidz,lidy,lidx+2)-lsxy_r(lidz,lidy,lidx-3))+
                  HC4*(lsxy_r(lidz,lidy,lidx+3)-lsxy_r(lidz,lidy,lidx-4))+
                  HC5*(lsxy_r(lidz,lidy,lidx+4)-lsxy_r(lidz,lidy,lidx-5)));
    
#elif FDOH ==6
    
    sxy_y_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy-1,lidx))+
                  HC2*(lsxy_r(lidz,lidy+1,lidx)-lsxy_r(lidz,lidy-2,lidx))+
                  HC3*(lsxy_r(lidz,lidy+2,lidx)-lsxy_r(lidz,lidy-3,lidx))+
                  HC4*(lsxy_r(lidz,lidy+3,lidx)-lsxy_r(lidz,lidy-4,lidx))+
                  HC5*(lsxy_r(lidz,lidy+4,lidx)-lsxy_r(lidz,lidy-5,lidx))+
                  HC6*(lsxy_r(lidz,lidy+5,lidx)-lsxy_r(lidz,lidy-6,lidx)));
    
    sxy_x_r = DTDH*(HC1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy,lidx-1))+
                  HC2*(lsxy_r(lidz,lidy,lidx+1)-lsxy_r(lidz,lidy,lidx-2))+
                  HC3*(lsxy_r(lidz,lidy,lidx+2)-lsxy_r(lidz,lidy,lidx-3))+
                  HC4*(lsxy_r(lidz,lidy,lidx+3)-lsxy_r(lidz,lidy,lidx-4))+
                  HC5*(lsxy_r(lidz,lidy,lidx+4)-lsxy_r(lidz,lidy,lidx-5))+
                  HC6*(lsxy_r(lidz,lidy,lidx+5)-lsxy_r(lidz,lidy,lidx-6)));
#endif
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lsyz_r(lidz,lidy,lidx)=syz_r(gidz,gidy,gidx);
        if (lidy<2*FDOH)
            lsyz_r(lidz,lidy-FDOH,lidx)=syz_r(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsyz_r(lidz,lidy+lsizey-3*FDOH,lidx)=syz_r(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsyz_r(lidz,lidy+FDOH,lidx)=syz_r(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsyz_r(lidz,lidy-lsizey+3*FDOH,lidx)=syz_r(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidz<2*FDOH)
            lsyz_r(lidz-FDOH,lidy,lidx)=syz_r(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lsyz_r(lidz+FDOH,lidy,lidx)=syz_r(gidz+FDOH,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH ==1
    syz_z_r = DTDH*HC1*(lsyz_r(lidz,lidy,lidx)   - lsyz_r(lidz-1,lidy,lidx));
    syz_y_r = DTDH*HC1*(lsyz_r(lidz,lidy,lidx)   - lsyz_r(lidz,lidy-1,lidx));
#elif FDOH ==2
    syz_z_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)   - lsyz_r(lidz-1,lidy,lidx))
                  +HC2*(lsyz_r(lidz+1,lidy,lidx) - lsyz_r(lidz-2,lidy,lidx)));
    syz_y_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)   - lsyz_r(lidz,lidy-1,lidx))
                  +HC2*(lsyz_r(lidz,lidy+1,lidx) - lsyz_r(lidz,lidy-2,lidx)));
#elif FDOH ==3
    syz_z_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz-1,lidy,lidx))+
                  HC2*(lsyz_r(lidz+1,lidy,lidx)-lsyz_r(lidz-2,lidy,lidx))+
                  HC3*(lsyz_r(lidz+2,lidy,lidx)-lsyz_r(lidz-3,lidy,lidx)));
    
    syz_y_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz,lidy-1,lidx))+
                  HC2*(lsyz_r(lidz,lidy+1,lidx)-lsyz_r(lidz,lidy-2,lidx))+
                  HC3*(lsyz_r(lidz,lidy+2,lidx)-lsyz_r(lidz,lidy-3,lidx)));
#elif FDOH ==4
    syz_z_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz-1,lidy,lidx))+
                  HC2*(lsyz_r(lidz+1,lidy,lidx)-lsyz_r(lidz-2,lidy,lidx))+
                  HC3*(lsyz_r(lidz+2,lidy,lidx)-lsyz_r(lidz-3,lidy,lidx))+
                  HC4*(lsyz_r(lidz+3,lidy,lidx)-lsyz_r(lidz-4,lidy,lidx)));
    
    syz_y_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)-lsyz_r(lidz,lidy-1,lidx))+
                  HC2*(lsyz_r(lidz,lidy+1,lidx)-lsyz_r(lidz,lidy-2,lidx))+
                  HC3*(lsyz_r(lidz,lidy+2,lidx)-lsyz_r(lidz,lidy-3,lidx))+
                  HC4*(lsyz_r(lidz,lidy+3,lidx)-lsyz_r(lidz,lidy-4,lidx)));
#elif FDOH ==5
    syz_z_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz-1,lidy,lidx))+
                  HC2*(lsyz_r(lidz+1,lidy,lidx)-lsyz_r(lidz-2,lidy,lidx))+
                  HC3*(lsyz_r(lidz+2,lidy,lidx)-lsyz_r(lidz-3,lidy,lidx))+
                  HC4*(lsyz_r(lidz+3,lidy,lidx)-lsyz_r(lidz-4,lidy,lidx))+
                  HC5*(lsyz_r(lidz+4,lidy,lidx)-lsyz_r(lidz-5,lidy,lidx)));
    
    syz_y_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz,lidy-1,lidx))+
                  HC2*(lsyz_r(lidz,lidy+1,lidx)-lsyz_r(lidz,lidy-2,lidx))+
                  HC3*(lsyz_r(lidz,lidy+2,lidx)-lsyz_r(lidz,lidy-3,lidx))+
                  HC4*(lsyz_r(lidz,lidy+3,lidx)-lsyz_r(lidz,lidy-4,lidx))+
                  HC5*(lsyz_r(lidz,lidy+4,lidx)-lsyz_r(lidz,lidy-5,lidx)));
#elif FDOH ==6
    syz_z_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz-1,lidy,lidx))+
                  HC2*(lsyz_r(lidz+1,lidy,lidx)-lsyz_r(lidz-2,lidy,lidx))+
                  HC3*(lsyz_r(lidz+2,lidy,lidx)-lsyz_r(lidz-3,lidy,lidx))+
                  HC4*(lsyz_r(lidz+3,lidy,lidx)-lsyz_r(lidz-4,lidy,lidx))+
                  HC5*(lsyz_r(lidz+4,lidy,lidx)-lsyz_r(lidz-5,lidy,lidx))+
                  HC6*(lsyz_r(lidz+5,lidy,lidx)-lsyz_r(lidz-6,lidy,lidx)));
    
    syz_y_r = DTDH*(HC1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz,lidy-1,lidx))+
                  HC2*(lsyz_r(lidz,lidy+1,lidx)-lsyz_r(lidz,lidy-2,lidx))+
                  HC3*(lsyz_r(lidz,lidy+2,lidx)-lsyz_r(lidz,lidy-3,lidx))+
                  HC4*(lsyz_r(lidz,lidy+3,lidx)-lsyz_r(lidz,lidy-4,lidx))+
                  HC5*(lsyz_r(lidz,lidy+4,lidx)-lsyz_r(lidz,lidy-5,lidx))+
                  HC6*(lsyz_r(lidz,lidy+5,lidx)-lsyz_r(lidz,lidy-6,lidx)));
#endif
    
#if LOCAL_OFF==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lsxz_r(lidz,lidy,lidx)=sxz_r(gidz,gidy,gidx);
    
        if (lidx<2*FDOH)
            lsxz_r(lidz,lidy,lidx-FDOH)=sxz_r(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxz_r(lidz,lidy,lidx+lsizex-3*FDOH)=sxz_r(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxz_r(lidz,lidy,lidx+FDOH)=sxz_r(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxz_r(lidz,lidy,lidx-lsizex+3*FDOH)=sxz_r(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lsxz_r(lidz-FDOH,lidy,lidx)=sxz_r(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lsxz_r(lidz+FDOH,lidy,lidx)=sxz_r(gidz+FDOH,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   FDOH ==1
    sxz_z_r = DTDH*HC1*(lsxz_r(lidz,lidy,lidx)   - lsxz_r(lidz-1,lidy,lidx));
    sxz_x_r = DTDH*HC1*(lsxz_r(lidz,lidy,lidx)   - lsxz_r(lidz,lidy,lidx-1));
#elif FDOH ==2
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)   - lsxz_r(lidz-1,lidy,lidx))
                  +HC2*(lsxz_r(lidz+1,lidy,lidx) - lsxz_r(lidz-2,lidy,lidx)));
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)   - lsxz_r(lidz,lidy,lidx-1))
                  +HC2*(lsxz_r(lidz,lidy,lidx+1) - lsxz_r(lidz,lidy,lidx-2)));
    
#elif FDOH ==3
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz-1,lidy,lidx))+
                  HC2*(lsxz_r(lidz+1,lidy,lidx)-lsxz_r(lidz-2,lidy,lidx))+
                  HC3*(lsxz_r(lidz+2,lidy,lidx)-lsxz_r(lidz-3,lidy,lidx)));
    
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz,lidy,lidx-1))+
                  HC2*(lsxz_r(lidz,lidy,lidx+1)-lsxz_r(lidz,lidy,lidx-2))+
                  HC3*(lsxz_r(lidz,lidy,lidx+2)-lsxz_r(lidz,lidy,lidx-3)));
#elif FDOH ==4
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz-1,lidy,lidx))+
                  HC2*(lsxz_r(lidz+1,lidy,lidx)-lsxz_r(lidz-2,lidy,lidx))+
                  HC3*(lsxz_r(lidz+2,lidy,lidx)-lsxz_r(lidz-3,lidy,lidx))+
                  HC4*(lsxz_r(lidz+3,lidy,lidx)-lsxz_r(lidz-4,lidy,lidx)));
    
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz,lidy,lidx-1))+
                  HC2*(lsxz_r(lidz,lidy,lidx+1)-lsxz_r(lidz,lidy,lidx-2))+
                  HC3*(lsxz_r(lidz,lidy,lidx+2)-lsxz_r(lidz,lidy,lidx-3))+
                  HC4*(lsxz_r(lidz,lidy,lidx+3)-lsxz_r(lidz,lidy,lidx-4)));
#elif FDOH ==5
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz-1,lidy,lidx))+
                  HC2*(lsxz_r(lidz+1,lidy,lidx)-lsxz_r(lidz-2,lidy,lidx))+
                  HC3*(lsxz_r(lidz+2,lidy,lidx)-lsxz_r(lidz-3,lidy,lidx))+
                  HC4*(lsxz_r(lidz+3,lidy,lidx)-lsxz_r(lidz-4,lidy,lidx))+
                  HC5*(lsxz_r(lidz+4,lidy,lidx)-lsxz_r(lidz-5,lidy,lidx)));
    
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz,lidy,lidx-1))+
                  HC2*(lsxz_r(lidz,lidy,lidx+1)-lsxz_r(lidz,lidy,lidx-2))+
                  HC3*(lsxz_r(lidz,lidy,lidx+2)-lsxz_r(lidz,lidy,lidx-3))+
                  HC4*(lsxz_r(lidz,lidy,lidx+3)-lsxz_r(lidz,lidy,lidx-4))+
                  HC5*(lsxz_r(lidz,lidy,lidx+4)-lsxz_r(lidz,lidy,lidx-5)));
#elif FDOH ==6
    sxz_z_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz-1,lidy,lidx))+
                  HC2*(lsxz_r(lidz+1,lidy,lidx)-lsxz_r(lidz-2,lidy,lidx))+
                  HC3*(lsxz_r(lidz+2,lidy,lidx)-lsxz_r(lidz-3,lidy,lidx))+
                  HC4*(lsxz_r(lidz+3,lidy,lidx)-lsxz_r(lidz-4,lidy,lidx))+
                  HC5*(lsxz_r(lidz+4,lidy,lidx)-lsxz_r(lidz-5,lidy,lidx))+
                  HC6*(lsxz_r(lidz+5,lidy,lidx)-lsxz_r(lidz-6,lidy,lidx)));
    
    
    sxz_x_r = DTDH*(HC1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz,lidy,lidx-1))+
                  HC2*(lsxz_r(lidz,lidy,lidx+1)-lsxz_r(lidz,lidy,lidx-2))+
                  HC3*(lsxz_r(lidz,lidy,lidx+2)-lsxz_r(lidz,lidy,lidx-3))+
                  HC4*(lsxz_r(lidz,lidy,lidx+3)-lsxz_r(lidz,lidy,lidx-4))+
                  HC5*(lsxz_r(lidz,lidy,lidx+4)-lsxz_r(lidz,lidy,lidx-5))+
                  HC6*(lsxz_r(lidz,lidy,lidx+5)-lsxz_r(lidz,lidy,lidx-6)));
    
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

// Backpropagate the forward velocity
#if BACK_PROP_TYPE==1
    {
        lvx=((sxx_x + sxy_y + sxz_z)/rip(gidz,gidy,gidx))+amp.x;
        lvy=((syy_y + sxy_x + syz_z)/rjp(gidz,gidy,gidx))+amp.y;
        lvz=((szz_z + sxz_x + syz_y)/rkp(gidz,gidy,gidx))+amp.z;
        
        vx(gidz,gidy,gidx)-= lvx;
        vy(gidz,gidy,gidx)-= lvy;
        vz(gidz,gidy,gidx)-= lvz;
        
        // Inject the boundary values
        m=evarm(gidz, gidy, gidx);
        if (m!=-1){
            vx(gidz,gidy, gidx)= vxbnd[m];
            vy(gidz,gidy, gidx)= vybnd[m];
            vz(gidz,gidy, gidx)= vzbnd[m];
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
            
            psi_sxz_z(k,j,i) = b_z[ind+1] * psi_sxz_z(k,j,i) + a_z[ind+1] * sxz_z_r;
            sxz_z_r = sxz_z_r / K_z[ind+1] + psi_sxz_z(k,j,i);
            psi_syz_z(k,j,i) = b_z[ind+1] * psi_syz_z(k,j,i) + a_z[ind+1] * syz_z_r;
            syz_z_r = syz_z_r / K_z[ind+1] + psi_syz_z(k,j,i);
            psi_szz_z(k,j,i) = b_z_half[ind] * psi_szz_z(k,j,i) + a_z_half[ind] * szz_z_r;
            szz_z_r = szz_z_r / K_z_half[ind] + psi_szz_z(k,j,i);
            
        }
        
#if FREESURF==0
        else if (gidz-FDOH<NAB){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;
            
            psi_sxz_z(k,j,i) = b_z[k] * psi_sxz_z(k,j,i) + a_z[k] * sxz_z_r;
            sxz_z_r = sxz_z_r / K_z[k] + psi_sxz_z(k,j,i);
            psi_syz_z(k,j,i) = b_z[k] * psi_syz_z(k,j,i) + a_z[k] * syz_z_r;
            syz_z_r = syz_z_r / K_z[k] + psi_syz_z(k,j,i);
            psi_szz_z(k,j,i) = b_z_half[k] * psi_szz_z(k,j,i) + a_z_half[k] * szz_z_r;
            szz_z_r = szz_z_r / K_z_half[k] + psi_szz_z(k,j,i);
            
        }
#endif
        
        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;
            
            psi_sxy_y(k,j,i) = b_y[j] * psi_sxy_y(k,j,i) + a_y[j] * sxy_y_r;
            sxy_y_r = sxy_y_r / K_y[j] + psi_sxy_y(k,j,i);
            psi_syy_y(k,j,i) = b_y_half[j] * psi_syy_y(k,j,i) + a_y_half[j] * syy_y_r;
            syy_y_r = syy_y_r / K_y_half[j] + psi_syy_y(k,j,i);
            psi_syz_y(k,j,i) = b_y[j] * psi_syz_y(k,j,i) + a_y[j] * syz_y_r;
            syz_y_r = syz_y_r / K_y[j] + psi_syz_y(k,j,i);
            
        }
        
        else if (gidy>NY-NAB-FDOH-1){
            
            i =gidx-FDOH;
            j =gidy - NY+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-j;
            
            psi_sxy_y(k,j,i) = b_y[ind+1] * psi_sxy_y(k,j,i) + a_y[ind+1] * sxy_y_r;
            sxy_y_r = sxy_y_r / K_y[ind+1] + psi_sxy_y(k,j,i);
            psi_syy_y(k,j,i) = b_y_half[ind] * psi_syy_y(k,j,i) + a_y_half[ind] * syy_y_r;
            syy_y_r = syy_y_r / K_y_half[ind] + psi_syy_y(k,j,i);
            psi_syz_y(k,j,i) = b_y[ind+1] * psi_syz_y(k,j,i) + a_y[ind+1] * syz_y_r;
            syz_y_r = syz_y_r / K_y[ind+1] + psi_syz_y(k,j,i);
            
            
        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;
            
            psi_sxx_x(k,j,i) = b_x_half[i] * psi_sxx_x(k,j,i) + a_x_half[i] * sxx_x_r;
            sxx_x_r = sxx_x_r / K_x_half[i] + psi_sxx_x(k,j,i);
            psi_sxy_x(k,j,i) = b_x[i] * psi_sxy_x(k,j,i) + a_x[i] * sxy_x_r;
            sxy_x_r = sxy_x_r / K_x[i] + psi_sxy_x(k,j,i);
            psi_sxz_x(k,j,i) = b_x[i] * psi_sxz_x(k,j,i) + a_x[i] * sxz_x_r;
            sxz_x_r = sxz_x_r / K_x[i] + psi_sxz_x(k,j,i);
            
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            
            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            
            psi_sxx_x(k,j,i) = b_x_half[ind] * psi_sxx_x(k,j,i) + a_x_half[ind] * sxx_x_r;
            sxx_x_r = sxx_x_r / K_x_half[ind] + psi_sxx_x(k,j,i);
            psi_sxy_x(k,j,i) = b_x[ind+1] * psi_sxy_x(k,j,i) + a_x[ind+1] * sxy_x_r;
            sxy_x_r = sxy_x_r / K_x[ind+1] + psi_sxy_x(k,j,i);
            psi_sxz_x(k,j,i) = b_x[ind+1] * psi_sxz_x(k,j,i) + a_x[ind+1] * sxz_x_r;
            sxz_x_r = sxz_x_r / K_x[ind+1] + psi_sxz_x(k,j,i);
            
            
            
        }
#endif
    }
#endif
    
    // Update adjoint velocities
    lvx=((sxx_x_r + sxy_y_r + sxz_z_r)/rip(gidz,gidy,gidx));
    lvy=((syy_y_r + sxy_x_r + syz_z_r)/rjp(gidz,gidy,gidx));
    lvz=((szz_z_r + sxz_x_r + syz_y_r)/rkp(gidz,gidy,gidx));
    vx_r(gidz,gidy,gidx)+= lvx;
    vy_r(gidz,gidy,gidx)+= lvy;
    vz_r(gidz,gidy,gidx)+= lvz;
 
    
// Absorbing boundary
#if ABS_TYPE==2
    {
        if (gidz-FDOH<NAB){
            vx_r(gidz,gidy,gidx)*=taper[gidz-FDOH];
            vy_r(gidz,gidy,gidx)*=taper[gidz-FDOH];
            vz_r(gidz,gidy,gidx)*=taper[gidz-FDOH];
        }
        
        if (gidz>NZ-NAB-FDOH-1){
            vx_r(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            vy_r(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            vz_r(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
        }
        
        if (gidy-FDOH<NAB){
            vx_r(gidz,gidy,gidx)*=taper[gidy-FDOH];
            vy_r(gidz,gidy,gidx)*=taper[gidy-FDOH];
            vz_r(gidz,gidy,gidx)*=taper[gidy-FDOH];
        }
        
        if (gidy>NY-NAB-FDOH-1){
            vx_r(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            vy_r(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            vz_r(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            vx_r(gidz,gidy,gidx)*=taper[gidx-FDOH];
            vy_r(gidz,gidy,gidx)*=taper[gidx-FDOH];
            vz_r(gidz,gidy,gidx)*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            vx_r(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            vy_r(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            vz_r(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif

// Density gradient calculation on the fly    
#if BACK_PROP_TYPE==1
    gradrho(gidz,gidx)+=vx(gidz,gidx)*lvx+vy(gidz,gidx)*lvy+vz(gidz,gidx)*lvz;
#endif
    
#if GRADSRCOUT==1
    //TODO
//    if (nsrc>0){
//        
//        
//        for (int srci=0; srci<nsrc; srci++){
//            
//            
//            int i=(int)(srcpos_loc(0,srci)/DH-0.5)+FDOH;
//            int j=(int)(srcpos_loc(1,srci)/DH-0.5)+FDOH;
//            int k=(int)(srcpos_loc(2,srci)/DH-0.5)+FDOH;
//            
//            if (i==gidx && j==gidy && k==gidz){
//                
//                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
//                
//                if (SOURCE_TYPE==2){
//                    /* single force in x */
//                    gradsrc(srci,nt)+= vx_r(gidz,gidy,gidx)/rip(gidx,gidy,gidz)/(DH*DH*DH);
//                }
//                else if (SOURCE_TYPE==3){
//                    /* single force in y */
//                    
//                    gradsrc(srci,nt)+= vy_r(gidz,gidy,gidx)/rip(gidx,gidy,gidz)/(DH*DH*DH);
//                }
//                else if (SOURCE_TYPE==4){
//                    /* single force in z */
//                    
//                    gradsrc(srci,nt)+= vz_r(gidz,gidy,gidx)/rip(gidx,gidy,gidz)/(DH*DH*DH);
//                }
//                
//            }
//        }
//        
//        
//    }
#endif
    
}

