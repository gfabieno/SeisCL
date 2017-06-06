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
#define lbnd (fdoh+nab)

#define rho(z,y,x)     rho[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rip(z,y,x)     rip[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rjp(z,y,x)     rjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,y,x)     rkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipjp(z,y,x) uipjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define ujpkp(z,y,x) ujpkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipkp(z,y,x) uipkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define u(z,y,x)         u[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define pi(z,y,x)       pi[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradrho(z,y,x)   gradrho[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradM(z,y,x)   gradM[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradmu(z,y,x)   gradmu[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
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

#define vx_r(z,y,x)   vx_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vy_r(z,y,x)   vy_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz_r(z,y,x)   vz_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxx_r(z,y,x) sxx_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syy_r(z,y,x) syy_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define szz_r(z,y,x) szz_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxy_r(z,y,x) sxy_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syz_r(z,y,x) syz_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxz_r(z,y,x) sxz_r[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]

#define psi_sxx_x(z,y,x) psi_sxx_x[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_sxy_x(z,y,x) psi_sxy_x[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_sxz_x(z,y,x) psi_sxz_x[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_sxy_y(z,y,x) psi_sxy_y[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_syy_y(z,y,x) psi_syy_y[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_syz_y(z,y,x) psi_syz_y[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_sxz_z(z,y,x) psi_sxz_z[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_syz_z(z,y,x) psi_syz_z[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_szz_z(z,y,x) psi_szz_z[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]

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


float3 ssource(int gidz, int gidy, int gidx,  int nsrc, __global float *srcpos_loc, __global float *signals, int nt, __global float * rip, __global float * rjp, __global float * rkp){
    
    float3 ampv={0.0,0.0,0.0};
    if (nsrc>0){
        
        
        for (int srci=0; srci<nsrc; srci++){
            
            
            int i=(int)(srcpos_loc(0,srci)/DH-0.5)+fdoh;
            int j=(int)(srcpos_loc(1,srci)/DH-0.5)+fdoh;
            int k=(int)(srcpos_loc(2,srci)/DH-0.5)+fdoh;
            
            if (i==gidx && j==gidy && k==gidz){
                //                float azi_rad=srcpos_loc(6,srci) * PI / 180;
                
                
                float amp=(DT*signals(srci,nt))/(DH*DH*DH); // scaled force amplitude with F= 1N
                
                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
                
                if (SOURCE_TYPE==2){
                    /* single force in x */
                    ampv.x  +=  amp/rip(k,j,i-offset);
                }
                else if (SOURCE_TYPE==3){
                    /* single force in y */
                    
                    ampv.y  +=  amp/rjp(k,j,i-offset);
                }
                else if (SOURCE_TYPE==4){
                    /* single force in z */
                    
                    ampv.z  +=  amp/rkp(k,j,i-offset);
                }
                
            }
        }
        
        
    }
    
    return ampv;
    
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


__kernel void update_adjv(int offcomm, int nsrc,  int ng, int nt,
                          __global float *vx,         __global float *vy,      __global float *vz,
                          __global float *sxx,        __global float *syy,     __global float *szz,
                          __global float *sxy,        __global float *syz,     __global float *sxz,
                          __global float *vxbnd,      __global float *vybnd,   __global float *vzbnd,
                          __global float *sxxbnd,     __global float *syybnd,  __global float *szzbnd,
                          __global float *sxybnd,     __global float *syzbnd,  __global float *sxzbnd,
                          __global float *vx_r,       __global float *vy_r,    __global float *vz_r,
                          __global float *sxx_r,      __global float *syy_r,   __global float *szz_r,
                          __global float *sxy_r,      __global float *syz_r,   __global float *sxz_r,
                          __global float *rx,         __global float *ry,      __global float *rz,
                          __global float *rip,        __global float *rjp,     __global float *rkp,
                          __global float *srcpos_loc, __global float *signals, __global float *rec_pos,
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
#elif local_off==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*fdoh);
    int glsizey = (NY-2*fdoh);
    int gidz = gid%glsizez+fdoh;
    int gidy = (gid/glsizez)%glsizey+fdoh;
    int gidx = gid/(glsizez*glsizey)+fdoh+offcomm;
    
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
#if back_prop_type==1
    {

#if local_off==0
        lsxx(lidz,lidy,lidx)=sxx(gidz,gidy,gidx);
        if (lidx<2*fdoh)
            lsxx(lidz,lidy,lidx-fdoh)=sxx(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxx(lidz,lidy,lidx+lsizex-3*fdoh)=sxx(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxx(lidz,lidy,lidx+fdoh)=sxx(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxx(lidz,lidy,lidx-lsizex+3*fdoh)=sxx(gidz,gidy,gidx-lsizex+3*fdoh);
        
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        sxx_x = dtdh*hc1*(lsxx(lidz,lidy,lidx+1) - lsxx(lidz,lidy,lidx));
#elif fdoh ==2
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1) - lsxx(lidz,lidy,lidx))
                      +hc2*(lsxx(lidz,lidy,lidx+2) - lsxx(lidz,lidy,lidx-1)));
#elif fdoh ==3
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      hc2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      hc3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2)));
#elif fdoh ==4
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      hc2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      hc3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2))+
                      hc4*(lsxx(lidz,lidy,lidx+4)-lsxx(lidz,lidy,lidx-3)));
#elif fdoh ==5
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      hc2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      hc3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2))+
                      hc4*(lsxx(lidz,lidy,lidx+4)-lsxx(lidz,lidy,lidx-3))+
                      hc5*(lsxx(lidz,lidy,lidx+5)-lsxx(lidz,lidy,lidx-4)));
#elif fdoh ==6
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      hc2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      hc3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2))+
                      hc4*(lsxx(lidz,lidy,lidx+4)-lsxx(lidz,lidy,lidx-3))+
                      hc5*(lsxx(lidz,lidy,lidx+5)-lsxx(lidz,lidy,lidx-4))+
                      hc6*(lsxx(lidz,lidy,lidx+6)-lsxx(lidz,lidy,lidx-5)));
#endif
        
        
#if local_off==0
        lsyy(lidz,lidy,lidx)=syy(gidz,gidy,gidx);
        if (lidy<2*fdoh)
            lsyy(lidz,lidy-fdoh,lidx)=syy(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lsyy(lidz,lidy+lsizey-3*fdoh,lidx)=syy(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lsyy(lidz,lidy+fdoh,lidx)=syy(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lsyy(lidz,lidy-lsizey+3*fdoh,lidx)=syy(gidz,gidy-lsizey+3*fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        syy_y = dtdh*hc1*(lsyy(lidz,lidy+1,lidx) - lsyy(lidz,lidy,lidx));
#elif fdoh ==2
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx) - lsyy(lidz,lidy,lidx))
                      +hc2*(lsyy(lidz,lidy+2,lidx) - lsyy(lidz,lidy-1,lidx)));
#elif fdoh ==3
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      hc2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      hc3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx)));
#elif fdoh ==4
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      hc2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      hc3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx))+
                      hc4*(lsyy(lidz,lidy+4,lidx)-lsyy(lidz,lidy-3,lidx)));
#elif fdoh ==5
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      hc2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      hc3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx))+
                      hc4*(lsyy(lidz,lidy+4,lidx)-lsyy(lidz,lidy-3,lidx))+
                      hc5*(lsyy(lidz,lidy+5,lidx)-lsyy(lidz,lidy-4,lidx)));
#elif fdoh ==6
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      hc2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      hc3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx))+
                      hc4*(lsyy(lidz,lidy+4,lidx)-lsyy(lidz,lidy-3,lidx))+
                      hc5*(lsyy(lidz,lidy+5,lidx)-lsyy(lidz,lidy-4,lidx))+
                      hc6*(lsyy(lidz,lidy+6,lidx)-lsyy(lidz,lidy-5,lidx)));
#endif
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lszz(lidz,lidy,lidx)=szz(gidz,gidy,gidx);
        if (lidz<2*fdoh)
            lszz(lidz-fdoh,lidy,lidx)=szz(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lszz(lidz+fdoh,lidy,lidx)=szz(gidz+fdoh,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        szz_z = dtdh*hc1*(lszz(lidz+1,lidy,lidx) - lszz(lidz,lidy,lidx));
#elif fdoh ==2
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx) - lszz(lidz,lidy,lidx))
                      +hc2*(lszz(lidz+2,lidy,lidx) - lszz(lidz-1,lidy,lidx)));
#elif fdoh ==3
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      hc2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      hc3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx)));
#elif fdoh ==4
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      hc2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      hc3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx))+
                      hc4*(lszz(lidz+4,lidy,lidx)-lszz(lidz-3,lidy,lidx)));
#elif fdoh ==5
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      hc2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      hc3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx))+
                      hc4*(lszz(lidz+4,lidy,lidx)-lszz(lidz-3,lidy,lidx))+
                      hc5*(lszz(lidz+5,lidy,lidx)-lszz(lidz-4,lidy,lidx)));
#elif fdoh ==6
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      hc2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      hc3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx))+
                      hc4*(lszz(lidz+4,lidy,lidx)-lszz(lidz-3,lidy,lidx))+
                      hc5*(lszz(lidz+5,lidy,lidx)-lszz(lidz-4,lidy,lidx))+
                      hc6*(lszz(lidz+6,lidy,lidx)-lszz(lidz-5,lidy,lidx)));
#endif
        
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsxy(lidz,lidy,lidx)=sxy(gidz,gidy,gidx);
        if (lidy<2*fdoh)
            lsxy(lidz,lidy-fdoh,lidx)=sxy(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lsxy(lidz,lidy+lsizey-3*fdoh,lidx)=sxy(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lsxy(lidz,lidy+fdoh,lidx)=sxy(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lsxy(lidz,lidy-lsizey+3*fdoh,lidx)=sxy(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lsxy(lidz,lidy,lidx-fdoh)=sxy(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxy(lidz,lidy,lidx+lsizex-3*fdoh)=sxy(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxy(lidz,lidy,lidx+fdoh)=sxy(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxy(lidz,lidy,lidx-lsizex+3*fdoh)=sxy(gidz,gidy,gidx-lsizex+3*fdoh);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        sxy_y = dtdh*hc1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy-1,lidx));
        sxy_x = dtdh*hc1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy,lidx-1));
#elif fdoh ==2
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy-1,lidx))
                      +hc2*(lsxy(lidz,lidy+1,lidx) - lsxy(lidz,lidy-2,lidx)));
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy,lidx-1))
                      +hc2*(lsxy(lidz,lidy,lidx+1) - lsxy(lidz,lidy,lidx-2)));
#elif fdoh ==3
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      hc2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      hc3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx)));
        
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      hc2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      hc3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3)));
#elif fdoh ==4
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      hc2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      hc3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx))+
                      hc4*(lsxy(lidz,lidy+3,lidx)-lsxy(lidz,lidy-4,lidx)));
        
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      hc2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      hc3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3))+
                      hc4*(lsxy(lidz,lidy,lidx+3)-lsxy(lidz,lidy,lidx-4)));
#elif fdoh ==5
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      hc2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      hc3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx))+
                      hc4*(lsxy(lidz,lidy+3,lidx)-lsxy(lidz,lidy-4,lidx))+
                      hc5*(lsxy(lidz,lidy+4,lidx)-lsxy(lidz,lidy-5,lidx)));
        
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      hc2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      hc3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3))+
                      hc4*(lsxy(lidz,lidy,lidx+3)-lsxy(lidz,lidy,lidx-4))+
                      hc5*(lsxy(lidz,lidy,lidx+4)-lsxy(lidz,lidy,lidx-5)));
        
#elif fdoh ==6
        
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      hc2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      hc3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx))+
                      hc4*(lsxy(lidz,lidy+3,lidx)-lsxy(lidz,lidy-4,lidx))+
                      hc5*(lsxy(lidz,lidy+4,lidx)-lsxy(lidz,lidy-5,lidx))+
                      hc6*(lsxy(lidz,lidy+5,lidx)-lsxy(lidz,lidy-6,lidx)));
        
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      hc2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      hc3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3))+
                      hc4*(lsxy(lidz,lidy,lidx+3)-lsxy(lidz,lidy,lidx-4))+
                      hc5*(lsxy(lidz,lidy,lidx+4)-lsxy(lidz,lidy,lidx-5))+
                      hc6*(lsxy(lidz,lidy,lidx+5)-lsxy(lidz,lidy,lidx-6)));
#endif
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsyz(lidz,lidy,lidx)=syz(gidz,gidy,gidx);
        if (lidy<2*fdoh)
            lsyz(lidz,lidy-fdoh,lidx)=syz(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lsyz(lidz,lidy+lsizey-3*fdoh,lidx)=syz(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lsyz(lidz,lidy+fdoh,lidx)=syz(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lsyz(lidz,lidy-lsizey+3*fdoh,lidx)=syz(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidz<2*fdoh)
            lsyz(lidz-fdoh,lidy,lidx)=syz(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lsyz(lidz+fdoh,lidy,lidx)=syz(gidz+fdoh,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        syz_z = dtdh*hc1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz-1,lidy,lidx));
        syz_y = dtdh*hc1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz,lidy-1,lidx));
#elif fdoh ==2
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz-1,lidy,lidx))
                      +hc2*(lsyz(lidz+1,lidy,lidx) - lsyz(lidz-2,lidy,lidx)));
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz,lidy-1,lidx))
                      +hc2*(lsyz(lidz,lidy+1,lidx) - lsyz(lidz,lidy-2,lidx)));
#elif fdoh ==3
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      hc2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      hc3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx)));
        
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz,lidy-1,lidx))+
                      hc2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      hc3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx)));
#elif fdoh ==4
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      hc2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      hc3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx))+
                      hc4*(lsyz(lidz+3,lidy,lidx)-lsyz(lidz-4,lidy,lidx)));
        
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)-lsyz(lidz,lidy-1,lidx))+
                      hc2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      hc3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx))+
                      hc4*(lsyz(lidz,lidy+3,lidx)-lsyz(lidz,lidy-4,lidx)));
#elif fdoh ==5
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      hc2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      hc3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx))+
                      hc4*(lsyz(lidz+3,lidy,lidx)-lsyz(lidz-4,lidy,lidx))+
                      hc5*(lsyz(lidz+4,lidy,lidx)-lsyz(lidz-5,lidy,lidx)));
        
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz,lidy-1,lidx))+
                      hc2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      hc3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx))+
                      hc4*(lsyz(lidz,lidy+3,lidx)-lsyz(lidz,lidy-4,lidx))+
                      hc5*(lsyz(lidz,lidy+4,lidx)-lsyz(lidz,lidy-5,lidx)));
#elif fdoh ==6
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      hc2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      hc3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx))+
                      hc4*(lsyz(lidz+3,lidy,lidx)-lsyz(lidz-4,lidy,lidx))+
                      hc5*(lsyz(lidz+4,lidy,lidx)-lsyz(lidz-5,lidy,lidx))+
                      hc6*(lsyz(lidz+5,lidy,lidx)-lsyz(lidz-6,lidy,lidx)));
        
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz,lidy-1,lidx))+
                      hc2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      hc3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx))+
                      hc4*(lsyz(lidz,lidy+3,lidx)-lsyz(lidz,lidy-4,lidx))+
                      hc5*(lsyz(lidz,lidy+4,lidx)-lsyz(lidz,lidy-5,lidx))+
                      hc6*(lsyz(lidz,lidy+5,lidx)-lsyz(lidz,lidy-6,lidx)));
#endif
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsxz(lidz,lidy,lidx)=sxz(gidz,gidy,gidx);
        
        if (lidx<2*fdoh)
            lsxz(lidz,lidy,lidx-fdoh)=sxz(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxz(lidz,lidy,lidx+lsizex-3*fdoh)=sxz(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxz(lidz,lidy,lidx+fdoh)=sxz(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxz(lidz,lidy,lidx-lsizex+3*fdoh)=sxz(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lsxz(lidz-fdoh,lidy,lidx)=sxz(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lsxz(lidz+fdoh,lidy,lidx)=sxz(gidz+fdoh,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        sxz_z = dtdh*hc1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz-1,lidy,lidx));
        sxz_x = dtdh*hc1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz,lidy,lidx-1));
#elif fdoh ==2
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz-1,lidy,lidx))
                      +hc2*(lsxz(lidz+1,lidy,lidx) - lsxz(lidz-2,lidy,lidx)));
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz,lidy,lidx-1))
                      +hc2*(lsxz(lidz,lidy,lidx+1) - lsxz(lidz,lidy,lidx-2)));
        
#elif fdoh ==3
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      hc2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      hc3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      hc2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      hc3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3)));
#elif fdoh ==4
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      hc2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      hc3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx))+
                      hc4*(lsxz(lidz+3,lidy,lidx)-lsxz(lidz-4,lidy,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      hc2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      hc3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3))+
                      hc4*(lsxz(lidz,lidy,lidx+3)-lsxz(lidz,lidy,lidx-4)));
#elif fdoh ==5
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      hc2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      hc3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx))+
                      hc4*(lsxz(lidz+3,lidy,lidx)-lsxz(lidz-4,lidy,lidx))+
                      hc5*(lsxz(lidz+4,lidy,lidx)-lsxz(lidz-5,lidy,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      hc2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      hc3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3))+
                      hc4*(lsxz(lidz,lidy,lidx+3)-lsxz(lidz,lidy,lidx-4))+
                      hc5*(lsxz(lidz,lidy,lidx+4)-lsxz(lidz,lidy,lidx-5)));
#elif fdoh ==6
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      hc2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      hc3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx))+
                      hc4*(lsxz(lidz+3,lidy,lidx)-lsxz(lidz-4,lidy,lidx))+
                      hc5*(lsxz(lidz+4,lidy,lidx)-lsxz(lidz-5,lidy,lidx))+
                      hc6*(lsxz(lidz+5,lidy,lidx)-lsxz(lidz-6,lidy,lidx)));
        
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      hc2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      hc3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3))+
                      hc4*(lsxz(lidz,lidy,lidx+3)-lsxz(lidz,lidy,lidx-4))+
                      hc5*(lsxz(lidz,lidy,lidx+4)-lsxz(lidz,lidy,lidx-5))+
                      hc6*(lsxz(lidz,lidy,lidx+5)-lsxz(lidz,lidy,lidx-6)));
        
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
    
// Calculation of the stress spatial derivatives of the adjoint wavefield
    {
#if local_off==0
    lsxx_r(lidz,lidy,lidx)=sxx_r(gidz,gidy,gidx);
        if (lidx<2*fdoh)
            lsxx_r(lidz,lidy,lidx-fdoh)=sxx_r(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxx_r(lidz,lidy,lidx+lsizex-3*fdoh)=sxx_r(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxx_r(lidz,lidy,lidx+fdoh)=sxx_r(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxx_r(lidz,lidy,lidx-lsizex+3*fdoh)=sxx_r(gidz,gidy,gidx-lsizex+3*fdoh);
    
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh ==1
    sxx_x_r = dtdh*hc1*(lsxx_r(lidz,lidy,lidx+1) - lsxx_r(lidz,lidy,lidx));
#elif fdoh ==2
    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidy,lidx+1) - lsxx_r(lidz,lidy,lidx))
                  +hc2*(lsxx_r(lidz,lidy,lidx+2) - lsxx_r(lidz,lidy,lidx-1)));
#elif fdoh ==3
    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidy,lidx+1)-lsxx_r(lidz,lidy,lidx))+
                  hc2*(lsxx_r(lidz,lidy,lidx+2)-lsxx_r(lidz,lidy,lidx-1))+
                  hc3*(lsxx_r(lidz,lidy,lidx+3)-lsxx_r(lidz,lidy,lidx-2)));
#elif fdoh ==4
    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidy,lidx+1)-lsxx_r(lidz,lidy,lidx))+
                  hc2*(lsxx_r(lidz,lidy,lidx+2)-lsxx_r(lidz,lidy,lidx-1))+
                  hc3*(lsxx_r(lidz,lidy,lidx+3)-lsxx_r(lidz,lidy,lidx-2))+
                  hc4*(lsxx_r(lidz,lidy,lidx+4)-lsxx_r(lidz,lidy,lidx-3)));
#elif fdoh ==5
    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidy,lidx+1)-lsxx_r(lidz,lidy,lidx))+
                  hc2*(lsxx_r(lidz,lidy,lidx+2)-lsxx_r(lidz,lidy,lidx-1))+
                  hc3*(lsxx_r(lidz,lidy,lidx+3)-lsxx_r(lidz,lidy,lidx-2))+
                  hc4*(lsxx_r(lidz,lidy,lidx+4)-lsxx_r(lidz,lidy,lidx-3))+
                  hc5*(lsxx_r(lidz,lidy,lidx+5)-lsxx_r(lidz,lidy,lidx-4)));
#elif fdoh ==6
    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidy,lidx+1)-lsxx_r(lidz,lidy,lidx))+
                  hc2*(lsxx_r(lidz,lidy,lidx+2)-lsxx_r(lidz,lidy,lidx-1))+
                  hc3*(lsxx_r(lidz,lidy,lidx+3)-lsxx_r(lidz,lidy,lidx-2))+
                  hc4*(lsxx_r(lidz,lidy,lidx+4)-lsxx_r(lidz,lidy,lidx-3))+
                  hc5*(lsxx_r(lidz,lidy,lidx+5)-lsxx_r(lidz,lidy,lidx-4))+
                  hc6*(lsxx_r(lidz,lidy,lidx+6)-lsxx_r(lidz,lidy,lidx-5)));
#endif
    
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lsyy_r(lidz,lidy,lidx)=syy_r(gidz,gidy,gidx);
        if (lidy<2*fdoh)
            lsyy_r(lidz,lidy-fdoh,lidx)=syy_r(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lsyy_r(lidz,lidy+lsizey-3*fdoh,lidx)=syy_r(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lsyy_r(lidz,lidy+fdoh,lidx)=syy_r(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lsyy_r(lidz,lidy-lsizey+3*fdoh,lidx)=syy_r(gidz,gidy-lsizey+3*fdoh,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh ==1
    syy_y_r = dtdh*hc1*(lsyy_r(lidz,lidy+1,lidx) - lsyy_r(lidz,lidy,lidx));
#elif fdoh ==2
    syy_y_r = dtdh*(hc1*(lsyy_r(lidz,lidy+1,lidx) - lsyy_r(lidz,lidy,lidx))
                  +hc2*(lsyy_r(lidz,lidy+2,lidx) - lsyy_r(lidz,lidy-1,lidx)));
#elif fdoh ==3
    syy_y_r = dtdh*(hc1*(lsyy_r(lidz,lidy+1,lidx)-lsyy_r(lidz,lidy,lidx))+
                  hc2*(lsyy_r(lidz,lidy+2,lidx)-lsyy_r(lidz,lidy-1,lidx))+
                  hc3*(lsyy_r(lidz,lidy+3,lidx)-lsyy_r(lidz,lidy-2,lidx)));
#elif fdoh ==4
    syy_y_r = dtdh*(hc1*(lsyy_r(lidz,lidy+1,lidx)-lsyy_r(lidz,lidy,lidx))+
                  hc2*(lsyy_r(lidz,lidy+2,lidx)-lsyy_r(lidz,lidy-1,lidx))+
                  hc3*(lsyy_r(lidz,lidy+3,lidx)-lsyy_r(lidz,lidy-2,lidx))+
                  hc4*(lsyy_r(lidz,lidy+4,lidx)-lsyy_r(lidz,lidy-3,lidx)));
#elif fdoh ==5
    syy_y_r = dtdh*(hc1*(lsyy_r(lidz,lidy+1,lidx)-lsyy_r(lidz,lidy,lidx))+
                  hc2*(lsyy_r(lidz,lidy+2,lidx)-lsyy_r(lidz,lidy-1,lidx))+
                  hc3*(lsyy_r(lidz,lidy+3,lidx)-lsyy_r(lidz,lidy-2,lidx))+
                  hc4*(lsyy_r(lidz,lidy+4,lidx)-lsyy_r(lidz,lidy-3,lidx))+
                  hc5*(lsyy_r(lidz,lidy+5,lidx)-lsyy_r(lidz,lidy-4,lidx)));
#elif fdoh ==6
    syy_y_r = dtdh*(hc1*(lsyy_r(lidz,lidy+1,lidx)-lsyy_r(lidz,lidy,lidx))+
                  hc2*(lsyy_r(lidz,lidy+2,lidx)-lsyy_r(lidz,lidy-1,lidx))+
                  hc3*(lsyy_r(lidz,lidy+3,lidx)-lsyy_r(lidz,lidy-2,lidx))+
                  hc4*(lsyy_r(lidz,lidy+4,lidx)-lsyy_r(lidz,lidy-3,lidx))+
                  hc5*(lsyy_r(lidz,lidy+5,lidx)-lsyy_r(lidz,lidy-4,lidx))+
                  hc6*(lsyy_r(lidz,lidy+6,lidx)-lsyy_r(lidz,lidy-5,lidx)));
#endif
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lszz_r(lidz,lidy,lidx)=szz_r(gidz,gidy,gidx);
    if (lidz<2*fdoh)
        lszz_r(lidz-fdoh,lidy,lidx)=szz_r(gidz-fdoh,gidy,gidx);
    if (lidz>(lsizez-2*fdoh-1))
        lszz_r(lidz+fdoh,lidy,lidx)=szz_r(gidz+fdoh,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh ==1
    szz_z_r = dtdh*hc1*(lszz_r(lidz+1,lidy,lidx) - lszz_r(lidz,lidy,lidx));
#elif fdoh ==2
    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidy,lidx) - lszz_r(lidz,lidy,lidx))
                  +hc2*(lszz_r(lidz+2,lidy,lidx) - lszz_r(lidz-1,lidy,lidx)));
#elif fdoh ==3
    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidy,lidx)-lszz_r(lidz,lidy,lidx))+
                  hc2*(lszz_r(lidz+2,lidy,lidx)-lszz_r(lidz-1,lidy,lidx))+
                  hc3*(lszz_r(lidz+3,lidy,lidx)-lszz_r(lidz-2,lidy,lidx)));
#elif fdoh ==4
    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidy,lidx)-lszz_r(lidz,lidy,lidx))+
                  hc2*(lszz_r(lidz+2,lidy,lidx)-lszz_r(lidz-1,lidy,lidx))+
                  hc3*(lszz_r(lidz+3,lidy,lidx)-lszz_r(lidz-2,lidy,lidx))+
                  hc4*(lszz_r(lidz+4,lidy,lidx)-lszz_r(lidz-3,lidy,lidx)));
#elif fdoh ==5
    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidy,lidx)-lszz_r(lidz,lidy,lidx))+
                  hc2*(lszz_r(lidz+2,lidy,lidx)-lszz_r(lidz-1,lidy,lidx))+
                  hc3*(lszz_r(lidz+3,lidy,lidx)-lszz_r(lidz-2,lidy,lidx))+
                  hc4*(lszz_r(lidz+4,lidy,lidx)-lszz_r(lidz-3,lidy,lidx))+
                  hc5*(lszz_r(lidz+5,lidy,lidx)-lszz_r(lidz-4,lidy,lidx)));
#elif fdoh ==6
    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidy,lidx)-lszz_r(lidz,lidy,lidx))+
                  hc2*(lszz_r(lidz+2,lidy,lidx)-lszz_r(lidz-1,lidy,lidx))+
                  hc3*(lszz_r(lidz+3,lidy,lidx)-lszz_r(lidz-2,lidy,lidx))+
                  hc4*(lszz_r(lidz+4,lidy,lidx)-lszz_r(lidz-3,lidy,lidx))+
                  hc5*(lszz_r(lidz+5,lidy,lidx)-lszz_r(lidz-4,lidy,lidx))+
                  hc6*(lszz_r(lidz+6,lidy,lidx)-lszz_r(lidz-5,lidy,lidx)));
#endif
    
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lsxy_r(lidz,lidy,lidx)=sxy_r(gidz,gidy,gidx);
        if (lidy<2*fdoh)
            lsxy_r(lidz,lidy-fdoh,lidx)=sxy_r(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lsxy_r(lidz,lidy+lsizey-3*fdoh,lidx)=sxy_r(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lsxy_r(lidz,lidy+fdoh,lidx)=sxy_r(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lsxy_r(lidz,lidy-lsizey+3*fdoh,lidx)=sxy_r(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lsxy_r(lidz,lidy,lidx-fdoh)=sxy_r(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxy_r(lidz,lidy,lidx+lsizex-3*fdoh)=sxy_r(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxy_r(lidz,lidy,lidx+fdoh)=sxy_r(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxy_r(lidz,lidy,lidx-lsizex+3*fdoh)=sxy_r(gidz,gidy,gidx-lsizex+3*fdoh);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh ==1
    sxy_y_r = dtdh*hc1*(lsxy_r(lidz,lidy,lidx)   - lsxy_r(lidz,lidy-1,lidx));
    sxy_x_r = dtdh*hc1*(lsxy_r(lidz,lidy,lidx)   - lsxy_r(lidz,lidy,lidx-1));
#elif fdoh ==2
    sxy_y_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)   - lsxy_r(lidz,lidy-1,lidx))
                  +hc2*(lsxy_r(lidz,lidy+1,lidx) - lsxy_r(lidz,lidy-2,lidx)));
    sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)   - lsxy_r(lidz,lidy,lidx-1))
                  +hc2*(lsxy_r(lidz,lidy,lidx+1) - lsxy_r(lidz,lidy,lidx-2)));
#elif fdoh ==3
    sxy_y_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy-1,lidx))+
                  hc2*(lsxy_r(lidz,lidy+1,lidx)-lsxy_r(lidz,lidy-2,lidx))+
                  hc3*(lsxy_r(lidz,lidy+2,lidx)-lsxy_r(lidz,lidy-3,lidx)));
    
    sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy,lidx-1))+
                  hc2*(lsxy_r(lidz,lidy,lidx+1)-lsxy_r(lidz,lidy,lidx-2))+
                  hc3*(lsxy_r(lidz,lidy,lidx+2)-lsxy_r(lidz,lidy,lidx-3)));
#elif fdoh ==4
    sxy_y_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy-1,lidx))+
                  hc2*(lsxy_r(lidz,lidy+1,lidx)-lsxy_r(lidz,lidy-2,lidx))+
                  hc3*(lsxy_r(lidz,lidy+2,lidx)-lsxy_r(lidz,lidy-3,lidx))+
                  hc4*(lsxy_r(lidz,lidy+3,lidx)-lsxy_r(lidz,lidy-4,lidx)));
    
    sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy,lidx-1))+
                  hc2*(lsxy_r(lidz,lidy,lidx+1)-lsxy_r(lidz,lidy,lidx-2))+
                  hc3*(lsxy_r(lidz,lidy,lidx+2)-lsxy_r(lidz,lidy,lidx-3))+
                  hc4*(lsxy_r(lidz,lidy,lidx+3)-lsxy_r(lidz,lidy,lidx-4)));
#elif fdoh ==5
    sxy_y_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy-1,lidx))+
                  hc2*(lsxy_r(lidz,lidy+1,lidx)-lsxy_r(lidz,lidy-2,lidx))+
                  hc3*(lsxy_r(lidz,lidy+2,lidx)-lsxy_r(lidz,lidy-3,lidx))+
                  hc4*(lsxy_r(lidz,lidy+3,lidx)-lsxy_r(lidz,lidy-4,lidx))+
                  hc5*(lsxy_r(lidz,lidy+4,lidx)-lsxy_r(lidz,lidy-5,lidx)));
    
    sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy,lidx-1))+
                  hc2*(lsxy_r(lidz,lidy,lidx+1)-lsxy_r(lidz,lidy,lidx-2))+
                  hc3*(lsxy_r(lidz,lidy,lidx+2)-lsxy_r(lidz,lidy,lidx-3))+
                  hc4*(lsxy_r(lidz,lidy,lidx+3)-lsxy_r(lidz,lidy,lidx-4))+
                  hc5*(lsxy_r(lidz,lidy,lidx+4)-lsxy_r(lidz,lidy,lidx-5)));
    
#elif fdoh ==6
    
    sxy_y_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy-1,lidx))+
                  hc2*(lsxy_r(lidz,lidy+1,lidx)-lsxy_r(lidz,lidy-2,lidx))+
                  hc3*(lsxy_r(lidz,lidy+2,lidx)-lsxy_r(lidz,lidy-3,lidx))+
                  hc4*(lsxy_r(lidz,lidy+3,lidx)-lsxy_r(lidz,lidy-4,lidx))+
                  hc5*(lsxy_r(lidz,lidy+4,lidx)-lsxy_r(lidz,lidy-5,lidx))+
                  hc6*(lsxy_r(lidz,lidy+5,lidx)-lsxy_r(lidz,lidy-6,lidx)));
    
    sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidy,lidx)  -lsxy_r(lidz,lidy,lidx-1))+
                  hc2*(lsxy_r(lidz,lidy,lidx+1)-lsxy_r(lidz,lidy,lidx-2))+
                  hc3*(lsxy_r(lidz,lidy,lidx+2)-lsxy_r(lidz,lidy,lidx-3))+
                  hc4*(lsxy_r(lidz,lidy,lidx+3)-lsxy_r(lidz,lidy,lidx-4))+
                  hc5*(lsxy_r(lidz,lidy,lidx+4)-lsxy_r(lidz,lidy,lidx-5))+
                  hc6*(lsxy_r(lidz,lidy,lidx+5)-lsxy_r(lidz,lidy,lidx-6)));
#endif
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lsyz_r(lidz,lidy,lidx)=syz_r(gidz,gidy,gidx);
        if (lidy<2*fdoh)
            lsyz_r(lidz,lidy-fdoh,lidx)=syz_r(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lsyz_r(lidz,lidy+lsizey-3*fdoh,lidx)=syz_r(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lsyz_r(lidz,lidy+fdoh,lidx)=syz_r(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lsyz_r(lidz,lidy-lsizey+3*fdoh,lidx)=syz_r(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidz<2*fdoh)
            lsyz_r(lidz-fdoh,lidy,lidx)=syz_r(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lsyz_r(lidz+fdoh,lidy,lidx)=syz_r(gidz+fdoh,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh ==1
    syz_z_r = dtdh*hc1*(lsyz_r(lidz,lidy,lidx)   - lsyz_r(lidz-1,lidy,lidx));
    syz_y_r = dtdh*hc1*(lsyz_r(lidz,lidy,lidx)   - lsyz_r(lidz,lidy-1,lidx));
#elif fdoh ==2
    syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)   - lsyz_r(lidz-1,lidy,lidx))
                  +hc2*(lsyz_r(lidz+1,lidy,lidx) - lsyz_r(lidz-2,lidy,lidx)));
    syz_y_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)   - lsyz_r(lidz,lidy-1,lidx))
                  +hc2*(lsyz_r(lidz,lidy+1,lidx) - lsyz_r(lidz,lidy-2,lidx)));
#elif fdoh ==3
    syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz-1,lidy,lidx))+
                  hc2*(lsyz_r(lidz+1,lidy,lidx)-lsyz_r(lidz-2,lidy,lidx))+
                  hc3*(lsyz_r(lidz+2,lidy,lidx)-lsyz_r(lidz-3,lidy,lidx)));
    
    syz_y_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz,lidy-1,lidx))+
                  hc2*(lsyz_r(lidz,lidy+1,lidx)-lsyz_r(lidz,lidy-2,lidx))+
                  hc3*(lsyz_r(lidz,lidy+2,lidx)-lsyz_r(lidz,lidy-3,lidx)));
#elif fdoh ==4
    syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz-1,lidy,lidx))+
                  hc2*(lsyz_r(lidz+1,lidy,lidx)-lsyz_r(lidz-2,lidy,lidx))+
                  hc3*(lsyz_r(lidz+2,lidy,lidx)-lsyz_r(lidz-3,lidy,lidx))+
                  hc4*(lsyz_r(lidz+3,lidy,lidx)-lsyz_r(lidz-4,lidy,lidx)));
    
    syz_y_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)-lsyz_r(lidz,lidy-1,lidx))+
                  hc2*(lsyz_r(lidz,lidy+1,lidx)-lsyz_r(lidz,lidy-2,lidx))+
                  hc3*(lsyz_r(lidz,lidy+2,lidx)-lsyz_r(lidz,lidy-3,lidx))+
                  hc4*(lsyz_r(lidz,lidy+3,lidx)-lsyz_r(lidz,lidy-4,lidx)));
#elif fdoh ==5
    syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz-1,lidy,lidx))+
                  hc2*(lsyz_r(lidz+1,lidy,lidx)-lsyz_r(lidz-2,lidy,lidx))+
                  hc3*(lsyz_r(lidz+2,lidy,lidx)-lsyz_r(lidz-3,lidy,lidx))+
                  hc4*(lsyz_r(lidz+3,lidy,lidx)-lsyz_r(lidz-4,lidy,lidx))+
                  hc5*(lsyz_r(lidz+4,lidy,lidx)-lsyz_r(lidz-5,lidy,lidx)));
    
    syz_y_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz,lidy-1,lidx))+
                  hc2*(lsyz_r(lidz,lidy+1,lidx)-lsyz_r(lidz,lidy-2,lidx))+
                  hc3*(lsyz_r(lidz,lidy+2,lidx)-lsyz_r(lidz,lidy-3,lidx))+
                  hc4*(lsyz_r(lidz,lidy+3,lidx)-lsyz_r(lidz,lidy-4,lidx))+
                  hc5*(lsyz_r(lidz,lidy+4,lidx)-lsyz_r(lidz,lidy-5,lidx)));
#elif fdoh ==6
    syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz-1,lidy,lidx))+
                  hc2*(lsyz_r(lidz+1,lidy,lidx)-lsyz_r(lidz-2,lidy,lidx))+
                  hc3*(lsyz_r(lidz+2,lidy,lidx)-lsyz_r(lidz-3,lidy,lidx))+
                  hc4*(lsyz_r(lidz+3,lidy,lidx)-lsyz_r(lidz-4,lidy,lidx))+
                  hc5*(lsyz_r(lidz+4,lidy,lidx)-lsyz_r(lidz-5,lidy,lidx))+
                  hc6*(lsyz_r(lidz+5,lidy,lidx)-lsyz_r(lidz-6,lidy,lidx)));
    
    syz_y_r = dtdh*(hc1*(lsyz_r(lidz,lidy,lidx)  -lsyz_r(lidz,lidy-1,lidx))+
                  hc2*(lsyz_r(lidz,lidy+1,lidx)-lsyz_r(lidz,lidy-2,lidx))+
                  hc3*(lsyz_r(lidz,lidy+2,lidx)-lsyz_r(lidz,lidy-3,lidx))+
                  hc4*(lsyz_r(lidz,lidy+3,lidx)-lsyz_r(lidz,lidy-4,lidx))+
                  hc5*(lsyz_r(lidz,lidy+4,lidx)-lsyz_r(lidz,lidy-5,lidx))+
                  hc6*(lsyz_r(lidz,lidy+5,lidx)-lsyz_r(lidz,lidy-6,lidx)));
#endif
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lsxz_r(lidz,lidy,lidx)=sxz_r(gidz,gidy,gidx);
    
        if (lidx<2*fdoh)
            lsxz_r(lidz,lidy,lidx-fdoh)=sxz_r(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxz_r(lidz,lidy,lidx+lsizex-3*fdoh)=sxz_r(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxz_r(lidz,lidy,lidx+fdoh)=sxz_r(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxz_r(lidz,lidy,lidx-lsizex+3*fdoh)=sxz_r(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lsxz_r(lidz-fdoh,lidy,lidx)=sxz_r(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lsxz_r(lidz+fdoh,lidy,lidx)=sxz_r(gidz+fdoh,gidy,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh ==1
    sxz_z_r = dtdh*hc1*(lsxz_r(lidz,lidy,lidx)   - lsxz_r(lidz-1,lidy,lidx));
    sxz_x_r = dtdh*hc1*(lsxz_r(lidz,lidy,lidx)   - lsxz_r(lidz,lidy,lidx-1));
#elif fdoh ==2
    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)   - lsxz_r(lidz-1,lidy,lidx))
                  +hc2*(lsxz_r(lidz+1,lidy,lidx) - lsxz_r(lidz-2,lidy,lidx)));
    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)   - lsxz_r(lidz,lidy,lidx-1))
                  +hc2*(lsxz_r(lidz,lidy,lidx+1) - lsxz_r(lidz,lidy,lidx-2)));
    
#elif fdoh ==3
    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz-1,lidy,lidx))+
                  hc2*(lsxz_r(lidz+1,lidy,lidx)-lsxz_r(lidz-2,lidy,lidx))+
                  hc3*(lsxz_r(lidz+2,lidy,lidx)-lsxz_r(lidz-3,lidy,lidx)));
    
    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz,lidy,lidx-1))+
                  hc2*(lsxz_r(lidz,lidy,lidx+1)-lsxz_r(lidz,lidy,lidx-2))+
                  hc3*(lsxz_r(lidz,lidy,lidx+2)-lsxz_r(lidz,lidy,lidx-3)));
#elif fdoh ==4
    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz-1,lidy,lidx))+
                  hc2*(lsxz_r(lidz+1,lidy,lidx)-lsxz_r(lidz-2,lidy,lidx))+
                  hc3*(lsxz_r(lidz+2,lidy,lidx)-lsxz_r(lidz-3,lidy,lidx))+
                  hc4*(lsxz_r(lidz+3,lidy,lidx)-lsxz_r(lidz-4,lidy,lidx)));
    
    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz,lidy,lidx-1))+
                  hc2*(lsxz_r(lidz,lidy,lidx+1)-lsxz_r(lidz,lidy,lidx-2))+
                  hc3*(lsxz_r(lidz,lidy,lidx+2)-lsxz_r(lidz,lidy,lidx-3))+
                  hc4*(lsxz_r(lidz,lidy,lidx+3)-lsxz_r(lidz,lidy,lidx-4)));
#elif fdoh ==5
    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz-1,lidy,lidx))+
                  hc2*(lsxz_r(lidz+1,lidy,lidx)-lsxz_r(lidz-2,lidy,lidx))+
                  hc3*(lsxz_r(lidz+2,lidy,lidx)-lsxz_r(lidz-3,lidy,lidx))+
                  hc4*(lsxz_r(lidz+3,lidy,lidx)-lsxz_r(lidz-4,lidy,lidx))+
                  hc5*(lsxz_r(lidz+4,lidy,lidx)-lsxz_r(lidz-5,lidy,lidx)));
    
    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz,lidy,lidx-1))+
                  hc2*(lsxz_r(lidz,lidy,lidx+1)-lsxz_r(lidz,lidy,lidx-2))+
                  hc3*(lsxz_r(lidz,lidy,lidx+2)-lsxz_r(lidz,lidy,lidx-3))+
                  hc4*(lsxz_r(lidz,lidy,lidx+3)-lsxz_r(lidz,lidy,lidx-4))+
                  hc5*(lsxz_r(lidz,lidy,lidx+4)-lsxz_r(lidz,lidy,lidx-5)));
#elif fdoh ==6
    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz-1,lidy,lidx))+
                  hc2*(lsxz_r(lidz+1,lidy,lidx)-lsxz_r(lidz-2,lidy,lidx))+
                  hc3*(lsxz_r(lidz+2,lidy,lidx)-lsxz_r(lidz-3,lidy,lidx))+
                  hc4*(lsxz_r(lidz+3,lidy,lidx)-lsxz_r(lidz-4,lidy,lidx))+
                  hc5*(lsxz_r(lidz+4,lidy,lidx)-lsxz_r(lidz-5,lidy,lidx))+
                  hc6*(lsxz_r(lidz+5,lidy,lidx)-lsxz_r(lidz-6,lidy,lidx)));
    
    
    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidy,lidx)  -lsxz_r(lidz,lidy,lidx-1))+
                  hc2*(lsxz_r(lidz,lidy,lidx+1)-lsxz_r(lidz,lidy,lidx-2))+
                  hc3*(lsxz_r(lidz,lidy,lidx+2)-lsxz_r(lidz,lidy,lidx-3))+
                  hc4*(lsxz_r(lidz,lidy,lidx+3)-lsxz_r(lidz,lidy,lidx-4))+
                  hc5*(lsxz_r(lidz,lidy,lidx+4)-lsxz_r(lidz,lidy,lidx-5))+
                  hc6*(lsxz_r(lidz,lidy,lidx+5)-lsxz_r(lidz,lidy,lidx-6)));
    
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

// Backpropagate the forward velocity
#if back_prop_type==1
    {
        float3 amp = ssource(gidz, gidy, gidx+offset, nsrc, srcpos_loc, signals, nt, rip, rjp, rkp);
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
#if abs_type==1
    {
        int ind;
        
        if (gidz>NZ-nab-fdoh-1){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz - NZ+nab+fdoh+nab;
            ind=2*nab-1-k;
            
            psi_sxz_z(k,j,i) = b_z[ind+1] * psi_sxz_z(k,j,i) + a_z[ind+1] * sxz_z_r;
            sxz_z_r = sxz_z_r / K_z[ind+1] + psi_sxz_z(k,j,i);
            psi_syz_z(k,j,i) = b_z[ind+1] * psi_syz_z(k,j,i) + a_z[ind+1] * syz_z_r;
            syz_z_r = syz_z_r / K_z[ind+1] + psi_syz_z(k,j,i);
            psi_szz_z(k,j,i) = b_z_half[ind] * psi_szz_z(k,j,i) + a_z_half[ind] * szz_z_r;
            szz_z_r = szz_z_r / K_z_half[ind] + psi_szz_z(k,j,i);
            
        }
        
#if freesurf==0
        else if (gidz-fdoh<nab){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_sxz_z(k,j,i) = b_z[k] * psi_sxz_z(k,j,i) + a_z[k] * sxz_z_r;
            sxz_z_r = sxz_z_r / K_z[k] + psi_sxz_z(k,j,i);
            psi_syz_z(k,j,i) = b_z[k] * psi_syz_z(k,j,i) + a_z[k] * syz_z_r;
            syz_z_r = syz_z_r / K_z[k] + psi_syz_z(k,j,i);
            psi_szz_z(k,j,i) = b_z_half[k] * psi_szz_z(k,j,i) + a_z_half[k] * szz_z_r;
            szz_z_r = szz_z_r / K_z_half[k] + psi_szz_z(k,j,i);
            
        }
#endif
        
        if (gidy-fdoh<nab){
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_sxy_y(k,j,i) = b_y[j] * psi_sxy_y(k,j,i) + a_y[j] * sxy_y_r;
            sxy_y_r = sxy_y_r / K_y[j] + psi_sxy_y(k,j,i);
            psi_syy_y(k,j,i) = b_y_half[j] * psi_syy_y(k,j,i) + a_y_half[j] * syy_y_r;
            syy_y_r = syy_y_r / K_y_half[j] + psi_syy_y(k,j,i);
            psi_syz_y(k,j,i) = b_y[j] * psi_syz_y(k,j,i) + a_y[j] * syz_y_r;
            syz_y_r = syz_y_r / K_y[j] + psi_syz_y(k,j,i);
            
        }
        
        else if (gidy>NY-nab-fdoh-1){
            
            i =gidx-fdoh;
            j =gidy - NY+nab+fdoh+nab;
            k =gidz-fdoh;
            ind=2*nab-1-j;
            
            psi_sxy_y(k,j,i) = b_y[ind+1] * psi_sxy_y(k,j,i) + a_y[ind+1] * sxy_y_r;
            sxy_y_r = sxy_y_r / K_y[ind+1] + psi_sxy_y(k,j,i);
            psi_syy_y(k,j,i) = b_y_half[ind] * psi_syy_y(k,j,i) + a_y_half[ind] * syy_y_r;
            syy_y_r = syy_y_r / K_y_half[ind] + psi_syy_y(k,j,i);
            psi_syz_y(k,j,i) = b_y[ind+1] * psi_syz_y(k,j,i) + a_y[ind+1] * syz_y_r;
            syz_y_r = syz_y_r / K_y[ind+1] + psi_syz_y(k,j,i);
            
            
        }
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_sxx_x(k,j,i) = b_x_half[i] * psi_sxx_x(k,j,i) + a_x_half[i] * sxx_x_r;
            sxx_x_r = sxx_x_r / K_x_half[i] + psi_sxx_x(k,j,i);
            psi_sxy_x(k,j,i) = b_x[i] * psi_sxy_x(k,j,i) + a_x[i] * sxy_x_r;
            sxy_x_r = sxy_x_r / K_x[i] + psi_sxy_x(k,j,i);
            psi_sxz_x(k,j,i) = b_x[i] * psi_sxz_x(k,j,i) + a_x[i] * sxz_x_r;
            sxz_x_r = sxz_x_r / K_x[i] + psi_sxz_x(k,j,i);
            
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            
            i =gidx - NX+nab+fdoh+nab;
            j =gidy-fdoh;
            k =gidz-fdoh;
            ind=2*nab-1-i;
            
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
#if abs_type==2
    {
        if (gidz-fdoh<nab){
            vx_r(gidz,gidy,gidx)*=taper[gidz-fdoh];
            vy_r(gidz,gidy,gidx)*=taper[gidz-fdoh];
            vz_r(gidz,gidy,gidx)*=taper[gidz-fdoh];
        }
        
        if (gidz>NZ-nab-fdoh-1){
            vx_r(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            vy_r(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            vz_r(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
        }
        
        if (gidy-fdoh<nab){
            vx_r(gidz,gidy,gidx)*=taper[gidy-fdoh];
            vy_r(gidz,gidy,gidx)*=taper[gidy-fdoh];
            vz_r(gidz,gidy,gidx)*=taper[gidy-fdoh];
        }
        
        if (gidy>NY-nab-fdoh-1){
            vx_r(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            vy_r(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            vz_r(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
        }
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            vx_r(gidz,gidy,gidx)*=taper[gidx-fdoh];
            vy_r(gidz,gidy,gidx)*=taper[gidx-fdoh];
            vz_r(gidz,gidy,gidx)*=taper[gidx-fdoh];
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            vx_r(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            vy_r(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            vz_r(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
        }
#endif
    }
#endif

// Density gradient calculation on the fly    
#if back_prop_type==1
    gradrho(gidz,gidx)+=vx(gidz,gidx)*lvx+vy(gidz,gidx)*lvy+vz(gidz,gidx)*lvz;
#endif
    
#if gradsrcout==1
    if (nsrc>0){
        
        
        for (int srci=0; srci<nsrc; srci++){
            
            
            int i=(int)(srcpos_loc(0,srci)/DH-0.5)+fdoh;
            int j=(int)(srcpos_loc(1,srci)/DH-0.5)+fdoh;
            int k=(int)(srcpos_loc(2,srci)/DH-0.5)+fdoh;
            
            if (i==gidx && j==gidy && k==gidz){
                
                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
                
                if (SOURCE_TYPE==2){
                    /* single force in x */
                    gradsrc(srci,nt)+= vx_r(gidz,gidy,gidx)/rip(gidx,gidy,gidz)/(DH*DH*DH);
                }
                else if (SOURCE_TYPE==3){
                    /* single force in y */
                    
                    gradsrc(srci,nt)+= vy_r(gidz,gidy,gidx)/rip(gidx,gidy,gidz)/(DH*DH*DH);
                }
                else if (SOURCE_TYPE==4){
                    /* single force in z */
                    
                    gradsrc(srci,nt)+= vz_r(gidz,gidy,gidx)/rip(gidx,gidy,gidz)/(DH*DH*DH);
                }
                
            }
        }
        
        
    }
#endif
    
}

