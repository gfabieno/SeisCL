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

/*Update of the velocity in 3D*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */


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

#define rxx(z,y,x,l) rxx[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryy(z,y,x,l) ryy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rzz(z,y,x,l) rzz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxy(z,y,x,l) rxy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryz(z,y,x,l) ryz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxz(z,y,x,l) rxz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]

#define psi_sxx_x(z,y,x) psi_sxx_x[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_sxy_x(z,y,x) psi_sxy_x[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_sxz_x(z,y,x) psi_sxz_x[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_sxy_y(z,y,x) psi_sxy_y[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_syy_y(z,y,x) psi_syy_y[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_syz_y(z,y,x) psi_syz_y[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_sxz_z(z,y,x) psi_sxz_z[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_syz_z(z,y,x) psi_syz_z[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_szz_z(z,y,x) psi_szz_z[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]



#if local_off==0

#define lvar(z,y,x)   lvar[(x)*lsizey*lsizez+(y)*lsizez+(z)]

#endif


#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define PI (3.141592653589793238462643383279502884197169)
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]




float3 ssource(int gidz, int gidy, int gidx,  int nsrc, __global float *srcpos_loc, __global float *signals, int nt, __global float * rip, __global float * rjp, __global float * rkp){
    
    float3 ampv={0.0,0.0,0.0};
    if (nsrc>0){
        
        
        for (int srci=0; srci<nsrc; srci++){
            
            
            int i=(int)floor(srcpos_loc(0,srci)/DH-0.5)+fdoh;
            int j=(int)floor(srcpos_loc(1,srci)/DH-0.5)+fdoh;
            int k=(int)floor(srcpos_loc(2,srci)/DH-0.5)+fdoh;
            
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
                
//                if (SOURCE_TYPE==2){
//                    /* single force in x */
//                    ampv.x  +=  amp;
//                }
//                else if (SOURCE_TYPE==3){
//                    /* single force in y */
//                    
//                    ampv.y  +=  amp;
//                }
//                else if (SOURCE_TYPE==4){
//                    /* single force in z */
//                    
//                    ampv.z  +=  amp;
//                }

            }
        }

        
    }
 
    return ampv;

}

__kernel void update_v(int offcomm, int nsrc,  int nt,
                       __global float *vx,         __global float *vy,           __global float *vz,
                       __global float *sxx,        __global float *syy,          __global float *szz,
                       __global float *sxy,        __global float *syz,          __global float *sxz,
                       __global float *rip,        __global float *rjp,          __global float *rkp,
                       __global float *srcpos_loc, __global float *signals,      __global float *rec_pos,
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
                       __local  float *lvar)
{

    float sxx_x;
    float syy_y;
    float szz_z;
    float sxy_y;
    float sxy_x;
    float syz_y;
    float syz_z;
    float sxz_x;
    float sxz_z;
    
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
#define lidx gidx
#define lidy gidy
#define lidz gidz
    
#endif
 
// Calculation of the stresses spatial derivatives
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
        barrier(CLK_LOCAL_MEM_FENCE);
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
    }

// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if local_off==0
#if comm12==0
    if ( gidy>(NY-fdoh-1) ||gidz>(NZ-fdoh-1) || (gidx-offcomm)>(NX-fdoh-1-lcomm) ){
        return;
    }
    
#else
    if ( gidy>(NY-fdoh-1) || gidz>(NZ-fdoh-1) ){
        return;
    }
#endif
#endif

    
    
    
// Correct spatial derivatives to implement CPML
#if abs_type==1
    {
        int i,j,k, ind;
        
        if (gidz>NZ-nab-fdoh-1){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz - NZ+nab+fdoh+nab;
            ind=2*nab-1-k;
            
            psi_sxz_z(k,j,i) = b_z[ind+1] * psi_sxz_z(k,j,i) + a_z[ind+1] * sxz_z;
            sxz_z = sxz_z / K_z[ind+1] + psi_sxz_z(k,j,i);
            psi_syz_z(k,j,i) = b_z[ind+1] * psi_syz_z(k,j,i) + a_z[ind+1] * syz_z;
            syz_z = syz_z / K_z[ind+1] + psi_syz_z(k,j,i);
            psi_szz_z(k,j,i) = b_z_half[ind] * psi_szz_z(k,j,i) + a_z_half[ind] * szz_z;
            szz_z = szz_z / K_z_half[ind] + psi_szz_z(k,j,i);
            
        }
        
#if freesurf==0
        else if (gidz-fdoh<nab){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_sxz_z(k,j,i) = b_z[k] * psi_sxz_z(k,j,i) + a_z[k] * sxz_z;
            sxz_z = sxz_z / K_z[k] + psi_sxz_z(k,j,i);
            psi_syz_z(k,j,i) = b_z[k] * psi_syz_z(k,j,i) + a_z[k] * syz_z;
            syz_z = syz_z / K_z[k] + psi_syz_z(k,j,i);
            psi_szz_z(k,j,i) = b_z_half[k] * psi_szz_z(k,j,i) + a_z_half[k] * szz_z;
            szz_z = szz_z / K_z_half[k] + psi_szz_z(k,j,i);
            
        }
#endif
        
        if (gidy-fdoh<nab){
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_sxy_y(k,j,i) = b_y[j] * psi_sxy_y(k,j,i) + a_y[j] * sxy_y;
            sxy_y = sxy_y / K_y[j] + psi_sxy_y(k,j,i);
            psi_syy_y(k,j,i) = b_y_half[j] * psi_syy_y(k,j,i) + a_y_half[j] * syy_y;
            syy_y = syy_y / K_y_half[j] + psi_syy_y(k,j,i);
            psi_syz_y(k,j,i) = b_y[j] * psi_syz_y(k,j,i) + a_y[j] * syz_y;
            syz_y = syz_y / K_y[j] + psi_syz_y(k,j,i);
            
        }
        
        else if (gidy>NY-nab-fdoh-1){
            
            i =gidx-fdoh;
            j =gidy - NY+nab+fdoh+nab;
            k =gidz-fdoh;
            ind=2*nab-1-j;
            
            psi_sxy_y(k,j,i) = b_y[ind+1] * psi_sxy_y(k,j,i) + a_y[ind+1] * sxy_y;
            sxy_y = sxy_y / K_y[ind+1] + psi_sxy_y(k,j,i);
            psi_syy_y(k,j,i) = b_y_half[ind] * psi_syy_y(k,j,i) + a_y_half[ind] * syy_y;
            syy_y = syy_y / K_y_half[ind] + psi_syy_y(k,j,i);
            psi_syz_y(k,j,i) = b_y[ind+1] * psi_syz_y(k,j,i) + a_y[ind+1] * syz_y;
            syz_y = syz_y / K_y[ind+1] + psi_syz_y(k,j,i);
            
            
        }
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_sxx_x(k,j,i) = b_x_half[i] * psi_sxx_x(k,j,i) + a_x_half[i] * sxx_x;
            sxx_x = sxx_x / K_x_half[i] + psi_sxx_x(k,j,i);
            psi_sxy_x(k,j,i) = b_x[i] * psi_sxy_x(k,j,i) + a_x[i] * sxy_x;
            sxy_x = sxy_x / K_x[i] + psi_sxy_x(k,j,i);
            psi_sxz_x(k,j,i) = b_x[i] * psi_sxz_x(k,j,i) + a_x[i] * sxz_x;
            sxz_x = sxz_x / K_x[i] + psi_sxz_x(k,j,i);
            
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            
            i =gidx - NX+nab+fdoh+nab;
            j =gidy-fdoh;
            k =gidz-fdoh;
            ind=2*nab-1-i;
            
            psi_sxx_x(k,j,i) = b_x_half[ind] * psi_sxx_x(k,j,i) + a_x_half[ind] * sxx_x;
            sxx_x = sxx_x / K_x_half[ind] + psi_sxx_x(k,j,i);
            psi_sxy_x(k,j,i) = b_x[ind+1] * psi_sxy_x(k,j,i) + a_x[ind+1] * sxy_x;
            sxy_x = sxy_x / K_x[ind+1] + psi_sxy_x(k,j,i);
            psi_sxz_x(k,j,i) = b_x[ind+1] * psi_sxz_x(k,j,i) + a_x[ind+1] * sxz_x;
            sxz_x = sxz_x / K_x[ind+1] + psi_sxz_x(k,j,i);
            
            
            
        }
#endif
    }
#endif

// Update the velocities
    {
        float3 amp = ssource(gidz, gidy, gidx+offset, nsrc, srcpos_loc, signals, nt, rip, rjp, rkp);
        vx(gidz,gidy,gidx)+= ((sxx_x + sxy_y + sxz_z)/rip(gidz,gidy,gidx))+amp.x;
        vy(gidz,gidy,gidx)+= ((syy_y + sxy_x + syz_z)/rjp(gidz,gidy,gidx))+amp.y;
        vz(gidz,gidy,gidx)+= ((szz_z + sxz_x + syz_y)/rkp(gidz,gidy,gidx))+amp.z;
    }
    
// Absorbing boundary
#if abs_type==2
    {
#if freesurf==0
        if (gidz-fdoh<nab){
            vx(gidz,gidy,gidx)*=taper[gidz-fdoh];
            vy(gidz,gidy,gidx)*=taper[gidz-fdoh];
            vz(gidz,gidy,gidx)*=taper[gidz-fdoh];
        }
#endif
        
        if (gidz>NZ-nab-fdoh-1){
            vx(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            vy(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            vz(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
        }
        
        if (gidy-fdoh<nab){
            vx(gidz,gidy,gidx)*=taper[gidy-fdoh];
            vy(gidz,gidy,gidx)*=taper[gidy-fdoh];
            vz(gidz,gidy,gidx)*=taper[gidy-fdoh];
        }
        
        if (gidy>NY-nab-fdoh-1){
            vx(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            vy(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            vz(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
        }
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            vx(gidz,gidy,gidx)*=taper[gidx-fdoh];
            vy(gidz,gidy,gidx)*=taper[gidx-fdoh];
            vz(gidz,gidy,gidx)*=taper[gidx-fdoh];
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            vx(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            vy(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            vz(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
        }
#endif
    }
#endif
    
    
}


