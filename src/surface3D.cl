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

/*This is the kernel that implement the free surface condition in 3D*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define rho(z,y,x)     rho[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,y,x)     rip[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,y,x)     rjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,y,x)     rkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipjp(z,y,x) muipjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define mujpkp(z,y,x) mujpkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipkp(z,y,x) muipkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define mu(z,y,x)         mu[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define M(z,y,x)       M[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define grad(z,y,x)   grad[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define grads(z,y,x) grads[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define amp1(z,y,x)   amp1[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define amp2(z,y,x)   amp2[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

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


#if LOCAL_OFF==0

#define lvx(z,y,x)   lvx[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lvy(z,y,x)   lvy[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lvz(z,y,x)   lvz[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsxx(z,y,x) lsxx[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsyy(z,y,x) lsyy[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lszz(z,y,x) lszz[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsxy(z,y,x) lsxy[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsyz(z,y,x) lsyz[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsxz(z,y,x) lsxz[(x)*lsizey*lsizez+(y)*lsizez+(z)]

#endif

#define psi_vxx(z,y,x) psi_vxx[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vyx(z,y,x) psi_vyx[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vzx(z,y,x) psi_vzx[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vxy(z,y,x) psi_vxy[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vyy(z,y,x) psi_vyy[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vzy(z,y,x) psi_vzy[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vxz(z,y,x) psi_vxz[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vyz(z,y,x) psi_vyz[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vzz(z,y,x) psi_vzz[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]



#define PI (3.141592653589793238462643383279502884197169)
#define signals(y,x) signals[(y)*NT+(x)]



FUNDEF void freesurface(        GLOBARG float *vx,         GLOBARG float *vy,       GLOBARG float *vz,
                              GLOBARG float *sxx,        GLOBARG float *syy,      GLOBARG float *szz,
                              GLOBARG float *sxy,        GLOBARG float *syz,      GLOBARG float *sxz,
                              GLOBARG float *M,         GLOBARG float *mu,        GLOBARG float *rxx,
                              GLOBARG float *ryy,        GLOBARG float *rzz,
                              GLOBARG float *taus,       GLOBARG float *taup,     GLOBARG float *eta, GLOBARG float *K_x, GLOBARG float *psi_vx_x,
                              GLOBARG float *K_y, GLOBARG float *psi_vy_y)
{
    /*Indice definition */
    #ifdef __OPENCL_VERSION__
    int gidy = get_global_id(0) + FDOH;
    int gidx = get_global_id(1) + FDOH;
    #else
    int gidy = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y + FDOH;
    #endif
    int gidz=FDOH;
    
    /* Global work size is padded to be a multiple of local work size. The padding elements must not be updated */
    if (gidy>(NY-FDOH-1) || gidx>(NX-FDOH-1) ){
        return;
    }
    
    float f, g, h;
    float  vxx, vyy, vzz;
    int m;
    
    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    szz(gidz, gidy, gidx)=0.0;
#if LVE>0
    rzz(gidz, gidy, gidx)=0.0;
#endif
    
    for (m=1; m<=FDOH; m++) {
        szz(gidz-m, gidy, gidx)=-szz(gidz+m, gidy, gidx);
        sxz(gidz-m, gidy, gidx)=-sxz(gidz+m-1, gidy, gidx);
        syz(gidz-m, gidy, gidx)=-syz(gidz+m-1, gidy, gidx);
				}
				
    
#if   FDOH==1
    {
        vxx = (vx(gidz,gidy,gidx)-vx(gidz,gidy,gidx-1));
        vyy = (vy(gidz,gidy,gidx)-vy(gidz,gidy-1,gidx));
        vzz = (vz(gidz,gidy,gidx)-vz(gidz-1,gidy,gidx));
    }
#elif FDOH==2
    {
        vxx = (HC1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               HC2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2)));
        
        vyy = (HC1*(vy(gidz,gidy,gidx)  -vy(gidz,gidy-1,gidx))+
               HC2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx)));
        
        vzz = (HC1*(vz(gidz,gidy,gidx)  -vz(gidz-1,gidy,gidx))+
               HC2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx)));
    }
#elif FDOH==3
    {
        vxx = (HC1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               HC2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2))+
               HC3*(vx(gidz,gidy,gidx+2)-vx(gidz,gidy,gidx-3)));
        
        vyy = (HC1*(vy(gidz,gidy,gidx)-vy(gidz,gidy-1,gidx))+
               HC2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx))+
               HC3*(vy(gidz,gidy+2,gidx)-vy(gidz,gidy-3,gidx)));
        
        vzz = (HC1*(vz(gidz,gidy,gidx)-vz(gidz-1,gidy,gidx))+
               HC2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx))+
               HC3*(vz(gidz+2,gidy,gidx)-vz(gidz-3,gidy,gidx)));
        
    }
#elif FDOH==4
    {
        vxx = (HC1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               HC2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2))+
               HC3*(vx(gidz,gidy,gidx+2)-vx(gidz,gidy,gidx-3))+
               HC4*(vx(gidz,gidy,gidx+3)-vx(gidz,gidy,gidx-4)));

        vyy = (HC1*(vy(gidz,gidy,gidx)  -vy(gidz,gidy-1,gidx))+
               HC2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx))+
               HC3*(vy(gidz,gidy+2,gidx)-vy(gidz,gidy-3,gidx))+
               HC4*(vy(gidz,gidy+3,gidx)-vy(gidz,gidy-4,gidx)));
        
        vzz = (HC1*(vz(gidz,gidy,gidx)  -vz(gidz-1,gidy,gidx))+
               HC2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx))+
               HC3*(vz(gidz+2,gidy,gidx)-vz(gidz-3,gidy,gidx))+
               HC4*(vz(gidz+3,gidy,gidx)-vz(gidz-4,gidy,gidx)));
    }
#elif FDOH==5
    {
        vxx = (HC1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               HC2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2))+
               HC3*(vx(gidz,gidy,gidx+2)-vx(gidz,gidy,gidx-3))+
               HC4*(vx(gidz,gidy,gidx+3)-vx(gidz,gidy,gidx-4))+
               HC5*(vx(gidz,gidy,gidx+4)-vx(gidz,gidy,gidx-5)));
        
        vyy = (HC1*(vy(gidz,gidy,gidx)  -vy(gidz,gidy-1,gidx))+
               HC2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx))+
               HC3*(vy(gidz,gidy+2,gidx)-vy(gidz,gidy-3,gidx))+
               HC4*(vy(gidz,gidy+3,gidx)-vy(gidz,gidy-4,gidx))+
               HC5*(vy(gidz,gidy+4,gidx)-vy(gidz,gidy-5,gidx)));
        
        vzz = (HC1*(vz(gidz,gidy,gidx)  -vz(gidz-1,gidy,gidx))+
               HC2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx))+
               HC3*(vz(gidz+2,gidy,gidx)-vz(gidz-3,gidy,gidx))+
               HC4*(vz(gidz+3,gidy,gidx)-vz(gidz-4,gidy,gidx))+
               HC5*(vz(gidz+4,gidy,gidx)-vz(gidz-5,gidy,gidx)));
        
        
    }
#elif FDOH==6
    {
        vxx = (HC1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               HC2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2))+
               HC3*(vx(gidz,gidy,gidx+2)-vx(gidz,gidy,gidx-3))+
               HC4*(vx(gidz,gidy,gidx+3)-vx(gidz,gidy,gidx-4))+
               HC5*(vx(gidz,gidy,gidx+4)-vx(gidz,gidy,gidx-5))+
               HC6*(vx(gidz,gidy,gidx+5)-vx(gidz,gidy,gidx-6)));
        
        vyy = (HC1*(vy(gidz,gidy,gidx)  -vy(gidz,gidy-1,gidx))+
               HC2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx))+
               HC3*(vy(gidz,gidy+2,gidx)-vy(gidz,gidy-3,gidx))+
               HC4*(vy(gidz,gidy+3,gidx)-vy(gidz,gidy-4,gidx))+
               HC5*(vy(gidz,gidy+4,gidx)-vy(gidz,gidy-5,gidx))+
               HC6*(vy(gidz,gidy+5,gidx)-vy(gidz,gidy-6,gidx)));
        
        vzz = (HC1*(vz(gidz,gidy,gidx)  -vz(gidz-1,gidy,gidx))+
               HC2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx))+
               HC3*(vz(gidz+2,gidy,gidx)-vz(gidz-3,gidy,gidx))+
               HC4*(vz(gidz+3,gidy,gidx)-vz(gidz-4,gidy,gidx))+
               HC5*(vz(gidz+4,gidy,gidx)-vz(gidz-5,gidy,gidx))+
               HC6*(vz(gidz+5,gidy,gidx)-vz(gidz-6,gidy,gidx)));
    }
#endif


//// Absorbing boundary
//#if ABS_TYPE==2
//{
//    if (gidy-FDOH<NAB){
//        sxx(gidz,gidy,gidx)*=1.0/taper[gidy-FDOH];
//        syy(gidz,gidy,gidx)*=1.0/taper[gidy-FDOH];
//    }
//
//    if (gidy>NY-NAB-FDOH-1){
//        sxx(gidz,gidy,gidx)*=1.0/taper[NY-FDOH-gidy-1];
//        syy(gidz,gidy,gidx)*=1.0/taper[NY-FDOH-gidy-1];
//    }
//
//#if DEV==0 & MYLOCALID==0
//    if (gidx-FDOH<NAB){
//        sxx(gidz,gidy,gidx)*=1.0/taper[gidx-FDOH];
//        syy(gidz,gidy,gidx)*=1.0/taper[gidx-FDOH];
//    }
//#endif
//
//#if DEV==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//    if (gidx>NX-NAB-FDOH-1){
//        sxx(gidz,gidy,gidx)*=1.0/taper[NX-FDOH-gidx-1];
//        syy(gidz,gidy,gidx)*=1.0/taper[NX-FDOH-gidx-1];
//    }
//#endif
//}
//#endif
    
//// Correct spatial derivatives to implement CPML
//#if ABS_TYPE==1
//    {
//        int i,j,k,ind;
//
//        if (gidy-FDOH<NAB){
//            i =gidx-FDOH;
//            j =gidy-FDOH;
//            k =gidz-FDOH;
//
//            vyy = vyy / K_y[j] + psi_vyy(k,j,i);
//        }
//
//        else if (gidy>NY-NAB-FDOH-1){
//
//            i =gidx-FDOH;
//            j =gidy - NY+NAB+FDOH+NAB;
//            k =gidz-FDOH;
//            ind=2*NAB-1-j;
//            vyy = vyy / K_y[ind+1] + psi_vyy(k,j,i);
//        }
//#if DEV==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//
//            i =gidx-FDOH;
//            j =gidy-FDOH;
//            k =gidz-FDOH;
//
//            vxx = vxx / K_x[i] + psi_vxx(k,j,i);
//        }
//#endif
//
//#if DEV==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//
//            i =gidx - NX+NAB+FDOH+NAB;
//            j =gidy-FDOH;
//            k =gidz-FDOH;
//            ind=2*NAB-1-i;
//            vxx = vxx /K_x[ind+1] + psi_vxx(k,j,i);
//        }
//#endif
//    }
//#endif

#if LVE==0
				f=mu(gidz, gidy, gidx)*2.0f;
				g=M(gidz, gidy, gidx);
				h=-((g-f)*(g-f)*(vxx+vyy)/g)-((g-f)*vzz);
				sxx(gidz, gidy, gidx)+=h;
				syy(gidz, gidy, gidx)+=h;
#else
    float b,d,e;
    /* partially updating sxx and syy in the same way*/
    f=mu(gidz, gidy, gidx)*2.0*(1.0+L*taus(gidz, gidy, gidx));
    g=M(gidz, gidy, gidx)*(1.0+L*taup(gidz, gidy, gidx));
    h=-((g-f)*(g-f)*(vxx+vyy)/g)-((g-f)*vzz);
    sxx(gidz, gidy, gidx)+=h-(DT/2.0*rxx(gidz, gidy, gidx));
    syy(gidz, gidy, gidx)+=h-(DT/2.0*ryy(gidz, gidy, gidx));
    
    /* updating the memory-variable rxx, ryy at the free surface */
    
    d=2.0*mu(gidz, gidy, gidx)*taus(gidz, gidy, gidx);
    e=M(gidz, gidy, gidx)*taup(gidz, gidy, gidx);
    for (m=0;m<LVE;m++){
        b=eta[m]/(1.0+(eta[m]*0.5));
        h=b*(((d-e)*((f/g)-1.0)*(vxx+vyy))-((d-e)*vzz));
        rxx(gidz, gidy, gidx)+=h;
        ryy(gidz, gidy, gidx)+=h;
    }
    
    /*completely updating the stresses sxx and syy */
    sxx(gidz, gidy, gidx)+=(DT/2.0*rxx(gidz, gidy, gidx));
    syy(gidz, gidy, gidx)+=(DT/2.0*ryy(gidz, gidy, gidx));
    
#endif

//// Absorbing boundary
//#if ABS_TYPE==2
//    {
//        if (gidy-FDOH<NAB){
//            sxx(gidz,gidy,gidx)*=taper[gidy-FDOH];
//            syy(gidz,gidy,gidx)*=taper[gidy-FDOH];
//        }
//        
//        if (gidy>NY-NAB-FDOH-1){
//            sxx(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
//            syy(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
//        }
//        
//#if DEV==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//            sxx(gidz,gidy,gidx)*=taper[gidx-FDOH];
//            syy(gidz,gidy,gidx)*=taper[gidx-FDOH];
//        }
//#endif
//        
//#if DEV==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//            sxx(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
//            syy(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
//        }
//#endif
//    }
//#endif
    
}



