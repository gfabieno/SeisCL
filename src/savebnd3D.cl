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

/*Kernels to save boundary wavefield in 3D if backpropagation is used in the computation of the gradient */


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



#define PI (3.141592653589793238462643383279502884197169)


__kernel void savebnd(__global float *vx,         __global float *vy,      __global float *vz,
                      __global float *sxx,        __global float *syy,     __global float *szz,
                      __global float *sxy,        __global float *syz,     __global float *sxz,
                      __global float *vxbnd,      __global float *vybnd,   __global float *vzbnd,
                      __global float *sxxbnd,     __global float *syybnd,  __global float *szzbnd,
                      __global float *sxybnd,     __global float *syzbnd,  __global float *sxzbnd)
{
// If we have one device and one processing element in the group, we need all 6 sides of the boundary
#if NUM_DEVICES==1 & NLOCALP==1
    int gid = get_global_id(0);
    int NXbnd = (NX-2*FDOH-2*NAB);
    int NYbnd = (NY-2*FDOH-2*NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    int i,j,k;
    int gidf;
    
    if (gid<NYbnd*NZbnd*FDOH){//front
        gidf=gid;
        i=gidf/(NYbnd*NZbnd)+lbnd;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NYbnd*NZbnd*2*FDOH){//back
        gidf=gid-NYbnd*NZbnd*FDOH;
        i=gidf/(NYbnd*NZbnd)+NXbnd+NAB;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NYbnd*NZbnd*2*FDOH+NZbnd*(NXbnd-2*FDOH)*FDOH){//left
        gidf=gid-NYbnd*NZbnd*2*FDOH;
        i=gidf/(NZbnd*FDOH)+lbnd+FDOH;
        j=(gidf/NZbnd)%FDOH+lbnd;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NYbnd*NZbnd*2*FDOH+NZbnd*(NXbnd-2*FDOH)*2*FDOH){//right
        gidf=gid-NYbnd*NZbnd*2*FDOH-NZbnd*(NXbnd-2*FDOH)*FDOH;
        i=gidf/(NZbnd*FDOH)+lbnd+FDOH;
        j=(gidf/NZbnd)%FDOH+NYbnd+NAB;
        k=gidf%NZbnd+lbnds;

    }
    else if (gid<NYbnd*NZbnd*2*FDOH+NZbnd*(NXbnd-2*FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-2*FDOH)*FDOH){//down
        gidf=gid-NYbnd*NZbnd*2*FDOH-NZbnd*(NXbnd-2*FDOH)*2*FDOH;
        i=gidf/(FDOH*(NYbnd-2*FDOH))+lbnd+FDOH;
        j=(gidf/FDOH)%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%FDOH+NZbnd+lbnds/DIV-FDOH/DIV;


    }
    else if (gid<NYbnd*NZbnd*2*FDOH+NZbnd*(NXbnd-2*FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-2*FDOH)*2*FDOH){//up
        gidf=gid-NYbnd*NZbnd*2*FDOH-NZbnd*(NXbnd-2*FDOH)*2*FDOH-(NYbnd-2*FDOH)*(NXbnd-2*FDOH)*FDOH;
        i=gidf/(FDOH*(NYbnd-2*FDOH))+lbnd+FDOH;
        j=(gidf/FDOH)%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%FDOH+lbnds;
    }
    else{
        return;
    }



    vxbnd[gid]=vx(k,j,i);
    vybnd[gid]=vy(k,j,i);
    vzbnd[gid]=vz(k,j,i);
    sxxbnd[gid]=sxx(k,j,i);
    syybnd[gid]=syy(k,j,i);
    szzbnd[gid]=szz(k,j,i);
    sxybnd[gid]=sxy(k,j,i);
    syzbnd[gid]=syz(k,j,i);
    sxzbnd[gid]=sxz(k,j,i);

// If we have domain decomposition and it is the first device, we need 5 sides of the boundary
#elif DEVID==0 & MYLOCALID==0
    int gid = get_global_id(0);
    int NXbnd = (NX-2*FDOH-NAB);
    int NYbnd = (NY-2*FDOH-2*NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    int i,j,k;
    int gidf;
    
    if (gid<NYbnd*NZbnd*FDOH){//front
        gidf=gid;
        i=gidf/(NYbnd*NZbnd)+lbnd;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*FDOH){//left
        gidf=gid-NYbnd*NZbnd*FDOH;
        i=gidf/(NZbnd*FDOH)+lbnd+FDOH;
        j=(gidf/NZbnd)%FDOH+lbnd;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH){//right
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*FDOH;
        i=gidf/(NZbnd*FDOH)+lbnd+FDOH;
        j=(gidf/NZbnd)%FDOH+NYbnd+NAB;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-FDOH)*FDOH){//down
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*2*FDOH;
        i=gidf/(FDOH*(NYbnd-2*FDOH))+lbnd+FDOH;
        j=(gidf/FDOH)%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%FDOH+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-FDOH)*2*FDOH){//up
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*2*FDOH-(NYbnd-2*FDOH)*(NXbnd-FDOH)*FDOH;
        i=gidf/(FDOH*(NYbnd-2*FDOH))+lbnd+FDOH;
        j=(gidf/FDOH)%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%FDOH+lbnds;
    }
    else{
        return;
    }
    
    
    vxbnd[gid]=vx(k,j,i);
    vybnd[gid]=vy(k,j,i);
    vzbnd[gid]=vz(k,j,i);
    sxxbnd[gid]=sxx(k,j,i);
    syybnd[gid]=syy(k,j,i);
    szzbnd[gid]=szz(k,j,i);
    sxybnd[gid]=sxy(k,j,i);
    syzbnd[gid]=syz(k,j,i);
    sxzbnd[gid]=sxz(k,j,i);

// If we have domain decomposition and it is the last device, we need 5 sides of the boundary
#elif DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    int gid = get_global_id(0);
    int NXbnd = (NX-2*FDOH-NAB);
    int NYbnd = (NY-2*FDOH-2*NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    int i,j,k;
    int gidf;
    
    if (gid<NYbnd*NZbnd*FDOH){//back
        gidf=gid;
        i=gidf/(NYbnd*NZbnd)+NXbnd;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*FDOH){//left
        gidf=gid-NYbnd*NZbnd*FDOH;
        i=gidf/(NZbnd*FDOH)+FDOH;
        j=(gidf/NZbnd)%FDOH+lbnd;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH){//right
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*FDOH;
        i=gidf/(NZbnd*FDOH)+FDOH;
        j=(gidf/NZbnd)%FDOH+NYbnd+NAB;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-FDOH)*FDOH){//down
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*2*FDOH;
        i=gidf/(FDOH*(NYbnd-2*FDOH))+FDOH;
        j=(gidf/FDOH)%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%FDOH+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-FDOH)*2*FDOH){//up
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*2*FDOH-(NYbnd-2*FDOH)*(NXbnd-FDOH)*FDOH;
        i=gidf/(FDOH*(NYbnd-2*FDOH))+FDOH;
        j=(gidf/FDOH)%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%FDOH+lbnds;
    }
    else{
        return;
    }
    
    
    vxbnd[gid]=vx(k,j,i);
    vybnd[gid]=vy(k,j,i);
    vzbnd[gid]=vz(k,j,i);
    sxxbnd[gid]=sxx(k,j,i);
    syybnd[gid]=syy(k,j,i);
    szzbnd[gid]=szz(k,j,i);
    sxybnd[gid]=sxy(k,j,i);
    syzbnd[gid]=syz(k,j,i);
    sxzbnd[gid]=sxz(k,j,i);

// If we have domain decomposition and it is a middle device, we need 4 sides of the boundary
#else 
    int gid = get_global_id(0);
    int NXbnd = (NX-2*FDOH);
    int NYbnd = (NY-2*FDOH-2*NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    int i,j,k;
    int gidf;
    

    if (gid<NZbnd*NXbnd*FDOH){//left
        gidf=gid;
        i=gidf/(NZbnd*FDOH)+FDOH;
        j=(gidf/NZbnd)%FDOH+lbnd;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NZbnd*NXbnd*2*FDOH){//right
        gidf=gid-NZbnd*NXbnd*FDOH;
        i=gidf/(NZbnd*FDOH)+FDOH;
        j=(gidf/NZbnd)%FDOH+NYbnd+NAB;
        k=gidf%NZbnd+lbnds;
    }
    else if (gid<NZbnd*NXbnd*2*FDOH+(NYbnd-2*FDOH)*NXbnd*FDOH){//down
        gidf=gid-NZbnd*NXbnd*2*FDOH;
        i=gidf/(FDOH*(NYbnd-2*FDOH))+FDOH;
        j=(gidf/FDOH)%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%FDOH+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NZbnd*NXbnd*2*FDOH+(NYbnd-2*FDOH)*NXbnd*2*FDOH){//up
        gidf=gid-NZbnd*NXbnd*2*FDOH-(NYbnd-2*FDOH)*NXbnd*FDOH;
        i=gidf/(FDOH*(NYbnd-2*FDOH))+FDOH;
        j=(gidf/FDOH)%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%FDOH+lbnds;
    }
    else{
        return;
    }
    
    
    vxbnd[gid]=vx(k,j,i);
    vybnd[gid]=vy(k,j,i);
    vzbnd[gid]=vz(k,j,i);
    sxxbnd[gid]=sxx(k,j,i);
    syybnd[gid]=syy(k,j,i);
    szzbnd[gid]=szz(k,j,i);
    sxybnd[gid]=sxy(k,j,i);
    syzbnd[gid]=syz(k,j,i);
    sxzbnd[gid]=sxz(k,j,i);

#endif
    
    
}



