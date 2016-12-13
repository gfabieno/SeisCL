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
#define grad(z,y,x)   grad[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define grads(z,y,x) grads[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define amp1(z,y,x)   amp1[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define amp2(z,y,x)   amp2[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

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
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]


__kernel void savebnd(__global float *vx,         __global float *vy,      __global float *vz,
                      __global float *sxx,        __global float *syy,     __global float *szz,
                      __global float *sxy,        __global float *syz,     __global float *sxz,
                      __global float *vxbnd,      __global float *vybnd,   __global float *vzbnd,
                      __global float *sxxbnd,     __global float *syybnd,  __global float *szzbnd,
                      __global float *sxybnd,     __global float *syzbnd,  __global float *sxzbnd)
{
// If we have one device and one processing element in the group, we need all 6 sides of the boundary
#if num_devices==1 & NLOCALP==1
    int gid = get_global_id(0);
    int NXbnd = (NX-2*fdoh-2*nab);
    int NYbnd = (NY-2*fdoh-2*nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    int i,j,k;
    int gidf;
    
    if (gid<NYbnd*NZbnd*fdoh){//front
        gidf=gid;
        i=gidf/(NYbnd*NZbnd)+lbnd;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NYbnd*NZbnd*2*fdoh){//back
        gidf=gid-NYbnd*NZbnd*fdoh;
        i=gidf/(NYbnd*NZbnd)+NXbnd+nab;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NYbnd*NZbnd*2*fdoh+NZbnd*(NXbnd-2*fdoh)*fdoh){//left
        gidf=gid-NYbnd*NZbnd*2*fdoh;
        i=gidf/(NZbnd*fdoh)+lbnd+fdoh;
        j=(gidf/NZbnd)%fdoh+lbnd;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NYbnd*NZbnd*2*fdoh+NZbnd*(NXbnd-2*fdoh)*2*fdoh){//right
        gidf=gid-NYbnd*NZbnd*2*fdoh-NZbnd*(NXbnd-2*fdoh)*fdoh;
        i=gidf/(NZbnd*fdoh)+lbnd+fdoh;
        j=(gidf/NZbnd)%fdoh+NYbnd+nab;
        k=gidf%NZbnd+lbnd;

    }
    else if (gid<NYbnd*NZbnd*2*fdoh+NZbnd*(NXbnd-2*fdoh)*2*fdoh+(NYbnd-2*fdoh)*(NXbnd-2*fdoh)*fdoh){//up
        gidf=gid-NYbnd*NZbnd*2*fdoh-NZbnd*(NXbnd-2*fdoh)*2*fdoh;
        i=gidf/(fdoh*(NYbnd-2*fdoh))+lbnd+fdoh;
        j=(gidf/fdoh)%(NYbnd-2*fdoh)+lbnd+fdoh;
        k=gidf%fdoh+lbnd;

    }
    else if (gid<NYbnd*NZbnd*2*fdoh+NZbnd*(NXbnd-2*fdoh)*2*fdoh+(NYbnd-2*fdoh)*(NXbnd-2*fdoh)*2*fdoh){//down
        gidf=gid-NYbnd*NZbnd*2*fdoh-NZbnd*(NXbnd-2*fdoh)*2*fdoh-(NYbnd-2*fdoh)*(NXbnd-2*fdoh)*fdoh;
        i=gidf/(fdoh*(NYbnd-2*fdoh))+lbnd+fdoh;
        j=(gidf/fdoh)%(NYbnd-2*fdoh)+lbnd+fdoh;
        k=gidf%fdoh+NZbnd+nab;
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
#elif dev==0 & MYGROUPID==0
    int gid = get_global_id(0);
    int NXbnd = (NX-2*fdoh-nab);
    int NYbnd = (NY-2*fdoh-2*nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    int i,j,k;
    int gidf;
    
    if (gid<NYbnd*NZbnd*fdoh){//front
        gidf=gid;
        i=gidf/(NYbnd*NZbnd)+lbnd;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NYbnd*NZbnd*fdoh+NZbnd*(NXbnd-fdoh)*fdoh){//left
        gidf=gid-NYbnd*NZbnd*fdoh;
        i=gidf/(NZbnd*fdoh)+lbnd+fdoh;
        j=(gidf/NZbnd)%fdoh+lbnd;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NYbnd*NZbnd*fdoh+NZbnd*(NXbnd-fdoh)*2*fdoh){//right
        gidf=gid-NYbnd*NZbnd*fdoh-NZbnd*(NXbnd-fdoh)*fdoh;
        i=gidf/(NZbnd*fdoh)+lbnd+fdoh;
        j=(gidf/NZbnd)%fdoh+NYbnd+nab;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NYbnd*NZbnd*fdoh+NZbnd*(NXbnd-fdoh)*2*fdoh+(NYbnd-2*fdoh)*(NXbnd-fdoh)*fdoh){//up
        gidf=gid-NYbnd*NZbnd*fdoh-NZbnd*(NXbnd-fdoh)*2*fdoh;
        i=gidf/(fdoh*(NYbnd-2*fdoh))+lbnd+fdoh;
        j=(gidf/fdoh)%(NYbnd-2*fdoh)+lbnd+fdoh;
        k=gidf%fdoh+lbnd;
    }
    else if (gid<NYbnd*NZbnd*fdoh+NZbnd*(NXbnd-fdoh)*2*fdoh+(NYbnd-2*fdoh)*(NXbnd-fdoh)*2*fdoh){//down
        gidf=gid-NYbnd*NZbnd*fdoh-NZbnd*(NXbnd-fdoh)*2*fdoh-(NYbnd-2*fdoh)*(NXbnd-fdoh)*fdoh;
        i=gidf/(fdoh*(NYbnd-2*fdoh))+lbnd+fdoh;
        j=(gidf/fdoh)%(NYbnd-2*fdoh)+lbnd+fdoh;
        k=gidf%fdoh+NZbnd+nab;
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
#elif dev==num_devices-1 & MYGROUPID==NLOCALP-1
    int gid = get_global_id(0);
    int NXbnd = (NX-2*fdoh-nab);
    int NYbnd = (NY-2*fdoh-2*nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    int i,j,k;
    int gidf;
    
    if (gid<NYbnd*NZbnd*fdoh){//back
        gidf=gid;
        i=gidf/(NYbnd*NZbnd)+NXbnd;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NYbnd*NZbnd*fdoh+NZbnd*(NXbnd-fdoh)*fdoh){//left
        gidf=gid-NYbnd*NZbnd*fdoh;
        i=gidf/(NZbnd*fdoh)+fdoh;
        j=(gidf/NZbnd)%fdoh+lbnd;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NYbnd*NZbnd*fdoh+NZbnd*(NXbnd-fdoh)*2*fdoh){//right
        gidf=gid-NYbnd*NZbnd*fdoh-NZbnd*(NXbnd-fdoh)*fdoh;
        i=gidf/(NZbnd*fdoh)+fdoh;
        j=(gidf/NZbnd)%fdoh+NYbnd+nab;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NYbnd*NZbnd*fdoh+NZbnd*(NXbnd-fdoh)*2*fdoh+(NYbnd-2*fdoh)*(NXbnd-fdoh)*fdoh){//up
        gidf=gid-NYbnd*NZbnd*fdoh-NZbnd*(NXbnd-fdoh)*2*fdoh;
        i=gidf/(fdoh*(NYbnd-2*fdoh))+fdoh;
        j=(gidf/fdoh)%(NYbnd-2*fdoh)+lbnd+fdoh;
        k=gidf%fdoh+lbnd;
    }
    else if (gid<NYbnd*NZbnd*fdoh+NZbnd*(NXbnd-fdoh)*2*fdoh+(NYbnd-2*fdoh)*(NXbnd-fdoh)*2*fdoh){//down
        gidf=gid-NYbnd*NZbnd*fdoh-NZbnd*(NXbnd-fdoh)*2*fdoh-(NYbnd-2*fdoh)*(NXbnd-fdoh)*fdoh;
        i=gidf/(fdoh*(NYbnd-2*fdoh))+fdoh;
        j=(gidf/fdoh)%(NYbnd-2*fdoh)+lbnd+fdoh;
        k=gidf%fdoh+NZbnd+nab;
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
    int NXbnd = (NX-2*fdoh);
    int NYbnd = (NY-2*fdoh-2*nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    int i,j,k;
    int gidf;
    

    if (gid<NZbnd*NXbnd*fdoh){//left
        gidf=gid;
        i=gidf/(NZbnd*fdoh)+fdoh;
        j=(gidf/NZbnd)%fdoh+lbnd;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NZbnd*NXbnd*2*fdoh){//right
        gidf=gid-NZbnd*NXbnd*fdoh;
        i=gidf/(NZbnd*fdoh)+fdoh;
        j=(gidf/NZbnd)%fdoh+NYbnd+nab;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NZbnd*NXbnd*2*fdoh+(NYbnd-2*fdoh)*NXbnd*fdoh){//up
        gidf=gid-NZbnd*NXbnd*2*fdoh;
        i=gidf/(fdoh*(NYbnd-2*fdoh))+fdoh;
        j=(gidf/fdoh)%(NYbnd-2*fdoh)+lbnd+fdoh;
        k=gidf%fdoh+lbnd;
    }
    else if (gid<NZbnd*NXbnd*2*fdoh+(NYbnd-2*fdoh)*NXbnd*2*fdoh){//down
        gidf=gid-NZbnd*NXbnd*2*fdoh-(NYbnd-2*fdoh)*NXbnd*fdoh;
        i=gidf/(fdoh*(NYbnd-2*fdoh))+fdoh;
        j=(gidf/fdoh)%(NYbnd-2*fdoh)+lbnd+fdoh;
        k=gidf%fdoh+NZbnd+nab;
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



