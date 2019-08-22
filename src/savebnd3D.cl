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

/*Kernels to save boundary wavefield in 3D if backpropagation is used in
 the computation of the gradient */

#if FP16==0
    #define __prec float
    #define __prec2 float
    #define DIV 1
#elif FP16==1
    #define __prec float
    #define __prec2 float2
    #define DIV 2
#else
    #define __prec half
    #define __prec2 half2
    #define DIV 2
#endif


FUNDEF void savebnd(GLOBARG __prec2 *vx,         GLOBARG __prec2 *vy,      GLOBARG __prec2 *vz,
                    GLOBARG __prec2 *sxx,        GLOBARG __prec2 *syy,     GLOBARG __prec2 *szz,
                    GLOBARG __prec2 *sxy,        GLOBARG __prec2 *syz,     GLOBARG __prec2 *sxz,
                    GLOBARG __prec2 *vxbnd,      GLOBARG __prec2 *vybnd,   GLOBARG __prec2 *vzbnd,
                    GLOBARG __prec2 *sxxbnd,     GLOBARG __prec2 *syybnd,  GLOBARG __prec2 *szzbnd,
                    GLOBARG __prec2 *sxybnd,     GLOBARG __prec2 *syzbnd,  GLOBARG __prec2 *sxzbnd)
{

    int i,j,k,indv;
    int gidf;

// If we have one device and one processing element in the group, we need all 6 sides of the boundary
#if NUM_DEVICES==1 & NLOCALP==1
    #ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    #else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    #endif
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

    if (gid<NYbnd*NZbnd*FDOH){//front
        gidf=gid;
        i=gidf/(NYbnd*NZbnd)+lbnd;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NYbnd*NZbnd*2*FDOH){//back
        gidf=gid-NYbnd*NZbnd*FDOH;
        i=gidf/(NYbnd*NZbnd)+NXbnd+NAB;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NYbnd*NZbnd*2*FDOH+NZbnd*(NXbnd-2*FDOH)*FDOH){//left
        gidf=gid-NYbnd*NZbnd*2*FDOH;
        i=gidf/(NZbnd*FDOH)+lbnd+FDOH;
        j=(gidf/NZbnd)%FDOH+lbnd;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NYbnd*NZbnd*2*FDOH+NZbnd*(NXbnd-2*FDOH)*2*FDOH){//right
        gidf=gid-NYbnd*NZbnd*2*FDOH-NZbnd*(NXbnd-2*FDOH)*FDOH;
        i=gidf/(NZbnd*FDOH)+lbnd+FDOH;
        j=(gidf/NZbnd)%FDOH+NYbnd+NAB;
        k=gidf%NZbnd+lbnds/DIV;

    }
    else if (gid<NYbnd*NZbnd*2*FDOH+NZbnd*(NXbnd-2*FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-2*FDOH)*FDOH/DIV){//down
        gidf=gid-NYbnd*NZbnd*2*FDOH-NZbnd*(NXbnd-2*FDOH)*2*FDOH;
        i=gidf/(FDOH/DIV*(NYbnd-2*FDOH))+lbnd+FDOH;
        j=(gidf/(FDOH/DIV))%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%(FDOH/DIV)+NZbnd+lbnds/DIV-FDOH/DIV;


    }
    else if (gid<NYbnd*NZbnd*2*FDOH+NZbnd*(NXbnd-2*FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-2*FDOH)*2*FDOH/DIV){//up
        gidf=gid-NYbnd*NZbnd*2*FDOH-NZbnd*(NXbnd-2*FDOH)*2*FDOH-(NYbnd-2*FDOH)*(NXbnd-2*FDOH)*FDOH/DIV;
        i=gidf/(FDOH/DIV*(NYbnd-2*FDOH))+lbnd+FDOH;
        j=(gidf/(FDOH/DIV))%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%(FDOH/DIV)+lbnds/DIV;
    }
    else{
        return;
    }


// If we have domain decomposition and it is the first device, we need 5 sides of the boundary
#elif DEVID==0 & MYLOCALID==0
    #ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    #else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    #endif
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


    if (gid<NYbnd*NZbnd*FDOH){//front
        gidf=gid;
        i=gidf/(NYbnd*NZbnd)+lbnd;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*FDOH){//left
        gidf=gid-NYbnd*NZbnd*FDOH;
        i=gidf/(NZbnd*FDOH)+lbnd+FDOH;
        j=(gidf/NZbnd)%FDOH+lbnd;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH){//right
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*FDOH;
        i=gidf/(NZbnd*FDOH)+lbnd+FDOH;
        j=(gidf/NZbnd)%FDOH+NYbnd+NAB;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-FDOH)*FDOH/DIV){//down
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*2*FDOH;
        i=gidf/(FDOH/DIV*(NYbnd-2*FDOH))+lbnd+FDOH;
        j=(gidf/(FDOH/DIV))%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%(FDOH/DIV)+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-FDOH)*2*FDOH/DIV){//up
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*2*FDOH-(NYbnd-2*FDOH)*(NXbnd-FDOH)*FDOH/DIV;
        i=gidf/(FDOH/DIV*(NYbnd-2*FDOH))+lbnd+FDOH;
        j=(gidf/(FDOH/DIV))%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%(FDOH/DIV)+lbnds/DIV;
    }
    else{
        return;
    }


// If we have domain decomposition and it is the last device, we need 5 sides of the boundary
#elif DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    #ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    #else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    #endif
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

    if (gid<NYbnd*NZbnd*FDOH){//back
        gidf=gid;
        i=gidf/(NYbnd*NZbnd)+NXbnd;
        j=(gidf/NZbnd)%NYbnd+lbnd;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*FDOH){//left
        gidf=gid-NYbnd*NZbnd*FDOH;
        i=gidf/(NZbnd*FDOH)+FDOH;
        j=(gidf/NZbnd)%FDOH+lbnd;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH){//right
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*FDOH;
        i=gidf/(NZbnd*FDOH)+FDOH;
        j=(gidf/NZbnd)%FDOH+NYbnd+NAB;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-FDOH)*FDOH/DIV){//down
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*2*FDOH;
        i=gidf/(FDOH/DIV*(NYbnd-2*FDOH))+FDOH;
        j=(gidf/(FDOH/DIV))%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%(FDOH/DIV)+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NYbnd*NZbnd*FDOH+NZbnd*(NXbnd-FDOH)*2*FDOH+(NYbnd-2*FDOH)*(NXbnd-FDOH)*2*FDOH/DIV){//up
        gidf=gid-NYbnd*NZbnd*FDOH-NZbnd*(NXbnd-FDOH)*2*FDOH-(NYbnd-2*FDOH)*(NXbnd-FDOH)*FDOH/DIV;
        i=gidf/(FDOH/DIV*(NYbnd-2*FDOH))+FDOH;
        j=(gidf/(FDOH/DIV))%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%(FDOH/DIV)+lbnds/DIV;
    }
    else{
        return;
    }

// If we have domain decomposition and it is a middle device, we need 4 sides of the boundary
#else
    #ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    #else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    #endif
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

    if (gid<NZbnd*NXbnd*FDOH){//left
        gidf=gid;
        i=gidf/(NZbnd*FDOH)+FDOH;
        j=(gidf/NZbnd)%FDOH+lbnd;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NZbnd*NXbnd*2*FDOH){//right
        gidf=gid-NZbnd*NXbnd*FDOH;
        i=gidf/(NZbnd*FDOH)+FDOH;
        j=(gidf/NZbnd)%FDOH+NYbnd+NAB;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NZbnd*NXbnd*2*FDOH+(NYbnd-2*FDOH)*NXbnd*FDOH/DIV){//down
        gidf=gid-NZbnd*NXbnd*2*FDOH;
        i=gidf/(FDOH/DIV*(NYbnd-2*FDOH))+FDOH;
        j=(gidf/(FDOH/DIV))%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%(FDOH/DIV)+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NZbnd*NXbnd*2*FDOH+(NYbnd-2*FDOH)*NXbnd*2*FDOH/DIV){//up
        gidf=gid-NZbnd*NXbnd*2*FDOH-(NYbnd-2*FDOH)*NXbnd*FDOH/DIV;
        i=gidf/(FDOH/DIV*(NYbnd-2*FDOH))+FDOH;
        j=(gidf/(FDOH/DIV))%(NYbnd-2*FDOH)+lbnd+FDOH;
        k=gidf%(FDOH/DIV)+lbnds/DIV;
    }
    else{
        return;
    }

#endif

    indv = i*NY*NZ+j*NZ+k;

    vxbnd[gid]=vx[indv];
    vybnd[gid]=vy[indv];
    vzbnd[gid]=vz[indv];
    sxxbnd[gid]=sxx[indv];
    syybnd[gid]=syy[indv];
    szzbnd[gid]=szz[indv];
    sxybnd[gid]=sxy[indv];
    syzbnd[gid]=syz[indv];
    sxzbnd[gid]=sxz[indv];
    
}



