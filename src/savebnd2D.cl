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

/*Kernels to save boundary wavefield in 2D if backpropagation is used in the computation of the gradient */

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */

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

FUNDEF void savebnd(GLOBARG __prec2 *sxx, GLOBARG __prec2 *sxz, GLOBARG __prec2 *szz,
                                   GLOBARG __prec2 *vx, GLOBARG __prec2 *vz,
                                   GLOBARG __prec2 *sxxbnd, GLOBARG __prec2 *sxzbnd, GLOBARG __prec2 *szzbnd,
                                   GLOBARG __prec2 *vxbnd, GLOBARG __prec2 *vzbnd)
{
    
    int i,k,indv;
    int gidf;
    
#if NUM_DEVICES==1 & NLOCALP==1
#ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
#else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
#endif
    int NXbnd = (NX- 2*FDOH- 2*NAB);
    #if FREESURF==0
    int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
    #else
    int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
    #endif

    if (gid<NZbnd*FDOH){//front
        gidf=gid;
        i=gidf/NZbnd+lbnd;
        k=gidf%NZbnd+lbnds/DIV;
    }
    else if (gid<NZbnd*FDOH*2){//back
        gidf=gid-NZbnd*FDOH;
        i=gidf/(NZbnd)+NXbnd+NAB;
        k=gidf%NZbnd+lbnds/DIV;
        
    }
    else if (gid<NZbnd*FDOH*2+(NXbnd - 2*FDOH)*FDOH/DIV){//bottom
        gidf=gid-NZbnd*FDOH*2;
        i=gidf%(NXbnd- 2*FDOH)+lbnd+FDOH;
        k=gidf/(NXbnd- 2*FDOH)+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NZbnd*FDOH*2+(NXbnd- 2*FDOH)*2*FDOH/DIV){//up
        gidf=gid-NZbnd*FDOH*2-(NXbnd- 2*FDOH)*FDOH/DIV;
        i=gidf%(NXbnd - 2*FDOH)+lbnd+FDOH;
        k=gidf/(NXbnd- 2*FDOH)+lbnds/DIV;
    }
    else{
        return;
    }


#elif DEVID==0 & MYLOCALID==0

#ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
#else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
#endif
    int NXbnd = (NX- 2*FDOH- NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif

    if (gid<NZbnd*FDOH){//front
        gidf=gid;
        i=gidf/NZbnd+lbnd;
        k=gidf%NZbnd+lbnds;
    }

    else if (gid<NZbnd*FDOH+(NXbnd-FDOH)*FDOH/DIV){//bottom
        gidf=gid-NZbnd*FDOH;
        i=gidf%(NXbnd-FDOH)+lbnd+FDOH;
        k=gidf/(NXbnd-FDOH)+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NZbnd*FDOH+(NXbnd-FDOH)*2*FDOH/DIV){//up
        gidf=gid-NZbnd*FDOH-(NXbnd-FDOH)*FDOH/DIV;
        i=gidf%(NXbnd-FDOH)+lbnd+FDOH;
        k=gidf/(NXbnd-FDOH)+lbnds/DIV;
    }
    else{
        return;
    }

#elif DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1

#ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
#else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
#endif
    int NXbnd = (NX- 2*FDOH- NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif

    if (gid<NZbnd*FDOH){//back
        gidf=gid;
        i=gidf/(NZbnd)+NXbnd;//+NAB;
        k=gidf%NZbnd+lbnds;
    }

    else if (gid<NZbnd*FDOH+(NXbnd-FDOH)*FDOH/DIV){//bottom
        gidf=gid-NZbnd*FDOH;
        i=gidf%(NXbnd-FDOH)+FDOH;
        k=gidf/(NXbnd-FDOH)+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NZbnd*FDOH+(NXbnd-FDOH)*2*FDOH/DIV){//up
        gidf=gid-NZbnd*FDOH-(NXbnd-FDOH)*FDOH/DIV;
        i=gidf%(NXbnd-FDOH)+FDOH;
        k=gidf/(NXbnd-FDOH)+lbnds/DIV;
    }
    else{
        return;
    }


#else

#ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
#else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
#endif
    int NXbnd = (NX- 2*FDOH);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif

    if (gid<(NXbnd)*FDOH){//bottom
        gidf=gid;
        i=gidf%(NXbnd)+FDOH;
        k=gidf/(NXbnd)+NZbnd+lbnds/DIV-FDOH/DIV;

    }
    else if (gid<NZbnd*FDOH+(NXbnd)*2*FDOH/DIV){//up
        gidf=gid-(NXbnd)*FDOH/DIV;
        i=gidf%(NXbnd)+FDOH;
        k=gidf/(NXbnd)+lbnds/DIV;
    }
    else{
        return;
    }

#endif

    indv = i*NZ+k;
#if ND==2
    vxbnd[gid]=vx[indv];
    vzbnd[gid]=vz[indv];
    sxxbnd[gid]=sxx[indv];
    szzbnd[gid]=szz[indv];
    sxzbnd[gid]=sxz[indv];
#endif
#if ND==21
    vybnd[gid]=vy[indv];
    sxybnd[gid]=sxy[indv];
    syzbnd[gid]=syz[indv];
#endif
    
}



