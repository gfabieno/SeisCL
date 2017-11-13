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
#define vx(z,x)  vx[(x)*NZ+(z)]
#define vy(z,x)  vy[(x)*NZ+(z)]
#define vz(z,x)  vz[(x)*NZ+(z)]
#define sxx(z,x) sxx[(x)*NZ+(z)]
#define szz(z,x) szz[(x)*NZ+(z)]
#define sxz(z,x) sxz[(x)*NZ+(z)]
#define sxy(z,x) sxy[(x)*NZ+(z)]
#define syz(z,x) syz[(x)*NZ+(z)]
#define lbnd (FDOH+NAB)

#if FP16==0

#define __prec float
#define __prec2 float2

#else

#define __prec half
#define __prec2 half2

#endif

extern "C" __global__ void savebnd(__prec2 *sxx,__prec2 *sxz,__prec2 *szz,
                                   __prec2 *vx,__prec2 *vz,
                                   __prec2 *sxxbnd,__prec2 *sxzbnd,__prec2 *szzbnd,
                                   __prec2 *vxbnd,__prec2 *vzbnd)
{
    
#if NUM_DEVICES==1 & NLOCALP==1
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int NXbnd = (NX- 2*FDOH- 2*NAB);
    int NZbnd = (NZ- FDOH- NAB);
    int i=0,k=0;
    int gidf;
    
    if (gid<NZbnd*FDOH/2){//front
        gidf=gid;
        i=gidf/NZbnd+lbnd;
        k=gidf%NZbnd+lbnd/2;
    }
    else if (gid<NZbnd*FDOH){//back
        gidf=gid-NZbnd*FDOH/2;
        i=gidf/(NZbnd)+NXbnd+NAB;
        k=gidf%NZbnd+lbnd/2;
    }
    else if (gid<NZbnd*FDOH+(NXbnd - 2*FDOH)*FDOH){//up
        gidf=gid-NZbnd*FDOH;
        i=gidf%(NXbnd - 2*FDOH)+lbnd+FDOH;
        k=gidf/(NXbnd- 2*FDOH)/2+lbnd/2;
    }
    else if (gid<NZbnd*FDOH+(NXbnd- 2*FDOH)*2*FDOH){//bottom
        gidf=gid-NZbnd*FDOH-(NXbnd- 2*FDOH)*FDOH;
        i=gidf%(NXbnd- 2*FDOH)+lbnd+FDOH;
        k=gidf/(NXbnd- 2*FDOH)/2+NZbnd+NAB/2;
    }

    else{
        return;
    }
    

#elif DEVID==0 & MYGROUPID==0
    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int NXbnd = (NX- 2*FDOH- NAB);
    int NZbnd = (NZ*2- 2*FDOH-2*NAB);
    int i,k;
    int gidf;
    
    if (gid<NZbnd*FDOH){//front
        gidf=gid;
        i=gidf/NZbnd+lbnd;
        k=gidf%NZbnd+lbnd;
    }

    else if (gid<NZbnd*FDOH+(NXbnd-FDOH)*FDOH){//up
        gidf=gid-NZbnd*FDOH;
        i=gidf%(NXbnd-FDOH)+lbnd+FDOH;
        k=gidf/(NXbnd-FDOH)+lbnd;
    }
    else if (gid<NZbnd*FDOH+(NXbnd-FDOH)*2*FDOH){//bottom
        gidf=gid-NZbnd*FDOH-(NXbnd-FDOH)*FDOH;
        i=gidf%(NXbnd-FDOH)+lbnd+FDOH;
        k=gidf/(NXbnd-FDOH)+NZbnd+NAB;
    }
    
    else{
        return;
    }

#elif DEVID==NUM_DEVICES-1 & MYGROUPID==NLOCALP-1
    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int NXbnd = (NX- 2*FDOH- NAB);
    int NZbnd = (NZ*2- 2*FDOH-2*NAB);
    int i,k;
    int gidf;
    
    if (gid<NZbnd*FDOH){//back
        gidf=gid;
        i=gidf/(NZbnd)+NXbnd;//+NAB;
        k=gidf%NZbnd+lbnd;
    }
    
    else if (gid<NZbnd*FDOH+(NXbnd-FDOH)*FDOH){//up
        gidf=gid-NZbnd*FDOH;
        i=gidf%(NXbnd-FDOH)+FDOH;
        k=gidf/(NXbnd-FDOH)+lbnd;
    }
    else if (gid<NZbnd*FDOH+(NXbnd-FDOH)*2*FDOH){//bottom
        gidf=gid-NZbnd*FDOH-(NXbnd-FDOH)*FDOH;
        i=gidf%(NXbnd-FDOH)+FDOH;
        k=gidf/(NXbnd-FDOH)+NZbnd+NAB;
    }
    
//    else{
//        return;
//    }
    
    
#else 
    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int NXbnd = (NX- 2*FDOH);
    int NZbnd = (NZ*2- 2*FDOH-2*NAB);
    int i,k;
    int gidf;
    

    if (gid<(NXbnd)*FDOH){//up
        gidf=gid;
        i=gidf%(NXbnd)+FDOH;
        k=gidf/(NXbnd)+lbnd;
    }
    else if (gid<NZbnd*FDOH+(NXbnd)*2*FDOH){//bottom
        gidf=gid-(NXbnd)*FDOH;
        i=gidf%(NXbnd)+FDOH;
        k=gidf/(NXbnd)+NZbnd+NAB;
    }
    
//    else{
//        return;
//    }
    
#endif

    if (gid>NBND){
        return;
    }
    
#if ND==2
    float2 out;
    out.x = gid;
    out.y = gid;
    
    vxbnd[gid]=out;//vx(k,i);
    vzbnd[gid]=vz(k,i);
    sxxbnd[gid]=sxx(k,i);
    szzbnd[gid]=szz(k,i);
    sxzbnd[gid]=sxz(k,i);
#endif
#if ND==21
    vybnd[gid]=vy(k,i);
    sxybnd[gid]=sxy(k,i);
    syzbnd[gid]=syz(k,i);
#endif
    
}



