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
#define vx(z,x)  vx[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vy(z,x)  vy[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz(z,x)  vz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxx(z,x) sxx[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define szz(z,x) szz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxz(z,x) sxz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxy(z,x) sxy[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syz(z,x) syz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define lbnd (fdoh+nab)



__kernel void savebnd(__global float *vx,         __global float *vy,      __global float *vz,
                      __global float *sxx,        __global float *syy,     __global float *szz,
                      __global float *sxy,        __global float *syz,     __global float *sxz,
                      __global float *vxbnd,      __global float *vybnd,   __global float *vzbnd,
                      __global float *sxxbnd,     __global float *syybnd,  __global float *szzbnd,
                      __global float *sxybnd,     __global float *syzbnd,  __global float *sxzbnd)
{
    
#if num_devices==1 & NLOCALP==1
    int gid = get_global_id(0);
    int NXbnd = (NX- 2*fdoh- 2*nab);
    int NZbnd = (NZ- 2*fdoh- 2*nab);
    int i=0,k=0;
    int gidf;
    
    if (gid<NZbnd*fdoh){//front
        gidf=gid;
        i=gidf/NZbnd+lbnd;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NZbnd*2*fdoh){//back
        gidf=gid-NZbnd*fdoh;
        i=gidf/(NZbnd)+NXbnd+nab;
        k=gidf%NZbnd+lbnd;
    }
    else if (gid<NZbnd*2*fdoh+(NXbnd - 2*fdoh)*fdoh){//up
        gidf=gid-NZbnd*2*fdoh;
        i=gidf%(NXbnd - 2*fdoh)+lbnd+fdoh;
        k=gidf/(NXbnd- 2*fdoh)+lbnd;
    }
    else if (gid<NZbnd*2*fdoh+(NXbnd- 2*fdoh)*2*fdoh){//bottom
        gidf=gid-NZbnd*2*fdoh-(NXbnd- 2*fdoh)*fdoh;
        i=gidf%(NXbnd- 2*fdoh)+lbnd+fdoh;
        k=gidf/(NXbnd- 2*fdoh)+NZbnd+nab;
    }

    else{
        return;
    }
    

#elif dev==0 & MYGROUPID==0
    
    int gid = get_global_id(0);
    int NXbnd = (NX- 2*fdoh- nab);
    int NZbnd = (NZ- 2*fdoh-2*nab);
    int i,k;
    int gidf;
    
    if (gid<NZbnd*fdoh){//front
        gidf=gid;
        i=gidf/NZbnd+lbnd;
        k=gidf%NZbnd+lbnd;
    }

    else if (gid<NZbnd*fdoh+(NXbnd-fdoh)*fdoh){//up
        gidf=gid-NZbnd*fdoh;
        i=gidf%(NXbnd-fdoh)+lbnd+fdoh;
        k=gidf/(NXbnd-fdoh)+lbnd;
    }
    else if (gid<NZbnd*fdoh+(NXbnd-fdoh)*2*fdoh){//bottom
        gidf=gid-NZbnd*fdoh-(NXbnd-fdoh)*fdoh;
        i=gidf%(NXbnd-fdoh)+lbnd+fdoh;
        k=gidf/(NXbnd-fdoh)+NZbnd+nab;
    }
    
    else{
        return;
    }

#elif dev==num_devices-1 & MYGROUPID==NLOCALP-1
    
    int gid = get_global_id(0);
    int NXbnd = (NX- 2*fdoh- nab);
    int NZbnd = (NZ- 2*fdoh-2*nab);
    int i,k;
    int gidf;
    
    if (gid<NZbnd*fdoh){//back
        gidf=gid;
        i=gidf/(NZbnd)+NXbnd;//+nab;
        k=gidf%NZbnd+lbnd;
    }
    
    else if (gid<NZbnd*fdoh+(NXbnd-fdoh)*fdoh){//up
        gidf=gid-NZbnd*fdoh;
        i=gidf%(NXbnd-fdoh)+fdoh;
        k=gidf/(NXbnd-fdoh)+lbnd;
    }
    else if (gid<NZbnd*fdoh+(NXbnd-fdoh)*2*fdoh){//bottom
        gidf=gid-NZbnd*fdoh-(NXbnd-fdoh)*fdoh;
        i=gidf%(NXbnd-fdoh)+fdoh;
        k=gidf/(NXbnd-fdoh)+NZbnd+nab;
    }
    
    else{
        return;
    }
    
    
#else 
    
    int gid = get_global_id(0);
    int NXbnd = (NX- 2*fdoh);
    int NZbnd = (NZ- 2*fdoh-2*nab);
    int i,k;
    int gidf;
    

    if (gid<(NXbnd)*fdoh){//up
        gidf=gid;
        i=gidf%(NXbnd)+fdoh;
        k=gidf/(NXbnd)+lbnd;
    }
    else if (gid<NZbnd*fdoh+(NXbnd)*2*fdoh){//bottom
        gidf=gid-(NXbnd)*fdoh;
        i=gidf%(NXbnd)+fdoh;
        k=gidf/(NXbnd)+NZbnd+nab;
    }
    
    else{
        return;
    }
    
#endif

#if ND==2
    vxbnd[gid]=vx(k,i);
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



