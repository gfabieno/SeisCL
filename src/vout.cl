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

/*Output seismograms */

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#if ND==3
#define vx(z,y,x)   vx[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vy(z,y,x)   vy[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz(z,y,x)   vz[(x)*NY*(NZ+NZ_al16)+(y)*(NZ+NZ_al16)+(z)+NZ_al0]
#endif

#if ND==2 || ND==21
#define vx(z,y,x)  vx[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vy(z,y,x)  vy[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz(z,y,x)  vz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#endif

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]


#define rec_pos(y,x) rec_pos[(y)*8+(x)]


__kernel void vout( __global float *vx,     __global float *vy,    __global float *vz,
                    __global float *vxout,  __global float *vyout, __global float *vzout,
                    __global float *rec_pos, int nt)
{
 
    int gid = get_global_id(0);
    
    int i=(int)(rec_pos(gid,0)/DH-0.5)+fdoh;
    int j=(int)(rec_pos(gid,1)/DH-0.5)+fdoh;
    int k=(int)(rec_pos(gid,2)/DH-0.5)+fdoh;
    
    if ( (i-offset)>=fdoh && (i-offset)<(NX-fdoh) ){

#if ND!=21
        vxout(gid, nt)= vx(k,j,i-offset);
        vzout(gid, nt)= vz(k,j,i-offset);
#endif
#if ND==3 || ND==21
        vyout(gid, nt)= vy(k,j,i-offset);
#endif

        

    }
    
}

