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

/*This is the kernel that implement the free surface condition for SH waves*/

#define syz(z,x) syz[(x)*NZ+(z)]

__kernel void surface(        __global float *syz)
{
    /*Indice definition */
    int gidx = get_global_id(0) + fdoh;
    int gidz=fdoh;
    
    /* Global work size is padded to be a multiple of local work size. The padding elements must not be updated */
    if ( gidx>(NX-fdoh-1) ){
        return;
    }
    
    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    
    for (int m=1; m<=fdoh; m++) {
        syz(gidz-m, gidx)=-syz(gidz+m-1, gidx);
    }


}



