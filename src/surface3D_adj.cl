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

/*This is the kernel that implement the free surface condition in 2D*/

//TODO: Adjoint surface for viscoelastic propagation

#define indv(z,y,x)  (x)*(NY*NZ*DIV)+(y)*(NZ*DIV)+(z)
#define indp(z,y,x)  ((x)-FDOH)*(NY-2*FDOH)*(NZ*DIV-2*FDOH)+((y)-FDOH)*(NZ*DIV-2*FDOH)+((z)-FDOH)



FUNDEF void surface_adj(GLOBARG __prec *vxr,   GLOBARG __prec *vyr,
                        GLOBARG __prec *vzr,   GLOBARG __prec *sxxr,
                        GLOBARG __prec *syyr,  GLOBARG __prec *szzr,
                        GLOBARG __prec *sxyr,  GLOBARG __prec *syzr,
                        GLOBARG __prec *sxzr,  GLOBARG __prec *M,
                        GLOBARG __prec *rip,   GLOBARG __prec *rjp,
                        GLOBARG __prec *rkp,   GLOBARG __prec *mu,
                        GLOBARG __prec *rxx,   GLOBARG __prec *rzz,
                        GLOBARG __prec *taus,  GLOBARG __prec *taup,
                        GLOBARG float *eta)
{
    /*Indice definition */
    #ifdef __OPENCL_VERSION__
    int gidx = get_global_id(0) + FDOH;
    int gidy = get_global_id(1) + FDOH;
    #else
    int gidy = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y + FDOH;
    #endif
    int gidz=FDOH;
    __cprec1 hc[6] = {HC1, HC2, HC3, HC4, HC5, HC6};

    /* Global work size is padded to be a multiple of local work size.
     The padding elements must not be updated */
    if ( gidx>(NX-FDOH-1) || gidy>(NY-FDOH-1) ){
        return;
    }

    __cprec1 h, hz;
    __cprec1 lszz, lsyy, lsxx;
    __cprec1 lmu, lM;
    
    
    //Read variables in register
    lszz = -szzr[indv(gidz,gidy,gidx)];
    lsxx = -sxxr[indv(gidz,gidy,gidx)];
    lsyy = -sxxr[indv(gidz,gidy,gidx)];
    lmu = mu[indp(gidz,gidy,gidx)];
    lM = M[indp(gidz,gidy,gidx)];
    
    //Preadjoint transformations
    {
        __cprec1 temp1, temp2, temp3, a;
        temp1 = lsxx;
        temp2 = lsyy;
        temp3 = lszz;
        a = (__cprec1)1.0/( (__cprec1)6.0*lM*lmu - (__cprec1)8.0*lmu*lmu);
        lsxx = a * ( (__cprec1)2.0*(lM-lmu) * temp1
                    + (-lM + (__cprec1)2.0 * lmu) * (temp2 +temp3) );
        lsyy = a * ( (__cprec1)2.0*(lM-lmu) * temp2
                    + (-lM + (__cprec1)2.0 * lmu) * (temp1 +temp3) );
        lszz = a * ( (__cprec1)2.0*(lM-lmu) * temp3
                    + (-lM + (__cprec1)2.0 * lmu) * (temp1 +temp2) );
    }
    /*Adjoint of the mirror method*/
    lszz=0.0;
    
    h=-((lM-lmu*(__cprec1)2.0)*(lM-lmu*(__cprec1)2.0)/lM);
    hz=-(lM-lmu*(__cprec1)2.0);
    
    for (int m=0; m<FDOH; m++){
        if ( (gidx-FDOH + m) < (NX - 2*FDOH))
            vxr[indv(gidz,gidy,gidx+m)]+=(hc[m] * h * (lsxx + lsyy) *
                                         (__cprec1)rip[indp(gidz,gidy,gidx+m)]);
        if ( (gidx - (m + 1)) > (FDOH - 1))
            vxr[indv(gidz,gidy,gidx-(m+1))]+=(-hc[m] * h* (lsxx + lsyy) *
                                             (__cprec1)rip[indp(gidz,gidy,gidx-(m+1))]);
        if ( (gidy-FDOH + m) < (NY - 2*FDOH))
            vyr[indv(gidz,gidy+m,gidx)]+=(hc[m] * h * (lsxx + lsyy) *
                                         (__cprec1)rjp[indp(gidz,gidy+m,gidx)]);
        if ( (gidy - (m + 1)) > (FDOH - 1))
            vyr[indv(gidz,gidy-(m+1),gidx)]+=(-hc[m] * h* (lsxx + lsyy) *
                                              (__cprec1)rjp[indp(gidz,gidy-(m+1),gidx)]);
            
        vzr[indv(gidz+m,gidy,gidx)]+=(hc[m] *hz * (lsxx + lsyy)
                                      * (__cprec1)rkp[indp(gidz+m,gidy, gidx)]);
    }
    
    //Perform the post-adjoint transformation
    sxxr[indv(gidz,gidy,gidx)] = -(lM * lsxx + (lM-(__cprec1)2.0*lmu) * (lsyy+lszz));
    syyr[indv(gidz,gidy,gidx)] = -(lM * lsyy + (lM-(__cprec1)2.0*lmu) * (lsxx+lszz));
    szzr[indv(gidz,gidy,gidx)] = -(lM * lszz + (lM-(__cprec1)2.0*lmu) * (lsxx+lsyy));
    

}



