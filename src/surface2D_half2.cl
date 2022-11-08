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


#define psi_vx_x(z,x) psi_vx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_vzx(z,x) psi_vzx[(x)*(NZ-2*FDOH)+(z)]

#define psi_vxz(z,x) psi_vxz[(x)*(2*NAB)+(z)]
#define psi_vzz(z,x) psi_vzz[(x)*(2*NAB)+(z)]

#define indv(z,x)  (x)*(NZ*DIV)+(z)

FUNDEF void freesurface(GLOBARG __prec2 *vx,  GLOBARG __prec *vz,
                        GLOBARG __prec *sxx, GLOBARG __prec *szz,
                        GLOBARG __prec *sxz, GLOBARG __prec *M,
                        GLOBARG __prec *mu,  GLOBARG __prec *rxx,
                        GLOBARG __prec *rzz, GLOBARG __prec *taus,
                        GLOBARG __prec *taup,GLOBARG float *eta,
                        GLOBARG float *K_x,   GLOBARG __prec *psi_vx_x,
                        GLOBARG float *taper, int pdir)
{
    /*Indice definition */
    #ifdef __OPENCL_VERSION__
    int gidx = get_global_id(0) + FDOH;
    #else
    int gidx = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    #endif
    int gidz=FDOH;

    //For the FD templates in header_FD to work, we must define:
    int lidx= gidx;
    int lidz= gidz/DIV;
    int lsizez=NZ;

    /* Global work size is padded to be a multiple of local work size.
     The padding elements must not be updated */
    if ( gidx>(NX-FDOH-1) ){
        return;
    }

    __prec f, g, h;
    __prec sump;
    __cprec  vxx2, vzz2;
    int m;
    int indp = (gidx-FDOH)*(NZ*DIV-2*FDOH)+(gidz-FDOH);

    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    szz[indv(gidz, gidx)]=0.0;
    #if LVE>0
    int l;
    for (l=0; l<LVE; l++){
        rzz[(l)*NX*NZ*DIV+(gidx)*NZ*DIV+(gidz)]=0.0;
    }
    #endif

    for (m=1; m<=FDOH; m++) {
        szz[indv(gidz-m,gidx)]=-szz[indv(gidz+m,gidx)];
        sxz[indv(gidz-m,gidx)]=-sxz[indv(gidz+m-1,gidx)];
    }

    vxx2 = Dxm(vx);
    vzz2 = Dzm(vz);
    #if FP16>0
    __prec  vxx = vxx2.x;
    __prec  vzz = vzz2.x;
    #else
    __prec  vxx = vxx2;
    __prec  vzz = vzz2;
    #endif

    // Correct spatial derivatives to implement CPML
    #if ABS_TYPE==1
    {
        int i,k,ind;
    #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            k =gidz-FDOH;

            vxx = vxx / K_x[i] + psi_vx_x(k,i);
        }
    #endif

    #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            vxx = vxx /K_x[ind+1] + psi_vx_x(k,i);
        }
    #endif
    }
    #endif

    #if LVE==0
    f=mu[indp]*(__prec)2.0f;
    g=M[indp];
    h=-((g-f)*(g-f)*(vxx)/g)-((g-f)*vzz);

//    // Absorbing boundary
//    #if ABS_TYPE==2
//        {
//
//        #if DEVID==0 & MYLOCALID==0
//            if (gidx-FDOH<NAB){
//                h*=taper[gidx-FDOH];
//            }
//        #endif
//
//        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//            if (gidx>NX-NAB-FDOH-1){
//                h*=taper[NX-FDOH-gidx-1];
//            }
//        #endif
//        }
//    #endif
    sxx[indv(gidz,gidx)]+=(__prec)pdir*h;

    #else
    __prec b,d,e;
    /* partially updating sxx  in the same way*/
    f=mu[indp]*(__prec)2.0*((__prec)1.0+(__prec)LVE*taus[indp]);
    g=M[indp]*((__prec)1.0+(__prec)LVE*taup[indp]);
    h=-((g-f)*(g-f)*(vxx)/g)-((g-f)*vzz);

    sump=0;
    for (l=0;l<LVE;l++){
        sump+=rxx[(l)*NX*NZ*DIV+(gidx)*NZ*DIV+(gidz)];
    }
    h+=-(__prec)DT2*sump;
//    #if ABS_TYPE==2
//        {
//        #if DEVID==0 & MYLOCALID==0
//            if (gidx-FDOH<NAB){
//                h*=taper[gidx-FDOH];
//            }
//        #endif
//
//        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//            if (gidx>NX-NAB-FDOH-1){
//                h*=taper[NX-FDOH-gidx-1];
//            }
//        #endif
//        }
//    #endif
    sxx[indv(gidz,gidx)]+=(__prec)pdir * (h);

    /* updating the memory-variable rxx at the free surface */
    d=(__prec)2.0*mu[indp]*taus[indp]/(__prec)DT;
    e=M[indp]*taup[indp]/(__prec)DT;

    sump=0;
    for (l=0;l<LVE;l++){
        b=(__prec)(eta[l]/(1.0+(eta[l]*0.5)));
        h=b*(((d-e)*((f/g)-(__prec)1.0)*vxx)-((d-e)*vzz));
        rxx[(l)*NX*NZ*DIV+(gidx)*NZ*DIV+(gidz)]+=(__prec)pdir*h;
        sump+=rxx[(l)*NX*NZ*DIV+(gidx)*NZ*DIV+(gidz)];
    }
//    #if ABS_TYPE==2
//        {
//        #if DEVID==0 & MYLOCALID==0
//            if (gidx-FDOH<NAB){
//                sump*=taper[gidx-FDOH];
//            }
//        #endif
//
//        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//            if (gidx>NX-NAB-FDOH-1){
//                sump*=taper[NX-FDOH-gidx-1];
//            }
//        #endif
//        }
//    #endif
    sxx[indv(gidz,gidx)]+=(__prec)pdir*((__prec)DT2*sump);
#endif

}



