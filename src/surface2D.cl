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

#define indv(z,x)  (x)*(NZ)+(z)

FUNDEF void surface(GLOBARG float *vx,  GLOBARG float *vz,
                    GLOBARG float *sxx, GLOBARG float *szz,
                    GLOBARG float *sxz, GLOBARG float *M,
                    GLOBARG float *mu,  GLOBARG float *rxx,
                    GLOBARG float *rzz, GLOBARG float *taus,
                    GLOBARG float *taup,GLOBARG float *eta,
                    GLOBARG float *K_x, GLOBARG float *psi_vx_x,
                    GLOBARG float *taper, pdir)
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
    int lidz= gidz;
    int lsizez=NZ;
    
    /* Global work size is padded to be a multiple of local work size.
     The padding elements must not be updated */
    if ( gidx>(NX-FDOH-1) ){
        return;
    }

    float f, g, h;
    float sump;
    float  vxx, vzz;
    int l, m;
    int indp = (gidx-FDOH)*(NZ-2*FDOH)+(gidz-FDOH);

    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    szz[indv(gidz, gidx)]=0.0;
    #if LVE>0
    for (l=0; l<LVE; l++){
        rzz[(l)*NX*NZ+(gidx)*NZ+(gidz)]=0.0;
    }
    #endif

    for (m=1; m<=FDOH; m++) {
        szz[indv(gidz-m,gidx)]=-szz[indv(gidz+m,gidx)];
        sxz[indv(gidz-m,gidx)]=-sxz[indv(gidz+m-1,gidx)];
    }

    vxx = Dxm(vx);
    vzz = Dzm(vz);

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
    f=mu[indp]*2.0;
    g=M[indp];
    h=-((g-f)*(g-f)*(vxx)/g)-((g-f)*vzz);

    // Absorbing boundary
    #if ABS_TYPE==2
        {

        #if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                h*=taper[gidx-FDOH];
            }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                h*=taper[NX-FDOH-gidx-1];
            }
        #endif
        }
    #endif
    sxx[indv(gidz,gidx)]+=pdir*h;

    #else
    float b,d,e;
    /* partially updating sxx  in the same way*/
    f=mu[indp]*2.0*(1.0+LVE*taus[indp]);
    g=M[indp]*(1.0+LVE*taup[indp]);
    h=-((g-f)*(g-f)*(vxx)/g)-((g-f)*vzz);
    #if ABS_TYPE==2
        {

        #if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                h*=taper[gidx-FDOH];
            }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                h*=taper[NX-FDOH-gidx-1];
            }
        #endif
        }
    #endif

    sump=0;
    for (l=0;l<LVE;l++){
        sump+=rxx[(l)*NX*NZ+(gidx)*NZ+(gidz)];
    }
    sxx[indv(gidz,gidx)]+=pdir* (h - DT/2.0*sump);

    /* updating the memory-variable rxx at the free surface */
    d=2.0*mu[indp]*taus[indp];
    e=M[indp]*taup[indp];

    sump=0;
    for (l=0;l<LVE;l++){
        b=eta[l]/(1.0+(eta[l]*0.5));
        h=b*(((d-e)*((f/g)-1.0)*vxx)-((d-e)*vzz));
        rxx[(l)*NX*NZ+(gidx)*NZ+(gidz)]+=pdir*h;
        /*completely updating the stresses sxx  */
    }
    sxx[indv(gidz,gidx)]+=pdir*(DT/2.0*sump);
#endif
    
}



