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

/*This is the kernel that implement the free surface condition in 3D*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */


#define I_psi_vx_x(z,y,x) ((x)*(NYS-2*FDOH)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vy_x(z,y,x) ((x)*(NYS-2*FDOH)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vz_x(z,y,x) ((x)*(NYS-2*FDOH)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))

#define I_psi_vx_y(z,y,x) ((x)*(2*NAB)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vy_y(z,y,x) ((x)*(2*NAB)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vz_y(z,y,x) ((x)*(2*NAB)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))

#define I_psi_vx_z(z,y,x) ((x)*(NYS-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z))
#define I_psi_vy_z(z,y,x) ((x)*(NYS-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z))
#define I_psi_vz_z(z,y,x) ((x)*(NYS-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z))

#define indv(z,y,x) ((x)*NZS*NYS+(y)*NZS+(z))

FUNDEF void freesurface(GLOBARG STO *vx,  GLOBARG STO *vy,
                        GLOBARG STO *vz,  GLOBARG STO *sxx,
                        GLOBARG STO *syy, GLOBARG STO *szz,
                        GLOBARG STO *sxy, GLOBARG STO *syz,
                        GLOBARG STO *sxz, GLOBARG STO *M,
                        GLOBARG STO *mu,  GLOBARG STO *rxx,
                        GLOBARG STO *ryy, GLOBARG STO *rzz,
                        GLOBARG STO *taus,GLOBARG STO *taup,
                        GLOBARG float *eta, GLOBARG float *K_x,
                        GLOBARG STO *psi_vx_x, GLOBARG float *K_y,
                        GLOBARG STO *psi_vy_y, GLOBARG float *taper,
                        int pdir)
{
    /*Indice definition */
    #ifdef __OPENCL_VERSION__
    int gidy = get_global_id(0) + FDOH;
    int gidx = get_global_id(1) + FDOH;
    #else
    int gidy = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y + FDOH;
    #endif
    int gidz=FDOH;

    int indp = ((gidx)-FDOH)*(NYS-2*FDOH)*(NZS-2*FDOH)+((gidy)-FDOH)*(NZS-2*FDOH)+((gidz)-FDOH);

    //For the FD templates in header_FD to work, we must define:

    /* Global work size is padded to be a multiple of local work size. The padding elements must not be updated */
    if (gidy>(NYS-FDOH-1) || gidx>(NX-FDOH-1) ){
        return;
    }
    
    float f, g, h;
    float  vxx, vyy, vzz;
    int m;
    
    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    stf(szz, indv(gidz, gidy, gidx), 0.0);
    #if LVE>0
    int l;
    for (l=0; l<LVE; l++){
        stf(rzz, l*NX*NYS*NZS + gidx*NYS*NZS + gidy*NZS +gidz, 0.0);
    }
    #endif
    
    for (m=1; m<=FDOH; m++) {
        stf(szz, indv(gidz-m, gidy, gidx), -ldf(szz, indv(gidz+m, gidy, gidx)));
        stf(sxz, indv(gidz-m, gidy, gidx), -ldf(sxz, indv(gidz+m-1, gidy, gidx)));
        stf(syz, indv(gidz-m, gidy, gidx), -ldf(syz, indv(gidz+m-1, gidy, gidx)));
	}
				
    {
        const float hc[6] = HCV;
        vxx = vyy = vzz = 0.0f;
        for (int mm=0; mm<FDOH; mm++){
            vxx += hc[mm]*(ldf(vx, indv(gidz, gidy, gidx+mm)) - ldf(vx, indv(gidz, gidy, gidx-mm-1)));
            vyy += hc[mm]*(ldf(vy, indv(gidz, gidy+mm, gidx)) - ldf(vy, indv(gidz, gidy-mm-1, gidx)));
            vzz += hc[mm]*(ldf(vz, indv(gidz+mm, gidy, gidx)) - ldf(vz, indv(gidz-mm-1, gidy, gidx)));
        }
    }


//// Correct spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
        int i,j,k,ind;

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            vyy = vyy / K_y[j] + ldf(psi_vy_y, I_psi_vy_y(k,j,i));
        }

        else if (gidy>NYS-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NYS+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-j;
            vyy = vyy / K_y[ind+1] + ldf(psi_vy_y, I_psi_vy_y(k,j,i));
        }
        #if DEV==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            vxx = vxx / K_x[i] + ldf(psi_vx_x, I_psi_vx_x(k,j,i));
        }
        #endif

        #if DEV==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            vxx = vxx /K_x[ind+1] + ldf(psi_vx_x, I_psi_vx_x(k,j,i));
        }
        #endif
    }
#endif

#if LVE==0
    f=ldf(mu, indp)*2.0f;
    g=ldf(M, indp);
    h=-((g-f)*(g-f)*(vxx+vyy)/g)-((g-f)*vzz);

//    #if ABS_TYPE==2
//        {
//        if (gidy-FDOH<NAB){
//                h*=taper[gidy-FDOH];
//        }
//        if (gidy>NYS-NAB-FDOH-1){
//                h*=taper[NYS-FDOH-gidy-1];
//        }
//        #if DEVID==0 & MYLOCALID==0
//            if (gidx-FDOH<NAB){
//                h*=taper[gidx-FDOH];
//            }
//        #endif
//        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//            if (gidx>NX-NAB-FDOH-1){
//                h*=taper[NX-FDOH-gidx-1];
//            }
//        #endif
//        }
//    #endif

    stf(sxx, indv(gidz, gidy, gidx), ldf(sxx, indv(gidz, gidy, gidx)) + (h));
    stf(syy, indv(gidz, gidy, gidx), ldf(syy, indv(gidz, gidy, gidx)) + (h));

#else
    float b,d,e, sumx, sumy;
    int indr;
    /* partially updating sxx and syy in the same way*/
    f=ldf(mu, indp)*2.0f*(1.0f+LVE*ldf(taus, indp));
    g=ldf(M, indp)*(1.0f+LVE*ldf(taup, indp));
    h=-((g-f)*(g-f)*(vxx+vyy)/g)-((g-f)*vzz);

    sumx=0;
    sumy=0;
    for (l=0;l<LVE;l++){
        indr = l*NX*NYS*NZS + gidx*NYS*NZS + gidy*NZS +gidz;
        sumx+=ldf(rxx, indr);
        sumy+=ldf(ryy, indr);
    }

//    #if ABS_TYPE==2
//        {
//        if (gidy-FDOH<NAB){
//                h*=taper[gidy-FDOH];
//                sumx*=taper[gidy-FDOH];
//                sumy*=taper[gidy-FDOH];
//        }
//        if (gidy>NYS-NAB-FDOH-1){
//                h*=taper[NYS-FDOH-gidy-1];
//                sumx*=taper[NYS-FDOH-gidy-1];
//                sumy*=taper[NYS-FDOH-gidy-1];
//        }
//        #if DEVID==0 & MYLOCALID==0
//            if (gidx-FDOH<NAB){
//                h*=taper[gidx-FDOH];
//                sumx*=taper[gidx-FDOH];
//                sumy*=taper[gidx-FDOH];
//            }
//        #endif
//        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//            if (gidx>NX-NAB-FDOH-1){
//                h*=taper[NX-FDOH-gidx-1];
//                sumx*=taper[NX-FDOH-gidx-1];
//                sumy*=taper[NX-FDOH-gidx-1];
//            }
//        #endif
//        }
//    #endif

    stf(sxx, indv(gidz, gidy, gidx), ldf(sxx, indv(gidz, gidy, gidx)) + (pdir*(h-DT2*sumx)));
    stf(syy, indv(gidz, gidy, gidx), ldf(syy, indv(gidz, gidy, gidx)) + (pdir*(h-DT2*sumy)));
    
    /* updating the memory-variable rxx, ryy at the free surface */
    
    d=2.0f*ldf(mu, indp)*ldf(taus, indp)/DT;
    e=ldf(M, indp)*ldf(taup, indp)/DT;
    sumx=0;
    sumy=0;
    for (l=0;l<LVE;l++){
        indr = l*NX*NYS*NZS + gidx*NYS*NZS + gidy*NZS +gidz;
        b=eta[l]/(1.0f+(eta[l]*0.5f));
        h=b*(((d-e)*((f/g)-1.0f)*(vxx+vyy))-((d-e)*vzz));
        stf(rxx, indr, ldf(rxx, indr) + (h));
        stf(ryy, indr, ldf(ryy, indr) + (h));
        sumx+=ldf(rxx, indr);
        sumy+=ldf(ryy, indr);
    }

//    #if ABS_TYPE==2
//        {
//        if (gidy-FDOH<NAB){
//                sumx*=taper[gidy-FDOH];
//                sumy*=taper[gidy-FDOH];
//        }
//        if (gidy>NYS-NAB-FDOH-1){
//                sumx*=taper[NYS-FDOH-gidy-1];
//                sumy*=taper[NYS-FDOH-gidy-1];
//        }
//        #if DEVID==0 & MYLOCALID==0
//            if (gidx-FDOH<NAB){
//                sumx*=taper[gidx-FDOH];
//                sumy*=taper[gidx-FDOH];
//            }
//        #endif
//        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//            if (gidx>NX-NAB-FDOH-1){
//                sumx*=taper[NX-FDOH-gidx-1];
//                sumy*=taper[NX-FDOH-gidx-1];
//            }
//        #endif
//        }
//    #endif

    /*completely updating the stresses sxx and syy */
    stf(sxx, indv(gidz, gidy, gidx), ldf(sxx, indv(gidz, gidy, gidx)) + ((DT2*sumx)));
    stf(syy, indv(gidz, gidy, gidx), ldf(syy, indv(gidz, gidy, gidx)) + ((DT2*sumy)));
    
#endif

    
}



