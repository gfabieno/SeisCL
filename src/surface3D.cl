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


#define psi_vx_x(z,y,x) psi_vx_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vy_x(z,y,x) psi_vy_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vz_x(z,y,x) psi_vz_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vx_y(z,y,x) psi_vx_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vy_y(z,y,x) psi_vy_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vz_y(z,y,x) psi_vz_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vx_z(z,y,x) psi_vx_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vy_z(z,y,x) psi_vy_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vz_z(z,y,x) psi_vz_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]

#define indv(z,y,x) (x)*NZ*NY+(y)*NZ+(z)

FUNDEF void freesurface(GLOBARG float *vx,  GLOBARG float *vy,
                        GLOBARG float *vz,  GLOBARG float *sxx,
                        GLOBARG float *syy, GLOBARG float *szz,
                        GLOBARG float *sxy, GLOBARG float *syz,
                        GLOBARG float *sxz, GLOBARG float *M,
                        GLOBARG float *mu,  GLOBARG float *rxx,
                        GLOBARG float *ryy, GLOBARG float *rzz,
                        GLOBARG float *taus,GLOBARG float *taup,
                        GLOBARG float *eta, GLOBARG float *K_x,
                        GLOBARG float *psi_vx_x, GLOBARG float *K_y,
                        GLOBARG float *psi_vy_y, GLOBARG float *taper,
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

    int indp = ((gidx)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((gidy)-FDOH)*(NZ-2*FDOH)+((gidz)-FDOH);

    //For the FD templates in header_FD to work, we must define:
    int lidx= gidx;
    int lidy= gidy;
    int lidz= gidz;
    int lsizez=NZ;
    int lsizey=NY;

    /* Global work size is padded to be a multiple of local work size. The padding elements must not be updated */
    if (gidy>(NY-FDOH-1) || gidx>(NX-FDOH-1) ){
        return;
    }
    
    float f, g, h;
    float  vxx, vyy, vzz;
    int m;
    
    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    szz[indv(gidz, gidy, gidx)]=0.0;
    #if LVE>0
    int l;
    for (l=0; l<LVE; l++){
        rzz[l*NX*NY*NZ + gidx*NY*NZ + gidy*NZ +gidz]=0.0;
    }
    #endif
    
    for (m=1; m<=FDOH; m++) {
        szz[indv(gidz-m, gidy, gidx)]=-szz[indv(gidz+m, gidy, gidx)];
        sxz[indv(gidz-m, gidy, gidx)]=-sxz[indv(gidz+m-1, gidy, gidx)];
        syz[indv(gidz-m, gidy, gidx)]=-syz[indv(gidz+m-1, gidy, gidx)];
	}
				
    vxx = Dxm(vx);
    vyy = Dym(vy);
    vzz = Dzm(vz);


// Correct spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
        int i,j,k,ind;

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            vyy = vyy / K_y[j] + psi_vy_y(k,j,i);
        }

        else if (gidy>NY-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NY+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-j;
            vyy = vyy / K_y[ind+1] + psi_vy_y(k,j,i);
        }
        #if DEV==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            vxx = vxx / K_x[i] + psi_vx_x(k,j,i);
        }
        #endif

        #if DEV==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            vxx = vxx /K_x[ind+1] + psi_vx_x(k,j,i);
        }
        #endif
    }
#endif

#if LVE==0
    f=mu[indp]*2.0f;
    g=M[indp];
    h=-((g-f)*(g-f)*(vxx+vyy)/g)-((g-f)*vzz);

    #if ABS_TYPE==2
        {
        if (gidy-FDOH<NAB){
                h*=taper[gidy-FDOH];
        }
        if (gidy>NY-NAB-FDOH-1){
                h*=taper[NY-FDOH-gidy-1];
        }
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

    sxx[indv(gidz, gidy, gidx)]+=h;
    syy[indv(gidz, gidy, gidx)]+=h;

#else
    float b,d,e, sumx, sumy;
    int indr;
    /* partially updating sxx and syy in the same way*/
    f=mu[indp]*2.0*(1.0+LVE*taus[indp]);
    g=M[indp]*(1.0+LVE*taup[indp]);
    h=-((g-f)*(g-f)*(vxx+vyy)/g)-((g-f)*vzz);

    sumx=0;
    sumy=0;
    for (l=0;l<LVE;l++){
        indr = l*NX*NY*NZ + gidx*NY*NZ + gidy*NZ +gidz;
        sumx+=rxx[indr];
        sumy+=ryy[indr];
    }

    #if ABS_TYPE==2
        {
        if (gidy-FDOH<NAB){
                h*=taper[gidy-FDOH];
                sumx*=taper[gidy-FDOH];
                sumy*=taper[gidy-FDOH];
        }
        if (gidy>NY-NAB-FDOH-1){
                h*=taper[NY-FDOH-gidy-1];
                sumx*=taper[NY-FDOH-gidy-1];
                sumy*=taper[NY-FDOH-gidy-1];
        }
        #if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                h*=taper[gidx-FDOH];
                sumx*=taper[gidx-FDOH];
                sumy*=taper[gidx-FDOH];
            }
        #endif
        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                h*=taper[NX-FDOH-gidx-1];
                sumx*=taper[NX-FDOH-gidx-1];
                sumy*=taper[NX-FDOH-gidx-1];
            }
        #endif
        }
    #endif

    sxx[indv(gidz, gidy, gidx)]+=pdir*(h-DT2*sumx);
    syy[indv(gidz, gidy, gidx)]+=pdir*(h-DT2*sumy);
    
    /* updating the memory-variable rxx, ryy at the free surface */
    
    d=2.0*mu[indp]*taus[indp]/DT;
    e=M[indp]*taup[indp]/DT;
    sumx=0;
    sumy=0;
    for (l=0;l<LVE;l++){
        indr = l*NX*NY*NZ + gidx*NY*NZ + gidy*NZ +gidz;
        b=eta[l]/(1.0+(eta[l]*0.5));
        h=b*(((d-e)*((f/g)-1.0)*(vxx+vyy))-((d-e)*vzz));
        rxx[indr]+=h;
        ryy[indr]+=h;
        sumx+=rxx[indr];
        sumy+=ryy[indr];
    }

    #if ABS_TYPE==2
        {
        if (gidy-FDOH<NAB){
                sumx*=taper[gidy-FDOH];
                sumy*=taper[gidy-FDOH];
        }
        if (gidy>NY-NAB-FDOH-1){
                sumx*=taper[NY-FDOH-gidy-1];
                sumy*=taper[NY-FDOH-gidy-1];
        }
        #if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                sumx*=taper[gidx-FDOH];
                sumy*=taper[gidx-FDOH];
            }
        #endif
        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                sumx*=taper[NX-FDOH-gidx-1];
                sumy*=taper[NX-FDOH-gidx-1];
            }
        #endif
        }
    #endif

    /*completely updating the stresses sxx and syy */
    sxx[indv(gidz, gidy, gidx)]+=(DT2*sumx);
    syy[indv(gidz, gidy, gidx)]+=(DT2*sumy);
    
#endif

    
}



