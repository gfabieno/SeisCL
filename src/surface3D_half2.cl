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

#define indv(z,y,x) (x)*(NY*NZ*DIV)+(y)*(NZ*DIV)+(z)

FUNDEF void freesurface(GLOBARG __prec2 *vx,   GLOBARG __prec2 *vy,
                        GLOBARG __prec *vz,   GLOBARG __prec *sxx,
                        GLOBARG __prec *syy,  GLOBARG __prec *szz,
                        GLOBARG __prec *sxy,  GLOBARG __prec *syz,
                        GLOBARG __prec *sxz,  GLOBARG __prec *M,
                        GLOBARG __prec *mu,   GLOBARG __prec *rxx,
                        GLOBARG __prec *ryy,  GLOBARG __prec *rzz,
                        GLOBARG __prec *taus, GLOBARG __prec *taup,
                        GLOBARG float *eta, GLOBARG float *taper,
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
    
    //For the FD templates in header_FD to work, we must define:
    int lidx= gidx;
    int lidy= gidy;
    int lidz= gidz/DIV;
    int lsizez=NZ;
    int lsizey=NY;
    
    /* Global work size is padded to be a multiple of local work size.
       The padding elements must not be updated */
    if (gidy>(NY-FDOH-1) || gidx>(NX-FDOH-1) ){
        return;
    }
    
    __prec f, g, h;
    __cprec  vxx2, vyy2, vzz2;
    int m, l;
    int indp = ( (gidx-FDOH)*(NY-2*FDOH)*(NZ*DIV-2*FDOH)
                +(gidy-FDOH)*(NZ*DIV-2*FDOH)
                +(gidz-FDOH));
    
    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    szz[indv(gidz, gidy, gidx)]=0.0;
    #if LVE>0
        for (l=0; l<LVE; l++){
            rzz[l*NX*NY*NZ*DIV + gidx*NY*NZ*DIV + gidy*NZ*DIV +gidz]=0.0;
        }
    #endif
    
    for (m=1; m<=FDOH; m++) {
        szz[indv(gidz-m, gidy, gidx)]=-szz[indv(gidz+m, gidy, gidx)];
        sxz[indv(gidz-m, gidy, gidx)]=-sxz[indv(gidz+m-1, gidy, gidx)];
        syz[indv(gidz-m, gidy, gidx)]=-syz[indv(gidz+m-1, gidy, gidx)];
    }
				
    vxx2 = Dxm(vx);
    vyy2 = Dym(vy);
    vzz2 = Dzm(vz);
    #if FP16>0
        __prec  vxx = vxx2.x;
        __prec  vyy = vyy2.x;
        __prec  vzz = vzz2.x;
    #else
        __prec  vxx = vxx2;
        __prec  vyy = vyy2;
        __prec  vzz = vzz2;
    #endif

    #if LVE==0
        f=mu[indp]*(__prec)2.0f;
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
        sxx[indv(gidz,gidy,gidx)]+=(__prec)pdir*h;
        syy[indv(gidz,gidy,gidx)]+=(__prec)pdir*h;
    #else
        float b,d,e, sumxx, sumyy;
        /* partially updating sxx and syy in the same way*/
        f=mu[indp]*(__prec)2.0*((__prec)1.0+LVE*taus[indp]);
        g=M[indp]*(1.0+LVE*taup[indp]);
        h=-((g-f)*(g-f)*(vxx+vyy)/g)-((g-f)*vzz);
    
        sumxx=0;sumyy=0;
        for (l=0;l<LVE;l++){
            sumxx+=rxx[l*NX*NY*NZ*DIV + gidx*NY*NZ*DIV + gidy*NZ*DIV +gidz];
            sumyy+=ryy[l*NX*NY*NZ*DIV + gidx*NY*NZ*DIV + gidy*NZ*DIV +gidz];
        }
        #if ABS_TYPE==2
            {
            if (gidy-FDOH<NAB){
                    h*=taper[gidy-FDOH];
                    sumxx*=taper[gidy-FDOH];
                    sumyy*=taper[gidy-FDOH];
            }
            if (gidy>NY-NAB-FDOH-1){
                    h*=taper[NY-FDOH-gidy-1];
                    sumxx*=taper[NY-FDOH-gidy-1];
                    sumyy*=taper[NY-FDOH-gidy-1];
            }
            #if DEVID==0 & MYLOCALID==0
                if (gidx-FDOH<NAB){
                    h*=taper[gidx-FDOH];
                    sumxx*=taper[gidx-FDOH];
                    sumyy*=taper[gidx-FDOH];
                }
            #endif
            #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
                if (gidx>NX-NAB-FDOH-1){
                    h*=taper[NX-FDOH-gidx-1];
                    sumxx*=taper[NX-FDOH-gidx-1];
                    sumyy*=taper[NX-FDOH-gidx-1];
                }
            #endif
            }
        #endif
        sxx[indv(gidz,gidy,gidx)]+=(__prec)pdir*(h-(DT2*sumxx));
        syy[indv(gidz,gidy,gidx)]+=(__prec)pdir*(h-(DT2*sumyy));
    
        /* updating the memory-variable rxx, ryy at the free surface */
    
        d=2.0*mu[indp]*taus[indp]/DT;
        e=M[indp]*taup[indp]/DT;
        sumxx=0;sumyy=0;
        for (l=0;l<LVE;l++){
            b=eta[l]/(1.0+(eta[l]*0.5));
            h=b*(((d-e)*((f/g)-1.0)*(vxx+vyy))-((d-e)*vzz));
            rxx[l*NX*NY*NZ*DIV + gidx*NY*NZ*DIV + gidy*NZ*DIV +gidz]+=(__prec)pdir*h;
            ryy[l*NX*NY*NZ*DIV + gidx*NY*NZ*DIV + gidy*NZ*DIV +gidz]+=(__prec)pdir*h;
            
            sumxx+=rxx[l*NX*NY*NZ*DIV + gidx*NY*NZ*DIV + gidy*NZ*DIV +gidz];
            sumyy+=ryy[l*NX*NY*NZ*DIV + gidx*NY*NZ*DIV + gidy*NZ*DIV +gidz];
        }
        #if ABS_TYPE==2
            {
            if (gidy-FDOH<NAB){
                    sumxx*=taper[gidy-FDOH];
                    sumyy*=taper[gidy-FDOH];
            }
            if (gidy>NY-NAB-FDOH-1){
                    sumxx*=taper[NY-FDOH-gidy-1];
                    sumyy*=taper[NY-FDOH-gidy-1];
            }
            #if DEVID==0 & MYLOCALID==0
                if (gidx-FDOH<NAB){
                    sumxx*=taper[gidx-FDOH];
                    sumyy*=taper[gidx-FDOH];
                }
            #endif
            #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
                if (gidx>NX-NAB-FDOH-1){
                    sumxx*=taper[NX-FDOH-gidx-1];
                    sumyy*=taper[NX-FDOH-gidx-1];
                }
            #endif
            }
        #endif
        /*completely updating the stresses sxx and syy */
        sxx[indv(gidz,gidy,gidx)]+=(__prec)pdir*(DT2*sumxx);
        syy[indv(gidz,gidy,gidx)]+=(__prec)pdir*(DT2*sumyy);
    
    #endif

    
}



