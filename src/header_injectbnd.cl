/*Macros for writing kernels compatible with CUDA and OpenCL */

#if FP16==0

#define __prec float
#define __prec2 float
#define DIV 1

#elif FP16==1

#define __prec float
#define __prec2 float2
#define DIV 2

#else

#define __prec half
#define __prec2 half2
#define DIV 2

#endif

#define HAVE_INJECT_IND 1
#ifndef GRADABS
#define GRADABS 0
#endif
/* Frame of stored forward values, single device, FP16 0 (Init_OpenCL.c, NBND):
   the cells between an outer box (FR_OL cells inside the grid edge; FR_OT on top)
   and an inner box FR_W cells further in (FR_WT on top).
   GRADABS 1: the whole absorbing layers + the FDOH band (FR_OL = 0).
   GRADABS 0: the FDOH band + the last layer cell (FR_OL = NAB-1): the gradient
   of the averaged parameters (rip, rkp, muipkp, ...) stored at that cell
   straddles the layer and the band, and the averaging transpose spreads it
   into the first band cell; its forward field must be exact.
   Free surface (FREESURF != 0): no layer on top, FDOH rows from the surface. */
#ifndef FR_OL
#if GRADABS==1
#define FR_OL 0
#else
#define FR_OL ((NAB)>0 ? (NAB)-1 : 0)
#endif
#define FR_W (NAB+FDOH-FR_OL)
#if FREESURF==0
#define FR_OT FR_OL
#define FR_WT FR_W
#else
#define FR_OT 0
#define FR_WT FDOH
#endif
#endif

#if ND==3
    // Find boundary indice for boundary injection in backpropagation
    LFUNDEF int inject_ind(int k, int j, int i){
        
    #if NUM_DEVICES==1 & NLOCALP==1 & FP16==0
        /* Inverse of savebnd3D.cl's frame (FR_* above). */
        int NZo = NZ-2*FDOH-FR_OT-FR_OL, NYo = NY-2*FDOH-2*FR_OL, NXo = NX-2*FDOH-2*FR_OL;
        int W = FR_W, Wt = FR_WT;
        int NXm = NXo-2*W, NYm = NYo-2*W;
        int A = W*NYo*NZo, B = NXm*W*NZo, C = NXm*NYm*Wt;
        int m=-1;
        i-=FDOH+FR_OL; j-=FDOH+FR_OL; k-=FDOH+FR_OT;
        if (k<0 || k>NZo-1 || j<0 || j>NYo-1 || i<0 || i>NXo-1)
            m=-1;
        else if (i<W)
            m=i*NYo*NZo+j*NZo+k;
        else if (i>NXo-1-W)
            m=A+(i-NXo+W)*NYo*NZo+j*NZo+k;
        else if (j<W)
            m=2*A+(i-W)*W*NZo+j*NZo+k;
        else if (j>NYo-1-W)
            m=2*A+B+(i-W)*W*NZo+(j-NYo+W)*NZo+k;
        else if (k<Wt)
            m=2*A+2*B+(i-W)*NYm*Wt+(j-W)*Wt+k;
        else if (k>NZo-1-W)
            m=2*A+2*B+C+(i-W)*NYm*W+(j-W)*W+(k-NZo+W);
        return m;
    #elif NUM_DEVICES==1 & NLOCALP==1
        
        int NXbnd = (NX-2*FDOH-2*NAB);
        int NYbnd = (NY-2*FDOH-2*NAB);
    #if FREESURF==0
        int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH+NAB;
    #else
        int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH;
    #endif
        
        int m=-1;
        i-=lbnd;
        j-=lbnd;
        k-=lbnds/DIV;
        
        if ( (k>FDOH/DIV-1 && k<NZbnd-FDOH/DIV) && (j>FDOH-1 && j<NYbnd-FDOH) && (i>FDOH-1 && i<NXbnd-FDOH) )
            m=-1;
        else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1 || i<0 || i>NXbnd-1 )
            m=-1;
        else if (i<FDOH){//front
            m=i*NYbnd*NZbnd+j*NZbnd+k;
        }
        else if (i>NXbnd-1-FDOH){//back
            i=i-NXbnd+FDOH;
            m=NYbnd*NZbnd*FDOH+i*NYbnd*NZbnd+j*NZbnd+k;
        }
        else if (j<FDOH){//left
            i=i-FDOH;
            m=NYbnd*NZbnd*FDOH*2+i*FDOH*NZbnd+j*NZbnd+k;
        }
        else if (j>NYbnd-1-FDOH){//right
            i=i-FDOH;
            j=j-NYbnd+FDOH;
            m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
        }
        else if (k<FDOH/DIV){//up
            i=i-FDOH;
            j=j-FDOH;
            
//        #if FREESURF==0
            m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH*2+(NXbnd-2*FDOH)*(NYbnd-2*FDOH)*FDOH/DIV+i*(NYbnd-2*FDOH)*FDOH/DIV+j*FDOH/DIV+k;
//        #else
//            m=-1;
//        #endif
            
        }
        else {//down
            i=i-FDOH;
            j=j-FDOH;
            k=k-NZbnd+FDOH/DIV;
            m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH/DIV+j*FDOH/DIV+k;
        }
        
        
        
    #elif DEVID==0 & MYLOCALID==0
        int NXbnd = (NX-2*FDOH-NAB);
        int NYbnd = (NY-2*FDOH-2*NAB);
    #if FREESURF==0
        int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH+NAB;
    #else
        int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH;
    #endif
        
        int m=-1;
        i-=lbnd;
        j-=lbnd;
        k-=lbnds/DIV;
        
        if ( (k>FDOH/DIV-1 && k<NZbnd-FDOH/DIV) && (j>FDOH-1 && j<NYbnd-FDOH) && i>FDOH-1  )
            m=-1;
        else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1 || i<0 || i>NXbnd-1 )
            m=-1;
        else if (i<FDOH){//front
            m=i*NYbnd*NZbnd+j*NZbnd+k;
        }
        else if (j<FDOH){//left
            i=i-FDOH;
            m=NYbnd*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
        }
        else if (j>NYbnd-1-FDOH){//right
            i=i-FDOH;
            j=j-NYbnd+FDOH;
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
        }
        else if (k<FDOH/DIV){//up
            i=i-FDOH;
            j=j-FDOH;
//            #if FREESURF==0
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+(NXbnd-FDOH)*(NYbnd-2*FDOH)*FDOH/DIV+i*(NYbnd-2*FDOH)*FDOH/DIV+j*FDOH/DIV+k;
//            #else
//            m=-1;
//            #endif
        }
        else {//down
            i=i-FDOH;
            j=j-FDOH;
            k=k-NZbnd+FDOH/DIV;
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH/DIV+j*FDOH/DIV+k;
        }
    #elif DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        int NXbnd = (NX-2*FDOH-NAB);
        int NYbnd = (NY-2*FDOH-2*NAB);
    #if FREESURF==0
        int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH+NAB;
    #else
        int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH;
    #endif
        
        int m=-1;
        i-=FDOH;
        j-=lbnd;
        k-=lbnds/DIV;
        
        if ( (k>FDOH/DIV-1 && k<NZbnd-FDOH/DIV) && (j>FDOH-1 && j<NYbnd-FDOH) && i<NXbnd-FDOH )
            m=-1;
        else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1  || i>NXbnd-1 )
            m=-1;
        else if (i>NXbnd-1-FDOH){//back
            i=i-NXbnd+FDOH;
            m=i*NYbnd*NZbnd+j*NZbnd+k;
        }
        else if (j<FDOH){//left
            m=NYbnd*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
        }
        else if (j>NYbnd-1-FDOH){//right
            j=j-NYbnd+FDOH;
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
        }
        else if (k<FDOH/DIV){//up
            j=j-FDOH;
//            #if FREESURF==0
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+(NXbnd-FDOH)*(NYbnd-2*FDOH)*FDOH/DIV+i*(NYbnd-2*FDOH)*FDOH/DIV+j*FDOH/DIV+k;
//            #else
//            m=-1;
//            #endif
        }
        else {//down
            j=j-FDOH;
            k=k-NZbnd+FDOH/DIV;
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH/DIV+j*FDOH/DIV+k;
        }
        
    #else
        int NXbnd = (NX-2*FDOH);
        int NYbnd = (NY-2*FDOH-2*NAB);
    #if FREESURF==0
        int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH+NAB;
    #else
        int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH;
    #endif
        
        int m=-1;
        i-=FDOH;
        j-=lbnd;
        k-=lbnds/DIV;
        
        if ( (k>FDOH/DIV-1 && k<NZbnd-FDOH/DIV) && (j>FDOH-1 && j<NYbnd-FDOH) )
            m=-1;
        else if (k<0 || k>NZbnd-1 || j<0 || j>NYbnd-1  || i<0 || i>NXbnd-1 )
            m=-1;
        else if (j<FDOH){//left
            m=i*FDOH*NZbnd+j*NZbnd+k;
        }
        else if (j>NYbnd-1-FDOH){//right
            j=j-NYbnd+FDOH;
            m=NXbnd*NZbnd*FDOH+i*FDOH*NZbnd+j*NZbnd+k;
        }
        else if (k<FDOH/DIV){//up
            j=j-FDOH;
            
//        #if FREESURF==0
            m=NXbnd*NZbnd*FDOH*2+NXbnd*(NYbnd-2*FDOH)*FDOH/DIV+i*(NYbnd-2*FDOH)*FDOH/DIV+j*FDOH/DIV+k;
//        #else
//            m=-1;
//        #endif
        }
        else {//down
            j=j-FDOH;
            k=k-NZbnd+FDOH/DIV;
            m=NXbnd*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH/DIV+j*FDOH/DIV+k;
        }
        
    #endif
        
        
        return m;
        
    }

#else


    // Find boundary indice for boundary injection in backpropagation
    LFUNDEF int inject_ind( int k, int i){
        
    #if NUM_DEVICES==1 & NLOCALP==1 & FP16==0
        /* Inverse of savebnd2D.cl's frame (FR_* above). */
        int NZo = NZ-2*FDOH-FR_OT-FR_OL, NXo = NX-2*FDOH-2*FR_OL;
        int W = FR_W, Wt = FR_WT;
        int A = W*NZo, NXm = NXo-2*W;
        int m=-1;
        i-=FDOH+FR_OL; k-=FDOH+FR_OT;
        if (k<0 || k>NZo-1 || i<0 || i>NXo-1)
            m=-1;
        else if (i<W)
            m=i*NZo+k;
        else if (i>NXo-1-W)
            m=A+(i-NXo+W)*NZo+k;
        else if (k<Wt)
            m=2*A+(i-W)*Wt+k;
        else if (k>NZo-1-W)
            m=2*A+NXm*Wt+(i-W)*W+(k-NZo+W);
        return m;
    #elif NUM_DEVICES==1 & NLOCALP==1
        
        int NXbnd = (NX-2*FDOH-2*NAB);
    #if FREESURF==0
        int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH+NAB;
    #else
        int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH;
    #endif
        
        int m=-1;
        i-=lbnd;
        k-=lbnds/DIV;
        
        if ( (k>FDOH/DIV-1 && k<NZbnd-FDOH/DIV)  && (i>FDOH-1 && i<NXbnd-FDOH) )
            m=-1;
        else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
            m=-1;
        else if (i<FDOH){//front
            m=i*NZbnd+k;
        }
        else if (i>NXbnd-1-FDOH){//back
            i=i-NXbnd+FDOH;
            m=NZbnd*FDOH+i*NZbnd+k;
        }
        else if (k<FDOH/DIV){//up
            i=i-FDOH;
//            #if FREESURF==0
            m=NZbnd*FDOH*2+(NXbnd-2*FDOH)*FDOH/DIV+i+k*(NXbnd-2.0*FDOH);
//            #else
//            m=-1;
//            #endif
            
        }
        else {//down
            i=i-FDOH;
            k=k-NZbnd+FDOH/DIV;
            m=NZbnd*FDOH*2+i+k*(NXbnd-2.0*FDOH);
        }
        
        
        
    #elif DEVID==0 & MYLOCALID==0
        
        int NXbnd = (NX-2*FDOH-NAB);
    #if FREESURF==0
        int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH+NAB;
    #else
        int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH;
    #endif
        
        int m=-1;
        i-=lbnd;
        k-=lbnds/DIV;
        
        if ( (k>FDOH/DIV-1 && k<NZbnd-FDOH/DIV)  && i>FDOH-1  )
            m=-1;
        else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
            m=-1;
        else if (i<FDOH){//front
            m=i*NZbnd+k;
        }
        else if (k<FDOH/DIV){//up
            i=i-FDOH;
//            #if FREESURF==0
            m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH/DIV+i+k*(NXbnd-FDOH);
//            #else
//            m=-1;
//            #endif
        }
        else {//down
            i=i-FDOH;
            k=k-NZbnd+FDOH/DIV;
            m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
        }
        
    #elif DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        int NXbnd = (NX-2*FDOH-NAB);
    #if FREESURF==0
        int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH+NAB;
    #else
        int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH;
    #endif
        
        int m=-1;
        i-=FDOH;
        k-=lbnds/DIV;
        
        if ( (k>FDOH/DIV-1 && k<NZbnd-FDOH/DIV) && i<NXbnd-FDOH )
            m=-1;
        else if (k<0 || k>NZbnd-1 || i>NXbnd-1 )
            m=-1;
        else if (i>NXbnd-1-FDOH){
            i=i-NXbnd+FDOH;
            m=i*NZbnd+k;
        }
        else if (k<FDOH/DIV){//up
//            #if FREESURF==0
            m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH/DIV+i+k*(NXbnd-FDOH);
//            #else
//            m=-1;
//            #endif
        }
        else {//down
            k=k-NZbnd+FDOH/DIV;
            m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
        }
        
    #else
        int NXbnd = (NX-2*FDOH);
    #if FREESURF==0
        int NZbnd = (NZ- 2*FDOH/DIV- 2*NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH+NAB;
    #else
        int NZbnd = (NZ- 2*FDOH/DIV- NAB/DIV);
        int lbnd = FDOH+NAB;
        int lbnds = FDOH;
    #endif
        
        int m=-1;
        i-=FDOH;
        k-=lbnds/DIV;
        
        if ( (k>FDOH/DIV-1 && k<NZbnd-FDOH/DIV) )
            m=-1;
        else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
            m=-1;
        else if (k<FDOH/DIV){//up
//            #if FREESURF==0
            m=(NXbnd)*FDOH/DIV+i+k*(NXbnd);
//            #else
//            m=-1;
//            #endif
        }
        else {//down
            k=k-NZbnd+FDOH/DIV;
            m=i+k*(NXbnd);
        }
        
        
    #endif
        
        
        return m;
        
    }

#endif
