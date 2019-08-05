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

#if ND==3
    // Find boundary indice for boundary injection in backpropagation
    LFUNDEF int inject_ind(int k, int j, int i){
        
        
    #if NUM_DEVICES==1 & NLOCALP==1
        
        int NXbnd = (NX-2*FDOH-2*NAB);
        int NYbnd = (NY-2*FDOH-2*NAB);
    #if FREESURF==0
        int NZbnd = (NZ- 2*FDOH- 2*NAB);
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
            
        #if FREESURF==0
            m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH*2+(NXbnd-2*FDOH)*(NYbnd-2*FDOH)*FDOH/DIV+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
        #else
            m=-1;
        #endif
            
        }
        else {//down
            i=i-FDOH;
            j=j-FDOH;
            k=k-NZbnd+FDOH/DIV;
            m=NYbnd*NZbnd*FDOH*2+(NXbnd-2*FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
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
            #if FREESURF==0
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+(NXbnd-FDOH)*(NYbnd-2*FDOH)*FDOH/DIV+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
            #else
            m=-1;
            #endif
        }
        else {//down
            i=i-FDOH;
            j=j-FDOH;
            k=k-NZbnd+FDOH/DIV;
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
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
        else if (k<FDOH){//up
            j=j-FDOH;
            #if FREESURF==0
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+(NXbnd-FDOH)*(NYbnd-2*FDOH)*FDOH/DIV+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
            #else
            m=-1;
            #endif
        }
        else {//down
            j=j-FDOH/DIV;
            k=k-NZbnd+FDOH;
            m=NYbnd*NZbnd*FDOH+(NXbnd-FDOH)*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
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
            
        #if FREESURF==0
            m=NXbnd*NZbnd*FDOH*2+NXbnd*(NYbnd-2*FDOH)*FDOH/DIV+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
        #else
            m=-1;
        #endif
        }
        else {//down
            j=j-FDOH;
            k=k-NZbnd+FDOH/DIV;
            m=NXbnd*NZbnd*FDOH*2+i*(NYbnd-2*FDOH)*FDOH+j*FDOH+k;
        }
        
    #endif
        
        
        return m;
        
    }

#else


    // Find boundary indice for boundary injection in backpropagation
    LFUNDEF int inject_ind( int k, int i){
        
        
    #if NUM_DEVICES==1 & NLOCALP==1
        
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
        k-=lbnds/2;
        
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
            #if FREESURF==0
            m=NZbnd*FDOH*2+(NXbnd-2*FDOH)*FDOH/DIV+i+k*(NXbnd-2.0*FDOH);
            #else
            m=-1;
            #endif
            
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
            #if FREESURF==0
            m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH/DIV+i+k*(NXbnd-FDOH);
            #else
            m=-1;
            #endif
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
            #if FREESURF==0
            m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH/DIV+i+k*(NXbnd-FDOH);
            #else
            m=-1;
            #endif
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
            #if FREESURF==0
            m=(NXbnd)*FDOH/DIV+i+k*(NXbnd);
            #else
            m=-1;
            #endif
        }
        else {//down
            k=k-NZbnd+FDOH/DIV;
            m=i+k*(NXbnd);
        }
        
        
    #endif
        
        
        return m;
        
    }

#endif
