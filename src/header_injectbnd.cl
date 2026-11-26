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
/* Frame of stored forward values (Init_OpenCL.c, NBND), one layout for every case:
   the cells between an outer box (FR_OL cells inside the grid edge; FR_OT on top)
   and an inner box FR_W cells further in (FR_WT on top), stored as scalar cells
   (FP16 1 and 2 included).
   GRADABS 1: the whole absorbing layers + the FDOH band (FR_OL = 0).
   GRADABS 0: the FDOH band + the last layer cell (FR_OL = NAB-1): the gradient
   of the averaged parameters (rip, rkp, muipkp, ...) stored at that cell
   straddles the layer and the band, and the averaging transpose spreads it
   into the first band cell; its forward field must be exact.
   Free surface (FREESURF != 0): no layer on top, FDOH rows from the surface.
   Domain decomposition along x: each subdomain stores the global frame clipped
   to its own x range. Only the first subdomain has the low x face (FR_OX0,
   FR_WX0) and only the last one the high x face (FR_OX1, FR_WX1); the z (and
   y) faces span the whole local x interior of a middle subdomain. */
#ifndef NZS
#define NZS (NZ*DIV)
#endif
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
#define FR_X_FIRST (DEVID==0 && MYLOCALID==0)
#define FR_X_LAST (DEVID==NUM_DEVICES-1 && MYLOCALID==NLOCALP-1)
#define FR_OX0 (FR_X_FIRST ? FR_OL : 0)
#define FR_OX1 (FR_X_LAST ? FR_OL : 0)
#define FR_WX0 (FR_X_FIRST ? FR_W : 0)
#define FR_WX1 (FR_X_LAST ? FR_W : 0)
#endif

#if ND==3
    /* Index in the stored frame of cell (k, j, i) (padded local indices), or -1
       if the cell is not stored. Inverse of savebnd3D.cl. */
    LFUNDEF int inject_ind(int k, int j, int i){
        int NZo = NZS-2*FDOH-FR_OT-FR_OL, NYo = NY-2*FDOH-2*FR_OL;
        int NXo = NX-2*FDOH-FR_OX0-FR_OX1;
        int W = FR_W, Wt = FR_WT, W0 = FR_WX0, W1 = FR_WX1;
        int NXm = NXo-W0-W1, NYm = NYo-2*W;
        int A0 = W0*NYo*NZo, A1 = W1*NYo*NZo, B = NXm*W*NZo, C = NXm*NYm*Wt;
        int m=-1;
        i-=FDOH+FR_OX0; j-=FDOH+FR_OL; k-=FDOH+FR_OT;
        if (k<0 || k>NZo-1 || j<0 || j>NYo-1 || i<0 || i>NXo-1)
            m=-1;
        else if (i<W0)
            m=i*NYo*NZo+j*NZo+k;
        else if (i>NXo-1-W1)
            m=A0+(i-NXo+W1)*NYo*NZo+j*NZo+k;
        else if (j<W)
            m=A0+A1+(i-W0)*W*NZo+j*NZo+k;
        else if (j>NYo-1-W)
            m=A0+A1+B+(i-W0)*W*NZo+(j-NYo+W)*NZo+k;
        else if (k<Wt)
            m=A0+A1+2*B+(i-W0)*NYm*Wt+(j-W)*Wt+k;
        else if (k>NZo-1-W)
            m=A0+A1+2*B+C+(i-W0)*NYm*W+(j-W)*W+(k-NZo+W);
        return m;
    }
#else
    /* Index in the stored frame of cell (k, i) (padded local indices), or -1
       if the cell is not stored. Inverse of savebnd2D.cl. */
    LFUNDEF int inject_ind(int k, int i){
        int NZo = NZS-2*FDOH-FR_OT-FR_OL, NXo = NX-2*FDOH-FR_OX0-FR_OX1;
        int W = FR_W, Wt = FR_WT, W0 = FR_WX0, W1 = FR_WX1;
        int A0 = W0*NZo, A1 = W1*NZo, NXm = NXo-W0-W1;
        int m=-1;
        i-=FDOH+FR_OX0; k-=FDOH+FR_OT;
        if (k<0 || k>NZo-1 || i<0 || i>NXo-1)
            m=-1;
        else if (i<W0)
            m=i*NZo+k;
        else if (i>NXo-1-W1)
            m=A0+(i-NXo+W1)*NZo+k;
        else if (k<Wt)
            m=A0+A1+(i-W0)*Wt+k;
        else if (k>NZo-1-W)
            m=A0+A1+NXm*Wt+(i-W0)*W+(k-NZo+W);
        return m;
    }
#endif
