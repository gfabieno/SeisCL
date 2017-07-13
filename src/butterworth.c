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


#include "F.h"


int butterworth(float * data, float fcl, float fch, float dt, int NT, int tmax, int ntrace, int order){
    int state=0;
    int nfft, n,t;
    float *buf=NULL, *H=NULL;
    kiss_fft_cpx * bufout=NULL;
    float freq;
    kiss_fftr_cfg stf=NULL, sti=NULL;
    
    nfft=kiss_fft_next_fast_size(tmax);
    GMALLOC(buf,sizeof(float)*nfft);
    GMALLOC(H,sizeof(float)*(nfft/2+1));
    GMALLOC(bufout,sizeof(kiss_fft_cpx)*nfft);

    if (!(stf = kiss_fftr_alloc( nfft ,0 ,0,0))) {state=1;fprintf(stderr,"Butterworth failed\n");};
    if (!(sti = kiss_fftr_alloc( nfft ,1 ,0,0))) {state=1;fprintf(stderr,"Butterworth failed\n");};

    
    for (t=0;t<nfft/2+1;t++){
        freq=t/(dt*nfft);
        H[t]=1.;
    }
    if (fcl>0){
        for (t=0;t<nfft/2+1;t++){
            freq=t/(dt*nfft);
            H[t]*=1.-sqrt(1./(1.+pow(freq/fcl,2*order)));
        }
    }
    if (fch>0){
        for (t=0;t<nfft/2+1;t++){
            freq=t/(dt*nfft);
            H[t]*=sqrt(1./(1.+pow(freq/fch,2*order)));
        }
    }
    
    for (n=0;n<ntrace;n++){
        for (t=0;t<tmax;t++){
            buf[t]=data[t+n*NT];

        }
        for (t=tmax;t<nfft;t++){
            buf[t]=data[t-tmax+n*NT];
        }
        kiss_fftr( stf , buf, (kiss_fft_cpx*)bufout );
        for (t=0;t<nfft/2+1;t++){
            bufout[t].r*=H[t];
            bufout[t].i*=H[t];
        }
        kiss_fftri( sti , (kiss_fft_cpx*)bufout, buf);
        for (t=0;t<tmax;t++){
            data[t+n*NT]=buf[t]/nfft;
        }
    }
    
    free(buf);
    free(bufout);
    free(H);
    free(stf);
    free(sti);
    kiss_fft_cleanup();
    
    return 0;
}

