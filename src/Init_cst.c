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
#include "third_party/NVIDIA_FP16/fp16_conversion.h"
#include "third_party/SOFI3D/holbergcoeff.h"

int Init_cst(model * m) {
    
    int state=0;
    int i;
    
    
    __GUARD holbergcoeff(m->FDORDER, m->MAXRELERROR, m->hc);
    if (m->halfpar==2){
        for (i=0;i<7;i++){
            m->hc[i] =half_to_float(float_to_half_full_rtne(m->hc[i]));
        }
    }

    for (i=0;i<m->ncsts;i++){
        if (m->csts[i].transform !=NULL){
            m->csts[i].transform( (void*)m, (void*) m->csts, m->ncsts );
        }
    }
    
    
    return state;

}
