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

/* In-memory entry point used by the PyTorch binding (SeisCL/torch/) to
 * drive the engine without HDF5-based model/geometry/data files or MPI.
 * Everything else that binding needs (assign_modeling_case, Init_cst,
 * Init_data, Init_model, Init_CUDA, time_stepping, Free_OpenCL, get_par,
 * get_var) already exists as plain functions declared in F.h and is called
 * directly from bindings.cpp. This file only ports the source/receiver
 * geometry grouping logic of read_srcrec() (src/read_hdf5.c), which has
 * nothing to do with HDF5 beyond the three reads it replaces.
 */

#include "F.h"

int seiscl_set_srcrec(model * m,
                      const float * src_pos, int allns,
                      const float * src,
                      const float * rec_pos, int allng) {

    int state = 0;
    int i, n, p, nsg;
    float thisid;
    float *src_pos0 = NULL, *src0 = NULL, *rec_pos0 = NULL;

    m->src_recs.allns = allns;
    m->src_recs.allng = allng;

    /* Own copies: src_recs.* is freed later by Free_OpenCL (Free_OpenCL.c),
     * so this API cannot alias caller-owned (e.g. numpy/torch) memory. */
    GMALLOC(src_pos0, sizeof(float)*allns*5);
    GMALLOC(src0, sizeof(float)*allns*m->NT);
    GMALLOC(rec_pos0, sizeof(float)*allng*8);
    if (state) return state;
    memcpy(src_pos0, src_pos, sizeof(float)*allns*5);
    memcpy(src0, src, sizeof(float)*allns*m->NT);
    memcpy(rec_pos0, rec_pos, sizeof(float)*allng*8);

    /* Determine the number of shots to simulate. Ported from read_srcrec()
     * (src/read_hdf5.c:307): source/receiver ids (column index 3, 0-based,
     * i.e. Python's src_pos[:,3]/rec_pos[:,3]) must be sorted ascending. */
    m->src_recs.ns = 0;
    thisid = -9999;
    for (i = 0; i < allns; i++) {
        if (thisid < src_pos0[3+i*5]) {
            thisid = src_pos0[3+i*5];
            m->src_recs.ns += 1;
        } else if (thisid > src_pos0[3+i*5]) {
            fprintf(stderr, "Error: Sources ids must be sorted in ascending order\n");
            return 1;
        }
    }

    nsg = 0;
    thisid = -9999;
    for (i = 0; i < allng; i++) {
        if (thisid < rec_pos0[3+i*8]) {
            thisid = rec_pos0[3+i*8];
            nsg += 1;
        } else if (thisid > rec_pos0[3+i*8]) {
            fprintf(stderr, "Error: Src ids in rec_pos must be sorted in ascending order\n");
            return 1;
        }
    }
    if (nsg != m->src_recs.ns) {
        fprintf(stderr, "Error: Number of source ids in src_pos and rec_pos "
                        "are not the same\n");
        return 1;
    }

    GMALLOC(m->src_recs.src_pos, sizeof(float*)*m->src_recs.ns);
    GMALLOC(m->src_recs.src, sizeof(float*)*m->src_recs.ns);
    GMALLOC(m->src_recs.nsrc, sizeof(int)*m->src_recs.ns);
    GMALLOC(m->src_recs.rec_pos, sizeof(float*)*m->src_recs.ns);
    GMALLOC(m->src_recs.nrec, sizeof(int)*m->src_recs.ns);
    if (state) return state;

    // Number of source positions per shot
    thisid = src_pos0[3];
    n = 1;
    p = 0;
    for (i = 1; i < allns; i++) {
        if (thisid == src_pos0[3+i*5]) {
            n += 1;
        } else {
            m->src_recs.nsrc[p] = n;
            n = 1;
            p += 1;
            thisid = src_pos0[3+i*5];
        }
    }
    m->src_recs.nsrc[m->src_recs.ns-1] = n;

    // Number of receiver positions per shot
    thisid = rec_pos0[3];
    n = 1;
    p = 0;
    for (i = 1; i < allng; i++) {
        if (thisid == rec_pos0[3+i*8]) {
            n += 1;
        } else {
            m->src_recs.nrec[p] = n;
            n = 1;
            p += 1;
            thisid = rec_pos0[3+i*8];
        }
    }
    m->src_recs.nrec[m->src_recs.ns-1] = n;

    // Assign the right number of shots and geophones for each shot
    m->src_recs.src_pos[0] = src_pos0;
    m->src_recs.src[0] = src0;
    m->src_recs.rec_pos[0] = rec_pos0;
    for (i = 1; i < m->src_recs.ns; i++) {
        m->src_recs.src_pos[i] = m->src_recs.src_pos[i-1]
                                 + m->src_recs.nsrc[i-1]*5;
        m->src_recs.src[i] = m->src_recs.src[i-1]
                             + m->src_recs.nsrc[i-1]*m->NT;
        m->src_recs.rec_pos[i] = m->src_recs.rec_pos[i-1]
                                 + m->src_recs.nrec[i-1]*8;
    }

    // Maximum number of geophones and shots within a source id
    m->src_recs.nsmax = 0;
    m->src_recs.ngmax = 0;
    for (i = 0; i < m->src_recs.ns; i++) {
        m->src_recs.nsmax = fmax(m->src_recs.nsmax, m->src_recs.nsrc[i]);
        m->src_recs.ngmax = fmax(m->src_recs.ngmax, m->src_recs.nrec[i]);
    }

    return state;
}
