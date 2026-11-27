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

/* The argument-list builders below append "<arg>, " (or "<expr>+") in a loop and
 * then need to drop the dangling separator. This trims the last n bytes of s in
 * place, guarded so an unexpectedly empty string cannot underflow the buffer.
 * Replaces a "char *p=temp; while(*p) p++; p[-n]='\0';" idiom whose walk-back
 * GCC's -Wstringop-overflow could not prove in-bounds. */
static void chop_tail(char * s, size_t n){
    size_t len = strlen(s);
    if (n <= len)
        s[len - n] = '\0';
}

/* vx, vy, vz carry the FP16 scale par_scale (set_par_scale_value); the
   generated kernels take it as their argument par_scale, not as a constant. */
static int vel_scaled(const char * name){
    return strcmp(name, "vx")==0 || strcmp(name, "vy")==0 || strcmp(name, "vz")==0;
}

int kernel_varout(device * dev,
                  clprogram * prog){
 
    int state=0;
    int i,j,k;
    int scaler=0;
    
    char temp[MAX_KERN_STR]={0};;
    char temp2[1000]={0};   /* one generated line; 100 overflowed at FP16 2 */
    variable * vars = dev->vars;
    variable * tvars = dev->trans_vars;
    
    strcat(temp, FUNDEF"void varsout"
                 "(int nt, int nrec,"GLOBARG"float * rec_pos, int src_scale, int par_scale,");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            if (dev->FP16<=1){
                strcat(temp, GLOBARG"float * ");
            }
            else{
                strcat(temp, GLOBARG"half * ");
            }
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, GLOBARG"float * ");
            strcat(temp, vars[i].name);
            strcat(temp, "out, ");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            for (j=0;j<tvars[i].n2ave;j++){
                /* Check if the variable has already been included in arg list */
                unsigned included = 0;
                for(k=0; k<dev->nvars; k++){ // Simple but ugly
                    if(strcmp(tvars[i].var2ave[j], vars[k].name) == 0 && vars[k].to_output)
                        included = 1;
                }
                if(!included){
                    if (dev->FP16<=1){
                        strcat(temp, GLOBARG"float * ");
                    }
                    else{
                        strcat(temp, GLOBARG"half * ");
                    }
                    strcat(temp, tvars[i].var2ave[j]);
                    strcat(temp, ", ");
                }
            }

            strcat(temp, GLOBARG"float * ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out, ");
        }
    }
    chop_tail(temp, 2);
    strcat(temp, "){\n\n");

    //This only supports 3 dimensions (need for more ?)
    #ifdef __SEISCL__
    strcat(temp,"    int gid = get_global_id(0);\n");
    #else
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
    #endif
    sprintf(temp2,"    if (gid > nrec-1){\n" );
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    };\n\n");
    
    strcat(temp,"    int i=(int)(rec_pos[0+8*gid]/DH)+FDOH;\n"
           "    int j=(int)(rec_pos[1+8*gid]/DH)+FDOH;\n"
           "    int k=(int)(rec_pos[2+8*gid]/DH)+FDOH;\n"
           "\n");
    
    sprintf(temp2,"    if (i-OFFSET<FDOH || i-OFFSET>N%s-FDOH-1){\n",
            dev->N_names[dev->NDIM-1] );
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    };\n\n");

    
    char posstr[100]={0};
    if (dev->FP16==0){
        if (dev->NDIM==2){
            sprintf(posstr,"[(i-OFFSET)*N%s+k]",dev->N_names[0]);
        }
        else if (dev->NDIM==3){
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s+j*(N%s)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
    }
    else{
        if (dev->NDIM==2){
            sprintf(posstr,"[(i-OFFSET)*N%s*2+k]",dev->N_names[0]);
        }
        else if (dev->NDIM==3){
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s*2+j*(N%s*2)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
    }
    
    /* Stress-type outputs (the stresses and the pressure) are recorded at the
       state BEFORE this step's stress update: varsout runs after update_s, so
       the value written now is stored at index nt+1 (index 0 stays at its
       initial value, 0). The adjoint then injects stress residual nt at
       adjoint iteration nt, like the velocity residuals, and every sample is
       used. Before 2026-11-18 the stress at the end of step nt was stored at
       nt and its residual delayed by one sample (res_scale), which dropped the
       last pressure sample from the adjoint: an O(dt) gradient error whenever
       the record ends before the waves have died out. In time, the new
       pressure trace sits half a sample after the velocity trace, the old one
       half a sample before. */
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            int isstress = (vars[i].name[0]=='s' && strlen(vars[i].name)==3);
            strcat(temp, "    ");
            if (isstress) strcat(temp, "if (nt+1<NT) ");
            strcat(temp, vars[i].name);
            strcat(temp, isstress ? "out[NT*gid+nt+1]=" : "out[NT*gid+nt]=");
            if (dev->FP16>0){
                #ifdef __SEISCL__
                strcat(temp, "ldexp(");
                #else
                strcat(temp, "scalbnf(");
                if (dev->FP16>1){
                    strcat(temp, "__half2float(");
                }
                #endif

            }
            strcat(temp, vars[i].name);
            strcat(temp, posstr);
            if (dev->FP16>0){
                if (dev->FP16>1){
                    strcat(temp, ")");
                }
                strcat(temp,", -src_scale");
                if (vel_scaled(vars[i].name)){
                    strcat(temp, "+ par_scale");
                }
                strcat(temp, ")");
            }
            strcat(temp, ";\n");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            /* the pressure is a stress-type output: see above */
            strcat(temp, "    if (nt+1<NT) ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out[NT*gid+nt+1]=(");
            for (j=0;j<tvars->n2ave;j++){
                if (dev->FP16>0){
                    scaler = vel_scaled(tvars[i].var2ave[j]);
                    #ifdef __SEISCL__
                    strcat(temp, "ldexp(");
                    #else
                    strcat(temp, "scalbnf(");
                    if (dev->FP16>1){
                        strcat(temp, "__half2float(");
                    }
                    #endif
                }
                strcat(temp, tvars[i].var2ave[j]);
                strcat(temp, posstr);
                if (dev->FP16>0){
                    if (dev->FP16>1){
                        strcat(temp, ")");
                    }
                    strcat(temp,", -src_scale");
                    if (scaler){
                        strcat(temp, "+ par_scale");
                    }
                    strcat(temp, ")");
                }
                strcat(temp, "+");
            }
            chop_tail(temp, 1);
            strcat(temp, ")/");
            sprintf(temp2,"%f",(float)tvars->n2ave);
            strcat(temp, temp2);
            strcat(temp, ";\n");
        }
    }

    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
       __GUARD prog_source(prog, "varsout", temp, 0, NULL);
    
    prog->wdim=1;
    
    return state;

}


int kernel_varoutinit(device * dev,
                      clprogram * prog){
    
    int state=0;
    int i;
    variable * vars = dev->vars;
    variable * tvars = dev->trans_vars;
    char temp[MAX_KERN_STR]={0};
    
    
    strcat(temp, FUNDEF"void varsoutinit(int nrec,");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            strcat(temp, GLOBARG"float * ");
            strcat(temp, vars[i].name);
            strcat(temp, "out, ");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            strcat(temp, GLOBARG"float * ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out, ");
        }
    }
    chop_tail(temp, 2);
    strcat(temp, "){\n\n");
    
    #ifdef __SEISCL__
    strcat(temp,"    int gid = get_global_id(0);\n\n");
    #else
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n\n");
    #endif
    
    strcat(temp,"    if (gid > nrec*NT-1){\n"
                "        return;\n"
                "    };\n\n");
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "    ");
            strcat(temp, vars[i].name);
            strcat(temp, "out[gid]=0;\n");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            strcat(temp, "    ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out[gid]=0;\n");
        }
    }

    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
       __GUARD prog_source(prog, "varsoutinit", temp, 0, NULL);
    
    prog->wdim=1;
    
    return state;
    
}

int kernel_varinit(device * dev,
                   model * m,
                   variable * vars,
                   clprogram * prog,
                   int BACK_PROP_TYPE){
    
    int state=0;
    int i;
    
    char temp[MAX_KERN_STR]={0};;
    
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize){
            maxsize=vars[i].num_ele;
        }
    }
    
    
    strcat(temp, FUNDEF"void vars_init(");
    for (i=0;i<dev->nvars;i++){
        if (!var_is_half(m, vars[i].name)){
            strcat(temp, GLOBARG"float * ");
        }
        else{
            strcat(temp, GLOBARG"half * ");
        }
        strcat(temp, vars[i].name);
        if (BACK_PROP_TYPE==1){
            strcat(temp, "r");
        }
        strcat(temp, ", ");
    }
    chop_tail(temp, 2);
    strcat(temp, "){\n\n");
    
    /* Flat 1-D launch over the LARGEST buffer, in both backends (as
     * initsavefreqs). The memory variables hold L grids (size_varmem:
     * num_ele = prod(N+FDORDER)*L), but the launch used to cover one grid:
     * OpenCL exactly prod(N+FDORDER), CUDA a 2-D/3-D launch whose fastest
     * extent is rounded up to BLOCK_SIZE (about 2.5 grids in 2-D tests). The
     * slices l >= 1 (OpenCL) or l >= 2 (CUDA, partly) were never zeroed: the
     * back_prop_type 2 adjoint, which reuses the forward buffers, started from
     * the forward's last memory variables (2-D OpenCL L=2 gradient 1e13 too
     * large), and later shots started from the previous shot's. */
    #ifdef __SEISCL__
    strcat(temp,"    int gid = get_global_id(0);\n\n");
    #else
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n\n");
    #endif
    
    for (i=0;i<dev->nvars;i++){
        sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
        strcat(temp,ptemp);
        strcat(temp, "    ");
        strcat(temp, "    ");
        strcat(temp, vars[i].name);
        if (BACK_PROP_TYPE==1){
            strcat(temp, "r");
        }
        if (!var_is_half(m, vars[i].name)){
            strcat(temp, "[gid]=0;\n");
        }
        else{
            strcat(temp, "[gid]= __float2half(0);\n");
        }
        
    }
    
    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
       __GUARD prog_source(prog, "vars_init", temp, 0, NULL);
    
    prog->gsize[0] = maxsize;
    prog->wdim = 1;
    
    return state;
    
}

int kernel_sources(model * m, 
                   device * dev,
                   clprogram * prog){
    
    int state=0;
    int i,j;
    variable * vars = dev->vars;
    variable * tvars = dev->trans_vars;
    const char * src_names[10] = {"vx", "vy", "vz", "p",
                                  "sxx", "syy", "szz",
                                  "sxz", "sxy", "syz"};
    int src_codes[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    char temp[MAX_KERN_STR]={0};
    char temp2[1000]={0};   /* one generated line; 100 overflowed at FP16 2 */
//    
    
    
    
    int * tosources=NULL;
    int * tosources2=NULL;
    int ntypes=0;
    int ind;
    GMALLOC(tosources,dev->nvars*sizeof(int));
    GMALLOC(tosources2,dev->ntvars*sizeof(int));
    for (i=0;i<dev->src_recs.allns;i++){
        ind =dev->src_recs.src_pos[0][4+i*5];
        if (ind<10 && ind>-1){
            for (j=0;j<dev->nvars;j++){
                if (strcmp(vars[j].name,src_names[ind])==0){
                    tosources[j]=1;
                    src_codes[j]=ind;
                }
            }
        }
        if (ind<10 && ind>-1){
            for (j=0;j<dev->ntvars;j++){
                if (strcmp(tvars[j].name,src_names[ind])==0){
                    tosources2[j]=1;
                }
            }
        }
        // backward compatibility
        if (ind==100){
            tosources2[0]=1;
        }

    }

    for (i=0;i<dev->nvars;i++){
        if (tosources[i]==1){
            ntypes++;
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tosources2[i]==1){
            ntypes++;
        }
    }
    
    if (ntypes==0){
        state=1;
        fprintf(stderr,"Error: No sources for variable list found\n");
    }
    
    strcat(temp, FUNDEF"void sources(int nt, int nsrc,"
                 "int src_scale, int par_scale,"GLOBARG"float * src_pos,"GLOBARG"float * src, int pdir, ");
    for (i=0;i<dev->nvars;i++){
        if (tosources[i]){
            if (dev->FP16<=1){
                strcat(temp, GLOBARG"float * ");
                
            }
            else{
                strcat(temp, GLOBARG"half * ");
            }
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tosources2[i]){
            for (j=0;j<tvars[i].n2ave;j++){
                if (dev->FP16<=1){
                    strcat(temp, GLOBARG"float * ");
                }
                else{
                    strcat(temp, GLOBARG"half * ");
                }
                strcat(temp, tvars[i].var2ave[j]);
                strcat(temp, ", ");
            }
        }
    }

    chop_tail(temp, 2);
    strcat(temp, "){\n\n");
    
    //This only supports 3 dimensions (need for more ?)
    #ifdef __SEISCL__
    strcat(temp,"    int gid = get_global_id(0);\n");
    #else
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
    sprintf(temp2,"    if (gid > nsrc -1){\n");
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    }\n\n");
    #endif
    strcat(temp,"    int i=(int)(src_pos[0+5*gid]/DH)+FDOH;\n"
           "    int j=(int)(src_pos[1+5*gid]/DH)+FDOH;\n"
           "    int k=(int)(src_pos[2+5*gid]/DH)+FDOH;\n"
           "\n");

    /* Inject in the own cells and in the x halo of an inner cut, which holds
       the neighbour's copy of the cell: the halo is refreshed only after the
       next update, so a source within FDOH of a cut would otherwise be
       missing from the neighbour's stencil for one update (domain
       decomposition). The outer padding never holds a source. */
    sprintf(temp2,"    if (i-OFFSET<0 || i-OFFSET>N%s-1){\n",
            dev->N_names[dev->NDIM-1] );
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    }\n\n");
    if (m->FREESURF==1){
        sprintf(temp2,"    if (pdir==-1 && k<2*FDOH){\n" );
        strcat(temp, temp2);
        strcat(temp,
               "        return;\n"
               "    }\n\n");
    }
   
    
    strcat(temp,"    int source_type= src_pos[4+5*gid];\n");
    if (dev->FP16==0){
        strcat(temp,"    float amp=(float)pdir*(DT*src[gid*NT+nt]);\n\n");
    }
    else{
        #ifdef __SEISCL__
        strcat(temp,"    float amp=ldexp((float)pdir*(DT*src[gid*NT+nt]), src_scale);\n\n");
        #else
        strcat(temp,"    float amp=scalbnf((float)pdir*(DT*src[gid*NT+nt]), src_scale);\n\n");
        #endif
    }
    
    char posstr[100]={0};

    
    if (dev->NDIM==2){
        if (dev->FP16==0){
            sprintf(posstr,"[(i-OFFSET)*N%s+k]",dev->N_names[0]);
        }
        else{
            sprintf(posstr,"[(i-OFFSET)*N%s*2+k]",dev->N_names[0]);
        }

    }
    else if (dev->NDIM==3){
        if (dev->FP16==0){
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s+j*(N%s)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
        else{
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s*2+j*(N%s*2)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
    }
    else{
        state=1;
        fprintf(stderr,"Error: Sources for a number of dimensions higher "
                       "than 3 are not supported yet\n");
    }
    
    
    /* Each variable is stored in its OWN scaled units at FP16>0:
     * set_par_scale() gives vx/vy/vz a scaler equal to par_scale and leaves the
     * stresses at 0, and kernel_varout() recovers a stored value as
     * ldexp(var, -src_scale + var->scaler) -- i.e.
     *     stored = physical * 2^(src_scale - scaler).
     * `amp` above carries 2^src_scale only, which is right for a stress
     * (scaler==0) but too large by 2^scaler for a velocity. Injecting it raw
     * therefore left FORCE sources mis-scaled by 2^par_scale at FP16>0 (~2^33
     * for a typical model) while pressure sources were fine -- the gradient
     * and the seismograms both came back on the wrong scale, and no amount of
     * unscaling downstream could recover it because the forward run itself had
     * the wrong source. Undo the variable's scaler here. */
    for (i=0;i<dev->nvars;i++){
        
        if (tosources[i]){
            char ampstr[64];
            if (dev->FP16>0 && vel_scaled(vars[i].name)){
                #ifdef __SEISCL__
                sprintf(ampstr, "ldexp(amp, -par_scale)");
                #else
                sprintf(ampstr, "scalbnf(amp, -par_scale)");
                #endif
            }
            else{
                sprintf(ampstr, "amp");
            }
            if (ntypes>1){
                sprintf(temp2,"    if (source_type==%d)\n", src_codes[i]);
                strcat(temp, temp2);
                strcat(temp, "    ");
            }
            strcat(temp, "    ");
            strcat(temp, vars[i].name);
            strcat(temp, posstr);
            if (dev->FP16<=1){
                strcat(temp, "+=");
                strcat(temp, ampstr);
                strcat(temp, ";\n");
            }
            else{
                strcat(temp, "=__float2half(__half2float(");
                strcat(temp, vars[i].name);
                strcat(temp, posstr);
                strcat(temp, ")+");
                strcat(temp, ampstr);
                strcat(temp, ");\n");
            }

            
        }
    }
    for (i=0;i<dev->ntvars;i++){
        
        if (tosources2[i]){
            if (ntypes>1){
                sprintf(temp2,"    if (source_type==%d){\n", i+100);
                strcat(temp, temp2);
                strcat(temp, "    ");
            }
            for (j=0;j<tvars[i].n2ave;j++){
                strcat(temp, "    ");
                strcat(temp, tvars[i].var2ave[j]);
                strcat(temp, posstr);
                if (dev->FP16<=1){
                    sprintf(temp2,"+=amp/%f;\n", (float)tvars[i].n2ave);
                    strcat(temp, temp2);
                }
                else{
                    strcat(temp, "=__float2half(__half2float(");
                    strcat(temp, tvars[i].var2ave[j]);
                    strcat(temp, posstr);
                    sprintf(temp2,")+amp/%f);\n", (float)tvars[i].n2ave);
                    strcat(temp, temp2);
                }
            }
            if (ntypes>1){
                strcat(temp, "    }");
            }
        }
    }
    
    strcat(temp, "\n}");
    
       __GUARD prog_source(prog, "sources", temp, 0, NULL);
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    free(tosources);
    free(tosources2);
    
    prog->wdim=1;
    
    return state;
    
}

int kernel_residuals(device * dev,
                     clprogram * prog,
                     int BACK_PROP_TYPE){
    
    int state=0;
    int i,j;
    
    char temp[MAX_KERN_STR]={0};
    char temp2[1000]={0};   /* one generated line; 100 overflowed at FP16 2 */
    variable * vars = dev->vars;
    variable * tvars = dev->trans_vars;
    
   
    
    strcat(temp, FUNDEF"void residuals(int nt, int nrec,"
                 GLOBARG"float * rec_pos, int res_scale, ");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            if (dev->FP16<=1){
                strcat(temp, GLOBARG"float * ");
                
            }
            else{
                strcat(temp, GLOBARG"half * ");
            }
            strcat(temp, vars[i].name);
            if (BACK_PROP_TYPE==1)
                strcat(temp, "r");
            strcat(temp, ", ");
            strcat(temp, GLOBARG"float * ");
            strcat(temp, vars[i].name);
            strcat(temp, "out, ");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            for (j=0;j<tvars[i].n2ave;j++){
                /* Skip a variable that is already a parameter (e.g. sxx and
                   szz when both the stresses and the pressure are output):
                   a duplicate parameter name does not compile. */
                char token[100];
                snprintf(token, sizeof(token), " %s%s, ", tvars[i].var2ave[j],
                         BACK_PROP_TYPE==1 ? "r" : "");
                if (strstr(temp, token)) continue;
                if (dev->FP16<=1){
                    strcat(temp, GLOBARG"float * ");
                    
                }
                else{
                    strcat(temp, GLOBARG"half * ");
                }
                strcat(temp, tvars[i].var2ave[j]);
                if (BACK_PROP_TYPE==1)
                    strcat(temp, "r");
                strcat(temp, ", ");
            }
            
            strcat(temp, GLOBARG"float * ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out, ");
        }
    }
    /* Viscoelastic model with pressure output: the adjoint memory variables
       get a source too (see the end of this function). */
    int pmem = 0;
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output && strcmp(tvars[i].name, "p")==0) pmem = 1;
    }
    {
        int hasr = 0;
        for (i=0;i<dev->nvars;i++){
            if (strcmp(vars[i].name, "rxx")==0) hasr = 1;
        }
        pmem = pmem && hasr;
    }
    if (pmem){
        const char * pt = (dev->FP16>1) ? GLOBARG"half * " : GLOBARG"float * ";
        const char * rs = (BACK_PROP_TYPE==1) ? "r" : "";
        sprintf(temp2, "%sM, %smu, %staup, %staus, "GLOBARG"float * eta, ",
                pt, pt, pt, pt);
        strcat(temp, temp2);
        sprintf(temp2, "%srxx%s, %srzz%s, ", pt, rs, pt, rs);
        strcat(temp, temp2);
        if (dev->NDIM==3){
            sprintf(temp2, "%sryy%s, ", pt, rs);
            strcat(temp, temp2);
        }
    }
    chop_tail(temp, 2);
    strcat(temp, "){\n\n");
    
    #ifdef __SEISCL__
    strcat(temp,"    int gid = get_global_id(0);\n");
    #else
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
    #endif
    sprintf(temp2,"    if (gid > nrec-1){\n");
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    };\n\n");
    
     strcat(temp,"    int i=(int)(rec_pos[0+8*gid]/DH)+FDOH;\n"
           "    int j=(int)(rec_pos[1+8*gid]/DH)+FDOH;\n"
           "    int k=(int)(rec_pos[2+8*gid]/DH)+FDOH;\n\n");
    
    /* Own cells and the x halo of an inner cut (see kernel_sources). */
    sprintf(temp2,"    if (i-OFFSET<0 || i-OFFSET>N%s-1){\n",
            dev->N_names[dev->NDIM-1] );
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    };\n\n");
    
    char posstr[100]={0};

    if (dev->NDIM==2){
        if (dev->FP16==0){
            sprintf(posstr,"[(i-OFFSET)*N%s+k]",dev->N_names[0]);
        }
        else
        {
            sprintf(posstr,"[(i-OFFSET)*N%s*2+k]",dev->N_names[0]);
        }
    }
    else if (dev->NDIM==3){
        if (dev->FP16==0){
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s+j*(N%s)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
        else
        {
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s*2+j*(N%s*2)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
    }

    
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "        ");
            strcat(temp, vars[i].name);
            if (BACK_PROP_TYPE==1)
            strcat(temp, "r");
            strcat(temp, posstr);
            if (dev->FP16==0){
                strcat(temp, "+=");
            }
            else{
                #ifdef __SEISCL__
                strcat(temp, "+=ldexp(");
                #else
                if (dev->FP16==1){
                    strcat(temp, "+=scalbnf(");
                }
                else{
                    strcat(temp, "=__float2half(__half2float(");
                    strcat(temp, vars[i].name);
                    if (BACK_PROP_TYPE==1)
                    strcat(temp, "r");
                    strcat(temp, posstr);
                    strcat(temp, ")+scalbnf(");
                }
                #endif
                
            }
            strcat(temp, vars[i].name);
            strcat(temp, "out[NT*gid+nt]");
            if (dev->FP16==0){
                strcat(temp, ";\n");
            }
            else if (dev->FP16==1){
                strcat(temp, ",res_scale);\n");
            }
            else{
                strcat(temp, ",res_scale));\n");
            }
            
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            for (j=0;j<tvars[i].n2ave;j++){
                strcat(temp, "        ");
                strcat(temp, tvars[i].var2ave[j]);
                if (BACK_PROP_TYPE==1)
                strcat(temp, "r");
                strcat(temp, posstr);
                if (dev->FP16==0){
                    strcat(temp, "+=");
                }
                else{
                    #ifdef __SEISCL__
                    strcat(temp, "+=ldexp(");
                    #else
                    if (dev->FP16==1){
                        strcat(temp, "+=scalbnf(");
                    }
                    else{
                        strcat(temp, "=__float2half(__half2float(");
                        strcat(temp, tvars[i].var2ave[j]);
                        if (BACK_PROP_TYPE==1)
                            strcat(temp, "r");
                        strcat(temp, posstr);
                        strcat(temp, ")+scalbnf(");
                    }
                    #endif
                }
                strcat(temp, tvars[i].name);
                strcat(temp, "out[NT*gid+nt]/");
                sprintf(temp2,"%d",tvars[i].n2ave);
                strcat(temp, temp2);
                if (dev->FP16==0){
                    strcat(temp, ";\n");
                }
                else if (dev->FP16==1){
                    strcat(temp, ",res_scale);\n");
                }
                else{
                    strcat(temp, ",res_scale));\n");
                }
            }
        }
    }
    
    /* Adjoint memory-variable source for pressure data (viscoelastic).
       Pressure depends on sigma = sigma' + sum_l R^l, so dX/dR^l = dX/dsigma
       and the adjoint source G dX/dphi (thesis eq. 3.29, G = blockdiag(1/rho,
       Psi0, Psi1..PsiL)) also drives each adjoint memory variable: in
       traditional variables r~ = -dt R~,
           dt r~_l + (1/tau_l) Psi_l D v~ + r~_l/tau_l = (1/tau_l) Psi_l dX/dsigma.
       The value s added above to each normal adjoint stress acts as an
       isotropic strain increment theta = ND s / P; through the memory update
       of update_s (r = b (c r - eta (e theta - d ...))) it adds -b eta Q/P s
       to every normal memory variable, with the moduli of update_s:
       P = ND g - (ND-1) f, Q = ND e - (ND-1) d. Without it the gradient with
       pressure receivers was off by a common factor growing with tau
       (notes viscoelastic-gradient-theory.md, 2026-11-17). */
    if (pmem){
        const char * rs = (BACK_PROP_TYPE==1) ? "r" : "";
        int div = (dev->FP16>0) ? 2 : 1;
        char nz[40], ny[40], nx[40], stride[200], ip[300], ld[100], st[200];
        sprintf(nz, "(N%s*%d)", dev->N_names[0], div);
        sprintf(nx, "N%s", dev->N_names[dev->NDIM-1]);
        if (dev->NDIM==3){
            sprintf(ny, "N%s", dev->N_names[1]);
            sprintf(stride, "%s*%s*%s", nx, ny, nz);
            sprintf(ip, "(((i-OFFSET-FDOH)*(%s-2*FDOH)+(j-FDOH))*(%s-2*FDOH)+(k-FDOH))", ny, nz);
        }
        else {
            sprintf(stride, "%s*%s", nx, nz);
            sprintf(ip, "((i-OFFSET-FDOH)*(%s-2*FDOH)+(k-FDOH))", nz);
        }
        if (dev->FP16>1){
            strcpy(ld, "__half2float");
        }
        else {
            strcpy(ld, "");
        }
        /* Own cells only: the parameters (M, mu, taup, taus) have no x halo,
           so ip would be out of range in the halo of an inner cut, and a halo
           memory variable is never read by the neighbour's stencil. The halo
           copy of the stress keeps the injection above but not the O(dt)
           correction below (domain decomposition). */
        sprintf(temp2, "#if LVE>0\n    if (i-OFFSET>=FDOH && i-OFFSET<=N%s-FDOH-1){\n",
                dev->N_names[dev->NDIM-1]);
        strcat(temp, temp2);
        sprintf(temp2, "        int ip = %s;\n", ip);
        strcat(temp, temp2);
        sprintf(temp2, "        float lM=%s(M[ip]), lmu=%s(mu[ip]), ltp=%s(taup[ip]), lts=%s(taus[ip]);\n", ld, ld, ld, ld);
        strcat(temp, temp2);
        strcat(temp,
               "        float g = lM*(1.0f+LVE*ltp), f = 2.0f*lmu*(1.0f+LVE*lts);\n"
               "        float e = lM*ltp/DT, d = 2.0f*lmu*lts/DT;\n"
               "        float P = ND*g - (ND-1)*f, Q = ND*e - (ND-1)*d;\n");
        /* s: the value added to each normal adjoint stress above */
        for (i=0;i<dev->ntvars;i++){
            if (tvars[i].to_output && strcmp(tvars[i].name, "p")==0){
                if (dev->FP16==0){
                    sprintf(temp2, "        float s = pout[NT*gid+nt]/%d;\n", tvars[i].n2ave);
                }
                else {
                    #ifdef __SEISCL__
                    sprintf(temp2, "        float s = ldexp(pout[NT*gid+nt]/%d, res_scale);\n", tvars[i].n2ave);
                    #else
                    sprintf(temp2, "        float s = scalbnf(pout[NT*gid+nt]/%d, res_scale);\n", tvars[i].n2ave);
                    #endif
                }
                strcat(temp, temp2);
            }
        }
        sprintf(temp2, "        int im = %s;\n", posstr+1);   /* posstr without its brackets */
        temp2[strlen(temp2)-3] = ';'; temp2[strlen(temp2)-2] = '\n'; temp2[strlen(temp2)-1] = '\0';
        strcat(temp, temp2);
        /* Timing: this runs before update_adjs, which advances each memory
           variable as r = b (c r_old - ...) and adds DT2 (sum r_old + sum r)
           to the stresses. To leave exactly -b eta Q/P s in r after that
           update, inject x = -eta Q/P s / c; the stresses then also receive
           DT2 sum_l x from the old memory, which the true source does not
           produce: it is removed here. Without these two corrections the
           pressure-receiver gradient was off by O(dt) (1.4 % at dt = 1 ms). */
        strcat(temp,
               "        if (P!=0.0f){\n"
               "            float sx = 0.0f;\n"
               "            for (int l=0; l<LVE; l++){\n"
               "                float c = 1.0f - eta[l]*0.5f;\n"
               "                float a = -eta[l]*Q/P*s/c;\n"
               "                sx += a;\n"
               "                int ir = l*");
        strcat(temp, stride);
        strcat(temp, " + im;\n");
        {
            const char * names[3] = {"rxx", "rzz", "ryy"};
            int nn = (dev->NDIM==3) ? 3 : 2;
            int q;
            for (q=0;q<nn;q++){
                if (dev->FP16>1){
                    sprintf(st, "                %s%s[ir] = __float2half(__half2float(%s%s[ir]) + a);\n",
                            names[q], rs, names[q], rs);
                }
                else {
                    sprintf(st, "                %s%s[ir] += a;\n", names[q], rs);
                }
                strcat(temp, st);
            }
        }
        strcat(temp, "            }\n");
        {
            const char * names[3] = {"sxx", "szz", "syy"};
            int nn = (dev->NDIM==3) ? 3 : 2;
            int q;
            for (q=0;q<nn;q++){
                if (dev->FP16>1){
                    sprintf(st, "            %s%s[im] = __float2half(__half2float(%s%s[im]) - DT2*sx);\n",
                            names[q], rs, names[q], rs);
                }
                else {
                    sprintf(st, "            %s%s[im] -= DT2*sx;\n", names[q], rs);
                }
                strcat(temp, st);
            }
        }
        strcat(temp, "        }\n    }\n#endif\n");
    }

    strcat(temp, "\n}");
    
    __GUARD prog_source(prog, "residuals", temp, 0, NULL);

//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    prog->wdim=1;
    
    return state;
    
}



int kernel_gradinit(device * dev,
                    parameter * pars,
                    clprogram * prog){
    
    int state=0;
    int i;
    
    char temp[MAX_KERN_STR]={0};
    char temp2[1000]={0};   /* one generated line; 100 overflowed at FP16 2 */

    strcat(temp, FUNDEF"void gradinit(");
    for (i=0;i<dev->npars;i++){
        if (pars[i].to_grad){
            strcat(temp, GLOBARG"float * grad");
            strcat(temp, pars[i].name);
            strcat(temp, ", ");
            if (pars[i].cl_H.host){
                strcat(temp, GLOBARG"float * H");
                strcat(temp, pars[i].name);
                strcat(temp, ", ");
            }
        }
    }
    chop_tail(temp, 2);
    strcat(temp, "){\n\n");
    
    #ifdef __SEISCL__
    strcat(temp,"    int gid = get_global_id(0);\n\n");
    #else
    /* One flat index over the prod(N) cells, launched 1-D (below), as the
       OpenCL branch does. The 3-D version used blockIdx.y for gidx (so
       gidx == gidy): only a sparse subset of the gradient and Hessian cells
       was zeroed, and the others kept whatever the memory held -- the
       previous call's gradient on a reused buffer (PyTorch engine), which
       was then added to the new one. */
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
    if (dev->npars>0){
        sprintf(temp2,"    if (gid>%d-1){\n", pars[0].num_ele);
        strcat(temp,temp2);
        strcat(temp,  "        return;\n"
                      "    };\n\n");
    }
    #endif
    
    for (i=0;i<dev->npars;i++){
        if (pars[i].to_grad){
            strcat(temp, "    grad");
            strcat(temp, pars[i].name);
            strcat(temp, "[gid]=0;\n");
            if (pars[i].cl_H.host){
                strcat(temp, "    H");
                strcat(temp, pars[i].name);
                strcat(temp, "[gid]=0;\n");
            }
        }
    }

    strcat(temp, "\n}");
    
       __GUARD prog_source(prog, "gradinit", temp, 0, NULL);
    #ifdef __SEISCL__
    prog->gsize[0]=1;
    for (i=0;i<dev->NDIM;i++){
        prog->gsize[0]*=dev->N[i];
    }
    prog->wdim=1;
    #else
    prog->gsize[0]=1;
    for (i=0;i<dev->NDIM;i++){
        prog->gsize[0]*=dev->N[i];
    }
    prog->wdim=1;
    #endif
    
//        printf("%s\n\n%lu\n",temp, strlen(temp));
    
    return state;
    
}

int kernel_initsavefreqs(device * dev,
                         variable * vars,
                         clprogram * prog){
    
    int state=0;
    int i;
    
    char temp[MAX_KERN_STR]={0};;
    
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize && vars[i].for_grad){
            maxsize=vars[i].num_ele;
        }
    }
    
    strcat(temp, FUNDEF"void initsavefreqs(");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            strcat(temp, GLOBARG"float2 * f");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
        }
    }
    chop_tail(temp, 2);
    strcat(temp, "){\n\n");


    /* Flat 1-D launch in both backends. This kernel is a pure elementwise map
     * over the padded buffer, so any bijection of [0, num_ele) is correct and a
     * 1-D index is the natural one. The former CUDA path built gid from a
     * multi-dimensional launch whose strides were the block-rounded extents
     * rather than the padded ones; in 3D it also read
     * "gidx = blockIdx.y*blockDim.z" (blockIdx.z was meant), and since
     * blockDim.y == blockDim.z == 1 that made gidx == gidy, so only a diagonal
     * of the buffer was ever touched. A flat launch removes both problems and
     * stops over-launching (the 2-D form dispatched ~5x more threads than there
     * are elements for a short, wide grid). */
    #ifdef __SEISCL__
        strcat(temp,"    int gid = get_global_id(0);\n");
        strcat(temp,"    float2 f20 = 0.0;\n");
    #else
        strcat(temp,"    float2 f20 = make_float2(0.0, 0.0);\n");
        strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
    #endif
    
    
    /* Zero every frequency slice, not just slice 0. The buffer is
     * num_ele*NFREQS float2 and clbuf_create does not zero-initialize
     * (clbuf.c:202-220), so without the freq loop slices 1..NFREQS-1 started
     * from uninitialized device memory on the first shot and then accumulated
     * across shots, because this kernel is also what resets the buffers between
     * sources (time_stepping.c:1048-1049). */
    strcat(temp,"    int freq;\n");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            sprintf(ptemp,"    if (gid<%d){\n", vars[i].num_ele);
            strcat(temp,ptemp);
            strcat(temp, "        for (freq=0;freq<NFREQS;freq++){\n");
            strcat(temp, "            f");
            strcat(temp, vars[i].name);
            sprintf(ptemp, "[gid+freq*%d]=f20;\n", vars[i].num_ele);
            strcat(temp,ptemp);
            strcat(temp, "        }\n");
            strcat(temp, "    }\n");
        }
    }

    strcat(temp, "\n}");

       __GUARD prog_source(prog, "initsavefreqs", temp, 0, NULL);
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    #ifdef __SEISCL__
    /* Launch one work-item per element of the *padded* buffer. prod(dev->N) is
     * the unpadded count and is smaller than num_ele = prod(N+FDORDER), so it
     * left a tail of each buffer untouched while the in-kernel guard uses
     * the padded count. */
    prog->gsize[0] = maxsize;
    #else
    /* Same flat 1-D launch as OpenCL: prog_launch picks tsize={BLOCK_SIZE,1,1}
     * and bsize=ceil(gsize/tsize), which covers [0, maxsize) exactly. */
    prog->gsize[0] = maxsize;
    #endif
    prog->wdim=1;

    return state;

}

int kernel_savefreqs(device * dev,
                     variable * vars,
                     clprogram * prog){
    
    int state=0;
    int i;
    
    char temp[MAX_KERN_STR]={0};;
    
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize && vars[i].for_grad){
            maxsize=vars[i].num_ele;
        }
    }
    
    /* cospi/sinpi are overloaded in OpenCL C but are double-only in CUDA,
     * where the float versions are cospif/sinpif. */
    strcat(temp,
        "#ifdef __OPENCL_VERSION__\n"
        "    #define COSPI(x) cospi(x)\n"
        "    #define SINPI(x) sinpi(x)\n"
        "#else\n"
        "    #define COSPI(x) cospif(x)\n"
        "    #define SINPI(x) sinpif(x)\n"
        "#endif\n\n");
    strcat(temp, FUNDEF"void savefreqs("GLOBARG"float * gradfreqsn, int nt, ");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            /* FP16>1 stores the wavefield as half; <=1 keeps one float per
               scalar element (FP16==1 is float2-vectorized, same bytes).
               Mirrors kernel_varout's argument typing above. The spectra
               stay float2 either way -- Init_OpenCL.c sizes cl_fvar from
               num_ele regardless of FP16 -- so only the read below changes. */
            if (dev->FP16<=1){
                strcat(temp, GLOBARG"float * ");
            }
            else{
                strcat(temp, GLOBARG"half * ");
            }
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, GLOBARG"float2 * f");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
        }
    }
    chop_tail(temp, 2);
    strcat(temp, "){\n\n");

    /* No float2 fact[NFREQS] array: it was a per-thread array indexed in a
     * loop, which costs registers and can spill as NFREQS grows. The twiddle is
     * now computed inside the frequency loop and consumed immediately. */
    strcat(temp,"    int freq;\n");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            strcat(temp, "    float  l");
            strcat(temp, vars[i].name);
            strcat(temp, ";\n");
        }
    }
    chop_tail(temp, 2);
    strcat(temp, ";\n\n");
    
    /* Flat 1-D launch in both backends -- see kernel_initsavefreqs. */
    #ifdef __SEISCL__
    strcat(temp,"    int gid = get_global_id(0);\n\n" );
    #else
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n\n");
    #endif

    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);

            strcat(temp,ptemp);
            strcat(temp, "    ");
            strcat(temp, "    l");
            strcat(temp, vars[i].name);
            strcat(temp, "=");
            /* OpenCL enables cl_khr_fp16 (header_CUDACL.cl), so a half load
               converts implicitly; CUDA needs the explicit intrinsic. Same
               split kernel_varout uses. */
            #ifdef __SEISCL__
            strcat(temp, vars[i].name);
            strcat(temp, "[gid];\n");
            #else
            if (dev->FP16>1){
                strcat(temp, "__half2float(");
                strcat(temp, vars[i].name);
                strcat(temp, "[gid]);\n");
            }
            else{
                strcat(temp, vars[i].name);
                strcat(temp, "[gid];\n");
            }
            #endif
        }
    }






    /* One pass over the frequencies, applying each twiddle to every variable
     * while it is still in a register.
     *
     * The twiddle argument is computed in *float*. It used to be
     * "2.0*gradfreqsn[freq]*nt/NTNYQ", where 2.0 is a double literal, so the
     * whole expression promoted to double and called the double-precision
     * cospi/sinpi -- 98 double-precision FLOPs per grid point per frequency on
     * hardware whose FP64 rate is 1/32 of FP32. That single literal made
     * savefreqs 84% of all GPU time. Float costs ~1e-6 relative error in the
     * accumulated spectrum, well under the fp32 accumulation noise floor, so
     * there is nothing to buy by keeping double here.
     *
     * The accumulator is also read and written as one float2 rather than as
     * two separate .x/.y accesses, halving the memory instructions. */
    strcat(temp,
        "\n"
        "    for (freq=0;freq<NFREQS;freq++){\n"
        "        float ang = 2.0f*gradfreqsn[freq]*(float)nt/(float)NTNYQ;\n"
        "        float fr =  DTNYQ*DT*COSPI(ang);\n"
        "        float fi = -DTNYQ*DT*SINPI(ang);\n");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            sprintf(ptemp,"        if (gid<%d){\n", vars[i].num_ele);
            strcat(temp,ptemp);
            sprintf(ptemp,"            int ind = gid+freq*%d;\n",
                    vars[i].num_ele);
            strcat(temp,ptemp);
            strcat(temp, "            float2 acc = f");
            strcat(temp, vars[i].name);
            strcat(temp, "[ind];\n");
            strcat(temp, "            acc.x += fr*l");
            strcat(temp, vars[i].name);
            strcat(temp, ";\n            acc.y += fi*l");
            strcat(temp, vars[i].name);
            strcat(temp, ";\n            f");
            strcat(temp, vars[i].name);
            strcat(temp, "[ind] = acc;\n");
            strcat(temp, "        }\n");
        }
    }
    strcat(temp, "    }\n");
    
    strcat(temp, "\n}");

       __GUARD prog_source(prog, "savefreqs", temp, 0, NULL);
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
#ifdef __SEISCL__
    /* One work-item per element of the padded buffer -- see the same comment in
     * kernel_initsavefreqs. prod(dev->N) under-covers by
     * prod(N+FDORDER) - prod(N) elements. */
    prog->gsize[0] = maxsize;
#else
    /* Same flat 1-D launch as OpenCL -- see kernel_initsavefreqs. */
    prog->gsize[0] = maxsize;
#endif
    prog->wdim=1;

    return state;
    
}

int kernel_init_gradsrc(clprogram * prog){
    
    int state=0;
    
    char temp[MAX_KERN_STR]={0};;
    
    
    strcat(temp,
           FUNDEF"void init_gradsrc("GLOBARG"float *gradsrc, int nsrc)\n"
           "{\n\n");
    #ifdef __SEISCL__
    strcat(temp,
           "    int gid = get_global_id(0);\n"
           "    gradsrc[gid]=0.0;\n\n"
           "}");
    #else
    strcat(temp,
           "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
           "    if (gid>nsrc*NT-1){\n"
           "        return;\n"
           "    }\n"
           "    gradsrc[gid]=0.0;\n\n"
           "}");
    #endif
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
       __GUARD prog_source(prog, "init_gradsrc", temp, 0, NULL);
    
    prog->wdim=1;
    
    return state;
    
}

int kernel_fcom_out(device * dev,
                    variable * vars,
                    clprogram * prog,
                    int upid, int buff12, int adj){
    
    int state=0;
    int i,j;
    
    char temp[MAX_KERN_STR]={0};;
    
    char ptemp[200];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize){
            maxsize=vars[i].num_ele;
        }
    }
    
    char prec[7];
    
    if (dev->FP16==0){
        sprintf(prec, "float");
    }
    else if (dev->FP16==1){
        sprintf(prec, "float2");
    }
    else{
        sprintf(prec, "half2");
    }
    
    strcat(temp, FUNDEF"void fill_transfer_buff_out(");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_comm==upid){
            strcat(temp, GLOBARG);
            strcat(temp, prec);
            strcat(temp, " * ");
            strcat(temp, vars[i].name);
            if (adj>0){
                strcat(temp, "r");
            }
            strcat(temp, ", ");
            strcat(temp, GLOBARG);
            strcat(temp, prec);
            strcat(temp, " * ");
            strcat(temp, vars[i].name);
            if (adj>0){
                strcat(temp, "r");
            }
            if (buff12==1)
                strcat(temp, "_buf1, ");
            else if (buff12==2)
                strcat(temp, "_buf2, ");
        }
    }
    chop_tail(temp, 2);
    strcat(temp, "){\n\n");
    
    
    //Indice if local memory is used
    char * names[] = {"x","y","z"};
    if (dev->LOCAL_OFF==0){
        for (i=0;i<dev->NDIM;i++){
            #ifdef __SEISCL__
            sprintf(ptemp,"    int gid%s=get_global_id(%d);\n",
                    dev->N_names[i],i );
            #else
            sprintf(ptemp,"    int gid%s=blockIdx.%s*blockDim.%s + threadIdx.%s;\n",
                            dev->N_names[i],names[i],names[i] ,names[i]  );
            #endif
            strcat(temp, ptemp);
        }
    }
    //if we use directly global memory, with 1 working dim
    else{
        #ifdef __SEISCL__
        strcat(temp,"    int gid = get_global_id(0);\n");
        #else
        strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
        #endif
        sprintf(ptemp,"    int gid%s=gid%%(N%s-2*FDOH);\n",
                           dev->N_names[0],dev->N_names[0]);
        strcat(temp, ptemp);
        
        for (i=1;i<dev->NDIM;i++){
            sprintf(ptemp,"    int gid%s=(gid",dev->N_names[i]);
            strcat(temp, ptemp);
            for (j=0;j<i;j++){
                sprintf(ptemp,"/(N%s-2*FDOH)",dev->N_names[j]);
                strcat(temp, ptemp);
            }
            if (i<dev->NDIM-1){
                sprintf(ptemp,")%%(N%s-2*FDOH);\n",dev->N_names[i]);
                strcat(temp, ptemp);
            }
            else
                strcat(temp,");\n");
        }
        
        
    }
    strcat(temp,"\n");
    
    strcat(temp,"    int idbuf=");
    for (i=0;i<dev->NDIM;i++){
        sprintf(ptemp,"gid%s",dev->N_names[i]);
        strcat(temp, ptemp);
        for (j=0;j<i;j++){
            if (j==0 && dev->FP16>0){
                sprintf(ptemp,"*(N%s-FDOH)",dev->N_names[j]);
            }
            else{
                sprintf(ptemp,"*(N%s-2*FDOH)",dev->N_names[j]);
            }
            strcat(temp, ptemp);
        }
        if (i!=dev->NDIM-1){
            strcat(temp, "+");
        }
        else{
            strcat(temp, ";\n");
        }
    }
    char temp2[1000]={0};   /* one generated line; 100 overflowed at FP16 2 */
    int precsize;
    int div;
    if (dev->FP16==0){
        precsize = sizeof(float);
        div=1;
    }
    else if (dev->FP16==1){
        precsize = sizeof(float)*2;
        div=2;
    }
    else{
        precsize = sizeof(float);
        div=2;
    }
    
//    if (buff12==1){
//        sprintf(temp2,"    if (idbuf>%d-1){\n",
//                (int)(vars[0].cl_buf1.size/precsize));
//    }
//    else{
//        sprintf(temp2,"    if (idbuf>%d-1){\n",
//                (int)(vars[0].cl_buf2.size/precsize));
//    }
//    strcat(temp,temp2);
//    strcat(temp,  "        return;\n"
//           "    }\n\n");
    
    if (dev->FP16==0){
        sprintf(temp2,"    if (gid%s > (N%s - 2*FDOH-1)",
                dev->N_names[0], dev->N_names[0]);
    }
    else{
        sprintf(temp2,"    if (gid%s > (N%s - FDOH-1)",
                dev->N_names[0], dev->N_names[0]);
    }
    strcat(temp, temp2);
    for (i=1;i<dev->NDIM-1;i++){
        sprintf(temp2," || gid%s > (N%s - 2*FDOH-1)",
                dev->N_names[i], dev->N_names[i]);
        strcat(temp, temp2);
    }
    sprintf(temp2," || gid%s > FDOH-1 ){\n", dev->N_names[dev->NDIM-1]);
    strcat(temp, temp2);
    strcat(temp, "        return;\n"
           "    }\n\n");

    strcat(temp,"    int idvar=");
    for (i=0;i<dev->NDIM;i++){
        if (i!=dev->NDIM-1){
            if (i==0 && dev->FP16>0){
                sprintf(ptemp,"(gid%s+FDOH/2)",dev->N_names[i]);
            }
            else {
                sprintf(ptemp,"(gid%s+FDOH)",dev->N_names[i]);
            }
        }
        else{
            if (buff12==1)
                sprintf(ptemp,"(gid%s+FDOH)",dev->N_names[i]);
            else
                sprintf(ptemp,"(gid%s+N%s-2*FDOH)",
                              dev->N_names[i],dev->N_names[i]);
        }
        strcat(temp, ptemp);
        for (j=0;j<i;j++){
            sprintf(ptemp,"*N%s",dev->N_names[j]);
            strcat(temp, ptemp);
        }
        if (i!=dev->NDIM-1){
            strcat(temp, "+");
        }
    }
    strcat(temp,";\n\n");

    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_comm==upid){
            if (adj==0){
                sprintf(ptemp,"    %s_buf%d[idbuf]=%s[idvar];\n",
                        vars[i].name, buff12, vars[i].name);
            }
            else{
                sprintf(ptemp,"    %sr_buf%d[idbuf]=%sr[idvar];\n",
                        vars[i].name, buff12, vars[i].name);
            }
            strcat(temp, ptemp);
        }
    }
    
    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    __GUARD prog_source(prog, "fill_transfer_buff_out", temp, 0, NULL);
    
    return state;
    
}

int kernel_fcom_in(device * dev,
                   variable * vars,
                   clprogram * prog,
                   int upid, int buff12, int adj){
    
    int state=0;
    int i,j;
    
    char temp[MAX_KERN_STR]={0};;
    
    char ptemp[200];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize){
            maxsize=vars[i].num_ele;
        }
    }
    
    char prec[7];
    int div;
    if (dev->FP16==0){
        sprintf(prec, "float");
        div=1;
    }
    else if (dev->FP16==1){
        sprintf(prec, "float2");
        div=2;
    }
    else{
        sprintf(prec, "half2");
        div=2;
    }
    
    strcat(temp, FUNDEF"void fill_transfer_buff_in(");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_comm==upid){
            strcat(temp, GLOBARG);
            strcat(temp, prec);
            strcat(temp, " * ");
            strcat(temp, vars[i].name);
            if (adj>0){
                strcat(temp, "r");
            }
            strcat(temp, ", ");
            strcat(temp, GLOBARG);
            strcat(temp, prec);
            strcat(temp, " * ");
            strcat(temp, vars[i].name);
            if (adj>0){
                strcat(temp, "r");
            }
            if (buff12==1)
                strcat(temp, "_buf1, ");
            else if (buff12==2)
                strcat(temp, "_buf2, ");
        }
    }
    chop_tail(temp, 2);
    strcat(temp, "){\n\n");
    
    char * names[] = {"x","y","z"};
    if (dev->LOCAL_OFF==0){
        for (i=0;i<dev->NDIM;i++){
            #ifdef __SEISCL__
            sprintf(ptemp,"    int gid%s=get_global_id(%d);\n",
                    dev->N_names[i],i );
            #else
            sprintf(ptemp,"    int gid%s=blockIdx.%s*blockDim.%s + threadIdx.%s;\n",
                    dev->N_names[i],names[i],names[i] ,names[i]  );
            #endif
            strcat(temp, ptemp);
        }
    }
    //if we use directly global memory, with 1 working dim
    else{
        #ifdef __SEISCL__
        strcat(temp,"    int gid = get_global_id(0);\n");
        #else
        strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
        #endif
        sprintf(ptemp,"    int gid%s=gid%%(N%s-2*FDOH);\n",
                          dev->N_names[0],dev->N_names[0]);
        strcat(temp, ptemp);
        
        for (i=1;i<dev->NDIM;i++){
            sprintf(ptemp,"    int gid%s=(gid",dev->N_names[i]);
            strcat(temp, ptemp);
            for (j=0;j<i;j++){
                sprintf(ptemp,"/(N%s-2*FDOH)",dev->N_names[j]);
                strcat(temp, ptemp);
            }
            if (i<dev->NDIM-1){
                sprintf(ptemp,")%%(N%s-2*FDOH);\n",dev->N_names[i]);
                strcat(temp, ptemp);
            }
            else
                strcat(temp,");\n");
        }
        
        
    }
    strcat(temp,"\n");
    
    strcat(temp,"    int idbuf=");
    for (i=0;i<dev->NDIM;i++){
        sprintf(ptemp,"gid%s",dev->N_names[i]);
        strcat(temp, ptemp);
        for (j=0;j<i;j++){
            if (j==0 && dev->FP16>0){
                sprintf(ptemp,"*(N%s-FDOH)",dev->N_names[j]);
            }
            else{
                sprintf(ptemp,"*(N%s-2*FDOH)",dev->N_names[j]);
            }
            strcat(temp, ptemp);
        }
        if (i!=dev->NDIM-1){
            strcat(temp, "+");
        }
    }
    strcat(temp,";\n");

    strcat(temp,"    int idvar=");
    for (i=0;i<dev->NDIM;i++){
        if (i!=dev->NDIM-1){
            if (i==0 && dev->FP16>0){
                sprintf(ptemp,"(gid%s+FDOH/2)",dev->N_names[i]);
            }
            else {
                sprintf(ptemp,"(gid%s+FDOH)",dev->N_names[i]);
            }
        }
        else{
            if (buff12==1)
                sprintf(ptemp,"(gid%s)",dev->N_names[i]);
            else
                sprintf(ptemp,"(gid%s+N%s-FDOH)",
                             dev->N_names[i],dev->N_names[i]);
        }
        strcat(temp, ptemp);
        for (j=0;j<i;j++){
            sprintf(ptemp,"*N%s",dev->N_names[j]);
            strcat(temp, ptemp);
        }
        if (i!=dev->NDIM-1){
            strcat(temp, "+");
        }
    }
    strcat(temp,";\n\n");
    
    
    int precsize;
    if (dev->FP16==0){
        precsize = sizeof(float);
    }
    else if (dev->FP16==1){
        precsize = sizeof(float)*2;
    }
    else{
        precsize = sizeof(float);
    }
    char temp2[1000]={0};   /* one generated line; 100 overflowed at FP16 2 */
//    if (buff12==1){
//        sprintf(temp2,"    if (idbuf>%d-1){\n",
//                (int)(vars[0].cl_buf1.size/precsize));
//    }
//    else{
//        sprintf(temp2,"    if (idbuf>%d-1){\n",
//                (int)(vars[0].cl_buf1.size/precsize));
//    }
//    strcat(temp,temp2);
//    strcat(temp,  "        return;\n"
//           "    };\n\n");
    
    if (dev->FP16==0){
        sprintf(temp2,"    if (gid%s > (N%s - 2*FDOH-1)",
                dev->N_names[0], dev->N_names[0]);
    }
    else{
        sprintf(temp2,"    if (gid%s > (N%s - FDOH-1)",
                dev->N_names[0], dev->N_names[0]);
    }
     strcat(temp, temp2);
    for (i=1;i<dev->NDIM-1;i++){
        sprintf(temp2," || gid%s > (N%s - 2*FDOH-1)",
                dev->N_names[i], dev->N_names[i]);
        strcat(temp, temp2);
    }
    sprintf(temp2," || gid%s > FDOH-1 ){\n", dev->N_names[dev->NDIM-1]);
    strcat(temp, temp2);
    strcat(temp, "        return;\n"
                 "    }\n\n");
    
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_comm==upid){
            sprintf(ptemp,"    %s[idvar]=%s_buf%d[idbuf];\n",
                               vars[i].name, vars[i].name, buff12);
            if (adj==0){
                sprintf(ptemp,"    %s[idvar]=%s_buf%d[idbuf];\n",
                        vars[i].name, vars[i].name, buff12);
            }
            else{
                sprintf(ptemp,"    %sr[idvar]=%sr_buf%d[idbuf];\n",
                        vars[i].name, vars[i].name, buff12);
            }
            strcat(temp, ptemp);
        }
    }
//    sprintf(ptemp,"    printf(\"%%d\\n\", gidX);\n");
    strcat(temp, ptemp);
    strcat(temp, "\n}");
    
    
//        printf("%s\n\n%lu\n",temp, strlen(temp));
    
       __GUARD prog_source(prog, "fill_transfer_buff_in", temp, 0, NULL);
    
    
    return state;
    
}



