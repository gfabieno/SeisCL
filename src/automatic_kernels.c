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


//int kernel_varout(device * dev,
//                  clprogram * prog){
// 
//    int state=0;
//    int i,j;
//    
//    char temp[MAX_KERN_STR]={0};;
//    char temp2[100]={0};
//    char * p=(char*)temp;
//    variable * vars = dev->vars;
//    variable * tvars = dev->trans_vars;
//    
//    strcat(temp, "__kernel void varsout(int nt, __global float * rec_pos, ");
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_output){
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            strcat(temp, ", ");
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            strcat(temp, "out, ");
//        }
//    }
//    for (i=0;i<dev->ntvars;i++){
//        if (tvars[i].to_output){
//            for (j=0;j<tvars[i].n2ave;j++){
//                strcat(temp, "__global float * ");
//                strcat(temp, tvars[i].var2ave[j]);
//                strcat(temp, ", ");
//            }
//
//            strcat(temp, "__global float * ");
//            strcat(temp, tvars[i].name);
//            strcat(temp, "out, ");
//        }
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//
//    //This only supports 3 dimensions (need for more ?)
//    strcat(temp,"    int gid = get_global_id(0);\n"
//           "    int i=(int)(rec_pos[0+8*gid]/DH)+FDOH;\n"
//           "    int j=(int)(rec_pos[1+8*gid]/DH)+FDOH;\n"
//           "    int k=(int)(rec_pos[2+8*gid]/DH)+FDOH;\n"
//           "\n");
//    
//    sprintf(temp2,"    if (i-OFFSET<FDOH || i-OFFSET>N%s-FDOH-1){\n",
//            dev->N_names[dev->NDIM-1] );
//    strcat(temp, temp2);
//    strcat(temp,
//           "        return;\n"
//           "    };\n\n");
//
//    
//    char posstr[100]={0};
//    
//    if (dev->NDIM==2){
//        sprintf(posstr,"[(i-OFFSET)*N%s+k]",dev->N_names[0]);
//    }
//    else if (dev->NDIM==3){
//        sprintf(posstr,"[(i-OFFSET)*N%s*N%s+j*(N%s)+k]",
//                dev->N_names[1], dev->N_names[0], dev->N_names[0]);
//    }
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_output){
//            strcat(temp, "    ");
//            strcat(temp, vars[i].name);
//            strcat(temp, "out[NT*gid+nt]=");
//            strcat(temp, vars[i].name);
//            strcat(temp, posstr);
//            strcat(temp, ";\n");
//        }
//    }
//    for (i=0;i<dev->ntvars;i++){
//        if (tvars[i].to_output){
//            strcat(temp, "    ");
//            strcat(temp, tvars[i].name);
//            strcat(temp, "out[NT*gid+nt]=(");
//            for (j=0;j<tvars->n2ave;j++){
//                strcat(temp, tvars[i].var2ave[j]);
//                strcat(temp, posstr);
//                strcat(temp, "+");
//            }
//            while (*p)
//                p++;
//            p[-1]='\0';
//            strcat(temp, ")/");
//            sprintf(temp2,"%d",tvars->n2ave);
//            strcat(temp, temp2);
//            strcat(temp, ";\n");
//        }
//    }
//
//    strcat(temp, "\n}");
//    
//    
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    
//    (*prog).src=temp;
//    
//    __GUARD prog_source(prog, "varsout", (*prog).src);
//    
//    prog->wdim=1;
//    
//    return state;
//
//}
//
//
//int kernel_varoutinit(device * dev,
//                      clprogram * prog){
//    
//    int state=0;
//    int i;
//    variable * vars = dev->vars;
//    variable * tvars = dev->trans_vars;
//    char temp[MAX_KERN_STR]={0};;
//    
//    char * p=(char*)temp;
//    
//    strcat(temp, "__kernel void varsoutinit(");
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_output){
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            strcat(temp, "out, ");
//        }
//    }
//    for (i=0;i<dev->ntvars;i++){
//        if (tvars[i].to_output){
//            strcat(temp, "__global float * ");
//            strcat(temp, tvars[i].name);
//            strcat(temp, "out, ");
//        }
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//    
//    
//    strcat(temp,"    int gid = get_global_id(0);\n\n");
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_output){
//            strcat(temp, "    ");
//            strcat(temp, vars[i].name);
//            strcat(temp, "out[gid]=0;\n");
//        }
//    }
//    for (i=0;i<dev->ntvars;i++){
//        if (tvars[i].to_output){
//            strcat(temp, "    ");
//            strcat(temp, tvars[i].name);
//            strcat(temp, "out[gid]=0;\n");
//        }
//    }
//
//    strcat(temp, "\n}");
//    
//    
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    
//    (*prog).src=temp;
//    
//    __GUARD prog_source(prog, "varsoutinit", (*prog).src);
//    
//    prog->wdim=1;
//    
//    return state;
//    
//}
//
//int kernel_varinit(device * dev,
//                   variable * vars,
//                   clprogram * prog){
//    
//    int state=0;
//    int i;
//    
//    char temp[MAX_KERN_STR]={0};;
//    
//    char * p=(char*)temp;
//    char ptemp[50];
//    
//    int maxsize=0;
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].num_ele>maxsize){
//            maxsize=vars[i].num_ele;
//        }
//    }
//    
//    
//    strcat(temp, "__kernel void vars_init(");
//    for (i=0;i<dev->nvars;i++){
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            strcat(temp, ", ");
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//    
//    
//    strcat(temp,"    int gid = get_global_id(0);\n\n");
//    
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].num_ele<maxsize){
//            sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
//            strcat(temp,ptemp);
//            strcat(temp, "    ");
//        }
//        strcat(temp, "    ");
//        strcat(temp, vars[i].name);
//        strcat(temp, "[gid]=0;\n");
//    }
//    
//    strcat(temp, "\n}");
//    
//    
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    
//    (*prog).src=temp;
//    
//    __GUARD prog_source(prog, "vars_init", (*prog).src);
//    
//    prog->wdim=1;
//    
//    return state;
//    
//}
//
//int kernel_sources(device * dev,
//                   clprogram * prog){
//    
//    int state=0;
//    int i,j;
//    variable * vars = dev->vars;
//    variable * tvars = dev->trans_vars;
//    char temp[MAX_KERN_STR]={0};
//    char temp2[100]={0};
////    
//    char * p=(char*)temp;
//    
//    
//    
//    int * tosources=NULL;
//    int * tosources2=NULL;
//    int ntypes=0;
//    int ind;
//    GMALLOC(tosources,dev->nvars*sizeof(int));
//    GMALLOC(tosources2,dev->ntvars*sizeof(int));
//    for (i=0;i<dev->src_recs.allns;i++){
//        ind =dev->src_recs.src_pos[0][4+i*5];
//        if (ind<dev->nvars && ind>-1)
//            tosources[ind]=1;
//        if (ind-100<dev->ntvars && ind-100>-1)
//            tosources2[ind-100]=1;
//    }
//
//    for (i=0;i<dev->nvars;i++){
//        if (tosources[i]==1){
//            ntypes++;
//        }
//    }
//    for (i=0;i<dev->ntvars;i++){
//        if (tosources2[i]==1){
//            ntypes++;
//        }
//    }
//    
//    if (ntypes==0){
//        state=1;
//        fprintf(stderr,"Error: No sources for variable list found\n");
//    }
//    
//    strcat(temp, "__kernel void sources(int nt, __global float * src_pos,"
//                 " __global float * src, int pdir, ");
//    for (i=0;i<dev->nvars;i++){
//        if (tosources[i]){
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            strcat(temp, ", ");
//        }
//    }
//    for (i=0;i<dev->ntvars;i++){
//        if (tosources2[i]){
//            for (j=0;j<tvars[i].n2ave;j++){
//                strcat(temp, "__global float * ");
//                strcat(temp, tvars[i].var2ave[j]);
//                strcat(temp, ", ");
//            }
//        }
//    }
//
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//    
//    //This only supports 3 dimensions (need for more ?)
//    strcat(temp,"    int gid = get_global_id(0);\n"
//           "    int i=(int)(src_pos[0+5*gid]/DH)+FDOH;\n"
//           "    int j=(int)(src_pos[1+5*gid]/DH)+FDOH;\n"
//           "    int k=(int)(src_pos[2+5*gid]/DH)+FDOH;\n"
//           "\n");
//
//    sprintf(temp2,"    if (i-OFFSET<FDOH || i-OFFSET>N%s-FDOH-1){\n",
//            dev->N_names[dev->NDIM-1] );
//    strcat(temp, temp2);
//    strcat(temp,
//           "        return;\n"
//           "    }\n\n"
//           "    int source_type= src_pos[4+5*gid];\n"
//           "    float amp=(DT*src[gid*NT+nt])/(DH*DH*DH);\n\n");
//
//    char posstr[100]={0};
//
//    
//    if (dev->NDIM==2){
//        sprintf(posstr,"[(i-OFFSET)*N%s+k]",dev->N_names[0]);
//    }
//    else if (dev->NDIM==3){
//        sprintf(posstr,"[(i-OFFSET)*N%s*N%s+j*(N%s)+k]",
//                dev->N_names[1], dev->N_names[0], dev->N_names[0]);
//    }
//    else{
//        state=1;
//        fprintf(stderr,"Error: Sources for a number of dimensions higher "
//                       "than 3 are not supported yet\n");
//    }
//    
//    
//    for (i=0;i<dev->nvars;i++){
//        
//        if (tosources[i]){
//            if (ntypes>1){
//                sprintf(temp2,"    if (source_type==%d)\n", i);
//                strcat(temp, temp2);
//                strcat(temp, "    ");
//            }
//            strcat(temp, "    ");
//            strcat(temp, vars[i].name);
//            strcat(temp, posstr);
//            strcat(temp, "+=pdir*amp;\n");
//            
//        }
//    }
//    for (i=0;i<dev->ntvars;i++){
//        
//        if (tosources2[i]){
//            if (ntypes>1){
//                sprintf(temp2,"    if (source_type==%d){\n", i+100);
//                strcat(temp, temp2);
//                strcat(temp, "    ");
//            }
//            for (j=0;j<tvars[i].n2ave;j++){
//                strcat(temp, "    ");
//                strcat(temp, tvars[i].var2ave[j]);
//                strcat(temp, posstr);
//                sprintf(temp2,"+=pdir*amp/%d;\n", tvars[i].n2ave);
//                strcat(temp, temp2);
//            }
//            if (ntypes>1){
//                strcat(temp, "    }");
//            }
//        }
//    }
//    
//    strcat(temp, "\n}");
//    
//    (*prog).src=temp;
//    
//    __GUARD prog_source(prog, "sources", (*prog).src);
//    
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    free(tosources);
//    free(tosources2);
//    
//    prog->wdim=1;
//    
//    return state;
//    
//}
//
//int kernel_residuals(device * dev,
//                     clprogram * prog,
//                     int BACK_PROP_TYPE){
//    
//    int state=0;
//    int i,j;
//    
//    char temp[MAX_KERN_STR]={0};;
//    char temp2[100]={0};
//    char * p=(char*)temp;
//    variable * vars = dev->vars;
//    variable * tvars = dev->trans_vars;
//    
//   
//    
//    strcat(temp, "__kernel void residuals(int nt, __global float * rec_pos,");
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_output){
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            if (BACK_PROP_TYPE==1)
//                strcat(temp, "_r");
//            strcat(temp, ", ");
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            strcat(temp, "out, ");
//        }
//    }
//    for (i=0;i<dev->ntvars;i++){
//        if (tvars[i].to_output){
//            for (j=0;j<tvars[i].n2ave;j++){
//                strcat(temp, "__global float * ");
//                strcat(temp, tvars[i].var2ave[j]);
//                if (BACK_PROP_TYPE==1)
//                    strcat(temp, "_r");
//                strcat(temp, ", ");
//            }
//            
//            strcat(temp, "__global float * ");
//            strcat(temp, tvars[i].name);
//            strcat(temp, "out, ");
//        }
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//    
//    
//    strcat(temp,"    int gid = get_global_id(0);\n"
//           "    int i=(int)(rec_pos[0+8*gid]/DH)+FDOH;\n"
//           "    int j=(int)(rec_pos[1+8*gid]/DH)+FDOH;\n"
//           "    int k=(int)(rec_pos[2+8*gid]/DH)+FDOH;\n\n");
//    
//    sprintf(temp2,"    if (i-OFFSET<FDOH || i-OFFSET>N%s-FDOH-1){\n",
//            dev->N_names[dev->NDIM-1] );
//    strcat(temp, temp2);
//    strcat(temp,
//           "        return;\n"
//           "    };\n\n");
//    
//    char posstr[100]={0};
//
//    if (dev->NDIM==2){
//        sprintf(posstr,"[(i-OFFSET)*N%s+k]",dev->N_names[0]);
//    }
//    else if (dev->NDIM==3){
//        sprintf(posstr,"[(i-OFFSET)*N%s*N%s+j*(N%s)+k]",
//                dev->N_names[1], dev->N_names[0], dev->N_names[0]);
//    }
//
//    
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_output){
//            strcat(temp, "        ");
//            strcat(temp, vars[i].name);
//            if (BACK_PROP_TYPE==1)
//                strcat(temp, "_r");
//            strcat(temp, posstr);
//            strcat(temp, "+=");
//            strcat(temp, vars[i].name);
//            strcat(temp, "out[NT*gid+nt];\n");
//
//        }
//    }
//    for (i=0;i<dev->ntvars;i++){
//        if (tvars[i].to_output){
//            for (j=0;j<tvars[i].n2ave;j++){
//                strcat(temp, "        ");
//                strcat(temp, tvars[i].var2ave[j]);
//                if (BACK_PROP_TYPE==1)
//                    strcat(temp, "_r");
//                strcat(temp, posstr);
//                strcat(temp, "+=");
//                strcat(temp, tvars[i].name);
//                strcat(temp, "out[NT*gid+nt]/");
//                sprintf(temp2,"%d",tvars[i].n2ave);
//                strcat(temp, temp2);
//                strcat(temp, ";\n");
//            }
//        }
//    }
//    
//    strcat(temp, "\n}");
//    
//    (*prog).src=temp;
//    
//    __GUARD prog_source(prog, "residuals", (*prog).src);
//
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    prog->wdim=1;
//    
//    return state;
//    
//}
//
//
//
//int kernel_gradinit(device * dev,
//                    parameter * pars,
//                    clprogram * prog){
//    
//    int state=0;
//    int i;
//    
//    char temp[MAX_KERN_STR]={0};;
//    
//    char * p=(char*)temp;
//    
//
//    strcat(temp, "__kernel void gradinit(");
//    for (i=0;i<dev->npars;i++){
//        if (pars[i].to_grad){
//            strcat(temp, "__global float * grad");
//            strcat(temp, pars[i].name);
//            strcat(temp, ", ");
//        }
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//    
//    
//    strcat(temp,"    int gid = get_global_id(0);\n\n");
//    
//    
//    for (i=0;i<dev->npars;i++){
//        if (pars[i].to_grad){
//            strcat(temp, "    grad");
//            strcat(temp, pars[i].name);
//            strcat(temp, "[gid]=0;\n");
//        }
//    }
//
//    strcat(temp, "\n}");
//    
//   (*prog).src=temp;
//    
//    __GUARD prog_source(prog, "gradinit", (*prog).src);
//    
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    prog->wdim=1;
//    
//    return state;
//    
//}
//
//int kernel_initsavefreqs(device * dev,
//                         variable * vars,
//                         clprogram * prog){
//    
//    int state=0;
//    int i;
//    
//    char temp[MAX_KERN_STR]={0};;
//    
//    char * p=(char*)temp;
//    char ptemp[50];
//    
//    int maxsize=0;
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].num_ele>maxsize && vars[i].for_grad){
//            maxsize=vars[i].num_ele;
//        }
//    }
//    
//    strcat(temp, "__kernel void initsavefreqs(");
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].for_grad){
//            strcat(temp, "__global float2 * f");
//            strcat(temp, vars[i].name);
//            strcat(temp, ", ");
//        }
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//    
//    
//    strcat(temp,"    int gid = get_global_id(0);\n");
//    
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].for_grad){
//            if (vars[i].num_ele<maxsize){
//                sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
//                strcat(temp,ptemp);
//                strcat(temp, "    ");
//            }
//            strcat(temp, "    f");
//            strcat(temp, vars[i].name);
//            strcat(temp, "[gid]=0;\n");
//        }
//    }
//    
//    strcat(temp, "\n}");
//    
//    (*prog).src=temp;
//    
//    __GUARD prog_source(prog, "initsavefreqs", (*prog).src);
//    
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    prog->wdim=1;
//    
//    return state;
//    
//}
//
//int kernel_savefreqs(device * dev,
//                     variable * vars,
//                     clprogram * prog){
//    
//    int state=0;
//    int i;
//    
//    char temp[MAX_KERN_STR]={0};;
//    
//    char * p=(char*)temp;
//    char ptemp[50];
//    
//    int maxsize=0;
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].num_ele>maxsize && vars[i].for_grad){
//            maxsize=vars[i].num_ele;
//        }
//    }
//    
//    strcat(temp,"__kernel void savefreqs(__constant float *gradfreqsn, int nt, ");
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].for_grad){
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            strcat(temp, ", ");
//            strcat(temp, "__global float2 * f");
//            strcat(temp, vars[i].name);
//            strcat(temp, ", ");
//        }
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//    
//    strcat(temp,"    int freq,l;\n"
//                "    float2 fact[NFREQS]={0};\n");
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].for_grad){
//            strcat(temp, "    float  l");
//            strcat(temp, vars[i].name);
//            strcat(temp, ";\n");
//        }
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, ";\n\n");
//
//    strcat(temp,"    int gid = get_global_id(0);\n"
//                "    int gsize=get_global_size(0);\n\n" );
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].for_grad){
//            if (vars[i].num_ele<maxsize){
//                sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
//                strcat(temp,ptemp);
//                strcat(temp, "    ");
//            }
//            strcat(temp, "    l");
//            strcat(temp, vars[i].name);
//            strcat(temp, "=");
//            strcat(temp, vars[i].name);
//            strcat(temp, "[gid];\n");
//        }
//    }
//
//    
//    strcat(temp,"\n"
//        "    for (freq=0;freq<NFREQS;freq++){\n"
//        "        fact[freq].x = DTNYQ*DT*cospi(2.0*gradfreqsn[freq]*nt/NTNYQ);\n"
//        "        fact[freq].y = -DTNYQ*DT*sinpi(2.0*gradfreqsn[freq]*nt/NTNYQ);\n"
//        "    }\n\n"
//           );
//        
//        
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].for_grad){
//            if (vars[i].num_ele<maxsize){
//                sprintf(ptemp,"    if (gid<%d){\n", vars[i].num_ele);
//                strcat(temp,ptemp);
//            }
//            strcat(temp, "    #pragma unroll\n");
//            strcat(temp, "    for (freq=0;freq<NFREQS;freq++){\n");
//            strcat(temp, "        f");
//            strcat(temp, vars[i].name);
//            strcat(temp, "[gid+freq*");
//            sprintf(ptemp, "%d]+=fact[freq]*l", vars[i].num_ele);
//            strcat(temp,ptemp);
//            strcat(temp, vars[i].name);
//            strcat(temp, ";\n");
//            strcat(temp, "    }\n");
//            if (vars[i].num_ele<maxsize){
//                strcat(temp,"    }\n");
//            }
//        }
//    }
//    
//    strcat(temp, "\n}");
//    
//    (*prog).src=temp;
//    __GUARD prog_source(prog, "savefreqs", (*prog).src);
//    
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    prog->wdim=1;
//    
//    return state;
//    
//}
//
//int kernel_init_gradsrc(clprogram * prog){
//    
//    int state=0;
//    
//    char temp[MAX_KERN_STR]={0};;
//    
//    
//    strcat(temp,
//           "__kernel void init_gradsrc(__global float *gradsrc)\n"
//           "{\n\n"
//           "    int gid = get_global_id(0);\n"
//           "    gradsrc[gid]=0.0;\n\n"
//           "}");
//    
//    (*prog).src=temp;
//    
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    __GUARD prog_source(prog, "init_gradsrc", (*prog).src);
//    
//    prog->wdim=1;
//    
//    return state;
//    
//}
//
//int kernel_fcom_out(device * dev,
//                    variable * vars,
//                    clprogram * prog,
//                    int upid, int buff12){
//    
//    int state=0;
//    int i,j;
//    
//    char temp[MAX_KERN_STR]={0};;
//    
//    char * p=(char*)temp;
//    char ptemp[200];
//    
//    int maxsize=0;
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].num_ele>maxsize){
//            maxsize=vars[i].num_ele;
//        }
//    }
//    
//    strcat(temp, "__kernel void fill_transfer_buff_out(");
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_comm==upid){
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            strcat(temp, ", ");
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            if (buff12==1)
//                strcat(temp, "_buf1, ");
//            else if (buff12==2)
//                strcat(temp, "_buf2, ");
//        }
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//    
//    
//    //Indice if local memory is used
//    if (dev->LOCAL_OFF==0){
//        for (i=0;i<dev->NDIM;i++){
//            sprintf(ptemp,"    int gid%s=get_global_id(%d)+FDOH;\n",
//                                                dev->N_names[i],i );
//            strcat(temp, ptemp);
//        }
//    }
//    //if we use directly global memory, with 1 working dim
//    else{
//        strcat(temp,"    int gid = get_global_id(0);\n");
//
//        sprintf(ptemp,"    int gid%s=gid%%(N%s-2*FDOH);\n",
//                           dev->N_names[0],dev->N_names[0]);
//        strcat(temp, ptemp);
//        
//        for (i=1;i<dev->NDIM;i++){
//            sprintf(ptemp,"    int gid%s=(gid",dev->N_names[i]);
//            strcat(temp, ptemp);
//            for (j=0;j<i;j++){
//                sprintf(ptemp,"/(N%s-2*FDOH)",dev->N_names[j]);
//                strcat(temp, ptemp);
//            }
//            if (i<dev->NDIM-1){
//                sprintf(ptemp,")%%(N%s-2*fdoh);\n",dev->N_names[i]);
//                strcat(temp, ptemp);
//            }
//            else
//                strcat(temp,");\n");
//        }
//        
//        
//    }
//    strcat(temp,"\n");
//    
//    strcat(temp,"    int idbuf=");
//    for (i=0;i<dev->NDIM;i++){
//        sprintf(ptemp,"gid%s",dev->N_names[i]);
//        strcat(temp, ptemp);
//        for (j=0;j<i;j++){
//            sprintf(ptemp,"*(N%s-2*FDOH)",dev->N_names[j]);
//            strcat(temp, ptemp);
//        }
//        if (i!=dev->NDIM-1){
//            strcat(temp, "+");
//        }
//    }
//    strcat(temp,";\n");
//    strcat(temp,"    int idvar=");
//    for (i=0;i<dev->NDIM;i++){
//        if (i!=dev->NDIM-1)
//            sprintf(ptemp,"(gid%s+FDOH)",dev->N_names[i]);
//        else{
//            if (buff12==1)
//                sprintf(ptemp,"(gid%s+FDOH)",dev->N_names[i]);
//            else
//                sprintf(ptemp,"(gid%s+N%s-2*FDOH)",
//                              dev->N_names[i],dev->N_names[i]);
//        }
//        strcat(temp, ptemp);
//        for (j=0;j<i;j++){
//            sprintf(ptemp,"*N%s",dev->N_names[j]);
//            strcat(temp, ptemp);
//        }
//        if (i!=dev->NDIM-1){
//            strcat(temp, "+");
//        }
//    }
//    strcat(temp,";\n\n");
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_comm==upid){
//            sprintf(ptemp,"    %s_buf%d[idbuf]=%s[idvar];\n",
//                          vars[i].name, buff12, vars[i].name);
//            strcat(temp, ptemp);
//        }
//    }
//    
//    strcat(temp, "\n}");
//    
//    
////    printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    
//    (*prog).src=temp;
//    
//    __GUARD prog_source(prog, "fill_transfer_buff_out", (*prog).src);
//    
//    return state;
//    
//}
//
//int kernel_fcom_in(device * dev,
//                   variable * vars,
//                   clprogram * prog,
//                   int upid, int buff12){
//    
//    int state=0;
//    int i,j;
//    
//    char temp[MAX_KERN_STR]={0};;
//    
//    char * p=(char*)temp;
//    char ptemp[200];
//    
//    int maxsize=0;
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].num_ele>maxsize){
//            maxsize=vars[i].num_ele;
//        }
//    }
//    
//    strcat(temp, "__kernel void fill_transfer_buff_in(");
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_comm==upid){
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            strcat(temp, ", ");
//            strcat(temp, "__global float * ");
//            strcat(temp, vars[i].name);
//            if (buff12==1)
//                strcat(temp, "_buf1, ");
//            else if (buff12==2)
//                strcat(temp, "_buf2, ");
//        }
//    }
//    while (*p)
//        p++;
//    p[-2]='\0';
//    strcat(temp, "){\n\n");
//    
//    
//    //Indice if local memory is used
//    if (dev->LOCAL_OFF==0){
//        for (i=0;i<dev->NDIM;i++){
//            sprintf(ptemp,"    int gid%s=get_global_id(%d)+FDOH;\n",
//                                                dev->N_names[i],i );
//            strcat(temp, ptemp);
//        }
//    }
//    //if we use directly global memory, with 1 working dim
//    else{
//        strcat(temp,"    int gid = get_global_id(0);\n");
//        
//        sprintf(ptemp,"    int gid%s=gid%%(N%s-2*FDOH);\n",
//                          dev->N_names[0],dev->N_names[0]);
//        strcat(temp, ptemp);
//        
//        for (i=1;i<dev->NDIM;i++){
//            sprintf(ptemp,"    int gid%s=(gid",dev->N_names[i]);
//            strcat(temp, ptemp);
//            for (j=0;j<i;j++){
//                sprintf(ptemp,"/(N%s-2*FDOH)",dev->N_names[j]);
//                strcat(temp, ptemp);
//            }
//            if (i<dev->NDIM-1){
//                sprintf(ptemp,")%%(N%s-2*fdoh);\n",dev->N_names[i]);
//                strcat(temp, ptemp);
//            }
//            else
//                strcat(temp,");\n");
//        }
//        
//        
//    }
//    strcat(temp,"\n");
//    
//    strcat(temp,"    int idbuf=");
//    for (i=0;i<dev->NDIM;i++){
//        sprintf(ptemp,"gid%s",dev->N_names[i]);
//        strcat(temp, ptemp);
//        for (j=0;j<i;j++){
//            sprintf(ptemp,"*(N%s-2*FDOH)",dev->N_names[j]);
//            strcat(temp, ptemp);
//        }
//        if (i!=dev->NDIM-1){
//            strcat(temp, "+");
//        }
//    }
//    strcat(temp,";\n");
//    strcat(temp,"    int idvar=");
//    for (i=0;i<dev->NDIM;i++){
//        if (i!=dev->NDIM-1)
//            sprintf(ptemp,"(gid%s+FDOH)",dev->N_names[i]);
//        else{
//            if (buff12==1)
//                sprintf(ptemp,"(gid%s)",dev->N_names[i]);
//            else
//                sprintf(ptemp,"(gid%s+N%s-FDOH)",
//                             dev->N_names[i],dev->N_names[i]);
//        }
//        strcat(temp, ptemp);
//        for (j=0;j<i;j++){
//            sprintf(ptemp,"*N%s",dev->N_names[j]);
//            strcat(temp, ptemp);
//        }
//        if (i!=dev->NDIM-1){
//            strcat(temp, "+");
//        }
//    }
//    strcat(temp,";\n\n");
//    
//    for (i=0;i<dev->nvars;i++){
//        if (vars[i].to_comm==upid){
//            sprintf(ptemp,"    %s[idvar]=%s_buf%d[idbuf];\n",
//                               vars[i].name, vars[i].name, buff12);
//            strcat(temp, ptemp);
//        }
//    }
//    
//    strcat(temp, "\n}");
//    
//    
//        printf("%s\n\n%lu\n",temp, strlen(temp));
//    
//    
//    (*prog).src=temp;
//    
//    __GUARD prog_source(prog, "fill_transfer_buff_in", (*prog).src);
//    
//    
//    return state;
//    
//}
//


