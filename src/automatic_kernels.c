//
//  automatic_kernels.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-11.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include "F.h"


int kernel_varout(int NDIM, int nvars, struct variable * vars, const char ** source){
 
    int state=0;
    int i;
    
    char * temp=NULL;
    GMALLOC(temp, sizeof(char)*MAX_KERN_STR);
    char * p=(char*)temp;
    
    strcat(temp, "__kernel void seisout(int nt, __global float * rec_pos, ");
    for (i=0;i<nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "__global float * ");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, "__global float * ");
            strcat(temp, vars[i].name);
            strcat(temp, "out, ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");

    
    strcat(temp,"    int gid = get_global_id(0);\n"
           "    int i=(int)(rec_pos[0*8+gid]/DH)+FDOH;\n"
           "    int j=(int)(rec_pos[0*8+gid]/DH)+FDOH;\n"
           "    int k=(int)(rec_pos[0*8+gid]/DH)+FDOH;\n"
           "\n"
           "    if ( (i-OFFSET)>=FDOH && (i-OFFSET)<(NX-FDOH) ){\n\n");
    
    char * posstr=NULL;
    if (NDIM==2){
        posstr="[(i-OFFSET)*(NZ)+k];\n";
    }
    else{
        posstr="[(i-OFFSET)*NY*(NZ)+j*(NZ)+k];\n";
    }
    
    for (i=0;i<nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "        ");
            strcat(temp, vars[i].name);
            strcat(temp, "out[nt*8+gid]=");
            strcat(temp, vars[i].name);
            strcat(temp, posstr);
        }
    }
    
    strcat(temp, "\n    }");
    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    *source=temp;
    
    return state;

}


int kernel_varoutinit(int NDIM, int nvars, struct variable * vars, const char ** source){
    
    int state=0;
    int i;
    
    char * temp=NULL;
    GMALLOC(temp, sizeof(char)*MAX_KERN_STR);
    char * p=(char*)temp;
    
    strcat(temp, "__kernel void seisoutinit(");
    for (i=0;i<nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "__global float * ");
            strcat(temp, vars[i].name);
            strcat(temp, "out, ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    strcat(temp,"    int gid = get_global_id(0);\n\n");
    
    for (i=0;i<nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "    ");
            strcat(temp, vars[i].name);
            strcat(temp, "out[gid]=0;\n");
        }
    }

    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    *source=temp;
    
    return state;
    
}

int kernel_varinit(int NDIM, int nvars, struct variable * vars, const char ** source){
    
    int state=0;
    int i;
    
    char * temp=NULL;
    GMALLOC(temp, sizeof(char)*MAX_KERN_STR);
    char * p=(char*)temp;
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<nvars;i++){
        if (vars[i].num_ele>maxsize){
            maxsize=vars[i].num_ele;
        }
    }
    
    
    strcat(temp, "__kernel void vars_init(");
    for (i=0;i<nvars;i++){
            strcat(temp, "__global float * ");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    strcat(temp,"    int gid = get_global_id(0);\n\n");
    
    
    for (i=0;i<nvars;i++){
        if (vars[i].num_ele<maxsize){
            sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
            strcat(temp,ptemp);
            strcat(temp, "    ");
        }
        strcat(temp, "    ");
        strcat(temp, vars[i].name);
        strcat(temp, "[gid]=0;\n");
    }
    
    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    *source=temp;
    
    return state;
    
}

int kernel_residuals(int NDIM, int nvars, struct variable * vars, const char ** source){
    
    int state=0;
    int i;
    
    char * temp=NULL;
    GMALLOC(temp, sizeof(char)*MAX_KERN_STR);
    char * p=(char*)temp;

    
   
    
    strcat(temp, "__kernel void residuals(int nt, __global float * rec_pos,");
    for (i=0;i<nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "__global float * ");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, "__global float * ");
            strcat(temp, vars[i].name);
            strcat(temp, "out, ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    strcat(temp,"    int gid = get_global_id(0);\n"
           "    int i=(int)(rec_pos[0*8+gid]/DH)+FDOH;\n"
           "    int j=(int)(rec_pos[0*8+gid]/DH)+FDOH;\n"
           "    int k=(int)(rec_pos[0*8+gid]/DH)+FDOH;\n"
           "\n"
           "    if ( (i-OFFSET)>=FDOH && (i-OFFSET)<(NX-FDOH) ){\n\n");
    
    char * posstr=NULL;
    if (NDIM==2){
        posstr="[(i-OFFSET)*NZ+k]";
    }
    else{
        posstr="[(i-OFFSET)*NY*NZ+j*(NZ)+k]";
    }
    
    for (i=0;i<nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "        ");
            strcat(temp, vars[i].name);
            strcat(temp, posstr);
            strcat(temp, "+=");
            strcat(temp, vars[i].name);
            strcat(temp, "out[nt*8+gid];\n");

        }
    }
    
    strcat(temp, "\n    }");
    strcat(temp, "\n}");
    
    *source=temp;

//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    return state;
    
}

int kernel_gradinit(int NDIM, int nparams, struct parameter * params, const char ** source){
    
    int state=0;
    int i;
    
    char * temp=NULL;
    GMALLOC(temp, sizeof(char)*MAX_KERN_STR);
    char * p=(char*)temp;
    

    strcat(temp, "__kernel void gradinit(");
    for (i=0;i<nparams;i++){
        if (params[i].to_grad){
            strcat(temp, "__global float * grad");
            strcat(temp, params[i].name);
            strcat(temp, ", ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    strcat(temp,"    int gid = get_global_id(0);\n\n");
    
    
    for (i=0;i<nparams;i++){
        if (params[i].to_grad){
            strcat(temp, "    grad");
            strcat(temp, params[i].name);
            strcat(temp, "[gid]=0;\n");
        }
    }

    strcat(temp, "\n}");
    
    *source=temp;
    
    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    return state;
    
}

int kernel_initsavefreqs(int NDIM, int nvars, struct variable * vars, const char ** source){
    
    int state=0;
    int i;
    
    char * temp=NULL;
    GMALLOC(temp, sizeof(char)*MAX_KERN_STR);
    char * p=(char*)temp;
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<nvars;i++){
        if (vars[i].num_ele>maxsize && vars[i].for_grad){
            maxsize=vars[i].num_ele;
        }
    }
    
    strcat(temp, "__kernel void initsavefreqs(");
    for (i=0;i<nvars;i++){
        if (vars[i].for_grad){
            strcat(temp, "__global float * f");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    strcat(temp,"    int gid = get_global_id(0);\n\n");
    
    
    for (i=0;i<nvars;i++){
        if (vars[i].for_grad){
            if (vars[i].num_ele<maxsize){
                sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
                strcat(temp,ptemp);
                strcat(temp, "    ");
            }
            strcat(temp, "    f");
            strcat(temp, vars[i].name);
            strcat(temp, "[gid]=0;\n");
        }
    }
    
    strcat(temp, "\n}");
    
    *source=temp;
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    return state;
    
}

int kernel_savefreqs(int NDIM, int nvars, struct variable * vars, const char ** source){
    
    int state=0;
    int i;
    
    char * temp=NULL;
    GMALLOC(temp, sizeof(char)*MAX_KERN_STR);
    char * p=(char*)temp;
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<nvars;i++){
        if (vars[i].num_ele>maxsize && vars[i].for_grad){
            maxsize=vars[i].num_ele;
        }
    }
    
    strcat(temp, "__kernel void savefreqs(__constant float *freqs,  int nt, ");
    for (i=0;i<nvars;i++){
        if (vars[i].for_grad){
            strcat(temp, "__global float * ");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, "__global float2 * f");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    strcat(temp,"    int freq,l;\n"
                "    float2 fact[NFREQS]={0};\n");
    
    for (i=0;i<nvars;i++){
        if (vars[i].for_grad){
            strcat(temp, "    float  l");
            strcat(temp, vars[i].name);
            strcat(temp, ";\n");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, ";\n\n");

    strcat(temp,"    int gid = get_global_id(0);\n"
                "    int gsize=get_global_size(0);\n\n" );
    
    for (i=0;i<nvars;i++){
        if (vars[i].for_grad){
            if (vars[i].num_ele<maxsize){
                sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
                strcat(temp,ptemp);
                strcat(temp, "    ");
            }
            strcat(temp, "    l");
            strcat(temp, vars[i].name);
            strcat(temp, "=");
            strcat(temp, vars[i].name);
            strcat(temp, "[gid];\n");
        }
    }

    
    strcat(temp,"\n"
        "    for (freq=0;freq<NFREQS;freq++){\n"
        "        fact[freq].x =  DTNYQ*DT*cospi(2.0*freqs[freq]*nt/NTNYQ);\n"
        "        fact[freq].y = -DTNYQ*DT*sinpi(2.0*freqs[freq]*nt/NTNYQ);\n"
        "    }\n\n"
           );
        
        
    
    for (i=0;i<nvars;i++){
        if (vars[i].for_grad){
            if (vars[i].num_ele<maxsize){
                sprintf(ptemp,"    if (gid<%d){\n", vars[i].num_ele);
                strcat(temp,ptemp);
            }
            strcat(temp, "    #pragma unroll\n");
            strcat(temp, "    for (freq=0;freq<NFREQS;freq++){\n");
            strcat(temp, "        f");
            strcat(temp, vars[i].name);
            strcat(temp, "[gid+freq*gsize]+=fact[freq]*l");
            strcat(temp, vars[i].name);
            strcat(temp, ";\n");
            strcat(temp, "    }\n");
            if (vars[i].num_ele<maxsize){
                strcat(temp,"    }\n");
            }
        }
    }
    
    strcat(temp, "\n}");
    
    *source=temp;
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    return state;
    
}

int kernel_init_gradsrc(int NDIM, const char ** source){
    
    int state=0;
    
    char * temp=NULL;
    GMALLOC(temp, sizeof(char)*MAX_KERN_STR);
    
    strcat(temp,
           "__kernel void init_gradsrc(__global float *gradsrc)\n"
           "{\n\n"
           "    int gid = get_global_id(0);\n"
           "    gradsrc[gid]=0.0;\n\n"
           "}");
    
    *source=temp;
    
    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    return state;
    
}

int kernel_fillbuff(int NDIM, char ** N_names, int local_off, int nvars, struct variable * vars, const char ** source, int upid, int in_out, int buff12){
    
    int state=0;
    int i;
    
    char * temp=NULL;
    GMALLOC(temp, sizeof(char)*MAX_KERN_STR);
    char * p=(char*)temp;
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<nvars;i++){
        if (vars[i].num_ele>maxsize){
            maxsize=vars[i].num_ele;
        }
    }
    
    if (in_out==0){
        strcat(temp, "__kernel void fill_transfer_buff_out(");
    }
    else{
        strcat(temp, "__kernel void fill_transfer_buff_in(");
    }
    for (i=0;i<nvars;i++){
        if (vars[i].to_comm==upid){
            strcat(temp, "__global float * ");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, "__global float * ");
            strcat(temp, vars[i].name);
            if (buff12==1)
                strcat(temp, "_buf1, ");
            else if (buff12==2)
                strcat(temp, "_buf2, ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    //Indice if local memory is used
    if (local_off==0){
        for (i=0;i<NDIM;i++){
            sprintf(ptemp,"    int gid%s=get_global_id(%d)+FDOH;\n",N_names[i],i );
            strcat(temp, ptemp);
        }
    }
    //if we use directly global memory, with 1 working dim
    else{
        strcat(temp,"    int gid = get_global_id(0);\n");
        sprintf(ptemp,"    int gid%s=get_global_id(%d)+FDOH;\n",N_names[i],i );
        strcat(temp, ptemp);
        
        for (i=0;i<NDIM;i++){
            if (i==0){

            }
        }
    }
    
//    strcat(temp,"    int gid = get_global_id(0);\n\n");
//    
//    
//    for (i=0;i<nvars;i++){
//        if (vars[i].num_ele<maxsize){
//            sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
//            strcat(temp,ptemp);
//            strcat(temp, "    ");
//        }
//        strcat(temp, "    ");
//        strcat(temp, vars[i].name);
//        strcat(temp, "[gid]=0;\n");
//    }
    
    strcat(temp, "\n}");
    
    
    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    *source=temp;
    
    return state;
    
}
