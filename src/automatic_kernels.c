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
            strcat(temp,"    if (gid<%d)\n");
            strcat(temp, "    ");
        }
        strcat(temp, "    ");
        strcat(temp, vars[i].name);
        strcat(temp, "out[gid]=0;\n");
    }
    
    strcat(temp, "\n}");
    
    
    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    *source=temp;
    
    return state;
    
}
