//
//  string_parse.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-08.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include <F.h>



int split (const char *str, char c, char ***arr)
{
    int count = 1;
    int token_len = 1;
    int i = 0;
    char *p;
    char *t;
    
    p = (char*)str;
    while (*p != '\0')
    {
        if (*p == c)
            count++;
        p++;
    }
    
    *arr = (char**) malloc(sizeof(char*) * count);
    if (*arr == NULL)
        exit(1);
    
    p = (char*)str;
    while (*p != '\0')
    {
        if (*p == c)
        {
            (*arr)[i] = (char*) malloc( sizeof(char) * token_len );
            if ((*arr)[i] == NULL)
                exit(1);
            
            token_len = 0;
            i++;
        }
        p++;
        token_len++;
    }
    (*arr)[i] = (char*) malloc( sizeof(char) * token_len );
    if ((*arr)[i] == NULL)
        exit(1);
    
    i = 0;
    p = (char*)str;
    t = ((*arr)[i]);
    while (*p != '\0')
    {
        if (*p != c && *p != '\0')
        {
            *t = *p;
            t++;
        }
        else
        {
            *t = '\0';
            i++;
            t = ((*arr)[i]);
        }
        p++;
    }
    *t='\0';
    
    return count;
}



char * extract_name(const char *str){
    
    char * output=NULL;
    
    int len=(int)strlen(str);
    
    char *p1=(char*)str+len-1;
    
    while (  !(isalnum(*p1) || *p1=='_') && p1>=str){
        
        p1--;
    }
    
    char *p2=p1;
    
    while (  (isalnum(*p2) || *p2=='_') && p2>=str){
        
        p2--;
    }
    p2++;
    
    output = malloc(sizeof(str)*(p1-p2+1));
    
    sprintf(output,"%.*s", (int)(p1-p2+1), p2);
    
    return output;
}


int extract_args(const char *str, char *name, char *** argnames, int * ninputs){
    
    int c = 0;
    char **arr = NULL;
    char * args = NULL;
    char del2[2] = ")";
    
    
    char del1[100];
    
    sprintf(del1,"__kernel void %s(", name);

    char * strbeg = strstr(str, del1);
    
    if (!strbeg){
        free(del1);
        fprintf(stderr, "Could not extract kernel arguments of %s\n",name);
        return 1;
    }
    
    char * strend = strstr(strbeg, del2);
    
    if (!strend){
        free(del1);
        fprintf(stderr, "Could not extract kernel arguments of %s\n",name);
        return 1;
    }

    
    args = malloc(sizeof(str)*(strend- strbeg-strlen(del1)+1 ));
    
    sprintf(args,"%.*s", (int)(strend- strbeg-strlen(del1)), strbeg + strlen(del1));


    
    c = split(args, ',', &arr);
    
    (*argnames)=malloc(c*sizeof(char *));
    
    
    for (int i = 0; i < c; i++){
        (*argnames)[i]=extract_name(arr[i]);
        free(arr[i]);
    }
    *ninputs=c;
    free(arr);
    free(args);

    return 0;
}
