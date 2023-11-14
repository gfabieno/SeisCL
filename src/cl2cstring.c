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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <ctype.h>
#include <string.h>


int main(int argc, char **argv) {
    struct stat statbuf;
    char * filename=NULL, * program_source=NULL, *progname=NULL;
    FILE *fh=NULL;
    char ch;
    int ii;
    
    filename = argv[1];
    fh = fopen(filename, "rb");
    if (fh==NULL){
        
        fprintf(stderr,"Error: Could not open the file: %s \n", filename);
    }
    else{
        stat(filename, &statbuf);
        program_source = (char *) malloc(statbuf.st_size);
        fread(program_source, statbuf.st_size, 1, fh);
        fclose(fh);
    }
    
    filename = argv[2];
    fh = fopen(filename, "w");
    if (fh==NULL){
        
        fprintf(stderr,"Error: Could not open the file: %s \n", filename);
    }
    else{
        //fprintf(fh, "const char * ");
        fprintf(fh, "const char ");
        ii=0;
        while ( (filename[ii]!='.') & (filename[ii]!='\0') & ii<1000){
            fputc(filename[ii], fh);
            ii++;
        }
        //fprintf(fh, "_source = \"");
        fprintf(fh, "_source[] = \"");
        for (ii=0;ii<statbuf.st_size;ii++){
            ch = program_source[ii];
            if (ch >=0 && ch <=125){
                switch (ch) {
                    case '\"':
                        fputs("\\\"", fh);
                        break;
                    case '\'':
                        fputs("\\\'", fh);
                        break;
                    case '\\':
                        fputs("\\\\", fh);
                        break;
                    case '\a':
                        fputs("\\a", fh);
                        break;
                    case '\b':
                        fputs("\\b", fh);
                        break;
                    case '\n':
                        fputs("\\n\"\n\"", fh);
                        break;
                    case '\t':
                        fputs("\\t", fh);
                        break;
                    case '\0':
                        break;
                    case '\r':
                        break;
                    default:
                        fputc(ch, fh);
                }
            }
        }
        fprintf(fh, "\";  ");
    }
    fclose(fh);
    free(program_source);
    free(progname);
    return 0;
}


