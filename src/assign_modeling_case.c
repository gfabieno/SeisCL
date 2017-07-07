//
//  assign_modeling_case.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-06.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include "F.h"


//Assign parameters list depending on which case of modeling is desired
int assign_modeling_case(struct modcsts * m){
    
    int i,j;
    int state =0;
    
    int sizeparams=1;
    int sizevars=1;
    int sizebnd[10];
    for (i=0;i<m->numdim;i++){
        sizeparams*=m->N[i];
        sizevars*=m->N[i]+m->FDORDER;
    }
    for (i=0;i<m->numdim;i++){
        sizebnd[i]=1;
        for (j=0;j<m->numdim;j++){
            if (i!=j)
                sizebnd[i]*=m->N[j];
        }
    }
    
    m->ncsts=23;
    GMALLOC(m->csts, sizeof(struct constants)*m->ncsts)
    m->csts[0].name="taper";   m->csts[0].num_ele=m->nab;
    m->csts[1].name="K_z";      m->csts[1].num_ele=2*m->nab;
    m->csts[2].name="a_z";      m->csts[2].num_ele=2*m->nab;
    m->csts[3].name="b_z";      m->csts[3].num_ele=2*m->nab;
    m->csts[4].name="K_z_half"; m->csts[4].num_ele=2*m->nab;
    m->csts[5].name="a_z_half"; m->csts[5].num_ele=2*m->nab;
    m->csts[6].name="b_z_half"; m->csts[6].num_ele=2*m->nab;
    
    m->csts[7].name="K_x";      m->csts[7].num_ele=2*m->nab;
    m->csts[8].name="a_x";      m->csts[8].num_ele=2*m->nab;
    m->csts[9].name="b_x";      m->csts[9].num_ele=2*m->nab;
    m->csts[10].name="K_x_half"; m->csts[10].num_ele=2*m->nab;
    m->csts[11].name="a_x_half"; m->csts[11].num_ele=2*m->nab;
    m->csts[12].name="b_x_half"; m->csts[12].num_ele=2*m->nab;
    
    m->csts[13].name="K_y";      m->csts[13].num_ele=2*m->nab;
    m->csts[14].name="a_y";      m->csts[14].num_ele=2*m->nab;
    m->csts[15].name="b_y";      m->csts[15].num_ele=2*m->nab;
    m->csts[16].name="K_y_half"; m->csts[16].num_ele=2*m->nab;
    m->csts[17].name="a_y_half"; m->csts[17].num_ele=2*m->nab;
    m->csts[18].name="b_y_half"; m->csts[18].num_ele=2*m->nab;
    m->csts[19].name="FL"; m->csts[19].num_ele=m->L; m->csts[19].to_read="/FL";
    m->csts[20].name="eta"; m->csts[20].num_ele=m->L;
    m->csts[21].name="gradfreqs"; m->csts[21].num_ele=m->nfreqs; m->csts[21].to_read="/gradfreqs";
    m->csts[22].name="gradfreqsn"; m->csts[22].num_ele=m->nfreqs;
    
    if (m->L>0){
        m->csts[19].active=1;
        m->csts[20].active=1;
    }
    if (m->gradout && m->back_prop_type==2){
        m->csts[21].active=1;
        m->csts[22].active=1;
    }
    if (m->abs_type==2){
        m->csts[0].active=1;
    }
    else if (m->abs_type==1){
        for (i=1;i<19;i++){
            m->csts[i].active=1;
        }
        if (m->ND!=3){
            for (i=13;i<19;i++){
                m->csts[i].active=0;
            }
        }
    }
        
    
    //Define parameters and variables
    {
        if (m->ND==3 && m->L>0){
            m->nparams=14;
            
            GMALLOC(m->params, sizeof(struct parameter)*m->nparams)
            m->params[0].name="mu";        m->params[0].to_read="/mu";
            m->params[1].name="M";         m->params[1].to_read="/M";
            m->params[2].name="rho";       m->params[2].to_read="/rho";
            m->params[3].name="taup";      m->params[3].to_read="/taup";
            m->params[4].name="taus";      m->params[4].to_read="/taus";
            m->params[5].name="rip";         
            m->params[6].name="rjp";         
            m->params[7].name="rkp";         
            m->params[8].name="muipjp";
            m->params[9].name="mujpkp";
            m->params[10].name="muipkp";
            m->params[11].name="tausipjp";  
            m->params[12].name="tausjpkp";  
            m->params[13].name="tausipkp"; 


 
            m->nvars=15;
            if (m->abs_type==2)
                m->nparams+=18;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars)
            m->vars[0].name="vx";   m->vars[0].num_ele=sizevars;
            m->vars[1].name="vy";   m->vars[1].num_ele=sizevars;
            m->vars[2].name="vz";   m->vars[2].num_ele=sizevars;
            m->vars[3].name="sxx";  m->vars[3].num_ele=sizevars;
            m->vars[4].name="syy";  m->vars[4].num_ele=sizevars;
            m->vars[5].name="szz";  m->vars[5].num_ele=sizevars;
            m->vars[6].name="sxy";  m->vars[6].num_ele=sizevars;
            m->vars[7].name="sxz";  m->vars[7].num_ele=sizevars;
            m->vars[8].name="syz";  m->vars[8].num_ele=sizevars;
            m->vars[9].name="rxx";  m->vars[9].num_ele=sizevars*m->L;
            m->vars[10].name="ryy"; m->vars[10].num_ele=sizevars*m->L;
            m->vars[11].name="rzz"; m->vars[11].num_ele=sizevars*m->L;
            m->vars[12].name="rxy"; m->vars[12].num_ele=sizevars*m->L;
            m->vars[13].name="rxz"; m->vars[13].num_ele=sizevars*m->L;
            m->vars[14].name="ryz"; m->vars[14].num_ele=sizevars*m->L;
            
            if (m->abs_type==2){
                m->vars[15].name="psi_sxx_x";   m->vars[15].num_ele=sizebnd[2];
                m->vars[16].name="psi_sxy_x";   m->vars[16].num_ele=sizebnd[2];
                m->vars[17].name="psi_sxz_x";   m->vars[17].num_ele=sizebnd[2];
                m->vars[18].name="psi_szz_z";   m->vars[18].num_ele=sizebnd[0];
                m->vars[19].name="psi_sxz_z";   m->vars[19].num_ele=sizebnd[0];
                m->vars[20].name="psi_syz_z";   m->vars[20].num_ele=sizebnd[0];
                m->vars[21].name="psi_vx_x";    m->vars[21].num_ele=sizebnd[2];
                m->vars[22].name="psi_vy_x";    m->vars[22].num_ele=sizebnd[2];
                m->vars[23].name="psi_vz_x";    m->vars[23].num_ele=sizebnd[2];
                m->vars[24].name="psi_vx_y";    m->vars[24].num_ele=sizebnd[1];
                m->vars[25].name="psi_vy_y";    m->vars[25].num_ele=sizebnd[1];
                m->vars[26].name="psi_vz_y";    m->vars[26].num_ele=sizebnd[1];
                m->vars[27].name="psi_vx_z";    m->vars[27].num_ele=sizebnd[0];
                m->vars[28].name="psi_vy_z";    m->vars[28].num_ele=sizebnd[0];
                m->vars[29].name="psi_vz_z";    m->vars[29].num_ele=sizebnd[0];

            }
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars)
            m->trans_vars[0].name="p"; m->vars[0].num_ele=sizevars;
            
        }
        else if (m->ND==3 && m->L==0){
            m->nparams=9;
            GMALLOC(m->params, sizeof(struct parameter)*m->nparams)
            m->params[0].name="mu";    m->params[0].to_read="/mu";
            m->params[1].name="M";     m->params[1].to_read="/M";
            m->params[2].name="rho";   m->params[2].to_read="/rho";
            m->params[3].name="rip";    
            m->params[4].name="rjp";    
            m->params[5].name="rkp";    
            m->params[6].name="uipjp";  
            m->params[7].name="ujpkp";  
            m->params[8].name="uipkp";  
            
            m->nvars=9;
            if (m->abs_type==2)
                m->nparams+=18;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars)
            m->vars[0].name="vx";   m->vars[0].num_ele=sizevars;
            m->vars[1].name="vy";   m->vars[1].num_ele=sizevars;
            m->vars[2].name="vz";   m->vars[2].num_ele=sizevars;
            m->vars[3].name="sxx";  m->vars[3].num_ele=sizevars;
            m->vars[4].name="syy";  m->vars[4].num_ele=sizevars;
            m->vars[5].name="szz";  m->vars[5].num_ele=sizevars;
            m->vars[6].name="sxy";  m->vars[6].num_ele=sizevars;
            m->vars[7].name="sxz";  m->vars[7].num_ele=sizevars;
            m->vars[8].name="syz";  m->vars[8].num_ele=sizevars;
            
            if (m->abs_type==2){
                m->vars[9].name="psi_sxx_x";   m->vars[9].num_ele=sizebnd[2];
                m->vars[10].name="psi_sxy_x";   m->vars[10].num_ele=sizebnd[2];
                m->vars[11].name="psi_sxz_x";   m->vars[11].num_ele=sizebnd[2];
                m->vars[12].name="psi_syy_y";   m->vars[12].num_ele=sizebnd[1];
                m->vars[13].name="psi_sxy_y";   m->vars[13].num_ele=sizebnd[1];
                m->vars[14].name="psi_syz_y";   m->vars[14].num_ele=sizebnd[1];
                m->vars[15].name="psi_szz_z";   m->vars[15].num_ele=sizebnd[0];
                m->vars[16].name="psi_sxz_z";   m->vars[16].num_ele=sizebnd[0];
                m->vars[17].name="psi_syz_z";   m->vars[17].num_ele=sizebnd[0];
                m->vars[18].name="psi_vx_x";    m->vars[18].num_ele=sizebnd[2];
                m->vars[19].name="psi_vy_x";    m->vars[19].num_ele=sizebnd[2];
                m->vars[20].name="psi_vz_x";    m->vars[20].num_ele=sizebnd[2];
                m->vars[21].name="psi_vx_y";    m->vars[21].num_ele=sizebnd[1];
                m->vars[22].name="psi_vy_y";    m->vars[22].num_ele=sizebnd[1];
                m->vars[23].name="psi_vz_y";    m->vars[23].num_ele=sizebnd[1];
                m->vars[24].name="psi_vx_z";    m->vars[24].num_ele=sizebnd[0];
                m->vars[25].name="psi_vy_z";    m->vars[25].num_ele=sizebnd[0];
                m->vars[26].name="psi_vz_z";    m->vars[26].num_ele=sizebnd[0];
                
            }
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars)
            m->trans_vars[0].name="p"; m->vars[0].num_ele=sizevars;
        }
        else if (m->ND==2 && m->L>0){
            m->nparams=9;
            GMALLOC(m->params, sizeof(struct parameter)*m->nparams)
            m->params[0].name="mu";       m->params[0].to_read="/mu";
            m->params[1].name="M";        m->params[1].to_read="/M";
            m->params[2].name="rho";      m->params[2].to_read="/rho";
            m->params[3].name="taup";     m->params[3].to_read="/taup";
            m->params[4].name="taus";     m->params[4].to_read="/taus";
            m->params[5].name="rip";       
            m->params[6].name="rkp";       
            m->params[7].name="uipkp";     
            m->params[8].name="tausipkp";  

 
            m->nvars=8;
            if (m->abs_type==2)
                m->nparams+=8;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars)
            m->vars[0].name="vx";   m->vars[0].num_ele=sizevars;
            m->vars[1].name="vz";   m->vars[1].num_ele=sizevars;
            m->vars[2].name="sxx";  m->vars[2].num_ele=sizevars;
            m->vars[3].name="szz";  m->vars[3].num_ele=sizevars;
            m->vars[4].name="sxz";  m->vars[4].num_ele=sizevars;
            m->vars[5].name="rxx";  m->vars[5].num_ele=sizevars*m->L;
            m->vars[6].name="rzz";  m->vars[6].num_ele=sizevars*m->L;
            m->vars[7].name="rxz";  m->vars[7].num_ele=sizevars*m->L;
            
            if (m->abs_type==2){
                m->vars[8].name="psi_sxx_x";   m->vars[8].num_ele=sizebnd[1];
                m->vars[9].name="psi_sxz_x";   m->vars[9].num_ele=sizebnd[1];
                m->vars[10].name="psi_szz_z";   m->vars[10].num_ele=sizebnd[0];
                m->vars[11].name="psi_sxz_z";   m->vars[11].num_ele=sizebnd[0];
                m->vars[12].name="psi_vx_x";    m->vars[12].num_ele=sizebnd[1];
                m->vars[13].name="psi_vz_x";    m->vars[13].num_ele=sizebnd[1];
                m->vars[14].name="psi_vx_z";    m->vars[14].num_ele=sizebnd[0];
                m->vars[15].name="psi_vz_z";    m->vars[15].num_ele=sizebnd[0];
                
            }
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars)
            m->trans_vars[0].name="p"; m->vars[0].num_ele=sizevars;
        }
        else if (m->ND==2 && m->L==0){
            m->nparams=6;

            GMALLOC(m->params, sizeof(struct parameter)*m->nparams)
            m->params[0].name="mu";    m->params[0].to_read="/mu";
            m->params[1].name="M";     m->params[1].to_read="/M";
            m->params[2].name="rho";   m->params[2].to_read="/rho";
            m->params[3].name="rip";    
            m->params[4].name="rkp";    
            m->params[5].name="uipkp";  
            
            m->nvars=5;
            if (m->abs_type==2)
                m->nparams+=8;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars)
            m->vars[0].name="vx";   m->vars[0].num_ele=sizevars;
            m->vars[1].name="vz";   m->vars[1].num_ele=sizevars;
            m->vars[2].name="sxx";  m->vars[2].num_ele=sizevars;
            m->vars[3].name="szz";  m->vars[3].num_ele=sizevars;
            m->vars[4].name="sxz";  m->vars[4].num_ele=sizevars;
            

            if (m->abs_type==2){
                m->vars[5].name="psi_sxx_x";   m->vars[5].num_ele=sizebnd[1];
                m->vars[6].name="psi_sxz_x";   m->vars[6].num_ele=sizebnd[1];
                m->vars[7].name="psi_szz_z";   m->vars[7].num_ele=sizebnd[0];
                m->vars[8].name="psi_sxz_z";   m->vars[8].num_ele=sizebnd[0];
                m->vars[9].name="psi_vx_x";    m->vars[9].num_ele=sizebnd[1];
                m->vars[10].name="psi_vz_x";    m->vars[10].num_ele=sizebnd[1];
                m->vars[11].name="psi_vx_z";    m->vars[11].num_ele=sizebnd[0];
                m->vars[12].name="psi_vz_z";    m->vars[12].num_ele=sizebnd[0];
                
            }
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars)
            m->trans_vars[0].name="p"; m->vars[0].num_ele=sizevars;
        }
        else if (m->ND==21 && m->L>0){
            m->nparams=7;

            GMALLOC(m->params, sizeof(struct parameter)*m->nparams)
            m->params[0].name="mu";        m->params[0].to_read="/mu";
            m->params[1].name="rho";      m->params[1].to_read="/rho";
            m->params[2].name="taus";     m->params[2].to_read="/taus";
            m->params[3].name="rip";       
            m->params[4].name="rkp";       
            m->params[5].name="uipkp";     
            m->params[6].name="tausipkp";  

            
            m->nvars=5;
            if (m->abs_type==2)
                m->nparams+=4;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars)
            m->vars[0].name="vy";   m->vars[0].num_ele=sizevars;
            m->vars[1].name="sxy";  m->vars[1].num_ele=sizevars;
            m->vars[2].name="syz";  m->vars[2].num_ele=sizevars;
            m->vars[3].name="rxy";  m->vars[3].num_ele=sizevars*m->L;
            m->vars[4].name="ryz";  m->vars[4].num_ele=sizevars*m->L;
            
            if (m->abs_type==2){
                m->vars[5].name="psi_sxy_x";   m->vars[5].num_ele=sizebnd[1];
                m->vars[6].name="psi_sxy_z";   m->vars[6].num_ele=sizebnd[0];
                m->vars[7].name="psi_vy_x";    m->vars[7].num_ele=sizebnd[1];
                m->vars[8].name="psi_vy_z";    m->vars[8].num_ele=sizebnd[0];
               
            }
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars)
            m->trans_vars[0].name="p"; m->vars[0].num_ele=sizevars;
            
        }
        else if (m->ND==21 && m->L==0){
            m->nparams=5;

            GMALLOC(m->params, sizeof(struct parameter)*m->nparams)
            m->params[0].name="mu";     m->params[0].to_read="/mu";
            m->params[1].name="rho";   m->params[1].to_read="/rho";
            m->params[2].name="rip";    
            m->params[3].name="rkp";    
            m->params[4].name="uipkp";  

            m->nvars=3;
            if (m->abs_type==2)
                m->nparams+=4;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars)
            m->vars[0].name="vy";   m->vars[0].num_ele=sizevars;
            m->vars[1].name="sxy";  m->vars[1].num_ele=sizevars;
            m->vars[2].name="syz";  m->vars[2].num_ele=sizevars;

            if (m->abs_type==2){
                m->vars[3].name="psi_sxy_x";   m->vars[3].num_ele=sizebnd[1];
                m->vars[4].name="psi_sxy_z";   m->vars[4].num_ele=sizebnd[0];
                m->vars[5].name="psi_vy_x";    m->vars[5].num_ele=sizebnd[1];
                m->vars[6].name="psi_vy_z";    m->vars[6].num_ele=sizebnd[0];
                
            }
            
        }
    }

    //Assign the number of elements of the parameters
    for (i=0;i<m->nparams;i++){
        m->params[i].num_ele=sizeparams;
    }
    
    //Check to name of variable to be read depending on the chosen parametrization
    {

        if (m->param_type==2){
            for (i=0;i<m->nparams;i++){
                if (strcmp(m->params[i].name,"M")==0)
                    m->params[i].to_read="/Ip";
                if (strcmp(m->params[i].name,"mu")==0)
                    m->params[i].to_read="/Is";
            }
        }
        else if (m->param_type==3){
            if (m->L==0) {state=1;fprintf(stderr, "Viscoelastic modeling is required for param_type 3\n");};
            for (i=0;i<m->nparams;i++){
                if (strcmp(m->params[i].name,"M")==0)
                    m->params[i].to_read="/vpR";
                if (strcmp(m->params[i].name,"mu")==0)
                    m->params[i].to_read="/vsR";
                if (strcmp(m->params[i].name,"taup")==0)
                    m->params[i].to_read="/vpI";
                if (strcmp(m->params[i].name,"taus")==0)
                    m->params[i].to_read="/vsI";
            }
        }
        else {
            m->param_type=0;
            for (i=0;i<m->nparams;i++){
                if (strcmp(m->params[i].name,"M")==0)
                    m->params[i].to_read="/vp";
                if (strcmp(m->params[i].name,"mu")==0)
                    m->params[i].to_read="/vs";
            }
        }
    }
    
    //Flags variables to output
    {
        if (m->seisout==1){
            for (i=0;i<m->nvars;i++){
                if (strcmp(m->vars[i].name,"vx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vz")==0)
                    m->vars[i].to_output=1;
            }
        }
        if (m->seisout==2){
            for (i=0;i<m->ntvars;i++){
                if (strcmp(m->trans_vars[i].name,"p")==0)
                    m->trans_vars[i].to_output=1;
            }
        }
        if (m->seisout==3){
            for (i=0;i<m->nvars;i++){
                if (strcmp(m->vars[i].name,"sxx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"szz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"sxx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"szz")==0)
                    m->vars[i].to_output=1;
            }
        }
        if (m->seisout==4){
            for (i=0;i<m->nvars;i++){
                if (strcmp(m->vars[i].name,"vx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"sxx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"szz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"sxx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"szz")==0)
                    m->vars[i].to_output=1;
            }
            for (i=0;i<m->ntvars;i++){
                if (strcmp(m->trans_vars[i].name,"p")==0)
                    m->trans_vars[i].to_output=1;
            }
        }
    }

    //Define the update kernels
    m->nupdates=2;
    GMALLOC(m->update_names, m->nupdates=2*sizeof(char *))
    m->update_names[0]="v";
    m->update_names[0]="s";
    
    
    
    //Allocate memory of constants
    for (i=0;i<m->ncsts;i++){
        if (m->csts[i].active)
            GMALLOC(m->csts[i].gl_cst, sizeof(float)*m->csts[i].num_ele)
            }
    
    
    //Allocate memory of parameters
    for (i=0;i<m->nparams;i++){
        GMALLOC(m->params[i].gl_param, sizeof(float)*m->params[i].num_ele)
        if (m->gradout && m->params[i].to_read){
            m->params[i].to_grad=1;
            GMALLOC(m->params[i].gl_grad, sizeof(double)*m->params[i].num_ele)
            memset((void*)m->params[i].gl_grad, 0, sizeof(double)*m->params[i].num_ele);
            if (m->Hout){
                GMALLOC(m->params[i].gl_H, sizeof(double)*m->params[i].num_ele)
                memset((void*)m->params[i].gl_H, 0, sizeof(double)*m->params[i].num_ele);
            }
        }
    }
    
    //Allocate memory of variables
    for (i=0;i<m->nvars;i++){
        if (m->vars[i].to_output){
            alloc_seismo(&m->vars[i].gl_varout, m->ns, m->allng, m->NT, m->src_recs.nrec);
            if (m->movout>0){
                GMALLOC(m->vars[i].gl_mov,sizeof(float)*m->ns*m->vars[i].num_ele*m->NT/m->movout);
                memset((void*)m->vars[i].gl_mov, 0, sizeof(float)*m->ns*m->vars[i].num_ele*m->NT/m->movout);
            }
            if (m->gradout && m->back_prop_type==2){
                GMALLOC(m->vars[i].gl_fvar, sizeof(cl_float2)*m->vars[i].num_ele)
                memset((void*)m->vars[i].gl_fvar, 0, sizeof(cl_float2)*m->vars[i].num_ele);
            }
        }
    }
    
    if (m->gradsrcout==1){
        GMALLOC(m->src_recs.gradsrc,sizeof(float*)*m->ns)
        if (!state) memset((void*)m->src_recs.gradsrc, 0, sizeof(float*)*m->ns);
        GMALLOC(m->src_recs.gradsrc[0],sizeof(float)*m->allns*m->NT)
        for (i=1;i<m->ns;i++){
            m->src_recs.gradsrc[i]=m->src_recs.gradsrc[i-1]+m->src_recs.nsrc[i-1]*m->NT;
        }
    }
    
    return state;
}
