//
//  assign_modeling_case.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-06.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include "F.h"

/*Loading files autmatically created by the makefile that contain the *.cl kernels in a c string.
 This way, no .cl file need to be read and there is no need to be in the executable directory to execute SeisCL.*/
#include "initialize.hcl"
#include "residuals.hcl"
#include "savebnd2D.hcl"
#include "savebnd3D.hcl"
#include "savefreqs.hcl"
#include "surface2D.hcl"
#include "surface2D_SH.hcl"
#include "surface3D.hcl"
#include "update_adjs2D.hcl"
#include "update_adjs2D_SH.hcl"
#include "update_adjs3D.hcl"
#include "update_adjv2D.hcl"
#include "update_adjv2D_SH.hcl"
#include "update_adjv3D.hcl"
#include "update_s2D.hcl"
#include "update_s2D_SH.hcl"
#include "update_s3D.hcl"
#include "update_v2D.hcl"
#include "update_v2D_SH.hcl"
#include "update_v3D.hcl"
#include "update_v_CPML.hcl"
//#include "varsout.hcl"
//#include "fill_transfer_buff_s.hcl"
//#include "fill_transfer_buff_v.hcl"


//Assign parameters list depending on which case of modeling is desired
int assign_modeling_case(struct modcsts * m){
    
    int i,j;
    int state =0;
    
    int sizepars=1;
    int sizevars=1;
    int sizebnd[10];
    for (i=0;i<m->NDIM;i++){
        sizepars*=m->N[i];
        sizevars*=m->N[i]+m->FDORDER;
    }
    for (i=0;i<m->NDIM;i++){
        sizebnd[i]=1;
        for (j=0;j<m->NDIM;j++){
            if (i!=j)
                sizebnd[i]*=m->N[j];
        }
    }
    
    //Arrays of constants size on all devices
    {
        m->ncsts=23;
        GMALLOC(m->csts, sizeof(struct constants)*m->ncsts);
        m->csts[0].name="taper";   m->csts[0].num_ele=m->NAB;
        m->csts[1].name="K_z";      m->csts[1].num_ele=2*m->NAB;
        m->csts[2].name="a_z";      m->csts[2].num_ele=2*m->NAB;
        m->csts[3].name="b_z";      m->csts[3].num_ele=2*m->NAB;
        m->csts[4].name="K_z_half"; m->csts[4].num_ele=2*m->NAB;
        m->csts[5].name="a_z_half"; m->csts[5].num_ele=2*m->NAB;
        m->csts[6].name="b_z_half"; m->csts[6].num_ele=2*m->NAB;
        
        m->csts[7].name="K_x";      m->csts[7].num_ele=2*m->NAB;
        m->csts[8].name="a_x";      m->csts[8].num_ele=2*m->NAB;
        m->csts[9].name="b_x";      m->csts[9].num_ele=2*m->NAB;
        m->csts[10].name="K_x_half"; m->csts[10].num_ele=2*m->NAB;
        m->csts[11].name="a_x_half"; m->csts[11].num_ele=2*m->NAB;
        m->csts[12].name="b_x_half"; m->csts[12].num_ele=2*m->NAB;
        
        m->csts[13].name="K_y";      m->csts[13].num_ele=2*m->NAB;
        m->csts[14].name="a_y";      m->csts[14].num_ele=2*m->NAB;
        m->csts[15].name="b_y";      m->csts[15].num_ele=2*m->NAB;
        m->csts[16].name="K_y_half"; m->csts[16].num_ele=2*m->NAB;
        m->csts[17].name="a_y_half"; m->csts[17].num_ele=2*m->NAB;
        m->csts[18].name="b_y_half"; m->csts[18].num_ele=2*m->NAB;
        m->csts[19].name="FL"; m->csts[19].num_ele=m->L; m->csts[19].to_read="/FL";
        m->csts[20].name="eta"; m->csts[20].num_ele=m->L;
        m->csts[21].name="gradfreqs"; m->csts[21].num_ele=m->NFREQS; m->csts[21].to_read="/gradfreqs";
        m->csts[22].name="gradfreqsn"; m->csts[22].num_ele=m->NFREQS;
        
        if (m->L>0){
            m->csts[19].active=1;
            m->csts[20].active=1;
        }
        if (m->GRADOUT && m->BACK_PROP_TYPE==2){
            m->csts[21].active=1;
            m->csts[22].active=1;
        }
        if (m->ABS_TYPE==2){
            m->csts[0].active=1;
        }
        else if (m->ABS_TYPE==1){
            for (i=1;i<19;i++){
                m->csts[i].active=1;
            }
            if (m->ND!=3){
                for (i=13;i<19;i++){
                    m->csts[i].active=0;
                }
            }
        }
    }
    
    //Define the update kernels
    m->nupdates=2;
    GMALLOC(m->ups_f, m->nupdates*sizeof(struct update));
    m->ups_f[0].name="update_v";
    m->ups_f[1].name="update_s";

    
    if (m->GRADOUT){
        GMALLOC(m->ups_adj, m->nupdates*sizeof(struct update));
        m->ups_adj[0].name="update_adjv";
        m->ups_adj[1].name="update_adjs";
    }



    

    //Define parameters and variables
    {
        if (m->ND==3 && m->L>0){

            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s3D_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s3D_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s3D_source);
            
            
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs3D_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs3D_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs3D_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface3D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd3D_source);
            }
            
            m->npars=14;
            
            GMALLOC(m->pars, sizeof(struct parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";        m->pars[0].to_read="/mu";
            m->pars[1].name="M";         m->pars[1].to_read="/M";
            m->pars[2].name="rho";       m->pars[2].to_read="/rho";
            m->pars[3].name="taup";      m->pars[3].to_read="/taup";
            m->pars[4].name="taus";      m->pars[4].to_read="/taus";
            m->pars[5].name="rip";
            m->pars[6].name="rjp";
            m->pars[7].name="rkp";
            m->pars[8].name="muipjp";
            m->pars[9].name="mujpkp";
            m->pars[10].name="muipkp";
            m->pars[11].name="tausipjp";
            m->pars[12].name="tausjpkp";
            m->pars[13].name="tausipkp";
            }
            
            m->nvars=15;
            if (m->ABS_TYPE==1)
                m->nvars+=18;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars);
            
            
            if (!state){
            m->vars[0].name="vx"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="vy"; m->vars[1].for_grad=1; m->vars[1].to_comm=1;
            m->vars[2].name="vz"; m->vars[2].for_grad=1; m->vars[2].to_comm=1;
            m->vars[3].name="sxx"; m->vars[3].for_grad=1; m->vars[3].to_comm=2;
            m->vars[4].name="syy"; m->vars[4].for_grad=1; m->vars[4].to_comm=2;
            m->vars[5].name="szz"; m->vars[5].for_grad=1; m->vars[5].to_comm=2;
            m->vars[6].name="sxy"; m->vars[6].for_grad=1; m->vars[6].to_comm=2;
            m->vars[7].name="sxz"; m->vars[7].for_grad=1; m->vars[7].to_comm=2;
            m->vars[8].name="syz"; m->vars[8].for_grad=1; m->vars[8].to_comm=2;
            m->vars[9].name="rxx"; m->vars[9].for_grad=1; 
            m->vars[10].name="ryy"; m->vars[10].for_grad=1; 
            m->vars[11].name="rzz"; m->vars[11].for_grad=1; 
            m->vars[12].name="rxy"; m->vars[12].for_grad=1; 
            m->vars[13].name="rxz"; m->vars[13].for_grad=1; 
            m->vars[14].name="ryz"; m->vars[14].for_grad=1; 
            
            if (m->ABS_TYPE==1){
                m->vars[15].name="psi_sxx_x"; m->vars[15].for_grad=0; 
                m->vars[16].name="psi_sxy_x"; m->vars[16].for_grad=0; 
                m->vars[17].name="psi_sxz_x"; m->vars[17].for_grad=0; 
                m->vars[18].name="psi_syy_y"; m->vars[18].for_grad=0; 
                m->vars[19].name="psi_sxy_y"; m->vars[19].for_grad=0; 
                m->vars[20].name="psi_syz_y"; m->vars[20].for_grad=0; 
                m->vars[21].name="psi_szz_z"; m->vars[21].for_grad=0; 
                m->vars[22].name="psi_sxz_z"; m->vars[22].for_grad=0; 
                m->vars[23].name="psi_syz_z"; m->vars[23].for_grad=0; 
                m->vars[24].name="psi_vx_x"; m->vars[24].for_grad=0; 
                m->vars[25].name="psi_vy_x"; m->vars[25].for_grad=0; 
                m->vars[26].name="psi_vz_x"; m->vars[26].for_grad=0; 
                m->vars[27].name="psi_vx_y"; m->vars[27].for_grad=0; 
                m->vars[28].name="psi_vy_y"; m->vars[28].for_grad=0; 
                m->vars[29].name="psi_vz_y"; m->vars[29].for_grad=0; 
                m->vars[30].name="psi_vx_z"; m->vars[30].for_grad=0; 
                m->vars[31].name="psi_vy_z"; m->vars[31].for_grad=0; 
                m->vars[32].name="psi_vz_z"; m->vars[32].for_grad=0; 
                
            }}
            
           
            
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars);
            m->trans_vars[0].name="p";
            
        }
        else if (m->ND==3 && m->L==0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s3D_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s3D_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s3D_source);
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs3D_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs3D_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs3D_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface3D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd3D_source);
            }
            
            
            m->npars=9;
            GMALLOC(m->pars, sizeof(struct parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";    m->pars[0].to_read="/mu";
            m->pars[1].name="M";     m->pars[1].to_read="/M";
            m->pars[2].name="rho";   m->pars[2].to_read="/rho";
            m->pars[3].name="rip";
            m->pars[4].name="rjp";
            m->pars[5].name="rkp";
            m->pars[6].name="muipjp";
            m->pars[7].name="mujpkp";
            m->pars[8].name="muipkp";
            }
            
            m->nvars=9;
            if (m->ABS_TYPE==1)
                m->nvars+=18;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars);
            if (!state){
            m->vars[0].name="vx"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="vy"; m->vars[1].for_grad=1; m->vars[1].to_comm=1;
            m->vars[2].name="vz"; m->vars[2].for_grad=1; m->vars[2].to_comm=1;
            m->vars[3].name="sxx"; m->vars[3].for_grad=1; m->vars[3].to_comm=2;
            m->vars[4].name="syy"; m->vars[4].for_grad=1; m->vars[4].to_comm=2;
            m->vars[5].name="szz"; m->vars[5].for_grad=1; m->vars[5].to_comm=2;
            m->vars[6].name="sxy"; m->vars[6].for_grad=1; m->vars[6].to_comm=2;
            m->vars[7].name="sxz"; m->vars[7].for_grad=1; m->vars[7].to_comm=2;
            m->vars[8].name="syz"; m->vars[8].for_grad=1; m->vars[8].to_comm=2;
            
            if (m->ABS_TYPE==1){
                m->vars[9].name="psi_sxx_x"; m->vars[9].for_grad=0; 
                m->vars[10].name="psi_sxy_x"; m->vars[10].for_grad=0; 
                m->vars[11].name="psi_sxz_x"; m->vars[11].for_grad=0; 
                m->vars[12].name="psi_syy_y"; m->vars[12].for_grad=0; 
                m->vars[13].name="psi_sxy_y"; m->vars[13].for_grad=0; 
                m->vars[14].name="psi_syz_y"; m->vars[14].for_grad=0; 
                m->vars[15].name="psi_szz_z"; m->vars[15].for_grad=0; 
                m->vars[16].name="psi_sxz_z"; m->vars[16].for_grad=0; 
                m->vars[17].name="psi_syz_z"; m->vars[17].for_grad=0; 
                m->vars[18].name="psi_vx_x"; m->vars[18].for_grad=0; 
                m->vars[19].name="psi_vy_x"; m->vars[19].for_grad=0; 
                m->vars[20].name="psi_vz_x"; m->vars[20].for_grad=0; 
                m->vars[21].name="psi_vx_y"; m->vars[21].for_grad=0; 
                m->vars[22].name="psi_vy_y"; m->vars[22].for_grad=0; 
                m->vars[23].name="psi_vz_y"; m->vars[23].for_grad=0; 
                m->vars[24].name="psi_vx_z"; m->vars[24].for_grad=0; 
                m->vars[25].name="psi_vy_z"; m->vars[25].for_grad=0; 
                m->vars[26].name="psi_vz_z"; m->vars[26].for_grad=0; 
                
            }}
            
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars);
            m->trans_vars[0].name="p";
        }
        else if (m->ND==2 && m->L>0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s2D_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s2D_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s2D_source);
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs2D_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs2D_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs2D_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface2D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd2D_source);
            }
            
            m->npars=9;
            GMALLOC(m->pars, sizeof(struct parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";       m->pars[0].to_read="/mu";
            m->pars[1].name="M";        m->pars[1].to_read="/M";
            m->pars[2].name="rho";      m->pars[2].to_read="/rho";
            m->pars[3].name="taup";     m->pars[3].to_read="/taup";
            m->pars[4].name="taus";     m->pars[4].to_read="/taus";
            m->pars[5].name="rip";
            m->pars[6].name="rkp";
            m->pars[7].name="muipkp";
            m->pars[8].name="tausipkp";
            }
            
            m->nvars=8;
            if (m->ABS_TYPE==1)
                m->nvars+=8;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars);
            if (!state){
            m->vars[0].name="vx"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="vz"; m->vars[1].for_grad=1; m->vars[1].to_comm=1;
            m->vars[2].name="sxx"; m->vars[2].for_grad=1; m->vars[2].to_comm=2;
            m->vars[3].name="szz"; m->vars[3].for_grad=1; m->vars[3].to_comm=2;
            m->vars[4].name="sxz"; m->vars[4].for_grad=1; m->vars[4].to_comm=2;
            m->vars[5].name="rxx"; m->vars[5].for_grad=1; 
            m->vars[6].name="rzz"; m->vars[6].for_grad=1; 
            m->vars[7].name="rxz"; m->vars[7].for_grad=1; 
            
            if (m->ABS_TYPE==1){
                m->vars[8].name="psi_sxx_x"; m->vars[8].for_grad=0; 
                m->vars[9].name="psi_sxz_x"; m->vars[9].for_grad=0; 
                m->vars[10].name="psi_szz_z"; m->vars[10].for_grad=0; 
                m->vars[11].name="psi_sxz_z"; m->vars[11].for_grad=0; 
                m->vars[12].name="psi_vx_x"; m->vars[12].for_grad=0; 
                m->vars[13].name="psi_vz_x"; m->vars[13].for_grad=0; 
                m->vars[14].name="psi_vx_z"; m->vars[14].for_grad=0; 
                m->vars[15].name="psi_vz_z"; m->vars[15].for_grad=0; 
                
            }}
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars);
            if (!state){m->trans_vars[0].name="p";}
        }
        else if (m->ND==2 && m->L==0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s2D_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s2D_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s2D_source);
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs2D_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs2D_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs2D_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface2D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd2D_source);
            }
            
            m->npars=6;
            
            GMALLOC(m->pars, sizeof(struct parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";    m->pars[0].to_read="/mu";
            m->pars[1].name="M";     m->pars[1].to_read="/M";
            m->pars[2].name="rho";   m->pars[2].to_read="/rho";
            m->pars[3].name="rip";
            m->pars[4].name="rkp";
            m->pars[5].name="muipkp";
            }
            
            m->nvars=5;
            if (m->ABS_TYPE==1)
                m->nvars+=8;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars);
            if (!state){
            m->vars[0].name="vx"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="vz"; m->vars[1].for_grad=1; m->vars[1].to_comm=1;
            m->vars[2].name="sxx"; m->vars[2].for_grad=1; m->vars[2].to_comm=2;
            m->vars[3].name="szz"; m->vars[3].for_grad=1; m->vars[3].to_comm=2;
            m->vars[4].name="sxz"; m->vars[4].for_grad=1; m->vars[4].to_comm=2;
            
            
            if (m->ABS_TYPE==1){
                m->vars[5].name="psi_sxx_x"; m->vars[5].for_grad=0; 
                m->vars[6].name="psi_sxz_x"; m->vars[6].for_grad=0; 
                m->vars[7].name="psi_szz_z"; m->vars[7].for_grad=0; 
                m->vars[8].name="psi_sxz_z"; m->vars[8].for_grad=0; 
                m->vars[9].name="psi_vx_x"; m->vars[9].for_grad=0; 
                m->vars[10].name="psi_vz_x"; m->vars[10].for_grad=0; 
                m->vars[11].name="psi_vx_z"; m->vars[11].for_grad=0; 
                m->vars[12].name="psi_vz_z"; m->vars[12].for_grad=0; 
                
            }}
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars);
            m->trans_vars[0].name="p";
        }
        else if (m->ND==21 && m->L>0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s2D_SH_source);
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs2D_SH_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface2D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd2D_source);
            }
            
            m->npars=7;
            
            GMALLOC(m->pars, sizeof(struct parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";        m->pars[0].to_read="/mu";
            m->pars[1].name="rho";      m->pars[1].to_read="/rho";
            m->pars[2].name="taus";     m->pars[2].to_read="/taus";
            m->pars[3].name="rip";
            m->pars[4].name="rkp";
            m->pars[5].name="muipkp";
            m->pars[6].name="tausipkp";
            }
            
            m->nvars=5;
            if (m->ABS_TYPE==1)
                m->vars+=4;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars);
            if (!state){
            m->vars[0].name="vy"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="sxy"; m->vars[1].for_grad=1; m->vars[1].to_comm=2;
            m->vars[2].name="syz"; m->vars[2].for_grad=1; m->vars[2].to_comm=2;
            m->vars[3].name="rxy"; m->vars[3].for_grad=1; 
            m->vars[4].name="ryz"; m->vars[4].for_grad=1; 
            
            if (m->ABS_TYPE==1){
                m->vars[5].name="psi_sxy_x"; m->vars[5].for_grad=0; 
                m->vars[6].name="psi_sxy_z"; m->vars[6].for_grad=0; 
                m->vars[7].name="psi_vy_x"; m->vars[7].for_grad=0; 
                m->vars[8].name="psi_vy_z"; m->vars[8].for_grad=0; 
                
            }}
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(struct variable)*m->ntvars);
            m->trans_vars[0].name="p";
            
        }
        else if (m->ND==21 && m->L==0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s2D_SH_source);

            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs2D_SH_source);

            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface2D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd2D_source);
            }
            
            m->npars=5;
            
            GMALLOC(m->pars, sizeof(struct parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";     m->pars[0].to_read="/mu";
            m->pars[1].name="rho";   m->pars[1].to_read="/rho";
            m->pars[2].name="rip";
            m->pars[3].name="rkp";
            m->pars[4].name="muipkp";
            }
            
            m->nvars=3;
            if (m->ABS_TYPE==1)
                m->nvars+=4;
            GMALLOC(m->vars, sizeof(struct variable)*m->nvars);
            if (!state){
            m->vars[0].name="vy"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="sxy"; m->vars[1].for_grad=1; m->vars[1].to_comm=2;
            m->vars[2].name="syz"; m->vars[2].for_grad=1; m->vars[2].to_comm=2;
            
            if (m->ABS_TYPE==1){
                m->vars[3].name="psi_sxy_x"; m->vars[3].for_grad=0; 
                m->vars[4].name="psi_sxy_z"; m->vars[4].for_grad=0;     
                m->vars[5].name="psi_vy_x"; m->vars[5].for_grad=0;      
                m->vars[6].name="psi_vy_z"; m->vars[6].for_grad=0;      
                
            }}
                
            
        }
    }
    


    
    //Create adjoint variables if necessary
    if (m->GRADOUT && m->BACK_PROP_TYPE==1){
        GMALLOC(m->vars_adj, sizeof(struct variable)*m->nvars);
        for (i=0;i<m->nvars;i++){
            m->vars_adj[i]=m->vars[i];
        }
    }

    //Assign dimensions name
    if (m->ND==3){
        m->N_names[0]="Z";
        m->N_names[1]="Y";
        m->N_names[2]="X";
    }
    else{
        m->N_names[0]="Z";
        m->N_names[1]="X";
    }

    //Assign the number of elements of the parameters
    if (!state){
    for (i=0;i<m->npars;i++){
        m->pars[i].num_ele=sizepars;
    }}
    __GUARD assign_var_size(m->N, m->NDIM, m->FDORDER, m->nvars, m->L, m->vars);
    
    //Check to name of variable to be read depending on the chosen paretrization
    if (!state){

        if (m->par_type==2){
            for (i=0;i<m->npars;i++){
                if (strcmp(m->pars[i].name,"M")==0)
                    m->pars[i].to_read="/Ip";
                if (strcmp(m->pars[i].name,"mu")==0)
                    m->pars[i].to_read="/Is";
            }
        }
        else if (m->par_type==3){
            if (m->L==0) {state=1;fprintf(stderr, "Viscoelastic modeling is required for par_type 3\n");};
            for (i=0;i<m->npars;i++){
                if (strcmp(m->pars[i].name,"M")==0)
                    m->pars[i].to_read="/vpR";
                if (strcmp(m->pars[i].name,"mu")==0)
                    m->pars[i].to_read="/vsR";
                if (strcmp(m->pars[i].name,"taup")==0)
                    m->pars[i].to_read="/vpI";
                if (strcmp(m->pars[i].name,"taus")==0)
                    m->pars[i].to_read="/vsI";
            }
        }
        else {
            m->par_type=0;
            for (i=0;i<m->npars;i++){
                if (strcmp(m->pars[i].name,"M")==0)
                    m->pars[i].to_read="/vp";
                if (strcmp(m->pars[i].name,"mu")==0)
                    m->pars[i].to_read="/vs";
            }
        }
    }
    
    //Flags variables to output
    if (!state){
        if (m->VARSOUT==1){
            for (i=0;i<m->nvars;i++){
                if (strcmp(m->vars[i].name,"vx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vz")==0)
                    m->vars[i].to_output=1;
            }
        }
        if (m->VARSOUT==2){
            for (i=0;i<m->ntvars;i++){
                if (strcmp(m->trans_vars[i].name,"p")==0)
                    m->trans_vars[i].to_output=1;
            }
        }
        if (m->VARSOUT==3){
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
        if (m->VARSOUT==4){
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

    //Allocate memory of constants
    for (i=0;i<m->ncsts;i++){
        if (m->csts[i].active)
            GMALLOC(m->csts[i].gl_cst, sizeof(float)*m->csts[i].num_ele);
    }
    
    //Allocate memory of parameters
    for (i=0;i<m->npars;i++){
        GMALLOC(m->pars[i].gl_par, sizeof(float)*m->pars[i].num_ele);
        if (m->GRADOUT && m->pars[i].to_read){
            m->pars[i].to_grad=1;
            GMALLOC(m->pars[i].gl_grad, sizeof(double)*m->pars[i].num_ele);
            if (m->HOUT){
                GMALLOC(m->pars[i].gl_H, sizeof(double)*m->pars[i].num_ele);
            }
        }
    }
    
    //Allocate memory of variables
    for (i=0;i<m->nvars;i++){
        if (m->vars[i].to_output){
            alloc_seismo(&m->vars[i].gl_varout, m);
            if (m->MOVOUT>0){
                GMALLOC(m->vars[i].gl_mov,sizeof(float)*m->ns*m->vars[i].num_ele*m->NT/m->MOVOUT);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==2){
                GMALLOC(m->vars[i].gl_fvar, sizeof(cl_float2)*m->vars[i].num_ele);
            }
        }
    }
    
    if (m->GRADSRCOUT==1){
        GMALLOC(m->src_recs.gradsrc,sizeof(float*)*m->ns);
        GMALLOC(m->src_recs.gradsrc[0],sizeof(float)*m->src_recs.allns*m->NT);
        for (i=1;i<m->ns;i++){
            m->src_recs.gradsrc[i]=m->src_recs.gradsrc[i-1]+m->src_recs.nsrc[i-1]*m->NT;
        }
    }
    
    

    
    return state;
}

int assign_var_size(int* N, int NDIM, int FDORDER, int numvar, int L, struct variable * vars){
    int i,j;
    int sizevars=1;
    int sizebnd[10];
    for (i=0;i<NDIM;i++){
       sizevars*=N[i]+FDORDER;
    }
    for (i=0;i<NDIM;i++){
        sizebnd[i]=1;
        for (j=0;j<NDIM;j++){
            if (i!=j)
                sizebnd[i]*=N[j];
        }
    }
    
    for (i=0;i<numvar;i++){
        
        if (strcmp(vars[i].name,"vx")==0 ||
            strcmp(vars[i].name,"vy")==0 ||
            strcmp(vars[i].name,"vz")==0 ||
            strcmp(vars[i].name,"sxx")==0 ||
            strcmp(vars[i].name,"syy")==0 ||
            strcmp(vars[i].name,"szz")==0 ||
            strcmp(vars[i].name,"sxy")==0 ||
            strcmp(vars[i].name,"sxz")==0 ||
            strcmp(vars[i].name,"syz")==0 ){
            vars[i].num_ele=sizevars;
        }
        else if(strcmp(vars[i].name,"rxx")==0 ||
                strcmp(vars[i].name,"ryy")==0 ||
                strcmp(vars[i].name,"rzz")==0 ||
                strcmp(vars[i].name,"rxy")==0 ||
                strcmp(vars[i].name,"rxz")==0 ||
                strcmp(vars[i].name,"ryz")==0 )
        {
            vars[i].num_ele=sizevars*L;
        }
        else if(strcmp(vars[i].name,"psi_szz_z")==0 ||
                strcmp(vars[i].name,"psi_sxz_z")==0 ||
                strcmp(vars[i].name,"psi_syz_z")==0 ||
                strcmp(vars[i].name,"psi_vx_z")==0 ||
                strcmp(vars[i].name,"psi_vy_z")==0 ||
                strcmp(vars[i].name,"psi_vz_z")==0 ){
            vars[i].num_ele=sizebnd[0];
        }
        else if(strcmp(vars[i].name,"psi_syy_y")==0 ||
                strcmp(vars[i].name,"psi_sxy_y")==0 ||
                strcmp(vars[i].name,"psi_syz_y")==0 ||
                strcmp(vars[i].name,"psi_vx_y")==0 ||
                strcmp(vars[i].name,"psi_vy_y")==0 ||
                strcmp(vars[i].name,"psi_vz_y")==0 ){
            vars[i].num_ele=sizebnd[1];
        }
        else if(strcmp(vars[i].name,"psi_sxx_x")==0 ||
                strcmp(vars[i].name,"psi_sxy_x")==0 ||
                strcmp(vars[i].name,"psi_sxz_x")==0 ||
                strcmp(vars[i].name,"psi_vx_x")==0 ||
                strcmp(vars[i].name,"psi_vy_x")==0 ||
                strcmp(vars[i].name,"psi_vz_x")==0 ){
            vars[i].num_ele=sizebnd[NDIM-1];
        }

        

    }
    
    return 0;

    
}
