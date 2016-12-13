/*Those are the kernels that update the velocities. Each finite difference order has a different kernels, to optimize performance. */

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define rho(z,y,x)     rho[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rip(z,y,x)     rip[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rjp(z,y,x)     rjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,y,x)     rkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipjp(z,y,x) uipjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define ujpkp(z,y,x) ujpkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipkp(z,y,x) uipkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define u(z,y,x)         u[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define pi(z,y,x)       pi[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define grad(z,y,x)   grad[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define grads(z,y,x) grads[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define amp1(z,y,x)   amp1[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define amp2(z,y,x)   amp2[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]


#define psi_sxx_x(z,y,x)   psi_sxx_x[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define psi_sxy_x(z,y,x)   psi_sxy_x[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define psi_sxz_x(z,y,x)   psi_sxz_x[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define psi_syy_y(z,y,x)   psi_syy_y[((x)-fdoh)*(2*nab)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define psi_sxy_y(z,y,x)   psi_sxy_y[((x)-fdoh)*(2*nab)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define psi_syz_y(z,y,x)   psi_syz_y[((x)-fdoh)*(2*nab)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define psi_szz_z(z,y,x)   psi_szz_z[((x)-fdoh)*(NY-2*fdoh)*(2*nab)+((y)-fdoh)*(2*nab)+((z)-fdoh)]
#define psi_sxz_z(z,y,x)   psi_sxz_z[((x)-fdoh)*(NY-2*fdoh)*(2*nab)+((y)-fdoh)*(2*nab)+((z)-fdoh)]
#define psi_syz_z(z,y,x)   psi_syz_z[((x)-fdoh)*(NY-2*fdoh)*(2*nab)+((y)-fdoh)*(2*nab)+((z)-fdoh)]

#define taus(z,y,x)         taus[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipjp(z,y,x) tausipjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausjpkp(z,y,x) tausjpkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipkp(z,y,x) tausipkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define taup(z,y,x)         taup[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define vx(z,y,x)   vx[(x)*NY*NZ+(y)*NZ+(z)]
#define vy(z,y,x)   vy[(x)*NY*NZ+(y)*NZ+(z)]
#define vz(z,y,x)   vz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxx(z,y,x) sxx[(x)*NY*NZ+(y)*NZ+(z)]
#define syy(z,y,x) syy[(x)*NY*NZ+(y)*NZ+(z)]
#define szz(z,y,x) szz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxy(z,y,x) sxy[(x)*NY*NZ+(y)*NZ+(z)]
#define syz(z,y,x) syz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxz(z,y,x) sxz[(x)*NY*NZ+(y)*NZ+(z)]

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#if local_off==0

#define lvx(z,y,x)   lvx[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lvy(z,y,x)   lvy[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lvz(z,y,x)   lvz[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsxx(z,y,x) lsxx[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsyy(z,y,x) lsyy[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lszz(z,y,x) lszz[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsxy(z,y,x) lsxy[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsyz(z,y,x) lsyz[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsxz(z,y,x) lsxz[(x)*lsizey*lsizez+(y)*lsizez+(z)]

#endif



#define PI (3.141592653589793238462643383279502884197169)
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]

float3 ssource(int gidz, int gidy, int gidx,  int nsrc, __global float *srcpos_loc, __global float *signals, int nt, __global float * rip, __global float * rjp, __global float * rkp){
    
    float3 ampv={0.0,0.0,0.0};
    if (nsrc>0){
        
        
        for (int srci=0; srci<nsrc; srci++){
            
            
            int i=(int)srcpos_loc(0,srci)/DH+fdoh;
            int j=(int)srcpos_loc(1,srci)/DH+fdoh;
            int k=(int)srcpos_loc(2,srci)/DH+fdoh;
            
            if (i==gidx && j==gidy && k==gidz){
                //                float azi_rad=srcpos_loc(6,srci) * PI / 180;
                
                float amp=(DT*signals(srci,nt))/(DH*DH*DH); // scaled force amplitude with F= 1N
                
                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
                
                if (SOURCE_TYPE==2){
                    /* single force in x */
                    ampv.x  +=  amp/rip(k,j,i);
                }
                else if (SOURCE_TYPE==3){
                    /* single force in y */
                    
                    ampv.y  +=  amp/rjp(k,j,i);
                }
                else if (SOURCE_TYPE==4){
                    /* single force in y */
                    
                    ampv.z  +=  amp/rkp(k,j,i);
                }
                
            }
        }
        
        
    }
    
    return ampv;
    
}


__kernel void update_v_CPML(__global float *vx,         __global float *vy,      __global float *vz,
                            __global float *sxx,        __global float *syy,     __global float *szz,
                            __global float *sxy,        __global float *syz,     __global float *sxz,
                            __global float *rip,        __global float *rjp,     __global float *rkp,
                            __global float * K_x,       __global float * a_x,    __global float * b_x,
                            __global float * K_x_half,  __global float * a_x_half, __global float * b_x_half,
                            __global float * K_y, __global float * a_y, __global float * b_y,
                            __global float * K_y_half, __global float * a_y_half, __global float * b_y_half,
                            __global float * K_z, __global float * a_z, __global float * b_z,
                            __global float * K_z_half, __global float * a_z_half, __global float * b_z_half,
                            __global float * psi_sxx_x, __global float * psi_sxy_x, __global float * psi_sxz_x,
                            __global float * psi_sxy_y, __global float * psi_syy_y, __global float * psi_syz_y,
                            __global float * psi_sxz_z, __global float * psi_syz_z, __global float * psi_szz_z,
                            __global float *srcpos_loc, __global float *signals, __global float *rec_pos,
                            __global float *taper,
                            __local  float *lsxx,       __local  float *lsyy,    __local  float *lszz,
                            __local  float *lsxy,       __local  float *lsyz,    __local  float *lsxz,
                            int nsrc,  int nt)
{
/* Standard staggered grid kernel, finite difference order of 2 to 12.  */
    
    int gid = get_global_id(0)+get_global_id(1)*get_global_size(0)+get_global_id(2)*get_global_size(0)*get_global_size(1);
    int lz = get_local_size(0);
    int ly = get_local_size(1);
    int glsizez = (NZ-2*fdoh)+(lz-(NZ-2*fdoh)%lz)%lz;
    int glsizey = (NY-2*fdoh)+(ly-(NY-2*fdoh)%ly)%ly;
    int gidz = gid%glsizez+fdoh;
    int gidy = (gid/glsizez)%glsizey+fdoh;
    int gidx = gid/(glsizez*glsizey)+fdoh;
        
#if local_off==0
    int lsizez0 = get_local_size(0);
    int lsizey0 = get_local_size(1);
    int lsizex0 = get_local_size(2);
    int lsizez = lsizez0+2*fdoh;
    int lsizey = lsizey0+2*fdoh;
    int lsizex = lsizex0+2*fdoh;
    int lidz = get_local_id(0);
    int lidy = get_local_id(1);
    int lidx = get_local_id(2);
    int lid = lidz+lidy*lsizez0+lidx*lsizez0*lsizey0;
    int n=(lsizez*lsizey*lsizex)/(lsizez0*lsizey0*lsizex0);
    int lidz2=0;
    int lidy2=0;
    int lidx2=0;
    
    int gidz0=get_local_size(0)*get_group_id(0);
    int gidy0=get_local_size(1)*get_group_id(1);
    int gidx0=get_local_size(2)*get_group_id(2);
    
    

    for (int i=0;i<n;i++){
        
        lidz2=lid%lsizez;
        lidy2=(lid/lsizez)%lsizey;
        lidx2=lid/(lsizez*lsizey);
        
        lsxx(lidz2,lidy2,lidx2)=sxx(gidz0+lidz2,gidy0+lidy2,gidx0+lidx2);
        lsyy(lidz2,lidy2,lidx2)=syy(gidz0+lidz2,gidy0+lidy2,gidx0+lidx2);
        lszz(lidz2,lidy2,lidx2)=szz(gidz0+lidz2,gidy0+lidy2,gidx0+lidx2);
        lsxy(lidz2,lidy2,lidx2)=sxy(gidz0+lidz2,gidy0+lidy2,gidx0+lidx2);
        lsyz(lidz2,lidy2,lidx2)=syz(gidz0+lidz2,gidy0+lidy2,gidx0+lidx2);
        lsxz(lidz2,lidy2,lidx2)=sxz(gidz0+lidz2,gidy0+lidy2,gidx0+lidx2);
        
        lid+=(lsizez0*lsizey0*lsizex0);

    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
   
    
    lidz+= fdoh;
    lidy+= fdoh;
    lidx+= fdoh;
    
//    int lsizez = get_local_size(0)+2*fdoh;
//    int lsizey = get_local_size(1)+2*fdoh;
//    int lsizex = get_local_size(2)+2*fdoh;
//    int lidz = get_local_id(0)+fdoh;
//    int lidy = get_local_id(1)+fdoh;
//    int lidx = get_local_id(2)+fdoh;
//    
//    lsxx(lidz,lidy,lidx)=sxx(gidz,gidy,gidx);
//    lsyy(lidz,lidy,lidx)=syy(gidz,gidy,gidx);
//    lszz(lidz,lidy,lidx)=szz(gidz,gidy,gidx);
//    lsxy(lidz,lidy,lidx)=sxy(gidz,gidy,gidx);
//    lsyz(lidz,lidy,lidx)=syz(gidz,gidy,gidx);
//    lsxz(lidz,lidy,lidx)=sxz(gidz,gidy,gidx);
//    
//    if (lidz<2*fdoh)
//    {
//        lsxx(lidz-fdoh,lidy,lidx)=sxx(gidz-fdoh,gidy,gidx);
//        lsyy(lidz-fdoh,lidy,lidx)=syy(gidz-fdoh,gidy,gidx);
//        lszz(lidz-fdoh,lidy,lidx)=szz(gidz-fdoh,gidy,gidx);
//        lsxy(lidz-fdoh,lidy,lidx)=sxy(gidz-fdoh,gidy,gidx);
//        lsyz(lidz-fdoh,lidy,lidx)=syz(gidz-fdoh,gidy,gidx);
//        lsxz(lidz-fdoh,lidy,lidx)=sxz(gidz-fdoh,gidy,gidx);
//        
//    }
//    if (lidz>(lsizez-2*fdoh-1))
//    {
//        lsxx(lidz+fdoh,lidy,lidx)=sxx(gidz+fdoh,gidy,gidx);
//        lsyy(lidz+fdoh,lidy,lidx)=syy(gidz+fdoh,gidy,gidx);
//        lszz(lidz+fdoh,lidy,lidx)=szz(gidz+fdoh,gidy,gidx);
//        lsxy(lidz+fdoh,lidy,lidx)=sxy(gidz+fdoh,gidy,gidx);
//        lsyz(lidz+fdoh,lidy,lidx)=syz(gidz+fdoh,gidy,gidx);
//        lsxz(lidz+fdoh,lidy,lidx)=sxz(gidz+fdoh,gidy,gidx);
//        
//    }
//    
//    if (lidy<2*fdoh)
//    {
//        lsxx(lidz,lidy-fdoh,lidx)=sxx(gidz,gidy-fdoh,gidx);
//        lsyy(lidz,lidy-fdoh,lidx)=syy(gidz,gidy-fdoh,gidx);
//        lszz(lidz,lidy-fdoh,lidx)=szz(gidz,gidy-fdoh,gidx);
//        lsxy(lidz,lidy-fdoh,lidx)=sxy(gidz,gidy-fdoh,gidx);
//        lsyz(lidz,lidy-fdoh,lidx)=syz(gidz,gidy-fdoh,gidx);
//        lsxz(lidz,lidy-fdoh,lidx)=sxz(gidz,gidy-fdoh,gidx);
//        
//    }
//    if (lidy>(lsizey-2*fdoh-1))
//    {
//        lsxx(lidz,lidy+fdoh,lidx)=sxx(gidz,gidy+fdoh,gidx);
//        lsyy(lidz,lidy+fdoh,lidx)=syy(gidz,gidy+fdoh,gidx);
//        lszz(lidz,lidy+fdoh,lidx)=szz(gidz,gidy+fdoh,gidx);
//        lsxy(lidz,lidy+fdoh,lidx)=sxy(gidz,gidy+fdoh,gidx);
//        lsyz(lidz,lidy+fdoh,lidx)=syz(gidz,gidy+fdoh,gidx);
//        lsxz(lidz,lidy+fdoh,lidx)=sxz(gidz,gidy+fdoh,gidx);
//        
//    }
//    
//    if (lidx<2*fdoh)
//    {
//        lsxx(lidz,lidy,lidx-fdoh)=sxx(gidz,gidy,gidx-fdoh);
//        lsyy(lidz,lidy,lidx-fdoh)=syy(gidz,gidy,gidx-fdoh);
//        lszz(lidz,lidy,lidx-fdoh)=szz(gidz,gidy,gidx-fdoh);
//        lsxy(lidz,lidy,lidx-fdoh)=sxy(gidz,gidy,gidx-fdoh);
//        lsyz(lidz,lidy,lidx-fdoh)=syz(gidz,gidy,gidx-fdoh);
//        lsxz(lidz,lidy,lidx-fdoh)=sxz(gidz,gidy,gidx-fdoh);
//        
//    }
//    if (lidx>(lsizex-2*fdoh-1))
//    {
//        lsxx(lidz,lidy,lidx+fdoh)=sxx(gidz,gidy,gidx+fdoh);
//        lsyy(lidz,lidy,lidx+fdoh)=syy(gidz,gidy,gidx+fdoh);
//        lszz(lidz,lidy,lidx+fdoh)=szz(gidz,gidy,gidx+fdoh);
//        lsxy(lidz,lidy,lidx+fdoh)=sxy(gidz,gidy,gidx+fdoh);
//        lsyz(lidz,lidy,lidx+fdoh)=syz(gidz,gidy,gidx+fdoh);
//        lsxz(lidz,lidy,lidx+fdoh)=sxz(gidz,gidy,gidx+fdoh);
//        
//    }
//    
//    
//    
//    barrier(CLK_LOCAL_MEM_FENCE);
#elif local_off==1
    
    #define lsxx sxx
    #define lsyy syy
    #define lszz szz
    #define lsxy sxy
    #define lsyz syz
    #define lsxz sxz
    #define lidx gidx
    #define lidy gidy
    #define lidz gidz
    
#endif

    
    if (gidy>(NY-fdoh-1) || gidx>(NX-fdoh-1) || gidz>(NZ-fdoh-1) ){
        return;
    }

    
    float3 amp = ssource(gidz, gidy, gidx+offset, nsrc, srcpos_loc, signals, nt, rip, rjp, rkp);
    
    float sxx_x;
    float syy_y;
    float szz_z;
    float sxy_y;
    float sxy_x;
    float syz_y;
    float syz_z;
    float sxz_x;
    float sxz_z;
    
#if   fdoh ==1
    {
        sxx_x = dtdh*hc1*(lsxx(lidz,lidy,lidx+1) - lsxx(lidz,lidy,lidx));
        sxy_y = dtdh*hc1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy-1,lidx));
        sxz_z = dtdh*hc1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz-1,lidy,lidx));
        
        syy_y = dtdh*hc1*(lsyy(lidz,lidy+1,lidx) - lsyy(lidz,lidy,lidx));
        sxy_x = dtdh*hc1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy,lidx-1));
        syz_z = dtdh*hc1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz-1,lidy,lidx));
        
        szz_z = dtdh*hc1*(lszz(lidz+1,lidy,lidx) - lszz(lidz,lidy,lidx));
        sxz_x = dtdh*hc1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz,lidy,lidx-1));
        syz_y = dtdh*hc1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz,lidy-1,lidx));
    }
#elif fdoh ==2
    {
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1) - lsxx(lidz,lidy,lidx))
                     +hc2*(lsxx(lidz,lidy,lidx+2) - lsxx(lidz,lidy,lidx-1)));
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy-1,lidx))
                     +hc2*(lsxy(lidz,lidy+1,lidx) - lsxy(lidz,lidy-2,lidx)));
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz-1,lidy,lidx))
                     +hc2*(lsxz(lidz+1,lidy,lidx) - lsxz(lidz-2,lidy,lidx)));
        
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx) - lsyy(lidz,lidy,lidx))
                     +hc2*(lsyy(lidz,lidy+2,lidx) - lsyy(lidz,lidy-1,lidx)));
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)   - lsxy(lidz,lidy,lidx-1))
                     +hc2*(lsxy(lidz,lidy,lidx+1) - lsxy(lidz,lidy,lidx-2)));
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz-1,lidy,lidx))
                     +hc2*(lsyz(lidz+1,lidy,lidx) - lsyz(lidz-2,lidy,lidx)));
        
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx) - lszz(lidz,lidy,lidx))
                     +hc2*(lszz(lidz+2,lidy,lidx) - lszz(lidz-1,lidy,lidx)));
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)   - lsxz(lidz,lidy,lidx-1))
                     +hc2*(lsxz(lidz,lidy,lidx+1) - lsxz(lidz,lidy,lidx-2)));
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)   - lsyz(lidz,lidy-1,lidx))
                     +hc2*(lsyz(lidz,lidy+1,lidx) - lsyz(lidz,lidy-2,lidx)));
    }
#elif fdoh ==3
    {
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      hc2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      hc3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2)));
        
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      hc2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      hc3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx)));
        
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      hc2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      hc3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx)));
        
        
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      hc2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      hc3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx)));
        
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      hc2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      hc3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3)));
        
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      hc2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      hc3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx)));
        
        
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      hc2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      hc3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      hc2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      hc3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3)));
        
        
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz,lidy-1,lidx))+
                      hc2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      hc3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx)));
    }
#elif fdoh ==4
    {
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      hc2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      hc3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2))+
                      hc4*(lsxx(lidz,lidy,lidx+4)-lsxx(lidz,lidy,lidx-3)));
        
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      hc2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      hc3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx))+
                      hc4*(lsxy(lidz,lidy+3,lidx)-lsxy(lidz,lidy-4,lidx)));
        
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      hc2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      hc3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx))+
                      hc4*(lsxz(lidz+3,lidy,lidx)-lsxz(lidz-4,lidy,lidx)));
        
        
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      hc2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      hc3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx))+
                      hc4*(lsyy(lidz,lidy+4,lidx)-lsyy(lidz,lidy-3,lidx)));
        
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      hc2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      hc3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3))+
                      hc4*(lsxy(lidz,lidy,lidx+3)-lsxy(lidz,lidy,lidx-4)));
        
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      hc2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      hc3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx))+
                      hc4*(lsyz(lidz+3,lidy,lidx)-lsyz(lidz-4,lidy,lidx)));
        
        
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      hc2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      hc3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx))+
                      hc4*(lszz(lidz+4,lidy,lidx)-lszz(lidz-3,lidy,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      hc2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      hc3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3))+
                      hc4*(lsxz(lidz,lidy,lidx+3)-lsxz(lidz,lidy,lidx-4)));
        
        
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)-lsyz(lidz,lidy-1,lidx))+
                      hc2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      hc3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx))+
                      hc4*(lsyz(lidz,lidy+3,lidx)-lsyz(lidz,lidy-4,lidx)));
    }
#elif fdoh ==5
    {
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      hc2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      hc3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2))+
                      hc4*(lsxx(lidz,lidy,lidx+4)-lsxx(lidz,lidy,lidx-3))+
                      hc5*(lsxx(lidz,lidy,lidx+5)-lsxx(lidz,lidy,lidx-4)));
        
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      hc2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      hc3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx))+
                      hc4*(lsxy(lidz,lidy+3,lidx)-lsxy(lidz,lidy-4,lidx))+
                      hc5*(lsxy(lidz,lidy+4,lidx)-lsxy(lidz,lidy-5,lidx)));
        
        sxz_z = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz-1,lidy,lidx))+
                      hc2*(lsxz(lidz+1,lidy,lidx)-lsxz(lidz-2,lidy,lidx))+
                      hc3*(lsxz(lidz+2,lidy,lidx)-lsxz(lidz-3,lidy,lidx))+
                      hc4*(lsxz(lidz+3,lidy,lidx)-lsxz(lidz-4,lidy,lidx))+
                      hc5*(lsxz(lidz+4,lidy,lidx)-lsxz(lidz-5,lidy,lidx)));
        
        
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      hc2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      hc3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx))+
                      hc4*(lsyy(lidz,lidy+4,lidx)-lsyy(lidz,lidy-3,lidx))+
                      hc5*(lsyy(lidz,lidy+5,lidx)-lsyy(lidz,lidy-4,lidx)));
        
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      hc2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      hc3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3))+
                      hc4*(lsxy(lidz,lidy,lidx+3)-lsxy(lidz,lidy,lidx-4))+
                      hc5*(lsxy(lidz,lidy,lidx+4)-lsxy(lidz,lidy,lidx-5)));
        
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      hc2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      hc3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx))+
                      hc4*(lsyz(lidz+3,lidy,lidx)-lsyz(lidz-4,lidy,lidx))+
                      hc5*(lsyz(lidz+4,lidy,lidx)-lsyz(lidz-5,lidy,lidx)));
        
        
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      hc2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      hc3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx))+
                      hc4*(lszz(lidz+4,lidy,lidx)-lszz(lidz-3,lidy,lidx))+
                      hc5*(lszz(lidz+5,lidy,lidx)-lszz(lidz-4,lidy,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      hc2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      hc3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3))+
                      hc4*(lsxz(lidz,lidy,lidx+3)-lsxz(lidz,lidy,lidx-4))+
                      hc5*(lsxz(lidz,lidy,lidx+4)-lsxz(lidz,lidy,lidx-5)));
        
        
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz,lidy-1,lidx))+
                      hc2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      hc3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx))+
                      hc4*(lsyz(lidz,lidy+3,lidx)-lsyz(lidz,lidy-4,lidx))+
                      hc5*(lsyz(lidz,lidy+4,lidx)-lsyz(lidz,lidy-5,lidx)));
    }
#elif fdoh ==6
    {
        sxx_x = dtdh*(hc1*(lsxx(lidz,lidy,lidx+1)-lsxx(lidz,lidy,lidx))+
                      hc2*(lsxx(lidz,lidy,lidx+2)-lsxx(lidz,lidy,lidx-1))+
                      hc3*(lsxx(lidz,lidy,lidx+3)-lsxx(lidz,lidy,lidx-2))+
                      hc4*(lsxx(lidz,lidy,lidx+4)-lsxx(lidz,lidy,lidx-3))+
                      hc5*(lsxx(lidz,lidy,lidx+5)-lsxx(lidz,lidy,lidx-4))+
                      hc6*(lsxx(lidz,lidy,lidx+6)-lsxx(lidz,lidy,lidx-5)));
        
        sxy_y = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy-1,lidx))+
                      hc2*(lsxy(lidz,lidy+1,lidx)-lsxy(lidz,lidy-2,lidx))+
                      hc3*(lsxy(lidz,lidy+2,lidx)-lsxy(lidz,lidy-3,lidx))+
                      hc4*(lsxy(lidz,lidy+3,lidx)-lsxy(lidz,lidy-4,lidx))+
                      hc5*(lsxy(lidz,lidy+4,lidx)-lsxy(lidz,lidy-5,lidx))+
                      hc6*(lsxy(lidz,lidy+5,lidx)-lsxy(lidz,lidy-6,lidx)));
        
        sxz_z = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz-1,lidy,lidx))+
                      hc2*(lsxy(lidz+1,lidy,lidx)-lsxy(lidz-2,lidy,lidx))+
                      hc3*(lsxy(lidz+2,lidy,lidx)-lsxy(lidz-3,lidy,lidx))+
                      hc4*(lsxy(lidz+3,lidy,lidx)-lsxy(lidz-4,lidy,lidx))+
                      hc5*(lsxy(lidz+4,lidy,lidx)-lsxy(lidz-5,lidy,lidx))+
                      hc6*(lsxy(lidz+5,lidy,lidx)-lsxy(lidz-6,lidy,lidx)));
        
        
        syy_y = dtdh*(hc1*(lsyy(lidz,lidy+1,lidx)-lsyy(lidz,lidy,lidx))+
                      hc2*(lsyy(lidz,lidy+2,lidx)-lsyy(lidz,lidy-1,lidx))+
                      hc3*(lsyy(lidz,lidy+3,lidx)-lsyy(lidz,lidy-2,lidx))+
                      hc4*(lsyy(lidz,lidy+4,lidx)-lsyy(lidz,lidy-3,lidx))+
                      hc5*(lsyy(lidz,lidy+5,lidx)-lsyy(lidz,lidy-4,lidx))+
                      hc6*(lsyy(lidz,lidy+6,lidx)-lsyy(lidz,lidy-5,lidx)));
        
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidy,lidx)  -lsxy(lidz,lidy,lidx-1))+
                      hc2*(lsxy(lidz,lidy,lidx+1)-lsxy(lidz,lidy,lidx-2))+
                      hc3*(lsxy(lidz,lidy,lidx+2)-lsxy(lidz,lidy,lidx-3))+
                      hc4*(lsxy(lidz,lidy,lidx+3)-lsxy(lidz,lidy,lidx-4))+
                      hc5*(lsxy(lidz,lidy,lidx+4)-lsxy(lidz,lidy,lidx-5))+
                      hc6*(lsxy(lidz,lidy,lidx+5)-lsxy(lidz,lidy,lidx-6)));
        
        syz_z = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz-1,lidy,lidx))+
                      hc2*(lsyz(lidz+1,lidy,lidx)-lsyz(lidz-2,lidy,lidx))+
                      hc3*(lsyz(lidz+2,lidy,lidx)-lsyz(lidz-3,lidy,lidx))+
                      hc4*(lsyz(lidz+3,lidy,lidx)-lsyz(lidz-4,lidy,lidx))+
                      hc5*(lsyz(lidz+4,lidy,lidx)-lsyz(lidz-5,lidy,lidx))+
                      hc6*(lsyz(lidz+5,lidy,lidx)-lsyz(lidz-6,lidy,lidx)));
        
        
        szz_z = dtdh*(hc1*(lszz(lidz+1,lidy,lidx)-lszz(lidz,lidy,lidx))+
                      hc2*(lszz(lidz+2,lidy,lidx)-lszz(lidz-1,lidy,lidx))+
                      hc3*(lszz(lidz+3,lidy,lidx)-lszz(lidz-2,lidy,lidx))+
                      hc4*(lszz(lidz+4,lidy,lidx)-lszz(lidz-3,lidy,lidx))+
                      hc5*(lszz(lidz+5,lidy,lidx)-lszz(lidz-4,lidy,lidx))+
                      hc6*(lszz(lidz+6,lidy,lidx)-lszz(lidz-5,lidy,lidx)));
        
        sxz_x = dtdh*(hc1*(lsxz(lidz,lidy,lidx)  -lsxz(lidz,lidy,lidx-1))+
                      hc2*(lsxz(lidz,lidy,lidx+1)-lsxz(lidz,lidy,lidx-2))+
                      hc3*(lsxz(lidz,lidy,lidx+2)-lsxz(lidz,lidy,lidx-3))+
                      hc4*(lsxz(lidz,lidy,lidx+3)-lsxz(lidz,lidy,lidx-4))+
                      hc5*(lsxz(lidz,lidy,lidx+4)-lsxz(lidz,lidy,lidx-5))+
                      hc6*(lsxz(lidz,lidy,lidx+5)-lsxz(lidz,lidy,lidx-6)));
        
        
        syz_y = dtdh*(hc1*(lsyz(lidz,lidy,lidx)  -lsyz(lidz,lidy-1,lidx))+
                      hc2*(lsyz(lidz,lidy+1,lidx)-lsyz(lidz,lidy-2,lidx))+
                      hc3*(lsyz(lidz,lidy+2,lidx)-lsyz(lidz,lidy-3,lidx))+
                      hc4*(lsyz(lidz,lidy+3,lidx)-lsyz(lidz,lidy-4,lidx))+
                      hc5*(lsyz(lidz,lidy+4,lidx)-lsyz(lidz,lidy-5,lidx))+
                      hc6*(lsyz(lidz,lidy+5,lidx)-lsyz(lidz,lidy-6,lidx)));
    }
#endif
    
#if freesurf==0
    if (gidz-fdoh<nab){
        psi_sxz_z(gidz,gidy,gidx) = b_z[gidz] * psi_sxz_z(gidz,gidy,gidx) + a_z[gidz] * sxz_z;
        sxz_z = sxz_z / K_z[gidz] + psi_sxz_z(gidz,gidy,gidx);
        psi_syz_z(gidz,gidy,gidx) = b_z[gidz] * psi_syz_z(gidz,gidy,gidx) + a_z[gidz] * syz_z;
        syz_z = syz_z / K_z[gidz] + psi_syz_z(gidz,gidy,gidx);
        psi_szz_z(gidz,gidy,gidx) = b_z_half[gidz] * psi_szz_z(gidz,gidy,gidx) + a_z_half[gidz] * szz_z;
        szz_z = szz_z / K_z_half[gidz] + psi_szz_z(gidz,gidy,gidx);
    }
#endif
    
    if (gidz>NZ-nab-fdoh-1){
        h1 = (gidz-NZ+2*nab+fdoh);
        
        psi_sxz_z(h1,gidy,gidx) = b_z[h1] * psi_sxz_z(h1,gidy,gidx) + a_z[h1] * sxz_z;
        sxz_z = sxz_z / K_z[h1] + psi_sxz_z(h1,gidy,gidx);
        psi_syz_z(h1,gidy,gidx) = b_z[h1] * psi_syz_z(h1,gidy,gidx) + a_z[h1] * syz_z;
        syz_z = syz_z / K_z[h1] + psi_syz_z(h1,gidy,gidx);
        psi_szz_z(h1,gidy,gidx) = b_z_half[h1] * psi_szz_z(h1,gidy,gidx) + a_z_half[h1] * szz_z;
        szz_z = szz_z / K_z_half[h1] + psi_szz_z(h1,gidy,gidx);
    }
    
    if (gidy-fdoh<nab){
       
        psi_sxy_y(gidz,gidy,gidx) = b_y[gidy] * psi_sxy_y(gidz,gidy,gidx) + a_y[gidy] * sxy_y;
        sxy_y = sxy_y / K_y[gidy] + psi_sxy_y(gidz,gidy,gidx);
        
        psi_syy_y(gidz,gidy,gidx) = b_y_half[gidy] * psi_syy_y(gidz,gidy,gidx) + a_y_half[gidy] * syy_y;
        syy_y = syy_y / K_y_half[gidy] + psi_syy_y(gidz,gidy,gidx);
        
        psi_syz_y(gidz,gidy,gidx) = b_y[gidy] * psi_syz_y(gidz,gidy,gidx) + a_y[gidy] * syz_y;
        syz_y = syz_y / K_y[gidy] + psi_syz_y(gidz,gidy,gidx);
        
#if freesurf==0
        if (gidz-fdoh<nab){
            psi_sxz_z(gidz,gidy,gidx) = b_z[gidz] * psi_sxz_z(gidz,gidy,gidx) + a_z[gidz] * sxz_z;
            sxz_z = sxz_z / K_z[gidz] + psi_sxz_z(gidz,gidy,gidx);
            psi_syz_z(gidz,gidy,gidx) = b_z[gidz] * psi_syz_z(gidz,gidy,gidx) + a_z[gidz] * syz_z;
            syz_z = syz_z / K_z[gidz] + psi_syz_z(gidz,gidy,gidx);
            psi_szz_z(gidz,gidy,gidx) = b_z_half[gidz] * psi_szz_z(gidz,gidy,gidx) + a_z_half[gidz] * szz_z;
            szz_z = szz_z / K_z_half[gidz] + psi_szz_z(gidz,gidy,gidx);}
#endif
        
        if (gidz>NZ-nab-fdoh-1){
            h1 = (gidz-NZ+2*nab+fdoh);
            psi_sxz_z(h1,gidy,gidx) = b_z[h1] * psi_sxz_z(h1,gidy,gidx) + a_z[h1] * sxz_z;
            sxz_z = sxz_z / K_z[h1] + psi_sxz_z(h1,gidy,gidx);
            psi_syz_z(h1,gidy,gidx) = b_z[h1] * psi_syz_z(h1,gidy,gidx) + a_z[h1] * syz_z;
            syz_z = syz_z / K_z[h1] + psi_syz_z(h1,gidy,gidx);
            psi_szz_z(h1,gidy,gidx) = b_z_half[h1] * psi_szz_z(h1,gidy,gidx) + a_z_half[h1] * szz_z;
            szz_z = szz_z / K_z_half[h1] + psi_szz_z(h1,gidy,gidx);}

    }
    
    if (gidy>NY-nab-fdoh-1){
        
        h1 = (gidy-NY+2*nab+fdoh);
        
        psi_sxy_y(gidz,h1,gidx) = b_y[h1] * psi_sxy_y(gidz,h1,gidx) + a_y[h1] * sxy_y;
        sxy_y = sxy_y / K_y[h1] + psi_sxy_y(gidz,h1,gidx);
        psi_syy_y(gidz,h1,gidx) = b_y_half[h1] * psi_syy_y(gidz,h1,gidx) + a_y_half[h1] * syy_y;
        syy_y = syy_y / K_y_half[h1] + psi_syy_y(gidz,h1,gidx);
        psi_syz_y(gidz,h1,gidx) = b_y[h1] * psi_syz_y(gidz,h1,gidx) + a_y[h1] * syz_y;
        syz_y = syz_y / K_y[h1] + psi_syz_y(gidz,h1,gidx);
        
#if freesurf==0
        if (gidz-fdoh<nab){
            psi_sxz_z(gidz,gidy,gidx) = b_z[gidz] * psi_sxz_z(gidz,gidy,gidx) + a_z[gidz] * sxz_z;
            sxz_z = sxz_z / K_z[gidz] + psi_sxz_z(gidz,gidy,gidx);
            psi_syz_z(gidz,gidy,gidx) = b_z[gidz] * psi_syz_z(gidz,gidy,gidx) + a_z[gidz] * syz_z;
            syz_z = syz_z / K_z[gidz] + psi_syz_z(gidz,gidy,gidx);
            psi_szz_z(gidz,gidy,gidx) = b_z_half[gidz] * psi_szz_z(gidz,gidy,gidx) + a_z_half[gidz] * szz_z;
            szz_z = szz_z / K_z_half[gidz] + psi_szz_z(gidz,gidy,gidx);}
#endif
        
        if (gidz>NZ-nab-fdoh-1){
            h1 = (gidz-NZ+2*nab+fdoh);
            psi_sxz_z(h1,gidy,gidx) = b_z[h1] * psi_sxz_z(h1,gidy,gidx) + a_z[h1] * sxz_z;
            sxz_z = sxz_z / K_z[h1] + psi_sxz_z(h1,gidy,gidx);
            psi_syz_z(h1,gidy,gidx) = b_z[h1] * psi_syz_z(h1,gidy,gidx) + a_z[h1] * syz_z;
            syz_z = syz_z / K_z[h1] + psi_syz_z(h1,gidy,gidx);
            psi_szz_z(h1,gidy,gidx) = b_z_half[h1] * psi_szz_z(h1,gidy,gidx) + a_z_half[h1] * szz_z;
            szz_z = szz_z / K_z_half[h1] + psi_szz_z(h1,gidy,gidx);}
        
    }
    
#if dev==0
    if (gidx-fdoh<nab){
        
        psi_sxx_x(gidz,gidy,gidx) = b_x_half[gidx] * psi_sxx_x(gidz,gidy,gidx) + a_x_half[gidx] * sxx_x;
        sxx_x = sxx_x / K_x_half[gidx] + psi_sxx_x(gidz,gidy,gidx);
        psi_sxy_x(gidz,gidy,gidx) = b_x[gidx] * psi_sxy_x(gidz,gidy,gidx) + a_x[gidx] * sxy_x;
        sxy_x = sxy_x / K_x[gidx] + psi_sxy_x(gidz,gidy,gidx);
        psi_sxz_x(gidz,gidy,gidx) = b_x[gidx] * psi_sxz_x(gidz,gidy,gidx) + a_x[gidx] * sxz_x;
        sxz_x = sxz_x / K_x[gidx] + psi_sxz_x(gidz,gidy,gidx);
        
        if (gidy-fdoh<=nab){
            
            psi_sxy_y(gidz,gidy,gidx) = b_y[gidy] * psi_sxy_y(gidz,gidy,gidx) + a_y[gidy] * sxy_y;
            sxy_y = sxy_y / K_y[gidy] + psi_sxy_y(gidz,gidy,gidx);
            psi_syy_y(gidz,gidy,gidx) = b_y_half[gidy] * psi_syy_y(gidz,gidy,gidx) + a_y_half[gidy] * syy_y;
            syy_y = syy_y / K_y_half[gidy] + psi_syy_y(gidz,gidy,gidx);
            psi_syz_y(gidz,gidy,gidx) = b_y[gidy] * psi_syz_y(gidz,gidy,gidx) + a_y[gidy] * syz_y;
            syz_y = syz_y / K_y[gidy] + psi_syz_y(gidz,gidy,gidx); 	 }
        
        
        if (gidy>NY-nab-fdoh-1){
            h1 = (gidy-NY+2*nab+fdoh);
            psi_sxy_y(gidz,h1,gidx) = b_y[h1] * psi_sxy_y(gidz,h1,gidx) + a_y[h1] * sxy_y;
            sxy_y = sxy_y / K_y[h1] + psi_sxy_y(gidz,h1,gidx);
            psi_syy_y(gidz,h1,gidx) = b_y_half[h1] * psi_syy_y(gidz,h1,gidx) + a_y_half[h1] * syy_y;
            syy_y = syy_y / K_y_half[h1] + psi_syy_y(gidz,h1,gidx);
            psi_syz_y(gidz,h1,gidx) = b_y[h1] * psi_syz_y(gidz,h1,gidx) + a_y[h1] * syz_y;
            syz_y = syz_y / K_y[h1] + psi_syz_y(gidz,h1,gidx);
        }
        
        
        if (gidz-fdoh<nab && freesurf==0){
            psi_sxz_z(gidz,gidy,gidx) = b_z[gidz] * psi_sxz_z(gidz,gidy,gidx) + a_z[gidz] * sxz_z;
            sxz_z = sxz_z / K_z[gidz] + psi_sxz_z(gidz,gidy,gidx);
            psi_syz_z(gidz,gidy,gidx) = b_z[gidz] * psi_syz_z(gidz,gidy,gidx) + a_z[gidz] * syz_z;
            syz_z = syz_z / K_z[gidz] + psi_syz_z(gidz,gidy,gidx);
            psi_szz_z(gidz,gidy,gidx) = b_z_half[gidz] * psi_szz_z(gidz,gidy,gidx) + a_z_half[gidz] * szz_z;
            szz_z = szz_z / K_z_half[gidz] + psi_szz_z(gidz,gidy,gidx);}
        
        
        if (gidz>NZ-nab-fdoh-1){
            h1 = (gidz-NZ+2*nab+fdoh);
            psi_sxz_z(h1,gidy,gidx) = b_z[h1] * psi_sxz_z(h1,gidy,gidx) + a_z[h1] * sxz_z;
            sxz_z = sxz_z / K_z[h1] + psi_sxz_z(h1,gidy,gidx);
            psi_syz_z(h1,gidy,gidx) = b_z[h1] * psi_syz_z(h1,gidy,gidx) + a_z[h1] * syz_z;
            syz_z = syz_z / K_z[h1] + psi_syz_z(h1,gidy,gidx);
            psi_szz_z(h1,gidy,gidx) = b_z_half[h1] * psi_szz_z(h1,gidy,gidx) + a_z_half[h1] * szz_z;
            szz_z = szz_z / K_z_half[h1] + psi_szz_z(h1,gidy,gidx);}

    }
#endif
    
#if dev==num_devices-1
    if (gidx>NX-nab-fdoh-1){
        
        h1 = (gidx-NX+2*nab+fdoh);
        
        psi_sxx_x(gidz,gidy,h1) = b_x_half[h1] * psi_sxx_x(gidz,gidy,h1) + a_x_half[h1] * sxx_x;
        sxx_x = sxx_x / K_x_half[h1] + psi_sxx_x(gidz,gidy,h1);
        psi_sxy_x(gidz,gidy,h1) = b_x[h1] * psi_sxy_x(gidz,gidy,h1) + a_x[h1] * sxy_x;
        sxy_x = sxy_x / K_x[h1] + psi_sxy_x(gidz,gidy,h1);
        psi_sxz_x(gidz,gidy,h1) = b_x[h1] * psi_sxz_x(gidz,gidy,h1) + a_x[h1] * sxz_x;
        sxz_x = sxz_x / K_x[h1] + psi_sxz_x(gidz,gidy,h1);
        
        if (gidy-fdoh<=nab){
            psi_sxy_y(gidz,gidy,gidx) = b_y[gidy] * psi_sxy_y(gidz,gidy,gidx) + a_y[gidy] * sxy_y;
            sxy_y = sxy_y / K_y[gidy] + psi_sxy_y(gidz,gidy,gidx);
            psi_syy_y(gidz,gidy,gidx) = b_y_half[gidy] * psi_syy_y(gidz,gidy,gidx) + a_y_half[gidy] * syy_y;
            syy_y = syy_y / K_y_half[gidy] + psi_syy_y(gidz,gidy,gidx);
            psi_syz_y(gidz,gidy,gidx) = b_y[gidy] * psi_syz_y(gidz,gidy,gidx) + a_y[gidy] * syz_y;
            syz_y = syz_y / K_y[gidy] + psi_syz_y(gidz,gidy,gidx); 	 }
        
        
        if (gidy>NY-nab-fdoh-1){
            h1 = (gidy-NY+2*nab+fdoh);
            psi_sxy_y(gidz,h1,gidx) = b_y[h1] * psi_sxy_y(gidz,h1,gidx) + a_y[h1] * sxy_y;
            sxy_y = sxy_y / K_y[h1] + psi_sxy_y(gidz,h1,gidx);
            psi_syy_y(gidz,h1,gidx) = b_y_half[h1] * psi_syy_y(gidz,h1,gidx) + a_y_half[h1] * syy_y;
            syy_y = syy_y / K_y_half[h1] + psi_syy_y(gidz,h1,gidx);
            psi_syz_y(gidz,h1,gidx) = b_y[h1] * psi_syz_y(gidz,h1,gidx) + a_y[h1] * syz_y;
            syz_y = syz_y / K_y[h1] + psi_syz_y(gidz,h1,gidx);
        }
        
        
        if (gidz-fdoh<nab && freesurf==0){
            psi_sxz_z(gidz,gidy,gidx) = b_z[gidz] * psi_sxz_z(gidz,gidy,gidx) + a_z[gidz] * sxz_z;
            sxz_z = sxz_z / K_z[gidz] + psi_sxz_z(gidz,gidy,gidx);
            psi_syz_z(gidz,gidy,gidx) = b_z[gidz] * psi_syz_z(gidz,gidy,gidx) + a_z[gidz] * syz_z;
            syz_z = syz_z / K_z[gidz] + psi_syz_z(gidz,gidy,gidx);
            psi_szz_z(gidz,gidy,gidx) = b_z_half[gidz] * psi_szz_z(gidz,gidy,gidx) + a_z_half[gidz] * szz_z;
            szz_z = szz_z / K_z_half[gidz] + psi_szz_z(gidz,gidy,gidx);}
        
        
        if (gidz>NZ-nab-fdoh-1){
            h1 = (gidz-NZ+2*nab+fdoh);
            psi_sxz_z(h1,gidy,gidx) = b_z[h1] * psi_sxz_z(h1,gidy,gidx) + a_z[h1] * sxz_z;
            sxz_z = sxz_z / K_z[h1] + psi_sxz_z(h1,gidy,gidx);
            psi_syz_z(h1,gidy,gidx) = b_z[h1] * psi_syz_z(h1,gidy,gidx) + a_z[h1] * syz_z;
            syz_z = syz_z / K_z[h1] + psi_syz_z(h1,gidy,gidx);
            psi_szz_z(h1,gidy,gidx) = b_z_half[h1] * psi_szz_z(h1,gidy,gidx) + a_z_half[h1] * szz_z;
            szz_z = szz_z / K_z_half[h1] + psi_szz_z(h1,gidy,gidx);}

    }
#endif
    
    
    vx(gidz,gidy,gidx)+= ((sxx_x + sxy_y + sxz_z)/rip(gidz,gidy,gidx))+amp.x;
    vy(gidz,gidy,gidx)+= ((syy_y + sxy_x + syz_z)/rjp(gidz,gidy,gidx))+amp.y;
    vz(gidz,gidy,gidx)+= ((szz_z + sxz_x + syz_y)/rkp(gidz,gidy,gidx))+amp.z;
    
 
}


