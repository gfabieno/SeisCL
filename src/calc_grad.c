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
#include "third_party/NVIDIA_FP16/fp16_conversion.h"

/*Gradient calculation in the frequency domain */


//TODO Write gradient computation in frequency for CUDA
#ifdef __SEISCL__
//Some functions to perfom complex operations with OpenCl vectors

static inline float
cl_itreal(cl_float2 a, cl_float2 b)
{
    float output =(a.s[1]*b.s[0]-a.s[0]*b.s[1]);
    return output;
}

static inline cl_float2
cl_add(cl_float2 a, cl_float2 b, cl_float2 c)
{
    cl_float2 output;
    output.s[0]=a.s[0]+b.s[0]+c.s[0];
    output.s[1]=a.s[1]+b.s[1]+c.s[1];
    return output;
}
static inline cl_float2
cl_diff(cl_float2 a, cl_float2 b, cl_float2 c)
{
    cl_float2 output;
    output.s[0]=a.s[0]-b.s[0]-c.s[0];
    output.s[1]=a.s[1]-b.s[1]-c.s[1];
    return output;
}

static inline cl_float2
cl_add2(cl_float2 a, cl_float2 b)
{
    cl_float2 output;
    output.s[0]=a.s[0]+b.s[0];
    output.s[1]=a.s[1]+b.s[1];
    return output;
}
static inline cl_float2
cl_diff2(cl_float2 a, cl_float2 b)
{
    cl_float2 output;
    output.s[0]=a.s[0]-b.s[0];
    output.s[1]=a.s[1]-b.s[1];
    return output;
}

static inline float
cl_rm(cl_float2 a,cl_float2 b, float tausig, float w)
{

    return tausig*(a.s[0]*b.s[0]+a.s[1]*b.s[1])+(a.s[0]*b.s[1]-a.s[1]*b.s[0])/w;
}

static inline cl_float2
cl_stat(cl_float2 a, float dt, float nf, float Nt)
{
    float fcos=cosf(2*PI*dt*nf/Nt);
    float fsin=sinf(2*PI*dt*nf/Nt);
    cl_float2 output;
    output.s[0]=a.s[0]*fcos-a.s[1]*fsin;
    output.s[1]=a.s[0]*fsin+a.s[1]*fcos;
    return output;
}
static inline cl_float2
cl_integral(cl_float2 a, float w)
{
    cl_float2 output;
    output.s[0]=a.s[1]/w;
    output.s[1]=-a.s[0]/w;
    return output;
}
static inline cl_float2
cl_derivative(cl_float2 a, float w)
{
    cl_float2 output;
    output.s[0]=-a.s[1]*w;
    output.s[1]=a.s[0]*w;
    return output;
}
static inline float
cl_norm(cl_float2 a)
{
    return pow(a.s[0],2)+pow(a.s[1],2);
}

// Coefficient of the scalar products
int grad_coefvisc_0(double (*c)[24],
                    float M,
                    float mu,
                    float taup,
                    float taus,
                    float rho,
                    float ND,
                    float L,
                    float al){
    
    double fact1=pow(ND*M*(1.0+L*taup)*(1.0+al*taus)-2.0*(ND-1.0)*mu*(1.0+L*taus)*(1.0+al*taup),2.0);
    double fact2=pow(ND*M*taup*(1.0+al*taus)-2.0*(ND-1.0)*mu*taus*(1.0+al*taup),2.0);

    (*c)[0]= 2.0*sqrtf(rho*M)*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[1]= 2.0*sqrtf(rho*M)*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[2]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[3]= 2.0*sqrtf(rho*mu)*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[4]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[5]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( mu*mu*taus );
    (*c)[6]= 2.0*sqrtf(rho*mu)*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[7]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    (*c)[8]= M*(L-al)*pow(1+al*taus,2)/fact1;
    (*c)[9]= M*pow(1+al*taus,2)/fact2;
    (*c)[10]= (L-al)/( mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[11]= (ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2)/fact1;
    (*c)[12]= (L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[13]= 1.0/( mu*taus*taus );
    (*c)[14]= (ND+1.0)/3.0*mu*pow(1+al*taup,2)/fact2;
    (*c)[15]= 1.0/( 2*ND*mu*taus*taus );
    
    (*c)[16]= M/rho*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[17]= M/rho*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[18]= mu/rho*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[19]= mu/rho*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[20]= mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[21]= mu/rho*(1.0+al*taus)/( mu*mu*taus );
    (*c)[22]= mu/rho*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[23]= mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    


    return 1;
}
int grad_coefelast_0(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    (*c)[0]= 2.0*sqrtf(rho*M)*1.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    
    (*c)[2]= 2.0*sqrtf(rho*mu)*1.0/( mu*mu);
    (*c)[3]= 2.0*sqrtf(rho*mu)*(ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[4]= 2.0*sqrtf(rho*mu)*1.0/( 2*ND*mu*mu );
    
    (*c)[16]= M/rho*1.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    
    (*c)[18]= mu/rho*1.0/( mu*mu);
    (*c)[19]= mu/rho*(ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[20]= mu/rho*1.0/( 2*ND*mu*mu );
    
    return 1;
}
int grad_coefvisc_1(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    
    
    double fact1=pow(ND*M*(1.0+L*taup)*(1.0+al*taus)-2.0*(ND-1.0)*mu*(1.0+L*taus)*(1.0+al*taup),2.0);
    double fact2=pow(ND*M*taup*(1.0+al*taus)-2.0*(ND-1.0)*mu*taus*(1.0+al*taup),2.0);
    
    (*c)[0]= (1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[1]= taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[2]= (1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[3]= (ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[4]= (1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[5]= (1.0+al*taus)/( mu*mu*taus );
    (*c)[6]= (ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[7]= (1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    (*c)[8]= M*(L-al)*pow(1+al*taus,2)/fact1;
    (*c)[9]= M*pow(1+al*taus,2)/fact2;
    (*c)[10]= (L-al)/( mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[11]= (ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2)/fact1;
    (*c)[12]= (L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[13]= 1.0/( mu*taus*taus );
    (*c)[14]= (ND+1.0)/3.0*mu*pow(1+al*taup,2)/fact2;
    (*c)[15]= 1.0/( 2*ND*mu*taus*taus );
    
    return 1;
}
int grad_coefelast_1(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    (*c)[0]= 1.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    
    (*c)[2]= 1.0/( mu*mu);
    (*c)[3]= (ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[4]= 1.0/( 2*ND*mu*mu );
    
    
    return 1;
}
int grad_coefvisc_2(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    
    
    double fact1=pow(ND*M*(1.0+L*taup)*(1.0+al*taus)-2.0*(ND-1.0)*mu*(1.0+L*taus)*(1.0+al*taup),2.0);
    double fact2=pow(ND*M*taup*(1.0+al*taus)-2.0*(ND-1.0)*mu*taus*(1.0+al*taup),2.0);
    
    (*c)[0]= 2.0*sqrtf(M/rho)*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[1]= 2.0*sqrtf(M/rho)*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[2]= 2.0*sqrtf(mu/rho)*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[3]= 2.0*sqrtf(mu/rho)*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[4]= 2.0*sqrtf(mu/rho)*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[5]= 2.0*sqrtf(mu/rho)*(1.0+al*taus)/( mu*mu*taus );
    (*c)[6]= 2.0*sqrtf(mu/rho)*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[7]= 2.0*sqrtf(mu/rho)*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    (*c)[8]= M*(L-al)*pow(1+al*taus,2)/fact1;
    (*c)[9]= M*pow(1+al*taus,2)/fact2;
    (*c)[10]= (L-al)/( mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[11]= (ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2)/fact1;
    (*c)[12]= (L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[13]= 1.0/( mu*taus*taus );
    (*c)[14]= (ND+1.0)/3.0*mu*pow(1+al*taup,2)/fact2;
    (*c)[15]= 1.0/( 2*ND*mu*taus*taus );
    
    (*c)[16]= -M/rho*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[17]= -M/rho*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[18]= -mu/rho*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[19]= -mu/rho*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[20]= -mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[21]= -mu/rho*(1.0+al*taus)/( mu*mu*taus );
    (*c)[22]= -mu/rho*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[23]= -mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    
    return 1;
}
int grad_coefelast_2(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    (*c)[0]= 2.0*sqrtf(M/rho)*1.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    
    (*c)[2]= 2.0*sqrtf(mu/rho)*1.0/( mu*mu);
    (*c)[3]= 2.0*sqrtf(mu/rho)*(ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[4]= 2.0*sqrtf(mu/rho)*1.0/( 2*ND*mu*mu );
    
    (*c)[16]= -M/rho*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2);
    
    (*c)[18]= -mu/rho*1.0/( mu*mu);
    (*c)[19]= -mu/rho*(ND+1.0)/3.0/pow(ND*M-2.0*(ND-1.0)*mu,2.0);
    (*c)[20]= -mu/rho*1.0/( 2*ND*mu*mu );
    
    return 1;
}
int grad_coefvisc_3(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    
    
    double fact1=pow(ND*M*(1.0+L*taup)*(1.0+al*taus)-2.0*(ND-1.0)*mu*(1.0+L*taus)*(1.0+al*taup),2.0);
    double fact2=pow(ND*M*taup*(1.0+al*taus)-2.0*(ND-1.0)*mu*taus*(1.0+al*taup),2.0);
    
    (*c)[0]= (2.0*sqrtf(rho*M)*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2) -taup/sqrtf(M/rho)*M*(L-al)*pow(1+al*taus,2))/fact1;
    (*c)[1]= (2.0*sqrtf(rho*M)*taup*(1.0+al*taup)*pow(1.0+al*taus,2) - taup/sqrtf(M/rho)*M*pow(1+al*taus,2) ) /fact2;
    (*c)[2]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/(mu*mu*(1.0+L*taus))  -  taup/sqrtf(M/rho)*(L-al)/( mu*(1.0+L*taus)*(1.0+L*taus))   ;
    (*c)[3]= (2.0*sqrtf(rho*mu)*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)-  taup/sqrtf(M/rho)*(ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2))/fact1 ;
    (*c)[4]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) )-taup/sqrtf(M/rho)*(L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[5]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( mu*mu*taus ) - taup/sqrtf(M/rho)*1.0/( mu*taus*taus ) ;
    (*c)[6]= (2.0*sqrtf(rho*mu)*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)  - taup/sqrtf(M/rho)*(ND+1.0)/3.0*mu*pow(1+al*taup,2) )/fact2;
    (*c)[7]= 2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*taus ) - taup/sqrtf(M/rho)*1.0/( 2*ND*mu*taus*taus );
    (*c)[8]= (-2.0*sqrtf(rho*M)*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2) + (1.0+taup)/sqrtf(M/rho)*M*(L-al)*pow(1+al*taus,2))/fact1;
    (*c)[9]= (-2.0*sqrtf(rho*M)*taup*(1.0+al*taup)*pow(1.0+al*taus,2) + (1.0+taup)/sqrtf(M/rho)*M*pow(1+al*taus,2) ) /fact2;
    (*c)[10]= -2.0*sqrtf(rho*mu)*(1.0+al*taus)/(mu*mu*(1.0+L*taus))  + (1.0+taup)/sqrtf(M/rho)*(L-al)/( mu*(1.0+L*taus)*(1.0+L*taus))   ;
    (*c)[11]= (-2.0*sqrtf(rho*mu)*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)+ (1.0+taup)/sqrtf(M/rho)*(ND+1.0)/3.0*mu*(L-al)*pow(1+al*taup,2))/fact1 ;
    (*c)[12]= -2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) )+ (1.0+taup)/sqrtf(M/rho)*(L-al)/( 2*ND*mu*(1.0+L*taus)*(1.0+L*taus) );
    (*c)[13]= -2.0*sqrtf(rho*mu)*(1.0+al*taus)/( mu*mu*taus ) + (1.0+taup)/sqrtf(M/rho)*1.0/( mu*taus*taus ) ;
    (*c)[14]= (-2.0*sqrtf(rho*mu)*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)  + (1.0+taup)/sqrtf(M/rho)*(ND+1.0)/3.0*mu*pow(1+al*taup,2) )/fact2;
    (*c)[15]= -2.0*sqrtf(rho*mu)*(1.0+al*taus)/( 2.0*ND*mu*mu*taus ) + (1.0+taup)/sqrtf(M/rho)*1.0/( 2*ND*mu*taus*taus );
    
    (*c)[16]= M/rho*(1.0+L*taup)*(1.0+al*taup)*pow(1.0+al*taus,2)/fact1;
    (*c)[17]= M/rho*taup*(1.0+al*taup)*pow(1.0+al*taus,2)/fact2;
    (*c)[18]= mu/rho*(1.0+al*taus)/(mu*mu*(1.0+L*taus));
    (*c)[19]= mu/rho*(ND+1.0)/3.0*(1.0+L*taus)*(1.0+al*taus)*pow(1.0+al*taup,2)/fact1;
    (*c)[20]= mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*(1.0+L*taus) );
    (*c)[21]= mu/rho*(1.0+al*taus)/( mu*mu*taus );
    (*c)[22]= mu/rho*(ND+1.0)/3*taus*(1.0+al*taus)*pow(1.0+al*taup,2)/fact2;
    (*c)[23]= mu/rho*(1.0+al*taus)/( 2.0*ND*mu*mu*taus );
    
    return 1;
}
int grad_coefvisc_0_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 2.0*sqrtf(rho*mu)*(1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[1]= 2.0*sqrtf(rho*mu)*(1+al*taus)/taus/pow(mu,2);
    (*c)[2]= (L-al)/pow(1+L*taus,2)/mu;
    (*c)[3]= 1/pow(taus,2)/mu;
    
    (*c)[4]= mu/rho*(1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[5]= mu/rho*(1+al*taus)/taus/pow(mu,2);


    
    return 1;
}
int grad_coefelast_0_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 2.0*sqrtf(rho*mu)/pow(mu,2);
    
    (*c)[4]= mu/rho/pow(mu,2);

    
    
    
    return 1;
}
int grad_coefvisc_1_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){

    
    (*c)[0]= (1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[1]= (1+al*taus)/taus/pow(mu,2);
    (*c)[2]= (L-al)/pow(1+L*taus,2)/mu;
    (*c)[3]= 1/pow(taus,2)/mu;

    return 1;
}
int grad_coefelast_1_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 1.0/pow(mu,2);

    
    return 1;
}
int grad_coefvisc_2_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 2.0*sqrtf(mu/rho)*(1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[1]= 2.0*sqrtf(mu/rho)*(1+al*taus)/taus/pow(mu,2);
    (*c)[2]= (L-al)/pow(1+L*taus,2)/mu;
    (*c)[3]= 1/pow(taus,2)/mu;
    
    (*c)[4]= -mu/rho*(1+al*taus)/(1+L*taus)/pow(mu,2);
    (*c)[5]= -mu/rho*(1+al*taus)/taus/pow(mu,2);
    
    return 1;
}
int grad_coefelast_2_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    
    
    (*c)[0]= 2.0*sqrtf(mu/rho)/pow(mu,2);
    
    (*c)[4]= -mu/rho*(1+al*taus)/(1+L*taus)/pow(mu,2);
    
    return 1;
}
int grad_coefvisc_3_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    //A faire

    
    return 1;
}
int grad_coefelast_3_SH(double (*c)[24],float M, float mu, float taup, float taus, float rho, float ND, float L, float al){
    //A faire
    
    
    return 1;
}


int calc_grad(model * m, device * dev)  {
    
    int i,j,k,f,l, n;
    float df,freq,ND, al,w0;
    double c[24]={0}, dot[17]={0};
    /* Coefficient of the sxz correlation, evaluated at muipkp rather than the
       cell-centred mu. Declared out here, alongside c[], because the
       coefficients are computed in one block and consumed in the frequency
       loop that follows it. */
    double imuipkp2=0.0;
    float * tausigl=NULL;
    cl_float2 sxxyyzz, sxxyyzzr, sxx_myyzz, syy_mxxzz, szz_mxxyy;
    cl_float2 rxxyyzz, rxxyyzzr, rxx_myyzz, ryy_mxxzz, rzz_mxxyy;
    
    cl_float2 sxxzz, sxxzzr, sxx_mzz, szz_mxx;
    cl_float2 rxxzz, rxxzzr, rxx_mzz, rzz_mxx;

    int NX=0, NY=0, NZ=0;
    int indfd, indm, indL;
    
    int (*c_calc)(double (*c)[24],
                  float M,
                  float mu,
                  float taup,
                  float taus,
                  float rho,
                  float ND,
                  float L,
                  float al)=NULL;
    
    ND=(float)m->ND;
    df=1.0/m->NTNYQ/m->dt/m->DTNYQ;
    /* Parseval normalization for the frequency-domain dot products. With
     * A_k = sum_n a_n * dteff * exp(-2i.pi.kn/N) and dteff = DTNYQ*dt,
     * sum_n a_n b_n dteff = (1/(N*dteff)) sum_k A_k conj(B_k), so the factor is
     * 1/(NTNYQ*DTNYQ*dt), not 1/NTNYQ. The missing DTNYQ was invisible while
     * DTNYQ was always 1, but it scales the whole gradient linearly with DTNYQ:
     * measured error was exactly DTNYQ-1 across a dft_osamp sweep. Any run with
     * a low fmax or a small dt, where DTNYQ > 1, was affected. */
    double dftnorm = (double)m->NTNYQ*(double)m->DTNYQ;
    
    w0=2.0*PI*m->f0;
    al=0;
    float * gradfreqsn = get_cst( m->csts, m->ncsts, "gradfreqsn")->gl_cst;
    
    if (m->L>0){
        tausigl=malloc(sizeof(float)*m->L);
        float * FL = get_cst( m->csts,m->ncsts, "FL")->gl_cst;
        for (l=0;l<m->L;l++){
            tausigl[l]=  1.0/(2.0*PI*FL[l]);
            al+=      pow(w0/(2.0*PI*FL[l]),2)
                /(1.0+pow(w0/(2.0*PI*FL[l]),2));
        }
    }
    
    /* The correlation always emits the *internal* (M, mu, rho) gradient; the
       parameterization chain rule is chain_rule_par_type()'s job, run once
       after the shot loop. The _1 family is exactly the _0 family with the
       chain-rule factors stripped (compare grad_coefelast_1 with
       grad_coefelast_0: c[0] loses its 2*sqrt(rho*M), and c[16..23] -- which
       were c[0..7] again, times M/rho or mu/rho -- disappear), so par_type==0
       simply selects it too.

       This removes a whole bug class rather than a single bug. The c[16..23]
       group had to be kept sign-consistent with the gradM/gradmu expressions
       by hand, and was not: the elastic block carries a documented fix for
       exactly that, while the viscoelastic block above it still had the
       opposite signs -- the third instance in this file of a fix applied to
       one branch and never ported to its twin. Deriving the density
       contribution from gradM/gradmu in one place makes the two impossible to
       disagree. */
    if (m->ND!=21){
        if (m->par_type==0 || m->par_type==1){
            if (m->L>0)
                c_calc=&grad_coefvisc_1;
            else
                c_calc=&grad_coefelast_1;
        }
        else if (0){
            if (m->L>0)
                c_calc=&grad_coefvisc_1;
            else
                c_calc=&grad_coefelast_1;
        }
        else if (m->par_type==2){
            if (m->L>0)
                c_calc=&grad_coefvisc_2;
            else
                c_calc=&grad_coefelast_2;
        }
        else if (m->par_type==3){
            c_calc=&grad_coefvisc_3;
            
        }
    }
    else if (m->ND==21){
        if (m->par_type==0 || m->par_type==1){
            if (m->L>0)
                c_calc=&grad_coefvisc_1_SH;
            else
                c_calc=&grad_coefelast_1_SH;
        }
        else if (0){
            if (m->L>0)
                c_calc=&grad_coefvisc_1_SH;
            else
                c_calc=&grad_coefelast_1_SH;
        }
        else if (m->par_type==2){
            if (m->L>0)
                c_calc=&grad_coefvisc_2_SH;
            else
                c_calc=&grad_coefelast_2_SH;
        }
        else if (m->par_type==3){
            c_calc=&grad_coefvisc_3_SH;
            
        }
        
        
    }
    
    cl_float2 * fvx=NULL;
    cl_float2 * fvy=NULL;
    cl_float2 * fvz=NULL;
    
    cl_float2 * fvxr=NULL;
    cl_float2 * fvyr=NULL;
    cl_float2 * fvzr=NULL;
    
    cl_float2 * fsxx=NULL;
    cl_float2 * fsyy=NULL;
    cl_float2 * fszz=NULL;
    cl_float2 * fsxy=NULL;
    cl_float2 * fsxz=NULL;
    cl_float2 * fsyz=NULL;
    
    cl_float2 * fsxxr=NULL;
    cl_float2 * fsyyr=NULL;
    cl_float2 * fszzr=NULL;
    cl_float2 * fsxyr=NULL;
    cl_float2 * fsxzr=NULL;
    cl_float2 * fsyzr=NULL;
    
    cl_float2 * frxx=NULL;
    cl_float2 * fryy=NULL;
    cl_float2 * frzz=NULL;
    cl_float2 * frxy=NULL;
    cl_float2 * frxz=NULL;
    cl_float2 * fryz=NULL;
    
    cl_float2 * frxxr=NULL;
    cl_float2 * fryyr=NULL;
    cl_float2 * frzzr=NULL;
    cl_float2 * frxyr=NULL;
    cl_float2 * frxzr=NULL;
    cl_float2 * fryzr=NULL;
    
    for (i=0;i<m->nvars;i++){
        
        if (strcmp(dev->vars[i].name,"vx")==0){
            fvx=(cl_float2*)dev->vars[i].cl_fvar.host;
            fvxr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"vy")==0){
            fvy=(cl_float2*)dev->vars[i].cl_fvar.host;
            fvyr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"vz")==0){
            fvz=(cl_float2*)dev->vars[i].cl_fvar.host;
            fvzr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"sxx")==0){
            fsxx=(cl_float2*)dev->vars[i].cl_fvar.host;
            fsxxr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"syy")==0){
            fsyy=(cl_float2*)dev->vars[i].cl_fvar.host;
            fsyyr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"szz")==0){
            fszz=(cl_float2*)dev->vars[i].cl_fvar.host;
            fszzr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"sxy")==0){
            fsxy=(cl_float2*)dev->vars[i].cl_fvar.host;
            fsxyr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"sxz")==0){
            fsxz=(cl_float2*)dev->vars[i].cl_fvar.host;
            fsxzr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"syz")==0){
            fsyz=(cl_float2*)dev->vars[i].cl_fvar.host;
            fsyzr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"rxx")==0){
            frxx=(cl_float2*)dev->vars[i].cl_fvar.host;
            frxxr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"ryy")==0){
            fryy=(cl_float2*)dev->vars[i].cl_fvar.host;
            fryyr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"rzz")==0){
            frzz=(cl_float2*)dev->vars[i].cl_fvar.host;
            frzzr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"rxy")==0){
            frxy=(cl_float2*)dev->vars[i].cl_fvar.host;
            frxyr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"rxz")==0){
            frxz=(cl_float2*)dev->vars[i].cl_fvar.host;
            frxzr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        if (strcmp(dev->vars[i].name,"ryz")==0){
            fryz=(cl_float2*)dev->vars[i].cl_fvar.host;
            fryzr=(cl_float2*)dev->vars[i].cl_fvar_adj.host;
        }
        
    }
    
    float *rho=NULL, *gradrho=NULL, *Hrho=NULL;
    float *M=NULL, *gradM=NULL, *HM=NULL;
    float *mu=NULL, *gradmu=NULL, *Hmu=NULL;
    float *muipkp=NULL, *gradmuipkp=NULL;
    float *gradrip=NULL, *gradrkp=NULL;
    float *taup=NULL, *gradtaup=NULL, *Htaup=NULL;
    float *taus=NULL, *gradtaus=NULL, *Htaus=NULL;
    
    for (i=0;i<m->npars;i++){
        if (strcmp(dev->pars[i].name,"rho")==0){
            rho=dev->pars[i].cl_par.host;
            gradrho=dev->pars[i].cl_grad.host;
            Hrho=dev->pars[i].cl_H.host;
        }
        if (strcmp(dev->pars[i].name,"M")==0){
            M=dev->pars[i].cl_par.host;
            gradM=dev->pars[i].cl_grad.host;
            HM=dev->pars[i].cl_H.host;
        }
        if (strcmp(dev->pars[i].name,"mu")==0){
            mu=dev->pars[i].cl_par.host;
            gradmu=dev->pars[i].cl_grad.host;
            Hmu=dev->pars[i].cl_H.host;
        }
        if (strcmp(dev->pars[i].name,"muipkp")==0){
            muipkp=dev->pars[i].cl_par.host;
            gradmuipkp=dev->pars[i].cl_grad.host;
        }
        if (strcmp(dev->pars[i].name,"rip")==0){
            gradrip=dev->pars[i].cl_grad.host;
        }
        if (strcmp(dev->pars[i].name,"rkp")==0){
            gradrkp=dev->pars[i].cl_grad.host;
        }
        if (strcmp(dev->pars[i].name,"taup")==0){
            taup=dev->pars[i].cl_par.host;
            gradtaup=dev->pars[i].cl_grad.host;
            Htaup=dev->pars[i].cl_H.host;
        }
        if (strcmp(dev->pars[i].name,"taus")==0){
            taus=dev->pars[i].cl_par.host;
            gradtaus=dev->pars[i].cl_grad.host;
            Htaus=dev->pars[i].cl_H.host;
        }
    }
    
    if (ND==3){
        NX=dev->N[2];
        NY=dev->N[1];
        NZ=dev->N[0];
    }
    else{
        NX=dev->N[1];
        NZ=dev->N[0];
    }
    
    if (ND==3){
        for (i=0;i<NX;i++){
            for (j=0;j<NY;j++){
                for (k=0;k<NZ;k++){
                    indm=i*NY*NZ+j*NZ+k;

                    /* Undo the internal non-dimensionalization before feeding
                     * the grad_coef* formulas, which are expressions in the
                     * *physical* stiffnesses and density -- cl_par.host holds
                     * the internally non-dimensionalized values. This mirrors
                     * the ND==2 branch below (calc_grad.c, "Undo the
                     * transform" comment) and the on-device grad_dft3D.cl
                     * kernel, both of which already do this. The ND==3 host
                     * branch never got this fix because back_prop_type=2 had
                     * no 3D path to exercise it until grad_dft3D.cl; without
                     * it this reference disagrees with the (correct) device
                     * kernel by a scale factor that depends on par_scale,
                     * dh, dt and the local density -- not a clean constant,
                     * which is what made it look like a device-kernel bug at
                     * first. */
                    double s2 = pow(2.0, -(double)m->par_scale);
                    double dhdt = (double)m->dh/(double)m->dt;
                    double rho_p = (rho[indm]!=0.0)
                                 ? (1.0/rho[indm])*((double)m->dt/(double)m->dh)*s2
                                 : 0.0;
                    double M_p   = M  ? M[indm]*dhdt*s2  : 0.0;
                    double mu_p  = mu ? mu[indm]*dhdt*s2 : 0.0;
                    double taup_p = (m->L>0 && taup) ? taup[indm] : 0.0;
                    double taus_p = (m->L>0 && taus) ? taus[indm] : 0.0;
                    /* Vacuum cells (M == mu == rho == 0): skip rather than
                     * evaluate, same guard and same reason as ND==2 -- see
                     * ../notes/back-prop-type1-zero-material-nan.md. */
                    double den_p = ND*M_p - 2.0*(ND-1.0)*mu_p;
                    if (!(rho_p>0.0) || !(den_p*den_p>0.0)){
                        for (n=0;n<24;n++) c[n]=0;
                    }
                    else{
                        c_calc(&c, M_p, mu_p, taup_p, taus_p, rho_p, ND, m->L, al);
                        /* Fluid cells: drop every shear-related coefficient.
                         * Tested on the physical mu, not the scaled one. */
                        if (mu_p<1.0){
                            for (n=2;n<8;n++)   c[n]=0;
                            for (n=10;n<16;n++) c[n]=0;
                            for (n=18;n<24;n++) c[n]=0;
                        }
                    }

                    for (f=0;f<m->NFREQS;f++){

                        indfd= f*(NX+m->FDORDER)*(NY+m->FDORDER)*(NZ+m->FDORDER)
                             +(i+m->FDOH)*(NY+m->FDORDER)*(NZ+m->FDORDER)
                             +(j+m->FDOH)*(NZ+m->FDORDER)
                             +(k+m->FDOH);

                        freq=2.0*PI*df* gradfreqsn[f];

                        dot[1]=0;dot[5]=0;dot[6]=0;dot[7]=0;
                        for (l=0;l<m->L;l++){
                            indL= f*(NX+m->FDORDER)*(NY+m->FDORDER)*(NZ+m->FDORDER)*m->L
                                +l*(NX+m->FDORDER)*(NY+m->FDORDER)*(NZ+m->FDORDER)
                                +(i+m->FDOH)*(NY+m->FDORDER)*(NZ+m->FDORDER)
                                +(j+m->FDOH)*(NZ+m->FDORDER)
                                +(k+m->FDOH);
                            fsxx[indfd]=cl_diff2(fsxx[indfd], cl_integral(frxx[indL],freq));
                            fszz[indfd]=cl_diff2(fszz[indfd], cl_integral(frzz[indL],freq));
                            fsyy[indfd]=cl_diff2(fsyy[indfd], cl_integral(fryy[indL],freq));
                            fsxz[indfd]=cl_diff2(fsxz[indfd], cl_integral(frxz[indL],freq));
                            fsxy[indfd]=cl_diff2(fsxy[indfd], cl_integral(frxy[indL],freq));
                            fsyz[indfd]=cl_diff2(fsyz[indfd], cl_integral(fryz[indL],freq));
                            
                            fsxxr[indfd]=cl_diff2(fsxxr[indfd], cl_integral(frxxr[indL],freq));
                            fszzr[indfd]=cl_diff2(fszzr[indfd], cl_integral(frzzr[indL],freq));
                            fsyyr[indfd]=cl_diff2(fsyyr[indfd], cl_integral(fryyr[indL],freq));
                            fsxzr[indfd]=cl_diff2(fsxzr[indfd], cl_integral(frxzr[indL],freq));
                            fsxyr[indfd]=cl_diff2(fsxyr[indfd], cl_integral(frxyr[indL],freq));
                            /* fryzr, not fryz: the adjoint syz must be corrected with the
                             * *adjoint* memory variable. The other five lines
                             * above use their r-suffixed spectrum; this one was
                             * copy-pasted from the forward block. Same class as
                             * the rxx_myyzz/ryy_mxxzz/rzz_mxxyy paste fixed just
                             * below, and unreachable until a 3D viscoelastic DFT
                             * gradient existed to exercise it. */
                            fsyzr[indfd]=cl_diff2(fsyzr[indfd], cl_integral(fryzr[indL],freq));
                            
                            
                            rxxyyzz=    cl_add(frxx[indL], fryy[indL], frzz[indL]);
                            rxxyyzzr=   cl_add(frxxr[indL], fryyr[indL], frzzr[indL]);
                            /* Each memory variable's own stress minus the sum
                             * of the other two, matching sxx_myyzz/syy_mxxzz/
                             * szz_mxxyy below. Previously all three lines
                             * were copy-pasted as cl_diff(frxx,fryy,frzz)
                             * (todo item 6 / notes/3d-gradient-findings.md),
                             * so ryy_mxxzz and rzz_mxxyy silently held the
                             * same value as rxx_myyzz. Unreachable except
                             * through calc_grad's L>0 (viscoelastic) DFT
                             * path, which has no on-device kernel yet. */
                            rxx_myyzz= cl_diff(frxx[indL], fryy[indL], frzz[indL]);
                            ryy_mxxzz= cl_diff(fryy[indL], frxx[indL], frzz[indL]);
                            rzz_mxxyy= cl_diff(frzz[indL], frxx[indL], fryy[indL]);
                            dot[1]+=cl_rm( rxxyyzzr, rxxyyzz, tausigl[l],freq )/dftnorm;
                            
                            dot[5]+=(+cl_rm( frxyr[indL], frxy[indL] , tausigl[l],freq)
                                     +cl_rm( frxzr[indL], frxz[indL] , tausigl[l],freq)
                                     +cl_rm( fryzr[indL], fryz[indL] , tausigl[l],freq))/dftnorm;
                            dot[6]=dot[1];
                            dot[7]+=(+cl_rm( frxxr[indL], rxx_myyzz , tausigl[l],freq)
                                     +cl_rm( fryyr[indL], ryy_mxxzz , tausigl[l],freq)
                                     +cl_rm( frzzr[indL], rzz_mxxyy , tausigl[l],freq))/dftnorm;
                        }
                        
                        sxxyyzz=    cl_add(fsxx[indfd], fsyy[indfd], fszz[indfd]);
                        sxxyyzzr=   cl_add(fsxxr[indfd],fsyyr[indfd],fszzr[indfd]);
                        sxx_myyzz= cl_diff(fsxx[indfd], fsyy[indfd], fszz[indfd]);
                        syy_mxxzz= cl_diff(fsyy[indfd], fsxx[indfd], fszz[indfd]);
                        szz_mxxyy= cl_diff(fszz[indfd], fsxx[indfd], fsyy[indfd]);

                        dot[0]=freq*cl_itreal( sxxyyzzr, sxxyyzz )/dftnorm;
                        dot[2]=freq*(+cl_itreal( fsxyr[indfd], fsxy[indfd] )
                                     +cl_itreal( fsxzr[indfd], fsxz[indfd] )
                                     +cl_itreal( fsyzr[indfd], fsyz[indfd] ))/dftnorm;
                        dot[3]=dot[0];
                        dot[4]=freq*(+cl_itreal( fsxxr[indfd], sxx_myyzz )
                                     +cl_itreal( fsyyr[indfd], syy_mxxzz )
                                     +cl_itreal( fszzr[indfd], szz_mxxyy ))/dftnorm;

                        
                        dot[8]=freq*(
                                     cl_itreal( fvxr[indfd], fvx[indfd] ) +
                                     cl_itreal( fvyr[indfd], fvy[indfd] ) +
                                     cl_itreal( fvzr[indfd], fvz[indfd] )
                                     )/dftnorm;
                        
                        gradM[indm]+=   -c[0]*dot[0]
                                        +c[1]*dot[1];
                        gradmu[indm]+=  -c[2]*dot[2]
                                        +c[3]*dot[3]
                                        -c[4]*dot[4]
                                        +c[5]*dot[5]
                                        -c[6]*dot[6]
                                        +c[7]*dot[7];
                        
                        if (m->L>0){
                             gradtaup[indm]+=-c[8]*dot[0]
                                             +c[9]*dot[1];
                             gradtaus[indm]+=-c[10]*dot[2]
                                             +c[11]*dot[3]
                                             -c[12]*dot[4]
                                             +c[13]*dot[5]
                                             -c[14]*dot[6]
                                             +c[15]*dot[7];
                        }
                        
                    /* Density gets the velocity correlation and nothing
                     * else. The parameterization's dependence of M and mu on
                     * rho is chain_rule_par_type()'s job now (gradrho +=
                     * M/rho*gradM + mu/rho*gradmu), derived from the very
                     * numbers accumulated above.
                     *
                     * This replaces a parallel c[16..23] group that had to be
                     * kept sign-consistent with gradM/gradmu by hand, and was
                     * not: the ND==2 branch carried a fix that this ND==3 one
                     * never received, found by comparing grad_dft3D.cl
                     * against this host oracle (gradvp/gradvs matched to fp32
                     * while gradrho did not, isolated to exactly this group --
                     * notes/3d-gradient-findings.md, "Item 6"). Deriving it in
                     * one place makes that class of divergence impossible; the
                     * _1 coefficient family the selection now uses does not
                     * define c[16..23] at all. */
                         gradrho[indm]+=-dot[8];

                    }
                    
                }
            }
        }
        
    }
    else if (ND==2){
        
        for (i=0;i<NX;i++){
            for (k=0;k<NZ;k++){
                /* The grad_coef* formulas are expressions in the *physical*
                 * stiffnesses and density, but cl_par.host holds the internally
                 * non-dimensionalized parameters. Feeding those in directly made
                 * every coefficient wrong, and -- because the formulas are
                 * nonlinear (sqrt, 1/mu^2) -- wrong by a *different* factor per
                 * coefficient: with par_scale==0 the c[0]/c[2] group came out 5x
                 * too large while c[16] came out 4e14x too large, so gradvp and
                 * gradvs were off by a clean constant while gradrho was garbage.
                 * Undo the transform, using the same relations transf_grad()
                 * applies in reverse (calc_grad.c:1004-1019).
                 * Hoisted out of the frequency loop: none of this depends on f. */
                indm=i*NZ+k;
                {
                    double s2 = pow(2.0, -(double)m->par_scale);
                    double dhdt = (double)m->dh/(double)m->dt;
                    double rho_p = (rho[indm]!=0.0)
                                 ? (1.0/rho[indm])*((double)m->dt/(double)m->dh)*s2
                                 : 0.0;
                    double M_p   = M  ? M[indm]*dhdt*s2  : 0.0;
                    double mu_p  = mu ? mu[indm]*dhdt*s2 : 0.0;
                    /* sxz is driven by muipkp, not the cell-centred mu, so
                     * its correlation's coefficient is evaluated here and its
                     * gradient stored at that staggered slot. Mirrors
                     * grad_dft2D.cl. */
                    double muipkp_p = muipkp ? muipkp[indm]*dhdt*s2 : 0.0;
                    imuipkp2 = (muipkp_p>=1.0)
                             ? 1.0/(muipkp_p*muipkp_p) : 0.0;
                    double taup_p = (m->L>0 && taup) ? taup[indm] : 0.0;
                    double taus_p = (m->L>0 && taus) ? taus[indm] : 0.0;
                    /* Vacuum cells (M == mu == rho == 0) contribute nothing,
                     * and must be skipped rather than evaluated: the coefficient
                     * formulas divide by rho and by (ND*M-2(ND-1)mu)^2, so a
                     * zero cell yields 0/0 = NaN which then propagates through
                     * the whole gradient. The device kernel is already immune
                     * because its (ND*M-2(ND-1)mu)^2 > 0 guard short-circuits
                     * there; this keeps the host reference in step, so
                     * SEISCL_DFT_CHECK does not compare against NaN. See
                     * ../notes/back-prop-type1-zero-material-nan.md, which
                     * root-causes the same class of failure in transf_grad()
                     * for BACK_PROP_TYPE==1. */
                    double den_p = ND*M_p - 2.0*(ND-1.0)*mu_p;
                    if (!(rho_p>0.0) || !(den_p*den_p>0.0)){
                        for (n=0;n<24;n++) c[n]=0;
                    }
                    else{
                    c_calc(&c, M_p, mu_p, taup_p, taus_p, rho_p, ND, m->L, al);

                    /* Fluid cells: drop every shear-related coefficient. Tested
                     * on the physical mu, not the scaled one. */
                    if (mu_p<1.0){
                        for (n=2;n<8;n++)   c[n]=0;
                        for (n=10;n<16;n++) c[n]=0;
                        for (n=18;n<24;n++) c[n]=0;
                    }
                    }
                }
                for (f=0;f<m->NFREQS;f++){

                    indfd= f*(NX+m->FDORDER)*(NZ+m->FDORDER)
                         +(i+m->FDOH)*(NZ+m->FDORDER)
                         +(k+m->FDOH);

                    freq=2.0*PI*df* gradfreqsn[f];

                    dot[1]=0;dot[5]=0;dot[6]=0;dot[7]=0;
                    for (l=0;l<m->L;l++){
                        indL= f*(NX+m->FDORDER)*(NZ+m->FDORDER)*m->L
                            +l*(NX+m->FDORDER)*(NZ+m->FDORDER)
                            +(i+m->FDOH)*(NZ+m->FDORDER)
                            +(k+m->FDOH);

                        fsxx[indfd]=cl_diff2(fsxx[indfd], cl_integral(frxx[indL],freq) );
                        fszz[indfd]=cl_diff2(fszz[indfd], cl_integral(frzz[indL],freq) );
                        fsxz[indfd]=cl_diff2(fsxz[indfd], cl_integral(frxz[indL],freq) );
                        fsxxr[indfd]=cl_diff2(fsxxr[indfd], cl_integral(frxxr[indL],freq) );
                        fszzr[indfd]=cl_diff2(fszzr[indfd], cl_integral(frzzr[indL],freq) );
                        fsxzr[indfd]=cl_diff2(fsxzr[indfd], cl_integral(frxzr[indL],freq) );
                        
                        rxxzz=    cl_add2(frxx[indL], frzz[indL]);
                        rxxzzr=   cl_add2(frxxr[indL],frzzr[indL]);
                        rxx_mzz= cl_diff2(frxx[indL], frzz[indL]);
                        rzz_mxx= cl_diff2(frzz[indL], frxx[indL]);
                        
                        dot[1]+=cl_rm( rxxzzr, rxxzz, tausigl[l],freq )/dftnorm;
                        
                        dot[5]+=(cl_rm( frxzr[indL], frxz[indL] , tausigl[l],freq) )/dftnorm;
                        dot[6]=dot[1];
                        dot[7]+=(+cl_rm( frxxr[indL], rxx_mzz , tausigl[l],freq)
                                 +cl_rm( frzzr[indL], rzz_mxx , tausigl[l],freq))/dftnorm;
                        
                    }
                    sxxzz=    cl_add2(fsxx[indfd], fszz[indfd]);
                    sxxzzr=   cl_add2(fsxxr[indfd],fszzr[indfd]);
                    sxx_mzz= cl_diff2(fsxx[indfd], fszz[indfd]);
                    szz_mxx= cl_diff2(fszz[indfd], fsxx[indfd]);
                    

                    
                    dot[0]=freq*cl_itreal( sxxzzr, sxxzz )/dftnorm;
                    dot[2]=freq* ( cl_itreal( fsxzr[indfd], fsxz[indfd])  )/dftnorm;
                    dot[3]=dot[0];
                    dot[4]=freq*(+cl_itreal( fsxxr[indfd], sxx_mzz )
                                 +cl_itreal( fszzr[indfd], szz_mxx ))/dftnorm;

                    /* vx sits at the rip position, vz at rkp: different
                     * parameters, so they are not summed. */
                    dot[8]=freq*cl_itreal( fvxr[indfd], fvx[indfd] )/dftnorm;
                    dot[9]=freq*cl_itreal( fvzr[indfd], fvz[indfd] )/dftnorm;
                    
                    
                    gradM[indm]+= -c[0]*dot[0]
                                  +c[1]*dot[1];

                    gradmu[indm]+= +c[3]*dot[3]
                                 -c[4]*dot[4]
                                 +c[5]*dot[5]
                                 -c[6]*dot[6]
                                 +c[7]*dot[7];
                    /* The staggered split is elastic-only for now. imuipkp2
                       is 1/muipkp^2, which equals c[2] only when L==0; the
                       viscoelastic c[2] is (1+al*taus)/(muipkp^2*(1+L*taus)),
                       and its taus twin c[10] would likewise have to go to a
                       tausipkp slot rather than to gradtaus. Doing that needs
                       the averaged taus as well, so until then L>0 keeps the
                       pre-existing cell-centred accumulation -- which is also
                       what grad_dft2D_visc.cl does, so device and host stay
                       comparable under SEISCL_DFT_CHECK. */
                    if (m->L==0){
                        if (gradmuipkp) gradmuipkp[indm]+= -imuipkp2*dot[2];
                        if (gradrip) gradrip[indm]+= -dot[8];
                        if (gradrkp) gradrkp[indm]+= -dot[9];
                    }
                    else {
                        gradmu[indm]  += -c[2]*dot[2];
                        gradrho[indm] += -(dot[8]+dot[9]);
                    }
                    
                    if (m->L>0){
                        gradtaup[indm]+= -c[8]*dot[0]
                                        +c[9]*dot[1];
                        gradtaus[indm]+= -c[10]*dot[2]
                                        +c[11]*dot[3]
                                        -c[12]*dot[4]
                                        +c[13]*dot[5]
                                        -c[14]*dot[6]
                                        +c[15]*dot[7];
                    }
                    
                    /* gradrho gets no correlation term: density enters the
                     * physics only through rip/rkp, accumulated above. It is
                     * filled by average_grad_transpose and then by
                     * chain_rule_par_type's M/rho and mu/rho terms. */
                    
                    if(m->HOUT){
                        dot[1]=0;dot[5]=0;dot[6]=0;dot[7]=0;
                        for (l=0;l<m->L;l++){
                            indL= f*(NX+m->FDORDER)*(NZ+m->FDORDER)*m->L
                            +l*(NX+m->FDORDER)*(NZ+m->FDORDER)
                            +(i+m->FDOH)*(NZ+m->FDORDER)
                            +(k+m->FDOH);

                            rxxzz=    cl_add2(frxx[indL], frzz[indL]);
                            rxx_mzz= cl_diff2(frxx[indL], frzz[indL]);
                            rzz_mxx= cl_diff2(frzz[indL], frxx[indL]);
                            
                            dot[1]+=cl_norm(cl_add2( rxxzz, cl_derivative(rxxzz, freq*tausigl[l])) )/dftnorm;
                            dot[5]+=cl_norm(cl_add2( frxz[indL], cl_derivative(frxz[indL], freq*tausigl[l])) )/dftnorm;
                            dot[6]=dot[1];
                            dot[7]+=(cl_norm(cl_add2( rxx_mzz, cl_derivative(rxx_mzz, freq*tausigl[l])) )
                                    +cl_norm(cl_add2( rzz_mxx, cl_derivative(rzz_mxx, freq*tausigl[l])) ))/dftnorm;
                            
                        }
                        sxxzz=    cl_add2(fsxx[indfd], fszz[indfd]);
                        sxx_mzz= cl_diff2(fsxx[indfd], fszz[indfd]);
                        szz_mxx= cl_diff2(fszz[indfd], fsxx[indfd]);
                        
                        
                        dot[0]=cl_norm(cl_derivative(sxxzz, freq))/dftnorm;
                        dot[2]=cl_norm(cl_derivative(fsxz[indfd], freq))/dftnorm;
                        dot[3]=dot[0];
                        dot[4]=(cl_norm(cl_derivative(sxx_mzz, freq))
                                    +cl_norm(cl_derivative(szz_mxx, freq)))/dftnorm;
                        dot[8]=(cl_norm(cl_derivative(fvx[indfd], freq))
                                +cl_norm(cl_derivative(fvz[indfd], freq)))/dftnorm;
                        
                        HM[indm]+=   c[0]*dot[0]
                                    -c[1]*dot[1];
                        Hmu[indm]+=  c[2]*dot[2]
                                    -c[3]*dot[3]
                                    +c[4]*dot[4]
                                    -c[5]*dot[5]
                                    +c[6]*dot[6]
                                    -c[7]*dot[7];
                        
                        if (m->L>0){
                            Htaup[indm]+=c[8]*dot[0]
                                        -c[9]*dot[1];
                            Htaus[indm]+=c[10]*dot[2]
                                        -c[11]*dot[3]
                                        +c[12]*dot[4]
                                        -c[13]*dot[5]
                                        +c[14]*dot[6]
                                        -c[15]*dot[7];
                        }
                        
                        /* As for the gradient: chain_rule_par_type()
                         * carries M and mu's rho-dependence over. */
                        Hrho[indm]+=dot[8];
                        
                    }
                    
                    
                }
            }
        }
        
    }
    else if (ND==21){
        
        for (i=0;i<NX;i++){
            for (k=0;k<NZ;k++){
                for (f=0;f<m->NFREQS;f++){
                    
                    indfd= f*(NX+m->FDORDER)*(NZ+m->FDORDER)
                    +(i+m->FDOH)*(NZ+m->FDORDER)
                    +(k+m->FDOH);
                    indm=i*NZ+k;
                    
                    freq=2.0*PI*df* gradfreqsn[f];
                    if (m->L>0)
                        c_calc(&c,M[indm], mu[indm], taup[indm], taus[indm], rho[indm], ND,m->L,al);
                    else
                        c_calc(&c,M[indm], mu[indm], 0, 0, rho[indm], ND,m->L,al);
                    
                    
                    dot[0]=freq*(cl_itreal(fsxyr[indfd],fsxy[indfd])+ cl_itreal(fsyzr[indfd],fsyz[indfd]) )/dftnorm;

                    for (l=0;l<m->L;l++){
                        indL= f*(NX+m->FDORDER)*(NZ+m->FDORDER)*m->L
                        +l*(NX+m->FDORDER)*(NZ+m->FDORDER)
                        +(i+m->FDOH)*(NZ+m->FDORDER)
                        +(k+m->FDOH);
                        dot[1]=(cl_rm( frxyr[indL], frxy[indL],tausigl[l],freq )+cl_rm( fryzr[indL], fryz[indL],tausigl[l],freq ))/dftnorm;
                    }
                    
                    dot[2]=freq*(cl_itreal( fvyr[indfd], fvy[indfd] ))/dftnorm;
                    

                    gradmu[indm]+=-c[0]*dot[0]+c[1]*dot[1];
                    
                    if (m->L>0){
                        gradtaus[indm]+=-c[2]*dot[0]+c[3]*dot[1];
                    }
                    
                    /* Velocity correlation only -- see the P-SV block. */
                    gradrho[indm]+=-dot[2];
                    
                }
            }
        }
        
    }
    
    
    if (tausigl) free(tausigl);
    return 0;
    
}
#else
int calc_grad(struct model * m, struct device * dev){
    return 0;
}
#endif

/* ---------------------------------------------------------------------------
   Transpose of the material-parameter averaging.

   The FD physics is evaluated at staggered, averaged parameters (rip/rjp/rkp
   from the buoyancy, muipkp/muipjp/mujpkp from mu, tausipkp/... from taus), so
   that is where the correlation accumulates its gradient. Mapping those back
   onto the cell-centred parameters the user inverts for is the transpose of
   the averaging operator's Jacobian.

   Written as a scatter, mirroring the forward routines in
   assign_modeling_case.c loop for loop -- including the trailing copy region
   each of them ends with, where the Jacobian is 1 rather than the averaging
   formula. Specified and dot-tested independently in
   SeisCL/tests/dot_prod_average.py.

   Must run *before* unscale_par(): the Jacobians are evaluated at the stored
   (internal) parameter values, which unscale_par overwrites. It commutes with
   unscale_grad(), which is a uniform factor on a linear operator.
   --------------------------------------------------------------------------*/

static void grad_T_arithmetic_rho(float * pin, float * gin, float * gout,
                                  int * N, int ndim, int * dir){
    int i,j,k;
    int NX, NY, NZ;
    int NX0=0, NY0=0, NZ0=0;
    int ind1, ind2;
    double avg, s;
    if (ndim==3){ NX=N[2]; NY=N[1]; NZ=N[0]; }
    else        { NX=N[1]; NY=1;    NZ=N[0]; }

    /* pout = 2/(1/p1 + 1/p2)  =>  d(pout)/d(p_j) = (pout^2/2)/p_j^2 */
    for (k=0;k<NZ-dir[0];k++){
        for (j=0;j<NY-dir[1];j++){
            for (i=0;i<NX-dir[2];i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                ind2 = (i+dir[2])*NY*NZ+(j+dir[1])*NZ+(k+dir[0]);
                s = 1.0/pin[ind1] + 1.0/pin[ind2];
                if (s==0.0) continue;
                avg = 2.0/s;
                gout[ind1] += gin[ind1]*0.5*avg*avg/(pin[ind1]*pin[ind1]);
                gout[ind2] += gin[ind1]*0.5*avg*avg/(pin[ind2]*pin[ind2]);
            }
        }
    }
    if (dir[2]==1)      NX0=NX-1;
    else if (dir[1]==1) NY0=NY-1;
    if (dir[0]==1)      NZ0=NZ-1;
    for (k=NZ0;k<NZ;k++){
        for (j=NY0;j<NY;j++){
            for (i=NX0;i<NX;i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                gout[ind1] += gin[ind1];
            }
        }
    }
}

static void grad_T_harmonic_mu(float * pin, float * gin, float * gout,
                               int * N, int ndim, int (*dir)[3]){
    int i,j,k,d;
    int NX, NY, NZ;
    int NX0=0, NY0=0, NZ0=0;
    int ind[4];
    double avg, s;
    if (ndim==3){ NX=N[2]; NY=N[1]; NZ=N[0]; }
    else        { NX=N[1]; NY=1;    NZ=N[0]; }

    /* pout = 4/sum(1/p_j)  =>  d(pout)/d(p_j) = (pout^2/4)/p_j^2, and all four
       are zero in a vacuum cell, where ave_harmonic_mu forces pout to 0. */
    for (k=0;k<NZ-dir[0][0]-dir[1][0];k++){
        for (j=0;j<NY-dir[0][1]-dir[1][1];j++){
            for (i=0;i<NX-dir[0][2]-dir[1][2];i++){
                ind[0] = (i)*NY*NZ+(j)*NZ+(k);
                ind[1] = (i+dir[0][2])*NY*NZ+(j+dir[0][1])*NZ+(k+dir[0][0]);
                ind[2] = (i+dir[1][2])*NY*NZ+(j+dir[1][1])*NZ+(k+dir[1][0]);
                ind[3] = (i+dir[0][2]+dir[1][2])*NY*NZ
                        +(j+dir[0][1]+dir[1][1])*NZ
                        +(k+dir[0][0]+dir[1][0]);
                if (pin[ind[0]]==0 || pin[ind[1]]==0
                    || pin[ind[2]]==0 || pin[ind[3]]==0){
                    continue;
                }
                s = 1.0/pin[ind[0]] + 1.0/pin[ind[1]]
                  + 1.0/pin[ind[2]] + 1.0/pin[ind[3]];
                if (s==0.0) continue;
                avg = 4.0/s;
                for (d=0;d<4;d++){
                    gout[ind[d]] += gin[ind[0]]*0.25*avg*avg
                                    /(pin[ind[d]]*pin[ind[d]]);
                }
            }
        }
    }
    for (d=0;d<2;d++){
        NX0=0; NY0=0; NZ0=0;
        if (dir[d][2]==1)      NX0=NX-1;
        else if (dir[d][1]==1) NY0=NY-1;
        if (dir[d][0]==1)      NZ0=NZ-1;
        for (k=NZ0;k<NZ;k++){
            for (j=NY0;j<NY;j++){
                for (i=NX0;i<NX;i++){
                    ind[0] = (i)*NY*NZ+(j)*NZ+(k);
                    gout[ind[0]] += gin[ind[0]];
                }
            }
        }
    }
}

static void grad_T_arithmetic_tau(float * gin, float * gout,
                                  int * N, int ndim, int (*dir)[3]){
    int i,j,k,d;
    int NX, NY, NZ;
    int NX0=0, NY0=0, NZ0=0;
    int ind[4];
    if (ndim==3){ NX=N[2]; NY=N[1]; NZ=N[0]; }
    else        { NX=N[1]; NY=1;    NZ=N[0]; }

    /* Plain 4-point arithmetic mean: every Jacobian entry is 0.25. */
    for (k=0;k<NZ-dir[0][0]-dir[1][0];k++){
        for (j=0;j<NY-dir[0][1]-dir[1][1];j++){
            for (i=0;i<NX-dir[0][2]-dir[1][2];i++){
                ind[0] = (i)*NY*NZ+(j)*NZ+(k);
                ind[1] = (i+dir[0][2])*NY*NZ+(j+dir[0][1])*NZ+(k+dir[0][0]);
                ind[2] = (i+dir[1][2])*NY*NZ+(j+dir[1][1])*NZ+(k+dir[1][0]);
                ind[3] = (i+dir[0][2]+dir[1][2])*NY*NZ
                        +(j+dir[0][1]+dir[1][1])*NZ
                        +(k+dir[0][0]+dir[1][0]);
                for (d=0;d<4;d++){
                    gout[ind[d]] += 0.25*gin[ind[0]];
                }
            }
        }
    }
    for (d=0;d<2;d++){
        NX0=0; NY0=0; NZ0=0;
        if (dir[d][2]==1)      NX0=NX-1;
        else if (dir[d][1]==1) NY0=NY-1;
        if (dir[d][0]==1)      NZ0=NZ-1;
        for (k=NZ0;k<NZ;k++){
            for (j=NY0;j<NY;j++){
                for (i=NX0;i<NX;i++){
                    ind[0] = (i)*NY*NZ+(j)*NZ+(k);
                    gout[ind[0]] += gin[ind[0]];
                }
            }
        }
    }
}

/* One helper so a missing parameter (SH has no M, elastic has no taus, ...)
   is skipped rather than dereferenced. */
static float * par_grad(model * m, const char * name, float ** value){
    parameter * p = get_par(m->pars, m->npars, name);
    if (!p || !p->gl_grad) return NULL;
    if (value) *value = p->gl_par;
    return p->gl_grad;
}

int average_grad_transpose(model * m) {
    int state=0;
    float * gsrc; float * psrc;
    float * gavg; float * pavg;
    int d1[3];
    int d2[2][3];

    /* buoyancy -> rip / rjp / rkp */
    gsrc = par_grad(m, "rho", &psrc);
    if (gsrc){
        gavg = par_grad(m, "rip", &pavg);
        if (gavg){ d1[0]=0; d1[1]=0; d1[2]=1;
            grad_T_arithmetic_rho(psrc, gavg, gsrc, m->N, m->NDIM, d1); }
        gavg = par_grad(m, "rjp", &pavg);
        if (gavg){ d1[0]=0; d1[1]=1; d1[2]=0;
            grad_T_arithmetic_rho(psrc, gavg, gsrc, m->N, m->NDIM, d1); }
        gavg = par_grad(m, "rkp", &pavg);
        if (gavg){ d1[0]=1; d1[1]=0; d1[2]=0;
            grad_T_arithmetic_rho(psrc, gavg, gsrc, m->N, m->NDIM, d1); }
    }

    /* mu -> muipkp / muipjp / mujpkp */
    gsrc = par_grad(m, "mu", &psrc);
    if (gsrc){
        gavg = par_grad(m, "muipkp", &pavg);
        if (gavg){ d2[0][0]=0; d2[0][1]=0; d2[0][2]=1;
                   d2[1][0]=1; d2[1][1]=0; d2[1][2]=0;
            grad_T_harmonic_mu(psrc, gavg, gsrc, m->N, m->NDIM, d2); }
        gavg = par_grad(m, "mujpkp", &pavg);
        if (gavg){ d2[0][0]=0; d2[0][1]=1; d2[0][2]=0;
                   d2[1][0]=1; d2[1][1]=0; d2[1][2]=0;
            grad_T_harmonic_mu(psrc, gavg, gsrc, m->N, m->NDIM, d2); }
        gavg = par_grad(m, "muipjp", &pavg);
        if (gavg){ d2[0][0]=0; d2[0][1]=0; d2[0][2]=1;
                   d2[1][0]=0; d2[1][1]=1; d2[1][2]=0;
            grad_T_harmonic_mu(psrc, gavg, gsrc, m->N, m->NDIM, d2); }
    }

    /* taus -> tausipkp / tausipjp / tausjpkp */
    gsrc = par_grad(m, "taus", &psrc);
    if (gsrc){
        gavg = par_grad(m, "tausipkp", &pavg);
        if (gavg){ d2[0][0]=0; d2[0][1]=0; d2[0][2]=1;
                   d2[1][0]=1; d2[1][1]=0; d2[1][2]=0;
            grad_T_arithmetic_tau(gavg, gsrc, m->N, m->NDIM, d2); }
        gavg = par_grad(m, "tausjpkp", &pavg);
        if (gavg){ d2[0][0]=0; d2[0][1]=1; d2[0][2]=0;
                   d2[1][0]=1; d2[1][1]=0; d2[1][2]=0;
            grad_T_arithmetic_tau(gavg, gsrc, m->N, m->NDIM, d2); }
        gavg = par_grad(m, "tausipjp", &pavg);
        if (gavg){ d2[0][0]=0; d2[0][1]=0; d2[0][2]=1;
                   d2[1][0]=0; d2[1][1]=1; d2[1][2]=0;
            grad_T_arithmetic_tau(gavg, gsrc, m->N, m->NDIM, d2); }
    }

    return state;
}

/* unpack_par_fp16, unscale_par_grad and chain_rule_par_type used to be one
   function. They are three unrelated jobs -- storage format, units, and
   parameterization -- and fusing them made it impossible to say which of them
   a given caller wanted. BACK_PROP_TYPE==2 in particular needs the
   parameterization but not the scaling (its correlation kernel already
   converts to physical units on the device), and the forthcoming
   material-averaging transpose has to run *between* the scaling and the
   parameterization. Keeping them separate is what makes those compositions
   expressible; transf_grad() below is simply all three in the original order,
   so existing callers are unaffected. */

/* FP16>1 stores the parameters as half in the same buffer. Expand in place,
   backwards, so the wider floats do not overwrite unread halves. */
int unpack_par_fp16(model * m) {
    int state=0;
    int i, j;
    half * hpar;
    if (m->FP16>1){
        for (i=0;i<m->npars;i++){
            hpar = (half*)m->pars[i].gl_par;
            for (j=m->pars[i].num_ele-1;j>=0;j--){
                m->pars[i].gl_par[j] = half_to_float(hpar[j]);
            }
            
        }
    }
    return state;
}

/* Parameters back to physical units: rho from buoyancy, M and mu from their
   dt/dh scaling, both times 2^-par_scale. Every path needs this before
   chain_rule_par_type(), which evaluates its Jacobians at the physical
   values. */
int unscale_par(model * m) {
    int state=0;
    int i, num_ele=0;

    float * rho = get_par(m->pars, m->npars, "rho")->gl_par;
    num_ele = get_par(m->pars, m->npars, "rho")->num_ele;
    float * M = get_par(m->pars, m->npars, "M")->gl_par;
    float * mu = get_par(m->pars, m->npars, "mu")->gl_par;

    int scaler=m->par_scale;

    for (i=0;i<num_ele;i++){
        rho[i]= 1.0/rho[i]*m->dt/m->dh*powf(2,-scaler);
    }
    if (M){
        for (i=0;i<num_ele;i++){
            M[i]*=m->dh/m->dt*powf(2,-scaler);
        }
    }
    if (mu){
        for (i=0;i<num_ele;i++){
            mu[i]*=m->dh/m->dt*powf(2,-scaler);
        }
    }
    return state;
}

/* Drop the dt the time integration put into the gradient. Separate from
   unscale_par() because it is *not* universal: BACK_PROP_TYPE==2 accumulates
   its gradient from frequency-domain spectra with its own normalization
   (dftnorm/DTNYQ) and must not be given this factor, while still needing the
   parameters unscaled above. */
int unscale_grad(model * m) {
    int state=0;
    int i, num_ele=0;

    float * gradrho = get_par(m->pars, m->npars, "rho")->gl_grad;
    num_ele = get_par(m->pars, m->npars, "rho")->num_ele;
    float * M = get_par(m->pars, m->npars, "M")->gl_par;
    float * gradM = get_par(m->pars, m->npars, "M")->gl_grad;
    float * mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float * gradmu = get_par(m->pars, m->npars, "mu")->gl_grad;

    for (i=0;i<num_ele;i++){
        gradrho[i]/=m->dt;
    }
    if (M){
        for (i=0;i<num_ele;i++){
            gradM[i]/=m->dt;
        }
    }
    if (mu){
        for (i=0;i<num_ele;i++){
            gradmu[i]/=m->dt;
        }
    }
    return state;
}

/* Map the internal (M, mu, rho) gradient onto m->par_type. Expects physical
   units, i.e. unscale_par_grad() already run (and, once it exists, the
   material-averaging transpose too -- that maps the staggered gradients onto
   the cell-centred ones, which is a different chain rule and belongs before
   this one). */
int chain_rule_par_type(model * m) {
    int state=0;
    int i, num_ele=0;

    float * rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float * gradrho = get_par(m->pars, m->npars, "rho")->gl_grad;
    float * Hrho = get_par(m->pars, m->npars, "rho")->gl_H;
    num_ele = get_par(m->pars, m->npars, "rho")->num_ele;
    float * M = get_par(m->pars, m->npars, "M")->gl_par;
    float * gradM = get_par(m->pars, m->npars, "M")->gl_grad;
    float * HM = get_par(m->pars, m->npars, "M")->gl_H;
    float * mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float * gradmu = get_par(m->pars, m->npars, "mu")->gl_grad;
    float * Hmu = get_par(m->pars, m->npars, "mu")->gl_H;

    if (m->par_type==0){

        for (i=0;i<num_ele;i++){
            gradrho[i]= gradrho[i]+M[i]/rho[i]*gradM[i];
            if (mu[i]>0){
                gradrho[i]= gradrho[i]+mu[i]/rho[i]*gradmu[i];
            }
        }
        if (Hrho){
            for (i=0;i<num_ele;i++){
                Hrho[i]= Hrho[i]+M[i]/rho[i]*HM[i];
                if (mu[i]>0){
                    Hrho[i]= Hrho[i]+mu[i]/rho[i]*Hmu[i];
                }
            }
        }
        if (M){
            for (i=0;i<num_ele;i++){
                gradM[i]  = 2.0*sqrt((double)rho[i]*(double)M[i])*gradM[i];
            }
        }
        if (HM){
            for (i=0;i<num_ele;i++){
                HM[i]  = 2.0*sqrt((double)rho[i]*(double)M[i])*HM[i];
            }
        }
        if (mu){
            for (i=0;i<num_ele;i++){
                gradmu[i] = 2.0*sqrt((double)rho[i]*(double)mu[i])*gradmu[i];
            }
        }
        if (Hmu){
            for (i=0;i<num_ele;i++){
                Hmu[i] = 2.0*sqrt((double)rho[i]*(double)mu[i])*Hmu[i];
            }
        }
    }
    else if (m->par_type==1){

    }
    else if (m->par_type==2){
        for (i=0;i<num_ele;i++){
            gradrho[i]= gradrho[i]+M[i]/rho[i]*gradM[i];
            if (mu[i]>0){
                gradrho[i]= gradrho[i]+mu[i]/rho[i]*gradmu[i];
            }
        }

        if (Hrho){
            for (i=0;i<num_ele;i++){
                Hrho[i]= Hrho[i]+M[i]/rho[i]*HM[i];
                if (mu[i]>0){
                    Hrho[i]= Hrho[i]+mu[i]/rho[i]*Hmu[i];
                }
            }
        }
        if (M){
            for (i=0;i<num_ele;i++){
                gradM[i]  = 2.0*sqrt((double)M[i]/(double)rho[i])*gradM[i];
            }
        }
        if (HM){
            for (i=0;i<num_ele;i++){
                HM[i]  = 2.0*sqrt((double)M[i]/(double)rho[i])*HM[i];
            }
        }
        if (mu){
            for (i=0;i<num_ele;i++){
                gradmu[i] = 2.0*sqrt((double)mu[i]/(double)rho[i])*gradmu[i];
            }
        }
        if (Hmu){
            for (i=0;i<num_ele;i++){
                Hmu[i] = 2.0*sqrt((double)mu[i]/(double)rho[i])*Hmu[i];
            }
        }
    }
    else{
        fprintf(stdout,"Warning: Gradiant transformation not implemented: ");
        fprintf(stdout,"Outputting grad for M,mu,rho parametrization\n");
    }
    return state;
}

/* The original composition: storage, then units, then parameterization. */
int transf_grad(model * m) {
    int state=0;
    __GUARD unpack_par_fp16(m);
    __GUARD average_grad_transpose(m);
    __GUARD unscale_par(m);
    __GUARD unscale_grad(m);
    __GUARD chain_rule_par_type(m);
    
    
    
    return state;

}



