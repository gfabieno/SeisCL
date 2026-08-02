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
    
    // Choose the right parameters depending on the dimensions
    if (m->ND!=21){
        if (m->par_type==0){
            if (m->L>0)
                c_calc=&grad_coefvisc_0;
            else
                c_calc=&grad_coefelast_0;
        }
        else if (m->par_type==1){
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
        if (m->par_type==0){
            if (m->L>0)
                c_calc=&grad_coefvisc_0_SH;
            else
                c_calc=&grad_coefelast_0_SH;
        }
        else if (m->par_type==1){
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
                            fsyzr[indfd]=cl_diff2(fsyzr[indfd], cl_integral(fryz[indL],freq));
                            
                            
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
                        
                        /* The c[16..23] group must carry the *same* sign as
                         * the gradM/gradmu expressions above -- see the
                         * identical, already-fixed ND==2 branch below
                         * (calc_grad.c, "The c[16..23] group is the
                         * parameterization chain rule" comment) for the full
                         * derivation. That fix, made while validating the 2D
                         * on-device DFT kernel, was never ported to this
                         * ND==3 branch because nothing exercised
                         * back_prop_type=2 in 3D until grad_dft3D.cl existed.
                         * Found by comparing grad_dft3D.cl (which already
                         * uses the correct sign, copied from the fixed 2D
                         * kernel) against this host oracle: gradvp/gradvs
                         * matched to fp32 precision but gradrho did not,
                         * isolated to exactly this group -- see
                         * notes/3d-gradient-findings.md, "Item 6". */
                         gradrho[indm]+=-dot[8]
                                        -c[16]*dot[0]
                                        +c[17]*dot[1]
                                        -c[18]*dot[2]
                                        +c[19]*dot[3]
                                        -c[20]*dot[4]
                                        +c[21]*dot[5]
                                        -c[22]*dot[6]
                                        +c[23]*dot[7];

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

                    dot[8]=freq*(cl_itreal( fvxr[indfd], fvx[indfd] ) + cl_itreal( fvzr[indfd], fvz[indfd] ))/dftnorm;
                    
                    
                    gradM[indm]+= -c[0]*dot[0]
                                  +c[1]*dot[1];

                    gradmu[indm]+=-c[2]*dot[2]
                                 +c[3]*dot[3]
                                 -c[4]*dot[4]
                                 +c[5]*dot[5]
                                 -c[6]*dot[6]
                                 +c[7]*dot[7];
                    
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
                    
                    /* The c[16..23] group is the parameterization chain rule:
                     * for par_type=0, M = rho*vp^2 and mu = rho*vs^2 both depend
                     * on rho, so d(J)/d(rho) at fixed vp,vs picks up
                     * vp^2*gradM + vs^2*gradmu on top of the density kernel
                     * -dot[8]. c[16] is vp^2/den and c[18..20] carry the mu/rho
                     * = vs^2 factor, so these terms must enter with the *same*
                     * signs as the gradM and gradmu expressions above -- which
                     * is what transf_grad does for back_prop_type=1
                     * (calc_grad.c:1036-1041: gradrho += M/rho*gradM and
                     * += mu/rho*gradmu, both positive). The group was negated,
                     * which left gradrho anti-correlated with the time-domain
                     * gradient (cos=-0.64) while gradrho in the (M,mu,rho)
                     * parameterization -- where this group is absent -- agreed
                     * at cos=0.99. */
                    gradrho[indm]+=-dot[8]
                                    -c[16]*dot[0]
                                    +c[17]*dot[1]
                                    -c[18]*dot[2]
                                    +c[19]*dot[3]
                                    -c[20]*dot[4]
                                    +c[21]*dot[5]
                                    -c[22]*dot[6]
                                    +c[23]*dot[7];
                    
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
                        
                        Hrho[indm]+=dot[8]
                                    -c[16]*dot[0]
                                    +c[17]*dot[1]
                                    -c[18]*dot[2]
                                    +c[19]*dot[3]
                                    -c[20]*dot[4]
                                    +c[21]*dot[5]
                                    -c[22]*dot[6]
                                    +c[23]*dot[7];
                        
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
                    
                    /* Same chain-rule sign as the P-SV case: c[4] is
                     * (mu/rho)/mu^2 = vs^2 * (internal coefficient of dot[0]),
                     * so it must carry the same sign as gradmu's -c[0]*dot[0],
                     * matching transf_grad's gradrho += mu/rho*gradmu. */
                    gradrho[indm]+=-dot[2] -c[4]*dot[0]+c[5]*dot[1]  ;
                    
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

int transf_grad(model * m) {
    //TODO perform forward and back transform to replace Init_model and trans_grad
    int state=0;
    int i, j, num_ele=0;
    half * hpar;
    
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

    int scaler=m->par_scale;
    
    if (m->FP16>1){
        for (i=0;i<m->npars;i++){
            hpar = (half*)m->pars[i].gl_par;
            for (j=m->pars[i].num_ele-1;j>=0;j--){
                m->pars[i].gl_par[j] = half_to_float(hpar[j]);
            }
            
        }
    }
    
    for (i=0;i<num_ele;i++){
        rho[i]= 1.0/rho[i]*m->dt/m->dh*powf(2,-scaler);
        gradrho[i]/=m->dt;
    }
    if (M){
        for (i=0;i<num_ele;i++){
            M[i]*=m->dh/m->dt*powf(2,-scaler);
            gradM[i]/=m->dt;
        }
    }
    if (mu){
        for (i=0;i<num_ele;i++){
            mu[i]*=m->dh/m->dt*powf(2,-scaler);
            gradmu[i]/=m->dt;
        }
    }

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



