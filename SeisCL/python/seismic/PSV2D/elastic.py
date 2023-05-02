from SeisCL.python import (ComputeGrid, FunctionGPU, ReversibleFunctionGPU, ReversibleFunction,
                           ComputeRessource)
from SeisCL.python import get_header_stencil, cudacl
from SeisCL.python.seismic.PSV2D.elastic_numpy import ricker, Cerjan
from SeisCL.python.seismic.common.sources import Source
from SeisCL.python.seismic.common.receivers import Receiver
import numpy as np
from copy import copy
from pyopencl.array import max
from SeisCL.python.seismic.common.vel2lame import Velocity2LameGPU
from SeisCL.python.seismic.common.scaling import ScaledParameters
from SeisCL.python.seismic.common.averaging import ArithmeticAveraging, HarmonicAveraging
import matplotlib.pyplot as plt

#TODO interface with backward compatibility with SeisCL
#TODO abs cerjan
#TODO PML


class UpdateStress(ReversibleFunctionGPU):

    def __init__(self, queue, order=8, local_size=(16, 16)):
        super().__init__(queue, local_size=local_size)
        self.header, self.local_header = get_header_stencil(order, 2,
                                                            local_size=local_size,
                                                            with_local_ops=True)
        self.order = order

    def forward(self, vx, vz, sxx, szz, sxz, M, mu, muipkp, backpropagate=0):
        """
        Update the stresses
        """
        src = self.local_header
        src += """        
        float vxx, vzz, vzx, vxz;
                
        // Calculation of the velocity spatial derivatives
        #if LOCAL_OFF==0
            load_local_in(vx, lvar);
            load_local_haloz(vx, lvar);
            load_local_halox(vx, lvar);
            BARRIER
        #endif
        vxx = Dxm(lvar);
        vxz = Dzp(lvar);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vz, lvar);
            load_local_haloz(vz, lvar);
            load_local_halox(vz, lvar);
            BARRIER
        #endif
        vzz = Dzm(lvar);
        vzx = Dxp(lvar);
        
        gridstop(g);
    
        // Update the stresses
        int sign = -2*backpropagate+1; 
        sxz(g.z, g.x) = backpropagate;
        //sxz(g.z, g.x) += sign * (muipkp(g.z, g.x)*(vxz+vzx));
        sxx(g.z, g.x) += sign * ((M(g.z, g.x)*(vxx+vzz))-(2.0*mu(g.z, g.x)*vzz));
        szz(g.z, g.x) += sign * ((M(g.z, g.x)*(vxx+vzz))-(2.0*mu(g.z, g.x)*vxx));
        """
        grid = ComputeGrid(shape=[s - self.order for s in vx.shape],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "forward", grid, vx, vz, sxx, szz, sxz, M, mu,
                       muipkp, backpropagate=backpropagate)
        return sxx, szz, sxz

    def linear(self, vx, vz, sxx, szz, sxz, M, mu, muipkp, backpropagate=0):

        src = self.local_header
        src += """ 
        float vxx, vzz, vzx, vxz;
        float vxx_lin, vzz_lin, vzx_lin, vxz_lin;
        
        // Calculation of the velocity spatial derivatives
        #if LOCAL_OFF==0
            load_local_in(vx, lvar);
            load_local_haloz(vx, lvar);
            load_local_halox(vx, lvar);
            BARRIER
        #endif
        vxx = Dxm(lvar);
        vxz = Dzp(lvar);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vx_lin, lvar);
            load_local_haloz(vx_lin, lvar);
            load_local_halox(vx_lin, lvar);
            BARRIER
        #endif
        vxx_lin = Dxm(lvar);
        vxz_lin = Dzp(lvar);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vz, lvar);
            load_local_haloz(vz, lvar);
            load_local_halox(vz, lvar);
            BARRIER
        #endif
        vzz = Dzm(lvar);
        vzx = Dxp(lvar);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vz_lin, lvar);
            load_local_haloz(vz_lin, lvar);
            load_local_halox(vz_lin, lvar);
            BARRIER
        #endif
        vzz_lin = Dzm(lvar);
        vzx_lin = Dxp(lvar);
        gridstop(g);
        
        sxz_lin(g.z, g.x)+= muipkp(g.z, g.x)*(vxz_lin+vzx_lin)\
                       +muipkp_lin(g.z, g.x)*(vxz+vzx);
        sxx_lin(g.z, g.x)+= M(g.z, g.x)*(vxx_lin+vzz_lin)-(2.0*mu(g.z, g.x)*vzz_lin) \
                       +M_lin(g.z, g.x)*(vxx+vzz)-(2.0*mu_lin(g.z, g.x)*vzz);
        szz_lin(g.z, g.x)+= M(g.z, g.x)*(vxx_lin+vzz_lin)-(2.0*mu(g.z, g.x)*vxx_lin) \
                       +M_lin(g.z, g.x)*(vxx+vzz)-(2.0*mu_lin(g.z, g.x)*vxx);
    """
        grid = ComputeGrid(shape=[s - self.order for s in vx.shape],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "linear", grid,
                       vx, vz, sxx, szz, sxz, M, mu, muipkp)
        return sxx, szz, sxz

    def adjoint(self, vx, vz, sxx, szz, sxz, M, mu, muipkp):

        src = self.local_header
        src += """ 
        float vxx, vzz, vzx, vxz;
        float sxx_x_adj, sxx_z_adj, szz_x_adj, szz_z_adj, sxz_x_adj, sxz_z_adj;
        
        // Calculation of the velocity spatial derivatives
        #if LOCAL_OFF==0
            load_local_in(vx, lvar);
            load_local_haloz(vx, lvar);
            load_local_halox(vx, lvar);
            BARRIER
        #endif
            vxx = Dxm(lvar);
            vxz = Dzp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vz, lvar);
            load_local_haloz(vz, lvar);
            load_local_halox(vz, lvar);
            BARRIER
        #endif
            vzz = Dzm(lvar);
            vzx = Dxp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(sxz_adj, lvar);
            mul_local_in(muipkp, lvar);
            load_local_haloz(sxz_adj, lvar);
            mul_local_haloz(muipkp, lvar);
            load_local_halox(sxz_adj, lvar);
            mul_local_halox(muipkp, lvar);
            BARRIER
        #endif
            sxz_x_adj = -Dxm(lvar);
            sxz_z_adj = -Dzm(lvar);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(sxx_adj, lvar);
            mul_local_in(M, lvar);
            load_local_haloz(sxx_adj, lvar);
            mul_local_haloz(M, lvar);
            load_local_halox(sxx_adj, lvar);
            mul_local_halox(M, lvar);
            BARRIER
        #endif
            sxx_x_adj = -Dxp(lvar);
            sxx_z_adj = -Dzp(lvar);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(sxx_adj, lvar);
            mul_local_in(mu, lvar);
            load_local_haloz(sxx_adj, lvar);
            mul_local_haloz(mu, lvar);
            BARRIER
        #endif
            sxx_z_adj += 2.0*Dzp(lvar);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(szz_adj, lvar);
            mul_local_in(M, lvar);
            load_local_haloz(szz_adj, lvar);
            mul_local_haloz(M, lvar);
            load_local_halox(szz_adj, lvar);
            mul_local_halox(M, lvar);
            BARRIER
        #endif
            szz_x_adj = -Dxp(lvar);
            szz_z_adj = -Dzp(lvar);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(szz_adj, lvar);
            mul_local_in(mu, lvar);
            load_local_halox(szz_adj, lvar);
            mul_local_halox(mu, lvar);
            BARRIER
        #endif
        szz_x_adj += 2.0*Dxp(lvar);
          
        gridstop(g);
        
        vx_adj(g.z, g.x) += sxx_x_adj + szz_x_adj + sxz_z_adj;
        vz_adj(g.z, g.x) += sxx_z_adj + szz_z_adj + sxz_x_adj;
        
        M_adj(g.z, g.x) += (vxx + vzz) * (sxx_adj(g.z, g.x) + szz_adj(g.z, g.x));
        mu_adj(g.z, g.x) += - 2.0 * (vzz * sxx_adj(g.z, g.x) + vxx * szz_adj(g.z, g.x));
        muipkp_adj(g.z, g.x) += (vxz + vzx) * sxz_adj(g.z, g.x);
        """
        grid = ComputeGrid(shape=[s - self.order for s in vx.shape],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "adjoint", grid,
                       vx, vz, sxx, szz, sxz, M, mu, muipkp)
        return vx, vz, M, mu, muipkp


class UpdateVelocity(ReversibleFunctionGPU):

    def __init__(self, queue, order=8, local_size=(16, 16)):
        super().__init__(queue, local_size=local_size)
        self.header, self.local_header = get_header_stencil(order, 2,
                                                            local_size=local_size,
                                                            with_local_ops=True)
        self.order = order

    def forward(self, vx, vz, sxx, szz, sxz, rip, rkp, backpropagate=0):
        """
        Update the velocity field using the stress field.
        """
        src = self.local_header
        src += """

        float sxx_x, szz_z, sxz_x, sxz_z;
                
        // Calculation of the stresses spatial derivatives
        #if LOCAL_OFF==0
            load_local_in(sxx, lvar);
            load_local_halox(sxx, lvar);
            BARRIER
        #endif
        sxx_x = Dxp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(szz, lvar);
            load_local_haloz(szz, lvar);
            BARRIER
        #endif
        szz_z = Dzp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(sxz, lvar);
            load_local_haloz(sxz, lvar);
            load_local_halox(sxz, lvar);
            BARRIER
        #endif
        sxz_z = Dzm(lvar);
        sxz_x = Dxm(lvar);
    
        gridstop(g);
        
        // Update the velocities
        int sign = -2*backpropagate+1; 
        vx(g.z, g.x)+= sign * ((sxx_x + sxz_z)*rip(g.z, g.x));
        vz(g.z, g.x)+= sign * ((szz_z + sxz_x)*rkp(g.z, g.x));
        """
        grid = ComputeGrid(shape=[s - self.order for s in vx.shape],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "forward", grid, vx, vz, sxx, szz, sxz, rip, rkp,
                       backpropagate=backpropagate)
        return vx, vz

    def linear(self, vx, vz, sxx, szz, sxz, rip, rkp):

        src = self.local_header
        src += """ 
        float sxx_x, szz_z, sxz_x, sxz_z;
        float sxx_x_lin, szz_z_lin, sxz_x_lin, sxz_z_lin;
        
        // Calculation of the stresses spatial derivatives
        #if LOCAL_OFF==0
            load_local_in(sxx, lvar);
            load_local_halox(sxx, lvar);
            BARRIER
        #endif
        sxx_x = Dxp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(szz, lvar);
            load_local_haloz(szz, lvar);
            BARRIER
        #endif
        szz_z = Dzp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(sxz, lvar);
            load_local_haloz(sxz, lvar);
            load_local_halox(sxz, lvar);
            BARRIER
        #endif
        sxz_z = Dzm(lvar);
        sxz_x = Dxm(lvar);
    
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(sxx_lin, lvar);
            load_local_halox(sxx_lin, lvar);
            BARRIER
        #endif
        sxx_x_lin = Dxp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(szz_lin, lvar);
            load_local_haloz(szz_lin, lvar);
            BARRIER
        #endif
        szz_z_lin = Dzp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(sxz_lin, lvar);
            load_local_haloz(sxz_lin, lvar);
            load_local_halox(sxz_lin, lvar);
            BARRIER
        #endif
        sxz_z_lin = Dzm(lvar);
        sxz_x_lin = Dxm(lvar);
    
        gridstop(g);
        
        // Update the velocities
        vx_lin(g.z, g.x)+= (sxx_x_lin + sxz_z_lin)*rip(g.z, g.x) \
                       +(sxx_x + sxz_z)*rip_lin(g.z, g.x);
        vz_lin(g.z, g.x)+= (szz_z_lin + sxz_x_lin)*rkp(g.z, g.x) \
                       +(szz_z + sxz_x)*rkp_lin(g.z, g.x);
        """
        grid = ComputeGrid(shape=[s - self.order for s in vx.shape],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "linear", grid, vx, vz, sxx, szz, sxz, rip, rkp)
        return vx, vz

    def adjoint(self, vx, vz, sxx, szz, sxz, rip, rkp):

        src = self.local_header
        src += """ 
        float sxx_x, szz_z, sxz_x, sxz_z;
        float vxx_adj, vxz_adj, vzz_adj, vzx_adj;
        
        // Calculation of the stresses spatial derivatives
        #if LOCAL_OFF==0
            load_local_in(sxx, lvar);
            load_local_halox(sxx, lvar);
            BARRIER
        #endif
        sxx_x = Dxp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(szz, lvar);
            load_local_haloz(szz, lvar);
            BARRIER
        #endif
        szz_z = Dzp(lvar);
            
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(sxz, lvar);
            load_local_haloz(sxz, lvar);
            load_local_halox(sxz, lvar);
            BARRIER
        #endif
        sxz_z = Dzm(lvar);
        sxz_x = Dxm(lvar);
    
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vx_adj, lvar);
            mul_local_in(rip, lvar);
            load_local_haloz(vx_adj, lvar);
            mul_local_haloz(rip, lvar);
            load_local_halox(vx_adj, lvar);
            mul_local_halox(rip, lvar);
            BARRIER
        #endif
        vxx_adj = -Dxm(lvar);
        vxz_adj = -Dzp(lvar);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vz_adj, lvar);
            mul_local_in(rkp, lvar);
            load_local_haloz(vz_adj, lvar);
            mul_local_haloz(rkp, lvar);
            load_local_halox(vz_adj, lvar);
            mul_local_halox(rkp, lvar);
            BARRIER
        #endif
        vzx_adj = -Dxp(lvar);
        vzz_adj = -Dzm(lvar);
        
        gridstop(g);
        
        // Update the velocities
        sxx_adj(g.z, g.x) += vxx_adj;
        szz_adj(g.z, g.x) += vzz_adj;
        sxz_adj(g.z, g.x) += vxz_adj + vzx_adj;
        
        rip_adj(g.z, g.x) += (sxx_x + sxz_z) * vx_adj(g.z, g.x);
        rkp_adj(g.z, g.x) += (szz_z + sxz_x) * vz_adj(g.z, g.x);
        """
        grid = ComputeGrid(shape=[s - self.order for s in vx.shape],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "adjoint", grid, vx, vz, sxx, szz, sxz, rip, rkp)
        return sxx, szz, sxz, rip, rkp


class FreeSurface1(ReversibleFunctionGPU):

    def __init__(self, queue, order=8):
        super().__init__(queue, local_size=None)
        self.header, _ = get_header_stencil(order, 2,
                                                            local_size=None,
                                                            with_local_ops=True)
        hc = ["HC%d" % (i+1) for i in range(order//2)]
        self.header += "float constant hc[%d] = {%s};\n" % (order//2,
                                                            ", ".join(hc))
        self.order = order

    def forward(self, vx, vz, sxx, szz, sxz, M, mu, rkp, rip,
                backpropagate=False):

        src = """ 
        float vxx, vzz;
        float a, b;
        int sign = -2*backpropagate+1; 
            
        vxx = Dxm(vx);
        vzz = Dzm(vz);
        //int i, j;
        //vxx = vzz = 0;
        //for (i=0; i<__FDOH__; i++){
        //    vxx += hc[i] * (vx[indg(pos, 0, 0, i)] - vx[indg(pos, 0, 0, -i-1)]);
        //    vzz += hc[i] * (vz[indg(pos, i, 0, 0)] - vz[indg(pos, -i-1, 0, 0)]);
        //}
        b = mu(g.z, g.x) * 2.0;
        a = M(g.z, g.x);
        sxx(g.z, g.x) += sign * (-((a - b) * (a - b) * vxx / a) - ((a - b) * vzz));
        //szz(g.z, g.x) = 0;
        szz(g.z, g.x)+= sign * -((a*(vxx+vzz))-(b*vxx));
        """
        grid = ComputeGrid(shape=[1, vx.shape[1] - self.order],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "forward", grid, vx, vz, sxx, szz, sxz,
                       M, mu, rkp, rip, backpropagate=backpropagate)
        return sxx, szz

    def linear(self, vx, vz, sxx, szz, sxz, M, mu, rkp, rip):

        src = """ 
        float vxx, vzz;
        float a, b, da, db;
        
        vxx = Dxm(vx_lin);
        vzz = Dzm(vz_lin);
        b = mu(g.z, g.x) * 2.0;
        a = M(g.z, g.x);
        sxx_lin(g.z, g.x) += -((a - b) * (a - b) * vxx / a) - ((a - b) * vzz);
        //szz_lin(g.z, g.x) = 0;
        szz_lin(g.z, g.x)+= -((a*(vxx+vzz))-(b*vxx));
        
        vxx = Dxm(vx);
        vzz = Dzm(vz);
    
        db = mu_lin(g.z, g.x) * 2.0;
        da = M_lin(g.z, g.x);
        sxx_lin(g.z, g.x) += (2.0 * (a - b) * vxx / a + vzz) * db +\
                         (-2.0 * (a - b) * vxx / a 
                         + (a - b)*(a - b) / a / a * vxx - vzz) * da;
        szz_lin(g.z, g.x) += -((da*(vxx+vzz))-(db*vxx));
        """
        grid = ComputeGrid(shape=[1, vx.shape[1] - self.order],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "linear", grid, vx, vz, sxx, szz, sxz,
                       M, mu, rkp, rip)
        return sxx, szz

    def adjoint(self, vx, vz, sxx, szz, sxz, M, mu, rkp, rip):

        src = """ 
        float vxx, vzz;
        float a, b, a_adj, b_adj;
        int i;
        float hx, hz;
    
        b = mu(g.z, g.x) * 2.0;
        a = M(g.z, g.x);
        hx = -((a - b) * (a - b) / a);
        hz = - (a - b);
        for (i=0; i<%d; i++){
            vx_adj(g.z, g.x+i) += hc[i] * hx * sxx_adj(g.z, g.x);
            vx_adj(g.z, g.x-i-1) += - hc[i] * hx * sxx_adj(g.z, g.x);
            vz_adj(g.z+i, g.x) += hc[i] * hz * sxx_adj(g.z, g.x);
            vz_adj(g.z-i-1, g.x) += - hc[i] * hz * sxx_adj(g.z, g.x);
            
            vx_adj(g.z, g.x+i) += hc[i] * hz * szz_adj(g.z, g.x);
            vx_adj(g.z, g.x-i-1) += - hc[i] * hz * szz_adj(g.z, g.x);
            vz_adj(g.z+i, g.x) += hc[i] * (-a) * szz_adj(g.z, g.x);
            vz_adj(g.z-i-1, g.x) += - hc[i] * (-a) * szz_adj(g.z, g.x);
        }
        
        //szz_adj(g.z, g.x) = 0;
        
        vxx = Dxm(vx);
        vzz = Dzm(vz);
    
        b_adj = mu_adj(g.z, g.x) * 2.0;
        a_adj = M_adj(g.z, g.x);
        M_adj(g.z, g.x) += (-2.0 * (a - b) * vxx / a 
                      + (a - b)*(a - b) / a / a * vxx - vzz) * sxx_adj(g.z, g.x);
        mu_adj(g.z, g.x) += 2.0 * (2.0 * (a - b) * vxx / a + vzz) * sxx_adj(g.z, g.x);
        
        M_adj(g.z, g.x) += -(vxx + vzz) *  szz_adj(g.z, g.x);
        mu_adj(g.z, g.x) += 2.0 * vxx *  szz_adj(g.z, g.x);
        """ % (self.order//2)

        grid = ComputeGrid(shape=[1, vx.shape[1] - self.order],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "adjoint", grid, vx, vz, sxx, szz, sxz,
                       M, mu, rkp, rip)
        return vx, vz, M, mu, rkp, rip


class FreeSurface2(ReversibleFunctionGPU):

    def __init__(self, queue, order=8):
        super().__init__(queue, local_size=None)
        self.header, _ = get_header_stencil(order, 2,
                                            local_size=None,
                                            with_local_ops=True)
        hc = ["HC%d" % (i+1) for i in range(order//2)]
        self.header += "float constant hc[%d] = {%s};\n" % (order//2,
                                                            ", ".join(hc))
        self.order = order

    def forward(self, vx, vz, sxx, szz, sxz, rkp, rip, backpropagate=False):

        src = """ 
        float szz_z, sxz_z;
        int i, j;
        
        int sign = -2*backpropagate+1; 
        for (i=0; i<%d; i++){
            sxz_z = szz_z = 0;
            for (j=i+1; j<%d; j++){
                szz_z += hc[j] * szz(g.z+j-i, g.x);
            }
            for (j=i; j<%d; j++){
                sxz_z += hc[j] * sxz(g.z+j-i+1, g.x);
            }
            vx(g.z, g.x) += sign * sxz_z * rip(g.z, g.x);
            vz(g.z, g.x) += sign * szz_z * rkp(g.z, g.x);
        }
        """ % ((self.order//2, )*3)
        grid = ComputeGrid(shape=[1, vx.shape[1] - self.order],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "forward", grid, vx, vz, sxx, szz, sxz,
                       rkp, rip, backpropagate=backpropagate)
        return vx, vz

    def linear(self, vx, vz, sxx, szz, sxz, rkp, rip):

        src = """ 
        float szz_z, sxz_z;
        int i, j;
        
        for (i=0; i<%d; i++){
            sxz_z = szz_z = 0;
            for (j=i+1; j<%d; j++){
                szz_z += hc[j] * szz_lin(g.z+j-i, g.x);
            }
            for (j=i; j<%d; j++){
                sxz_z += hc[j] * sxz_lin(g.z+j-i+1, g.x);
            }
            vx_lin(g.z, g.x) += sxz_z * rip(g.z, g.x);
            vz_lin(g.z, g.x) += szz_z * rkp(g.z, g.x);
        }
        
        for (i=0; i<%d; i++){
            sxz_z = szz_z = 0;
            for (j=i+1; j<%d; j++){
                szz_z += hc[j] * szz(g.z+j-i, g.x);
            }
            for (j=i; j<%d; j++){
                sxz_z += hc[j] * sxz(g.z+j-i+1, g.x);
            }
            vx_lin(g.z, g.x) += sxz_z * rip_lin(g.z, g.x);
            vz_lin(g.z, g.x) += szz_z * rkp_lin(g.z, g.x);
        }
        """ % ((self.order//2, )*6)
        grid = ComputeGrid(shape=[1, vx.shape[1] - self.order],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "linear", grid, vx, vz, sxx, szz, sxz,
                       rkp, rip)
        return vx, vz

    def adjoint(self, vx, vz, sxx, szz, sxz, rkp, rip):

        src = """ 
        float szz_z, sxz_z;
        int i, j;
    
        for (i=0; i<%d; i++){
            for (j=i+1; j<%d; j++){
                szz_adj(g.z+j-i, g.x) += rkp(g.z, g.x) * hc[j] * vz_adj(g.z, g.x);
            }
            for (j=i; j<%d; j++){
                sxz_adj(g.z+j-i+1, g.x) += rip(g.z, g.x) * hc[j] * vx_adj(g.z, g.x);
            }
        }
        
        for (i=0; i<%d; i++){
            sxz_z = szz_z = 0;
            for (j=i+1; j<%d; j++){
                szz_z += hc[j] * szz(g.z+j-i, g.x);
            }
            for (j=i; j<%d; j++){
                sxz_z += hc[j] * sxz(g.z+j-i+1, g.x);
            }
            rip_adj(g.z, g.x) += sxz_z * vx_adj(g.z, g.x);
            rkp_adj(g.z, g.x) += szz_z * vz_adj(g.z, g.x);
        }
        """  % ((self.order//2, )*6)

        grid = ComputeGrid(shape=[1, vx.shape[1] - self.order],
                           queue=self.queue,
                           origin=[self.order//2 for _ in vx.shape])
        self.gpukernel(src, "adjoint", grid, vx, vz, sxx, szz, sxz,
                       rkp, rip)
        return szz, sxz


class ScaledParameters(ReversibleFunction):

    def __init__(self, dt=1, dh=1):
        super().__init__()
        self.sc = 1.0
        self.dt = dt
        self.dh = dh

    @staticmethod
    def scale(M, dt, dx):
        return int(np.log2(max(M).get() * dt / dx))

    def forward(self, rho, rip, rkp, M, mu, muipkp, backpropagate=False):

        dt = self.dt
        dh = self.dh

        if backpropagate:
            sc = self.sc
            rho.data = 2 ** sc * dt / dh / (rho.data + np.sqrt(rho.smallest))
            rip.data = 2 ** sc * dt / dh / (rip.data + np.sqrt(rip.smallest))
            rkp.data = 2 ** sc * dt / dh / (rkp.data + np.sqrt(rkp.smallest))
            M.data = 2 ** sc / dt * dh * M.data
            mu.data = 2 ** sc / dt * dh * mu.data
            muipkp.data = 2 ** sc / dt * dh * muipkp.data
        else:
            self.sc = sc = self.scale(M.data, dt, dh)
            rho.data = 2 ** sc * dt / dh / (rho.data + np.sqrt(rho.smallest))
            rip.data = 2 ** sc * dt / dh / (rip.data + np.sqrt(rip.smallest))
            rkp.data = 2 ** sc * dt / dh / (rkp.data + np.sqrt(rkp.smallest))
            M.data = 2 ** -sc * dt / dh * M.data
            mu.data = 2 ** -sc * dt / dh * mu.data
            muipkp.data = 2 ** -sc * dt / dh * muipkp.data

        return rho, rip, rkp, M, mu, muipkp

    def linear(self, rho, rip, rkp, M, mu, muipkp):

        dt = self.dt
        dh = self.dh
        sc = self.sc
        rho.lin = -2 ** sc * dt / dh / (rho.data + np.sqrt(rho.smallest)) ** 2 * rho.lin
        rip.lin = -2 ** sc * dt / dh / (rip.data + np.sqrt(rip.smallest)) ** 2 * rip.lin
        rkp.lin = -2 ** sc * dt / dh / (rkp.data + np.sqrt(rkp.smallest)) ** 2 * rkp.lin
        M.lin = 2 ** -sc * dt / dh * M.lin
        mu.lin = 2 ** -sc * dt / dh * mu.lin
        muipkp.lin = 2 ** -sc * dt / dh * muipkp.lin

        return rho, rip, rkp, M, mu, muipkp

    def adjoint(self, rho, rip, rkp, M, mu, muipkp):

        sc = self.sc
        dt = self.dt
        dh = self.dh
        rho.grad = -2 ** sc * dt / dh / (rho.data + np.sqrt(rho.smallest)) ** 2 * rho.grad
        rip.grad = -2 ** sc * dt / dh / (rip.data + np.sqrt(rip.smallest)) ** 2 * rip.grad
        rkp.grad = -2 ** sc * dt / dh / (rkp.data + np.sqrt(rkp.smallest)) ** 2 * rkp.grad
        M.grad = 2 ** -sc * dt / dh * M.grad
        mu.grad = 2 ** -sc * dt / dh * mu.grad
        muipkp.grad = 2 ** -sc * dt / dh * muipkp.grad

        return rho, rip, rkp, M, mu, muipkp


class Cerjan(FunctionGPU):

    def __init__(self, grids=None, freesurf=False, abpc=4.0, nab=2,
                 required_states=(), **kwargs):
        self.abpc = abpc
        self.nab = nab
        self.required_states = required_states
        self.updated_states = required_states
        self.taper = np.exp(np.log(1.0-abpc/100)/nab**2 * np.arange(nab) **2)
        self.taper = np.expand_dims(self.taper, -1)
        self.freesurf = freesurf
        self.default_grids = {el: "gridfd" for el in self.required_states}
        super().__init__(grids, **kwargs)

    @property
    def updated_regions(self):
        regions = []
        pad = self.grids[self.updated_states[0]].pad
        ndim = len(self.grids[self.updated_states[0]].shape)
        b = self.nab + pad
        for dim in range(ndim):
            region = [Ellipsis for _ in range(ndim)]
            region[dim] = slice(pad, b)
            if dim != 0 or not self.freesurf:
                regions.append(region)
            region = [Ellipsis for _ in range(ndim)]
            region[dim] = slice(-b, -pad)
            regions.append(tuple(region))
        return {el: regions for el in self.updated_states}

    def forward(self, states, **kwargs):
        pad = self.grids[self.updated_states[0]].pad
        for el in self.required_states:
            if not self.freesurf:
                states[el][pad:self.nab+pad, :] *= self.taper[::-1]
            states[el][-self.nab-pad:-pad, :] *= self.taper

            tapert = np.transpose(self.taper)
            states[el][:, pad:self.nab+pad] *= tapert[:, ::-1]
            states[el][:, -self.nab-pad:-pad] *= tapert

        return states

    def adjoint(self, adj_states, states, **kwargs):

        return self.forward(adj_states, **kwargs)


def elastic2d(grid2D, gridout, gridsrc, nab):

    gridpar = copy(grid2D)
    gridpar.zero_boundary = False
    gridpar.pad = 0
    defs = {"gridfd": grid2D, "gridpar": gridpar, "gridout": gridout,
            "gridsrc": gridsrc}

    stepper = SequenceCL([Source(required_states=["vz"], grids=defs),
                          UpdateVelocity(grids=defs),
                          Cerjan(required_states=["vx", "vz"], freesurf=1, nab=nab),
                          Receiver(required_states=["vz"],
                                   updated_states=["vzout"],
                                   grids=defs),
                          UpdateStress(grids=defs),
                          FreeSurface(grids=defs),
                          Cerjan(required_states=["sxx", "szz", "sxz"],
                                 freesurf=1, nab=nab),
                          ])
    psv2D = SequenceCL([Velocity2LameCL(grids=defs),
                        ArithmeticAveraging(grids=defs,
                                            required_states=["rho", "rip"],
                                            updated_states=["rip"], dx=1),
                        ArithmeticAveraging(grids=defs,
                                            required_states=["rho", "rkp"],
                                            updated_states=["rkp"], dz=1),
                        HarmonicAveraging(grids=defs,
                                          required_states=["mu", "muipkp"],
                                          updated_states=["muipkp"],
                                          dx1=1, dz2=1),
                        ScaledParameters(grids=defs,
                                         dt=gridsrc.dt,
                                         dh=grid2D.dh),
                        PropagatorCL(stepper, gridsrc.nt)],
                       grids=defs)
    return psv2D


if __name__ == '__main__':

    resc = ComputeRessource()

    nx = 24
    nz = 24
    nrec = 1
    nt = 3
    nab = 2
    dh = 1.0
    dt = 0.0001

    grid2D = GridCL(resc.queues[0], shape=(nz, nx), pad=4, type=np.float32,
                    zero_boundary=True)
    src_linpos = grid2D.xyz2lin([0], [15]).astype(np.int32)
    xrec = np.arange(10, 15)
    zrec = xrec*0
    rec_linpos = grid2D.xyz2lin(zrec, xrec).astype(np.int32)
    gridsrc = GridCL(resc.queues[0], shape=(1,), pad=0, type=np.float32,
                     dt=dt, nt=nt)
    gridout = GridCL(resc.queues[0], shape=(nt, nrec), type=np.float32)
    psv2D = psv2D = elastic2d(grid2D, gridout, gridsrc, nab)

    # psv2D.backward_test(reclinpos=rec_linpos,
    #                     srclinpos=src_linpos)
    # psv2D.linear_test(reclinpos=rec_linpos,
    #                   srclinpos=src_linpos)
    # psv2D.dot_test(reclinpos=rec_linpos,
    #                srclinpos=src_linpos)

    nrec = 1
    nt = 7500
    nab = 16
    dh = 1.0
    dt = 0.0001

    grid2D = GridCL(resc.queues[0], shape=(160, 300), type=np.float32,
                    zero_boundary=True, dh=dh, pad=2)

    src_linpos = grid2D.xyz2lin([0], [50]).astype(np.int32)
    xrec = np.arange(50, 250)
    zrec = xrec*0
    rec_linpos = grid2D.xyz2lin(zrec, xrec).astype(np.int32)
    gridout = GridCL(resc.queues[0], shape=(nt, xrec.shape[0]), pad=0,
                     type=np.float32, dt=dt, nt=nt)
    gridsrc = GridCL(resc.queues[0], shape=(nt, 1), pad=0, type=np.float32,
                     dt=dt, nt=nt)
    psv2D = elastic2d(grid2D, gridout, gridsrc, nab)

    vs = np.full(grid2D.shape, 300.0)
    rho = np.full(grid2D.shape, 1800.0)
    vp = np.full(grid2D.shape, 1500.0)
    vs[80:, :] = 600
    rho[80:, :] = 2000
    vp[80:, :] = 2000
    vs0 = vs.copy()
    vs[5:10, 145:155] *= 1.05

    states = psv2D({"vs": vs,
                    "vp": vp,
                    "rho": rho,
                    "signal": ricker(10, dt, nt)},
                    reclinpos=rec_linpos,
                    srclinpos=src_linpos)
    plt.imshow(states["vx"].get())
    plt.show()
    #
    vzobs = states["vzout"].get()
    clip = 0.01
    vmin = np.min(vzobs) * 0.1
    vmax=-vmin
    plt.imshow(vzobs, aspect="auto", vmin=vmin, vmax=vmax)
    plt.show()

