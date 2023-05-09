import time

from SeisCL.python import (ComputeGrid, ReversibleFunctionGPU, ReversibleFunction,
                           ComputeRessource, VariableCL)
from SeisCL.python import get_header_stencil
from SeisCL.python.seismic.PSV2D.elastic_numpy import ricker
import numpy as np
from copy import copy
from pyopencl.array import max
from SeisCL.python.seismic import (Acquisition, Grid,
                                   Velocity2LameGPU,
                                   ArithmeticAveraging,
                                   HarmonicAveraging,
                                   GeophoneGPU2D,
                                   PointSources2DGPU,
                                   CerjanGPU
                                   )
from SeisCL.python.Propagator import FWI, Propagator
import matplotlib.pyplot as plt

#TODO interface with backward compatibility with SeisCL
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
        self.callgpu(src, "forward", grid, vx, vz, sxx, szz, sxz, M, mu,
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
        self.callgpu(src, "linear", grid,
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
        self.callgpu(src, "adjoint", grid,
                     vx, vz, sxx, szz, sxz, M, mu, muipkp)
        return vx, vz, M, mu, muipkp


class UpdateVelocity(ReversibleFunctionGPU):

    def __init__(self, queue, order=8, local_size=(16, 16)):
        super().__init__(queue, local_size=local_size)
        self.header, self.local_header = get_header_stencil(order, 2,
                                                            local_size=local_size,
                                                            with_local_ops=True)
        self.order = order
        self.grid = None

    def compute_grid(self, shape):
        if self.grid is None:
            self.grid = ComputeGrid(shape=[s - self.order for s in shape],
                                    queue=self.queue,
                                    origin=[self.order//2 for _ in shape])
        return self.grid

    #TODO memoize in a better way
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
        grid = self.compute_grid(vx.shape)
        self.callgpu(src, "forward", grid, vx, vz, sxx, szz, sxz, rip, rkp,
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
        self.callgpu(src, "linear", grid, vx, vz, sxx, szz, sxz, rip, rkp)
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
        self.callgpu(src, "adjoint", grid, vx, vz, sxx, szz, sxz, rip, rkp)
        return sxx, szz, sxz, rip, rkp


class FreeSurface1(ReversibleFunctionGPU):

    def __init__(self, queue, order=8):
        super().__init__(queue, local_size=None)
        self.header, _ = get_header_stencil(order, 2, local_size=None,
                                            with_local_ops=True)
        hc = ["HC%d" % (i+1) for i in range(order // 2)]
        self.header += "float constant hc[%d] = {%s};\n" % (order // 2,
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
        self.callgpu(src, "forward", grid, vx, vz, sxx, szz, sxz,
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
        self.callgpu(src, "linear", grid, vx, vz, sxx, szz, sxz,
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
        self.callgpu(src, "adjoint", grid, vx, vz, sxx, szz, sxz,
                     M, mu, rkp, rip)
        return vx, vz, M, mu, rkp, rip


class FreeSurface2(ReversibleFunctionGPU):

    def __init__(self, queue, order=8):
        super().__init__(queue, local_size=None)
        self.header, _ = get_header_stencil(order, 2,
                                            local_size=None,
                                            with_local_ops=True)
        hc = ["HC%d" % (i+1) for i in range(order // 2)]
        self.header += "float constant hc[%d] = {%s};\n" % (order // 2,
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
        self.callgpu(src, "forward", grid, vx, vz, sxx, szz, sxz,
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
        self.callgpu(src, "linear", grid, vx, vz, sxx, szz, sxz,
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
        self.callgpu(src, "adjoint", grid, vx, vz, sxx, szz, sxz,
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


class Elastic2dPropagatorGPU(Propagator):

    def __init__(self, grid: Grid, order=4,
                 local_size=(16, 16)):

        resc = ComputeRessource()
        queue = self.queue = resc.queues[0]
        self.queue = queue
        self.grid = grid
        self.fdorder = order
        shape = (self.grid.nz, self.grid.nx)
        pad = self.fdorder//2
        self.vx = VariableCL(queue, shape=shape, pad=pad)
        self.vz = VariableCL(queue, shape=shape, pad=pad)
        self.sxx = VariableCL(queue, shape=shape, pad=pad)
        self.szz = VariableCL(queue, shape=shape, pad=pad)
        self.sxz = VariableCL(queue, shape=shape, pad=pad)
        self.vs = VariableCL(queue, shape=shape, pad=pad)
        self.vp = VariableCL(queue, shape=shape, pad=pad)
        self.rho = VariableCL(queue, shape=shape, pad=pad)
        self.rip = VariableCL(queue, shape=shape, pad=pad)
        self.rkp = VariableCL(queue, shape=shape, pad=pad)
        self.muipkp = VariableCL(queue, shape=shape, pad=pad)
        self.vel2lame = Velocity2LameGPU(queue)
        self.arithemtic_average = ArithmeticAveraging(queue)
        self.harmonic_average = HarmonicAveraging(queue)
        self.scaledparameters = ScaledParameters(self.grid.dt,
                                                 self.grid.dh)
        self.src_fun = PointSources2DGPU(queue)
        self.rec_fun = GeophoneGPU2D(queue)
        self.updatev = UpdateVelocity(queue, order=order, local_size=local_size)
        self.updates = UpdateStress(queue, order=order, local_size=local_size)
        self.freesurface1 = FreeSurface1(queue, order=order)
        self.freesurface2 = FreeSurface2(queue, order=order)
        self.abs_v = CerjanGPU(queue, nab=self.grid.nab)
        self.abs_s = CerjanGPU(queue, nab=self.grid.nab)

    def propagate(self, shot, vp, vs, rho):

        self.vp.data = vp
        self.vs.data = vs
        self.rho.data = rho
        vp, vs, rho = (self.vp, self.vs, self.rho)

        self.vx.initialize()
        self.vz.initialize()
        self.sxx.initialize()
        self.szz.initialize()
        self.sxz.initialize()
        shot.dmod = VariableCL(self.queue,
                               shape=(shot.nt, len(shot.receivers)))
        rec_pos = self.rec_fun.rec_pos(shot, self.grid.dh, vp.shape)
        rec_type = self.rec_fun.rec_type(shot)
        src_pos = self.src_fun.src_pos(shot, self.grid.dh, vp.shape)
        src_type = self.src_fun.src_type(shot)
        M, mu = self.vel2lame(vp, vs, rho)
        rip = self.arithemtic_average(rho, self.rip, dx=1)
        rkp = self.arithemtic_average(rho, self.rkp, dz=1)
        muipkp = self.harmonic_average(mu, self.muipkp, dx1=1, dz2=1)
        rho, rip, rkp, M, mu, muipkp = self.scaledparameters(rho, rip, rkp,
                                                             M, mu, muipkp)
        vx, vz, sxx, szz, sxz = (self.vx, self.vz, self.sxx, self.szz, self.sxz)
        for t in range(self.grid.nt):
            # vx, vz, sxx, szz = self.src_fun(vx, vz, sxx, szz, shot.wavelet,
            #                                 src_pos, src_type, t)
            vx, vz = self.updatev(vx, vz, sxx, szz, sxz, rip, rkp)
            # vx, vz = self.abs_v(vx, vz)
            # sxx, szz, sxz = self.updates(vx, vz, sxx, szz, sxz, M, mu, muipkp)
            # sxx, szz = self.freesurface1(vx, vz, sxx, szz, sxz, M, mu, rkp, rip)
            # vx, vz = self.freesurface2(vx, vz, sxx, szz, sxz, rkp, rip)
            # sxx, szz, sxz = self.abs_s(sxx, szz, sxz)
            # shot.dmod = self.rec_fun(vx, vz, sxx, szz, shot.dmod, rec_pos,
            #                          rec_type, t)
            print(t)
        return shot.dmod


if __name__ == '__main__':

    shape = (160, 300)
    grid = Grid(nd=2, nx=shape[1], ny=None, nz=shape[0], nt=4500, dt=0.0001,
                dh=1.0, nab=16, pad=2, freesurf=True)
    acquisition = Acquisition(grid=grid)
    propagator = Elastic2dPropagatorGPU(grid)
    acquisition.regular2d(rec_types=["vx", "vz"], gz0=4, queue=propagator.queue)
    vp, vs, rho = (np.zeros(shape, dtype=np.float32) for _ in range(3))
    vp[:, :] = 1500
    vs[:, :] = 400
    rho[:, :] = 1800
    vs[80:, :] = 600
    vp[80:, :] = 2000
    rho[80:, :] = 2000
    vs0 = vs.copy()
    vs[5:10, 145:155] *= 1.05

    t1 = time.time()
    dmod = propagator.propagate(acquisition.shots[1], vp, vs, rho)
    t2 = time.time() -t1
    print(t2)

    # fwi = FWI(acquisition, propagator)
    # shots = fwi(acquisition.shots[:2], vp, vs, rho)
    # dmod = shots[1].dmod

    clip = 0.1
    vmin = np.min(dmod.data.get()) * clip
    vmax=-vmin
    plt.imshow(dmod.data.get(), aspect="auto", vmin=vmin, vmax=vmax)
    plt.show()


    # psv2D.backward_test(reclinpos=rec_linpos,
    #                     srclinpos=src_linpos)
    # psv2D.linear_test(reclinpos=rec_linpos,
    #                   srclinpos=src_linpos)
    # psv2D.dot_test(reclinpos=rec_linpos,
    #                srclinpos=src_linpos)


