from SeisCL.python import ReversibleFunction
from SeisCL.python import ComputeRessource, ReversibleFunctionGPU
import numpy as np
import pyopencl.clmath as math


class Velocity2Lame(ReversibleFunction):

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vp", "vs", "rho"]
        self.updated_states = ["M", "mu", "rhoi"]
        self.default_grids = {el: "gridpar"
                              for el in self.required_states+self.updated_states}

    def forward(self, states, **kwargs):

        vs = states["vp"]
        vp = states["vs"]
        rho = states["rho"]

        states["M"] = vp**2 * rho
        states["mu"] = vs**2 * rho
        states["rhoi"] = 1.0 / (rho + self.grids["rho"].smallest)

        return states

    def linear(self, dstates, states, **kwargs):

        vs = states["vs"]
        vp = states["vp"]
        rho = states["rho"]

        dvs = dstates["vs"]
        dvp = dstates["vp"]
        drho = dstates["rho"]

        dstates["M"] = 2.0 * (vp * rho) * dvp + vp**2 * drho
        dstates["mu"] = 2.0 * (vs * rho) * dvs + vs**2 * drho
        dstates["rhoi"] = -1.0 / (rho + self.grids["rho"].smallest)**2 * drho

        return dstates

    def adjoint(self, adj_states, states, dt=0.1, dx=2.0, **kwargs):

        vs = states["vs"]
        vp = states["vp"]
        rho = states["rho"]

        adj_mu = adj_states["mu"]
        adj_M = adj_states["M"]
        adj_rhoi = adj_states["rhoi"]

        adj_states["vp"] += 2.0 * (vp * rho) * adj_M
        adj_states["vs"] += 2.0 * (vs * rho) * adj_mu
        adj_states["rho"] += (vp**2 * adj_M
                             + vs**2 * adj_mu
                             - 1.0 / (rho + self.grids["rho"].smallest)**2 * adj_rhoi)
        return adj_states

    def backward(self, states, dt=0.1, dx=2.0, **kwargs):

        rhoi = states["rhoi"]
        M = states["M"]
        mu = states["mu"]

        states["rho"] = 1.0 / (rhoi + self.grids["rho"].smallest)
        states["vs"] = math.sqrt(mu * rhoi)
        states["vp"] = math.sqrt(M * rhoi)

        return states


class Velocity2LameGPU(ReversibleFunctionGPU):

    forward_src = """
    FUNDEF void Velocity2LameCL(grid pos,
                                GLOBARG float *rho,
                                GLOBARG float *vp,
                                GLOBARG float *vs,
                                int backpropagate){
    
    get_pos(&pos);
    int ind0 = indg(pos, 0, 0, 0);
    if (backpropagate){
        vp[ind0] = sqrt(vp[ind0] / rho[ind0]);
        vs[ind0] = sqrt(vs[ind0] / rho[ind0]);
    }
    else{
        vp[ind0] = pow(vp[ind0], 2) * rho[ind0];
        vs[ind0] = pow(vs[ind0], 2) * rho[ind0];
    }
}
    """

    linear_src = """
    FUNDEF void Velocity2LameCL_lin(grid pos,
                                GLOBARG float *rho,
                                GLOBARG float *vp,
                                GLOBARG float *vs,
                                GLOBARG float *rho_lin,
                                GLOBARG float *vp_lin,
                                GLOBARG float *vs_lin){
    
    get_pos(&pos);
    int ind0 = indg(pos, 0, 0, 0);
    vp_lin[ind0] = 2.0 * (vp[ind0] * rho[ind0]) * vp_lin[ind0] \
                  + pow(vp[ind0], 2) * rho_lin[ind0];
    vs_lin[ind0] = 2.0 * (vs[ind0] * rho[ind0]) * vs_lin[ind0] \
                  + pow(vs[ind0], 2) * rho_lin[ind0];
}
    """

    adjoint_src = """
    FUNDEF void Velocity2LameCL_adj(grid pos,
                                GLOBARG float *rho,
                                GLOBARG float *vp,
                                GLOBARG float *vs,
                                GLOBARG float *rho_adj,
                                GLOBARG float *vp_adj,
                                GLOBARG float *vs_adj){
    
    get_pos(&pos);
    int ind0 = indg(pos, 0, 0, 0);
    rho_adj[ind0] += pow(vp[ind0], 2) * vp_adj[ind0] \
                    + pow(vs[ind0], 2) * vs_adj[ind0] ;
    vp_adj[ind0] = 2.0 * (vp[ind0] * rho[ind0]) * vp_adj[ind0];
    vs_adj[ind0] = 2.0 * (vs[ind0] * rho[ind0]) * vs_adj[ind0];

}
    """

    def __init__(self, grids=None, computegrid=None, **kwargs):
        self.required_states = ["vp", "vs", "rho"]
        self.updated_states = ["vp", "vs", "rho"]
        self.default_grids = {el: "gridpar" for el in self.required_states}
        self.default_grids["M"] = "gridpar"
        self.default_grids["mu"] = "gridpar"
        self.copy_states = {"M": "vp", "mu": "vs"}
        super().__init__(grids=grids,
                         computegrid=computegrid,
                         **kwargs)


if __name__ == '__main__':

    resc = ComputeRessource()
    nx = 24
    nz = 24

    grid = GridCL(resc.queues[0], shape=(nz, nx), pad=4, dtype=np.float32,
                  zero_boundary=False)
    v = np.tile(np.arange(nx, dtype=np.float32), [nz, 1])

    # veltrans =Velocity2Lame(grids={"gridpar": grid})
    # veltrans.linear_test()
    # veltrans.dot_test()

    veltrans =Velocity2LameCL(grids={"gridpar": grid})
    veltrans.linear_test()
    veltrans.backward_test()
    veltrans.dot_test()