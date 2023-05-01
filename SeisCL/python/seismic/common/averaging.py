
from SeisCL.python import (FunctionGPU, ComputeRessource)
import numpy as np


class ArithmeticAveraging(FunctionGPU):

    forward_src = """
FUNDEF void ArithmeticAveraging(grid pos, 
                              GLOBARG float *v, 
                              GLOBARG float *vave,
                              int dx,
                              int dy,
                              int dz){
    
    get_pos(&pos);
    int ind1 = indg(pos,0,0,0);
    int ind2 = indg(pos, dz, dy, dx);
    vave[ind1]=0.5*(v[ind1]+v[ind2]);
}
"""


    adjoint_src = """
FUNDEF void ArithmeticAveraging_adj(grid pos, 
                                    GLOBARG float *v_adj, 
                                    GLOBARG float *vave_adj,
                                    int dx,
                                    int dy,
                                    int dz){
    
    get_pos(&pos);
    int ind1 = indg(pos,0,0,0);
    int ind2 = indg(pos, dz, dy, dx);
     
    v_adj[ind1] += 0.5*vave_adj[ind1];
    v_adj[ind2] += 0.5*vave_adj[ind1];
    vave_adj[ind1] = 0;
}
    """

    def __init__(self, grids=None, computegrid=None, required_states=(),
                 updated_states=(), dx=0, dy=0, dz=0, **kwargs):
        self.required_states = required_states
        self.updated_states = updated_states
        self.default_grids = {el: "gridpar" for el in self.required_states}
        self.args_renaming = {"v": required_states[0],
                              "vave": updated_states[0]}
        super().__init__(grids=grids,
                         computegrid=computegrid,
                         default_args={"dx": dx, "dy": dy, "dz": dz},
                         **kwargs)


class ArithmeticAveraging2(FunctionGPU):

    forward_src = """
FUNDEF void ArithmeticAveraging2(grid pos, 
                              GLOBARG float *v, 
                              GLOBARG float *vave,
                              int dx1,
                              int dy1,
                              int dz1,
                              int dx2,
                              int dy2,
                              int dz2){
    
    get_pos(&pos);
    int ind1 = indg(pos,0,0,0);
    int ind2 = indg(pos, dz1, dy1, dx1);
    int ind3 = indg(pos, dz2, dy2, dx2);
    int ind4 = indg(pos, dz1+dz2, dy1+dy2, dx1+dx2);
    vave[ind1]=0.25*(v[ind1]+v[ind2]+v[ind3]+v[ind4]);
}
"""


    adjoint_src = """
FUNDEF void ArithmeticAveraging2_adj(grid pos, 
                              GLOBARG float *v_adj, 
                              GLOBARG float *vave_adj,
                              int dx1,
                              int dy1,
                              int dz1,
                              int dx2,
                              int dy2,
                              int dz2){
    
    get_pos(&pos);
    get_pos(&pos);
    int ind1 = indg(pos,0,0,0);
    int ind2 = indg(pos, dz1, dy1, dx1);
    int ind3 = indg(pos, dz2, dy2, dx2);
    int ind4 = indg(pos, dz1+dz2, dy1+dy2, dx1+dx2);
     
    v_adj[ind1] += 0.25*vave_adj[ind1];
    v_adj[ind2] += 0.25*vave_adj[ind1];
    v_adj[ind3] += 0.25*vave_adj[ind1];
    v_adj[ind4] += 0.25*vave_adj[ind1];
    vave_adj[ind1] = 0;
}
    """

    def __init__(self, grids=None, computegrid=None, required_states=(),
                 updated_states=(), dx1=0, dy1=0, dz1=0, dx2=0, dy2=0, dz2=0, **kwargs):
        self.required_states = required_states
        self.updated_states = updated_states
        self.default_grids = {el: "gridpar" for el in self.required_states}
        self.args_renaming = {"v": required_states[0],
                              "vave": updated_states[0]}
        super().__init__(grids=grids,
                         computegrid=computegrid,
                         default_args={"dx1": dx1, "dy1": dy1, "dz1": dz1,
                                       "dx2": dx2, "dy2": dy2, "dz2": dz2},
                         **kwargs)


class HarmonicAveraging(FunctionGPU):

    forward_src = """        
    FUNDEF void HarmonicAveraging(grid pos,
                                   GLOBARG float *v,
                                   GLOBARG float *vave,
                                   int dx1,
                                   int dy1,
                                   int dz1,
                                   int dx2,
                                   int dy2,
                                   int dz2){

    get_pos(&pos);
    int ind1 = indg(pos,0,0,0);
    int ind2 = indg(pos, dz1, dy1, dx1);
    int ind3 = indg(pos, dz2, dy2, dx2);
    int ind4 = indg(pos, dz1+dz2, dy1+dy2, dx1+dx2);
    float d =0;
    float n = 0;
    if (v[ind1] > __EPS__){
        n+=1;
        d += 1.0 / v[ind1];
    }
    if (v[ind2] > __EPS__){
        n+=1;
        d += 1.0 / v[ind2];
    }
    if (v[ind3] > __EPS__){
        n+=1;
        d += 1.0 / v[ind3];
    }
    if (v[ind4] > __EPS__){
        n+=1;
        d += 1.0 / v[ind4];
    }
    vave[ind1]= n / d;
    }
    """

    linear_src = """
    FUNDEF void HarmonicAveraging_lin(grid pos,
                                   GLOBARG float *v,
                                   GLOBARG float *vave,
                                   GLOBARG float *v_lin,
                                   GLOBARG float *vave_lin,
                                   int dx1,
                                   int dy1,
                                   int dz1,
                                   int dx2,
                                   int dy2,
                                   int dz2){

    get_pos(&pos);
    int ind1 = indg(pos,0,0,0);
    int ind2 = indg(pos, dz1, dy1, dx1);
    int ind3 = indg(pos, dz2, dy2, dx2);
    int ind4 = indg(pos, dz1+dz2, dy1+dy2, dx1+dx2);
    float d1 = 0;
    float d2 = 0;
    float n = 0;
    if (v[ind1] > __EPS__){
        n+=1;
        d1 += 1.0 / v[ind1];
        d2 += 1.0/ pow(v[ind1],2) * v_lin[ind1];
    }
    if (v[ind2] > __EPS__){
        n+=1;
        d1 += 1.0 / v[ind2];
        d2 += 1.0/ pow(v[ind2],2) * v_lin[ind2];
    }
    if (v[ind3] > __EPS__){
        n+=1;
        d1 += 1.0 / v[ind3];
        d2 += 1.0/ pow(v[ind3],2) * v_lin[ind3];
    }
    if (v[ind4] > __EPS__){
        n+=1;
        d1 += 1.0 / v[ind4];
        d2 += 1.0/ pow(v[ind4],2) * v_lin[ind4];
    }
    vave_lin[ind1]= n / pow(d1, 2) * d2;
    }
    """

    adjoint_src = """
    FUNDEF void HarmonicAveraging_adj(grid pos,
                                   GLOBARG float *v,
                                   GLOBARG float *vave,
                                   GLOBARG float *v_adj,
                                   GLOBARG float *vave_adj,
                                   int dx1,
                                   int dy1,
                                   int dz1,
                                   int dx2,
                                   int dy2,
                                   int dz2){

    get_pos(&pos);
    int ind1 = indg(pos,0,0,0);
    int ind2 = indg(pos, dz1, dy1, dx1);
    int ind3 = indg(pos, dz2, dy2, dx2);
    int ind4 = indg(pos, dz1+dz2, dy1+dy2, dx1+dx2);
    float d1 = 0;
    float n = 0;
    if (v[ind1] > __EPS__){
        n+=1;
        d1 += 1.0 / v[ind1];
    }
    if (v[ind2] > __EPS__){
        n+=1;
        d1 += 1.0 / v[ind2];
    }
    if (v[ind3] > __EPS__){
        n+=1;
        d1 += 1.0 / v[ind3];
    }
    if (v[ind4] > __EPS__){
        n+=1;
        d1 += 1.0 / v[ind4];
    }
    
    
    if (v[ind1] > __EPS__){
        v_adj[ind1] += n / pow(d1, 2) * 1.0/ pow(v[ind1],2) * vave_adj[ind1];
    }
    if (v[ind2] > __EPS__){
        v_adj[ind2] += n / pow(d1, 2) * 1.0/ pow(v[ind2],2) * vave_adj[ind1];
    }
    if (v[ind3] > __EPS__){
        v_adj[ind3] += n / pow(d1, 2) * 1.0/ pow(v[ind3],2) * vave_adj[ind1];
    }
    if (v[ind4] > __EPS__){
        v_adj[ind4] += n / pow(d1, 2) * 1.0/ pow(v[ind4],2) * vave_adj[ind1];
    }
    vave_adj[ind1] = 0;
    }
    """

    def __init__(self, grids=None, computegrid=None, required_states=(),
                 updated_states=(), dx1=0, dy1=0, dz1=0, dx2=0, dy2=0, dz2=0, **kwargs):
        self.required_states = required_states
        self.updated_states = updated_states
        self.default_grids = {el: "gridpar" for el in self.required_states}
        self.args_renaming = {"v": required_states[0],
                              "vave": updated_states[0]}
        super().__init__(grids=grids,
                         computegrid=computegrid,
                         default_args={"dx1": dx1, "dy1": dy1, "dz1": dz1,
                                       "dx2": dx2, "dy2": dy2, "dz2": dz2},
                         **kwargs)


if __name__ == '__main__':

    resc = ComputeRessource()
    nx = 24
    nz = 24

    grid = GridCL(resc.queues[0], shape=(nz, nx), pad=4, dtype=np.float32,
                  zero_boundary=True)
    v = np.tile(np.arange(nx, dtype=np.float32), [nz, 1])

    avex = HarmonicAveraging(grids={"gridpar": grid},
                             required_states=["v1", "v1ave"],
                             updated_states=["v1ave"])
    avex.linear_test(dx1=1, dz1=0, dy1=0, dx2=0, dz2=1, dy2=0)
    avex.dot_test(dx1=1, dz1=0, dy1=0, dx2=0, dz2=1, dy2=0)

    avex =ArithmeticAveraging2(grids={"gridpar": grid},
                               required_states=["v1", "v1ave"],
                               updated_states=["v1ave"])
    avex.linear_test(dx1=1, dz1=0, dy1=0, dx2=0, dz2=1, dy2=0)
    avex.dot_test(dx1=1, dz1=0, dy1=0, dx2=0, dz2=1, dy2=0)

    avex = ArithmeticAveraging(grids={"gridpar": grid},
                               required_states=["v1", "v1ave"],
                               updated_states=["v1ave"])
    avex.linear_test(dx=1, dz=0, dy=0)
    avex.dot_test(dx=1, dz=0, dy=0)



