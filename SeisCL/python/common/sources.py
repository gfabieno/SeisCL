from SeisCL.python.pycl_backend import ReversibleFunctionCL
import numpy as np


class Source(ReversibleFunctionCL):

    forward_src = """
    FUNDEF void Source(grid pos,
                       GLOBARG float *val,
                       GLOBARG float *signal,
                       GLOBARG int *srclinpos,
                       int t,
                       int backpropagate)
{
    int i = get_global_id(0);
    int sign = -2*backpropagate+1;
    val[srclinpos[i]] += sign * signal[t + i*pos.nt] * pos.dt;
}
    """

    def __init__(self, grids=None, required_states=(), **kwargs):

        self.required_states = required_states + ["signal"]
        self.updated_states = required_states
        self.default_grids = {el: "gridfd" for el in required_states}
        self.default_grids["signal"] = "gridsrc"
        self.args_renaming = {"val": required_states[0]}
        super().__init__(grids=grids,
                         computegrid=grids["gridsrc"],
                         linear_forward=False,
                         **kwargs)

    def global_size_fw(self):
        gsize = np.int32(self.computegrid.shape)
        gsize[0] = 1
        return gsize


class PressureSource2D(ReversibleFunctionCL):

    forward_src = """
    FUNDEF void Source(GLOBARG float *sxx,
                       GLOBARG float *szz,
                       GLOBARG float *signal,
                       GLOBARG float *linpos,
                       int t,
                       int nt,
                       int backpropagate)
{
    int i = get_global_id(1);
    int sign = -2*backpropagate+1; 
    sxx[linpos[i]] += signal[t + i*nt]/2;
    szz[linpos[i]] += signal[t + i*nt]/2;
}
    """

    def __init__(self, grids=None, computegrid=None,
                 required_states=(), **kwargs):

        self.required_states = required_states
        self.updated_states = required_states
        self.default_grids = {el: "gridpar" for el in self.required_states}
        self.args_renaming = {"val": required_states[0]}
        super().__init__(grids=grids,
                         computegrid=computegrid,
                         linear_forward=False,
                         **kwargs)