from SeisCL.python.pycl_backend import (ReversibleKernelCL,
                                        GridCL,
                                        ComputeRessource,
                                        )
import numpy as np
import pyopencl as cl
import pyopencl.array


class Receiver(ReversibleKernelCL):

    forward_src = """
    FUNDEF void Receiver(grid pos,
                         GLOBARG float *val,
                         GLOBARG float *valout,
                         GLOBARG int *reclinpos,
                         int t)
{
    int i = get_global_id(1);
    valout[t + i*pos.nt] = val[reclinpos[i]];
}
    """

    linear_src = """
    FUNDEF void Receiver_lin(grid pos,
                         GLOBARG float *val_lin,
                         GLOBARG float *valout_lin,
                         GLOBARG int *reclinpos,
                         int t)
{
    int i = get_global_id(0);
    valout_lin[t + i*pos.nt] = val_lin[reclinpos[i]];
}
    """

    adjoint_src = """
    FUNDEF void Receiver_adj(grid pos,
                             GLOBARG float *val_adj,
                             GLOBARG float *valout_adj,
                             GLOBARG int *reclinpos,
                             int t)
{
    int i = get_global_id(0);
    val_adj[reclinpos[i]] += valout_adj[t + i*pos.nt];
}
    """

    def __init__(self, grids=None,  required_states=(), **kwargs):

        if len(required_states) > 1:
            raise ValueError("Required state should be a 1-element list, "
                             "got %d" % len(required_states))
        self.required_states = required_states + [required_states[0]+"out"]
        self.default_grids = {self.required_states[0]: "gridfd",
                              self.required_states[1]: "gridout"}
        self.args_renaming = {"val": self.required_states[0],
                              "valout": self.required_states[1]}
        self.zeroinit_states = [self.required_states[1]]
        super().__init__(grids=grids,
                         computegrid=grids["gridout"],
                         **kwargs)

    def backward(self, states, **kwargs):
        name = self.required_states[1]
        states[name] = self.grids[name].initialize()
        return states

    def global_size_fw(self):
        gsize = np.int32(self.computegrid.shape)
        gsize[0] = 1
        return gsize

if __name__ == '__main__':

    resc = ComputeRessource()
    nx = 24
    nz = 24
    nt = 4

    grid = GridCL(resc.queues[0], shape=(nx, ), pad=0, dtype=np.float32,
                  zero_boundary=False, nt=nt)
    gridfd = GridCL(resc.queues[0], shape=(nz, nx), pad=0, dtype=np.float32,
                    zero_boundary=False, nt=nt)
    v = np.tile(np.arange(nx, dtype=np.float32), [nz, 1])
    rec = Receiver(required_states=["v"],
                   grids={"gridout": grid, "gridfd": gridfd}, computegrid=grid)

    rec_linpos = cl.array.to_device(resc.queues[0],
                                    gridfd.xyz2lin(1, np.arange(nx)).astype(np.int32))
    rec.linear_test(t=0, reclinpos=rec_linpos)
    rec.dot_test(t=0, reclinpos=rec_linpos)
    rec.backward_test(t=0, reclinpos=rec_linpos)


