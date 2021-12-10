from SeisCL.python.pycl_backend import StateKernelGPU, GridCL, Propagator, Sequence, ReversibleKernel,



# TODO FD headers
# TODO allow transformations in dot product
class UpdateVelocity(ReversibleKernel):

    forward_src = """
__kernel void Sum
(
    __global const float *a, __global const float *b, __global float *res)
{
  int gid = get_global_id(0);
  res[gid] = a[gid] + b[gid];
}
"""

    adjoint_src = """
__kernel void Sum_adj(__global float *a_adj, 
                      __global float *b_adj, 
                      __global float *res_adj)
{
  int gid = get_global_id(0);
  a_adj[gid] += res_adj[gid];
  b_adj[gid] += res_adj[gid];
  res_adj[gid] = 0;
}
"""

    def __init__(self, grids=None, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "cv"]
        self.updated_states = ["vx", "vz"]