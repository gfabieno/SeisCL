from SeisCL.python.stencils.fdcoefficient import FDCoefficients









#TODO: fix derivative templates
class DerivativeTester(FunctionGPU):
    forward_src = """
FUNDEF void DerivativeTester(__global float *a, grid g)
{
    get_pos(&g);
    LOCID float la[__LSIZE__];
    #if LOCAL_OFF==0
        load_local_in(g, a, la);
        load_local_haloz(g, a, la);
        load_local_halox(g, a, la);
        BARRIER
    #endif
    float ax = Dxm(g, la);
    gridstop(g);
    a[indg(g, 0, 0, 0)] = ax;

}
"""

    def __init__(self, grids=None, computegrid=None, fdcoefs=FDCoefficients(),
                 local_size=(16, 16), **kwargs):
        self.required_states = ["a"]
        self.updated_states = ["a"]
        self.default_grids = {"a": "a"}
        self.headers = fdcoefs.header()
        super().__init__(grids=grids, computegrid=computegrid,
                         local_size=local_size, **kwargs)



