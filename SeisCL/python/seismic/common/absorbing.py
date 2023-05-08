from SeisCL.python import Function, FunctionGPU, ComputeGrid
import numpy as np
try:
    import pyopencl as cl
    from pyopencl.array import Array, to_device, empty, zeros
except ImportError:
    def compile(ctx, src, options):
        raise ImportError("pyopencl not installed")


class Cerjan(Function):

    def __init__(self, *args, freesurf=False, abpc=4.0, nab=2, pad=2):
        super().__init__(*args)
        self.abpc = abpc
        self.nab = nab
        self.pad = pad
        self.taper = np.exp(np.log(1.0-abpc/100)/nab**2 * np.arange(nab) **2)
        self.taper = np.expand_dims(self.taper, -1)

        self.taperi = self.taper[::-1]
        self.tapert = np.transpose(self.taper)
        self.taperti = np.transpose(self.taperi)
        self.freesurf = freesurf

    # def updated_regions(self, var):
    #     regions = []
    #     pad = self.pad
    #     ndim = len(var.shape)
    #     b = self.nab + pad
    #     for dim in range(ndim):
    #         region = [Ellipsis for _ in range(ndim)]
    #         region[dim] = slice(pad, b)
    #         if dim != 0 or not self.freesurf:
    #             regions.append(tuple(region))
    #         region = [Ellipsis for _ in range(ndim)]
    #         region[dim] = slice(-b, -pad)
    #         regions.append(tuple(region))
    #     return regions

    def forward(self, *args, direction="data"):
        pad = self.pad
        nab = self.nab
        for arg in args:
            d = getattr(arg, direction)
            if not self.freesurf:
                d[pad:nab+pad, :] *= self.taperi
            d[-nab-pad:-pad, :] *= self.taper

            d[:, pad:nab+pad] *= self.taperti
            d[:, -nab-pad:-pad] *= self.tapert
        return args

    def linear(self, *args):
        return self.forward(*args, direction="lin")

    def adjoint(self, *args):
        return self.forward(*args, direction="grad")


class CerjanGPU(Cerjan, FunctionGPU):

    def build_src(self, *args, direction="forward"):
        src = """
            float taper[%d] = {%s};
        """ % (self.nab, ", ".join([str(x) for x in self.taper.flatten()]))
        shape = args[0].shape
        nd = len(shape)
        if nd == 2:
            strpos = "g.z, g.x"
        elif nd == 3:
            strpos = "g.z, g.y, g.x"
        else:
            raise ValueError("Only 2D and 3D supported")

        # upper boundary in z if not freesurface
        if not self.freesurf:
            src += """
                if (g.z >= %d && g.z < %d) {
                """ % (self.pad, self.nab+self.pad)
            for ii, arg in enumerate(args):
                strname = "arg%d" % ii
                if direction == "linear":
                    strname += "_lin"
                elif direction == "adjoint":
                    strname += "_adj"
                src += """
                    %s(%s) *= taper[%d - (g.z - %d)];
                """ % (strname, strpos, self.nab-1, self.pad)
            src += """
                }
            """

        # lower boundary in z
        src += """
            if (g.z >= %d && g.z < %d) {
            """ % (shape[0]-self.nab-self.pad, shape[0]-self.pad)
        for ii, arg in enumerate(args):
            strname = "arg%d" % ii
            if direction == "linear":
                strname += "_lin"
            elif direction == "adjoint":
                strname += "_adj"
            src += """
                %s(%s) *= taper[g.z - %d];
            """ % (strname, strpos, shape[0]-self.nab-self.pad)
        src += """
            }
        """

        # left boundary in x
        src += """
            if (g.x >= %d && g.x < %d) {
            """ % (self.pad, self.nab+self.pad)
        for ii, arg in enumerate(args):
            strname = "arg%d" % ii
            if direction == "linear":
                strname += "_lin"
            elif direction == "adjoint":
                strname += "_adj"
            src += """
                %s(%s) *= taper[%d - (g.x - %d)];
            """ % (strname, strpos, self.nab-1, self.pad)
        src += """
            }
        """

        # right boundary in x
        src += """
            if (g.x >= %d && g.x < %d) {
            """ % (shape[nd-1]-self.nab-self.pad, shape[nd-1]-self.pad)
        for ii, arg in enumerate(args):
            strname = "arg%d" % ii
            if direction == "linear":
                strname += "_lin"
            elif direction == "adjoint":
                strname += "_adj"
            src += """
                %s(%s) *= taper[g.x - %d];
            """ % (strname, strpos, shape[nd-1]-self.nab-self.pad)
        src += """
            }
        """

        # boundaries in y (if 3D)
        if nd == 3:
            src += """
                }
                if (g.y >= %d && g.y < %d) {
                """ % (self.pad, self.nab+self.pad)
            for ii, arg in enumerate(args):
                strname = "arg%d" % ii
                if direction == "linear":
                    strname += "_lin"
                elif direction == "adjoint":
                    strname += "_adj"
                src += """
                    %s(%s) *= taper[%d - (g.y - %d)];
                """ % (strname, strpos, self.nab-1, self.pad)
            src += """
                }
            """

            # right boundary in y
            src += """
                if (g.y >= %d && g.y < %d) {
                """ % (shape[1]-self.nab-self.pad, shape[1]-self.pad)
            for ii, arg in enumerate(args):
                strname = "arg%d" % ii
                if direction == "linear":
                    strname += "_lin"
                elif direction == "adjoint":
                    strname += "_adj"
                src += """
                    %s(%s) *= taper[g.y - %d];
                """ % (strname, strpos, shape[1]-self.nab-self.pad)
                src += """
                }
            """
        return src

    def forward(self, *args):

        src = self.build_src(*args, direction="forward")
        grid = ComputeGrid(shape=[s - 2*args[0].pad for s in args[0].shape],
                           queue=self.queue,
                           origin=[args[0].pad for _ in args[0].shape])
        self.callgpu(src, "forward", grid, *args)
        return args

    def linear(self, *args):

        src = self.build_src(*args, direction="linear")
        grid = ComputeGrid(shape=[s - 2*args[0].pad for s in args[0].shape],
                           queue=self.queue,
                           origin=[args[0].pad for _ in args[0].shape])
        self.callgpu(src, "linear", grid, *args)
        return args

    def adjoint(self, *args):

        src = self.build_src(*args, direction="adjoint")
        grid = ComputeGrid(shape=[s - 2*args[0].pad for s in args[0].shape],
                           queue=self.queue,
                           origin=[args[0].pad for _ in args[0].shape])
        self.callgpu(src, "adjoint", grid, *args)
        return args