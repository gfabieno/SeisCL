from .function import Function, ReversibleFunction
from .kernel import Kernel

#TODO memoize forward, linear and adjoint kernels
class FunctionGPU(Function):
    """
    Base class for all GPU functions. As for `Function`, override the methods
    `forward`, `linear`  and `adjoint`. You should define the input and output
    signatures, write the body of the GPU kernel as a string and pass it to
    the method `gpukernel` to launch the kernel on GPU.
    """

    def __init__(self, queue, local_size=None, options=None):

        self.queue = queue
        self.local_size = local_size
        if options is None:
            options = []
        self.options = options
        #TODO clean up header creation
        self.header = ""
        self._gpukernel = {"forward": None, "linear": None, "adjoint": None}
        super().__init__()

    def gpukernel(self, src, mode, grid, *args, **kwargs):
        """
        Launch the GPU kernel. The kernel is compiled and cached the first time
        it is called. The kernel is then launched with the arguments passed to
        this method.

        :param src: String containing the body of the kernel. If a List of
                    string is passed, multiple kernels are compiled and
                    launched in order.
        :param mode: Either 'forward', 'linear' or 'adjoint'
        :param grid: A `Grid` object containing the grid on which the kernel
                     is launched.
        :param args: The list of arguments to pass to the kernel. Must contain
                     only `Variable` objects, int, float or bool.
        :param kwargs: Same as args, but passed as keyword arguments.
        """
        if self._gpukernel[mode] is None:
            if mode not in ["forward", "linear", "adjoint"]:
                raise ValueError("mode must be 'forward', 'linear' or "
                                 "'adjoint'")
            name = self.__class__.__name__ + "_" + mode

            self._gpukernel[mode] = Kernel(self.queue,
                                           name,
                                           self.signature,
                                           src,
                                           self.header,
                                           self.options,
                                           mode,
                                           self.local_size
                                           )
        self._gpukernel[mode](grid, *args, **kwargs)
    # def cache_states(self, states):
    #     for el in self.updated_states:
    #         if el in self.updated_regions:
    #             for ii, region in enumerate(self.updated_regions[el]):
    #                 self.grids[el].copy_array(self._forward_states[el][ii][..., self.ncall],
    #                                           states[el][region])
    #         else:
    #             self._forward_states[el][..., self.ncall] = states[el]
    #     self.ncall += 1


class ReversibleFunctionGPU(ReversibleFunction, FunctionGPU):
    pass
