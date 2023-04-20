import numpy as np

from SeisCL.python.common.Acquisition import Acquisition, Shot
from SeisCL.python.Losses import NormalizedL2
from SeisCL.python.tape.tape import TapedFunction, Variable
from typing import List


class Propagator:

    def __init__(self, grid, fdorder=4):

        self.fdorder = fdorder
        self.grid = grid

    def propagate(self, shot, *args, **kwargs):
        raise NotImplementedError


class FWI:

    def __init__(self, acquisition: Acquisition,
                 propagator: Propagator,
                 loss=NormalizedL2):

        self.acquisition = acquisition
        self.propagator = propagator
        self.lossfun = loss()
        self.records = {} #store the wavefields and seismograms
        self.loss = Variable(shape=(1,))


    @TapedFunction
    def compute_loss(self, shot, *args, **kwargs):
        dmod, (_) = self.propagator.propagate(shot, *args, **kwargs)
        self.lossfun(self.loss, shot.dobs, dmod)
        return dmod, self.loss

    def __call__(self, shots: List[Shot], *args,
                 compute_gradient: bool = False,
                 **kwargs):
        """
        Compute the modelled data for the provided shotids. If `compute_gradient`
        is True, compute the loss and the gradient as well.

        """
        # if shots is None:
        #     shots = self.acquisition.shots
        if compute_gradient:
            return self.gradient(shots, *args, **kwargs)
        else:
            return self.forward(shots, *args, **kwargs)

    def forward(self, shots: List[Shot], *args, **kwargs):
        """
        Compute the modelled data for the provided shotids.

        """
        for shot in shots:
            self.propagator.propagate(shot, *args, **kwargs)

        return shots

    def gradient(self, shots: List[Shot], *args, **kwargs):
        """
        Compute gradient
        """

        self.loss.initialize()
        gradients = {}
        for name, var in kwargs.items():
            if type(var) is Variable:
                gradients[name] = np.zeros_like(var.grad)

        for shot in shots:
            self.propagator.initialize(**kwargs)
            dmod, _ = self.compute_loss(shot, *args, **kwargs)
            self.compute_loss(shot, *args, mode="adjoint", **kwargs)
            for name in gradients:
                gradients[name] += getattr(self.propagator, name).grad
        for name in gradients:
            getattr(self.propagator, name).grad = gradients[name]
        return self.loss.data, gradients


