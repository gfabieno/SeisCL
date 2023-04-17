import numpy as np

from SeisCL.python.common.Acquisition import Acquisition
from SeisCL.python.Losses import NormalizedL2
from SeisCL.python.tape import TapedFunction, Variable, Function
from typing import List


class Propagator:

    def __init__(self, grid, fdorder=4):

        self.fdorder = fdorder
        self.grid = grid


    def initialize(self, **kwargs):
        """
        Set wavefield values to 0, and assign values to variables specified in
        **kwargs (such as model parameters vp, vs ...)
        """
        raise NotImplementedError

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
    def compute_loss(self, shot, dmod, dobs):
        dmod = self.propagator.propagate(shot, **dmod)
        self.lossfun(self.loss, dobs, dmod)
        return dmod, self.loss

    def __call__(self, shotids: List[int] = None,
                 compute_gradient: bool = False,
                 **kwargs):
        """
        Compute the modelled data for the provided shotids. If `compute_gradient`
        is True, compute the loss and the gradient as well.

        """
        if shotids is None:
            shotids = range(len(self.acquisition.shots))

        for ii in shotids:
            self.propagator.initialize(**kwargs)
            self.acquisition.shots[ii].init_dmod()
            shot = self.acquisition.shots[ii]
            dmod = self.acquisition.shots[ii].dmod
            self.propagator.propagate(shot, **dmod)

        return (self.acquisition.shots[ii] for ii in shotids)

    def gradient(self, shotids: List[int] = None, **kwargs):
        """
        Compute gradient
        """
        if shotids is None:
            shotids = range(len(self.acquisition.shots))

        self.loss.initialize()
        gradients = {}
        for name, var in kwargs.items():
            if type(var) is Variable:
                gradients[name] = np.zeros_like(var.grad)

        for ii in shotids:
            self.propagator.initialize(**kwargs)
            self.acquisition.shots[ii].init_dmod()
            shot = self.acquisition.shots[ii]
            dobs = self.acquisition.shots[ii].dobs
            dmod = self.acquisition.shots[ii].dmod
            dmod, _ = self.compute_loss(shot, dmod, dobs)
            self.compute_loss(shot, dmod, dobs, mode="adjoint")
            for name in gradients:
                gradients[name] += getattr(self.propagator, name).grad
        for name in gradients:
            getattr(self.propagator, name).grad = gradients[name]
        return self.loss.data, gradients


