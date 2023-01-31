from SeisCL.python.pycl_backend import (GridCL,
                                        FunctionGPU,
                                        ComputeRessource,
                                        State,
                                        )
import numpy as np

class Cerjan(FunctionGPU):

    def __init__(self, grids=None, freesurf=False, abpc=4.0, nab=2, pad=2,
                 required_states=(), **kwargs):
        super().__init__(grids, **kwargs)
        self.abpc = abpc
        self.nab = nab
        self.pad = pad
        self.required_states = required_states
        self.updated_states = required_states
        self.taper = np.exp(np.log(1.0-abpc/100)/nab**2 * np.arange(nab) **2)
        self.taper = np.concatenate([self.taper,  self.taper[-pad:][::-1]])
        self.taper = np.expand_dims(self.taper, -1)
        self.freesurf = freesurf

    def __call__(self, states, initialize=True, **kwargs):

        if initialize:
            states = self.initialize(states)
        kwargs = self.make_kwargs_compatible(**kwargs)
        saved = {}
        for el in self.updated_states:
            saved[el] = []
            b = self.nab + self.pad
            if not self.freesurf:
                saved[el].append(states[el][:b, :].copy())
            saved[el].append(states[el][-b:, :].copy())
            saved[el].append(states[el][:, :b].copy())
            saved[el].append(states[el][:, -b:].copy())

        self._forward_states.append(saved)

        return self.forward(states, **kwargs)

    def forward(self, states, **kwargs):

        for el in self.required_states:
            if not self.freesurf:
                states[el][:self.nab+2, :] *= self.taper[::-1]
            states[el][-self.nab-2:, :] *= self.taper

            tapert = np.transpose(self.taper)
            states[el][:, :self.nab+2] *= tapert[:, ::-1]
            states[el][:, -self.nab-2:] *= tapert

        return states

    def adjoint(self, adj_states, states, **kwargs):

        return self.forward(adj_states, **kwargs)

    def backward(self, states, **kwargs):

        torestore = self._forward_states.pop()
        for el in torestore:
            b = self.nab + self.pad
            if not self.freesurf:
                states[el][:b, :] = torestore[el][0]
            states[el][-b:, :] = torestore[el][-3]
            states[el][:, :b] = torestore[el][-2]
            states[el][:, -b:] = torestore[el][-1]

        return states