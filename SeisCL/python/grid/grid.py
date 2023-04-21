

class Grid:

    def __init__(self, shape, nt=0, dt=0, dh=0, nab=0, pad=0, freesurf=False):

        self.shape = shape
        self.nt = nt
        self.dt = dt
        self.dh = dh
        self.nab = nab
        self.pad = pad
        self.freesurf = freesurf