from copy import copy



class Grid:

    def __init__(self, shape, nt=0, dt=0, dh=0, pad=0):

        self.shape = shape
        self.nt = nt
        self.dt = dt
        self.dh = dh
        self.pad = pad

        #TODO support regions
        # self.regions = {}
        # self.add_region("all", Region([s-2*pad for s in shape],
        #                               [self.pad]*len(shape)))
    # def add_region(self, name, region):
    #     self.regions[name] = region

    @property
    def compute_shape(self):
        return tuple([s-2*self.pad for s in self.shape])

    @property
    def compute_offsets(self):
        return tuple([self.pad]*len(self.shape))


class GridSeismic(Grid):

    def __init__(self, shape, nt=0, dt=0, dh=0, nab=0, pad=0, freesurf=False):

        self.shape = shape
        self.nt = nt
        self.dt = dt
        self.dh = dh
        self.nab = nab
        self.pad = pad
        self.freesurf = freesurf



class Region:

    def __init__(self, shape, offsets, name=None):
        self.shape = shape
        self.offsets = offsets
        self.name = name

