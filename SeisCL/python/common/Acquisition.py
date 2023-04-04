import numpy as np
from typing import List
from copy import deepcopy
import matplotlib.pyplot as plt


class Receiver:

    def __init__(self, x: float = 0, y: float = 0, z: float = 0,
                 type: str = "z", trid: int = 0):
        """
        Define the position of a receiver.

        :param x: x location in meters
        :param y: y location in meters
        :param z: z location in meters
        :param type: Type of receiver:
                     "x": force in x
                     "y": force in y
                     "z": force in z
                     "p": pressure
        :param trid: Trace id in the segy file.
        """
        self.x = x
        self.y = y
        self.z = z
        self.type = type
        self.trid = trid


class Source:

    def __init__(self, x: float = 0, y: float = 0, z: float = 0,
                 wavelet: np.ndarray = None, type: str = "z"):
        """
        Define the position and wavelet of a source.

        :param x: x location in meters.
        :param y: y location in meters.
        :param z: z location in meters
        :param type: Type of receiver:
                     "x": force in x
                     "y": force in y
                     "z": force in z
                     "p": pressure
        :param wavelet: The source wavelet
        """
        self.wavelet = wavelet
        self.x = x
        self.y = y
        self.z = z
        self.type = type
        self.nt = wavelet.size


class Shot:

    def __init__(self, sources: List, receivers: List, sid: int):
        """
        Define a shot that may contain simultaneous sources
        and multiple receivers.

        :param sources: A list of Source objects fired simultaneously.
        :param receivers: A list of Receiver objects
        :param sid: The source unique shot id.
        """
        self.sources = sources
        self.receivers = receivers
        self.sid = sid


def ricker_wavelet(f0=None, nt=None, dt=None, tmin=None):
    """
    Compute a ricker wavelet

    :param f0: Peak frequency of the wavelet
    :param nt: Number of time steps
    :param dt: Sampling time
    :param tmin: Time delay before time 0 relative to the center of the
                 wavelet
    :return: ricker: An array containing the wavelet
    """
    if tmin is None:
        tmin = -1.5 / f0
    t = np.linspace(tmin, (nt - 1) * dt + tmin, num=int(nt))

    ricker = ((1.0 - 2.0 * (np.pi ** 2) * (f0 ** 2) * (t ** 2))
              * np.exp(-(np.pi ** 2) * (f0 ** 2) * (t ** 2)))

    return ricker


class Grid:

    def __init__(self, nd, nx, ny, nz, nt, dt, dh, nab, freesurf):

        self.nd = nd
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nt = nt
        self.dt = dt
        self.dh = dh
        self.nab = nab
        self.freesurf = freesurf


class Acquisition:
    """
    Defines the geometry of the seismic acquisition. The source and receiver
    position origin starts just outside the absorbing boundary. The system of
    coordinates is right-handed: z points downward, x points to the right
    and y points to the front.

    """

    def __init__(self, grid, shots: List[Shot] = None):
        """

        :param grid A `Grid` object defining the size of the model
        :param shots: A list of `Shot` objects defining all shots in a survey.
                      You can build shots with the regular2d method, if desired.
        """
        self.grid = grid
        self.shots = shots

    @property
    def shots(self):
        return self._shots

    @shots.setter
    def shots(self, shots):
        xmax = (self.grid.nx - 2*self.grid.nab) * self.grid.dh
        if self.grid.freesurf:
            zmax = (self.grid.nx - self.grid.nab) * self.grid.dh
        else:
            zmax = (self.grid.nx - 2*self.grid.nab) * self.grid.dh
        for shot in shots:
            for source in shot.sources:
                if source.x > xmax or source.x < 0:
                    raise ValueError("Source located at x=%f m is outside "
                                     "the grid with a size of %f m"
                                     % (source.x, xmax))
                if source.z > zmax or source.z < 0:
                    raise ValueError("Source located at z=%f m is outside "
                                     "the grid with a size of %f m"
                                     % (source.z, zmax))
            for receiver in shot.receivers:
                if receiver.x > xmax or receiver.x < 0:
                    raise ValueError("Receiver located at x=%f m is outside "
                                     "the grid with a size of %f m"
                                     % (receiver.x, xmax))
                if receiver.z > zmax or receiver.z < 0:
                    raise ValueError("Receiver located at z=%f m is outside "
                                     "the grid with a size of %f m"
                                     % (receiver.z, zmax))
        self._shots = shots


    def regular2d(self, dir: str = "x", dg: int = 2, ds: int = 5,
                  sx0: int = 2, sz0: int = 2, gx0: int = 2, gz0: int = 2,
                  src_type: str = "p", rec_types: List[str] = ["p"],
                  wavelet: np.ndarray = None, f0: float = 15):
        """
        Build source and receiver position for a regular surface acquisition.

        :param dir  The direction along which the source and receiver positions
                    vary. Set to "x" for surface acquistion and "z" for
                    crosshole. Default to "x"
        :param dg: Spacing between receivers (in grid points)
        :param ds: Spacing between sources (in grid points)
        :param sx0: X distance from the absorbing boundary of the first source
        :param sz0: Depth of the sources relative to the free surface or of the
                    absorbing boundary
        :param gx0: X distance from the absorbing boundary of the first receiver
        :param gz0: Depth of the receivers relative to the free surface or of
                    the absorbing boundary
        :param src_type: The type of sources. Default to "p" for pressure
        :param rec_types: A list of type of receivers to use. Default to ["p"]
        :param wavelet:   An array containing the source wavelet. Default to
                          a ricker wavelet with a center frequency of f0
        :param f0:      The center frequency of the default ricker wavelet
        """

        if wavelet is None:
            wavelet = ricker_wavelet(f0, nt=self.grid.nt, dt=self.grid.dt)
        nx = self.grid.nx
        nz = self.grid.nz
        dh = self.grid.dh

        receivers = []
        if dir == "x":
            for rec_type in rec_types:
                receivers.append([Receiver(x=x, z=gz0, type=rec_type)
                                  for x in np.arange(sx0, nx - sx0, dg) * dh])
        else:
            for rec_type in rec_types:
                receivers.append([Receiver(x=gx0, z=z, type=rec_type)
                                  for z in np.arange(gz0, nz - gz0, dg) * dh])
        for ii, rec in enumerate(receivers):
            rec.trid = ii

        sid = 0
        self.shots = []
        if dir == "x":
            spos = range(sx0, nx - sx0, ds)
        else:
            spos = range(sz0, nz - sz0, ds)
        for ii in spos:
            if dir == "x":
                x = ii*dh
                z = sz0*dh
            else:
                x = sx0*dh
                z = ii*dh
            source = Source(x=x, z=z, wavelet=wavelet, type=src_type)
            receivers = deepcopy(receivers)
            for rec in receivers:
                rec.trid += len(receivers)
            self.shots.append(Shot(sources=[source], receivers=receivers,
                                   sid=sid))
            sid += 1

        return self.shots

