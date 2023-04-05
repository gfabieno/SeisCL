import numpy as np
from typing import List, Tuple
from copy import deepcopy
import unittest
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
mpl.rcParams['hatch.linewidth'] = 0.5



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

    def __init__(self, grid, shots: Tuple[Shot] = None):
        """

        :param grid A `Grid` object defining the size of the model
        :param shots: A tuple of `Shot` objects defining all shots in a survey.
                      You can build shots with the regular2d method, if desired.
        """
        self.grid = grid
        self.shots = shots

    @property
    def shots(self):
        return self._shots

    @shots.setter
    def shots(self, shots):
        if shots is None:
            return
        xmax = (self.grid.nx - 2*self.grid.nab) * self.grid.dh
        if self.grid.freesurf:
            zmax = (self.grid.nz - self.grid.nab) * self.grid.dh
        else:
            zmax = (self.grid.nz - 2*self.grid.nab) * self.grid.dh
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
        self._shots = tuple(shots)

    def regular2d(self, dir: str = "x", dg: int = 2, ds: int = 5,
                  sx0: int = 2, sz0: int = 2, gx0: int = 2, gz0: int = 2,
                  src_type: str = "p", rec_types: List[str] = ("p",),
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
        xmax = nx - 2 * self.grid.nab
        if self.grid.freesurf:
            zmax = nz - self.grid.nab
        else:
            zmax = nz - 2 * self.grid.nab

        receivers = []
        if dir == "x":
            for rec_type in rec_types:
                receivers.append([Receiver(x=x, z=gz0, type=rec_type)
                                  for x in np.arange(gx0, xmax - gx0, dg) * dh])
        else:
            for rec_type in rec_types:
                receivers.append([Receiver(x=gx0, z=z, type=rec_type)
                                  for z in np.arange(gz0, zmax - gz0, dg) * dh])
        receivers = [item for sublist in receivers for item in sublist]
        for ii, rec in enumerate(receivers):
            rec.trid = ii

        sid = 0
        shots = []
        if dir == "x":
            spos = range(sx0, xmax - sx0, ds)
        else:
            spos = range(sz0, zmax - sz0, ds)
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
            shots.append(Shot(sources=[source], receivers=receivers, sid=sid))
            sid += 1
        self.shots = shots
        return self.shots

    def draw_model(self, model=None, ax=None, showabs=True, showsrcrec=True):
        """
        Draws the 2D model with absorbing boundary position or receivers and
        sources positions

        :param model: The 2D array of the model to draw
        :param ax: The axis on which to plot
        :param showabs: If True, draws the absorbing boundary
        :param showsrcrec: If True, draws the sources and receivers positions

        """

        nx = self.grid.nx
        nz = self.grid.nz
        dh = self.grid.dh
        nab = self.grid.nab

        xmin = -nab*dh
        xmax = (nx-nab)*dh
        if self.grid.freesurf:
            zmin = 0
            zmax = nz * dh
        else:
            zmin = -nab*dh
            zmax = (nz-nab)*dh

        if not ax:
            _, ax = plt.subplots(1, 1)

        if model is None:
            model = np.zeros((nz, nx), dtype=float)
        im = ax.imshow(model, extent=[xmin, xmax, zmax, zmin])
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Depth (m)')

        if np.max(model) > 0:
            cbar = plt.colorbar(im)
            cbar.set_label('Velocity (m/s)')

        if showsrcrec:
            for ii, shot in enumerate(self.shots):
                gx = [r.x for r in shot.receivers]
                gz = [r.z for r in shot.receivers]
                if ii == 0:
                    label = "receiver"
                else:
                    label = None
                ax.plot(gx, gz, marker='v', linestyle='none', markersize=3,
                        markerfacecolor="w", markeredgecolor='w',
                        markeredgewidth=1, label=label)
            for ii, shot in enumerate(self.shots):
                sx = [s.x for s in shot.sources]
                sz = [s.z for s in shot.sources]
                if ii == 0:
                    label = "source"
                else:
                    label = None
                ax.plot(sx, sz, marker='.', linestyle='none', markersize=7,
                        color='r', label=label)
            plt.legend(loc=4)

        if showabs:
            abs_rect = {'East': Rectangle((xmin, zmin), nab*dh, nz*dh,
                                          linewidth=2, edgecolor='k',
                                          facecolor='none',  hatch='/'),
                        'West': Rectangle((xmax - nab*dh, zmin), nab*dh, nz*dh,
                                          linewidth=2, edgecolor='k',
                                          facecolor='none', hatch='/'),
                        'South': Rectangle((xmin + nab*dh, zmax - nab*dh),
                                           (nx-2*nab)*dh, nab*dh,
                                           linewidth=2, edgecolor='k',
                                           facecolor='none', hatch='/')
                        }

            if not self.grid.freesurf:
                abs_rect['North'] = Rectangle((nab*dh, 0), (nx-2*nab)*dh,
                                              nab*dh, linewidth=2,
                                              edgecolor='k', facecolor='none',
                                              hatch='/')
            else:
                ax.set_title('free surface', fontsize=12)

            title = "Abs. Boundary"

            for r in abs_rect:
                ax.add_artist(abs_rect[r])
                rx, ry = abs_rect[r].get_xy()
                cx = rx + abs_rect[r].get_width()/2.0
                cy = ry + abs_rect[r].get_height()/2.0

                if r == 'North' or r == 'South':
                    ax.annotate(title, (cx, cy), color='k', weight='bold',
                                fontsize=12, ha='center', va='center',
                                path_effects=[withStroke(linewidth=3,
                                                         foreground="w")])
                elif r == 'East' or r == 'West':
                    ax.annotate(title, (cx, cy), color='k', weight='bold',
                                fontsize=12, ha='center', va='center',
                                rotation=90,
                                path_effects=[withStroke(linewidth=3,
                                                         foreground="w")])

        plt.show()

    def stacking_diagram(self):

        _, ax = plt.subplots(1, 1)
        for ii, shot in enumerate(self.shots):
            x = [r.x for r in shot.receivers]
            z = [shot.sources[0].x for _ in shot.receivers]
            if ii == 0:
                label = 'receiver'
            else:
                label = None
            ax.plot(x, z, marker='v', linestyle='none', markersize=4,
                    markerfacecolor="k", markeredgecolor='k',
                    markeredgewidth=1, label=label)

            x = [s.x for s in shot.sources]
            z = [shot.sources[0].x for _ in shot.sources]
            if ii == 0:
                label = 'source'
            else:
                label = None
            ax.plot(x, z, marker='.', linestyle='none', markersize=7,
                    color='r', label=label)
            plt.legend(loc=4)
        plt.legend(loc=4)
        plt.title("Stacking diagram")
        plt.xlabel("Receiver x (m)")
        plt.ylabel("Source x (m)")
        plt.show()


class AcquisitionTester(unittest.TestCase):

    def test_position_checking(self):
        grid = Grid(nd=2, nx=200, ny=None, nz=100, nt=500, dt=0.0001, dh=2,
                    nab=16, freesurf=True)
        shot = Shot(receivers=[], sources=[Source(x=600)], sid=0)
        with self.assertRaises(ValueError):
            acquisition = Acquisition(grid, [shot])


if __name__ == '__main__':
    grid = Grid(nd=2, nx=200, ny=None, nz=100, nt=500, dt=0.0001, dh=2,
                nab=16, freesurf=True)
    acquisition = Acquisition(grid)
    acquisition.regular2d()
    acquisition.draw_model()
    acquisition.stacking_diagram()