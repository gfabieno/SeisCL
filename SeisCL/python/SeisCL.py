#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface to SeisCL
"""
import h5py as h5
import numpy as np
import subprocess
import os
import shutil
from obspy.core import Trace, Stream
from obspy.io.segy.segy import _read_segy

import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
mpl.rcParams['hatch.linewidth'] = 0.5

class SeisCLError(Exception):
    pass

csts = [ 'N', 'ND', 'dh', 'dt', 'NT', 'freesurf', 'FDORDER', 'MAXRELERROR',
        'L', 'f0', 'FL', 'src_pos', 'rec_pos', 'src', 'abs_type', 'VPPML',
        'NPOWER', 'FPML', 'K_MAX_CPML', 'nab', 'abpc', 'pref_device_type',
        'no_use_GPUs', 'MPI_NPROC_SHOT', 'nmax_dev', 'back_prop_type',
        'param_type', 'gradfreqs', 'tmax', 'tmin', 'scalerms',
        'scalermsnorm', 'scaleshot',
        'fmin', 'fmax', 'gradout', 'Hout', 'gradsrcout', 'seisout', 'resout',
        'rmsout', 'movout', 'restype', 'inputres', 'FP16']


class SeisCL:
    """ A class that implements an interface to SeisCL
        (https://github.com/gfabieno/SeisCL.git)
    """

    def __init__(self,

                 N:  np.ndarray = None, ND: int = 2, dh: float = 10,
                 dt: float = 0.0008, NT: int = 875,

                 L: int = 0, f0: float = 15, FL: np.ndarray = np.array(15),

                 FDORDER: int = 8, MAXRELERROR: int = 1, FP16: int = 0,

                 src_pos_all:  np.ndarray = np.empty((5, 0)),
                 rec_pos_all:  np.ndarray = np.empty((8, 0)),
                 src_all:  np.ndarray = None,

                 freesurf: int = 0, abs_type: int = 1,
                 VPPML: float = 3500, NPOWER: float = 2,  FPML: float = 15,
                 K_MAX_CPML: float = 2, nab: int = 16, abpc: float = 6,

                 with_docker: bool = False, with_mpi: bool = False,
                 NP: int = 1, pref_device_type: int = 4,
                 MPI_NPROC_SHOT: int = 1, nmax_dev: int = 1,
                 no_use_GPUs:  np.ndarray = np.array([-1]),

                 gradout: int = 0, Hout: int = 0, gradsrcout: int = 0,
                 back_prop_type: int = 1, cropgrad=True,
                 gradfreqs:  np.ndarray = np.empty((1, 0)), param_type: int = 0,
                 tmax: float = 0, tmin: float = 0, fmin: float = 0,
                 fmax: float = 0, filter_offset: bool = False,
                 offmin: float = -float('Inf'), offmax: float = float('Inf'),
                 inputres: int = 0, restype: int = 0, scalerms: int = 0,
                 scalermsnorm: int = 0, scaleshot: int = 0,

                 seisout: int = 2, resout: int = 0, rmsout: int = 0,
                 movout: int = 0,

                 file: str = "SeisCL", workdir: str = "./seiscl",
                 ):
        """
        Parameters defining the spatial and temporal grid

        :param N:   Grid size [NZ, NX] in 2D or [NZ, NY, NX] in 3D
        :param ND:  Flag for dimension. 3: 3D, 2: 2D P-SV, 21: 2D
                    SH, 22: 2D acoustic
        :param dh:  Grid spatial spacing
        :param dt:  Time step size
        :param NT:  Number of time steps


        Parameters for the Generalized Standard linear solid implementing
        seismic attenuation.

        :param L:   Number of attenuation mechanism (L=0 elastic)
        :param f0:  Central frequency of the relaxation. Also used as the peak
                    frequency of the default rickerwavelet source
        :param FL:  Frequencies of the attenuation mechanism


        Parameters of the finite difference stencils

        :param FDORDER:          Order of the finite difference stencil
                                 Values: 2, 4, 6, 8, 10, 12
        :param MAXRELERROR:      Select method to compute FD coefficients
                                 0: Taylor-coeff.
                                 Holberg-coeff with maximum phase velocity error
                                 1: 0.1 % 2: 0.5 % 3: 1.0 % 4: 3.0 %
        :param FP16:             FP16 computation: 0:FP32
                                 1:FP32 vectorized, (faster than 0)
                                 2: FP16 IO (computation are performed in FP32)
                                 3: FP16 IO + COMP (computation in FP16,
                                 requires CUDA)


        Parameters controlling the acquisition

        :param src_pos_all:      Position of all shots containted in dataset
                                 Array [sx sy sz srcid src_type] x nb sources
                                 The same srcid are fired simulatneously
                                 src_type: 100: Explosive, 0: Force in X, 1:
                                 Force in Y, 2:Force in Z
        :param rec_pos_all:      Position of all the receivers in the dataset
                                 Array [gx gy gz srcid recid - - -] x nb traces
                                 srcid is the source number
                                 recid is the trace number in the record
        :param src_all:          Source signals. NT x number of sources



        Parameters defining the Boundary conditions

        :param freesurf:         Include a free surface 0: no, 1: yes
        :param abs_type:         Absorbing boundary type:
                                 1: CPML, 2: Absorbing layer of Cerjan
        :param VPPML:            Vp velocity near CPML boundary
        :param NPOWER:           Exponent used in CMPL frame update
        :param FPML:             Dominant frequency of the wavefield
        :param K_MAX_CPML:       Coefficient involved in CPML
                                 (may influence simulation stability)
        :param nab:              Width in grid points of the absorbing layer
        :param abpc:             Exponential decay for absorbing layer of Cerjan


        Parameters controlling parallelization

        :param with_docker:      If True, use a docker implementation of SeisCL
        :param with_mpi:         Use mpi parallelization (True) or not.
        :param NP:               Number of MPI processes to launch
        :param pref_device_type: Type of processor used (OpenCL version):
                                 2: CPU, 4: GPU, 8: Accelerator
        :param MPI_NPROC_SHOT:   Maximum number of MPI process (nodes or gpus)
                                 involved in domain decomposition
        :param nmax_dev:         Maximum number of GPUs per process
        :param no_use_GPUs:      Array of device numbers that should not be used


        Parameters controlling how the gradient is computed.

        :param gradout:          Output gradient 1:yes, 0: no
        :param Hout:             Output approximate Hessian 1:yes, 0: no
        :param gradsrcout:       Output source gradient 1:yes, 0: no
        :param back_prop_type:   Type of gradient calculation:
                                 1: backpropagation (elastic only)
                                 2: Discrete Fourier transform
        :param cropgrad:         If true, when calling read_grad, the gradient
                                 in the absorbing boundaries are set to 0.
        :param gradfreqs:        Frequencies of gradient with DFT
        :param param_type:       Type of parametrization:
                                 0:(rho,vp,vs,taup,taus)
                                 1:(rho, M, mu, taup, taus),
                                 2:(rho, Ip, Is, taup, taus)
        :param tmax:             Maximum time of gradient computation
        :param tmin:             Minimum time of gradient computation
        :param fmin:             Maximum frequency of gradient. A butterworh
                                 filter is used to remove frequencies lower
                                 than fmin
        :param fmax:             Minimum frequency of gradient. A butterworh
                                 filter is used to remove frequencies lower
                                 than fmin
        :param filter_offset:    If true, will only allow the minimum and
                                 maximum offset during computation
        :param offmin:           Maximum offset to compute
        :param offmax:           Minimum offset to compute
        :param inputres:         If 1, the gradient computation needs the
                                 residuals (adjoint sources) to be provided.
                                 This allows to  compute the cost and adjoint
                                 sources in python, providing more flexibility.
                                 If 0, SeisCL computes the cost and adjoint
                                 sources.


        The rest of these parameters apply if inputres=0

        :param restype:          Type of costfunction
                                 0: l2 cost. 1: Cross-correlation of traces
        :param scalerms:         Scale each modeled and recorded traces
                                 according to its rms value, then scale residual
                                 by recorded trace rms when computing cost
        :param scalermsnorm:     Scale each modeled and recorded traces
                                 according to its rms value before computing
                                 cost
        :param scaleshot:        Scale all of the traces in each shot by the
                                 shot total rms value when computing cost


        Parameters setting the type of outputs

        :param seisout:          Output seismograms
                                 1: output velocities,
                                 2: output pressure,
                                 3: output stresses, output everything
        :param resout:           Output residuals 1:yes, 0: no
        :param rmsout:           Output rms value of the cost 1:yes, 0: no
        :param movout:           Output movie every n frames


        Parameters for file creation

        :param file:            Base name of the file to create for SeisCL.
                                Created files will be appended thr right suffix,
                                i.e. the model file will be file_model.mat
        :param workdir:         The name of the directory in which to create
                                the file for SeisCL
        """

        self.N = N
        self.ND = ND
        self.dh = dh
        self.dt = dt
        self.NT = NT

        self.L = L
        self.f0 = f0
        self.FL = FL

        self.FDORDER = FDORDER
        self.MAXRELERROR = MAXRELERROR
        self.FP16 = FP16

        self.src_pos_all = src_pos_all
        self.rec_pos_all = rec_pos_all
        self.src_all = src_all
        self.src_pos = None
        self.rec_pos = None
        self.src = None

        self.freesurf = freesurf
        self.abs_type = abs_type
        self.VPPML = VPPML
        self.NPOWER = NPOWER
        self.FPML = FPML
        self.K_MAX_CPML = K_MAX_CPML
        self.nab = nab
        self.abpc = abpc

        self.with_docker = with_docker
        self.with_mpi = with_mpi
        self.NP = NP
        self.pref_device_type = pref_device_type
        self.MPI_NPROC_SHOT = MPI_NPROC_SHOT
        self.nmax_dev = nmax_dev
        self.no_use_GPUs = no_use_GPUs
        # __________Check if SeisCL exists and choose the implementation________
        self.progname = 'SeisCL_MPI'
        self.docker_name = 'seiscl:v0'
        if shutil.which(self.progname):
            if shutil.which("mpirun"):
                self.with_mpi = True
        elif shutil.which("docker"):
            pipes = subprocess.Popen("docker inspect --type=image "
                                     + self.docker_name,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, shell=True)
            stdout, stderr = pipes.communicate()
            if stderr:
                raise SeisCLError("No working SeisCL program found")
            else:
                self.with_docker = True
        else:
            raise SeisCLError("No working SeisCL program found")

        self.gradout = gradout
        self.Hout = Hout
        self.gradsrcout = gradsrcout
        self.back_prop_type = back_prop_type
        self.cropgrad = cropgrad
        self.gradfreqs = gradfreqs
        self.param_type = param_type
        self.tmax = tmax
        self.tmin = tmin
        self.fmin = fmin
        self.fmax = fmax
        self.filter_offset = filter_offset
        self.offmin = offmin
        self.offmax = offmax
        self.inputres = inputres

        self.restype = restype
        self.scalerms = scalerms
        self.scalermsnorm = scalermsnorm
        self.scaleshot = scaleshot

        self.seisout = seisout
        self.resout = resout
        self.rmsout = rmsout
        self.movout = movout

        self.file = file
        self.file_datalist = None
        self.workdir = workdir

        self.wavelet_generator = self.ricker_wavelet
        self.__to_load_names = None

    # _____________________Setters _______________________



    @property
    def csts(self):
        return self.__dict__

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def invert_source(self, src, datao, datam, srcid=None):
        """
        Invert for the source signature provided observed and modeled data

        :param src: The source function
        :param datao: The observed seismogram. Time should be axis 0.
        :param datam: The modeled seismogram. Time should be axis 0.
        :param srcid: The source id. If provided, will overwrite the source
                         function of srcid in self.src_all,
                         so new modeling will use the updated source.
        :return: An array containing the updated source
        """

        dm = np.fft.fft(datam, axis=0)
        do = np.fft.fft(datao, axis=0)
        s = np.fft.fft(np.squeeze(src), axis=0)
        a = np.sum(np.conj(dm) * do, axis=1) / np.sum(dm * np.conj(dm), axis=1)
        src_new = np.real(np.fft.ifft(a * s[:]))
        if srcid is not None:
            self.src_all[:, srcid] = src_new
        return src_new

    def ricker_wavelet(self, f0=None, NT=None, dt=None, tmin=None):
        """
        Compute a ricker wavelet

        :param f0: Peak frequency of the wavelet
        :param NT: Number of time steps
        :param dt: Sampling time
        :param tmin: Time delay before time 0 relative to the center of the
                     wavelet
        :return: ricker: An array containing the wavelet
        """
        if NT is None:
            NT = self.NT
        if f0 is None:
            f0 = self.f0
        if dt is None:
            dt = self.dt
        if tmin is None:
            tmin = -1.5 / f0
        t = np.linspace(tmin, (NT-1) * dt + tmin, num=int(NT))

        ricker = ((1.0 - 2.0 * (np.pi ** 2) * (f0 ** 2) * (t ** 2))
                  * np.exp(-(np.pi ** 2) * (f0 ** 2) * (t ** 2)))

        return ricker

    def save_segy(self, data, name):
        """
        Saves the data in a SEGY file. The
        only header that is set is the time interval.

        :param data: The array of dimension nt X nb traces containing the data
        :param name: Name of the SEGY file to write
        """
        #TODO write receivers and sources position and test

        out = Stream(Trace(data[:, ii], header=dict(delta=self.dt))
                     for ii in range(data.shape[1]))
        out.write(name, format='SEGY', data_encoding=5)

    def read_segy(self, name):
        """
        Read a SEGY file and places returns the data in an array of dimension
        nt X nb traces

        :param name: Name of the segyfile

        :return: A numpy array containing the data
        """
        segy = _read_segy(name)
        return np.transpose(np.array([trace.data for trace in segy.traces]))

    def surface_acquisition_2d(self, dg=2, ds=5, dsx=2, dsz=2, dgsz=0,
                               src_type=100):
        """
        Fills the sources and receivers position (src_pos_all and rec_pos_all)
        for a regular surface acquisition.

        :param dg: Spacing between receivers (in grid points)
        :param ds: Spacing between sources (in grid points)
        :param dsx: X distance from the absorbing boundary of the first source
        :param dsz: Depth of the sources relative to the free surface or of the
                    absorbing boundary
        :param dgsz: Depth of the receivers relative to the depth of the sources
        :param src_type: The type of sources
        """

        self.src_pos_all = np.empty((5, 0))
        self.rec_pos_all = np.empty((8, 0))
        self.src_all = None

        nx = self.N[1]
        dlx = self.nab + dsx
        if self.freesurf == 0:
            dlz = self.nab + dsz
        else:
            dlz = dsz
        gx = np.arange(dlx + dsx, nx - dlx, dg) * self.dh
        gz = gx * 0 + (dlz + dgsz) * self.dh

        for ii in range(dlx, nx - dlx, ds):
            idsrc = self.src_pos_all.shape[1]
            toappend = np.zeros((5, 1))
            toappend[0, :] = (ii) * self.dh
            toappend[1, :] = 0
            toappend[2, :] = dlz * self.dh
            toappend[3, :] = idsrc
            toappend[4, :] = src_type
            self.src_pos_all = np.append(self.src_pos_all, toappend, axis=1)

            gid = np.arange(0, len(gx)) + self.rec_pos_all.shape[1] + 1
            toappend = np.stack([gx,
                                 gz * 0,
                                 gz,
                                 gz * 0 + idsrc,
                                 gid,
                                 gx * 0,
                                 gx * 0,
                                 gx * 0], 0)
            self.rec_pos_all = np.append(self.rec_pos_all, toappend, axis=1)

    def crosshole_acquisition_2d(self, dg=2, ds=5, dsx=2, dsz=2):
        """
        Fills the sources and receivers position (src_pos_all and rec_pos_all)
        for 2D multi-offset crosshole acquisition.

        :param dg: Spacing of the receivers (in grid points)
        :param ds: Spacing of the source (in grid points)
        :param dsx: X distance from the absorbing boundary of the sources (left)
                    and receivers (right)
        :param dsz: Z distance from the absorbing boundary or the free surface
                    for both sources and receivers
        """

        nz, nx = self.N
        dlx = self.nab + dsx
        if self.freesurf == 0:
            dlz = self.nab + dsz
        else:
            dlz = dsz
        gz = np.arange(dlz, nz - dlz, dg) * self.dh
        gx = gz * 0 + (nx - dlx) * self.dh

        for ii in range(dlz, nz - dlz, ds):
            idsrc = self.src_pos_all.shape[1]
            toappend = np.zeros((5, 1))
            toappend[0, :] = dlx * self.dh
            toappend[1, :] = 0
            toappend[2, :] = (ii) * self.dh
            toappend[3, :] = idsrc
            toappend[4, :] = 100
            self.src_pos_all = np.append(self.src_pos_all, toappend, axis=1)

            gid = np.arange(0, len(gx)) + self.rec_pos_all.shape[1] + 1
            toappend = np.stack([gx,
                                 gx * 0,
                                 gz,
                                 gz * 0 + idsrc,
                                 gid,
                                 gx * 0 + 2,
                                 gx * 0,
                                 gx * 0], 0)
            self.rec_pos_all = np.append(self.rec_pos_all, toappend, axis=1)

    def DrawDomain2D(self, model, ax=None, showabs=False, showsrcrec=False):
        """
        Draws the 2D model with absorbing boundary position or receivers and
        sources positions

        :param model: The 2D array of the model to draw
        :param ax: The axis on which to plot
        :param showabs: If True, draws the absorbing boundary
        :param showsrcrec: If True, draws the sources and receivers positions

        """

        nz, nx = self.N
        dh = self.dh

        if not ax:
            _, ax = plt.subplots(1, 1)

        im = ax.imshow(model, extent=[0, nx*dh, nz*dh, 0])
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Depth (m)')

        cbar = plt.colorbar(im)
        cbar.set_label('Velocity (m/s)')

        if showsrcrec:
            self.DrawSrcRec(ax)

        if showabs:
            self.DrawLayers(ax)

        plt.show()

    def DrawSrcRec(self, ax):
        """
        Draws the sources and receivers position

        :param ax: Axis on which to draw the positions
        """
        sx = self.src_pos_all[0]
        sy = self.src_pos_all[2]

        gx = self.rec_pos_all[0, :]
        gy = self.rec_pos_all[2, :]

        ax.plot(sx, sy, marker='.', linestyle='none', markersize=15,
                color='k', label='source')

        ax.plot(gx, gy, marker='v', linestyle='none', markersize=8,
                markerfacecolor="None", markeredgecolor='k',
                markeredgewidth=1, label='receiver')

        plt.legend(loc=4)

    def DrawLayers(self, ax):

        nab = self.nab
        nz, nx = self.N
        dh = self.dh

        AbsRect = {'East': Rectangle((0, 0), nab*dh, nz*dh, linewidth=2,
                                     edgecolor='k', facecolor='none', hatch='/'),
                   'West': Rectangle(((nx-nab)*dh, 0), nab*dh, nz*dh, linewidth=2,
                                     edgecolor='k', facecolor='none', hatch='/'),
                   'South': Rectangle((nab*dh, (nz-nab)*dh), (nx-2*nab)*dh, nab*dh,
                                      linewidth=2, edgecolor='k',
                                      facecolor='none', hatch='/')
                   }

        if not self.freesurf:
            AbsRect['North'] = Rectangle((nab*dh, 0), (nx-2*nab)*dh, nab*dh,
                                         linewidth=2, edgecolor='k',
                                         facecolor='none', hatch='/')
        else:
            ax.spines['top'].set_linewidth(6)
            # Not the best way to do it
            ax.set_title('free surface', fontsize=12)

        if self.abs_type == 1:
            TextLayers = 'PML'
        else:
            TextLayers = 'Cerjan'

        for r in AbsRect:
            ax.add_artist(AbsRect[r])
            rx, ry = AbsRect[r].get_xy()
            cx = rx + AbsRect[r].get_width()/2.0
            cy = ry + AbsRect[r].get_height()/2.0

            if r is 'North' or r is 'South':
                ax.annotate(TextLayers, (cx, cy), color='k', weight='bold',
                            fontsize=12, ha='center', va='center',
                            path_effects=[withStroke(linewidth=3,
                                                     foreground="w")])
            elif r is 'East' or r is 'West':
                ax.annotate(TextLayers, (cx, cy), color='k', weight='bold',
                            fontsize=12, ha='center', va='center', rotation=90,
                            path_effects=[withStroke(linewidth=3,
                                                     foreground="w")])


if __name__ == "__main__":

    tmax = 0.25
    seis = SeisCL()
    seis.dh = 2.5
    seis.dt = 2e-4
    seis.NT = tmax // seis.dt
    seis.f0 = 40
    seis.freesurf = 0
    seis.FDORDER = 4
    seis.abs_type = 1
    seis.seisout = 2
    
    vp = np.zeros([200, 200]) + 3500
    vs = vp * 0 + 2000
    rho = vp * 0 + 2000

    seis.N = np.array(vp.shape)
    seis.surface_acquisition_2d()

    seis.set_forward(seis.src_pos_all[3, :],
                     {"vp": vp, "rho": rho, "vs": vs},
                     withgrad=False)
    seis.execute()
    sdata = seis.read_data()[0]

    clip = 0.01
    vmin = np.min(sdata) * clip
    vmax = -vmin
    plt.imshow(sdata, aspect='auto', vmin=vmin, vmax=vmax)
    plt.show()
