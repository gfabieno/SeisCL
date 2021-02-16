#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:35:05 2016

@author: gabrielfabien-ouellet
"""

import h5py as h5
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
sys.path.append('../')
from SeisCL.SeisCL import SeisCL, SeisCLError


def test_seisout(seis, plot=False):
    
    seis = define_rec_src(seis)
    pars = {}
    pars['vp'] = np.zeros(seis.N) + 3500
    pars['vs'] = pars['vp'] * 0 + 2000
    pars['rho'] = pars['vp'] * 0 + 2000
    if seis.L > 0:
        pars['taup'] = pars['vp'] * 0 + 0.1
        pars['taus'] = pars['vp'] * 0 + 0.1
    
    for seisout in range(1, 5):
        ref = None
        print("    Testing seisout=%d" % seisout)
        for fp16 in range(0, 2):
            print("        Testing fp16=%d....." % fp16, end='')
            seis.FP16 = fp16
            seis.seisout = seisout
            try:
                seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=False)
                seis.execute()
                data = seis.read_data()
                if plot:
                    clip = 1.0
                    vmin = np.min(data[0]) * clip
                    vmax = -vmin
                    plt.imshow(data[0], aspect='auto', vmin=vmin, vmax=vmax)
                    plt.show()
                if ref is None:
                    ref = data
                    print("passed")
                else:
                    err = (np.sum([np.sum((d-r)**2) for d, r in zip(data, ref)])
                           / np.sum([np.sum(r**2) for r in ref]))
                    if err > 0.001:
                        if plot:
                            plt.imshow(data[0]-ref[0], aspect='auto')
                            plt.show()
                        raise SeisCLError("    Error with reference too large: %e"
                                          % err)
                    print("passed (error %e)" % err)
            except SeisCLError as msg:
                print("failed:")
                print(msg)


def test_backpropagation(seis, plot=False, ngpu=1, nmpi=1):
    
    seis = define_rec_src(seis)
    pars = {}
    
    pars['vp'] = np.zeros(seis.N) + 3500
    pars['vs'] = pars['vp'] * 0 + 2000
    pars['rho'] = pars['vp'] * 0 + 2000
    slices = [slice(d//2-5, d//2+5) for d in seis.N]
    slicesp = [slice(seis.nab, -seis.nab) for _ in seis.N]
    
    slicesp[1:-1] = [d//2 for d in seis.N[1:-1]]
    if seis.csts["freesurf"] == 1:
        slices[0] = slice(5, 15)
        slicesp[0] = slice(0, -seis.nab)
    slicesp = [0, 1] + slicesp
    slicesp = tuple(slicesp)
    slices = tuple(slices)
    if seis.L > 0:
        pars['taup'] = pars['vp'] * 0 + 0.1
        pars['taus'] = pars['vp'] * 0 + 0.1
    
    seis.FP16 = 0
    seis.nmax_dev = 1
    seis.NP = 1
    seis.file_din = seis.workdir + '/SeisCL_din.mat'

    seis.movout = 1
    seis.seisout = 1
    pars['vp'] = np.zeros(seis.N) + 3500
    pars['vp'][slices] = 4000
    seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=False)
    seis.execute()
    data = seis.read_data()
    seis.write_data({"vx": data[0]}, filename="SeisCL_din.mat")
    pars['vp'] = np.zeros(seis.N) + 3500

    if ngpu > 1 or nmpi > 1:
        seis.nmax_dev = ngpu
        seis.NP = nmpi
    
    for fp16 in range(1, 4):
        print("    Testing FP16=%d....." % fp16, end='')
        seis.FP16 = fp16
        try:

            seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=True)
            seis.execute()
            file = h5.File("./seiscl/SeisCL_movie.mat", "r")
            mov = file['movvx'][slicesp]
            file.close()
            
            if plot:
                plt.imshow(mov, aspect='auto')
                plt.show()

            err = np.max(mov)
            if err > 1e-4:
                raise SeisCLError("    Error with data reference too large: %e"
                                  % err)
            print("passed (error %e)" % err)
        except SeisCLError as msg:
            print("failed:")
            print(msg)

    seis.movout = 0
    seis.seisout = 2
    seis.nmax_dev = 1
    seis.NP = 1


def test_fp16_forward(seis, ref=None, plot=False, ngpu=1, nmpi=1):
    
    seis = define_rec_src(seis)
    pars = {}
    pars['vp'] = np.zeros(seis.N) + 3500
    pars['vs'] = pars['vp'] * 0 + 2000
    pars['rho'] = pars['vp'] * 0 + 2000
    if seis.L > 0:
        pars['taup'] = pars['vp'] * 0 + 0.1
        pars['taus'] = pars['vp'] * 0 + 0.1
    
    if ngpu > 1 or nmpi > 1:
        seis.FP16 = 0
        seis.nmax_dev = 1
        seis.NP = 1
        seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=False)
        seis.execute()
        data = seis.read_data()
        ref = data
        seis.nmax_dev = ngpu
        seis.NP = nmpi

    for fp16 in range(0, 4):
        print("    Testing FP16=%d....." % fp16, end='')
        seis.FP16 = fp16
        try:
            seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=False)
            seis.execute()
            data = seis.read_data()
            if plot:
                clip = 1.0
                vmin = np.min(data) * clip
                vmax = -vmin
                plt.imshow(data[0], aspect='auto', vmin=vmin, vmax=vmax)
                plt.show()
            if ref is None:
                ref = data
                print("passed")
            else:
                err = (np.sum([np.sum((d-r)**2) for d, r in zip(data, ref)])
                       / np.sum([np.sum(r**2) for r in ref]))
                if err > 0.001:
                    if plot:
                        plt.imshow(data[0]-ref[0], aspect='auto')
                        plt.show()
                    raise SeisCLError("    Error with data referance too large:"
                                      " %e" % err)
                print("passed (error %e)" % err)
        except SeisCLError as msg:
            print("failed:")
            print(msg)

    seis.NP = 1
    seis.nmax_dev = 1


def test_fp16_grad(seis, ref=None, plot=False, ngpu=1, nmpi=1, inputres=0):
    
    seis = define_rec_src(seis)
    pars = {}
    pars['vp'] = np.zeros(seis.N) + 3500
    pars['vs'] = pars['vp'] * 0 + 2000
    pars['rho'] = pars['vp'] * 0 + 2000
    slices = [slice(d//2-5, d//2+5) for d in seis.N]
    slicesp = [slice(seis.nab, -seis.nab) for _ in seis.N]
    slicesp[1:-1] = [d//2 for d in seis.N[1:-1]]
    if seis.csts["freesurf"] == 1:
        slices[0] = slice(5, 15)
        slicesp[0] = slice(0, -seis.nab)
    slicesp = tuple(slicesp)
    slices = tuple(slices)
    if seis.L > 0:
        pars['taup'] = pars['vp'] * 0 + 0.1
        pars['taus'] = pars['vp'] * 0 + 0.1
    
    seis.FP16 = 0
    seis.nmax_dev = 1
    seis.NP = 1
    seis.file_din = seis.workdir + '/SeisCL_din.mat'

    pars['vp'] = np.zeros(seis.N) + 3500
    pars['vp'][slices] = 4000
    seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=False)
    seis.execute()
    data = seis.read_data()
    seis.write_data({"p": data[0]}, filename="SeisCL_din.mat")
    pars['vp'] = np.zeros(seis.N) + 3500

    if ngpu > 1 or nmpi > 1 or inputres == 1:
        seis.FP16 = 0
        seis.resout = 1
        seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=True)
        seis.execute()
        if inputres == 1:
            res = seis.read_data(residuals=True)
        else:
            res = None
        grad = seis.read_grad()
        ref = grad
        if plot:
            for g in grad:
                plt.imshow(g[slicesp], aspect='auto')
            plt.show()
        seis.nmax_dev = ngpu
        seis.NP = nmpi
        seis.resout = 0

    for fp16 in range(0, 4):
        print("    Testing FP16=%d....." % fp16, end='')
        seis.FP16 = fp16
        try:
            if inputres == 1:
                seis.inputres = 1
                seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=False)
                seis.execute()
                seis.set_backward(residuals=res)
                seis.execute()
                seis.inputres = 0
            else:
                seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=True)
                seis.execute()
            grad = seis.read_grad()
            if plot:
                for g in grad:
                    plt.imshow(g[slicesp], aspect='auto')
                plt.show()
            if ref is None:
                ref = grad
                print("passed")
            else:
                err = (np.sum([np.sum((g-r)**2) for g, r in zip(grad, ref)])
                       / np.sum([np.sum(r**2) for r in ref]))
                if err > 0.001:
                    if plot:
                        plt.imshow(grad[0][slicesp]-ref[0][slicesp],
                                   aspect='auto')
                        plt.show()
                    raise SeisCLError("    Error with referance too large: %e"
                                      % err)
                print("passed (error %e)" % err)
        except SeisCLError as msg:
            print("failed:")
            print(msg)

    seis.NP = 1
    seis.nmax_dev = 1


def define_rec_src(seis):

    """
        _________________________Sources and receivers__________________________
    """
    if seis.freesurf == 1:
        indz0 = 5
    else:
        indz0 = seis.nab+5

    ii = 0
    seis.rec_pos_all = np.empty((8, 0))
    seis.src_pos_all = np.empty((5, 0))
    
    toappend = np.zeros((5, 1))
    toappend[0, :] = (seis.N[-1]//2-15)*seis.dh
    toappend[1, :] = (seis.N[1]//2)*seis.dh
    toappend[2, :] = indz0 * seis.dh
    toappend[3, :] = ii
    toappend[4, :] = 100

    seis.src_pos_all = np.append(seis.src_pos_all, toappend, axis=1)
    for jj in range(0, seis.N[0]-seis.nab-indz0):
        toappend = np.zeros((8, 1))
        toappend[0, :] = (seis.N[-1]//2+15)*seis.dh
        toappend[1, :] = (seis.N[1]//2)*seis.dh
        toappend[2, :] = (indz0+jj)*seis.dh
        toappend[3, :] = ii
        toappend[4, :] = seis.rec_pos_all.shape[1]+1
        seis.rec_pos_all = np.append(seis.rec_pos_all, toappend, axis=1)

    seis.src_all = None
    return seis


if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",
                        type=str,
                        default='all',
                        help="Name of the test to run, default to all"
                        )
    parser.add_argument("--plot",
                        type=int,
                        default=0,
                        help="Plot the test results (1) or not (0). Default: 0."
                        )
    
    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()
    
    seis = SeisCL()
    seis.dh = 10
    seis.dt = 0.0008
    seis.NT = 875
    seis.FDORDER = 8
    seis.abs_type = 2
    seis.N = np.array([64, 64, 64])

    # ND, L, freesurf, abs_type, FDORDER, forward, NGPU, NPROC

    name = "2D_elastic_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "2D_elastic_grad"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        test_fp16_grad(seis, ref=None, plot=args.plot)

    name = "3D_elastic_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 64])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "3D_elastic_grad"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 64])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        test_fp16_grad(seis, ref=None, plot=args.plot)

    name = "2D_visco_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64])
        seis.L = 1
        seis.ND = 2
        seis.freesurf = 0
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "3D_visco_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 64])
        seis.ND = 3
        seis.L = 1
        seis.freesurf = 0
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "2D_elas_forward_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64])
        seis.ND = 2
        seis.L = 0
        seis.freesurf = 1
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "2D_visco_forward_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64])
        seis.ND = 2
        seis.L = 1
        seis.freesurf = 1
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "2D_elas_grad_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64])
        seis.ND = 2
        seis.L = 0
        seis.freesurf = 1
        test_fp16_grad(seis, ref=None, plot=args.plot)

    name = "3D_elas_forward_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 64])
        seis.ND = 3
        seis.L = 0
        seis.freesurf = 1
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "3D_visco_forward_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 64])
        seis.ND = 3
        seis.L = 1
        seis.freesurf = 1
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "3D_elas_grad_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 64])
        seis.ND = 3
        seis.L = 0
        seis.freesurf = 1
        test_fp16_grad(seis, ref=None, plot=args.plot)

    name = "2D_NGPU_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 256])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        test_fp16_forward(seis, ref=None, plot=args.plot, ngpu=3)

    name = "2D_NGPU_grad"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 256])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        test_fp16_grad(seis, ref=None, plot=args.plot, ngpu=3)

    name = "3D_NGPU_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 256])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        test_fp16_forward(seis, ref=None, plot=args.plot, ngpu=3)

    name = "3D_NGPU_grad"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 256])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        test_fp16_grad(seis, ref=None, plot=args.plot, ngpu=3)

    name = "2D_MPI_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 256])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        seis.MPI_NPROC_SHOT = 3
        test_fp16_forward(seis, ref=None, plot=args.plot, nmpi=3)
        seis.NP = 1
        seis.MPI_NPROC_SHOT = 1

    name = "2D_MPI_grad"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 256])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        seis.MPI_NPROC_SHOT = 4
        test_fp16_grad(seis, ref=None, plot=args.plot, nmpi=4)
        seis.NP = 1
        seis.MPI_NPROC_SHOT = 1

    name = "3D_MPI_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 256])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        seis.MPI_NPROC_SHOT = 3
        test_fp16_forward(seis, ref=None, plot=args.plot, nmpi=3)
        seis.NP = 1
        seis.MPI_NPROC_SHOT = 1

    name = "3D_MPI_grad"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 256])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        seis.MPI_NPROC_SHOT = 3
        test_fp16_grad(seis, ref=None, plot=args.plot, nmpi=3)
        seis.NP = 1
        seis.MPI_NPROC_SHOT = 1

    name = "3D_seisout"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 64])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        test_seisout(seis, plot=args.plot)

    name = "2D_NGPU_backpropagation"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 256])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        test_backpropagation(seis, plot=args.plot, ngpu=3)

    name = "2D_MPI_backpropagation"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 256])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        seis.MPI_NPROC_SHOT = 4
        test_backpropagation(seis, plot=args.plot, nmpi=3)
        seis.NP = 1
        seis.MPI_NPROC_SHOT = 1

    name = "3D_NGPU_backpropagation"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.NT = 300
        seis.N = np.array([48, 48, 144])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        test_backpropagation(seis, plot=args.plot, ngpu=3)
        seis.NT = 875

    name = "3D_MPI_backpropagation"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.NT = 300
        seis.N = np.array([48, 48, 144])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        seis.MPI_NPROC_SHOT = 3
        test_backpropagation(seis, plot=args.plot, nmpi=3)
        seis.NT = 875
        seis.NP = 1
        seis.MPI_NPROC_SHOT = 1

    name = "2D_inputres"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        test_fp16_grad(seis, ref=None, plot=args.plot, inputres=1)

    name = "3D_inputres"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 64])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        test_fp16_grad(seis, ref=None, plot=args.plot, inputres=1)

    name = "2D_NGPU_inputres"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 256])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        test_fp16_grad(seis, ref=None, plot=args.plot, ngpu=3, inputres=1)

    name = "2D_MPI_inputres"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 256])
        seis.L = 0
        seis.ND = 2
        seis.freesurf = 0
        seis.MPI_NPROC_SHOT = 4
        test_fp16_grad(seis, ref=None, plot=args.plot, nmpi=4, inputres=1)
        seis.NP = 1
        seis.MPI_NPROC_SHOT = 1

    name = "3D_NGPU_grad"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 256])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        test_fp16_grad(seis, ref=None, plot=args.plot, ngpu=3, inputres=1)

    name = "3D_MPI_inputres"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.N = np.array([64, 64, 256])
        seis.L = 0
        seis.ND = 3
        seis.freesurf = 0
        seis.MPI_NPROC_SHOT = 3
        test_fp16_grad(seis, ref=None, plot=args.plot, nmpi=3, inputres=1)
        seis.NP = 1
        seis.MPI_NPROC_SHOT = 1
