#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:35:05 2016

@author: gabrielfabien-ouellet
"""

import re, sys, os, inspect
import hdf5storage as h5mat
import h5py as h5
import numpy as np
import time
from shutil import copyfile
import sys
import matplotlib.pyplot as plt
import argparse
sys.path.append('../')
from python.SeisCL import SeisCL, SeisCLError



def test_seisout(seis, plot=False):
    
    seis = define_rec_src(seis)
    pars = {}
    pars['vp'] = np.zeros(seis.csts['N']) + 3500
    pars['vs'] = pars['vp'] * 0 + 2000
    pars['rho'] = pars['vp'] * 0 + 2000
    if seis.csts['L'] > 0:
        pars['taup'] = pars['vp'] * 0 + 0.1
        pars['taus'] = pars['vp'] * 0 + 0.1
    
    for seisout in range(1,5):
        ref = None
        print("    Testing seisout=%d" %seisout)
        for fp16 in range(0,2):
            print("        Testing fp16=%d....." %fp16 , end = '')
            seis.csts['FP16'] = fp16
            seis.csts['seisout'] = seisout
            try :
                seis.set_forward(seis.src_pos_all[3,:], pars, withgrad=False)
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
                    err = (np.sum([np.sum((d-r)**2) for d,r in zip(data, ref)])
                           /np.sum([np.sum((r)**2) for r in ref]))
                    if err > 0.001:
                        if plot:
                            plt.imshow(data[0]-ref[0], aspect='auto')
                            plt.show()
                        raise SeisCLError("    Error with data referance too large: %e"
                                          % err)
                    print("passed (error %e)" % err)
            except(SeisCLError) as msg:
                print("failed:")
                print(msg)

def test_fp16_forward(seis, ref=None, plot=False, ngpu=1):
    
    seis = define_rec_src(seis)
    pars = {}
    pars['vp'] = np.zeros(seis.csts['N']) + 3500
    pars['vs'] = pars['vp'] * 0 + 2000
    pars['rho'] = pars['vp'] * 0 + 2000
    if seis.csts['L'] > 0:
        pars['taup'] = pars['vp'] * 0 + 0.1
        pars['taus'] = pars['vp'] * 0 + 0.1
    
    if ngpu > 1:
        seis.csts['FP16'] = 0
        seis.csts['nmax_dev'] = 1
        seis.set_forward(seis.src_pos_all[3,:], pars, withgrad=False)
        seis.execute()
        data = seis.read_data()
        ref = data
        seis.csts['nmax_dev'] = ngpu


    for fp16 in range(0,4):
        print("    Testing FP16=%d....." %fp16 , end = '')
        seis.csts['FP16'] = fp16
        try :
            seis.set_forward(seis.src_pos_all[3,:], pars, withgrad=False)
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
                err = (np.sum([np.sum((d-r)**2) for d,r in zip(data, ref)])
                       /np.sum([np.sum((r)**2) for r in ref]))
                if err > 0.001:
                    if plot:
                        plt.imshow(data[0]-ref[0], aspect='auto')
                        plt.show()
                    raise SeisCLError("    Error with data referance too large: %e"
                                      % err)
                print("passed (error %e)" % err)
        except(SeisCLError) as msg:
            print("failed:")
            print(msg)

    seis.csts['nmax_dev'] = 1

def test_fp16_grad(seis, ref=None, plot=False):
    
    seis = define_rec_src(seis)
    pars = {}
    pars['vp'] = np.zeros(seis.csts['N']) + 3500
    pars['vs'] = pars['vp'] * 0 + 2000
    pars['rho'] = pars['vp'] * 0 + 2000
    slices = [slice(d//2-5,d//2+5) for d in seis.csts['N']]
    slicesp = [slice(seis.csts['nab'], -seis.csts['nab']) for _ in seis.csts['N']]
    slicesp[1:-1] = [d//2 for d in seis.csts['N'][1:-1]]
    if seis.csts["freesurf"] ==1:
        slices[0] = slice(5, 15)
        slicesp[0] = slice(0, -seis.csts['nab'])
    slicesp = tuple(slicesp)
    slices = tuple(slices)
    if seis.csts['L'] > 0:
        pars['taup'] = pars['vp'] * 0 + 0.1
        pars['taus'] = pars['vp'] * 0 + 0.1
    
    for fp16 in range(0,4):
        print("    Testing FP16=%d....." %fp16 , end = '')
        seis.csts['FP16'] = fp16
        try :
            if fp16==0:
                pars['vp'] = np.zeros(seis.csts['N']) + 3500
                pars['vp'][slices]= 4000
                seis.set_forward(seis.src_pos_all[3,:], pars, withgrad=False)
                seis.execute()
                data = seis.read_data()
                seis.write_data({"p":data[0]})
            pars['vp'] = np.zeros(seis.csts['N']) + 3500
            seis.set_forward(seis.src_pos_all[3,:], pars, withgrad=True)
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
                err = (np.sum([np.sum((g-r)**2) for g,r in zip(grad, ref)])
                       /np.sum([np.sum((r)**2) for r in ref]))
                if err > 0.01:
                    if plot:
                        plt.imshow(grad[0][slicesp]-ref[0][slicesp], aspect='auto')
                        plt.show()
                    raise SeisCLError("    Error with data referance too large: %e"
                                      % err)
                print("passed (error %e)" % err)
        except(SeisCLError) as msg:
            print("failed:")
            print(msg)

def define_rec_src(seis):

    """
        _________________________Sources and receivers__________________________
    """
    if seis.csts['freesurf'] == 1:
        indz0 = 5
    else:
        indz0 = seis.csts['nab']+5

    ii=0
    seis.rec_pos = np.empty((8,0))
    seis.src_pos = np.empty((5,0))
    
    toappend=np.zeros((5,1))
    toappend[0,:]=(seis.csts['N'][-1]//2-15)*seis.csts['dh'] #(seis.csts['nab']+5)*seis.csts['dh']
    toappend[1,:]=(seis.csts['N'][1]//2)*seis.csts['dh']
    toappend[2,:]=(indz0)*seis.csts['dh']
    toappend[3,:]=ii
    toappend[4,:]=100

    seis.src_pos=np.append(seis.src_pos, toappend, axis=1)
    for jj in range(0,seis.csts['N'][0]-seis.csts['nab']-indz0):
        toappend=np.zeros((8,1))
        toappend[0,:]=(seis.csts['N'][-1]//2+15)*seis.csts['dh']#(seis.csts['N'][1]-seis.csts['nab']-5)*seis.csts['dh']
        toappend[1,:]=(seis.csts['N'][1]//2)*seis.csts['dh']
        toappend[2,:]=(indz0+jj)*seis.csts['dh']
        toappend[3,:]=ii
        toappend[4,:]=seis.csts['rec_pos'].shape[1]+1
        seis.rec_pos=np.append(seis.rec_pos, toappend, axis=1)
    seis.fill_src()
    seis.rec_pos_all = seis.rec_pos
    seis.src_pos_all = seis.src_pos

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
    seis.csts['dh'] = 10
    seis.csts['dt'] = 0.0008
    seis.csts['NT'] = 875
    seis.csts['FDORDER']=8
    seis.csts['abs_type'] = 2
    seis.csts['N']=np.array([64,64,64])

    # ND, L, freesurf, abs_type, FDORDER, forward, NGPU, NPROC




    name = "2D_elastic_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64])
        seis.csts['L'] = 0
        seis.csts['ND'] = 2
        seis.csts['freesurf'] = 0
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "2D_elastic_grad"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64])
        seis.csts['L'] = 0
        seis.csts['ND'] = 2
        seis.csts['freesurf'] = 0
        test_fp16_grad(seis, ref=None, plot=args.plot)

    name = "3D_elastic_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64,64])
        seis.csts['L'] = 0
        seis.csts['ND'] = 3
        seis.csts['freesurf'] = 0
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "3D_elastic_grad"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64,64])
        seis.csts['L'] = 0
        seis.csts['ND'] = 3
        seis.csts['freesurf'] = 0
        test_fp16_grad(seis, ref=None, plot=args.plot)

    name = "2D_visco_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64])
        seis.csts['L'] = 1
        seis.csts['ND'] = 2
        seis.csts['freesurf'] = 0
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "3D_visco_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64,64])
        seis.csts['ND'] = 3
        seis.csts['L'] = 1
        seis.csts['freesurf'] = 0
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "2D_elas_forward_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64])
        seis.csts['ND'] = 2
        seis.csts['L'] = 0
        seis.csts['freesurf'] = 1
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "2D_elas_grad_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64])
        seis.csts['ND'] = 2
        seis.csts['L'] = 0
        seis.csts['freesurf'] = 1
        test_fp16_grad(seis, ref=None, plot=args.plot)

    name = "3D_elas_forward_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64,64])
        seis.csts['ND'] = 3
        seis.csts['L'] = 0
        seis.csts['freesurf'] = 1
        test_fp16_forward(seis, ref=None, plot=args.plot)

    name = "3D_elas_grad_surface"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64,64])
        seis.csts['ND'] = 3
        seis.csts['L'] = 0
        seis.csts['freesurf'] = 1
        test_fp16_grad(seis, ref=None, plot=args.plot)

    name = "2D_NGPU_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,256])
        seis.csts['L'] = 0
        seis.csts['ND'] = 2
        seis.csts['freesurf'] = 0
        test_fp16_forward(seis, ref=None, plot=args.plot, ngpu=3)

    name = "3D_NGPU_forward"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64,256])
        seis.csts['L'] = 0
        seis.csts['ND'] = 3
        seis.csts['freesurf'] = 0
        test_fp16_forward(seis, ref=None, plot=args.plot, ngpu=3)

    name = "3D_seisout"
    if args.test == name or args.test == "all":
        print("Testing %s" % name)
        seis.csts['N'] = np.array([64,64,64])
        seis.csts['L'] = 0
        seis.csts['ND'] = 3
        seis.csts['freesurf'] = 0
        test_seisout(seis, plot=args.plot)


