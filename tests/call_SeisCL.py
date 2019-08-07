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
import math
import subprocess

file="SeisCL"
filenames={}
filenames['model']=file+"_model.mat"    #File containing the model pareters
filenames['csts']=file+"_csts.mat"      #File containing the simulation constants
filenames['din']=file+"_din.mat"        #File containing the recorded data
filenames['dout']=file+"_dout.mat"      #File containing the seismograms output
filenames['gout']=file+"_gout.mat"      #File containing the gradient ouput
filenames['rms']=file+"_rms.mat"        #File containing the rms ouput
filenames['movout']=file+"_movie.mat"   #File containing the movie ouput


#_____________________Simulation constants input file_______________________
csts={}
csts['N']=np.array([64,64]) #Grid size ( z,x)
csts['ND']=2                #Flag for dimension. 3: 3D, 2: 2D P-SV,  21: 2D SH
csts['dh']=10                #Grid spatial spacing
csts['dt']=0.0008           # Time step size
csts['NT']=875              #Number of time steps
csts['freesurf']=0          #Include a free surface at z=0: 0: no, 1: yes
csts['FDORDER']=8           #Order of the finite difference stencil. Values: 2,4,6,8,10,12
csts['MAXRELERROR']=1       #Set to 1
csts['L']=1                 #Number of attenuation mechanism (L=0 elastic)
csts['f0']=15               #Central frequency for which the relaxation mechanism are corrected to the righ velocity
csts['FL']=np.array(15)     #Array of frequencies in Hz of the attenuation mechanism

csts['src_pos']=np.empty((5,0)) #Position of each shots. 5xnumber of sources. [sx sy sz srcid src_type]. srcid is the source number (two src with same srcid are fired simulatneously) src_type: 1: Explosive, 2: Force in X, 3: Force in Y, 4:Force in Z
csts['rec_pos']=np.empty((8,0)) #Position of the receivers. 8xnumber of traces. [gx gy gz srcid recid Not_used Not_used Not_used]. srcid is the source number recid is the trace number in the record
csts['src']=np.empty((csts['NT'],0))            #Source signals. NTxnumber of sources

csts['abs_type']=2          #Absorbing boundary type: 1: CPML, 2: Absorbing layer of Cerjan
csts['VPPML']=3500          #Vp velocity near CPML boundary
csts['NPOWER']=2            #Exponent used in CMPL frame update, the larger the more damping
csts['FPML']=15              #Dominant frequency of the wavefield
csts['K_MAX_CPML']=2        #Coeffienc involved in CPML (may influence simulation stability)
csts['nab']=16              #Width in grid points of the absorbing layer
csts['abpc']=6              #Exponential decay of the absorbing layer of Cerjan et. al.
csts['pref_device_type']=4  #Type of processor used: 2: CPU, 4: GPU, 8: Accelerator
csts['nmax_dev']=9999       #Maximum number of devices that can be used
csts['no_use_GPUs']=np.empty( (1,0) )  #Array of device numbers that should not be used for computation
csts['MPI_NPROC_SHOT']=1    #Maximum number of MPI process (nodes) per shot involved in domain decomposition

csts['back_prop_type']=1    #Type of gradient calculation: 1: backpropagation (elastic only) 2: Discrete Fourier transform
csts['par_type']=0        #Type of paretrization: 0:(rho,vp,vs,taup,taus), 1:(rho, M, mu, taup, taus), 2:(rho, Ip, Is, taup, taus)
csts['gradfreqs']=np.empty((1,0)) #Array of frequencies in Hz to calculate the gradient with DFT
csts['tmax']=csts['NT']*csts['dt']#Maximum time for which the gradient is to be computed
csts['tmin']=0              #Minimum time for which the gradient is to be computed
csts['scalerms']=0          #Scale each modeled and recorded traces according to its rms value, then scale residual by recorded trace rms
csts['scalermsnorm']=0      #Scale each modeled and recorded traces according to its rms value, normalized
csts['scaleshot']=0         #Scale all of the traces in each shot by the shot total rms value
csts['fmin']=0              #Maximum frequency for the gradient computation
csts['fmax']=45              #Minimum frequency for the gradient computation
csts['mute']=None           #Muting matrix 5xnumber of traces. [t1 t2 t3 t4 flag] t1 to t4 are mute time with cosine tapers, flag 0: keep data in window, 1: mute data in window
csts['weight']=None         # NTxnumber of geophones or 1x number of geophones. Weight each sample, or trace, according to the value of weight for gradient calculation.

csts['gradout']=0           #Output gradient 1:yes, 0: no
csts['gradsrcout']=0        #Output source gradient 1:yes, 0: no
csts['seisout']=2           #Output seismograms 1:velocities, 2: pressure, 3: velocities and pressure, 4: velocities and stresses
csts['resout']=0            #Output residuals 1:yes, 0: no
csts['rmsout']=0            #Output rms value 1:yes, 0: no
csts['movout']=0            #Output movie 1:yes, 0: no
csts['restype']=0           #Type of costfunction 0: raw seismic trace cost function. No other available at the moment
csts['FP16']=2              #Use half precision 1: yes 0: no

h5mat.savemat(filenames['csts'], csts , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)


#_________________Model File__________________
model={}
model['vp']=np.zeros( (csts['N'][0],csts['N'][1]))+3500  #Must contain the variables names of the chosen paretrization
model['vs']=np.zeros( (csts['N'][0],csts['N'][1]))+2000
model['rho']=np.zeros( (csts['N'][0],csts['N'][1]))+2000
model['taup']=np.zeros( (csts['N'][0],csts['N'][1]))+0.02
model['taus']=np.zeros( (csts['N'][0],csts['N'][1]))+0.02

h5mat.savemat(filenames['model'], model , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)



#________add src pos, src and rec_pos_______
tmin=-1.5/csts['f0']
t=np.zeros((csts['NT'],1))
t[:,0]=tmin+np.arange(0,csts['NT']*csts['dt'],csts['dt'] )
pf=math.pow(math.pi,2)*math.pow(csts['f0'],2)
ricker=np.multiply( (1.0-2.0*pf*np.power(t,2)), np.exp(-pf*np.power(t,2) )  )

for ii in range(0,csts['N'][0]-2*csts['nab']-10,60):
    toappend=np.zeros((5,1))
    toappend[0,:]=(csts['nab']+5)*csts['dh']
    toappend[1,:]=0
    toappend[2,:]=(csts['nab']+5+ii)*csts['dh']
    toappend[3,:]=ii
    toappend[4,:]=100
    csts['src_pos']=np.append(csts['src_pos'], toappend, axis=1)
    csts['src']=np.append(csts['src'], ricker  , axis=1)
    for jj in range(0,csts['N'][0]-2*csts['nab']-10):
        toappend=np.zeros((8,1))
        toappend[0,:]=(csts['N'][1]-csts['nab']-5)*csts['dh']
        toappend[1,:]=0
        toappend[2,:]=(csts['nab']+5+jj)*csts['dh']
        toappend[3,:]=ii
        toappend[4,:]=csts['rec_pos'].shape[1]+1
        csts['rec_pos']=np.append(csts['rec_pos'], toappend, axis=1)



#________________Launch simulation______________
#model['vp'][70:90,65:85]= 3550
model['taup'][110:130,65:85]= 0.03
h5mat.savemat(filenames['csts'], csts , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)
h5mat.savemat(filenames['model'], model , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)

filepath=os.getcwd()
cmdlaunch='cd ../src/; mpirun -np 1 ./SeisCL_MPI '+filepath+'/SeisCL > ../tests/out 2>../tests/err'
print(cmdlaunch)
pipes = subprocess.Popen(cmdlaunch,stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
while (pipes.poll() is None):
    time.sleep(1)
sys.stdout.write('Forward calculation completed \n')
sys.stdout.flush()

dout = h5mat.loadmat(filenames['dout'])
din={}
din['src_pos']=dout['src_pos']
din['rec_pos']=dout['rec_pos']
#din['vx']=np.transpose(dout['vxout'])
#din['vz']=np.transpose(dout['vzout'])
din['p']=np.transpose(dout['pout'])
h5mat.savemat(filenames['din'], din , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)


#________________Calculate gradient______________
#model['vp'][70:90,65:85]= 3500
model['vp'][:,:]= 4500
model['taup'][110:130,65:85]= 0.02
csts['gradout']=1
csts['resout']=1
csts['gradfreqs']=np.append(csts['gradfreqs'], csts['f0'])
h5mat.savemat(filenames['csts'], csts , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)
h5mat.savemat(filenames['model'], model , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)

filepath=os.getcwd()
cmdlaunch='cd ../src/; mpirun -np 1 ./SeisCL_MPI '+filepath+'/SeisCL > ../tests/out 2>../tests/err'
print(cmdlaunch)
pipes = subprocess.Popen(cmdlaunch,stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
while (pipes.poll() is None):
    time.sleep(1)
sys.stdout.write('Gradient calculation completed \n')
sys.stdout.flush()


