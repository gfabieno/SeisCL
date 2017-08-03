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
filenames['model']=file+"_model.mat"    #File containing the model parameters
filenames['csts']=file+"_csts.mat"      #File containing the simulation constants
filenames['din']=file+"_din.mat"       #File containing the recorded data
filenames['dout']=file+"_dout.mat"      #File containing the seismograms output
filenames['gout']=file+"_gout.mat"      #File containing the gradient ouput
filenames['rms']=file+"_rms.mat"        #File containing the rms ouput
filenames['movout']=file+"_movie.mat"   #File containing the movie ouput


#_____________________Simulation constants input file_______________________
csts={}
csts['NX']=50              #Grid size in X
csts['NY']=30                #Grid size in Y (set to 1 for 2D)
csts['NZ']=20              #Grid size in Z
csts['ND']=3                #Flag for dimension. 3: 3D, 2: 2D P-SV,  21: 2D SH
csts['dh']=10                #Grid spatial spacing
csts['dt']=0.0008           # Time step size
csts['NT']=875              #Number of time steps
csts['freesurf']=0          #Include a free surface at z=0: 0: no, 1: yes
csts['FDORDER']=8           #Order of the finite difference stencil. Values: 2,4,6,8,10,12
csts['MAXRELERROR']=1       #Set to 1
csts['L']=0                 #Number of attenuation mechanism (L=0 elastic)
csts['f0']=15               #Central frequency for which the relaxation mechanism are corrected to the righ velocity
csts['FL']=np.array(15)     #Array of frequencies in Hz of the attenuation mechanism

csts['src_pos']=np.empty((5,0)) #Position of each shots. 5xnumber of sources. [sx sy sz srcid src_type]. srcid is the source number (two src with same srcid are fired simulatneously) src_type: 1: Explosive, 2: Force in X, 3: Force in Y, 4:Force in Z
csts['rec_pos']=np.empty((8,0)) #Position of the receivers. 8xnumber of traces. [gx gy gz srcid recid Not_used Not_used Not_used]. srcid is the source number recid is the trace number in the record
csts['src']=np.empty((csts['NT'],0))            #Source signals. NTxnumber of sources

csts['abs_type']=1          #Absorbing boundary type: 1: CPML, 2: Absorbing layer of Cerjan
csts['VPPML']=3500          #Vp velocity near CPML boundary
csts['NPOWER']=2            #Exponent used in CMPL frame update, the larger the more damping
csts['FPML']=15              #Dominant frequency of the wavefield
csts['K_MAX_CPML']=2        #Coeffienc involved in CPML (may influence simulation stability)
csts['nab']=8              #Width in grid points of the absorbing layer
csts['abpc']=6              #Exponential decay of the absorbing layer of Cerjan et. al.
csts['pref_device_type']=2  #Type of processor used: 2: CPU, 4: GPU, 8: Accelerator
csts['nmax_dev']=9999       #Maximum number of devices that can be used
csts['no_use_GPUs']=np.empty( (1,0) )  #Array of device numbers that should not be used for computation
csts['MPI_NPROC_SHOT']=1    #Maximum number of MPI process (nodes) per shot involved in domain decomposition

csts['back_prop_type']=2    #Type of gradient calculation: 1: backpropagation (elastic only) 2: Discrete Fourier transform
csts['param_type']=0        #Type of parametrization: 0:(rho,vp,vs,taup,taus), 1:(rho, M, mu, taup, taus), 2:(rho, Ip, Is, taup, taus)
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
csts['seisout']=4           #Output seismograms 1:velocities, 2: pressure, 3: velocities and pressure, 4: velocities and stresses
csts['resout']=0            #Output residuals 1:yes, 0: no
csts['rmsout']=0            #Output rms value 1:yes, 0: no
csts['movout']=0            #Output movie 1:yes, 0: no
csts['restype']=0           #Type of costfunction 0: raw seismic trace cost function. No other available at the moment

h5mat.savemat(filenames['csts'], csts , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)


#_________________Model File__________________
model={}
model['vp']=np.zeros( (csts['NZ'],csts['NY'],csts['NX']))+3500  #Must contain the variables names of the chosen parametrization
model['vs']=np.zeros( (csts['NZ'],csts['NY'],csts['NX']))+2000
model['rho']=np.zeros( (csts['NZ'],csts['NY'],csts['NX']))+2000
model['taup']=np.zeros( (csts['NZ'],csts['NY'],csts['NX']))+0.02
model['taus']=np.zeros( (csts['NZ'],csts['NY'],csts['NX']))+0.02

h5mat.savemat(filenames['model'], model , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)



#________add src pos, src and rec_pos_______
tmin=-1.5/csts['f0']
t=np.zeros((csts['NT'],1))
t[:,0]=tmin+np.arange(0,csts['NT']*csts['dt'],csts['dt'] )
pf=math.pow(math.pi,2)*math.pow(csts['f0'],2)
ricker=np.multiply( (1.0-2.0*pf*np.power(t,2)), np.exp(-pf*np.power(t,2) )  )

srcXList = [20,30]  # just try out 4 point sources at constant depth
srcYList = [10,20]
for ii in srcXList:
    for ij in srcYList:
        print('source position '+str((ii,ij)))
        toappend=np.zeros((5,1))
        toappend[0,:]=(csts['nab']+ii)*csts['dh'] # x position
        toappend[1,:]=(csts['nab']+ij)*csts['dh'] # y position
        toappend[2,:]=(csts['nab']+5) # z position
        toappend[3,:]=ii*len(srcXList)+ij # source id
        toappend[4,:]=1 
        csts['src_pos']=np.append(csts['src_pos'], toappend, axis=1)
        csts['src']=np.append(csts['src'], ricker  , axis=1)
        for ji in range(10,40):
            for jj in range(10,20):
                toappend=np.zeros((8,1))
                toappend[0,:]=(csts['nab']+ji)*csts['dh'] # x position
                toappend[1,:]=(csts['nab']+jj)*csts['dh'] # y position
                toappend[2,:]= (csts['nab']+6)  #(csts['NZ']-csts['nab']-5)*csts['dh'] # z position
                toappend[3,:]=ii*len(srcXList)+ij # source id
                toappend[4,:]=csts['rec_pos'].shape[1]+1 # rec id
                csts['rec_pos']=np.append(csts['rec_pos'], toappend, axis=1)



#________________Launch simulation______________
model['vp'][20:40,10:20,5:10]= 3550
model['taup'][20:40,10:20,5:10]= 0.03
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
din['vx0']=dout['vxout']
din['vy0']=dout['vyout']
din['vz0']=dout['vzout']
h5mat.savemat(filenames['din'], din , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)


##________________Calculate gradient______________
#model['vp'][20:40,10:20,5:10]= 3500
#model['taup'][20:40,10:20,5:10]= 0.02
#csts['gradout']=1
#csts['resout']=1
#csts['gradfreqs']=np.append(csts['gradfreqs'], csts['f0'])
#h5mat.savemat(filenames['csts'], csts , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)
#h5mat.savemat(filenames['model'], model , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)
#              
#filepath=os.getcwd()
#cmdlaunch='cd ../src/; mpirun -np 1 ./SeisCL_MPI '+filepath+'/SeisCL > ../tests/out 2>../tests/err'
#print(cmdlaunch)
#pipes = subprocess.Popen(cmdlaunch,stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
#while (pipes.poll() is None):
#    time.sleep(1)
#sys.stdout.write('Gradient calculation completed \n')
#sys.stdout.flush()


