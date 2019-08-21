#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface to SeisCL
"""
import hdf5storage as h5mat
import h5py as h5
import numpy as np
import subprocess
import os
from obspy.core import Trace, Stream

class SeisCLError(Exception):
    pass

class SeisCL():
    """ A class that implements an interface to SeisCL
        (https://github.com/gfabieno/SeisCL.git)
    """
    def __init__(self):
        """
            Define variables that are needed for SeisCL input files
        """


        self.file = 'SeisCL'     #Filename for models and parameters (see setter)
        self.file_datalist = None     #File with a list of all data (see setter)
        self.progname = 'SeisCL_MPI'
        self.workdir = './seiscl'
        self.NP = 1

        #_____________________Simulation constants _______________________
        self.csts = {}
        self.csts['N'] = np.array([200,150])                 #Grid size [NZ, NX]
        self.csts['ND'] = 2   #Flag for dimension. 3: 3D, 2: 2D P-SV, 21: 2D SH,
                              #22: 2D acoustic
        self.csts['dh'] = 10                               #Grid spatial spacing
        self.csts['dt'] = 0.0008                                 #Time step size
        self.csts['NT'] = 875                              #Number of time steps
        self.csts['freesurf'] = 0          #Include a free surface 0: no, 1: yes
        self.csts['FDORDER'] = 8                 #Order of the finite difference
                                                         # Values: 2,4,6,8,10,12
        self.csts['MAXRELERROR'] = 1                     #Select FD coefficients
                                                         #(0: Taylor, 1:Holberg)
        self.csts['L'] = 0        #Number of attenuation mechanism (L=0 elastic)
        self.csts['f0'] = 15                #Central frequency of the relaxation
        self.csts['FL'] = np.array(15) #Frequencies of the attenuation mechanism

        # Position of each shots. Array [sx sy sz srcid src_type] x nb sources
        # srcid is the source number (same srcid are fired simulatneously)
        # src_type: 100: Explosive, 1: Force in X, 2: Force in Y, 3:Force in Z
        self.csts['src_pos'] = np.empty((5,0))
        # Position of the receivers. [gx gy gz srcid recid - - -] x nb receivers
        # srcid is the source number recid is the trace number in the record
        self.csts['rec_pos'] = np.empty((8,0))
        self.csts['src'] = np.empty((self.csts['NT'],0))         #Source signals.
                                                          # NTxnumber of sources
        
        self.csts['abs_type'] = 1                      #Absorbing boundary type:
                                         # 1: CPML, 2: Absorbing layer of Cerjan
        self.csts['VPPML'] = 3500                #Vp velocity near CPML boundary
        self.csts['NPOWER'] = 2              #Exponent used in CMPL frame update
        self.csts['FPML'] = 15              #Dominant frequency of the wavefield
        self.csts['K_MAX_CPML'] = 2                 # Coeffienc involved in CPML
                                          # (may influence simulation stability)
        self.csts['nab'] = 16       #Width in grid points of the absorbing layer
        self.csts['abpc'] = 6   #Exponential decay for absorbing layer of Cerjan
        self.csts['pref_device_type'] = 4               #Type of processor used:
                                                # 2: CPU, 4: GPU, 8: Accelerator
        self.csts['nmax_dev'] = 1                       # Maximum number of GPUs
        self.csts['no_use_GPUs'] = np.empty( (1,0) )    #Array of device numbers
                                       # that should not be used for computation
        self.csts['MPI_NPROC_SHOT'] = 1   #Maximum number of MPI process (nodes)
                                              # involved in domain decomposition
        
        self.csts['back_prop_type'] = 1           #Type of gradient calculation:
                                             # 1: backpropagation (elastic only)
                                                #  2: Discrete Fourier transform
        self.csts['param_type'] = 0                    #Type of parametrization:
                                                   # 0:(rho,vp,vs,taup,taus),
                                                   # 1:(rho, M, mu, taup, taus),
                                                   # 2:(rho, Ip, Is, taup, taus)
        self.csts['gradfreqs'] = np.empty((1,0)) #Frequencies of gradient with DFT
        self.csts['tmax'] = 0              #Maximum time of gradient computation
        self.csts['tmin'] = 0              #Minimum time of gradient computation
        self.csts['scalerms'] = 0          #Scale each modeled and recorded traces according to its rms value, then scale residual by recorded trace rms
        self.csts['scalermsnorm'] = 0      #Scale each modeled and recorded traces according to its rms value, normalized
        self.csts['scaleshot'] = 0         #Scale all of the traces in each shot by the shot total rms value
        self.csts['fmin'] = 0              #Maximum frequency of gradient
        self.csts['fmax'] = 0              #Minimum frequency of gradient
        self.csts['mute'] = None           #Muting matrix 5xnumber of traces. [t1 t2 t3 t4 flag] t1 to t4 are mute time with cosine tapers, flag 0: keep data in window, 1: mute data in window
        self.csts['weight'] = None         # NTxnumber of geophones or 1x number of geophones. Weight each sample, or trace, according to the value of weight for gradient calculation.
        
        self.csts['gradout'] = 0           #Output gradient 1:yes, 0: no
        self.csts['Hout'] = 0              #Output approximate Hessian 1:yes, 0: no
        self.csts['gradsrcout'] = 0        #Output source gradient 1:yes, 0: no
        self.csts['seisout'] = 2           #Output seismograms 1: output velocities, 2: output pressure, 3: output stresses, output everything
        self.csts['resout'] = 0            #Output residuals 1:yes, 0: no
        self.csts['rmsout'] = 0            #Output rms value 1:yes, 0: no
        self.csts['movout'] = 0            #Output movie every n frames
        self.csts['restype'] = 0           #Type of costfunction 0: raw seismic trace cost function. 1: Migration
        self.csts['inputres'] = 0          #Input the residuals for gradient computation

         
        # Variable for data maipulation, always equal to the self.csts member
        # (see setter)
        self.src_pos = np.empty((5,0))
        self.rec_pos = np.empty((8,0))
        self.src = np.empty((self.csts['NT'],0))
        
        #For data manipulation inside this class
        self.src_pos_all = np.empty((5,0))
        self.rec_pos_all = np.empty((8,0))
        self.allshotids = []
        self.srcids = np.empty((0,0))
        self.recids = np.empty((0,0))
        self.read_src = False

        self.mute = None
        self.mute_window = np.empty((4,0))
        self.mute_picks = np.empty((1,0))
        self.offmin = -float('Inf')
        self.offmax = float('Inf')

        self.wavelet_generator = self.ricker_wavelet

    #_____________________Setters _______________________
    
    #When setting a file for the datalist, load the datalist from it
    @property
    def workdir(self):
        return self.__workdir
    
    @workdir.setter
    def workdir(self, workdir):
        self.__workdir = workdir
        if not os.path.isdir(workdir):
            os.mkdir(workdir)

    #When setting a file for the datalist, load the datalist from it  
    @property
    def file_datalist(self):
        return self.__file_datalist

    @file_datalist.setter
    def file_datalist(self, file_datalist):
        self.__file_datalist = file_datalist
        if self.file_datalist:
            mat = h5.File(file_datalist, 'r')
            fields = {'src_pos': 'src_pos_all', 'rec_pos': 'rec_pos_all'}
            for word in fields.keys():
                data = mat[word]
                setattr(self, fields[word], np.transpose(data[:,:]) )
            self.allshotids = self.src_pos_all[3,:]
    
    #Params returns the list of parameters required by the simulation constants
    @property
    def params(self):
        if self.csts['param_type'] == 0:
            params = ['vp', 'vs', 'rho']
        elif self.csts['param_type']==1:
            params = ['M','mu','rho']
        elif self.csts['param_type']==2:
            params = ['Ip','Is','rho']
        else:
            raise NotImplementedError()
        if self.csts['L'] > 0:
            params.append('taup')
            params.append('taus')
        
        return params


    #Given the general filename, set the specific filenames of each files
    @property
    def file(self):
        return self.__file

    @file.setter
    def file(self, file):
        self.__file = file
        self.file_model = file+"_model.mat"
        self.file_csts = file+"_csts.mat"
        self.file_dout = file+"_dout.mat"
        self.file_gout = file+"_gout.mat"
        self.file_rms = file+"_rms.mat"
        self.file_movout = file+"_movie.mat"
        self.file_din = file+"_din.mat"
        self.file_res = file + "_res.mat"

    #The variable src_pos, rec_pos and src must always be reflected 
    #in self.csts to be written
    @property
    def src_pos(self):
        return self.csts['src_pos']

    @src_pos.setter
    def src_pos(self, src_pos):
        if not type(src_pos) is np.ndarray:
            raise TypeError('src_pos must be a numpy arrays')
        if src_pos.shape[0] != 5:
            raise TypeError('src_pos must be a numpy arrays with dim 5x num of src')
        self.csts['src_pos'] = src_pos
        
    @property
    def rec_pos(self):
        return self.csts['rec_pos']

    @rec_pos.setter
    def rec_pos(self, rec_pos):
        if not type(rec_pos) is np.ndarray:
            raise TypeError('rec_pos must be a numpy arrays')
        if rec_pos.shape[0] != 8:
            raise TypeError('rec_pos must be a numpy arrays with dim 8x num of rec')
        self.csts['rec_pos'] = rec_pos
        
    @property
    def src(self):
        return self.csts['src']

    @src.setter
    def src(self, src):
        if not type(src) is np.ndarray:
            raise TypeError('src must be a numpy arrays')
        if src.shape[0] != self.csts['NT']:
            raise TypeError('src must be a numpy arrays with dim NT x num of src')
        self.csts['src'] = src

    @property
    def to_load_names(self):
        toload = []
        if self.csts['seisout'] == 1:
            if self.csts['ND'] == 2:
                toload = ["vx", "vz"]
            if self.csts['ND'] == 21:
                toload = ["vy"]
            if self.csts['ND'] == 3:
                toload = ["vx", "vy", "vz"]
        if self.csts['seisout'] == 2:
            toload = ["p"]
        if self.csts['seisout'] == 3:
            if self.csts['ND'] == 2:
                toload = ["sxx", "szz", "sxz"]
            if self.csts['ND'] == 21:
                toload = ["sxy", "syz"]
            if self.csts['ND'] == 3:
                toload = ["sxx", "syy", "szz", "sxz", "sxy", "syz"]
        if self.csts['seisout'] == 4:
            if self.csts['ND'] == 2:
                toload = ["vx", "vz","sxx", "szz", "sxz"]
            if self.csts['ND'] == 21:
                toload = ["vy", "sxy", "syz"]
            if self.csts['ND'] == 3:
                toload = ["vx", "vy", "vz",
                          "sxx", "syy", "szz", "sxz", "sxy", "syz"]

        return toload

            
    def set_forward(self, jobids, params, workdir=None, withgrad=True):
        """
        Set up files to launch SeisCL on a selected number of shots

        @params:
        jobids (list): Source ids to compute
        params (dict): A dictionary containing material parameters
        workdir (str): The directory in which to write SeisCL files
        withgrad (bool): If true, SeisCL will output the gradient

        @returns:

        """
        if workdir is None:
            workdir = self.workdir
        
        self.srcids = np.empty((0, 0), 'int')
        self.recids = np.empty((0, 0), 'int')
        if len(np.atleast_1d(jobids)) > 1:
            shots = np.sort(jobids)
        else:
            shots = np.atleast_1d(jobids)
        for shot in shots:
            src_to_append = np.where(self.src_pos_all[3, :].astype(int) == shot)
            self.srcids = np.append(self.srcids, src_to_append)
            rec_to_append = np.where(self.rec_pos_all[3, :].astype(int) == shot)
            self.recids = np.append(self.recids, rec_to_append)
        if len(self.srcids) <= 0:
            raise ValueError('No shot found')
        self.src_pos = self.src_pos_all[:, self.srcids]
        self.rec_pos = self.rec_pos_all[:, self.recids]
        self.prepare_data()
        if withgrad:
            self.csts['gradout'] = 1
        else:
            self.csts['gradout'] = 0
        self.write_csts(workdir)
        self.write_model(params, workdir)

    def set_backward(self, residuals=None, workdir=None):
        """
        Set up files to launch SeisCL when inputing residuals for gradient
        computation

        @params:
        workdir (str): The directory in which to write SeisCL files
        residuals (list): A list of residuals for each seismic variable

        @returns:

        """
        
        if workdir is None:
            workdir = self.workdir
        if residuals is not None:
            data = {}
            for n, word in enumerate(self.to_load_names):
               data[word+"res"] = residuals[n]
            h5mat.savemat(workdir+self.file_res,
                          data,
                          appendmat=False,
                          format='7.3',
                          store_python_metadata=True,
                          truncate_existing=True)
            self.csts['inputres'] = 1
        else:
            self.csts['inputres'] = 0
        self.csts['gradout'] = 1
        self.write_csts(workdir)

    def callcmd(self, workdir=None):
        """
        Defines the command to launch SeisCL

        @params:
        workdir (str): The directory in which to write SeisCL files

        @returns:
        cmd (str): A string containing the command
        """
        if workdir is None:
            workdir = self.workdir
        cmd = 'mpirun -np ' + str(self.NP) + ' ' + self.progname \
              + ' '+workdir+'/'+self.file+' '+workdir+'/'+self.file_din
        return cmd

    def execute(self, workdir=None):
        """
        Launch SeisCL, a wait for it to return

        @params:
        workdir (str): The directory in which to write SeisCL files

        @returns:
        """
        if workdir is None:
            workdir = self.workdir
        pipes = subprocess.Popen(self.callcmd(workdir),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, shell=True)
        stdout, stderr = pipes.communicate()
        if stderr:
            raise SeisCLError(stderr.decode())

    def read_data(self, workdir=None):
        """
        Read the seismogram output by SeisCL

        @params:
        workdir (str): The directory in which to write SeisCL files

        @returns:
        output (list): A list containing each outputted variable
        """
        if workdir is None:
            workdir = self.workdir
        try:
            mat = h5.File(workdir + "/" + self.file_dout, 'r')
        except OSError:
            raise SeisCLError('Could not read data')
        output = []
        for word in self.to_load_names:
            if word+"out" in mat:
                datah5 = mat[word+"out"]
                data = np.transpose(datah5)
                output.append(data)  
                
        if not output:
            raise SeisCLError('Could not read data: variables not found')
            
        return output
    
    def read_grad(self, workdir=None, param_names=None):
        """
        Read the gradient output by SeisCL

        @params:
        workdir (str): The directory in which to write SeisCL files
        param_names (list) List the variable names to read the gradient

        @returns:
        output (list): A list containing each gradient
        """
        if workdir is None:
            workdir = self.workdir
        if param_names is None:
            param_names = self.params
        toread = ['grad'+name for name in param_names]
        try:
            mat = h5.File(workdir + "/" + self.file_gout, 'r')
            output = [np.transpose(mat[v]) for v in toread]
        except OSError:
            raise SeisCLError('Could not read grad')

        return output 
    
    def read_Hessian(self,  workdir=None, param_names=None):
        """
        Read the approximate hessian output by SeisCL

        @params:
        workdir (str): The directory in which to write SeisCL files
        param_names (list) List the variable names to read the gradient

        @returns:
        output (list): A list containing each gradient
        """
        if workdir is None:
            workdir = self.workdir
        if param_names is None:
            param_names = self.params
        toread = ['H' + name for name in param_names]
        try:
            mat = h5.File(workdir + "/" + self.file_gout, 'r')
            output = [np.transpose(mat[v]) for v in toread]
        except OSError:
            raise SeisCLError('Could not read Hessian')

        return output

    def read_rms(self, workdir=None):
        """
        Read the rms value output by SeisCL

        @params:
        workdir (str): The directory in which to write SeisCL files

        @returns:
        rms (float): The normalized rms value
        rms_norm (float) :Normalization factor
        """
        if workdir is None:
            workdir = self.workdir
        try:
            mat = h5.File(workdir + "/" + self.file_rms, 'r')
        except OSError:
            raise SeisCLError('Forward modeling failed, could not read rms\n')
            
        return mat['rms'][0]/mat['rms_norm'][0], mat['rms_norm'][0]
            
    def write_data(self, data, workdir=None, filename=None):
        """
        Write the data file for SeisCL

        @params:
        data (dict): A dictionary containing the variables names and the data

        @returns:

        """
        if workdir is None:
            workdir = self.workdir
        if filename is None:
            filename = self.file_din
        if 'src_pos' not in data:
            data['src_pos'] = self.src_pos
        if 'rec_pos' not in data:
            data['rec_pos'] = self.rec_pos
        if 'src' not in data:
            data['src'] = self.src
        h5mat.savemat(os.path.join(workdir, filename),
                      data,
                      appendmat=False,
                      format='7.3',
                      store_python_metadata=True,
                      truncate_existing=True)

    def read_csts(self, workdir=None):
        """
        Read the constants from the constants file

        @params:
        workdir (str): The directory in which to write SeisCL files

        @returns:

        """
        if workdir is None:
            workdir = self.workdir
        try:
           mat = h5mat.loadmat(os.path.join(workdir, self.file_csts),
                               variable_names=[param for param in self.csts])
           for word in mat:
               if word in self.csts:
                   self.csts[word] = mat[word]
        except (h5mat.lowlevel.CantReadError, NotImplementedError):
           raise SeisCLError('could not read parameter file \n')

    def write_csts(self, workdir=None):
        """
        Write the constants to the constants file

        @params:
        workdir (str): The directory in which to write SeisCL files

        @returns:

        """
        if workdir is None:
            workdir = self.workdir
        h5mat.savemat(os.path.join(workdir, self.file_csts),
                      self.csts,
                      appendmat=False,
                      format='7.3',
                      store_python_metadata=True,
                      truncate_existing=True)

    def read_srcs(self, workdir=None):
        """
        Read the source signal for SeisCL file

        @params:
        workdir (str): The directory in which to write SeisCL files

        @returns:

        """
        if workdir is None:
            workdir = self.workdir
        try:
            mat = h5.File(os.path.join(workdir, self.file_datalist), 'r')
            data = mat['src']
            setattr(self, 'src', np.transpose(data[self.srcids,:]))
        except :
            raise SeisCLError('could not read src\n')

    def write_model(self, params, workdir=None):
        """
        Write model parameters to the model files

        @params:
        workdir (str): The directory in which to write SeisCL files
        param_names (dict) Dictionary with parameters name and their value

        @returns:

        """
        if workdir is None:
            workdir = self.workdir
        for param in self.params:
            if param not in params:
                raise SeisCLError('Parameter with %s not defined\n' % param)
        h5mat.savemat(os.path.join(workdir, self.file_model),
                      params,
                      appendmat=False,
                      format='7.3',
                      store_python_metadata=True,
                      truncate_existing=True)
        
    def prepare_data(self):
        """
        Prepares receivers and sources arrays to launch computations

        @params:

        @returns:

        """
        validrec=[]
        for ii in range(0, self.rec_pos.shape[1] ):
            srcid = np.where(self.src_pos[3,:] == self.rec_pos[3, ii])
            offset = np.sqrt(np.square(self.rec_pos[0, ii]-self.src_pos[0, srcid])
                             +np.square(self.rec_pos[1, ii]-self.src_pos[1, srcid])
                             +np.square(self.rec_pos[2, ii]-self.src_pos[2, srcid])
                             )
            if offset <= self.offmax and offset >= self.offmin:
                validrec.append(ii)
        self.rec_pos = self.rec_pos[:, validrec]
        self.recids = self.recids[validrec]
        self.rec_pos[4, :] = [x+1 for x in self.recids]

        if self.read_src:
            self.read_srcs()
        else:
            self.fill_src()
      
        if np.any(self.mute_window):
            self.mute = np.transpose(np.tile(self.mute_window,
                                             (self.rec_pos.shape[1], 1)
                                             ))
            if np.any(self.mute_picks):
                for ii in range(0, self.recids.size):
                    self.mute[:3, ii] = self.mute[:3, ii] + self.mute_picks[self.recids[ii]]

    def fill_src(self):
        """
        Fill the source array with a signal generated by self.wavelet_generator

        @params:

        @returns:

        """
        src = self.wavelet_generator()

        nstot = self.src_pos.shape[1]
        self.src = np.stack([src] * nstot, 1)

    def ricker_wavelet(self):
        tmin = -1.5 / self.csts['f0']
        t = np.linspace(tmin,
                        self.csts['NT'] * self.csts['dt'] + tmin,
                        num=self.csts['NT'])

        ricker = ((1.0 - 2.0 * (np.pi ** 2) * (self.csts['f0'] ** 2) * (t ** 2))
                  * np.exp(-(np.pi ** 2) * (self.csts['f0'] ** 2) * (t ** 2)))

        return ricker

    def save_segy(self, data, name):

        dt = self.csts["dt"]
        out = Stream(Trace(data[:,ii], header=dict(delta=dt)) for ii in range(data.shape[1]))
        out.write(name, format='SEGY', data_encoding=5)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    tmax = 0.5
    seis = SeisCL()
    seis.csts['dh'] = 2.5
    seis.csts['dt'] = 2e-4
    seis.csts['NT'] = tmax // seis.csts['dt']

    seis.csts['f0'] = 40
    seis.csts['freesurf'] = 0
    seis.csts['FDORDER'] = 4
    seis.csts['abs_type'] = 1
    seis.csts['seisout'] = 2
    
    vp = np.zeros([200,200]) + 3500
    vs = vp * 0 + 2000
    rho = vp * 0 + 2000


    """
    _________________________Sources and receivers______________________________
    """
    seis.csts['N'] = np.array(vp.shape)

    sx = np.array([seis.csts['N'][1]/2]) * seis.csts['dh']
    sz = np.array([seis.csts['N'][0]/2]) * seis.csts['dh']
    sid = sx*0
    seis.src_pos = np.stack([sx, sx * 0, sz, sid, sx * 0 + 100], axis=0)
    seis.src_pos_all = seis.src_pos
    seis.fill_src()
    
    l1 = (seis.csts['nab'] + 1) * seis.csts['dh']
    l2 = (seis.csts['N'][1] - seis.csts['nab']) * seis.csts['dh']
    gx = np.arange(l1, l2, seis.csts['dh'])
    gsid = np.concatenate([s + gx * 0 for s in sid], axis=0)
    gz = gx * 0 + (seis.csts['N'][0]//2) * seis.csts['dh']
    gid = np.arange(0, len(gx))
    seis.rec_pos = np.stack(
                            [gx, gx * 0, gz, gsid, gid, gx * 0 + 2, gx * 0, gx * 0],
                            axis=0)
    seis.rec_pos_all = seis.rec_pos

    seis.set_forward(seis.src_pos_all[3,:], {"vp": vp, "rho": rho, "vs": vs},
                     withgrad=False)
    seis.execute()
    data = seis.read_data()[0]
    clip = 0.01
    vmin = np.min(data) * clip
    vmax = -vmin
    plt.imshow(data, aspect='auto', vmin=vmin, vmax=vmax)
    plt.show()

