# SeisCL

SeisCL is a program to perform seismic modeling, full waveform inversion and reverse time migration.
Modeling is performed with finite-difference in the time-domain, and can be either 3D or 2D,
isotropic acoustic, elastic or viscoelastic.
SeisCL can be run either on GPUs with CUDA or OpenCL, or CPUs with OpenCL.
Model decomposition and shot parallelization allows using multiples GPUs/nodes.
Although written is C/CUDA/OpenCL, a python interface is provided for implementing the
inversion/imaging workflows.

Cite this publication if you use this software:

Fabien-Ouellet, G., Gloaguen, E., & Giroux, B. (2016). Time-domain seismic modeling in
viscoelastic media for full waveform inversion on heterogeneous computing platforms with
OpenCL. Computers & Geosciences. http://dx.doi.org/10.1016/j.cageo.2016.12.004

## Installation

You should clone this repository

    git clone https://github.com/gfabieno/SeisCL.git

Two options are provided to install the software.

#### a) Use Docker (easiest, but limited)

This method only works with Nvidia GPUs, and is restricted to a single node execution.
However, multi-GPU domain decomposition is supported.

You first need to install Docker Engine, following the instructions [here](https://docs.docker.com/install/).
Because SeisCL uses GPUs, you also need to install the [Nvidia docker](https://github.com/NVIDIA/nvidia-docker).
For the later to work, Nvidia drivers should be installed.
Then, when in SeisCL repository, build the docker image as follows:

    docker build -t seiscl:v0

You can then direclty use the python interface to launch SeisCL.

#### b) Compile from source (all features available)

To obtain all features (multi-node, CUDA and OpenCL, GPUs and CPUs), you need to compile from source.
There are several pre-requisites:
*   [HDF5](https://www.hdfgroup.org/about-us/), for input and ouput files. On Unix system,
the easiest way is to install with a package manager (yum, apt-get). On a cluster, it should be already available.
*  Option 1[CUDA](https://developer.nvidia.com/cuda-toolkit). Install CUDA if SeisCL will
be mainly used on NVidia GPUs (most likely case). It is faster than OpenCL and supports
SeisCL FP16 modes, which are more or less 2x faster than conventional mode.
*  Option 2[OpenCL](https://www.khronos.org/opencl/). This option is slightly slower on
Nvidia GPUs for large models, but faster for smalller models (because of faster JIT compilation).
It also supports Intel and AMD CPUs and GPUs. To install OpenCL on Linux, see this very good [tutorial](
https://wiki.tiker.net/OpenCLHowTo)
*  (Optional) An MPI Library, either [OpenMPI](https://www.open-mpi.org)
or [MPICH](https://www.mpich.org). If working on a cluster, this should already be installed!

A Makefile is provided, which may need to be modified for your specific system.
To compile, just run:

    cd src;
    make all
    echo "export PATH=$(pwd):\$PATH" >> ~/.bashrc
    source ~/.bashrc

The last line is needed because SeisCL_MPI must be on PATH for the Python interface.

Several options can be passed to the make file to compile different flavors of SeisCL:
*  api. Use api=cuda to build SeisCL with Cuda instead of OpenCL
* nompi Use nompi=1 to compile without MPI support
* H5LIB Use option to set the path to hdf5 libraries (hdf5.so)
* H5LIB Use this option to set the path to hdf5 headers
* H5CC Define as the desired compiler, even if the h5cc wrapper is found.

For example, to compile with cuda, without MPI support with gcc:

    make all api=cuda nompi=1 H5CC=gcc

## Testing

Several tests can be found in ./tests.


```
cd PATH_TO_SEISCL
mpirun -np N ./SeisCL_MPI INPUT_MODEL_FILE INPUT_DATA_FILE
```

For the moment, the working directory in terminal must be the directory of the SeisCL binary, as kernels are read and compiled at each program call.

See the Python caller located in tests for the structure of the HDF5 input files.

