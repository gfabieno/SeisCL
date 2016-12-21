# SeisCL

SeisCL is a program to perform (visco)elastic full waveform inversion in 3D and 2D (SV and SH).
As input, it takes model parameters and recorded data.
It can ouput modelled seismograms, a movie of particle velocities, and the gradient for different parametrizations.


Cite this publication if you use this software:

Fabien-Ouellet, G., Gloaguen, E., & Giroux, B. (2016). Time-domain seismic modeling in viscoelastic media for full waveform inversion on heterogeneous computing platforms with OpenCL. Computers & Geosciences. http://dx.doi.org/10.1016/j.cageo.2016.12.004


To compile the program, do:
```
cd ./SeisCL_MPI_3D/
make all
```
You may have to adapt the makefile to your system.
Tested systems are Mac OS X and Linux Suse.
Prerequisites are HDF5, OpenCL, and MPI libraries.
To install OpenCL on Linux, see this very good tutorial:
https://wiki.tiker.net/OpenCLHowTo


To launch the program on N machines, call:
```
cd PATH_TO_SEISCL
mpirun -np N ./SeisCL_MPI INPUT_MODEL_FILE INPUT_DATA_FILE
```

For the moment, the working directory in terminal must be the directory of the SeisCL binary, as kernels are read and compiled at each program call.

See the Python caller located in tests for the structure of the HDF5 input files.

