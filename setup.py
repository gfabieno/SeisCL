
import os

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

ext_modules = []
cmdclass = {}

# SeisCL/torch's compiled extension (bindings.cpp) links against
# seiscl_core, a CUDA-only static library built separately via CMake
# (`cmake .. -DBUILD_TORCH_CORE=1`, see CMakeLists.txt). Building it here is
# opt-in via SEISCL_BUILD_TORCH=1 rather than tied to the `torch` extra
# alone, since it requires that CMake step to already have produced
# libseiscl_core.a and a CUDA toolkit to be present -- neither of which pip
# can arrange on its own.
if os.environ.get('SEISCL_BUILD_TORCH') == '1':
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    seiscl_core_dir = os.environ.get('SEISCL_CORE_DIR', 'build')
    # HDF5's install location varies a lot by system (same reason src/Makefile
    # exposes H5LIB/H5CC) -- point these at wherever `hdf5.h`/`libhdf5` live
    # if they aren't already on the compiler's default search path.
    hdf5_include_dir = os.environ.get('SEISCL_HDF5_INCLUDE_DIR')
    hdf5_lib_dir = os.environ.get('SEISCL_HDF5_LIB_DIR')
    include_dirs = ['src']
    if hdf5_include_dir:
        include_dirs.append(hdf5_include_dir)
    library_dirs = [seiscl_core_dir]
    if hdf5_lib_dir:
        library_dirs.append(hdf5_lib_dir)

    ext_modules = [
        CUDAExtension(
            name='SeisCL.torch._C',
            sources=['SeisCL/torch/bindings.cpp'],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['seiscl_core', 'hdf5', 'hdf5_hl', 'cuda', 'nvrtc'],
            # seiscl_core (CMakeLists.txt's BUILD_TORCH_CORE target) is
            # compiled with __NOMPI__, which changes the layout of the
            # `model` struct (F.h:446-448, an MPI_Comm field appears only
            # when __NOMPI__ is *not* defined). bindings.cpp must see the
            # exact same struct layout or field offsets silently mismatch.
            #
            # OMPI_SKIP_MPICXX: F.h's <hdf5.h> include pulls in <mpi.h>
            # unconditionally when HDF5 was built with parallel/MPI support
            # (true of most system HDF5 installs, e.g. Linux distros that
            # bundle HDF5 under an openmpi package). Under a C++ compiler
            # that in turn pulls in OpenMPI's legacy C++ bindings
            # (mpicxx.h), which fail to compile standalone on older OpenMPI
            # releases -- this is unrelated to __NOMPI__ (which only governs
            # SeisCL's own MPI usage) and is the standard OpenMPI-provided
            # escape hatch, safe regardless of OpenMPI version.
            define_macros=[('__NOMPI__', None), ('OMPI_SKIP_MPICXX', None)],
        )
    ]
    cmdclass = {'build_ext': BuildExtension}

setup(name='SeisCL',
      version='1.0',
      description='Interface to SeisCL, for seismic modeling and inversion',
      long_description=readme(),
      author='Gabriel Fabien-Ouellet',
      author_email='gabriel.fabien-ouellet@polymtl.ca',
      license='GNU General Public License v3.0',
      packages=['SeisCL', 'SeisCL.torch'],
      install_requires=['obspy',
                        'numpy',
                        'h5py',
                        'scipy'],
      # 'invert' pulls the stochastic L-BFGS used by tests/test_dft_inversion.py.
      # Not a hard dependency: nothing in the engine or the wrapper needs it,
      # and that test skips itself if it is absent.
      extras_require={'torch': ['torch'],
                      'invert': ['slbfgs']},
      ext_modules=ext_modules,
      cmdclass=cmdclass,
      zip_safe=False)
