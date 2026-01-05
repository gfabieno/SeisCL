
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='SeisCL',
      version='1.0',
      description='Interface to SeisCL, for seismic modeling and inversion',
      long_description=readme(),
      author='Gabriel Fabien-Ouellet',
      author_email='gabriel.fabien-ouellet@polymtl.ca',
      license='GNU General Public License v3.0',
      packages=['SeisCL'],
      install_requires=['obspy',
                        'numpy',
                        'h5py',
                        'scipy'],
      zip_safe=False)
