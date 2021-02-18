.. SeisCL documentation master file, created by
   sphinx-quickstart on Wed Feb 17 08:52:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. include:: readme.rst


.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   Installation <self>
   notebooks/1_SimpleExample

.. toctree::
   :maxdepth: 1
   :caption: Forward Modeling:

   notebooks/ForwardModeling/1_SourcesReceivers
   notebooks/ForwardModeling/2_AbsorbingBoundaries
   notebooks/ForwardModeling/3_FreeSurface
   notebooks/ForwardModeling/4_ChosingWaveEquation
   notebooks/ForwardModeling/5_Attenuation
   notebooks/ForwardModeling/6_Parallelization
   notebooks/ForwardModeling/7_CreatingMovie
   notebooks/ForwardModeling/8_FiniteDifferenceStencil
   notebooks/ForwardModeling/9_StabilityCriteria
   notebooks/ForwardModeling/10_ReadingWritingSEGY
   notebooks/ForwardModeling/11_FileFormat

.. toctree::
   :maxdepth: 1
   :caption: Accuracy tests:

   notebooks/Accuracy/AnalyticalSolutions
   notebooks/Accuracy/DotProductTests

.. toctree::
   :maxdepth: 1
   :caption: Inversion:

   notebooks/Inversion/ComputingGradient
   notebooks/Inversion/UsingTensorflow
   notebooks/Inversion/UsingScipy
   notebooks/Inversion/source_inversion

.. toctree::
   :maxdepth: 1
   :caption: Imaging:

   notebooks/Imaging/ReverseTimeMigration

.. toctree::
   :maxdepth: 1
   :caption: Deep Learning:

   notebooks/DeepLearning/SeisCLinTensorflow

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   gallery/index.rst

.. toctree::
   :maxdepth: 1
   :caption: API:

   api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
