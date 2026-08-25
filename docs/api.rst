API
===

SeisCL exposes two independent Python entry points to the same C/CUDA/OpenCL
engine:

* :ref:`api-seiscl-class` -- the general-purpose interface. Writes HDF5 files
  and drives the compiled ``SeisCL_MPI`` binary as a subprocess. Supports
  CUDA and OpenCL (GPU and CPU), MPI, and multi-GPU domain decomposition.
* :ref:`api-seiscl-torch` -- an in-memory, CUDA-only binding exposing forward
  modeling as a :class:`torch.autograd.Function`. No files, no subprocess.

The two do **not** share a parameter-naming convention; see the warning under
:func:`~SeisCL.torch.seiscl_forward` below.

.. _api-seiscl-class:

The ``SeisCL`` class
--------------------

The primary interface, used by every notebook in this documentation except
:doc:`notebooks/DeepLearning/SeisCLinPyTorch`.

Constructing ``SeisCL()`` locates the backend immediately: it looks for
``SeisCL_MPI`` (and ``mpirun``) on ``PATH``, else for a ``seiscl:v0`` Docker
image, and raises ``SeisCLError`` if neither is found. There is no
pip-installable backend -- the engine must be compiled first (see
:doc:`readme`).

.. autoclass:: SeisCL.SeisCL.SeisCL
   :members:

.. autoexception:: SeisCL.SeisCL.SeisCLError

.. _api-seiscl-torch:

``SeisCL.torch`` -- PyTorch binding
-----------------------------------

.. note::

   This subpackage is opt-in and requires both a CUDA GPU and the compiled
   extension (``cmake .. -DBUILD_TORCH_CORE=1``, then ``cmake --build .``,
   then ``SEISCL_BUILD_TORCH=1 SEISCL_CORE_DIR=build pip install
   -e .[torch]``). Importing ``SeisCL.SeisCL`` does not depend on it.

   Current limitations: CUDA only; host (CPU) tensors only; only
   ``BACK_PROP_TYPE = 1`` (boundary-storage gradient); no source-wavelet
   gradients; single process and single GPU group (no MPI).

See :doc:`notebooks/DeepLearning/SeisCLinPyTorch` for a worked example that
also cross-checks this binding against the ``SeisCL`` class.

.. py:function:: SeisCL.torch.seiscl_forward(cfg, params, src, src_pos, rec_pos, output_fields=None)

   Differentiable SeisCL forward modeling.

   :param cfg: Scalar run configuration -- see :ref:`api-torch-config`.
   :type cfg: SeisCL.torch.Config
   :param params: Flat ``(prod(cfg.N),)`` float32 CPU tensors holding the
      material parameters. Tensors with ``requires_grad=True`` receive
      gradients through ``backward()``.
   :type params: dict[str, torch.Tensor]
   :param src: Source wavelets, ``[allns, NT]``.
   :param src_pos: Source geometry, ``[allns, 5]`` --
      ``[sx, sy, sz, srcid, src_type]``.
   :param rec_pos: Receiver geometry, ``[allng, 8]`` --
      ``[gx, gy, gz, srcid, recid, -, -, -]``.
   :param output_fields: Which fields to record seismograms for, e.g.
      ``["vx", "vz"]``. Defaults to every field the modeling case declares.
      The seismogram-output kernel is generated per the specific requested
      combination, so pass only what you need.
   :type output_fields: list[str], optional
   :returns: Modeled seismograms at the receivers, ``[allng, NT]`` per field.
   :rtype: dict[str, torch.Tensor]

   .. warning::

      ``params`` keys are always the engine's *internal* parameter names --
      ``"M"``, ``"mu"``, ``"rho"`` (plus ``"taup"``/``"taus"`` if
      ``cfg.L > 0``) -- **regardless of** ``cfg.par_type``. ``par_type``
      changes how the values under ``"M"``/``"mu"`` are *interpreted*, not the
      dict keys: for ``par_type=0`` they are vp/vs in m/s, squared into Lamé
      parameters by the engine.

      This differs from the ``SeisCL`` class, whose ``params`` dict uses
      ``"vp"``/``"vs"``/``"rho"`` for the same ``par_type=0`` convention
      (matching the HDF5 dataset names). The *values* are the same; only the
      keys differ. Do not assume the two conventions match when porting code
      between the two APIs.

   .. warning::

      Array layout differs too. The ``SeisCL`` class accepts ``(NZ, NX)``
      numpy arrays and transposes them internally before writing to file.
      ``SeisCL.torch`` skips the file step, so the caller must supply
      parameters already flattened in the engine's internal ``(NX, NZ)``
      C-order layout -- i.e. ``arr_zx.T.ravel()``.

.. _api-torch-config:

``SeisCL.torch.Config``
~~~~~~~~~~~~~~~~~~~~~~~

A plain settings struct (exposed from C++ via pybind11) mirroring the subset
of the ``SeisCL`` class's constructor arguments needed to drive one forward or
gradient call. All attributes are read/write; defaults are shown below.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Attribute
     - Default
     - Meaning
   * - ``N``
     - *(unset)*
     - Grid size, ``[NZ, NX]`` in 2D or ``[NZ, NY, NX]`` in 3D. Required.
   * - ``ND``
     - ``2``
     - 3: 3D elastic, 2: 2D P-SV, 21: 2D SH, 22: 2D acoustic.
   * - ``dh``
     - ``1.0``
     - Grid spacing (m).
   * - ``dt``
     - ``1.0``
     - Time step (s).
   * - ``NT``
     - ``0``
     - Number of time steps. Required.
   * - ``FDORDER``
     - ``8``
     - Finite-difference order (2, 4, 6, 8, 10 or 12).
   * - ``MAXRELERROR``
     - ``0``
     - Stencil coefficient set: 0 for Taylor, 1 for Holberg.
   * - ``FREESURF``
     - ``0``
     - Free surface at the top boundary: 0 no, 1 yes.
   * - ``NAB``
     - ``16``
     - Width of the absorbing boundary, in grid cells.
   * - ``ABS_TYPE``
     - ``1``
     - 1: CPML, 2: Cerjan-style exponential taper.
   * - ``VPPML``
     - ``3500.0``
     - Vp used for the CPML coefficients (``ABS_TYPE=1``).
   * - ``FPML``
     - ``15.0``
     - CPML dominant frequency.
   * - ``NPOWER``
     - ``2.0``
     - CPML damping-profile exponent.
   * - ``K_MAX_CPML``
     - ``2.0``
     - CPML stretching coefficient.
   * - ``abpc``
     - ``4.0``
     - Attenuation at the edge of the Cerjan taper, in percent
       (``ABS_TYPE=2``).
   * - ``L``
     - ``0``
     - Number of Maxwell bodies for attenuation; 0 for purely elastic.
   * - ``f0``
     - ``15.0``
     - Central frequency of the attenuation mechanisms.
   * - ``par_type``
     - ``0``
     - How ``"M"``/``"mu"`` are interpreted: 0 for vp/vs, 1 for M/mu,
       2 for Ip/Is. See the warning above.
   * - ``FP16``
     - ``0``
     - 0 for float32; nonzero selects the packed-fp16 kernel variants.
   * - ``restype``
     - ``0``
     - Residual/misfit type used by the adjoint pass.
   * - ``GRADSRCOUT``
     - ``0``
     - Source-wavelet gradient. Not computed by this binding.
   * - ``HOUT``
     - ``0``
     - Output the approximate Hessian diagonal.
   * - ``BACK_PROP_TYPE``
     - ``1``
     - Gradient strategy. Only 1 (boundary storage) is supported here;
       requesting 2 (DFT) raises.
   * - ``nmax_dev``
     - ``1``
     - Maximum number of devices to use.
   * - ``pref_device_type``
     - ``4``
     - Preferred device type. Inert in this binding (meaningful only for
       OpenCL builds); kept for familiarity with the ``SeisCL`` class.

Other modules
-------------

.. autoclass:: SeisCL.Q_tau.QTAU
   :members:
