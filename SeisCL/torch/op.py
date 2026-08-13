"""torch.autograd.Function wrapper around SeisCL's in-memory CUDA binding
(the compiled extension in SeisCL/torch/bindings.cpp).

forward() and backward() reuse the engine's existing INPUTRES=1 two-call
protocol (src/time_stepping.c:850-859): forward() writes a small HDF5
checkpoint (boundary-wavefield shell, not the full model) only when at
least one parameter requires grad; backward() reads it back to compute the
adjoint gradient directly from grad_output, without recomputing the
forward pass. The checkpoint file is a private temp file, created and
deleted automatically -- callers never see it.

This binding is CUDA-only and does not go through SeisCL.py / HDF5 model
files / subprocess -- see the project plan for why (torch GPU tensors are
CUDA device pointers, with no OpenCL equivalent).
"""

import os
import tempfile

import torch

from . import _C

Config = _C.Config


class SeisCLForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cfg, param_names, field_names_out, output_fields, src,
                src_pos, rec_pos, *param_values):
        params = dict(zip(param_names, param_values))
        # ctx.needs_input_grad (one bool per forward() positional arg, in
        # order, False for non-tensor args) is the reliable way to check
        # this inside forward() -- checking .requires_grad directly on the
        # received tensors is not (see torch.autograd.Function docs).
        need_grad = any(ctx.needs_input_grad[7:])

        checkpoint_path = ""
        if need_grad:
            # time_stepping() creates/opens the file itself (see
            # checkpoint_d2h/checkpoint_h2d in src/time_stepping.c) -- only
            # reserve a unique path here.
            fd, checkpoint_path = tempfile.mkstemp(suffix="_checkpoint.mat")
            os.close(fd)
            os.remove(checkpoint_path)

        data = _C.run_forward(cfg, params, src, src_pos, rec_pos,
                              checkpoint_path, output_fields)
        data_names = list(data.keys())
        field_names_out.extend(data_names)

        ctx.cfg = cfg
        ctx.param_names = param_names
        ctx.data_names = data_names
        ctx.checkpoint_path = checkpoint_path if need_grad else None
        if need_grad:
            ctx.save_for_backward(src, src_pos, rec_pos, *param_values)

        return tuple(data[name] for name in data_names)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.checkpoint_path is None:
            raise RuntimeError(
                "SeisCLForward.backward() was called but no checkpoint was "
                "recorded during forward() -- this happens when none of "
                "the parameter tensors had requires_grad=True."
            )
        src, src_pos, rec_pos, *param_values = ctx.saved_tensors
        params = dict(zip(ctx.param_names, param_values))
        residuals = dict(zip(ctx.data_names, grad_outputs))

        try:
            grads = _C.run_backward(ctx.cfg, params, src, src_pos, rec_pos,
                                    residuals, ctx.checkpoint_path)
        finally:
            if os.path.exists(ctx.checkpoint_path):
                os.remove(ctx.checkpoint_path)

        # d(source wavelet)/d(geometry) gradients aren't computed here
        # (cfg.GRADSRCOUT support is a future extension, see the project
        # plan) -- always None for src/src_pos/rec_pos.
        grad_params = [grads.get(name) for name in ctx.param_names]

        # One None per non-tensor/non-grad-needing forward() arg:
        # (cfg, param_names, field_names_out, output_fields, src, src_pos,
        # rec_pos)
        return (None, None, None, None, None, None, None, *grad_params)


def seiscl_forward(cfg, params, src, src_pos, rec_pos, output_fields=None):
    """Differentiable SeisCL forward modeling.

    Parameters
    ----------
    cfg : Config
        Scalar run configuration (grid size, dt, dh, absorbing boundary,
        param_type, ...).
    params : dict[str, torch.Tensor]
        Flat (prod(cfg.N),) float32 tensors, keyed by the engine's
        *internal* parameter names -- "M", "mu", "rho" (add "taup"/"taus"
        if cfg.L>0) -- regardless of cfg.par_type. CUDA tensors are
        accepted and copied down automatically, which is a convenience
        rather than a fast path: the engine applies its parameter
        transforms on the host before uploading, so the values pass
        through host memory either way. par_type changes how the
        raw values under "M"/"mu" are interpreted, not the dict keys: for
        par_type=0 they're vp/vs in m/s. This differs from SeisCL.py's own
        `params` dict, which uses "vp"/"vs"/"rho" for par_type=0 (matching
        HDF5 dataset names) -- don't assume the two conventions match.
        Tensors with requires_grad=True receive gradients through
        backward().
    src, src_pos, rec_pos : torch.Tensor
        Source wavelets [allns, NT] and geometry [allns, 5] / [allng, 8],
        same convention as SeisCL.py's src_all/src_pos_all/rec_pos_all.
        These must be CPU tensors.
    output_fields : list[str], optional
        Which fields to record seismograms for (e.g. ["vx", "vz"],
        matching SeisCL.py's seisout=1 default for 2D). Defaults to every
        field the modeling case declares. The seismogram-output kernel is
        generated per the specific requested combination
        (automatic_kernels.c) -- pass only what you need.

    Returns
    -------
    dict[str, torch.Tensor]
        Modeled seismograms at the receivers, [allng, NT] per field name.
    """
    param_names = list(params.keys())
    param_values = [params[name] for name in param_names]
    field_names_out = []
    outputs = SeisCLForward.apply(cfg, param_names, field_names_out,
                                  output_fields or [], src, src_pos,
                                  rec_pos, *param_values)
    return dict(zip(field_names_out, outputs))
