# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Torch reference helpers referenced by generated reference code (and by emitted repro scripts)."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from .dtypes import FP


def f64(x: torch.Tensor, fn) -> torch.Tensor:  # noqa: ANN001
    """Evaluate fn in float64 and round once to x's dtype (magnetron computes half types in f32 and rounds once)."""
    if x.dtype in FP:
        return fn(x.double()).to(x.dtype)
    return fn(x)


def softmax_dv(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 0:
        return torch.zeros_like(x)
    s = torch.softmax(x, -1)
    return s * (1 - s)


def sigmoid_dv(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1 - s)


def silu_dv(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s + x * s * (1 - s)


def tanh_dv(x: torch.Tensor) -> torch.Tensor:
    t = torch.tanh(x)
    return 1 - t * t


def gelu_dv(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x * 0.7071067811865476)) + x * 0.3989422804014327 * torch.exp(-0.5 * x * x)


def step(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(x.dtype)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.softmax(x, dim) if x.dim() else torch.ones_like(x)


def bitnot(x: torch.Tensor) -> torch.Tensor:
    return torch.logical_not(x) if x.dtype == torch.bool else torch.bitwise_not(x)


def clamp_shift(y: torch.Tensor, bits: int) -> torch.Tensor:
    return y.abs() % bits


__all__ = ['F', 'bitnot', 'clamp_shift', 'f64', 'gelu_dv', 'sigmoid_dv', 'silu_dv', 'softmax', 'softmax_dv', 'step', 'tanh_dv', 'torch']


def binf(op, a, b):  # noqa: ANN001, ANN201
    """Binary op with magnetron's float model: half types are computed in f32 and rounded once."""
    rdt = torch.result_type(a, b)
    if rdt in FP and rdt != torch.float32:
        a = a.float() if isinstance(a, torch.Tensor) else a
        b = b.float() if isinstance(b, torch.Tensor) else b
        return op(a, b).to(rdt)
    return op(a, b)


def reduce_dims(fn, x: torch.Tensor, dims, keepdim: bool) -> torch.Tensor:  # noqa: ANN001
    """Apply a single-dim reduction (e.g. torch.prod) over several dims."""
    if dims is None:
        return fn(x.reshape(-1), 0) if not keepdim else fn(x.reshape(-1), 0).reshape([1] * x.dim())
    if isinstance(dims, int):
        dims = (dims,)
    dims = sorted({d % x.dim() for d in dims}, reverse=True)
    for d in dims:
        x = fn(x, d, keepdim=keepdim)
    return x


def cast_mask(x: torch.Tensor, to: torch.dtype) -> torch.Tensor:
    """Elements whose conversion x -> to is well defined (C leaves out-of-range float->int undefined)."""
    if to in FP and to != torch.float8_e4m3fn:
        return torch.ones(x.shape, dtype=torch.bool)
    if x.dtype == torch.bool or to == torch.bool:
        return torch.ones(x.shape, dtype=torch.bool)
    xd = x.double() if x.dtype in FP else x.to(torch.float64) if x.dtype != torch.uint64 else x.view(torch.int64).double()
    if to == torch.float8_e4m3fn:
        return (xd.abs() <= 448) | torch.isnan(xd)
    if x.dtype in FP:
        info = torch.iinfo(to)
        return torch.isfinite(xd) & (xd > float(info.min) - 1) & (xd < float(info.max) + 1)
    return torch.ones(x.shape, dtype=torch.bool)


__all__ += ['binf', 'cast_mask', 'reduce_dims']
