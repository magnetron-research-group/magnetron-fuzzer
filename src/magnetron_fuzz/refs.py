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


def step(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(x.dtype)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.softmax(x, dim) if x.dim() else torch.ones_like(x)


def bitnot(x: torch.Tensor) -> torch.Tensor:
    return torch.logical_not(x) if x.dtype == torch.bool else torch.bitwise_not(x)


def clamp_shift(y: torch.Tensor, bits: int) -> torch.Tensor:
    return y.abs() % bits


__all__ = ['F', 'bitnot', 'clamp_shift', 'f64', 'softmax', 'step', 'torch']


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


def prod_exact(x: torch.Tensor, dims, keepdim: bool) -> torch.Tensor:  # noqa: ANN001
    """Float product over dims tracked as mantissa * 2^exponent: no partial product overflows to inf (then inf*0 = NaN)
    or flushes to 0, so the result is the order-independent value a reordered (tree) reduction should produce."""
    nd = x.dim()
    red = list(range(nd)) if dims is None else sorted({d % nd for d in ((dims,) if isinstance(dims, int) else dims)})
    keep = [d for d in range(nd) if d not in red]
    flat = x.double().permute(keep + red).reshape([x.shape[d] for d in keep] + [-1])
    m, e = torch.frexp(flat)
    am, ae = torch.ones(flat.shape[:-1], dtype=torch.float64), torch.zeros(flat.shape[:-1], dtype=torch.int64)
    for i in range(flat.shape[-1]):
        am, k = torch.frexp(am * m[..., i])
        ae = ae + e[..., i] + k
    ae = ae.clamp(-3000, 3000)
    third = torch.div(ae, 3, rounding_mode='trunc')
    r = torch.ldexp(torch.ldexp(torch.ldexp(am, third), third), ae - 2 * third)  # each factor stays finite
    if keepdim and dims is not None:
        for d in red:
            r = r.unsqueeze(d)
    elif keepdim:
        r = r.reshape([1] * nd)
    return r


__all__ += ['binf', 'cast_mask', 'prod_exact', 'reduce_dims']


def nearest(x: torch.Tensor, **kw) -> torch.Tensor:
    """Nearest(-exact) reference: torch's source-index rounding depends on the dtype (float32 math for float32 inputs), so
    interpolate a float32 index grid and gather. This also keeps large integers exact."""
    sp = tuple(x.shape[2:])
    n = 1
    for d in sp:
        n *= d
    grid = torch.arange(n, dtype=torch.float32).reshape(1, 1, *sp)
    src = F.interpolate(grid, **kw).round().long()
    flat = x.reshape(x.shape[0], x.shape[1], n)
    return flat[:, :, src.reshape(-1)].reshape(x.shape[0], x.shape[1], *src.shape[2:])


__all__ += ['nearest']
