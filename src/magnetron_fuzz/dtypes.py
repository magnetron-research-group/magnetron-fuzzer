# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Dtype tables and lossless conversion between magnetron and torch tensors."""

from __future__ import annotations

import math

import torch
from magnetron import Tensor, dtype as D

MAG2TORCH: dict[D.DType, torch.dtype] = {
    D.float32: torch.float32,
    D.float16: torch.float16,
    D.bfloat16: torch.bfloat16,
    D.float8_e4m3fn: torch.float8_e4m3fn,
    D.boolean: torch.bool,
    D.uint8: torch.uint8,
    D.int8: torch.int8,
    D.uint16: torch.uint16,
    D.int16: torch.int16,
    D.uint32: torch.uint32,
    D.int32: torch.int32,
    D.uint64: torch.uint64,
    D.int64: torch.int64,
}
TORCH2MAG: dict[torch.dtype, D.DType] = {v: k for k, v in MAG2TORCH.items()}

# Python-side names, used when emitting code (e.g. `D.float32`, `torch.float32`).
MAG_NAME: dict[D.DType, str] = {d: f'D.{d.name}' for d in MAG2TORCH}
TORCH_NAME: dict[torch.dtype, str] = {t: str(t) for t in MAG2TORCH.values()}

FLOATS = (torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn)
SINTS = (torch.int8, torch.int16, torch.int32, torch.int64)
UINTS = (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
INTS = SINTS + UINTS
ALL = FLOATS + INTS + (torch.bool,)

# Masks mirroring MAG_DTYPE_MASK_* in magnetron/core/mag_operator.h.
FP = frozenset(FLOATS)
INTEGER = frozenset(INTS)
INTEGRAL = INTEGER | {torch.bool}
NUMERIC = INTEGER | FP
BOOL = frozenset({torch.bool})
ANY = frozenset(ALL)

# Dtypes for which torch lacks most compute kernels; the reference upcasts these.
TORCH_WEAK = frozenset({torch.float8_e4m3fn, torch.uint16, torch.uint32, torch.uint64})

ITEMSIZE: dict[torch.dtype, int] = {t: torch.empty((), dtype=t).element_size() for t in ALL}

# (rtol, atol) for elementwise comparison of a single rounding step.
TOLS: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-4, 1e-5),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1e-2),
    torch.float8_e4m3fn: (0.13, 0.02),
}
# Unit roundoff used to bound accumulation error (sum, matmul, conv, ...).
EPS: dict[torch.dtype, float] = {
    torch.float32: 2.0**-23,
    torch.float16: 2.0**-10,
    torch.bfloat16: 2.0**-7,
    torch.float8_e4m3fn: 2.0**-3,
}


def is_float(t: torch.dtype) -> bool:
    return t in FP


def upcast_dtype(t: torch.dtype) -> torch.dtype:
    """Dtype torch can compute in for a TORCH_WEAK dtype."""
    if t == torch.float8_e4m3fn:
        return torch.float32
    if t in (torch.uint16, torch.uint32, torch.uint64):
        return torch.int64
    return t


def to_torch(m: Tensor) -> torch.Tensor:
    """Bit-exact copy of a magnetron tensor (any layout) as a contiguous CPU torch tensor."""
    td = MAG2TORCH[m.dtype]
    shape = tuple(m.shape)
    if m.numel == 0:
        return torch.empty(shape, dtype=td)
    buf = bytearray(m.tobytes())
    return torch.frombuffer(buf, dtype=td).reshape(shape)


def to_mag(t: torch.Tensor) -> Tensor:
    """Contiguous magnetron copy of a torch tensor."""
    if t.numel() == 0:
        return Tensor.empty(*t.shape, dtype=TORCH2MAG[t.dtype]) if t.dim() else Tensor.empty(dtype=TORCH2MAG[t.dtype])
    return Tensor(t.detach().contiguous())


def lit(v: object) -> str:
    """Python literal for a scalar that also round-trips nan/inf."""
    if isinstance(v, float):
        if math.isnan(v):
            return "float('nan')"
        if math.isinf(v):
            return "float('inf')" if v > 0 else "-float('inf')"
    return repr(v)
