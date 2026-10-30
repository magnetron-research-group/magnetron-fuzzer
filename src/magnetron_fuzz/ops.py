# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Random step generators, one per magnetron operator family.

Each generator inspects the program's tensor pool and returns a Step (magnetron code + torch reference),
or None if no suitable inputs are available. Dtype masks mirror magnetron/core/mag_operator.h, so a
magnetron error on a supported dtype is reported while errors on unsupported dtypes only exercise validation.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .dtypes import ALL, ANY, BOOL, FP, INTEGRAL, INTS, ITEMSIZE, MAG_NAME, NUMERIC, TORCH2MAG, TORCH_NAME, UINTS, lit, upcast_dtype
from .engine import Cmp, Program, Step, Var, values_equal
from .gen import Gen, numel

Generator = Callable[[Program], Step | None]
OPS: list[tuple[str, float, Generator]] = []

DTYPE_WEIGHTS = {
    torch.float32: 0.33, torch.float16: 0.1, torch.bfloat16: 0.1, torch.float8_e4m3fn: 0.04, torch.bool: 0.07,
    torch.int64: 0.08, torch.int32: 0.07, torch.int16: 0.03, torch.int8: 0.04,
    torch.uint8: 0.04, torch.uint16: 0.03, torch.uint32: 0.03, torch.uint64: 0.04,
}  # fmt: skip


def op(name: str, weight: float) -> Callable[[Generator], Generator]:
    def deco(fn: Generator) -> Generator:
        OPS.append((name, weight, fn))
        return fn

    return deco


# -- helpers ----------------------------------------------------------------------------------------


def rand_dtype(g: Gen, mask: frozenset = ANY) -> torch.dtype:
    dts = [d for d in ALL if d in mask]
    return g.rng.choices(dts, [DTYPE_WEIGHTS[d] for d in dts])[0]


def mdt(t: torch.dtype) -> str:
    return MAG_NAME[TORCH2MAG[t]]


def tdt(t: torch.dtype) -> str:
    return TORCH_NAME[t]


def pick(p: Program, mask: frozenset = ANY, pred: Callable[[Var], bool] | None = None, p_any: float = 0.06) -> tuple[int, bool] | None:
    """Pick an input var; with probability p_any ignore the dtype mask to exercise magnetron's validation."""
    ok = (lambda v: pred is None or pred(v))
    if p.gen.chance(p_any):
        i = p.pick(ok)
        return None if i is None else (i, p.pool[i].dt in mask)
    i = p.pick(lambda v: v.dt in mask and ok(v))
    return None if i is None else (i, True)


def norm_dim(g: Gen, rank: int) -> int:
    d = g.randint(0, rank - 1)
    return d - rank if g.chance(0.3) else d


def bshape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...] | None:
    out = []
    for i in range(1, max(len(a), len(b)) + 1):
        x = a[-i] if i <= len(a) else 1
        y = b[-i] if i <= len(b) else 1
        if x != y and x != 1 and y != 1:
            return None
        out.append(y if x == 1 else x)
    return tuple(reversed(out))


def corrupt_index(g: Gen, ix: torch.Tensor, size: int) -> tuple[torch.Tensor, bool]:
    """Sometimes plant out-of-range indices (size, -size-1, huge) to probe bounds checking."""
    if not ix.numel() or not g.chance(0.08):
        return ix, False
    flat = ix.reshape(-1).clone()
    flat[g.randint(0, flat.numel() - 1)] = g.choice((size, size + 1, -size - 1, 2**31, -(2**31), 2**62, -(2**63)))
    return flat.reshape(ix.shape), True


def maybe_permuted_aux(g: Gen, step_aux: dict, name: str, t: torch.Tensor) -> str:
    """Register aux tensor `name` holding t; sometimes stored permuted so the operand is non-contiguous."""
    if t.dim() >= 2 and g.chance(0.25):
        perm = g.perm(t.dim())
        inv = [perm.index(i) for i in range(t.dim())]
        step_aux[name] = t.permute(*perm).contiguous()
        return f'{name}.permute({", ".join(map(str, inv))})'
    step_aux[name] = t
    return name


def scalar_for(g: Gen, dt: torch.dtype) -> object:
    if dt == torch.bool:
        return g.chance(0.5) if g.chance(0.7) else g.int_scalar(0, 3)
    if dt in FP:
        return g.float_scalar() if g.chance(0.85) else g.int_scalar()
    if g.chance(0.1):
        return g.float_scalar()
    return g.int_scalar(0, 20) if dt in UINTS else g.int_scalar()


# -- leaves -----------------------------------------------------------------------------------------


@op('leaf', 4.0)
def leaf(p: Program) -> Step | None:
    g = p.gen
    dt = rand_dtype(g)
    shape = g.shape()
    t = g.tensor(shape, dt)
    return Step('leaf', [], 'r = a0', 'r = a0', aux={'a0': t}, cmp=Cmp(exact=True))


@op('factory', 1.0)
def factory(p: Program) -> Step | None:
    g = p.gen
    dt = rand_dtype(g, NUMERIC)
    shape = g.shape(min_rank=1, budget=g.max_numel)
    kind = g.choice(('zeros', 'ones', 'full', 'arange', 'linspace', 'eye'))
    if kind in ('zeros', 'ones'):
        return Step(kind, [], f'r = Tensor.{kind}({shape!r}, dtype={mdt(dt)})', f'r = torch.{kind}({shape!r}, dtype={tdt(dt)})', cmp=Cmp(exact=True))
    if kind == 'full':
        v = scalar_for(g, dt)
        if dt in INTS and not isinstance(v, int):
            v = int(v) if math.isfinite(v) else 1
        return Step('full', [], f'r = Tensor.full({shape!r}, fill_value={lit(v)}, dtype={mdt(dt)})', f'r = torch.full({shape!r}, {lit(v)}, dtype={tdt(dt)})')
    if kind == 'eye':
        n, m = g.randint(0, 20), g.randint(0, 20)
        return Step('eye', [], f'r = Tensor.eye({n}, {m}, dtype={mdt(dt)})', f'r = torch.eye({n}, {m}, dtype={tdt(dt)})', cmp=Cmp(exact=True))
    if kind == 'arange':
        if dt in FP and g.chance(0.5):
            start, step = g.float_scalar(), g.choice((0.5, 0.25, 1.5, -0.5, 0.1, 3.0))
            if not math.isfinite(start):
                start = 0.0
        else:
            start, step = g.int_scalar(-20, 20), g.choice((1, 1, 2, 3, -1, -2))
        n = g.randint(0, 300)
        stop = start + n * step
        if dt in UINTS:
            start, step, stop = abs(int(start)), abs(int(step)) or 1, abs(int(start)) + n * (abs(int(step)) or 1)
        wide = 'torch.float64' if dt in FP else 'torch.int64'
        return Step(
            'arange', [], f'r = Tensor.arange({lit(start)}, {lit(stop)}, {lit(step)}, dtype={mdt(dt)})',
            f'r = torch.arange({lit(start)}, {lit(stop)}, {lit(step)}, dtype={wide}).to({tdt(dt)})',
            cmp=Cmp(scale=f's = torch.arange({lit(start)}, {lit(stop)}, {lit(step)}, dtype=torch.float64).abs() + {abs(float(start))!r}', acc_k=4),
        )  # fmt: skip
    if dt not in FP:
        dt = torch.float32
    a, b, n = g.float_scalar(), g.float_scalar(), g.randint(0, 200)
    if not (math.isfinite(a) and math.isfinite(b)):
        a, b = -1.0, 1.0
    return Step(
        'linspace', [], f'r = Tensor.linspace({lit(a)}, {lit(b)}, {n}, dtype={mdt(dt)})',
        f'r = torch.linspace({lit(a)}, {lit(b)}, {n}, dtype=torch.float64).to({tdt(dt)})',
        cmp=Cmp(scale=f's = torch.full(({n},), {abs(a) + abs(b)!r}, dtype=torch.float64)', acc_k=4),
    )  # fmt: skip


# -- views ------------------------------------------------------------------------------------------


def view_step(p: Program, name: str, i: int, code: str, ref: str | None = None, *, kind: str = 'view') -> Step:
    return Step(name, [i], f'r = x0{code}', f'r = x0{ref if ref is not None else code}', kind=kind, cmp=Cmp(exact=True), upcast_ok=False)


@op('view', 14.0)
def view_ops(p: Program) -> Step | None:
    g = p.gen
    i = p.pick()
    if i is None:
        return None
    v = p.pool[i]
    r, shape = v.rank, v.shape
    which = g.choice((
        'transpose', 'permute', 'reshape', 'view', 'squeeze', 'unsqueeze', 'flatten', 'unflatten', 'narrow', 'movedim',
        'select', 'split', 'unbind', 'expand', 'broadcast', 'flip', 'view_slice', 'getitem', 'T', 'contiguous', 'clone',
    ))  # fmt: skip
    if which in ('transpose', 'movedim') and r >= 1:
        a, b = norm_dim(g, r), norm_dim(g, r)
        return view_step(p, which, i, f'.{which}({a}, {b})')
    if which == 'permute' and r >= 1:
        return view_step(p, 'permute', i, f'.permute({", ".join(map(str, g.perm(r)))})')
    if which in ('reshape', 'view'):
        new = g.factorize(v.numel, g.randint(0, 5) if v.numel == 1 else g.randint(1, 5))
        if new and v.numel and g.chance(0.3):
            new[g.randint(0, len(new) - 1)] = -1
        if not new and v.numel != 1:
            return None
        args = ', '.join(map(str, new))
        return view_step(p, which, i, f'.{which}({args})')
    if which == 'squeeze':
        if r and g.chance(0.6):
            return view_step(p, 'squeeze', i, f'.squeeze({norm_dim(g, r)})')
        return view_step(p, 'squeeze', i, '.squeeze()')
    if which == 'unsqueeze':
        d = g.randint(-r - 1, r)
        return view_step(p, 'unsqueeze', i, f'.unsqueeze({d})')
    if which == 'flatten' and r >= 1:
        a = g.randint(0, r - 1)
        b = g.randint(a, r - 1)
        return view_step(p, 'flatten', i, f'.flatten({a}, {b})')
    if which == 'unflatten' and r >= 1:
        d = g.randint(0, r - 1)
        sizes = g.factorize(shape[d], g.randint(1, 3))
        if shape[d] and g.chance(0.3):
            sizes[g.randint(0, len(sizes) - 1)] = -1
        return view_step(p, 'unflatten', i, f'.unflatten({d}, {tuple(sizes)!r})')
    if which == 'narrow' and r >= 1:
        d = g.randint(0, r - 1)
        start = g.randint(0, shape[d])
        length = g.randint(0, shape[d] - start)
        if g.chance(0.2) and start:
            start -= shape[d]
        return view_step(p, 'narrow', i, f'.narrow({d}, {start}, {length})')
    if which == 'select' and r >= 1:
        d = g.randint(0, r - 1)
        if not shape[d]:
            return None
        idx = g.randint(-shape[d], shape[d] - 1)
        return view_step(p, 'select', i, f'.select({d - r if g.chance(0.3) else d}, {idx})')
    if which == 'split' and r >= 1:
        d = g.randint(0, r - 1)
        n = shape[d]
        if g.chance(0.5) and n:
            sizes = []
            rest = n
            while rest:
                k = g.randint(0, rest) if g.chance(0.2) else g.randint(1, rest)
                sizes.append(k)
                rest -= k
            st = Step('split', [i], f'r = x0.split({sizes!r}, {d})', f'r = x0.split({sizes!r}, {d})', kind='view', cmp=Cmp(exact=True), upcast_ok=False, keep='one')
            return st
        k = g.randint(1, max(1, n))
        return Step('split', [i], f'r = x0.split({k}, {d})', f'r = x0.split({k}, {d})', kind='view', cmp=Cmp(exact=True), upcast_ok=False, keep='one')
    if which == 'unbind' and r >= 1 and v.numel <= 4096:
        d = norm_dim(g, r)
        return Step('unbind', [i], f'r = x0.unbind({d})', f'r = x0.unbind({d})', kind='view', cmp=Cmp(exact=True), upcast_ok=False, keep='one')
    if which in ('expand', 'broadcast'):
        lead = [g.choice((1, 2, 3)) for _ in range(g.randint(0, 2) if r < 6 else 0)]
        tgt = lead + [g.choice((2, 3, 5)) if d == 1 and g.chance(0.6) else d for d in shape]
        if numel(tgt) > 4 * g.max_numel:
            return None
        if which == 'expand':
            args = [(-1 if (j >= len(lead) and tgt[j] == shape[j - len(lead)] and g.chance(0.3)) else tgt[j]) for j in range(len(tgt))]
            return view_step(p, 'expand', i, f'.expand({", ".join(map(str, args))})')
        return view_step(p, 'broadcast', i, f'.broadcast({", ".join(map(str, tgt))})', f'.broadcast_to({tuple(tgt)!r})')
    if which == 'flip' and r >= 1:
        dims = sorted(g.rng.sample(range(r), g.randint(1, r)))
        dims = [d - r if g.chance(0.2) else d for d in dims]
        return Step('flip', [i], f'r = x0.flip({", ".join(map(str, dims))})', f'r = torch.flip(x0, {dims!r})', kind='view', cmp=Cmp(exact=True), upcast_ok=False)
    if which == 'view_slice' and r >= 1:
        d = g.randint(0, r - 1)
        n = shape[d]
        if not n:
            return None
        step = g.choice((1, 1, 2, 3, 5))
        start = g.randint(0, n - 1)
        length = g.randint(1, (n - start - 1) // step + 1)
        idx = [':'] * r
        idx[d] = f'{start}:{start + (length - 1) * step + 1}:{step}'
        return view_step(p, 'view_slice', i, f'.view_slice({d}, {start}, {length}, {step})', f'[{", ".join(idx)}]')
    if which == 'getitem' and r >= 1:
        return view_step(p, 'getitem', i, f'[{basic_index(g, shape)}]')
    if which == 'T' and r == 2:
        return view_step(p, 'T', i, '.T')
    if which == 'contiguous':
        return view_step(p, 'contiguous', i, '.contiguous()')
    if which == 'clone':
        return view_step(p, 'clone', i, '.clone()', kind='new')
    return None


def basic_index(g: Gen, shape: tuple[int, ...]) -> str:
    """Random NumPy-style basic index (ints, slices, Ellipsis, None) for `shape`."""
    parts = []
    dims = list(shape)
    use_ellipsis = g.chance(0.15)
    n = g.randint(1, len(dims))
    ell_at = g.randint(0, n) if use_ellipsis else -1
    for k in range(n):
        if k == ell_at:
            parts.append('...')
        d = dims[k] if not use_ellipsis or k < ell_at else dims[len(dims) - (n - k)]
        c = g.rng.random()
        if c < 0.3 and d:
            parts.append(str(g.randint(-d, d - 1)))
        elif c < 0.33:
            parts.append('None')
        else:
            a = g.choice((None, 0, 1, -1, g.randint(-d - 2, d + 2)))
            b = g.choice((None, d, -1, g.randint(-d - 2, d + 2)))
            s = g.choice((None, 1, 2, 3))
            parts.append(f'{"" if a is None else a}:{"" if b is None else b}{"" if s is None else f":{s}"}')
    if ell_at == n:
        parts.append('...')
    return ', '.join(parts)


@op('strided_view', 1.5)
def strided_view(p: Program) -> Step | None:
    g = p.gen
    i = p.pick(lambda v: v.numel > 0 and v.m.is_contiguous and not v.m.is_view and v.t.storage_offset() == 0 and v.t.is_contiguous())
    if i is None:
        return None
    v = p.pool[i]
    n = v.numel
    rank = g.randint(0, 4)
    shape = [g.choice((1, 2, 3, 4, 5)) for _ in range(rank)]
    strides = [g.choice((0, 1, 1, 2, 3, 5)) for _ in range(rank)]
    span = sum((d - 1) * s for d, s in zip(shape, strides))
    if span >= n:
        return None
    off = g.randint(0, n - 1 - span)
    return Step(
        'strided_view', [i], f'r = x0.strided_view({tuple(shape)!r}, {tuple(strides)!r}, offset={off})',
        f'r = torch.as_strided(x0, {tuple(shape)!r}, {tuple(strides)!r}, {off})', kind='view', cmp=Cmp(exact=True), upcast_ok=False,
    )  # fmt: skip


@op('reinterpret_view', 0.7)
def reinterpret_view(p: Program) -> Step | None:
    g = p.gen
    i = p.pick(lambda v: v.numel > 0 and v.m.is_contiguous and v.t.is_contiguous() and v.rank >= 1)
    if i is None:
        return None
    v = p.pool[i]
    same = [d for d in ALL if ITEMSIZE[d] == ITEMSIZE[v.dt] and d != v.dt]
    if not same:
        return None
    to = g.choice(same)
    shape = ', '.join(map(str, v.shape))
    return Step(
        'reinterpret_view', [i], f'r = x0.reinterpret_view({mdt(to)}, {shape})', f'r = x0.view({tdt(to)})',
        kind='view', cmp=Cmp(exact=True), upcast_ok=False,
    )  # fmt: skip


# -- elementwise ------------------------------------------------------------------------------------

UNARY: list[tuple[str, frozenset, str]] = [
    ('abs', NUMERIC, 'torch.abs(x0)'), ('sgn', NUMERIC, 'torch.sign(x0)'), ('neg', NUMERIC, 'torch.neg(x0)'),
    ('log', FP, 'f64(x0, torch.log)'), ('log10', FP, 'f64(x0, torch.log10)'), ('log1p', FP, 'f64(x0, torch.log1p)'),
    ('log2', FP, 'f64(x0, torch.log2)'), ('sqr', NUMERIC, 'f64(x0, torch.square)'), ('rcp', FP, 'f64(x0, torch.reciprocal)'),
    ('sqrt', FP, 'f64(x0, torch.sqrt)'), ('rsqrt', FP, 'f64(x0, torch.rsqrt)'), ('sin', FP, 'f64(x0, torch.sin)'),
    ('cos', FP, 'f64(x0, torch.cos)'), ('tan', FP, 'f64(x0, torch.tan)'), ('sinh', FP, 'f64(x0, torch.sinh)'),
    ('cosh', FP, 'f64(x0, torch.cosh)'), ('tanh', FP, 'f64(x0, torch.tanh)'), ('asin', FP, 'f64(x0, torch.asin)'),
    ('acos', FP, 'f64(x0, torch.acos)'), ('atan', FP, 'f64(x0, torch.atan)'), ('asinh', FP, 'f64(x0, torch.asinh)'),
    ('acosh', FP, 'f64(x0, torch.acosh)'), ('atanh', FP, 'f64(x0, torch.atanh)'), ('step', FP, 'f64(x0, step)'),
    ('erf', FP, 'f64(x0, torch.erf)'), ('erfc', FP, 'f64(x0, torch.erfc)'), ('exp', FP, 'f64(x0, torch.exp)'),
    ('exp2', FP, 'f64(x0, torch.exp2)'), ('expm1', FP, 'f64(x0, torch.expm1)'), ('floor', FP, 'f64(x0, torch.floor)'),
    ('ceil', FP, 'f64(x0, torch.ceil)'), ('round', FP, 'f64(x0, torch.round)'), ('trunc', FP, 'f64(x0, torch.trunc)'),
    ('softmax_dv', FP, 'f64(x0, softmax_dv)'), ('sigmoid', FP, 'f64(x0, torch.sigmoid)'), ('sigmoid_dv', FP, 'f64(x0, sigmoid_dv)'),
    ('hard_sigmoid', FP, 'f64(x0, F.hardsigmoid)'), ('silu', FP, 'f64(x0, F.silu)'), ('silu_dv', FP, 'f64(x0, silu_dv)'),
    ('tanh_dv', FP, 'f64(x0, tanh_dv)'), ('relu', FP, 'f64(x0, torch.relu)'), ('relu_dv', FP, 'f64(x0, step)'),
    ('gelu', FP, 'f64(x0, F.gelu)'), ('gelu_approx', FP, "f64(x0, lambda v: F.gelu(v, approximate='tanh'))"),
    ('gelu_dv', FP, 'f64(x0, gelu_dv)'), ('logical_not', INTEGRAL, 'bitnot(x0)'),
]  # fmt: skip
# Periodic functions lose absolute accuracy proportional to |x| during range reduction.
RANGE_REDUCED = {'sin', 'cos', 'tan'}


def inplace_ok(v: Var) -> bool:
    return not v.overlapping()


@op('unary', 16.0)
def unary(p: Program) -> Step | None:
    g = p.gen
    name, mask, ref = g.choice(UNARY)
    sel = pick(p, mask)
    if sel is None:
        return None
    i, supported = sel
    v = p.pool[i]
    cmp = Cmp(scale='s = x0.double().abs()', acc_k=16) if name in RANGE_REDUCED else Cmp()
    if g.chance(0.3) and inplace_ok(v):
        return Step(f'{name}_', [i], f'r = x0.{name}_()', f'r = {ref}', kind='inplace', cmp=cmp, supported=supported)
    code = {'neg': '-x0', 'abs': 'abs(x0)', 'logical_not': '~x0'}.get(name) if g.chance(0.3) else None
    return Step(name, [i], f'r = {code or f"x0.{name}()"}', f'r = {ref}', cmp=cmp, supported=supported)


BINARY: list[tuple[str, str, frozenset]] = [
    ('add', '+', NUMERIC), ('sub', '-', NUMERIC), ('mul', '*', NUMERIC), ('truediv', '/', NUMERIC),
    ('floordiv', '//', NUMERIC), ('mod', '%', NUMERIC), ('pow', '**', NUMERIC),
    ('logical_and', '&', INTEGRAL), ('logical_or', '|', INTEGRAL), ('logical_xor', '^', INTEGRAL),
    ('lshift', '<<', INTEGRAL), ('rshift', '>>', INTEGRAL),
    ('lt', '<', ANY), ('le', '<=', ANY), ('gt', '>', ANY), ('ge', '>=', ANY), ('eq', '==', ANY), ('ne', '!=', ANY),
]  # fmt: skip
INPLACE_DUNDER = {'add': '+=', 'sub': '-=', 'mul': '*=', 'truediv': '/=', 'floordiv': '//=', 'mod': '%=', 'pow': '**=',
                  'logical_and': '&=', 'logical_or': '|=', 'logical_xor': '^=', 'lshift': '<<=', 'rshift': '>>='}  # fmt: skip
COMPARE = {'lt', 'le', 'gt', 'ge', 'eq', 'ne'}
DIVLIKE = {'truediv', 'floordiv', 'mod'}


def int_edges(name: str, dt: torch.dtype) -> list[int]:
    """Operand values that hit undefined/trapping/boundary behaviour in C kernels for this op."""
    info = torch.iinfo(dt)
    edges = [0, 1, info.max, info.min]
    if dt not in UINTS:
        edges += [-1]
    if name in ('lshift', 'rshift'):
        edges += [info.bits - 1, info.bits, info.bits + 1, 2 * info.bits, 63, 64] + ([-1, -info.bits] if dt not in UINTS else [])
    if name == 'pow':
        edges += [2, info.bits, info.bits + 1] + ([-1, -2] if dt not in UINTS else [])
    return [e for e in edges if info.min <= e <= info.max]


def edge_rhs(g: Gen, name: str, t: torch.Tensor) -> torch.Tensor:
    """Sprinkle edge values (zero divisors, -1, oversized/negative shifts, negative exponents) into an int rhs."""
    if t.dtype not in INTS or not t.numel() or not g.chance(0.5):
        return t
    edges = int_edges(name, t.dtype)
    flat = t.reshape(-1).clone()
    for _ in range(g.randint(1, max(1, min(flat.numel(), 6)))):
        e = g.choice(edges)
        flat[g.randint(0, flat.numel() - 1)] = torch.tensor(e, dtype=torch.int64).view(torch.uint64) if t.dtype == torch.uint64 else e
    return flat.reshape(t.shape)


def edge_scalar(g: Gen, name: str, dt: torch.dtype, s: object) -> object:
    return g.choice(int_edges(name, dt)) if dt in INTS and g.chance(0.4) else s


@op('binary', 18.0)
def binary(p: Program) -> Step | None:
    g = p.gen
    name, sym, mask = g.choice(BINARY)
    sel = pick(p, mask)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    inplace = name not in COMPARE and g.chance(0.25) and inplace_ok(x)
    aux: dict[str, torch.Tensor] = {}
    args = [i]
    c = g.rng.random()
    rhs_dt = x.dt if g.chance(0.8) else rand_dtype(g, mask)
    if c < 0.25:
        # scalar operand
        s = edge_scalar(g, name, x.dt, scalar_for(g, x.dt))
        rhs = lit(s)
        if not inplace and g.chance(0.25) and name not in COMPARE:
            # reflected: scalar op tensor
            return Step(f'r{name}', args, f'r = {rhs} {sym} x0', f'r = binf(lambda a, b: a {sym} b, {rhs}, x0)', supported=supported)
    elif c < 0.5:
        # pool operand
        def compatible(v: Var) -> bool:
            out = bshape(x.shape, v.shape)
            return out is not None and (not inplace or out == x.shape)

        j = p.pick(compatible)
        if j is None:
            return None
        args.append(j)
        rhs = 'x1'
    else:
        shape = g.broadcast_partner(x.shape, inplace)
        t = edge_rhs(g, name, g.tensor(shape, rhs_dt))
        rhs = maybe_permuted_aux(g, aux, 'a0', t)
    ref = f'r = binf(lambda a, b: a {sym} b, x0, {rhs})'
    if name in COMPARE:
        ref = f'r = x0 {sym} {rhs}'
    if inplace:
        form = g.choice(('method', 'dunder'))
        mag = f'r = x0.{name}_({rhs})' if form == 'method' else f'x0 {INPLACE_DUNDER[name]} {rhs}\nr = x0'
        return Step(f'{name}_', args, mag, ref, kind='inplace', aux=aux, supported=supported)
    mag = f'r = x0.{name}({rhs})' if g.chance(0.4) else f'r = x0 {sym} {rhs}'
    return Step(name, args, mag, ref, aux=aux, supported=supported)


@op('minmax_elementwise', 2.0)
def minmax_elementwise(p: Program) -> Step | None:
    g = p.gen
    sel = pick(p, ANY)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    name = g.choice(('max', 'min'))
    t = g.tensor(g.broadcast_partner(x.shape), x.dt)
    aux: dict[str, torch.Tensor] = {}
    rhs = maybe_permuted_aux(g, aux, 'a0', t)
    fn = 'torch.maximum' if name == 'max' else 'torch.minimum'
    return Step(f'{name}(other)', [i], f'r = x0.{name}({rhs})', f'r = {fn}(x0, {rhs})', aux=aux, cmp=Cmp(exact=True), supported=supported)


@op('where', 2.0)
def where(p: Program) -> Step | None:
    g = p.gen
    i = p.pick()
    if i is None:
        return None
    x = p.pool[i]
    aux = {'c': g.tensor(g.broadcast_partner(x.shape), torch.bool)}
    if g.chance(0.3):
        y = lit(scalar_for(g, x.dt))
    else:
        y = maybe_permuted_aux(g, aux, 'a0', g.tensor(g.broadcast_partner(x.shape), x.dt if g.chance(0.85) else rand_dtype(g)))
    if g.chance(0.5):
        return Step('where', [i], f'r = Tensor.where(c, x0, {y})', f'r = torch.where(c, x0, {y})', aux=aux, cmp=Cmp(exact=True))
    return Step('where', [i], f'r = Tensor.where(c, {y}, x0)', f'r = torch.where(c, {y}, x0)', aux=aux, cmp=Cmp(exact=True))


@op('clamp', 2.5)
def clamp(p: Program) -> Step | None:
    g = p.gen
    sel = pick(p, NUMERIC)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    lo, hi = sorted((scalar_for(g, x.dt), scalar_for(g, x.dt)), key=lambda z: float(z) if not (isinstance(z, float) and math.isnan(z)) else 0)
    if x.dt in INTS:
        lo, hi = int(lo) if math.isfinite(float(lo)) else 0, int(hi) if math.isfinite(float(hi)) else 10
    kind = g.choice(('clamp', 'clamp_t', 'clamp_min', 'clamp_max'))
    if kind == 'clamp':
        return Step('clamp', [i], f'r = x0.clamp({lit(lo)}, {lit(hi)})', f'r = torch.clamp(x0, {lit(lo)}, {lit(hi)})', supported=supported, cmp=Cmp(exact=True))
    if kind == 'clamp_t':
        aux = {'a0': g.tensor(g.broadcast_partner(x.shape, True), x.dt), 'a1': g.tensor(g.broadcast_partner(x.shape, True), x.dt)}
        return Step('clamp(tensor)', [i], 'r = x0.clamp(a0, a1)', 'r = torch.clamp(x0, a0, a1)', aux=aux, supported=supported, cmp=Cmp(exact=True))
    b = lo if kind == 'clamp_min' else hi
    return Step(kind, [i], f'r = x0.{kind}({lit(b)})', f'r = torch.{kind}(x0, {lit(b)})', supported=supported, cmp=Cmp(exact=True))


@op('lerp', 1.2)
def lerp(p: Program) -> Step | None:
    g = p.gen
    sel = pick(p, FP)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    aux = {'a0': g.tensor(g.broadcast_partner(x.shape, True), x.dt)}
    w = lit(g.choice((0.0, 0.25, 0.5, 1.0, 1.5, -0.5, g.float_scalar())))
    if g.chance(0.3):
        aux['w'] = g.tensor(g.broadcast_partner(x.shape, True), x.dt)
        w = 'w'
    ref = f'r = torch.lerp(x0.double(), a0.double(), {w if w != "w" else "w.double()"}).to(x0.dtype)'
    cmp = Cmp(scale=f's = x0.abs() + a0.abs() * (1 + {"w.abs()" if w == "w" else f"abs({w})"})', acc_k=4)
    if inplace_ok(x) and g.chance(0.3):
        return Step('lerp_', [i], f'r = x0.lerp_(a0, {w})', ref, kind='inplace', aux=aux, cmp=cmp, supported=supported)
    return Step('lerp', [i], f'r = x0.lerp(a0, {w})', ref, aux=aux, cmp=cmp, supported=supported)


@op('masked_fill', 1.5)
def masked_fill(p: Program) -> Step | None:
    g = p.gen
    i = p.pick()
    if i is None:
        return None
    x = p.pool[i]
    inplace = inplace_ok(x) and g.chance(0.4)
    aux = {'mk': g.tensor(g.broadcast_partner(x.shape, inplace), torch.bool)}
    val = scalar_for(g, x.dt)
    ref = f'r = torch.masked_fill(x0, mk, {lit(val)})'
    if inplace:
        return Step('masked_fill_', [i], f'r = x0.masked_fill_(mk, {lit(val)})', ref, kind='inplace', aux=aux, cmp=Cmp(exact=True))
    return Step('masked_fill', [i], f'r = x0.masked_fill(mk, {lit(val)})', ref, aux=aux, cmp=Cmp(exact=True))


@op('cast', 3.0)
def cast(p: Program) -> Step | None:
    g = p.gen
    i = p.pick()
    if i is None:
        return None
    to = rand_dtype(g)
    return Step(f'cast', [i], f'r = x0.cast({mdt(to)})', f'r = x0.to({tdt(to)})', cmp=Cmp(exact=True, mask=f'm = cast_mask(x0, {tdt(to)})'))


@op('fill', 2.0)
def fill(p: Program) -> Step | None:
    g = p.gen
    i = p.pick(inplace_ok)
    if i is None:
        return None
    x = p.pool[i]
    which = g.choice(('fill_', 'zero_', 'one_', 'copy_', 'setitem', 'setitem'))
    if which == 'fill_':
        val = scalar_for(g, x.dt)
        return Step('fill_', [i], f'r = x0.fill_({lit(val)})', f'r = torch.empty_like(x0).fill_({lit(val)})', kind='inplace', cmp=Cmp(exact=True))
    if which in ('zero_', 'one_'):
        return Step(which, [i], f'r = x0.{which}()', f'r = torch.{"zeros" if which == "zero_" else "ones"}_like(x0)', kind='inplace', cmp=Cmp(exact=True))
    if which == 'copy_':
        src = g.tensor(g.broadcast_partner(x.shape, True), rand_dtype(g) if g.chance(0.5) else x.dt)
        aux: dict[str, torch.Tensor] = {}
        a = maybe_permuted_aux(g, aux, 'a0', src)
        return Step(
            'copy_', [i], f'r = x0.copy_({a})', f'r = {a}.to(x0.dtype).expand_as(x0).clone()', kind='inplace', aux=aux,
            cmp=Cmp(exact=True, mask=f'm = cast_mask({a}, {tdt(x.dt)}).expand_as(x0)'),
        )  # fmt: skip
    if x.rank == 0:
        return None
    idx = basic_index(g, x.shape)
    try:
        region = x.t[eval(f'np_index[{idx}]', {'np_index': _IndexCapture()})]
    except Exception:  # noqa: BLE001
        return None
    if g.chance(0.5):
        val = lit(scalar_for(g, x.dt))
        aux = {}
    else:
        aux = {'a0': g.tensor(g.broadcast_partner(tuple(region.shape), True), x.dt)}
        val = 'a0'
    return Step('setitem', [i], f'x0[{idx}] = {val}\nr = x0', f'r = x0.clone()\nr[{idx}] = {val}', kind='inplace', aux=aux, cmp=Cmp(exact=True))


class _IndexCapture:
    def __getitem__(self, idx: object) -> object:
        return idx


@op('getitem_adv', 2.0)
def getitem_adv(p: Program) -> Step | None:
    g = p.gen
    i = p.pick(lambda v: v.rank >= 1)
    if i is None:
        return None
    x = p.pool[i]
    if g.chance(0.4):
        k = g.randint(1, x.rank)
        aux = {'mk': g.tensor(x.shape[:k], torch.bool)}
        return Step('getitem(bool)', [i], 'r = x0[mk]', 'r = x0[mk]', aux=aux, cmp=Cmp(exact=True), upcast_ok=False)
    d = g.randint(0, x.rank - 1) if g.chance(0.4) else 0
    if not x.shape[d]:
        return None
    ix, bad = corrupt_index(g, g.index(x.shape[d], (g.randint(0, 8),)), x.shape[d])
    pre = ':, ' * d
    return Step('getitem(int)', [i], f'r = x0[{pre}ix]', f'r = x0[{pre}ix]', aux={'ix': ix}, cmp=Cmp(exact=True), upcast_ok=False, invalid=bad)


# -- reductions -------------------------------------------------------------------------------------


def rand_dims(g: Gen, rank: int) -> tuple[object, int]:
    """(dim argument, number of dims reduced)."""
    if rank == 0 or g.chance(0.25):
        return None, rank
    if g.chance(0.6):
        return norm_dim(g, rank), 1
    k = g.randint(1, rank)
    dims = g.rng.sample(range(rank), k)
    dims = [d - rank if g.chance(0.2) else d for d in dims]
    return tuple(dims) if g.chance(0.5) else list(dims), k


def reduced_count(shape: tuple[int, ...], dims: object) -> int:
    if dims is None:
        return max(1, numel(shape))
    if isinstance(dims, int):
        dims = [dims]
    return max(1, math.prod(shape[d % len(shape)] for d in set(d % len(shape) for d in dims)))


@op('reduce', 10.0)
def reduce(p: Program) -> Step | None:
    g = p.gen
    name = g.choice(('sum', 'sum', 'mean', 'prod', 'amax', 'amin', 'all', 'any'))
    mask = {'mean': FP, 'all': ANY, 'any': ANY}.get(name, NUMERIC)
    sel = pick(p, mask)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    dims, _ = rand_dims(g, x.rank)
    n = reduced_count(x.shape, dims)
    keep = g.chance(0.4) and dims is not None
    kw = f'dim={dims!r}, keepdim={keep}' if dims is not None else ''
    tkw = f', dim={dims!r}, keepdim={keep}' if dims is not None else ''
    mag = f'r = x0.{name}({kw})'
    acc = 2 * math.sqrt(n) + 8
    if name == 'sum':
        return Step(name, [i], mag, f'r = torch.sum(x0{tkw})', cmp=Cmp(scale=f's = torch.sum(x0.double().abs(){tkw})', acc_k=acc), supported=supported)
    if name == 'mean':
        return Step(name, [i], mag, f'r = torch.mean(x0.double(){tkw}).to(x0.dtype)', cmp=Cmp(scale=f's = torch.mean(x0.double().abs(){tkw})', acc_k=acc), supported=supported)
    if name == 'prod':
        ref = f'r = reduce_dims(torch.prod, x0, {dims!r}, {keep})' if dims is not None else 'r = torch.prod(x0)'
        if x.dt in FP:
            ref = f'r = reduce_dims(torch.prod, x0.double(), {dims!r}, {keep}).to(x0.dtype)'
        return Step(name, [i], mag, ref, cmp=Cmp(rtol_mul=1 + n / 2), supported=supported)
    if name in ('all', 'any'):
        return Step(name, [i], mag, f'r = torch.{name}(x0{tkw})', cmp=Cmp(check_dtype=x.dt != torch.uint8), supported=supported)
    return Step(name, [i], mag, f'r = torch.{name}(x0{tkw})', cmp=Cmp(exact=True), supported=supported)


def check_index_select(x: torch.Tensor, dim: int | None, keep: bool, values: torch.Tensor, idx: torch.Tensor, ref_idx: torch.Tensor | None) -> str | tuple[str, str] | None:
    """Indices must point at elements equal to `values` (max/min/argmax...); ties may differ from torch."""
    if idx.dtype != torch.int64:
        return f'indices dtype {idx.dtype}, expected int64'
    if dim is None:
        flat = _cmp(x).reshape(-1)
        if idx.numel() != 1 or not (0 <= int(idx.reshape(-1)[0]) < max(1, flat.numel())):
            return f'index {idx.tolist()} out of range for numel {flat.numel()}'
        got = flat[idx.reshape(-1)].reshape(values.shape)
    else:
        size = x.shape[dim] if x.dim() else 1
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= size):
            return f'indices out of range [0, {size}): min={int(idx.min())} max={int(idx.max())}'
        ii = idx if keep or x.dim() == 0 else idx.unsqueeze(dim)
        got = _cmp(x).gather(dim, ii) if x.dim() else _cmp(x)
        got = got if keep or x.dim() == 0 else got.squeeze(dim)
    eq = values_equal(got, values)
    if not bool(eq.all()):
        return 'indices select elements that are not the extremum'
    if ref_idx is not None and not torch.equal(idx, ref_idx):
        return 'tiebreak', f'indices differ from torch on ties: magnetron={idx.reshape(-1)[:8].tolist()} torch={ref_idx.reshape(-1)[:8].tolist()}'
    return None


def _cmp(x: torch.Tensor) -> torch.Tensor:
    from .engine import _cmp_view

    return _cmp_view(x)


@op('reduce_index', 5.0)
def reduce_index(p: Program) -> Step | None:
    g = p.gen
    name = g.choice(('max', 'min', 'argmax', 'argmin', 'max_full', 'min_full'))
    sel = pick(p, NUMERIC)  # MINIMA/MAXIMA/ARGMIN/ARGMAX are NUMERIC in mag_operator.h
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    keep = g.chance(0.4)
    if name.endswith('_full'):
        fn = name[:3]
        return Step(fn, [i], f'r = x0.{fn}()', f'r = torch.{"amax" if fn == "max" else "amin"}(x0)', cmp=Cmp(exact=True), supported=supported)
    if name.startswith('arg'):
        dim = norm_dim(g, x.rank) if x.rank and g.chance(0.7) else None
        kw = f'dim={dim}, keepdim={keep}' if dim is not None else ''

        def check(mo: list, ro: list, ins: list) -> object:
            x0 = ins[0]
            d = None if dim is None else dim % max(1, x0.dim())
            return check_index_select(x0, d, keep, _extremum(x0, name, dim, keep), mo[0], ro[0])

        return Step(name, [i], f'r = x0.{name}({kw})', f'r = torch.{name}(x0{", " + kw if kw else ""})', check=check, supported=supported)
    if x.rank == 0:
        return None
    dim = norm_dim(g, x.rank)

    def check2(mo: list, ro: list, ins: list) -> object:
        msg = values_check(mo[0], ro[0])
        return msg or check_index_select(ins[0], dim % ins[0].dim(), keep, ro[0], mo[1], ro[1])

    return Step(name, [i], f'r = x0.{name}({dim}, keepdim={keep})', f'r = tuple(torch.{name}(x0, {dim}, keepdim={keep}))', check=check2, supported=supported)


def _extremum(x: torch.Tensor, name: str, dim: int | None, keep: bool) -> torch.Tensor:
    fn = torch.amax if name == 'argmax' else torch.amin
    xx = _cmp(x)
    return fn(xx) if dim is None else fn(xx, dim=dim, keepdim=keep)


def values_check(got: torch.Tensor, exp: torch.Tensor) -> str | None:
    if tuple(got.shape) != tuple(exp.shape):
        return f'values shape {tuple(got.shape)} vs torch {tuple(exp.shape)}'
    if got.dtype != exp.dtype and exp.dtype != upcast_dtype(got.dtype):
        return f'values dtype {got.dtype} vs torch {exp.dtype}'
    eq = values_equal(got, exp.to(got.dtype) if got.dtype != exp.dtype else exp)
    if not bool(eq.all()):
        from .engine import _fmt_bad

        return 'values: ' + _fmt_bad(got, exp.to(got.dtype), eq)
    return None


@op('cumulative', 4.0)
def cumulative(p: Program) -> Step | None:
    g = p.gen
    name = g.choice(('cusum', 'cuprod', 'cumax', 'cumin'))
    sel = pick(p, NUMERIC, lambda v: v.rank >= 1)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    dim = norm_dim(g, x.rank)
    n = max(1, x.shape[dim])
    if name == 'cusum':
        ref = 'r = torch.cumsum(x0.double(), {d}).to(x0.dtype)' if x.dt in FP else 'r = torch.cumsum(x0, {d})'
        return Step(name, [i], f'r = x0.cusum({dim})', ref.format(d=dim), cmp=Cmp(scale=f's = torch.cumsum(x0.double().abs(), {dim})', acc_k=2 * math.sqrt(n) + 8), supported=supported)
    if name == 'cuprod':
        ref = 'r = torch.cumprod(x0.double(), {d}).to(x0.dtype)' if x.dt in FP else 'r = torch.cumprod(x0, {d})'
        return Step(name, [i], f'r = x0.cuprod({dim})', ref.format(d=dim), cmp=Cmp(rtol_mul=1 + n / 2), supported=supported)
    fn = 'cummax' if name == 'cumax' else 'cummin'

    def check(mo: list, ro: list, ins: list) -> object:
        msg = values_check(mo[0], ro[0])
        if msg:
            return msg
        idx = mo[1]
        if idx.dtype != torch.int64:
            return f'indices dtype {idx.dtype}'
        d = dim % ins[0].dim()
        if idx.numel():
            pos = torch.arange(ins[0].shape[d]).reshape([-1 if k == d else 1 for k in range(ins[0].dim())])
            if bool((idx < 0).any()) or bool((idx > pos).any()):
                return 'cumulative indices point past the current position'
            if not bool(values_equal(_cmp(ins[0]).gather(d, idx), ro[0]).all()):
                return 'cumulative indices select wrong elements'
        if not torch.equal(idx, ro[1]):
            return 'tiebreak', f'indices differ from torch on ties: magnetron={idx.reshape(-1)[:8].tolist()} torch={ro[1].reshape(-1)[:8].tolist()}'
        return None

    return Step(name, [i], f'r = x0.{name}({dim})', f'r = tuple(torch.{fn}(x0, {dim}))', check=check, supported=supported)


@op('sort', 4.0)
def sort(p: Program) -> Step | None:
    g = p.gen
    name = g.choice(('sort', 'argsort', 'topk'))
    sel = pick(p, NUMERIC if name == 'topk' else ANY)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    dim = norm_dim(g, x.rank) if x.rank else -1
    size = x.shape[dim] if x.rank else 1
    desc = g.chance(0.5)
    stable = g.chance(0.5)
    if name == 'topk':
        k = g.randint(0, size)
        largest, srt = g.chance(0.7), g.chance(0.8)

        def check(mo: list, ro: list, ins: list) -> object:
            vals, idx = mo
            ref_v = ro[0]
            if srt:
                msg = values_check(vals, ref_v)
            else:
                msg = values_check(torch.sort(_cmp(vals), dim=dim, descending=largest).values, torch.sort(_cmp(ref_v), dim=dim, descending=largest).values)
            if msg:
                return msg
            if idx.dtype != torch.int64:
                return f'indices dtype {idx.dtype}'
            if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= size):
                return 'topk indices out of range'
            if ins[0].dim() and not bool(values_equal(_cmp(ins[0]).gather(dim, idx), vals).all()):
                return 'topk indices do not point at the returned values'
            return None

        return Step('topk', [i], f'r = x0.topk({k}, dim={dim}, largest={largest}, sorted={srt})', f'r = tuple(torch.topk(x0, {k}, dim={dim}, largest={largest}, sorted={srt}))', check=check, supported=supported)

    def check_sort(mo: list, ro: list, ins: list) -> object:
        if name == 'sort':
            msg = values_check(mo[0], ro[0])
            if msg:
                return msg
            idx, ref_idx, vals = mo[1], ro[1], ro[0]
        else:
            idx, ref_idx = mo[0], ro[0]
            vals = torch.sort(_cmp(ins[0]), dim=dim, descending=desc, stable=True).values if ins[0].dim() else ins[0]
        if idx.dtype != torch.int64:
            return f'indices dtype {idx.dtype}'
        if tuple(idx.shape) != tuple(ref_idx.shape):
            return f'indices shape {tuple(idx.shape)} vs {tuple(ref_idx.shape)}'
        if not ins[0].dim():
            return None if torch.equal(idx, ref_idx) else 'indices of 0-d sort must be 0'
        if idx.numel():
            if bool(((idx < 0) | (idx >= size)).any()):
                return 'sort indices out of range'
            if not torch.equal(torch.sort(idx, dim=dim).values, torch.sort(ref_idx, dim=dim).values):
                return 'sort indices are not a permutation'
            if not bool(values_equal(_cmp(ins[0]).gather(dim, idx), vals).all()):
                return 'sort indices do not produce sorted order'
        if stable and not torch.equal(idx, ref_idx):
            return 'stable sort indices differ from torch'
        return None

    kw = f'dim={dim}, descending={desc}, stable={stable}'
    if name == 'sort':
        return Step('sort', [i], f'r = x0.sort({kw})', f'r = tuple(torch.sort(x0, {kw}))', check=check_sort, supported=supported)
    return Step('argsort', [i], f'r = x0.argsort({kw})', f'r = torch.argsort(x0, {kw})', check=check_sort, supported=supported)


# -- indexing / scatter -----------------------------------------------------------------------------


@op('gather', 2.5)
def gather(p: Program) -> Step | None:
    g = p.gen
    sel = pick(p, ANY, lambda v: v.rank >= 1 and v.numel > 0)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    dim = norm_dim(g, x.rank)
    d = dim % x.rank
    ishape = tuple((g.randint(1, 2 * x.shape[k] + 1) if k == d else g.randint(1, x.shape[k])) for k in range(x.rank))
    if numel(ishape) > 4 * g.max_numel:
        return None
    ix, bad = corrupt_index(g, g.index(x.shape[d], ishape, negative=False), x.shape[d])
    aux = {'ix': ix}
    return Step('gather', [i], f'r = x0.gather({dim}, ix)', f'r = torch.gather(x0, {dim}, ix)', aux=aux, cmp=Cmp(exact=True), supported=supported, invalid=bad)


def unique_index(g: Gen, shape: tuple[int, ...], d: int, ishape: tuple[int, ...]) -> torch.Tensor:
    full = list(ishape)
    full[d] = shape[d]
    perm = torch.rand(full, generator=g.g).argsort(dim=d)
    return perm.narrow(d, 0, ishape[d]).contiguous()


@op('scatter', 2.5)
def scatter(p: Program) -> Step | None:
    g = p.gen
    add = g.chance(0.5)
    sel = pick(p, NUMERIC if add else ANY, lambda v: v.rank >= 1 and v.numel > 0)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    dim = norm_dim(g, x.rank)
    d = dim % x.rank
    ishape = tuple(g.randint(1, x.shape[k]) for k in range(x.rank))
    ix, bad = corrupt_index(g, g.index(x.shape[d], ishape, negative=False) if add else unique_index(g, x.shape, d, ishape), x.shape[d])
    aux = {'ix': ix, 'src': g.tensor(ishape, x.dt)}
    name = 'scatter_add' if add else 'scatter'
    cmp = Cmp(exact=not add or x.dt not in FP)
    if add and x.dt in FP:
        cmp = Cmp(scale=f's = torch.scatter_add(x0.double().abs(), {dim}, ix, src.double().abs())', acc_k=2 * math.sqrt(numel(ishape)) + 8)
    ref = f'r = torch.{name}(x0, {dim}, ix, src)'
    if add and x.dt in FP:
        ref = f'r = torch.scatter_add(x0.double(), {dim}, ix, src.double()).to(x0.dtype)'
    if inplace_ok(x) and g.chance(0.4):
        return Step(f'{name}_', [i], f'r = x0.{name}_({dim}, ix, src)', ref, kind='inplace', aux=aux, cmp=cmp, supported=supported, invalid=bad)
    return Step(name, [i], f'r = x0.{name}({dim}, ix, src)', ref, aux=aux, cmp=cmp, supported=supported, invalid=bad)


@op('index_add', 1.5)
def index_add(p: Program) -> Step | None:
    g = p.gen
    sel = pick(p, NUMERIC, lambda v: v.rank >= 1 and v.numel > 0)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    dim = norm_dim(g, x.rank)
    d = dim % x.rank
    L = g.randint(0, 2 * x.shape[d])
    sshape = tuple(L if k == d else x.shape[k] for k in range(x.rank))
    alpha = g.choice((1, 1, 2, -1)) if x.dt in INTS else g.choice((1.0, 0.5, -2.0))
    if x.dt in UINTS:
        alpha = 1
    ix, bad = corrupt_index(g, g.index(x.shape[d], (L,), negative=False), x.shape[d])
    aux = {'ix': ix, 'src': g.tensor(sshape, x.dt)}
    ref = f'r = torch.index_add(x0, {dim}, ix, src, alpha={alpha})'
    cmp = Cmp(exact=x.dt not in FP)
    if x.dt in FP:
        ref = f'r = torch.index_add(x0.double(), {dim}, ix, src.double(), alpha={alpha}).to(x0.dtype)'
        cmp = Cmp(scale=f's = torch.index_add(x0.double().abs(), {dim}, ix, src.double().abs(), alpha={abs(alpha)})', acc_k=2 * math.sqrt(L + 1) + 8)
    if inplace_ok(x) and g.chance(0.4):
        return Step('index_add_', [i], f'r = x0.index_add_({dim}, ix, src, alpha={alpha})', ref, kind='inplace', aux=aux, cmp=cmp, supported=supported, invalid=bad)
    return Step('index_add', [i], f'r = x0.index_add({dim}, ix, src, alpha={alpha})', ref, aux=aux, cmp=cmp, supported=supported, invalid=bad)


@op('embedding', 1.5)
def embedding(p: Program) -> Step | None:
    g = p.gen
    i = p.pick(lambda v: v.rank >= 1 and v.shape[0] > 0)
    if i is None:
        return None
    x = p.pool[i]
    ishape = g.shape(max_rank=3, budget=256, allow_zero=True)
    if numel(ishape) * (x.numel // max(1, x.shape[0])) > 8 * g.max_numel:
        return None
    ix, bad = corrupt_index(g, g.index(x.shape[0], ishape, negative=False), x.shape[0])
    if bad:
        ix = ix.abs()  # torch's x[ix] wraps negatives; embedding indices must be in [0, vocab)
    return Step('embedding', [i], 'r = x0.embedding(ix)', 'r = x0[ix]', aux={'ix': ix}, cmp=Cmp(exact=True), upcast_ok=False, invalid=bad)


@op('triangular', 1.5)
def triangular(p: Program) -> Step | None:
    g = p.gen
    i = p.pick(lambda v: v.rank >= 2)
    if i is None:
        return None
    x = p.pool[i]
    name = g.choice(('tril', 'triu'))
    k = g.randint(-4, 4)
    ref = f'r = torch.{name}(x0, {k})'
    if inplace_ok(x) and g.chance(0.4):
        return Step(f'{name}_', [i], f'r = x0.{name}_({k})', ref, kind='inplace', cmp=Cmp(exact=True))
    return Step(name, [i], f'r = x0.{name}({k})', ref, cmp=Cmp(exact=True))


@op('repeat', 2.0)
def repeat(p: Program) -> Step | None:
    g = p.gen
    i = p.pick()
    if i is None:
        return None
    x = p.pool[i]
    if g.chance(0.5):
        reps = [g.choice((1, 1, 2, 3)) for _ in range(x.rank + g.randint(0, 2))]
        if g.chance(0.05) and reps:
            reps[g.randint(0, len(reps) - 1)] = 0
        if not reps or numel(reps) * x.numel > 4 * g.max_numel:
            return None
        args = ', '.join(map(str, reps))
        return Step('repeat', [i], f'r = x0.repeat({args})', f'r = x0.repeat({args})', cmp=Cmp(exact=True), upcast_ok=False)
    dim = norm_dim(g, x.rank) if x.rank and g.chance(0.7) else None
    n = x.shape[dim] if dim is not None else x.numel
    dkw = f', dim={dim}' if dim is not None else ''
    if g.chance(0.5) or n == 0:
        k = g.randint(0, 3)
        if k * x.numel > 4 * g.max_numel:
            return None
        return Step('repeat_interleave', [i], f'r = x0.repeat_interleave({k}{dkw})', f'r = torch.repeat_interleave(x0, {k}{dkw})', cmp=Cmp(exact=True), upcast_ok=False)
    aux = {'rp': torch.randint(0, 4, (n,), generator=g.g)}
    return Step('repeat_interleave(tensor)', [i], f'r = x0.repeat_interleave(rp{dkw})', f'r = torch.repeat_interleave(x0, rp{dkw})', aux=aux, cmp=Cmp(exact=True), upcast_ok=False)


@op('pad', 1.5)
def pad(p: Program) -> Step | None:
    g = p.gen
    i = p.pick(lambda v: v.rank >= 1)
    if i is None:
        return None
    x = p.pool[i]
    mode = g.choice(('constant', 'constant', 'reflect', 'replicate', 'circular'))
    k = g.randint(1, min(x.rank, 3))
    pads = []
    for j in range(k):
        size = x.shape[-1 - j]
        lim = size - 1 if mode == 'reflect' else size if mode == 'circular' else 4
        lo = -min(2, size) if mode == 'constant' and g.chance(0.2) else 0
        pads += [g.randint(lo, max(lo, lim)), g.randint(lo, max(lo, lim))]
    pads_t = tuple(pads)
    if mode == 'constant':
        val = scalar_for(g, x.dt)
        return Step('pad(constant)', [i], f"r = x0.pad({pads_t!r}, mode='constant', value={lit(val)})", f"r = F.pad(x0, {pads_t!r}, mode='constant', value={lit(val)})", cmp=Cmp(exact=True))
    # Torch only implements non-constant padding for some dtypes; float64 represents all others exactly.
    exact_in_f64 = x.dt not in (torch.float32, torch.int64, torch.uint64)
    ref = f"r = F.pad(x0.double(), {pads_t!r}, mode='{mode}').to(x0.dtype)" if exact_in_f64 else f"r = F.pad(x0, {pads_t!r}, mode='{mode}')"
    return Step(f'pad({mode})', [i], f"r = x0.pad({pads_t!r}, mode='{mode}')", ref, cmp=Cmp(exact=True))


@op('concat', 2.5)
def concat(p: Program) -> Step | None:
    g = p.gen
    i = p.pick()
    if i is None:
        return None
    x = p.pool[i]
    name = g.choice(('cat', 'cat', 'stack', 'hstack', 'vstack', 'dstack'))
    n = g.randint(0, 3)
    aux: dict[str, torch.Tensor] = {}
    names = ['x0']
    dim = 0
    if name == 'cat':
        if x.rank == 0:
            return None
        dim = norm_dim(g, x.rank)
        d = dim % x.rank
        for j in range(n):
            shape = tuple(g.randint(0, 5) if k == d else x.shape[k] for k in range(x.rank))
            names.append(maybe_permuted_aux(g, aux, f'a{j}', g.tensor(shape, x.dt)))
    else:
        for j in range(n):
            names.append(maybe_permuted_aux(g, aux, f'a{j}', g.tensor(x.shape, x.dt)))
        dim = g.randint(-x.rank - 1, x.rank) if name == 'stack' else 0
    g.rng.shuffle(names)
    lst = f'[{", ".join(names)}]'
    if name in ('cat', 'stack'):
        return Step(name, [i], f'r = Tensor.{name}({lst}, dim={dim})', f'r = torch.{name}({lst}, dim={dim})', aux=aux, cmp=Cmp(exact=True), upcast_ok=False)
    return Step(name, [i], f'r = Tensor.{name}({lst})', f'r = torch.{name}({lst})', aux=aux, cmp=Cmp(exact=True), upcast_ok=False)


@op('bincount', 0.8)
def bincount(p: Program) -> Step | None:
    g = p.gen
    n = g.randint(0, 300)
    dt = g.choice((torch.int64, torch.int32, torch.uint8, torch.int16))
    aux = {'a0': torch.randint(0, g.choice((1, 5, 50, 200)), (n,), generator=g.g).to(dt)}
    ml = g.choice((0, 0, 3, 100))
    if g.chance(0.4):
        aux['w'] = g.tensor((n,), g.choice((torch.float32, torch.float32, torch.float16)), specials=0)
        return Step('bincount(weights)', [], f'r = a0.bincount(weights=w, minlength={ml})', f'r = torch.bincount(a0, weights=w.double(), minlength={ml}).to(w.dtype)', aux=aux,
                    cmp=Cmp(check_dtype=False, scale=f's = torch.bincount(a0, weights=w.double().abs(), minlength={ml})', acc_k=2 * math.sqrt(n + 1) + 8))  # fmt: skip
    return Step('bincount', [], f'r = a0.bincount(minlength={ml})', f'r = torch.bincount(a0, minlength={ml})', aux=aux, cmp=Cmp(exact=True))


@op('nonzero', 1.0)
def nonzero(p: Program) -> Step | None:
    i = p.pick(lambda v: v.numel <= 20000)
    if i is None:
        return None
    return Step('nonzero', [i], 'r = x0.nonzero()', 'r = torch.nonzero(x0)', cmp=Cmp(exact=True))


@op('one_hot', 0.7)
def one_hot(p: Program) -> Step | None:
    g = p.gen
    c = g.randint(1, 12)
    shape = g.shape(max_rank=3, budget=512)
    aux = {'a0': torch.randint(0, c, shape, generator=g.g)}
    nc = c if g.chance(0.7) else -1
    if nc == -1 and not numel(shape):
        nc = c
    return Step('one_hot', [], f'r = a0.one_hot({nc})', f'r = F.one_hot(a0, {nc})', aux=aux, cmp=Cmp(exact=True, check_dtype=False))


# -- linear algebra / nn ----------------------------------------------------------------------------


@op('matmul', 5.0)
def matmul(p: Program) -> Step | None:
    g = p.gen
    sel = pick(p, FP, lambda v: v.rank >= 1 and v.numel <= 65536)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    k = x.shape[-1]
    if x.rank == 1 or g.chance(0.15):
        bshape_ = (k,) if g.chance(0.5) else (k, g.choice((1, 3, 8, 17, 32)))
    else:
        n = g.choice((1, 2, 3, 7, 8, 16, 31, 64))
        batch = list(x.shape[:-2])
        batch = [1 if g.chance(0.2) else b for b in batch]
        if batch and g.chance(0.3):
            batch = batch[g.randint(0, len(batch)) :]
        bshape_ = (*batch, k, n)
    if numel(bshape_) > 65536:
        return None
    aux: dict[str, torch.Tensor] = {}
    b = maybe_permuted_aux(g, aux, 'a0', g.tensor(bshape_, x.dt))
    mag = f'r = x0 @ {b}' if g.chance(0.5) else f'r = x0.matmul({b})'
    cmp = Cmp(scale=f's = torch.matmul(x0.double().abs(), {b}.double().abs())', acc_k=2 * math.sqrt(k + 1) + 8)
    return Step('matmul', [i], mag, f'r = torch.matmul(x0.double(), {b}.double()).to(x0.dtype)', aux=aux, cmp=cmp, supported=supported)


@op('outer', 0.8)
def outer(p: Program) -> Step | None:
    g = p.gen
    sel = pick(p, NUMERIC, lambda v: v.rank == 1)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    aux = {'a0': g.tensor((g.randint(0, 40),), x.dt)}
    return Step('outer', [i], 'r = x0.outer(a0)', 'r = binf(torch.outer, x0, a0)', aux=aux, supported=supported)


EINSUM = (
    ('ij,jk->ik', 'ij jk'), ('bij,bjk->bik', 'bij bjk'), ('ij->ji', 'ij'), ('ij->', 'ij'), ('ii->i', 'ii'), ('ii->', 'ii'),
    ('i,i->', 'i i'), ('i,j->ij', 'i j'), ('ij,ij->ij', 'ij ij'), ('ij,ij->', 'ij ij'), ('bhqd,bhkd->bhqk', 'bhqd bhkd'),
    ('ijk->kji', 'ijk'), ('ij,j->i', 'ij j'), ('ab,cd->abcd', 'ab cd'), ('bij,jk->bik', 'bij jk'), ('ijk,ikl->ijl', 'ijk ikl'),
)  # fmt: skip


@op('einsum', 1.5)
def einsum(p: Program) -> Step | None:
    g = p.gen
    eq, spec = g.choice(EINSUM)
    sizes = {c: g.choice((1, 2, 3, 4, 5, 8, 13)) for c in 'abcdhijklq'}
    dt = rand_dtype(g, FP) if g.chance(0.9) else rand_dtype(g, NUMERIC)
    aux = {f'a{j}': g.tensor(tuple(sizes[c] for c in s), dt) for j, s in enumerate(spec.split())}
    names = ', '.join(aux)
    dnames = ', '.join(f'{n}.double()' for n in aux)
    anames = ', '.join(f'{n}.double().abs()' for n in aux)
    ref = f"r = torch.einsum('{eq}', {dnames}).to(a0.dtype)" if dt in FP else f"r = torch.einsum('{eq}', {names})"
    return Step(f'einsum({eq})', [], f"r = Tensor.einsum('{eq}', {names})", ref, aux=aux,
                cmp=Cmp(scale=f"s = torch.einsum('{eq}', {anames})", acc_k=24), supported=dt in FP)  # fmt: skip


@op('softmax', 3.0)
def softmax(p: Program) -> Step | None:
    g = p.gen
    sel = pick(p, FP)
    if sel is None:
        return None
    i, supported = sel
    x = p.pool[i]
    dim = norm_dim(g, x.rank) if x.rank else 0
    ref = f'r = f64(x0, lambda v: softmax(v, {dim}))'
    if inplace_ok(x) and g.chance(0.3):
        return Step('softmax_', [i], f'r = x0.softmax_({dim})', ref, kind='inplace', supported=supported)
    return Step('softmax', [i], f'r = x0.softmax({dim})', ref, supported=supported)


@op('conv', 3.0)
def conv(p: Program) -> Step | None:
    g = p.gen
    nd = g.choice((1, 1, 2, 2, 2, 3))
    transposed = g.chance(0.35)
    dt = rand_dtype(g, FP)
    groups = g.choice((1, 1, 1, 2, 3))
    cin, cout = groups * g.randint(1, 3), groups * g.randint(1, 3)
    n = g.randint(1, 2)
    lim = {1: 24, 2: 12, 3: 6}[nd]
    spatial = [g.randint(1, lim) for _ in range(nd)]
    ks = [g.randint(1, 4) for _ in range(nd)]
    stride = [g.choice((1, 1, 2, 3)) for _ in range(nd)]
    padding = [g.randint(0, 2) for _ in range(nd)]
    dilation = [g.choice((1, 1, 2)) for _ in range(nd)]
    outpad = [g.randint(0, max(stride[j], dilation[j]) - 1) for j in range(nd)] if transposed else None

    def arg(v: list[int]) -> str:
        return str(v[0]) if len(set(v)) == 1 and g.chance(0.5) else repr(tuple(v))

    i_sel = p.pick(lambda v: v.rank == nd + 2 and v.dt in FP and v.numel <= 20000 and v.shape[1] > 0)
    args: list[int] = []
    aux: dict[str, torch.Tensor] = {}
    if i_sel is not None and g.chance(0.4):
        x = p.pool[i_sel]
        dt = x.dt
        cin = x.shape[1]
        if cin % groups:
            groups = 1
        cout = groups * g.randint(1, 3)
        args = [i_sel]
        xs = 'x0'
    else:
        aux['x'] = g.tensor((n, cin, *spatial), dt)
        xs = 'x'
    if not transposed:
        sp = p.pool[args[0]].shape[2:] if args else spatial
        if any(sp[j] + 2 * padding[j] - dilation[j] * (ks[j] - 1) - 1 < 0 for j in range(nd)):
            return None
    wshape = (cin, cout // groups, *ks) if transposed else (cout, cin // groups, *ks)
    aux['w'] = g.tensor(wshape, dt)
    bias = g.chance(0.6)
    if bias:
        aux['b'] = g.tensor((cout,), dt)
    b = 'b' if bias else 'None'
    bd = 'b.double()' if bias else 'None'
    ba = 'b.double().abs()' if bias else 'None'
    st, pd, dl = arg(stride), arg(padding), arg(dilation)
    if transposed:
        op_ = arg(outpad)
        mag = f'r = {xs}.convT{nd}D(w, {b}, stride={st}, padding={pd}, output_padding={op_}, groups={groups}, dilation={dl})'
        fn = f'F.conv_transpose{nd}d'
        tail = f'stride={st}, padding={pd}, output_padding={op_}, groups={groups}, dilation={dl}'
    else:
        mag = f'r = {xs}.conv{nd}D(w, {b}, stride={st}, padding={pd}, dilation={dl}, groups={groups})'
        fn = f'F.conv{nd}d'
        tail = f'stride={st}, padding={pd}, dilation={dl}, groups={groups}'
    ref = f'r = {fn}({xs}.double(), w.double(), {bd}, {tail}).to(w.dtype)'
    red = (cin // groups) * math.prod(ks)
    cmp = Cmp(scale=f's = {fn}({xs}.double().abs(), w.double().abs(), {ba}, {tail})', acc_k=2 * math.sqrt(red) + 8)
    return Step(f'conv{"T" if transposed else ""}{nd}d', args, mag, ref, aux=aux, cmp=cmp)


@op('interpolate', 2.0)
def interpolate(p: Program) -> Step | None:
    g = p.gen
    nd = g.choice((1, 2, 2, 3))
    modes = {1: ('nearest', 'nearest-exact', 'linear', 'area'), 2: ('nearest', 'nearest-exact', 'bilinear', 'bicubic', 'area'), 3: ('nearest', 'nearest-exact', 'trilinear', 'area')}[nd]
    mode = g.choice(modes)
    dt = rand_dtype(g, FP) if mode not in ('nearest', 'nearest-exact') or g.chance(0.6) else rand_dtype(g, NUMERIC)
    lim = {1: 20, 2: 10, 3: 5}[nd]
    shape = (g.randint(1, 2), g.randint(1, 3), *[g.randint(1, lim) for _ in range(nd)])
    aux = {'x': g.tensor(shape, dt)}
    if g.chance(0.5):
        size = tuple(g.randint(1, 2 * lim) for _ in range(nd))
        kw = f'size={size!r}'
    else:
        sf = g.choice((0.5, 1.5, 2.0, 3.0, 0.75)) if g.chance(0.6) else tuple(g.choice((0.5, 1.5, 2.0, 3.0)) for _ in range(nd))
        kw = f'scale_factor={sf!r}'
    linear = mode in ('linear', 'bilinear', 'bicubic', 'trilinear')
    ac = g.chance(0.5) if linear else False
    aa = g.chance(0.3) if mode in ('bilinear', 'bicubic') else False
    mag = f"r = x.interpolate({kw}, mode='{mode}', align_corners={ac}, antialias={aa})"
    ref = f"r = F.interpolate(x.double(), {kw}, mode='{mode}', align_corners={ac if linear else None}, antialias={aa}).to(x.dtype)"
    cmp = Cmp(scale=f"s = F.interpolate(x.double().abs(), {kw}, mode='{mode}', align_corners={ac if linear else None}, antialias={aa})", acc_k=16)
    if dt not in FP:
        cmp = Cmp(exact=True)
    return Step(f'interpolate({mode})', [], mag, ref, aux=aux, cmp=cmp)


# -- random number generators: no reference values, check invariants only ------------------------


@op('random', 2.0)
def random_ops(p: Program) -> Step | None:
    g = p.gen
    which = g.choice(('uniform_', 'normal_', 'bernoulli_', 'uniform', 'multinomial', 'rand_perm'))
    if which in ('uniform_', 'normal_', 'bernoulli_'):
        mask = BOOL if which == 'bernoulli_' else FP
        i = p.pick(lambda v: v.dt in mask and inplace_ok(v))
        if i is None:
            return None
        if which == 'uniform_':
            lo = g.choice((0.0, -1.0, -3.0, 2.0))
            hi = lo + g.choice((1.0, 0.5, 4.0))

            def chk(mo: list, ins: list) -> str | None:
                x = mo[0].double()
                if x.numel() and not bool(((x >= lo) & (x <= hi)).all()):
                    return f'uniform_({lo}, {hi}) produced values outside range: min={x.min().item()} max={x.max().item()}'
                return None

            return Step('uniform_', [i], f'r = x0.uniform_({lo}, {hi})', '', kind='smoke_inplace', smoke=chk)
        if which == 'normal_':
            mean, std = g.choice((0.0, 1.0, -2.0)), g.choice((1.0, 0.1, 3.0))

            def chk(mo: list, ins: list) -> str | None:
                x = mo[0].double()
                if x.numel() and not bool(torch.isfinite(x).all()):
                    return 'normal_ produced non-finite values'
                if x.numel() > 2000 and abs(x.mean().item() - mean) > 6 * std / math.sqrt(x.numel()) + 0.05:
                    return f'normal_ mean {x.mean().item():.4g} far from {mean}'
                return None

            return Step('normal_', [i], f'r = x0.normal_({mean}, {std})', '', kind='smoke_inplace', smoke=chk)
        return Step('bernoulli_', [i], f'r = x0.bernoulli_({g.choice((0.0, 0.3, 0.5, 1.0))})', '', kind='smoke_inplace')
    if which == 'uniform':
        dt = rand_dtype(g, FP)
        shape = g.shape()

        def chk(mo: list, ins: list) -> str | None:
            x = mo[0].double()
            if tuple(mo[0].shape) != shape:
                return f'uniform shape {tuple(mo[0].shape)} != {shape}'
            if x.numel() and not bool(((x >= -1) & (x <= 1)).all()):
                return 'uniform(-1, 1) out of range'
            return None

        return Step('uniform', [], f'r = Tensor.uniform({", ".join(map(str, shape))}, low=-1.0, high=1.0, dtype={mdt(dt)})' if shape else f'r = Tensor.uniform(low=-1.0, high=1.0, dtype={mdt(dt)})', '', kind='smoke', smoke=chk)
    if which == 'multinomial':
        c = g.randint(1, 20)
        rows = g.choice((None, 1, 3))
        shape = (c,) if rows is None else (rows, c)
        probs = torch.rand(shape, generator=g.g) * (torch.rand(shape, generator=g.g) < 0.7)
        probs[..., 0] += 0.1
        k = g.randint(1, c)
        repl = g.chance(0.5)
        if repl:
            k = g.randint(1, 30)
        nz = int((probs > 0).sum(-1).min())
        if not repl and k > nz:
            k = nz
        aux = {'pr': probs.to(rand_dtype(g, frozenset({torch.float32, torch.float16, torch.bfloat16})))}

        def chk(mo: list, ins: list, aux_pr: torch.Tensor = aux['pr']) -> str | None:
            s = mo[0]
            if tuple(s.shape) != (*shape[:-1], k):
                return f'multinomial shape {tuple(s.shape)} expected {(*shape[:-1], k)}'
            if s.numel() and (int(s.min()) < 0 or int(s.max()) >= c):
                return 'multinomial index out of range'
            if s.numel() and not bool((aux_pr.float().gather(-1, s.long()) > 0).all()):
                return 'multinomial sampled a zero-probability class'
            if not repl and s.numel():
                srt = torch.sort(s.long(), -1).values
                if bool((srt[..., 1:] == srt[..., :-1]).any()):
                    return 'multinomial without replacement returned duplicates'
            return None

        return Step('multinomial', [], f'r = pr.multinomial({k}, replacement={repl})', '', kind='smoke', aux=aux, smoke=chk)
    n = g.randint(0, 500)
    dt = rand_dtype(g, frozenset({torch.int64, torch.int32}))

    def chk(mo: list, ins: list) -> str | None:
        s = mo[0].long()
        if tuple(s.shape) != (n,) or not torch.equal(torch.sort(s).values, torch.arange(n)):
            return f'rand_perm({n}) is not a permutation'
        return None

    return Step('rand_perm', [], f'r = Tensor.rand_perm({n}, dtype={mdt(dt)})', '', kind='smoke', smoke=chk)


def weighted_ops(include: str | None, exclude: str | None) -> list[tuple[str, float, Generator]]:
    import re

    out = []
    for name, w, fn in OPS:
        if include and not re.search(include, name):
            continue
        if exclude and re.search(exclude, name):
            continue
        out.append((name, w, fn))
    return out

