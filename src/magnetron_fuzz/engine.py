# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Execution of generated op sequences on magnetron and torch, with differential checking."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from magnetron import Tensor, dtype as D

from . import refs
from .dtypes import EPS, FP, ITEMSIZE, MAG2TORCH, TOLS, TORCH_WEAK, to_mag, to_torch, upcast_dtype
from .gen import Gen

MAG_NS: dict[str, object] = {'Tensor': Tensor, 'D': D, 'math': math}
REF_NS: dict[str, object] = {'torch': torch, 'math': math, **{k: getattr(refs, k) for k in refs.__all__}}


@dataclass
class Cmp:
    exact: bool = False  # Selection ops (sort, max, gather, views): values must be bit-identical.
    rtol_mul: float = 1.0
    scale: str | None = None  # Torch code over x0.. yielding `s`, the per-element magnitude of an accumulation.
    acc_k: float = 8.0  # Error allowance in units of eps * s.
    check_dtype: bool = True
    mask: str | None = None  # Torch code over x0.. yielding bool `m`: only compare where m is True.


@dataclass
class Step:
    op: str  # Name used for stats and finding signatures.
    args: list[int]  # Pool indices bound as x0, x1, ...
    mag: str  # Python statement(s) on magnetron tensors, must assign `r`.
    ref: str  # Torch statement(s), must assign `r`. For inplace steps: the functional result.
    kind: str = 'new'  # new | view | inplace | smoke | smoke_inplace
    aux: dict[str, torch.Tensor] = field(default_factory=dict)  # Extra inputs (indices, masks), bound by name.
    cmp: Cmp = field(default_factory=Cmp)
    check: Callable | None = None  # check(mag_outs, ref_outs, ins_t) -> str | None, replaces value comparison.
    smoke: Callable | None = None  # For kind=smoke: invariant check(mag_outs, ins_t) -> str | None.
    supported: bool = True  # Inside magnetron's dtype mask: a magnetron error is a finding.
    keep: str = 'all'  # all | one | none: which outputs enter the pool.
    upcast_ok: bool = True
    invalid: bool = False  # Deliberately invalid input (e.g. out-of-range index): magnetron must raise, not succeed.


class Var:
    __slots__ = ('m', 'name', 't')

    def __init__(self, m: Tensor, t: torch.Tensor, name: str) -> None:
        self.m = m
        self.t = t
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.t.shape)

    @property
    def dt(self) -> torch.dtype:
        return self.t.dtype

    @property
    def rank(self) -> int:
        return self.t.dim()

    @property
    def numel(self) -> int:
        return self.t.numel()

    def mstore(self) -> int:
        p = self.m.data_storage_ptr if self.m.numel else 0
        return p or id(self.m)

    def tstore(self) -> int:
        p = self.t.untyped_storage().data_ptr() if self.t.numel() else 0
        return p or id(self.t)

    def overlapping(self) -> bool:
        """Some element is reachable through more than one index (expanded views)."""
        return any(s == 0 and d > 1 for s, d in zip(self.m.strides, self.m.shape)) or any(
            s == 0 and d > 1 for s, d in zip(self.t.stride(), self.t.shape)
        )

    def layout(self) -> str:
        if self.numel == 0:
            return 'empty'
        if self.rank == 0:
            return 'scalar'
        st = self.m.strides
        if any(s < 0 for s in st):
            return 'negstride'
        if self.overlapping():
            return 'bcast'
        return 'contig' if self.m.is_contiguous else 'strided'

    def describe(self) -> str:
        return f'{self.name}: {str(self.dt).removeprefix("torch.")}{list(self.shape)} strides={tuple(self.m.strides)} {self.layout()}'


class CaseAbort(Exception):
    """Raised by hooks to stop the current case (e.g. replay reached its target step)."""


def _flat(r: object) -> list:
    if isinstance(r, (tuple, list)):
        out = []
        for x in r:
            out.extend(_flat(x))
        return out
    return [r]


def _upcast(t: torch.Tensor) -> torch.Tensor:
    return t.to(upcast_dtype(t.dtype)) if t.dtype in TORCH_WEAK else t


def _cmp_view(t: torch.Tensor) -> torch.Tensor:
    """Representation where ==, gather, sort work for every dtype."""
    if t.dtype in FP:
        return t.double()
    if t.dtype == torch.uint64:
        return t.view(torch.int64)
    if t.dtype in TORCH_WEAK:
        return t.to(torch.int64)
    return t


def values_equal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _cmp_view(a), _cmp_view(b)
    eq = a == b
    if a.dtype.is_floating_point:
        eq |= torch.isnan(a) & torch.isnan(b)
    return eq


def _fmt_bad(got: torch.Tensor, exp: torch.Tensor, eq: torch.Tensor) -> str:
    bad = (~eq).nonzero()
    n = bad.shape[0]
    lines = [f'{n}/{eq.numel()} elements differ']
    g, e = _cmp_view(got), _cmp_view(exp)
    if g.dtype.is_floating_point and n:
        d = (g - e).abs()[~eq]
        d = d[torch.isfinite(d)]
        if d.numel():
            lines.append(f'max abs diff {d.max().item():.6g}')
    for idx in bad[:6].tolist():
        ix = tuple(idx)
        lines.append(f'  at {list(ix)}: magnetron={g[ix].item()!r} torch={e[ix].item()!r}')
    return '\n'.join(lines)


def compare(got: torch.Tensor, exp: torch.Tensor, cmp: Cmp, *, upcast: bool, scale: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> tuple[str, str] | None:
    """(category, message) on mismatch, else None."""
    if tuple(got.shape) != tuple(exp.shape):
        return 'shape', f'shape mismatch: magnetron {tuple(got.shape)} vs torch {tuple(exp.shape)}'
    if got.dtype != exp.dtype:
        if cmp.check_dtype and not upcast:
            return 'dtype', f'dtype mismatch: magnetron {got.dtype} vs torch {exp.dtype}'
        exp = exp.to(got.dtype)
    if got.numel() == 0:
        return None
    if got.dtype not in FP or cmp.exact:
        eq = values_equal(got, exp)
    else:
        a, b = got.double(), exp.double()
        rtol, atol = TOLS[got.dtype]
        tol = (atol + rtol * b.abs()) * cmp.rtol_mul
        if scale is not None:
            # A NaN magnitude (NaN somewhere in the inputs) means no usable bound: be lenient there.
            tol = tol + cmp.acc_k * EPS[got.dtype] * torch.nan_to_num(scale.double(), nan=math.inf)
        close = (((a - b).abs() <= tol) & torch.isfinite(a) & torch.isfinite(b)) | (a == b)
        eq = close | (torch.isnan(a) & torch.isnan(b)) | ((a == b) & torch.isinf(b))
    if mask is not None:
        eq = eq | ~mask
    if bool(eq.all()):
        return None
    return 'mismatch', _fmt_bad(got, exp, eq)


@dataclass
class Outcome:
    status: str  # ok | mag_err | ref_err | both_err | finding | unsupported
    category: str | None = None
    message: str | None = None


class Program:
    """One fuzz case: a pool of (magnetron, torch) tensor pairs evolved by random steps."""

    def __init__(self, gen: Gen, reporter, hooks=None, max_pool: int = 10) -> None:  # noqa: ANN001
        self.gen = gen
        self.reporter = reporter
        self.hooks = hooks
        self.pool: list[Var] = []
        self.max_pool = max_pool
        self.trace: list[str] = []
        self.counter = 0
        self.step_idx = 0

    # -- pool -------------------------------------------------------------------------------------

    def fresh_name(self) -> str:
        self.counter += 1
        return f'v{self.counter}'

    def add(self, m: Tensor, t: torch.Tensor) -> Var:
        v = Var(m, t, self.fresh_name())
        self.pool.append(v)
        if len(self.pool) > self.max_pool:
            del self.pool[self.gen.randint(0, len(self.pool) - 2)]
        return v

    def pick(self, pred: Callable[[Var], bool] | None = None, exclude: tuple[int, ...] = ()) -> int | None:
        idx = [i for i, v in enumerate(self.pool) if i not in exclude and (pred is None or pred(v))]
        return self.gen.choice(idx) if idx else None

    def family(self, v: Var) -> tuple[list[Var], list[Var]]:
        """(consistent, divergent): vars aliasing v on both sides vs only on one side."""
        ms, ts = v.mstore(), v.tstore()
        same, diff = [], []
        for w in self.pool:
            a, b = w.mstore() == ms, w.tstore() == ts
            if a and b:
                same.append(w)
            elif a or b:
                diff.append(w)
        return same, diff

    def drop(self, vs: list[Var]) -> None:
        ids = {id(v) for v in vs}
        self.pool = [w for w in self.pool if id(w) not in ids]

    # -- execution --------------------------------------------------------------------------------

    def _code(self, ins: list[Var], code: str) -> str:
        names = [v.name for v in ins]
        return re.sub(r'\bx(\d+)\b', lambda mo: names[int(mo.group(1))] if int(mo.group(1)) < len(names) else mo.group(0), code)

    def run(self, step: Step) -> Outcome:
        ins = [self.pool[i] for i in step.args]
        inplace = step.kind == 'inplace'
        if step.kind in ('inplace', 'smoke_inplace'):
            _, diff = self.family(ins[0])
            self.drop(diff)
        self.trace.append(f'[{self.step_idx}] {self._code(ins, step.mag)}'.replace('\n', '; '))
        if self.hooks:
            self.hooks.before_mag(self, step, ins)

        # magnetron first: a crash here is attributed to this step by the supervisor.
        merr = None
        mouts_t: list = []
        mouts: list = []
        try:
            ns = dict(MAG_NS)
            for j, v in enumerate(ins):
                ns[f'x{j}'] = v.m
            for k, t in step.aux.items():
                ns[k] = to_mag(t)
            exec(step.mag, ns)
            mouts = _flat(ns['r'])
            mouts_t = [to_torch(o) if isinstance(o, Tensor) else o for o in mouts]
        except Exception as e:  # noqa: BLE001
            merr = e
        if self.hooks:
            self.hooks.after_mag(self, step)

        if step.kind.startswith('smoke'):
            return self._finish_smoke(step, ins, merr, mouts, mouts_t)

        # torch reference
        rerr, upcast = None, False
        routs: list = []
        ins_t = [v.t for v in ins]
        try:
            routs = self._ref(step, ins_t, step.aux, upcast=False)
        except Exception as e:  # noqa: BLE001
            rerr = e
            weak = any(t.dtype in TORCH_WEAK for t in ins_t) or any(t.dtype in TORCH_WEAK for t in step.aux.values())
            if weak and step.upcast_ok:
                try:
                    routs = self._ref(step, [_upcast(t) for t in ins_t], {k: _upcast(t) for k, t in step.aux.items()}, upcast=True)
                    rerr, upcast = None, True
                except Exception as e2:  # noqa: BLE001
                    rerr = e2
        if not rerr and inplace:
            rerr = self._inplace_ref_ok(ins[0], routs, upcast, mouts_t)

        if merr is not None and rerr is not None:
            if inplace:
                self._check_family(step, ins[0], 'partial_write', 'in-place op raised but modified data')
            return Outcome('both_err')
        if merr is not None:
            if inplace:
                self._check_family(step, ins[0], 'partial_write', 'in-place op raised but modified data')
            if not step.supported:
                return Outcome('unsupported')
            msg = f'{type(merr).__name__}: {merr}'
            self.reporter.finding('mag_error', step, ins, msg, self)
            return Outcome('finding', 'mag_error', msg)
        if rerr is not None:
            if inplace:
                self.drop(self.family(ins[0])[0])
            if step.invalid:
                msg = f'magnetron accepted invalid input that torch rejects ({type(rerr).__name__}: {str(rerr).splitlines()[0][:200]})'
                self.reporter.finding('accepts_invalid', step, ins, msg, self)
                return Outcome('finding', 'accepts_invalid', msg)
            return Outcome('ref_err')

        res = self._compare(step, ins, mouts_t, routs, upcast)
        if res is not None:
            cat, msg = res
            self.reporter.finding(cat, step, ins, msg, self)
            if inplace:
                self.drop(self.family(ins[0])[0])
            return Outcome('finding', cat, msg)

        if inplace:
            self._apply_inplace(step, ins[0], mouts_t[0])
        else:
            self._add_outputs(step, ins, mouts, mouts_t, routs)
        return Outcome('ok')

    def _ref(self, step: Step, ins_t: list[torch.Tensor], aux: dict[str, torch.Tensor], upcast: bool) -> list:
        ns = dict(REF_NS)
        for j, t in enumerate(ins_t):
            ns[f'x{j}'] = t
        ns.update(aux)
        exec(step.ref, ns)
        outs = _flat(ns['r'])
        return outs

    def _inplace_ref_ok(self, target: Var, routs: list, upcast: bool, mouts_t: list) -> Exception | None:
        """Torch rejects in-place ops whose result can't be stored in the target; mirror that."""
        r = routs[0]
        if not isinstance(r, torch.Tensor) or tuple(r.shape) != target.shape:
            return RuntimeError('in-place result shape differs from target')
        if not upcast and r.dtype != target.dt:
            return RuntimeError(f'in-place result type {r.dtype} cannot be stored in {target.dt}')
        if upcast and mouts_t and isinstance(mouts_t[0], torch.Tensor) and mouts_t[0].dtype != target.dt:
            return RuntimeError('in-place result type differs from target')
        return None

    def _compare(self, step: Step, ins: list[Var], mouts_t: list, routs: list, upcast: bool) -> tuple[str, str] | None:
        if len(mouts_t) != len(routs):
            return 'structure', f'magnetron returned {len(mouts_t)} outputs, torch {len(routs)}'
        for i, (a, b) in enumerate(zip(mouts_t, routs)):
            if isinstance(a, torch.Tensor) != isinstance(b, torch.Tensor):
                return 'structure', f'output {i}: magnetron {type(a).__name__} vs torch {type(b).__name__}'
        if step.check is not None:
            res = step.check(mouts_t, routs, [v.t for v in ins])
            if not res:
                return None
            return res if isinstance(res, tuple) else ('mismatch', res)
        scale = mask = None
        if step.cmp.scale or step.cmp.mask:
            ns = dict(REF_NS)
            for j, v in enumerate(ins):
                ns[f'x{j}'] = _upcast(v.t)
            ns.update({k: _upcast(t) for k, t in step.aux.items()})
            if step.cmp.scale:
                exec(step.cmp.scale, ns)
                scale = ns['s']
            if step.cmp.mask:
                exec(step.cmp.mask, ns)
                mask = ns['m']
        for i, (a, b) in enumerate(zip(mouts_t, routs)):
            if not isinstance(a, torch.Tensor):
                if a != b:
                    return 'mismatch', f'output {i}: magnetron {a!r} vs torch {b!r}'
                continue
            res = compare(a, b, step.cmp, upcast=upcast, scale=scale if i == 0 else None, mask=mask if i == 0 else None)
            if res:
                cat, msg = res
                return cat, (f'output {i}: ' if len(mouts_t) > 1 else '') + msg
        return None

    def _add_outputs(self, step: Step, ins: list[Var], mouts: list, mouts_t: list, routs: list) -> None:
        if step.keep == 'none':
            return
        pairs = [(m, mt, r) for m, mt, r in zip(mouts, mouts_t, routs) if isinstance(m, Tensor)]
        if step.keep == 'one' and pairs:
            pairs = [self.gen.choice(pairs)]
        in_stores = {v.t.untyped_storage().data_ptr() for v in ins if v.numel}
        for m, mt, r in pairs:
            if r.numel() and r.untyped_storage().data_ptr() in in_stores and r.dtype == mt.dtype:
                t = r  # torch view: keep aliasing so later in-place ops propagate like in magnetron.
            else:
                t = mt  # fresh result: resync to magnetron's bits so later steps see identical inputs.
            self.add(m, t)

    def _apply_inplace(self, step: Step, target: Var, r: torch.Tensor) -> None:
        # r is magnetron's (already verified) result: adopting its bits keeps both sides identical.
        with torch.no_grad():
            target.t.copy_(r)
        self._check_family(step, target, 'alias', 'in-place update not reflected consistently across aliasing tensors')

    def _check_family(self, step: Step, target: Var, category: str, what: str) -> None:
        same, _ = self.family(target)
        for v in same:
            try:
                got = to_torch(v.m)
            except Exception as e:  # noqa: BLE001
                self.reporter.finding(category, step, [target], f'{what}: reading {v.name} raised {e}', self)
                self.drop(same)
                return
            res = compare(got, v.t, Cmp(exact=True), upcast=False)
            if res:
                who = f'{v.name} (aliases {target.name})' if v is not target else v.name
                self.reporter.finding(category, step, [target], f'{what}: {who}: {res[1]}', self)
                self.drop(same)
                return

    def _finish_smoke(self, step: Step, ins: list[Var], merr: Exception | None, mouts: list, mouts_t: list) -> Outcome:
        if merr is not None:
            if not step.supported:
                return Outcome('unsupported')
            msg = f'{type(merr).__name__}: {merr}'
            self.reporter.finding('mag_error', step, ins, msg, self)
            return Outcome('finding', 'mag_error', msg)
        msg = step.smoke(mouts_t, [v.t for v in ins]) if step.smoke else None
        if msg:
            self.reporter.finding('invariant', step, ins, msg, self)
            return Outcome('finding', 'invariant', msg)
        if step.kind == 'smoke_inplace':
            # In-place random fill: adopt magnetron's values on the torch side, then verify neighbours are untouched.
            target = ins[0]
            with torch.no_grad():
                target.t.copy_(to_torch(target.m))
            self._check_family(step, target, 'alias', 'random fill wrote outside its view')
        elif step.keep != 'none':
            for m, mt in zip(mouts, mouts_t):
                if isinstance(m, Tensor):
                    self.add(m, mt)
        return Outcome('ok')

    def check_all(self, why: str) -> None:
        """Re-read every pool tensor from magnetron and compare: catches late corruption of unrelated tensors."""
        for v in list(self.pool):
            try:
                got = to_torch(v.m)
            except Exception as e:  # noqa: BLE001
                self.reporter.finding('corruption', None, [v], f'{why}: reading {v.name} raised {e}', self)
                self.drop([v])
                continue
            res = compare(got, v.t, Cmp(exact=True), upcast=False)
            if res:
                self.reporter.finding('corruption', None, [v], f'{why}: {v.name} changed unexpectedly: {res[1]}', self)
                self.drop(self.family(v)[0] + [v])


def item_size(t: torch.dtype) -> int:
    return ITEMSIZE[t]


__all__ = ['MAG2TORCH', 'Cmp', 'Outcome', 'Program', 'Step', 'Var', 'compare', 'values_equal']
