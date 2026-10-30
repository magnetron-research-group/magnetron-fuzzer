# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Deduplicated findings on disk, each with a replay command and a standalone single-op repro script."""

from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
import time
from pathlib import Path

import torch
from magnetron import Tensor

from .dtypes import ITEMSIZE, TORCH2MAG, to_torch

MAX_EXAMPLES = 3

# Deliberate differences from torch semantics, reported under 'api_diff' so they don't hide real bugs.
# Each entry: (category, op regex, message regex).
KNOWN_API_DIFFS = [
    ('mag_error', r'^topk$', r'k must be > |requires a tensor with rank'),
    ('mag_error', r'^transpose$', r'must be different|requires rank'),
    ('mag_error', r'^(getitem|setitem)$', r'incompatible function arguments'),
    ('mag_error', r'^(copy_|setitem)$', r'must have the same (shape|dtype)'),
    ('mag_error', r'_$', r'in-place (.* )?(would change the dtype|true division is not allowed)'),
    ('mag_error', r'^convT', r'computed output size 0'),  # torch skips the size check for an empty batch
    ('dtype', r'^(sum|prod|cusum|cuprod|einsum.*)$', r'magnetron torch.uint64 vs torch torch.int64'),  # unsigned reductions stay unsigned
    ('mag_error', r'^(arange|linspace)$', r'empty or invalid range|steps must be'),
    ('mag_error', r'^broadcast$', r'expected at least one dimension'),
]


def api_diff(category: str, op: str, msg: str, ins: list, step) -> bool:  # noqa: ANN001
    if category in ('dtype', 'mismatch') and step is not None:
        ops = [(v.rank, v.dt) for v in ins] + [(t.dim(), t.dtype) for t in step.aux.values()]
        dts = {d for _, d in ops}
        ranks = [r for r, _ in ops]
        # torch: 0-d tensors don't take part in type promotion against dimensioned tensors
        if len(dts) > 1 and 0 in ranks and any(r > 0 for r in ranks):
            return True
        # mixed-dtype comparisons: magnetron and torch promote int/half pairs differently (no NaN/precision issue on one side)
        if category == 'mismatch' and len(dts) > 1 and re.fullmatch(r'r?(lt|le|gt|ge|eq|ne)(@\w+)?', op):
            return True
    base_op = op.split('@', 1)[0]  # 'sum@cuda' follows the same rules as 'sum'
    return any(c == category and re.search(o, base_op) and re.search(m, msg) for c, o, m in KNOWN_API_DIFFS)


def slug(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s).strip('_')[:60] or 'op'


def normalize_message(msg: str) -> str:
    """Drop numbers, addresses and shapes so the same failure on different inputs dedupes."""
    first = msg.strip().splitlines()[0] if msg.strip() else ''
    first = re.sub(r'0x[0-9a-fA-F]+', 'X', first)
    first = re.sub(r'\(([^()]*\.c):\d+', r'(\1', first)
    first = re.sub(r'-?\d+(\.\d+)?(e[-+]?\d+)?', 'N', first)
    return first[:200]


def replay_command(seed: str, threads: int, device: str = 'cpu') -> str:
    dev = f' --device {device}' if device != 'cpu' else ''
    return f'MAG_LAUNCH_BLOCKING=1 MAG_CPU_THREADS={threads} .venv/bin/python -m magnetron_fuzz --replay {seed}{dev} -v'


def append_index(run_dir: Path, rec: dict) -> None:
    line = json.dumps(rec) + '\n'
    fd = os.open(run_dir / 'index.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())  # O_APPEND writes of one small line are atomic across workers.
    finally:
        os.close(fd)


def claim_example_slot(d: Path) -> int | None:
    """Create the finding dir if needed; return an example number if fewer than MAX_EXAMPLES exist."""
    d.mkdir(parents=True, exist_ok=True)
    for k in range(MAX_EXAMPLES):
        try:
            fd = os.open(d / f'.slot{k}', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return k
        except FileExistsError:
            continue
    return None


class Reporter:
    def __init__(self, run_dir: Path | None, seed: str, threads: int, verbose: bool = False, worker: int = -1, device: str = 'cpu') -> None:
        self.device = device
        self.run_dir = run_dir
        self.seed = seed
        self.threads = threads
        self.verbose = verbose
        self.worker = worker
        self.count = 0
        self.by_category: dict[str, int] = {}

    def finding(self, category: str, step, ins: list, msg: str, prog) -> None:  # noqa: ANN001
        op = step.op if step is not None else 'pool'
        if api_diff(category, op, msg, ins, step):
            category = 'api_diff'
        self.count += 1
        self.by_category[category] = self.by_category.get(category, 0) + 1
        dts = [str(v.dt).removeprefix('torch.') for v in ins]
        if step is not None:
            dts += [str(t.dtype).removeprefix('torch.') for t in step.aux.values()]
        # Coarse on purpose: one finding per distinct failure, not per input layout.
        detail = normalize_message(msg) if category in ('mag_error', 'api_diff', 'dtype', 'shape', 'structure', 'invariant', 'accepts_invalid') else ''
        sig = f'{category}|{op}|{dts[0] if dts else "-"}|{detail}'
        key = hashlib.sha1(sig.encode()).hexdigest()[:10]
        if self.verbose:
            print(f'\n=== {category} in {op} (step {prog.step_idx}) ===\n{msg}\n')
            for v in ins:
                print('  ' + v.describe())
            if step is not None:
                print('magnetron:', step.mag.replace('\n', '; '))
                print('torch    :', step.ref.replace('\n', '; '))
        if self.run_dir is None:
            return
        d = self.run_dir / 'findings' / category / f'{slug(op)}-{key}'
        k = claim_example_slot(d)
        append_index(self.run_dir, {'t': time.time(), 'sig': sig, 'key': key, 'category': category, 'op': op, 'seed': self.seed, 'step': prog.step_idx,
                                    'threads': self.threads, 'worker': self.worker, 'msg': msg.splitlines()[0][:300] if msg else ''})  # fmt: skip
        if k is None:
            return
        stem = f'ex{k}'
        write_report(d / f'{stem}.md', category, op, sig, msg, self.seed, prog.step_idx, self.threads, ins, step, prog.trace, self.device)
        if step is not None:
            try:
                write_repro(d, stem, category, step, ins, replay_command(self.seed, self.threads, self.device))
            except Exception as e:  # noqa: BLE001
                (d / f'{stem}.repro_error.txt').write_text(f'could not write repro: {type(e).__name__}: {e}\n')


def write_report(path: Path, category: str, op: str, sig: str, msg: str, seed: str, step_idx: int, threads: int, ins: list, step, trace: list[str], device: str = 'cpu') -> None:  # noqa: ANN001
    lines = [
        f'# {category}: {op}',
        '',
        f'- signature: `{sig}`',
        f'- seed: `{seed}`, step {step_idx}, MAG_CPU_THREADS={threads}',
        f'- replay: `{replay_command(seed, threads, device)}`',
        '',
        '## Message',
        '```',
        msg,
        '```',
        '## Inputs',
        *[f'- {v.describe()}' for v in ins],
    ]
    if step is not None:
        lines += [*[f'- aux {k}: {str(t.dtype).removeprefix("torch.")}{list(t.shape)}' for k, t in step.aux.items()]]
        lines += ['## magnetron', '```python', step.mag, '```', '## torch reference', '```python', step.ref, '```']
    lines += ['## Case trace (magnetron code, vN = pool tensors)', '```', *trace[-60:], '```', '']
    path.write_text('\n'.join(lines))


def _input_record(v) -> dict:  # noqa: ANN001
    """Magnetron-side layout of an input (storage + view), so the repro rebuilds the exact strides."""
    m: Tensor = v.m
    rec = {'dtype': m.dtype.name, 'shape': tuple(m.shape), 'strides': tuple(m.strides), 'ref': v.t.detach().clone(), 'view': False, 'exact_layout': True}
    if m.numel == 0:
        rec['base'] = torch.empty(tuple(m.shape), dtype=v.t.dtype)
        return rec
    base = m.view_base if m.is_view else m
    item = ITEMSIZE[v.t.dtype]
    if not (m.is_view or not m.is_contiguous):
        rec['base'] = to_torch(m)  # plain contiguous tensor: store it as is, shape included
    elif base.data_ptr == base.data_storage_ptr and base.is_contiguous:
        rec['base'] = to_torch(base).reshape(-1)
        rec['offset'] = (m.data_ptr - m.data_storage_ptr) // item
        rec['view'] = True
    else:
        rec['base'] = to_torch(m)
        rec['exact_layout'] = False
    return rec


def write_repro(d: Path, stem: str, category: str, step, ins: list, replay: str) -> None:  # noqa: ANN001
    data = {'inputs': [_input_record(v) for v in ins], 'aux': {k: t.clone() for k, t in step.aux.items()}}
    # Verify the magnetron-side reconstruction reproduces the input bit for bit.
    notes = []
    for j, (rec, v) in enumerate(zip(data['inputs'], ins)):
        if rec['view']:
            try:
                rebuilt = Tensor(rec['base']).strided_view(rec['shape'], rec['strides'], offset=rec['offset'])
                if not torch.equal(to_torch(rebuilt).view(torch.uint8), to_torch(v.m).view(torch.uint8)) or tuple(rebuilt.strides) != rec['strides']:
                    rec['exact_layout'] = False
            except Exception:  # noqa: BLE001
                rec['exact_layout'] = False
            if not rec['exact_layout']:
                rec['view'] = False
                rec['base'] = to_torch(v.m)
        if not rec['exact_layout']:
            notes.append(f'# note: x{j} is rebuilt contiguous; the original layout was {v.describe()}')
    torch.save(data, d / f'{stem}.pt')
    n = len(ins)
    xs = '\n'.join(f'x{j} = mk(d["inputs"][{j}])' for j in range(n))
    auxs = '\n'.join(f'{k} = Tensor(d["aux"]["{k}"]) if d["aux"]["{k}"].numel() else Tensor.empty(*d["aux"]["{k}"].shape, dtype=D.{TORCH2MAG[t.dtype].name})' for k, t in step.aux.items())
    xs_ref = '\n'.join(f'x{j} = d["inputs"][{j}]["ref"]' for j in range(n))
    auxs_ref = '\n'.join(f'{k} = d["aux"]["{k}"]' for k in step.aux)
    dev = step.op.rsplit('@', 1)[1] if '@' in step.op else None
    names = [f'x{j}' for j in range(n)] + list(step.aux)
    dev_lines = '\n'.join(f"{k} = mirror({k}, '{dev}')" for k in names) if dev else ''
    dev_code = f"with magnetron.device('{dev}'):\n" + textwrap.indent(step.mag, '    ') if dev else step.mag
    src = f'''\
# magnetron-fuzz repro: {category} in {step.op}
# Full case replay: {replay}
{chr(10).join(notes)}
import math
import pathlib

import torch
from magnetron import Tensor, dtype as D

from magnetron_fuzz.dtypes import to_torch
import magnetron
from magnetron_fuzz.engine import Cmp, _flat, compare, mirror
from magnetron_fuzz.refs import *

d = torch.load(pathlib.Path(__file__).with_suffix('.pt'), weights_only=False)


def mk(e):
    if not e['base'].numel():
        return Tensor.empty(*e['shape'], dtype=getattr(D, e['dtype']))
    base = Tensor(e['base'])
    return base.strided_view(e['shape'], e['strides'], offset=e['offset']) if e['view'] else base


# ---- magnetron ----
{xs}
{auxs}
{dev_lines}
for _n, _t in list(globals().items()):
    if isinstance(_t, Tensor) and _n[0] in 'xa':
        print(f'{{_n}}: shape={{_t.shape}} strides={{_t.strides}} dtype={{_t.dtype}}')
{dev_code}
mag_out = [to_torch(t) if isinstance(t, Tensor) else t for t in _flat(r)]

# ---- torch reference ----
{xs_ref}
{auxs_ref}
{step.ref if step.ref else 'r = None'}
ref_out = _flat(r)

for i, (a, b) in enumerate(zip(mag_out, ref_out)):
    print(f'output {{i}}:', compare(a, b, Cmp(), upcast=True) if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) else (a, b))
'''
    (d / f'{stem}.py').write_text(textwrap.dedent(src))
