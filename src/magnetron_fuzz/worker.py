# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Fuzz worker process: runs cases forever and publishes progress through shared memory.

Kept free of heavy top-level imports: the supervisor imports this module only to name the spawn target.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Shared-memory slot layout per worker (int64 each).
HB, CASE, STEP, CASES, STEPS, FINDINGS, MAG_ERR, REF_ERR, INTERNAL, STATE, PID = range(11)
NSLOT = 16
OPNAME = 64  # bytes per worker for the current op name


@dataclass
class Config:
    seed: str
    run_dir: str | None
    min_steps: int = 4
    max_steps: int = 40
    max_numel: int = 2048
    big_numel: int = 1 << 18
    p_big: float = 0.04
    p_zero: float = 0.03
    include: str | None = None
    exclude: str | None = None
    max_cases: int = 0
    check_every: float = 0.1
    threads: int = 1
    extra: dict = field(default_factory=dict)


class Hooks:
    """Publishes the step about to run so the supervisor can attribute crashes and hangs."""

    def __init__(self, shm, opbuf, wid: int) -> None:  # noqa: ANN001
        self.shm, self.opbuf, self.wid = shm, opbuf, wid
        self.base = wid * NSLOT

    def before_mag(self, prog, step, ins) -> None:  # noqa: ANN001
        b = self.base
        self.shm[b + HB] = time.monotonic_ns() // 1_000_000
        self.shm[b + STEP] = prog.step_idx
        self.shm[b + STATE] = 1
        name = step.op.encode()[: OPNAME - 1]
        o = self.wid * OPNAME
        self.opbuf[o : o + OPNAME] = name + b'\0' * (OPNAME - len(name))

    def after_mag(self, prog, step) -> None:  # noqa: ANN001
        self.shm[self.base + STATE] = 2


class ReplayHooks:
    """Replay mode: dump a single-op repro right before the target step runs on magnetron."""

    def __init__(self, dump_step: int | None, dump_dir: Path | None, reporter, verbose: bool) -> None:  # noqa: ANN001
        self.dump_step, self.dump_dir, self.reporter, self.verbose = dump_step, dump_dir, reporter, verbose

    def before_mag(self, prog, step, ins) -> None:  # noqa: ANN001
        if self.verbose:
            print(f'[{prog.step_idx}] {step.op}: {step.mag.replace(chr(10), "; ")}', flush=True)
            for v in ins:
                print(f'      {v.describe()}', flush=True)
        if self.dump_step is not None and prog.step_idx == self.dump_step and self.dump_dir is not None:
            from .report import replay_command, write_repro

            self.dump_dir.mkdir(parents=True, exist_ok=True)
            stem = f'replay-{self.reporter.seed}'
            write_repro(self.dump_dir, stem, 'crash', step, ins, replay_command(self.reporter.seed, self.reporter.threads))
            print(f'wrote {self.dump_dir / stem}.py', flush=True)

    def after_mag(self, prog, step) -> None:  # noqa: ANN001
        pass


def run_case(cfg: Config, seed: str, reporter, hooks, stats: dict, ops: list, on_internal=None) -> int:  # noqa: ANN001
    """Run one case; returns the number of steps executed."""
    import magnetron

    from .engine import Program
    from .gen import Gen
    from .ops import leaf

    gen = Gen(seed, cfg.max_numel, cfg.big_numel, cfg.p_big, cfg.p_zero)
    magnetron.context.manual_seed(gen.rng.getrandbits(32))
    prog = Program(gen, reporter, hooks)
    weights = [o[1] for o in ops]
    plan = ['leaf'] * gen.randint(1, 3) + [None] * gen.randint(cfg.min_steps, cfg.max_steps)
    for planned in plan:
        step = None
        try:
            for _ in range(10):
                fn = leaf if planned == 'leaf' or not prog.pool else ops[gen.rng.choices(range(len(ops)), weights)[0]][2]
                step = fn(prog)
                if step is not None:
                    break
            if step is None:
                continue
            out = prog.run(step)
        except Exception:  # noqa: BLE001
            if on_internal:
                on_internal(seed, prog.step_idx, step, traceback.format_exc())
            return prog.step_idx
        s = stats.setdefault(step.op, {})
        s[out.status] = s.get(out.status, 0) + 1
        prog.step_idx += 1
        if gen.chance(cfg.check_every):
            prog.check_all('periodic pool check')
    prog.check_all('end-of-case pool check')
    return prog.step_idx


def _setup_process(cfg: Config, wid: int) -> None:
    import faulthandler
    import sys
    import warnings

    run_dir = Path(cfg.run_dir)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    log = open(run_dir / 'logs' / f'worker{wid}.log', 'a', buffering=1)  # noqa: SIM115
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    fault = open(run_dir / 'logs' / f'worker{wid}.fault', 'w')  # noqa: SIM115
    faulthandler.enable(fault, all_threads=True)
    warnings.filterwarnings('ignore')


def worker_main(wid: int, cfg: Config, shm, opbuf, stop, start_idx: int) -> None:  # noqa: ANN001
    _setup_process(cfg, wid)
    import torch

    torch.set_num_threads(1)
    from .ops import weighted_ops
    from .report import Reporter

    run_dir = Path(cfg.run_dir)
    ops = weighted_ops(cfg.include, cfg.exclude)
    hooks = Hooks(shm, opbuf, wid)
    b = wid * NSLOT
    shm[b + PID] = os.getpid()
    prev = run_dir / 'stats' / f'worker{wid}.json'
    stats: dict = json.loads(prev.read_text()) if prev.exists() else {}  # continue counting across respawns
    last_dump = time.monotonic()

    def on_internal(seed: str, step_idx: int, step, tb: str) -> None:  # noqa: ANN001
        shm[b + INTERNAL] += 1
        with open(run_dir / 'internal_errors.log', 'a') as f:
            f.write(f'--- seed {seed} step {step_idx} op {getattr(step, "op", None)}\n{getattr(step, "mag", "")}\n{tb}\n')

    idx = start_idx
    while not stop.value:
        seed = f'{cfg.seed}-{wid}-{idx}'
        shm[b + CASE] = idx
        shm[b + HB] = time.monotonic_ns() // 1_000_000
        reporter = Reporter(run_dir, seed, cfg.threads, worker=wid)
        n = run_case(cfg, seed, reporter, hooks, stats, ops, on_internal)
        shm[b + STATE] = 0
        shm[b + CASES] += 1
        shm[b + STEPS] += n
        shm[b + FINDINGS] += reporter.count
        idx += 1
        now = time.monotonic()
        if now - last_dump > 5:
            _dump_stats(run_dir, wid, stats)
            last_dump = now
        if cfg.max_cases and shm[b + CASES] >= cfg.max_cases:
            break
    _dump_stats(run_dir, wid, stats)


def _dump_stats(run_dir: Path, wid: int, stats: dict) -> None:
    d = run_dir / 'stats'
    d.mkdir(exist_ok=True)
    tmp = d / f'.worker{wid}.json'
    tmp.write_text(json.dumps(stats))
    os.replace(tmp, d / f'worker{wid}.json')
