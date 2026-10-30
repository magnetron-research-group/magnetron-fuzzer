# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Supervisor: spawns one fuzz worker per core, detects crashes and hangs, replays them for repros."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .worker import CASE, CASES, FINDINGS, HB, INTERNAL, NSLOT, OPNAME, PID, STATE, STEP, STEPS, Config, worker_main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog='magnetron-fuzz', description='Differential fuzzer: random magnetron op sequences checked against torch.')
    ap.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1, help='worker processes (default: all cores)')
    ap.add_argument('--threads', default='1,1,1,1,1,1,1,4', help='MAG_CPU_THREADS per worker, cycled over workers (default: mostly 1, every 8th worker 4)')
    ap.add_argument('--seed', default=None, help='base seed (default: random)')
    ap.add_argument('-t', '--time', type=float, default=0, help='stop after this many seconds (0 = run until Ctrl-C)')
    ap.add_argument('--cases', type=int, default=0, help='stop each worker after this many cases (0 = unlimited)')
    ap.add_argument('-o', '--out', default='runs', help='output directory (a timestamped run dir is created inside)')
    ap.add_argument('--steps', default='4:40', help='min:max ops per case')
    ap.add_argument('--max-numel', type=int, default=2048, help='typical tensor size budget')
    ap.add_argument('--big-numel', type=int, default=1 << 18, help='size budget for occasional big tensors (exercises multithreaded kernels)')
    ap.add_argument('--p-big', type=float, default=0.04, help='probability a shape uses the big budget')
    ap.add_argument('--ops', default=None, help='regex: only fuzz op families matching (e.g. "unary|binary")')
    ap.add_argument('--skip-ops', default=None, help='regex: skip op families matching')
    ap.add_argument('--timeout', type=float, default=60, help='seconds without progress before a worker is considered hung')
    ap.add_argument('--no-replay', action='store_true', help='do not replay crashes to produce repro scripts')
    ap.add_argument('--replay', metavar='SEED', help='replay one case (seed from a finding) in this process')
    ap.add_argument('--dump-step', type=int, default=None, help='with --replay: write a single-op repro before this step runs')
    ap.add_argument('--dump-dir', default=None, help='with --replay --dump-step: where to write the repro')
    ap.add_argument('-v', '--verbose', action='store_true', help='with --replay: print every step')
    ap.add_argument('--list-ops', action='store_true', help='list op families and weights')
    return ap.parse_args(argv)


def make_config(a: argparse.Namespace, seed: str, run_dir: str | None) -> Config:
    lo, hi = (int(x) for x in a.steps.split(':'))
    return Config(seed=seed, run_dir=run_dir, min_steps=lo, max_steps=hi, max_numel=a.max_numel, big_numel=a.big_numel, p_big=a.p_big,
                  include=a.ops, exclude=a.skip_ops, max_cases=a.cases)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if a.list_ops:
        from .ops import weighted_ops

        for name, w, _ in weighted_ops(a.ops, a.skip_ops):
            print(f'{w:6.1f}  {name}')
        return 0
    if a.replay:
        return replay(a)
    return Supervisor(a).run()


# -- replay ---------------------------------------------------------------------------------------


def replay(a: argparse.Namespace) -> int:
    import warnings

    warnings.filterwarnings('ignore')
    import torch

    torch.set_num_threads(1)
    from .ops import weighted_ops
    from .report import Reporter
    from .worker import ReplayHooks, run_case

    base_seed = a.replay.rsplit('-', 2)[0]
    cfg = make_config(a, base_seed, None)
    threads = int(os.environ.get('MAG_CPU_THREADS', '0') or 0)
    reporter = Reporter(None, a.replay, threads, verbose=True)
    hooks = ReplayHooks(a.dump_step, Path(a.dump_dir) if a.dump_dir else None, reporter, a.verbose)
    stats: dict = {}

    def on_internal(seed: str, step_idx: int, step, tb: str) -> None:  # noqa: ANN001
        print(f'internal fuzzer error at step {step_idx}:\n{tb}')

    n = run_case(cfg, a.replay, reporter, hooks, stats, weighted_ops(a.ops, a.skip_ops), on_internal)
    print(f'replayed {n} steps, {reporter.count} findings: {reporter.by_category}')
    return 1 if reporter.count else 0


# -- supervisor -----------------------------------------------------------------------------------


class Supervisor:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.seed = a.seed or str(secrets.randbelow(10**6))
        self.run_dir = Path(a.out).resolve() / time.strftime(f'%Y%m%d-%H%M%S-s{self.seed}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = make_config(a, self.seed, str(self.run_dir))
        self.jobs = max(1, a.jobs)
        cycle = [max(0, int(x)) for x in a.threads.split(',') if x.strip()] or [1]
        self.threads = [cycle[w % len(cycle)] for w in range(self.jobs)]
        self.ctx = mp.get_context('spawn')
        self.shm = self.ctx.Array('q', self.jobs * NSLOT, lock=False)
        self.opbuf = self.ctx.Array('c', self.jobs * OPNAME, lock=False)
        self.stop = self.ctx.Value('b', 0, lock=False)
        self.procs: list = [None] * self.jobs
        self.done = [False] * self.jobs
        self.crashes = Counter()
        self.replays = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.t0 = time.monotonic()

    def spawn(self, w: int, start_idx: int) -> None:
        cfg = Config(**{**self.cfg.__dict__, 'threads': self.threads[w]})
        os.environ['MAG_CPU_THREADS'] = str(self.threads[w])
        p = self.ctx.Process(target=worker_main, args=(w, cfg, self.shm, self.opbuf, self.stop, start_idx), daemon=True, name=f'fuzz{w}')
        p.start()
        self.shm[w * NSLOT + HB] = time.monotonic_ns() // 1_000_000
        self.procs[w] = p

    def run(self) -> int:
        a = self.a
        print(f'magnetron-fuzz: seed {self.seed}, {self.jobs} workers, MAG_CPU_THREADS per worker cycle {a.threads}')
        print(f'run dir: {self.run_dir}')
        (self.run_dir / 'config.json').write_text(json.dumps({**self.cfg.__dict__, 'jobs': self.jobs, 'threads': self.threads, 'argv': sys.argv}, indent=2))
        for w in range(self.jobs):
            self.spawn(w, 0)
        interrupted = False
        old = signal.signal(signal.SIGINT, lambda *_: setattr(self, '_sigint', True))
        self._sigint = False
        last_print = 0.0
        try:
            while True:
                time.sleep(0.25)
                if self._sigint:
                    interrupted = True
                    break
                if a.time and time.monotonic() - self.t0 > a.time:
                    break
                self.supervise()
                if all(self.done):
                    break
                now = time.monotonic()
                if now - last_print > 2:
                    self.print_status(final=False)
                    last_print = now
        finally:
            signal.signal(signal.SIGINT, old)
            self.shutdown()
        self.print_status(final=True)
        self.summary(interrupted)
        return 0

    def supervise(self) -> None:
        now_ms = time.monotonic_ns() // 1_000_000
        for w, p in enumerate(self.procs):
            if self.done[w] or p is None:
                continue
            b = w * NSLOT
            if not p.is_alive():
                p.join()
                code = p.exitcode
                if code == 0:
                    self.done[w] = True
                    continue
                self.on_crash(w, code)
                self.spawn(w, self.shm[b + CASE] + 1)
            elif self.shm[b + STATE] != 0 and now_ms - self.shm[b + HB] > self.a.timeout * 1000:
                self.on_hang(w)
                p.kill()
                p.join()
                self.spawn(w, self.shm[b + CASE] + 1)

    def worker_state(self, w: int) -> tuple[str, int, str]:
        b = w * NSLOT
        seed = f'{self.seed}-{w}-{self.shm[b + CASE]}'
        raw = bytes(self.opbuf[w * OPNAME : (w + 1) * OPNAME])
        return seed, int(self.shm[b + STEP]), raw.split(b'\0', 1)[0].decode(errors='replace')

    def on_crash(self, w: int, code: int) -> None:
        seed, step, op = self.worker_state(w)
        try:
            sig = signal.Signals(-code).name if code < 0 else f'exit{code}'
        except ValueError:
            sig = f'signal{-code}'
        fault = (self.run_dir / 'logs' / f'worker{w}.fault').read_text(errors='replace') if (self.run_dir / 'logs' / f'worker{w}.fault').exists() else ''
        self.record_abort('crash', sig, w, seed, step, op, fault)

    def on_hang(self, w: int) -> None:
        seed, step, op = self.worker_state(w)
        pid = self.shm[w * NSLOT + PID]
        stack = ''
        try:  # Best effort: py-spy shows where the worker is stuck, if installed.
            stack = subprocess.run(['py-spy', 'dump', '--native', '--pid', str(pid)], capture_output=True, text=True, timeout=20).stdout
        except Exception:  # noqa: BLE001
            pass
        self.record_abort('hang', f'>{self.a.timeout:g}s', w, seed, step, op, stack)

    def record_abort(self, category: str, what: str, w: int, seed: str, step: int, op: str, trace: str) -> None:
        from .report import append_index, claim_example_slot, replay_command, slug

        self.crashes[(category, what, op)] += 1
        threads = self.threads[w]
        sig = f'{category}|{op}|{what}'
        key = hashlib.sha1(sig.encode()).hexdigest()[:10]
        d = self.run_dir / 'findings' / category / f'{slug(op)}-{slug(what)}-{key}'
        k = claim_example_slot(d)
        append_index(self.run_dir, {'t': time.time(), 'sig': sig, 'key': key, 'category': category, 'op': op, 'seed': seed, 'step': step, 'threads': threads, 'worker': w, 'msg': what})
        print(f'\n!! {category} ({what}) in worker {w} at op {op!r}, seed {seed} step {step}', flush=True)
        if k is None:
            return
        cmd = replay_command(seed, threads)
        (d / f'ex{k}.md').write_text(
            f'# {category}: {op} ({what})\n\n- seed: `{seed}`, step {step}, MAG_CPU_THREADS={threads}\n- replay: `{cmd}`\n\n'
            f'## Python stack at {category} (faulthandler / py-spy)\n```\n{trace.strip()}\n```\n'
        )
        if not self.a.no_replay:
            self.replays.submit(self.confirm, d, k, seed, step, threads, category)

    def confirm(self, d: Path, k: int, seed: str, step: int, threads: int, category: str) -> None:
        """Re-run the case in a fresh process: confirms reproducibility and writes a single-op repro script."""
        env = {**os.environ, 'MAG_CPU_THREADS': str(threads), 'PYTHONFAULTHANDLER': '1'}
        cmd = [sys.executable, '-m', 'magnetron_fuzz', '--replay', seed, '--dump-step', str(step), '--dump-dir', str(d), *self.forwarded_args()]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=max(120, 3 * self.a.timeout))
            status = f'exit code {r.returncode}' + (f' ({signal.Signals(-r.returncode).name})' if r.returncode < 0 else '')
            reproduced = r.returncode < 0 if category == 'crash' else False
            out = (r.stdout[-4000:] + '\n' + r.stderr[-8000:]).strip()
        except subprocess.TimeoutExpired as e:
            status, reproduced = 'timed out', category == 'hang'
            out = ((e.stdout or b'')[-4000:].decode(errors='replace') if isinstance(e.stdout, bytes) else str(e.stdout or ''))
        with open(d / f'ex{k}.md', 'a') as f:
            f.write(f'\n## Replay\n- `{" ".join(cmd)}`\n- {status}; reproduced: **{"yes" if reproduced else "no"}**\n')
            f.write(f'- single-op repro: `replay-{seed}.py` (run with `uv run python <file>`)\n\n```\n{out}\n```\n')

    def forwarded_args(self) -> list[str]:
        a = self.a
        out = ['--steps', a.steps, '--max-numel', str(a.max_numel), '--big-numel', str(a.big_numel), '--p-big', str(a.p_big)]
        if a.ops:
            out += ['--ops', a.ops]
        if a.skip_ops:
            out += ['--skip-ops', a.skip_ops]
        return out

    def shutdown(self) -> None:
        self.stop.value = 1
        deadline = time.monotonic() + 5
        for p in self.procs:
            if p is not None:
                p.join(timeout=max(0.0, deadline - time.monotonic()))
        for p in self.procs:
            if p is not None and p.is_alive():
                p.kill()
                p.join()
        self.replays.shutdown(wait=True)

    def totals(self) -> dict[str, int]:
        t = Counter()
        for w in range(self.jobs):
            b = w * NSLOT
            t['cases'] += self.shm[b + CASES]
            t['steps'] += self.shm[b + STEPS]
            t['findings'] += self.shm[b + FINDINGS]
            t['internal'] += self.shm[b + INTERNAL]
        return t

    def unique(self) -> Counter:
        c = Counter()
        root = self.run_dir / 'findings'
        if root.exists():
            for cat in root.iterdir():
                c[cat.name] = sum(1 for _ in cat.iterdir())
        return c

    def print_status(self, final: bool) -> None:
        t = self.totals()
        el = max(1e-9, time.monotonic() - self.t0)
        u = self.unique()
        uniq = ' '.join(f'{k}={v}' for k, v in sorted(u.items())) or 'none'
        line = f'[{el:7.0f}s] cases {t["cases"]:>8} steps {t["steps"]:>10} ({t["steps"] / el:7.0f}/s) | unique findings: {uniq}'
        if t['internal']:
            line += f' | fuzzer errors {t["internal"]}'
        if sys.stdout.isatty() and not final:
            print('\r' + line[: os.get_terminal_size().columns - 1].ljust(os.get_terminal_size().columns - 1), end='', flush=True)
        else:
            print(line, flush=True)

    def summary(self, interrupted: bool) -> None:
        print()
        idx = self.run_dir / 'index.jsonl'
        recs = [json.loads(line) for line in idx.read_text().splitlines()] if idx.exists() else []
        by_sig: dict[str, list] = {}
        for r in recs:
            by_sig.setdefault(r['sig'], []).append(r)
        stats = Counter()
        for f in (self.run_dir / 'stats').glob('worker*.json'):
            for op, d in json.loads(f.read_text()).items():
                for k, v in d.items():
                    stats[(op, k)] += v
        ops = sorted({op for op, _ in stats})
        if ops:
            print(f'{"op":<28} {"ok":>9} {"finding":>8} {"ref_err":>8} {"both_err":>8} {"unsupp":>7}')
            for op in ops:
                print(f'{op:<28} {stats[(op, "ok")]:>9} {stats[(op, "finding")]:>8} {stats[(op, "ref_err")]:>8} {stats[(op, "both_err")]:>8} {stats[(op, "unsupported")]:>7}')
            print()
        print(f'{len(by_sig)} unique findings ({len(recs)} total) in {self.run_dir / "findings"}')
        order = sorted(by_sig.items(), key=lambda kv: ({'crash': 0, 'hang': 1, 'corruption': 2, 'alias': 3, 'partial_write': 4, 'mismatch': 5}.get(kv[1][0]['category'], 9), -len(kv[1])))
        for sig, rs in order[:40]:
            r = rs[0]
            print(f'  {len(rs):>6}x  {r["category"]:<12} {r["op"]:<26} {r["msg"][:90]}')
            print(f'           seed {r["seed"]} step {r["step"]}  ->  findings/{r["category"]}/*-{r["key"]}')
        if len(order) > 40:
            print(f'  ... {len(order) - 40} more, see {idx}')
        if (self.run_dir / 'internal_errors.log').exists():
            print(f'fuzzer-internal errors were logged to {self.run_dir / "internal_errors.log"}')
        if interrupted:
            print('interrupted')
