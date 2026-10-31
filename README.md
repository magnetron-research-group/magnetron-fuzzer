# magnetron-fuzzer

A differential fuzzer for [magnetron](https://github.com/MarioSieg/magnetron). It builds random sequences of
magnetron tensor operations, runs each one alongside an equivalent PyTorch reference, and reports any difference
in values, shapes, dtypes, aliasing behaviour or error handling. It also catches crashes and hangs. Every
finding can be replayed deterministically from its seed and comes with a standalone single-op repro script.

## How it works

A **case** is a small program seeded by a single string. It starts with a few leaf tensors and then applies
4 to 40 random **steps**. Each step is picked by weight from about 37 operator families: views, unary and
binary ops, reductions, sort, gather and scatter, matmul, einsum, conv, interpolate, random fills and more.
Each tensor in the pool is a `(magnetron, torch)` pair, and every step runs on both sides:

1. **magnetron first.** The worker writes the current op and step to shared memory before calling into
   magnetron, so the supervisor can attribute a segfault or hang to that exact op.
2. **torch reference.** Dtypes that torch barely supports (`float8_e4m3fn`, `uint16/32/64`) are upcast for
   the reference automatically.
3. **compare.** Selection ops such as sort, max, gather and views must match bit for bit. Floating-point
   results are compared with per-dtype tolerances. Accumulating ops (sum, matmul, conv, ...) get an extra
   error bound scaled by machine epsilon and the magnitude of the accumulation.
4. **resync.** Fresh results take magnetron's exact bits on the torch side, so tolerated rounding
   differences don't compound over the case. Views stay real views on both sides, so a later in-place op
   has to propagate to every alias the same way in both libraries.

Inputs are skewed towards inputs that break kernels: dimension sizes around SIMD widths and tile
boundaries (`31, 32, 33, 63, 64, 65, ...`), broadcast, strided, negative-stride and zero-size layouts, and
special values (`nan`, `±inf`, `-0.0`, denormals, dtype limits). A small fraction of shapes use a much
larger size budget to exercise the multithreaded kernels.

The operator dtype masks mirror `magnetron/core/mag_operator.h`. A magnetron error on a dtype it claims to
support is a finding. An error on an unsupported dtype only exercises input validation and is not reported.

### Finding categories

| Category          | Meaning                                                                         |
|-------------------|---------------------------------------------------------------------------------|
| `crash`           | Worker process died from a signal or nonzero exit                               |
| `hang`            | No progress for `--timeout` seconds                                             |
| `corruption`      | A pool tensor changed although no step wrote to it                              |
| `alias`           | An in-place write wasn't reflected consistently across views of the same storage |
| `partial_write`   | An in-place op raised an error but still modified its target                    |
| `mismatch`        | Values differ from torch beyond tolerance                                       |
| `shape` / `dtype` | Result shape or dtype differs from torch                                        |
| `structure`       | Different number or kind of outputs                                             |
| `mag_error`       | magnetron raised on an input it claims to support                               |
| `accepts_invalid` | magnetron accepted an invalid input (e.g. out-of-range index) that torch rejects |
| `invariant`       | A random op produced output violating its invariants (range, distribution)      |
| `device_mismatch` | `--device cuda`: CUDA differs from magnetron CPU where torch has no reference   |
| `api_diff`        | Known, intentional difference from torch semantics (see `KNOWN_API_DIFFS`)       |

Findings are deduplicated by a signature made of category, op, dtypes, input layout and a normalized error
message. Each unique signature keeps at most 3 examples on disk.

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). The project builds magnetron from a sibling
checkout until the next release (see `[tool.uv.sources]` in `pyproject.toml`), so lay it out like this:

```
workspace/
├── magnetron/          # magnetron checkout
└── magnetron-fuzzer/   # this repo
```

```sh
uv sync
```

[py-spy](https://github.com/benfred/py-spy) is optional. If it's on `PATH`, hang reports include a native
stack dump of the stuck worker.

## Usage

```sh
# Fuzz on all cores until Ctrl-C
uv run magnetron-fuzz

# 10 minutes, 8 workers, fixed seed
uv run magnetron-fuzz -j 8 -t 600 --seed 1234

# Only fuzz some op families (regex), or skip some
uv run magnetron-fuzz --ops 'unary|binary|reduce'
uv run magnetron-fuzz --skip-ops 'conv|interpolate'

# List op families and their weights
uv run magnetron-fuzz --list-ops
```

While running, the supervisor prints a status line with throughput and unique findings per category. On exit
it prints a per-op outcome table and the top findings, ordered by severity and then frequency.

### Options

| Option                 | Default           | Description                                                     |
|------------------------|-------------------|-----------------------------------------------------------------|
| `-j, --jobs`           | all cores         | Worker processes                                                |
| `--threads`            | `1,1,1,1,1,1,1,4` | `MAG_CPU_THREADS` per worker, cycled over workers               |
| `--seed`               | random            | Base seed                                                       |
| `-t, --time`           | `0`               | Stop after this many seconds (`0` = until Ctrl-C)               |
| `--cases`              | `0`               | Stop each worker after this many cases (`0` = unlimited)        |
| `-o, --out`            | `runs`            | Output directory; a timestamped run dir is created inside       |
| `--steps`              | `4:40`            | `min:max` ops per case                                          |
| `--max-numel`          | `2048`            | Typical tensor size budget                                      |
| `--big-numel`          | `262144`          | Size budget for occasional big tensors                          |
| `--p-big`              | `0.04`            | Probability that a shape uses the big budget                    |
| `--ops` / `--skip-ops` | —                 | Regex filter on op family names                                 |
| `--timeout`            | `60`              | Seconds without progress before a worker counts as hung         |
| `--no-replay`          | off               | Don't replay crashes and hangs to produce repro scripts         |
| `--device`             | `cpu`             | `cuda`: also run every step on magnetron CUDA and check it against torch (and magnetron CPU); defaults to 4 workers |
| `--async-every`        | `4`               | Every Nth worker runs with async CPU execution; the others use `MAG_LAUNCH_BLOCKING=1` for exact crash attribution |
| `--recheck RUN_DIR`    | —                 | Replay one example per finding of an earlier run and report which still fail |

## Reproducing findings

All randomness in a case derives from its seed, `<base>-<worker>-<case>`. Each finding records that seed and
the `MAG_CPU_THREADS` value its worker used. To replay the whole case in the current process and print every
step:

```sh
MAG_CPU_THREADS=1 uv run magnetron-fuzz --replay 482913-3-1207 -v
```

If the original run used `--steps`, `--max-numel`, `--big-numel`, `--p-big`, `--ops` or `--skip-ops`, pass
the same values again. They change how the case is generated.

Each example also ships a standalone single-op repro. `exN.py` loads the exact inputs, including the original
strides and offsets of views, from `exN.pt`, then runs the failing op on magnetron and torch and compares the
results:

```sh
uv run python runs/<run>/findings/mismatch/<op>-<key>/ex0.py
```

For crashes and hangs, the supervisor reruns the case in a fresh process with `--replay --dump-step`. This
confirms the failure reproduces and writes `replay-<seed>.py` just before the faulting op.

## Output layout

```
runs/20261030-141502-s482913/
├── config.json              # effective configuration and argv
├── index.jsonl              # one line per finding occurrence (including duplicates)
├── findings/
│   └── <category>/
│       └── <op>-<key>/      # one dir per unique signature
│           ├── ex0.md       # message, inputs, magnetron and torch code, case trace, replay command
│           ├── ex0.py       # standalone single-op repro
│           └── ex0.pt       # its inputs
├── logs/                    # per-worker stdout/stderr and faulthandler output
├── stats/                   # per-worker op outcome counters
└── internal_errors.log      # bugs in the fuzzer itself, if any
```

## Code layout

| Module                  | Contents                                                                     |
|-------------------------|------------------------------------------------------------------------------|
| `cli.py`                | Argument parsing, supervisor (spawn, crash/hang detection, replay), summary |
| `worker.py`             | Worker process loop, shared-memory progress slots, case runner              |
| `engine.py`             | `Program` (tensor pool), step execution, differential comparison            |
| `ops.py`                | One random step generator per operator family, with weights                 |
| `gen.py`                | Seeded random shapes, layouts and kernel-stressing values                   |
| `dtypes.py`             | Dtype tables, tolerances, bit-exact magnetron ↔ torch conversion            |
| `refs.py`               | Torch reference helpers used by generated code and repro scripts            |
| `report.py`             | Finding signatures, dedup, reports and repro script generation              |

### Adding an operator

Write a generator in `ops.py` and register it with `@op(name, weight)`. It takes a `Program`, picks inputs
from its pool and returns a `Step`, or `None` if no suitable inputs exist. A `Step` holds the magnetron code
and the torch reference code as strings. Both read the inputs as `x0, x1, ...` and assign the result to `r`.
`Step.cmp` controls how results are compared: exact, relative tolerance, accumulation scale or mask.
`Step.supported` marks whether the dtype is in magnetron's mask for that op.

## License

Apache License 2.0, see [LICENSE](LICENSE).
