# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Random shapes and tensor contents, biased towards values that stress kernels."""

from __future__ import annotations

import math
import random

import torch

from .dtypes import FLOATS, INTS, UINTS

# Dimension sizes around SIMD widths, tile sizes and odd tails.
DIMS = (1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129)
RANK_WEIGHTS = (0.04, 0.2, 0.3, 0.2, 0.14, 0.07, 0.05)

FLOAT_SPECIALS = (float('nan'), float('inf'), -float('inf'), 0.0, -0.0, 1e38, -1e38, 1e-40, 6.1e-5, 65504.0, 448.0, 0.5, -0.5, 1.5, 2.5)


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


class Gen:
    """Random source for one fuzz case. All randomness flows from a single seed, so a case replays exactly."""

    def __init__(self, seed: str, max_numel: int, big_numel: int, p_big: float, p_zero: float) -> None:
        self.rng = random.Random(seed)
        self.g = torch.Generator().manual_seed(self.rng.getrandbits(63))
        self.max_numel = max_numel
        self.big_numel = big_numel
        self.p_big = p_big
        self.p_zero = p_zero

    # -- scalars --------------------------------------------------------------------------------

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def choice(self, seq):  # noqa: ANN001, ANN201
        return self.rng.choice(seq)

    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)

    def float_scalar(self) -> float:
        r = self.rng.random()
        if r < 0.1:
            return self.choice((0.0, -0.0, 1.0, -1.0, 0.5, 2.0))
        if r < 0.15:
            return self.choice(FLOAT_SPECIALS)
        if r < 0.4:
            return float(self.randint(-8, 8)) / 2
        return round(self.rng.gauss(0, 3), 4)

    def int_scalar(self, lo: int = -16, hi: int = 16) -> int:
        return self.randint(lo, hi)

    # -- shapes ---------------------------------------------------------------------------------

    def budget(self) -> int:
        return self.big_numel if self.chance(self.p_big) else self.max_numel

    def shape(self, min_rank: int = 0, max_rank: int = 6, budget: int | None = None, allow_zero: bool = True) -> tuple[int, ...]:
        budget = budget or self.budget()
        ranks = list(range(min_rank, max_rank + 1))
        weights = [RANK_WEIGHTS[r] if r < len(RANK_WEIGHTS) else 0.01 for r in ranks]
        rank = self.rng.choices(ranks, weights)[0]
        dims = [self.dim(budget) for _ in range(rank)]
        while numel(dims) > budget:
            i = max(range(rank), key=lambda k: dims[k])
            dims[i] = max(1, dims[i] // 2)
        if allow_zero and rank and self.chance(self.p_zero):
            dims[self.randint(0, rank - 1)] = 0
        return tuple(dims)

    def dim(self, budget: int) -> int:
        if budget > 10_000 and self.chance(0.3):
            return self.randint(100, 3000)
        return self.choice(DIMS)

    def factorize(self, n: int, rank: int) -> list[int]:
        """Random shape of `rank` dims whose product is n."""
        if rank == 0:
            return []
        dims = [1] * rank
        rest = n
        if n == 0:
            dims[self.randint(0, rank - 1)] = 0
            for i in range(rank):
                if dims[i]:
                    dims[i] = self.choice((1, 2, 3))
            return dims
        primes = []
        p = 2
        while p * p <= rest:
            while rest % p == 0:
                primes.append(p)
                rest //= p
            p += 1
        if rest > 1:
            primes.append(rest)
        for q in primes:
            dims[self.randint(0, rank - 1)] *= q
        return dims

    def broadcast_partner(self, shape: tuple[int, ...], inplace: bool = False) -> tuple[int, ...]:
        """A shape that broadcasts against `shape` (to exactly `shape` if inplace)."""
        out = list(shape)
        if out and self.chance(0.3):
            out = out[self.randint(0, len(out)) :]
        out = [1 if self.chance(0.25) else d for d in out]
        if not inplace and self.chance(0.15) and len(out) < 6:
            out = [self.choice((1, 2, 3))] + out
            if out[0] != 1 and len(out) <= len(shape):
                out[0] = 1
        return tuple(out)

    # -- tensors --------------------------------------------------------------------------------

    def tensor(self, shape: tuple[int, ...], dt: torch.dtype, *, specials: float = 0.12, positive: bool = False, nonzero: bool = False) -> torch.Tensor:
        n = numel(shape)
        if dt == torch.bool:
            return torch.rand(shape, generator=self.g) < self.choice((0.1, 0.5, 0.9))
        if dt in FLOATS:
            x = self._float_values(shape)
            if n and self.chance(specials) and not positive:
                k = self.randint(1, max(1, min(n, 4)))
                flat = x.reshape(-1)
                for _ in range(k):
                    flat[self.randint(0, n - 1)] = self.choice(FLOAT_SPECIALS)
            if positive:
                x = x.abs() + 1e-3
            if nonzero:
                x = torch.where(x == 0, torch.ones_like(x), x)
            if dt == torch.float8_e4m3fn:
                x = x.clamp(-448, 448)
            return x.to(dt)
        assert dt in INTS
        x = self._int_values(shape, dt)
        if positive:
            x = x.abs() if dt not in UINTS else x
        if nonzero:
            x = torch.where(x == 0, torch.ones_like(x), x)
        return x

    def _float_values(self, shape: tuple[int, ...]) -> torch.Tensor:
        style = self.rng.choices(('normal', 'uniform', 'int', 'half', 'wide', 'tiny'), (0.35, 0.25, 0.12, 0.1, 0.12, 0.06))[0]
        g = self.g
        if style == 'normal':
            return torch.randn(shape, generator=g, dtype=torch.float64) * self.choice((0.01, 0.1, 1.0, 1.0, 3.0, 10.0, 100.0))
        if style == 'uniform':
            return (torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1) * self.choice((0.5, 1.0, 4.0))
        if style == 'int':
            return torch.randint(-8, 9, shape, generator=g).double()
        if style == 'half':
            return torch.randint(-16, 17, shape, generator=g).double() / 2
        if style == 'wide':
            mag = torch.exp(torch.rand(shape, generator=g, dtype=torch.float64) * 40 - 20)
            return mag * torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0).double()
        return torch.randn(shape, generator=g, dtype=torch.float64) * 1e-39

    def _int_values(self, shape: tuple[int, ...], dt: torch.dtype) -> torch.Tensor:
        info = torch.iinfo(dt)
        style = self.rng.choices(('small', 'medium', 'full'), (0.55, 0.3, 0.15))[0]
        g = self.g
        if style == 'full':
            if info.bits == 64:
                x = torch.randint(-(2**63), 2**63 - 1, shape, generator=g, dtype=torch.int64)
                return x.view(torch.uint64) if dt == torch.uint64 else x
            return torch.randint(info.min, info.max + 1, shape, generator=g, dtype=torch.int64).to(dt)
        lim = 10 if style == 'small' else 1000
        lo = 0 if dt in UINTS else max(-lim, info.min)
        hi = min(lim, info.max)
        return torch.randint(lo, hi + 1, shape, generator=g, dtype=torch.int64).to(dt)

    def index(self, size: int, shape: tuple[int, ...], negative: bool = True) -> torch.Tensor:
        """int64 indices into a dim of length `size`, optionally using negative indices."""
        if size == 0:
            return torch.zeros(shape, dtype=torch.int64)
        idx = torch.randint(0, size, shape, generator=self.g)
        if negative and self.chance(0.3):
            idx = torch.where(torch.rand(shape, generator=self.g) < 0.5, idx - size, idx)
        return idx

    def perm(self, n: int) -> list[int]:
        p = list(range(n))
        self.rng.shuffle(p)
        return p
