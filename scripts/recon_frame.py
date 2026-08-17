"""A two-axis, twenty-epoch frame for the recon.  [EXPLORATORY-UNVERIFIED-PROVENANCE]

Nothing here may enter registered artifacts, paper claims, or be pooled with any registered
result.

``analysis.corrected.RunFrame`` cannot express the recon, for two reasons that are both
module-level constants rather than arguments:

* ``AXES = ("ID", "WC", "OOD")`` is read directly inside ``max_g`` and ``J``.  The recon has
  no OOD pool, so a three-axis lookup raises ``KeyError`` on a frame that legitimately has
  two axes.
* ``CKPT_GRID`` has length 24 and is the denominator of ``w_delta``.  The recon's grid is
  all 20 stored epochs, so the registered denominator would understate every window by the
  ratio 20/24 -- a run with 5 feasible checkpoints would report 0.208 instead of 0.250.

The registered module is deliberately **not modified**.  It is the code the adjudicated
battery ran, and widening it to accept an axis tuple would mean the frozen results and the
exploratory ones come out of a module that has changed since adjudication, for the benefit
of a track that is barred from being pooled with them anyway.

Subclassing costs three overrides.  ``max_g``, ``J`` and ``w_delta`` are the only methods
that read the two constants; ``rho_star_le``, ``feasible_set``, ``eta`` and
``uniform_baseline`` all reach them through ``self``, so they inherit correctly and the
estimator definitions stay shared rather than retyped.  That is the point of subclassing
here instead of copying the file: the recon must use the *same* corrected-frame arithmetic,
and the only differences permitted are the two the recon genuinely has.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from analysis.corrected import RunFrame                     # noqa: E402

RECON_AXES: Tuple[str, ...] = ("ID", "WC")
RECON_GRID: Tuple[int, ...] = tuple(range(20))              # every stored epoch
TAG = "EXPLORATORY-UNVERIFIED-PROVENANCE"


@dataclass
class ReconRunFrame(RunFrame):
    """A RunFrame whose axis set and grid are instance state, not module constants."""

    axis_names: Tuple[str, ...] = RECON_AXES
    grid: Tuple[int, ...] = RECON_GRID

    def max_g(self) -> np.ndarray:
        if not self.scorable:
            return np.full(len(self.grid), np.nan)
        return np.max(np.vstack([self.axes[a].ghat for a in self.axis_names]), axis=0)

    def J(self, grid_index: int) -> float:
        if not self.scorable:
            return float("nan")
        return float(max(self.axes[a].ghat[grid_index] for a in self.axis_names))

    def w_delta(self, delta: float) -> float:
        if not self.scorable:
            return float("nan")
        return float(len(self.feasible_set(delta)) / len(self.grid))
