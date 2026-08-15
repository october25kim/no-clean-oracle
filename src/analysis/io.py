"""Reading the audit's JSONL logs back into trajectories (pure NumPy, no GPU).

The log is deliberately raw: one metadata line plus one line per epoch holding the
values measured at that epoch and nothing derived. Everything an analysis needs that
is NOT a direct measurement -- the static tail-class set, the 3-epoch moving average,
oracle epochs -- is reconstructed here, so the same log can be re-analyzed under a
different smoothing or tail definition without re-running anything on a GPU.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from eval.tail import r_tail_static, static_tail_classes


@dataclass
class Run:
    """One run's log: its metadata line and its per-epoch records, in epoch order."""
    run_dir: str
    meta: dict
    epochs: List[dict]

    @property
    def run_id(self) -> str:
        return self.meta.get("run_id", os.path.basename(self.run_dir))

    @property
    def cell(self) -> str:
        """The PRE-REGISTERED analysis unit: one (dataset x noise x learner) task.

        There are 16 of these and the bootstrap resamples the 3 seeds inside one.
        The learner is a fixed factor, never a resampling unit.

        The eta is underscore-separated on purpose: ``ncr.verdict`` detects the
        highest-noise cells by matching ``_0.6``, which ``cifar10_symmetric_0.6_ce``
        satisfies and ``cifar10_symmetric0.6_ce`` would not.
        """
        return f"{self.tail_group}_{self.meta['learner']}"

    @property
    def tail_group(self) -> str:
        """The (dataset x noise) group: the scope of the static tail-class set, which
        CE and ELR share so their R_tail is measured on the same classes. Also the key
        of the pooled secondary table, which is diagnostic only."""
        split = self.meta.get("split")
        if split:                       # real-noise run: the split IS the group
            return str(split)
        return f"{self.meta['dataset']}_{self.meta['noise_type']}_{self.meta['eta']:g}"

    def series(self, key: str) -> np.ndarray:
        return np.asarray([e[key] for e in self.epochs], dtype=np.float64)

    @property
    def per_class_test_error(self) -> np.ndarray:
        return np.asarray([e["per_class_test_error"] for e in self.epochs], dtype=np.float64)


def load_run(run_dir: str) -> Run:
    """Load ``run_dir/metrics.jsonl``: line 0 is metadata, the rest are epochs."""
    path = os.path.join(run_dir, "metrics.jsonl")
    meta: dict = {}
    epochs: List[dict] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # a run appending its current epoch can leave a half-written last line
                break
            if i == 0 and rec.get("record") == "metadata":
                meta = rec
            else:
                epochs.append(rec)
    epochs.sort(key=lambda r: r["epoch"])
    return Run(run_dir=run_dir, meta=meta, epochs=epochs)


def static_tail_from_reference(reference: Run, tail_frac: Optional[float] = None) -> np.ndarray:
    """The cell's fixed tail-class set: bottom-frac classes of the reference run's
    FINAL epoch by clean-test per-class accuracy (the CE seed-0 run, per config)."""
    frac = reference.meta.get("tail_frac", 0.30) if tail_frac is None else tail_frac
    final_err = np.asarray(reference.epochs[-1]["per_class_test_error"], dtype=np.float64)
    return static_tail_classes(1.0 - final_err, frac)


def r_tail_static_series(run: Run, tail_classes: Sequence[int]) -> np.ndarray:
    """R_tail(t) on a FIXED class set, recomputed from the logged per-class errors."""
    return np.asarray([r_tail_static(np.asarray(e["per_class_test_error"], dtype=np.float64),
                                     np.asarray(tail_classes))
                       for e in run.epochs], dtype=np.float64)


def risk_trajectories(run: Run, tail_classes: Sequence[int]) -> Dict[str, np.ndarray]:
    """The three audited risks R_j(t), keyed as analysis.ncr.OBJECTIVES expects."""
    return {
        "ID": run.series("R_ID"),
        "tail": r_tail_static_series(run, tail_classes),
        "OOD": run.series("R_OOD_primary_energy_semantic"),
    }
