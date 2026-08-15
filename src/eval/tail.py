"""Tail-class set construction for R_tail (pure NumPy, no GPU).

Two definitions (both logged; primary endpoint uses the static set):
- static  : the bottom ``frac`` of classes by clean-test per-class ACCURACY of the
            CE seed-0 run at its FINAL epoch, fixed once per (dataset, noise) cell and
            reused for every run in that cell so the class set is constant.
- dynamic : per epoch, the worst ``frac`` of classes at THAT epoch.
R_tail is the mean per-class ERROR over the selected class set.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def per_class_error(confusion_correct: np.ndarray, per_class_total: np.ndarray) -> np.ndarray:
    """Per-class error = 1 - correct/total; classes with 0 samples -> nan."""
    total = np.asarray(per_class_total, dtype=np.float64)
    correct = np.asarray(confusion_correct, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(total > 0, correct / total, np.nan)
    return 1.0 - acc


def static_tail_classes(final_per_class_acc: np.ndarray, frac: float = 0.30) -> np.ndarray:
    """Indices of the bottom ``frac`` classes by accuracy (ties broken by class id)."""
    acc = np.asarray(final_per_class_acc, dtype=np.float64)
    n = acc.size
    k = max(1, int(round(frac * n)))
    order = np.lexsort((np.arange(n), acc))     # ascending acc, stable by class id
    return np.sort(order[:k])


def r_tail_static(epoch_per_class_error: np.ndarray, tail_classes: np.ndarray) -> float:
    err = np.asarray(epoch_per_class_error, dtype=np.float64)[np.asarray(tail_classes)]
    return float(np.nanmean(err))


def r_tail_dynamic(epoch_per_class_error: np.ndarray, frac: float = 0.30) -> float:
    err = np.asarray(epoch_per_class_error, dtype=np.float64)
    valid = err[~np.isnan(err)]
    k = max(1, int(round(frac * valid.size)))
    worst = np.sort(valid)[::-1][:k]
    return float(worst.mean())
