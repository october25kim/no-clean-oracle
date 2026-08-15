"""Label Wave early stopping, applied post-hoc to the G1 logs.

Yuan, Feng, Liu, "Early Stopping Against Label Noise Without Validation Data",
ICLR 2024 (arXiv:2502.07551). Transcribed from the paper, not reconstructed:

  Eq. (2)  prediction changes:  PC_t = sum_{i in D} 1[ y_hat_i^t != y_hat_i^(t-1) ]
           -- against the PREVIOUS EPOCH'S PREDICTION, over the whole noisy training
           set D. This is exactly what G1 logs as ``pred_flip_count``.

  Eq. (3)  moving average:      PC'_t = (PC_t + PC_(t-1) + ... + PC_(t-k+1)) / k
           -- a TRAILING mean over the recent k epochs, not a centred one.

  Eq. (4)  Early Stopping Point = t_first-min, the FIRST local minimum of PC'_t
           (not the global argmin), located by Algorithm 1's patience counter:

             v <- inf, i <- 0
             for each epoch t:
                 if PC'_t < v:  v <- PC'_t; i <- 0; t* <- t
                 else:          i <- i + 1
                 halt when i >= p

Two details the paper does not pin down numerically, flagged rather than invented:

  * ``k``: Appendix E sweeps k in {1, 2, 3, 5, 10} and reports the strongest
    PC'-vs-test-accuracy correlation at **k = 3** (-0.9637). No sentence names a
    default, so k=3 is used as the paper's own best-supported value and the others
    are reported alongside. NEEDS-VERIFICATION against the authors' code.
  * ``patience`` p: named in Algorithm 1, never given a number anywhere in the paper.
    Reported over a range instead of guessing one. NEEDS-VERIFICATION.
  * PC'_t needs k terms; the paper does not say what happens for t < k. Here PC'_t is
    only defined once the window is full (PC starts at epoch 1, so PC' starts at
    epoch k). NEEDS-VERIFICATION.
  * How the predictions y_hat_i^t are PRODUCED is never specified. The paper contains no
    occurrence of "model.eval", "eval mode" or "no_grad", and gives no code link. The
    variant G1 logs collects argmax inside the training loop (trainer.py:232-251), so it
    is train-mode, augmented, and -- because weights update batch by batch -- NOT a
    single-theta snapshot the way Eq. (2)'s y_hat_i^t notation implies. Each sample is
    predicted by whatever parameters existed when its batch came up, so consecutive-epoch
    comparisons absorb intra-epoch weight drift on top of genuine prediction change,
    plausibly biasing PC upward relative to a synchronized end-of-epoch pass. That the
    authors likewise reuse the training pass is an INFERENCE from their efficiency
    claims ("computationally efficient", "low computational overhead", "requires no
    preprocessing"), not something the paper states. NEEDS-VERIFICATION.

Applying an early-stopping rule to a finished log is a faithful simulation of it:
the rule reads only PC_1..PC_t to decide at epoch t, so replaying the logged sequence
selects exactly the epoch the rule would have halted at online.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

PAPER = ("Yuan, Feng & Liu, Early Stopping Against Label Noise Without Validation Data, "
         "ICLR 2024 (arXiv:2502.07551)")
PAPER_BEST_K = 3          # Appendix E: strongest correlation, -0.9637


def prediction_changes(epochs: Sequence[dict]) -> np.ndarray:
    """PC_t for t = 1..T-1 (Eq. 2), read straight from the log.

    Epoch 0 has no predecessor and is logged as null; it is dropped, so index j of the
    returned array is epoch j+1.
    """
    return np.asarray([e["pred_flip_count"] for e in epochs[1:]], dtype=np.float64)


def trailing_mean(pc: np.ndarray, k: int) -> np.ndarray:
    """PC'_t (Eq. 3): trailing mean over the recent k epochs, NaN until the window fills."""
    pc = np.asarray(pc, dtype=np.float64)
    out = np.full(pc.shape, np.nan)
    if k <= 0 or pc.size == 0:
        return out
    csum = np.concatenate([[0.0], np.cumsum(pc)])
    for j in range(k - 1, pc.size):
        out[j] = (csum[j + 1] - csum[j + 1 - k]) / k
    return out


@dataclass
class LabelWaveResult:
    stop_epoch: Optional[int]        # the selected epoch (Eq. 4), in log epoch numbering
    halted_epoch: Optional[int]      # where the patience counter tripped; None = never
    exhausted: bool                  # True if training ended before patience tripped
    k: int
    patience: int
    pc: np.ndarray
    pc_smoothed: np.ndarray


def label_wave(epochs: Sequence[dict], k: int = PAPER_BEST_K,
               patience: int = 5) -> LabelWaveResult:
    """Algorithm 1, replayed over a finished run's log.

    ``exhausted=True`` means PC' never worsened for ``patience`` consecutive epochs, so
    the rule ran to the end of training and returned the last improving epoch. That is
    the rule's own degenerate case, not an error: the paper (Appendix C.3) states that
    when the run has no learning-confusion stage the method "may not identify an
    appropriate stopping point" -- the source's own strength, which is weaker than
    "cannot find one".
    """
    pc = prediction_changes(epochs)
    sm = trailing_mean(pc, k)

    best = np.inf
    misses = 0
    t_star: Optional[int] = None
    halted: Optional[int] = None
    for j, v in enumerate(sm):
        if np.isnan(v):                       # window not full yet
            continue
        epoch = j + 1                         # PC index j corresponds to epoch j+1
        if v < best:
            best, misses, t_star = v, 0, epoch
        else:
            misses += 1
            if misses >= patience:
                halted = epoch
                break
    return LabelWaveResult(stop_epoch=t_star, halted_epoch=halted, exhausted=halted is None,
                           k=k, patience=patience, pc=pc, pc_smoothed=sm)


def naive_argmin(epochs: Sequence[dict]) -> int:
    """DEGENERATE VARIANT, not the baseline: the global argmin of the raw PC curve.

    Eq. (4) asks for the FIRST local minimum, not the global one. This is reported only
    to show what the shortcut would have picked, and must never be labelled Label Wave.
    """
    pc = prediction_changes(epochs)
    return int(np.argmin(pc)) + 1


def summarize(result: LabelWaveResult) -> dict:
    return dict(stop_epoch=result.stop_epoch, halted_epoch=result.halted_epoch,
                exhausted=result.exhausted, k=result.k, patience=result.patience)
