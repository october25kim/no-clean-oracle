"""Generalized Cross Entropy (GCE).

Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with
Noisy Labels", NeurIPS 2018.

    L_q(f(x), y) = (1 - f_y(x)^q) / q ,   q in (0, 1]

with f_y the softmax probability of the given label. q -> 0 recovers cross-entropy and
q = 1 recovers mean absolute error; the paper's CIFAR default is q = 0.7, which is what
is pinned here. The truncated variant (L_q with a pruning threshold and alternating
optimization) is NOT implemented -- the extension preregistration pins "no truncation
variant; loss replacement only, all else identical to CE".

Loss replacement only: no per-sample state, no extra parameters, no schedule change.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GCELoss:
    name = "gce"

    def __init__(self, n_train: int, n_classes: int, q: float = 0.7, eps: float = 1e-7):
        if not 0.0 < q <= 1.0:
            raise ValueError(f"GCE q must lie in (0, 1]; got {q}")
        self.n_train = int(n_train)
        self.n_classes = int(n_classes)
        self.q = float(q)
        self.eps = float(eps)

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor,
                 indices=None) -> torch.Tensor:
        p = F.softmax(logits, dim=1)
        p_y = p.gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(self.eps)
        return ((1.0 - p_y.pow(self.q)) / self.q).mean()

    # GCE is stateless; state/load_state are omitted so the trainer's
    # hasattr checks skip it, exactly as they do for CE.
