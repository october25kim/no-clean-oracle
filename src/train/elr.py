"""Early-Learning Regularization (ELR).

Liu, Niles-Weed, Razavian, Fernandez-Granda, "Early-Learning Regularization Prevents
Memorization of Noisy Labels", NeurIPS 2020. This is plain ELR (not ELR+): no mixup,
no weight averaging, no two-network ensemble -- the audit needs CE and ELR to differ
ONLY in the training objective.

    t_i  <- beta * t_i + (1 - beta) * p_i / sum(p_i)          (no gradient)
    L    =  CE(f(x_i), y_i) + lambda * mean_i log(1 - <t_i, p_i>)

``t_i`` is a per-sample running estimate of the model's own early-learning prediction.
Because <t_i, p_i> in (0, 1), the regularizer is negative and minimizing it pulls the
prediction back toward that estimate, which is what stops the network from later
fitting the corrupted label. Hyperparameters come from configs/base.yaml
(``learners.elr.per_dataset``): CIFAR-10 lambda=3, beta=0.7; CIFAR-100 lambda=7,
beta=0.9, as in the paper.

The target bank is a plain (n_train, n_classes) tensor, so it is part of the run's
state and is checkpointed via ``state``/``load_state`` for crash-safe resume.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class ELRLoss:
    name = "elr"

    def __init__(self, n_train: int, n_classes: int, lam: float = 3.0, beta: float = 0.7,
                 eps: float = 1e-4):
        self.n_train = int(n_train)
        self.n_classes = int(n_classes)
        self.lam = float(lam)
        self.beta = float(beta)
        self.eps = float(eps)
        # t_i, zero-initialised: the first update sets it to (1 - beta) * p_i.
        self.target = torch.zeros(self.n_train, self.n_classes, dtype=torch.float32)

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor,
                 indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if indices is None:
            raise ValueError("ELR needs per-sample dataset indices to key its target bank")
        if self.target.device != logits.device:
            self.target = self.target.to(logits.device)
        idx = torch.as_tensor(indices, device=logits.device, dtype=torch.long)

        p = F.softmax(logits, dim=1).clamp(self.eps, 1.0 - self.eps)
        with torch.no_grad():
            q = p.detach()
            q = q / q.sum(dim=1, keepdim=True)
            self.target[idx] = self.beta * self.target[idx] + (1.0 - self.beta) * q
        # <t, p> < 1 strictly: t rows sum to <= 1 and every p entry is <= 1 - eps.
        reg = torch.log(1.0 - (self.target[idx] * p).sum(dim=1)).mean()
        return F.cross_entropy(logits, targets) + self.lam * reg

    # ---- checkpointing -------------------------------------------------------

    def state(self) -> dict:
        return dict(target=self.target.detach().cpu(), lam=self.lam, beta=self.beta,
                    eps=self.eps, n_train=self.n_train, n_classes=self.n_classes)

    def load_state(self, state: Optional[dict]) -> None:
        if not state:
            return
        t = state["target"]
        if tuple(t.shape) != (self.n_train, self.n_classes):
            raise ValueError(f"ELR target bank shape {tuple(t.shape)} does not match "
                             f"({self.n_train}, {self.n_classes})")
        self.target = t.to(self.target.device).float()
        for k in ("lam", "beta", "eps"):
            if k in state and float(state[k]) != getattr(self, k):
                raise ValueError(f"resumed ELR {k}={state[k]} differs from configured "
                                 f"{getattr(self, k)}")
