"""SOP — Sparse Over-Parameterization (plain SOP, not SOP+).

Liu, Zhu, Qu, You, "Robust Training under Label Noise by Over-parameterization",
ICML 2022. Transcribed from the official implementation, not reconstructed; the source
commit, the file it came from, and the paper-vs-repo discrepancies are recorded in
docs/TRANSCRIPTION.md.

The forward pass reproduces ``overparametrization_loss.forward`` from
shengliu66/SOP ``model/loss.py`` @ 4d991ce line for line, with the two SOP+ terms
disabled (lambda_C = lambda_B = 0), which is what the paper's own Table A.1 lists for
plain SOP:

    U_square = clamp(u[i]^2 * y_onehot,       0, 1)      # u is (N, 1)
    V_square = clamp(v[i]^2 * (1 - y_onehot), 0, 1)      # v is (N, C)
    p        = softmax(f(x))
    p_adj    = normalize_L1(clamp(p + U_square - V_square.detach(), min=eps))
    y_hard   = onehot(argmax f(x).detach())
    L        = mean(-sum(y_onehot * log(p_adj)))  +  ||y_hard + U_square - V_square - y_onehot||_F^2 / B

u and v are per-sample training-only parameters optimized by a SEPARATE SGD optimizer
with momentum 0, weight decay 0, and per-group learning rates alpha_u / alpha_v. They
are never part of evaluation: ``logits`` written by the forward pass are the raw f(x),
excluding the sparse term. ``assert_raw_logits`` exists so that invariant is enforced by
an assertion rather than by convention.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-4              # official: eps = 1e-4
INIT_MEAN = 0.0         # official config reparam_arch.args.mean
INIT_STD = 1e-8         # official config reparam_arch.args.std, and Table A.1


class SOPLoss:
    """Plain SOP. ``ratio_consistency`` and ``ratio_balance`` are fixed at 0."""

    name = "sop"

    def __init__(self, n_train: int, n_classes: int, lr_u: float, lr_v: float,
                 init_std: float = INIT_STD, init_mean: float = INIT_MEAN,
                 eps: float = EPS, device: str = "cpu"):
        self.n_train = int(n_train)
        self.n_classes = int(n_classes)
        self.lr_u = float(lr_u)
        self.lr_v = float(lr_v)
        self.eps = float(eps)
        self.u = nn.Parameter(torch.empty(self.n_train, 1, dtype=torch.float32,
                                          device=device))
        self.v = nn.Parameter(torch.empty(self.n_train, self.n_classes,
                                          dtype=torch.float32, device=device))
        torch.nn.init.normal_(self.u, mean=init_mean, std=init_std)
        torch.nn.init.normal_(self.v, mean=init_mean, std=init_std)
        # separate optimizer, per the official train.py param groups:
        # momentum 0, weight decay 0 for BOTH u and v
        self.optimizer = torch.optim.SGD(
            [{"params": [self.u], "lr": self.lr_u, "weight_decay": 0.0},
             {"params": [self.v], "lr": self.lr_v, "weight_decay": 0.0}],
            lr=self.lr_u, momentum=0.0, weight_decay=0.0)

    # ---- the loss ------------------------------------------------------------

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor,
                 indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if indices is None:
            raise ValueError("SOP needs per-sample dataset indices for u and v")
        idx = torch.as_tensor(indices, device=logits.device, dtype=torch.long)
        label = F.one_hot(targets, num_classes=self.n_classes).float()

        U_square = torch.clamp(self.u[idx] ** 2 * label, 0, 1)
        V_square = torch.clamp(self.v[idx] ** 2 * (1 - label), 0, 1)

        prediction = F.softmax(logits, dim=1)
        prediction = torch.clamp(prediction + U_square - V_square.detach(), min=self.eps)
        prediction = F.normalize(prediction, p=1, eps=self.eps)
        prediction = torch.clamp(prediction, min=self.eps, max=1.0)

        with torch.no_grad():                       # official soft_to_hard
            label_one_hot = F.one_hot(logits.detach().argmax(dim=1),
                                      num_classes=self.n_classes).float()

        mse = F.mse_loss(label_one_hot + U_square - V_square, label,
                         reduction="sum") / len(label)
        ce = torch.mean(-torch.sum(label * torch.log(prediction), dim=-1))
        return ce + mse

    # ---- the extra optimizer, driven by the trainer ---------------------------

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def to(self, device: str) -> "SOPLoss":
        if self.u.device.type != torch.device(device).type:
            with torch.no_grad():
                self.u = nn.Parameter(self.u.detach().to(device))
                self.v = nn.Parameter(self.v.detach().to(device))
            self.optimizer = torch.optim.SGD(
                [{"params": [self.u], "lr": self.lr_u, "weight_decay": 0.0},
                 {"params": [self.v], "lr": self.lr_v, "weight_decay": 0.0}],
                lr=self.lr_u, momentum=0.0, weight_decay=0.0)
        return self

    # ---- checkpointing --------------------------------------------------------

    def state(self) -> dict:
        return dict(u=self.u.detach().cpu(), v=self.v.detach().cpu(),
                    optimizer=self.optimizer.state_dict(),
                    lr_u=self.lr_u, lr_v=self.lr_v, eps=self.eps)

    def load_state(self, state: Optional[dict]) -> None:
        if not state:
            return
        for k in ("lr_u", "lr_v", "eps"):
            if k in state and float(state[k]) != getattr(self, k):
                raise ValueError(f"resumed SOP {k}={state[k]} differs from configured "
                                 f"{getattr(self, k)}")
        with torch.no_grad():
            self.u.copy_(state["u"].to(self.u.device))
            self.v.copy_(state["v"].to(self.v.device))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])


def assert_raw_logits(model, x: torch.Tensor, logits: torch.Tensor) -> None:
    """Enforce that stored logits are the raw f(x), with no sparse term added.

    The preregistration requires SOP's stored logits to exclude s_i = u_i^2 - v_i^2.
    Rather than trusting that the forward-pass writer never adds it, this recomputes
    f(x) from the model and asserts bitwise equality with what is about to be written.
    """
    with torch.no_grad():
        raw = model(x)
    if not torch.equal(raw, logits):
        raise AssertionError(
            "stored logits are not the raw f(x): a sparse term or other adjustment has "
            f"been applied (max abs diff {float((raw - logits).abs().max()):.3e})")
