"""Dataset construction for G1 (torch/torchvision; runs in the torch container).

Loads CIFAR-10/100 from a LOCAL root (no download; the audit reuses the already
present ``cifar-10-batches-py`` / ``cifar-100-python``). Training labels are replaced
by the saved deterministic noisy labels so CE and ELR see identical corruption.
Held-out CLEAN test set is used for all R_ID / R_tail evaluation. OOD pools:
SVHN (semantic), the opposite CIFAR (semantic), and CIFAR-10-C/100-C (covariate).
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import torchvision
    import torchvision.transforms as T
except Exception:  # pragma: no cover - only importable in the torch container
    torchvision = None
    T = None

CIFAR_MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
CIFAR_STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}
N_CLASSES = {"cifar10": 10, "cifar100": 100}


def _tv_cifar(dataset: str, root: str, train: bool):
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    return cls(root=root, train=train, download=False)


def train_transform(dataset: str) -> "T.Compose":
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN[dataset], CIFAR_STD[dataset]),
    ])


def eval_transform(dataset: str) -> "T.Compose":
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN[dataset], CIFAR_STD[dataset])])


class ArrayImageDataset(Dataset):
    """(N,H,W,C) uint8 images + int labels + a torchvision transform on PIL."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        from PIL import Image
        img = Image.fromarray(self.images[i])
        return self.transform(img), int(self.labels[i])


def materialize_eval_tensors(ds: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a deterministic eval transform ONCE and keep the result in RAM.

    Eval transforms carry no randomness, so a per-epoch loader can iterate cached
    tensors instead of re-decoding PIL images every epoch. The tensors are
    bit-identical to transforming on the fly; this only removes repeated CPU work
    and the dataloader worker processes that exhausted the container's /dev/shm.
    """
    x0, _ = ds[0]
    xs = torch.empty((len(ds), *x0.shape), dtype=torch.float32)
    ys = torch.empty(len(ds), dtype=torch.int64)
    for i in range(len(ds)):
        x, y = ds[i]
        xs[i] = x
        ys[i] = y
    return xs, ys


def clean_labels(dataset: str, root: str, train: bool) -> np.ndarray:
    ds = _tv_cifar(dataset, root, train)
    return np.asarray(ds.targets, dtype=np.int64)


def make_train_dataset(dataset: str, root: str, noisy_labels: np.ndarray) -> ArrayImageDataset:
    ds = _tv_cifar(dataset, root, train=True)
    images = ds.data  # (N,32,32,3) uint8
    assert len(noisy_labels) == len(images)
    return ArrayImageDataset(images, noisy_labels, train_transform(dataset))


def make_clean_test_dataset(dataset: str, root: str) -> ArrayImageDataset:
    ds = _tv_cifar(dataset, root, train=False)
    return ArrayImageDataset(ds.data, np.asarray(ds.targets, np.int64), eval_transform(dataset))


def make_ood_svhn(root: str, id_dataset: str) -> ArrayImageDataset:
    svhn = torchvision.datasets.SVHN(root=os.path.join(root, "svhn"), split="test", download=True)
    images = np.transpose(svhn.data, (0, 2, 3, 1))  # (N,32,32,3)
    labels = np.zeros(len(images), dtype=np.int64)
    return ArrayImageDataset(images, labels, eval_transform(id_dataset))


def make_ood_cross_cifar(id_dataset: str, root: str) -> ArrayImageDataset:
    other = "cifar100" if id_dataset == "cifar10" else "cifar10"
    ds = _tv_cifar(other, root, train=False)
    return ArrayImageDataset(ds.data, np.zeros(len(ds.data), np.int64), eval_transform(id_dataset))
