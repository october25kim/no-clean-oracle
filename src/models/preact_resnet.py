"""PreAct ResNet-18 (He et al. 2016 identity-mapping variant) for CIFAR 32x32.

Same backbone for CE and ELR. Standard CIFAR stem (3x3, stride 1, no maxpool).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, 1, stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate 512-d embedding -- the input to ``linear``.

        Split out for the representation-based analyses, which need the embedding and
        the logits from the SAME pass. ``forward`` routes through it, so the logits are
        computed by exactly the same ops as before this split existed.
        """
        out = self.conv1(x)
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = F.relu(self.bn(out))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(x))


def preact_resnet18(num_classes: int) -> PreActResNet:
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes)


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "preact_resnet18":
        return preact_resnet18(num_classes)
    raise ValueError(f"unknown arch {arch!r}")
