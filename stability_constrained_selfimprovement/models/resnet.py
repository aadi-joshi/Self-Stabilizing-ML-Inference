# ============================================================================
# ResNet-based Models for Continual Learning (Experiment A)
# Small/Medium/Large variants for ablation
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BasicBlock(nn.Module):
    """Basic residual block for small ResNets."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SmallResNet(nn.Module):
    """
    Configurable small ResNet for CIFAR-scale experiments.
    
    Architecture scales:
        small:  [1, 1, 1, 1] blocks, base_width=16  (~44K params)
        medium: [2, 2, 2, 2] blocks, base_width=32  (~270K params)
        large:  [2, 2, 2, 2] blocks, base_width=64  (~1.1M params)
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: Optional[list] = None,
        base_width: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]

        self.in_planes = base_width
        self.conv1 = nn.Conv2d(in_channels, base_width, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)

        self.layer1 = self._make_layer(base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_width * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_width * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_width * 8, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representation (before classification head)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.fc(feat)


def build_resnet(variant: str = "small", num_classes: int = 10) -> SmallResNet:
    """Factory function for ResNet variants."""
    configs = {
        "resnet18_small": {"num_blocks": [1, 1, 1, 1], "base_width": 16},
        "resnet18_medium": {"num_blocks": [2, 2, 2, 2], "base_width": 32},
        "resnet18_large": {"num_blocks": [2, 2, 2, 2], "base_width": 64},
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    cfg = configs[variant]
    return SmallResNet(num_classes=num_classes, **cfg)
