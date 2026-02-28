# -*- coding: utf-8 -*-
"""3D ResNet18 backbone（支持 layer4 后 return_spatial 用于 FiLM）。"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SE3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, use_attention=False, attn_type="SE"):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.attn = SE3D(planes, reduction=16) if (use_attention and attn_type == "SE") else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.attn(out)
        out = self.relu(out)
        return out


class ResNet18_3D_Encoder(nn.Module):
    DF = 512

    def __init__(self, in_channels=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock3D, 64, 2, stride=1, use_attention=False, attn_type="SE")
        self.layer2 = self._make_layer(BasicBlock3D, 128, 2, stride=2, use_attention=False, attn_type="SE")
        self.layer3 = self._make_layer(BasicBlock3D, 256, 2, stride=2, use_attention=False, attn_type="SE")
        self.layer4 = self._make_layer(BasicBlock3D, 512, 2, stride=2, use_attention=False, attn_type="SE")

    def _make_layer(self, block, planes, blocks, stride=1, use_attention=False, attn_type="SE"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [
            block(self.inplanes, planes, stride=stride, downsample=downsample, use_attention=use_attention, attn_type=attn_type)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_attention=use_attention, attn_type=attn_type))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_spatial:
            return x
        x = F.adaptive_avg_pool3d(x, 1).flatten(1)
        return x
