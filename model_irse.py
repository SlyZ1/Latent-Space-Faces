"""
iResNet-50 with Squeeze-and-Excitation blocks (ir_se50).
Architecture matches the weights from TreB1eN/InsightFace_Pytorch,
used by the pSp encoder for ArcFace identity loss.

Input:  [B, 3, 112, 112], normalized to [-1, 1]
Output: [B, 512], L2-normalized embeddings
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

ARCFACE_HF_REPO = "matbee/model_ir_se50_pytorch"
ARCFACE_HF_FILE = "model_ir_se50.pth"


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return x * scale


class IrSEBlock(nn.Module):
    """Residual block with BN-Conv-PReLU-Conv-BN-SE.

    Shortcut follows TreB1eN convention:
    - same channels: MaxPool2d(1, stride)  — no learnable params
    - different channels: Conv(1×1) + BN
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEBlock(out_channels),
        )
        if in_channels == out_channels:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


def _make_layer(in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
    layers = [IrSEBlock(in_ch, out_ch, stride=2)]
    for _ in range(num_blocks - 1):
        layers.append(IrSEBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)


class IrSE50(nn.Module):
    """iResNet-50 with SE blocks. Matches `model_ir_se50.pth` weight layout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.body = nn.Sequential(
            *_make_layer(64, 64, 3),   # blocks 0-2
            *_make_layer(64, 128, 4),  # blocks 3-6
            *_make_layer(128, 256, 14), # blocks 7-20
            *_make_layer(256, 512, 3), # blocks 21-23
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return F.normalize(x, dim=1)


def load_ir_se50(local_cache: Path | None = None) -> IrSE50:
    """Load pretrained ir_se50 weights. Downloads from HuggingFace if not cached."""
    if local_cache is not None and local_cache.exists():
        weights_path = str(local_cache)
    else:
        weights_path = hf_hub_download(
            repo_id=ARCFACE_HF_REPO, filename=ARCFACE_HF_FILE
        )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = IrSE50()
    model.load_state_dict(state)
    return model
