"""RoadSense-SignNet: STN + SE-ResNet classifier for 48x48 traffic-sign crops."""
import torch
from torch import nn
import torch.nn.functional as F

from .blocks import conv_bn, stage


class STN(nn.Module):
    """Spatial transformer (Jaderberg et al., 2015) that learns to re-centre, de-rotate and
    re-scale the sign before classification. Initialised to the identity transform so
    training starts from the plain CNN and only bends the grid when it helps."""

    def __init__(self):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), nn.MaxPool2d(2), nn.SiLU(),   # 24
            nn.Conv2d(16, 32, 5, padding=2), nn.MaxPool2d(2), nn.SiLU(),  # 12
            nn.Conv2d(32, 64, 3, padding=1), nn.MaxPool2d(2), nn.SiLU(),  # 6
            nn.AdaptiveAvgPool2d(3), nn.Flatten(),
            nn.Linear(64 * 9, 64), nn.SiLU(), nn.Linear(64, 6),
        )
        nn.init.zeros_(self.loc[-1].weight)
        self.loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.loc(x).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="border"), theta


class SignNet(nn.Module):
    def __init__(self, num_classes=43, width=32, use_stn=True, use_se=True, drop=0.3, drop_path=0.05):
        super().__init__()
        w = width
        self.stn = STN() if use_stn else None
        self.stem = conv_bn(3, w)
        self.features = nn.Sequential(
            stage(w, w * 2, 2, 1, use_se, drop_path), nn.MaxPool2d(2),          # 24
            stage(w * 2, w * 4, 2, 1, use_se, drop_path), nn.MaxPool2d(2),      # 12
            stage(w * 4, w * 8, 2, 1, use_se, drop_path), nn.MaxPool2d(2),      # 6
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(w * 8, num_classes))

    def forward(self, x, return_theta=False):
        theta = None
        if self.stn is not None:
            x, theta = self.stn(x)
        logits = self.head(self.features(self.stem(x)))
        return (logits, theta) if return_theta else logits

    @property
    def cam_layer(self):
        return self.features[4]  # last 12x12 stage before the final pool: best CAM resolution / semantics trade-off
