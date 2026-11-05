import torch
from torch import nn
import torch.nn.functional as F


def conv_bn(cin, cout, k=3, s=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class SE(nn.Module):
    """Squeeze-and-excitation channel attention (Hu et al., 2018)."""

    def __init__(self, c, r=8):
        super().__init__()
        self.fc1, self.fc2 = nn.Conv2d(c, max(c // r, 8), 1), nn.Conv2d(max(c // r, 8), c, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(w))))


class ResBlock(nn.Module):
    """Pre-activation-free basic residual block with optional SE attention."""

    def __init__(self, cin, cout, stride=1, se=True, drop_path=0.0):
        super().__init__()
        self.body = nn.Sequential(conv_bn(cin, cout, 3, stride), conv_bn(cout, cout, 3, 1, act=False))
        self.se = SE(cout) if se else nn.Identity()
        self.skip = nn.Identity() if (cin == cout and stride == 1) else conv_bn(cin, cout, 1, stride, act=False)
        self.drop_path = drop_path

    def forward(self, x):
        y = self.se(self.body(x))
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
            y = y * keep / (1 - self.drop_path)
        return F.silu(y + self.skip(x))


def stage(cin, cout, n, stride, se, drop_path=0.0):
    return nn.Sequential(*[ResBlock(cin if i == 0 else cout, cout, stride if i == 0 else 1, se, drop_path) for i in range(n)])
