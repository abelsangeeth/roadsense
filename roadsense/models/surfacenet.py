"""Road-surface condition networks with three factorised heads.

friction (6) x material (4) x unevenness (3) instead of one flat 27-way softmax: the
heads share evidence across classes (e.g. "severe" looks alike on wet and dry asphalt),
heads whose attribute is undefined are masked out of the loss (snow hides the
material), and the advisory layer needs the factors separately anyway.
"""
import torch
from torch import nn
import torchvision

from ..labels import FRICTION, MATERIAL, UNEVENNESS, RSCD_CLASSES
from .blocks import conv_bn, stage


class MultiHead(nn.Module):
    def __init__(self, dim, drop):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.friction = nn.Linear(dim, len(FRICTION))
        self.material = nn.Linear(dim, len(MATERIAL))
        self.unevenness = nn.Linear(dim, len(UNEVENNESS))

    def forward(self, z):
        z = self.drop(z)
        return {"friction": self.friction(z), "material": self.material(z), "unevenness": self.unevenness(z)}


class FlatHead(nn.Module):
    """Ablation: one 27-way softmax over the joint RSCD label."""

    def __init__(self, dim, drop):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, len(RSCD_CLASSES)))

    def forward(self, z):
        return {"joint": self.fc(z)}


class SurfaceCNN(nn.Module):
    """From-scratch SE-ResNet for 128x192 road patches."""

    def __init__(self, width=32, drop=0.3, head="multi"):
        super().__init__()
        w = width
        self.stem = nn.Sequential(conv_bn(3, w, 3, 2), conv_bn(w, w))           # 64x96
        self.features = nn.Sequential(
            stage(w, w * 2, 2, 2, True, 0.05),                                   # 32x48
            stage(w * 2, w * 4, 2, 2, True, 0.05),                               # 16x24
            stage(w * 4, w * 8, 2, 2, True, 0.05),                               # 8x12
            stage(w * 8, w * 16, 2, 2, True, 0.05),                              # 4x6
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = (MultiHead if head == "multi" else FlatHead)(w * 16, drop)

    def forward(self, x):
        return self.head(self.pool(self.features(self.stem(x))))

    @property
    def cam_layer(self):
        return self.features[3]


class SurfaceEffNet(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 backbone, same heads (transfer-learning variant)."""

    def __init__(self, drop=0.3, head="multi", pretrained=True):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.efficientnet_b0(weights=weights)
        self.features = net.features
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = (MultiHead if head == "multi" else FlatHead)(1280, drop)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

    @property
    def cam_layer(self):
        return self.features[-1]


class SurfaceConvNeXt(nn.Module):
    """ImageNet-pretrained ConvNeXt-Tiny backbone, same heads (strongest variant)."""

    def __init__(self, drop=0.3, head="multi", pretrained=True):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.convnext_tiny(weights=weights, stochastic_depth_prob=0.1)
        self.features = net.features
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(768))
        self.head = (MultiHead if head == "multi" else FlatHead)(768, drop)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

    @property
    def cam_layer(self):
        return self.features[-1]


def build_surface_model(arch: str, head: str = "multi", pretrained: bool = True) -> nn.Module:
    if arch == "cnn":
        return SurfaceCNN(head=head)
    if arch == "effnet_b0":
        return SurfaceEffNet(head=head, pretrained=pretrained)
    if arch == "convnext_tiny":
        return SurfaceConvNeXt(head=head, pretrained=pretrained)
    raise ValueError(arch)


@torch.no_grad()
def joint_probs(out: dict) -> torch.Tensor:
    """Probability of each of the 27 RSCD classes from either head type.

    For the factorised head, P(class) = P(friction) * P(material) * P(unevenness) over
    the attributes that class defines (heads are trained independently, so this is the
    naive-Bayes-style product; it is only used for the argmax / 27-way metrics).
    """
    if "joint" in out:
        return out["joint"].softmax(-1)
    pf, pm, pu = out["friction"].softmax(-1), out["material"].softmax(-1), out["unevenness"].softmax(-1)
    cols = []
    for name in RSCD_CLASSES:
        parts = name.split("_")
        if name in ("fresh_snow", "melted_snow", "ice"):
            cols.append(pf[:, FRICTION.index(name)])
            continue
        p = pf[:, FRICTION.index(parts[0])] * pm[:, MATERIAL.index(parts[1])]
        if len(parts) == 3:
            p = p * pu[:, UNEVENNESS.index(parts[2])]
        cols.append(p)
    return torch.stack(cols, -1)
