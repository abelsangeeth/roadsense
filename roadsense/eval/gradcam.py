"""Grad-CAM (Selvaraju et al., 2017) for SignNet and the surface models.

For SignNet the CAM is computed after the spatial transformer, i.e. in the STN's
re-sampled frame; `signnet_cam` maps it back to input pixels with the inverse affine
so the heatmap overlays the image the user actually supplied.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self.h1 = layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        self.h2 = layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, score_fn):
        """score_fn(model_output) -> (N,) scalar per sample to explain."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        self.model.zero_grad(set_to_none=True)
        score_fn(out).sum().backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam / cam.flatten(1).amax(1).clamp(min=1e-8).view(-1, 1, 1, 1)
        return cam.detach(), out


def _invert_affine(theta):  # (N,2,3) output->input map  ->  input->output map
    A, t = theta[:, :, :2], theta[:, :, 2:]
    Ai = torch.linalg.inv(A)
    return torch.cat([Ai, -Ai @ t], 2)


def signnet_cam(model, x, target=None):
    """Returns (cam in input coordinates [N,1,H,W], logits, theta)."""
    cam_fn = GradCAM(model, model.cam_layer)
    try:
        out_holder = {}

        def score(logits):
            out_holder["logits"] = logits
            t = logits.argmax(1) if target is None else target
            return logits.gather(1, t.view(-1, 1)).squeeze(1)

        cam, _ = cam_fn(x, score)
        theta = None
        if model.stn is not None:
            with torch.no_grad():
                _, theta = model.stn(x)
                grid = F.affine_grid(_invert_affine(theta), cam.shape, align_corners=False)
                cam = F.grid_sample(cam, grid, align_corners=False, padding_mode="zeros")
        return cam, out_holder["logits"].detach(), theta
    finally:
        cam_fn.remove()


def surface_cam(model, x, head="friction", target=None):
    cam_fn = GradCAM(model, model.cam_layer)
    try:
        def score(out):
            logits = out[head] if head in out else out["joint"]
            t = logits.argmax(1) if target is None else target
            return logits.gather(1, t.view(-1, 1)).squeeze(1)
        cam, out = cam_fn(x, score)
        return cam, {k: v.detach() for k, v in out.items()}
    finally:
        cam_fn.remove()
