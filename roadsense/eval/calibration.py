"""Confidence calibration and selective prediction.

A sign classifier in a driving context must know when it does not know: the pipeline
abstains ("unrecognised sign") below a confidence threshold chosen on validation data.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Single-parameter temperature scaling (Guo et al., 2017), fitted by NLL on held-out data."""
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def ece(probs: np.ndarray, y: np.ndarray, bins=15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def risk_coverage(probs: np.ndarray, y: np.ndarray):
    """Error rate among accepted samples as the confidence threshold sweeps."""
    conf = probs.max(1)
    order = np.argsort(-conf)
    wrong = (probs.argmax(1) != y)[order]
    n = np.arange(1, len(y) + 1)
    return n / len(y), np.cumsum(wrong) / n, conf[order]


def threshold_for_risk(probs, y, target_risk) -> float:
    """Lowest confidence threshold whose accepted set has error <= target_risk."""
    cov, risk, conf = risk_coverage(probs, y)
    ok = np.where(risk <= target_risk)[0]
    return float(conf[ok[-1]]) if len(ok) else 1.0
