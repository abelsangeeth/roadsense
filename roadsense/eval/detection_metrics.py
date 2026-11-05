"""Detection metrics (COCO-style 101-point interpolated AP), dependency-free."""
from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), -1)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], -1)
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def match(preds, gts, iou_thr, classwise=False):
    """Greedy score-ordered matching per image.

    preds: list of (boxes[N,4], scores[N], labels[N]); gts: list of (boxes[M,4], labels[M]).
    Returns (scores, is_tp) over all predictions and the number of GT boxes.
    """
    all_s, all_tp, n_gt = [], [], 0
    for (pb, ps, pl), (gb, gl) in zip(preds, gts):
        n_gt += len(gb)
        order = np.argsort(-ps)
        pb, ps, pl = pb[order], ps[order], pl[order]
        iou = box_iou(pb, gb)
        used = np.zeros(len(gb), bool)
        for i in range(len(pb)):
            cand = iou[i].copy() if len(gb) else np.zeros(0)
            if len(gb):
                cand[used] = -1
                if classwise:
                    cand[gl != pl[i]] = -1
            j = int(np.argmax(cand)) if len(cand) else -1
            tp = j >= 0 and cand[j] >= iou_thr
            if tp:
                used[j] = True
            all_s.append(ps[i])
            all_tp.append(tp)
    return np.array(all_s), np.array(all_tp, bool), n_gt


def average_precision(scores, tp, n_gt) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rs, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def detection_report(preds, gts, classwise=False) -> dict:
    out = {}
    aps = []
    for thr in np.arange(0.5, 0.96, 0.05):
        s, tp, n = match(preds, gts, thr, classwise)
        ap = average_precision(s, tp, n)
        aps.append(ap)
        if abs(thr - 0.5) < 1e-6:
            out["AP50"] = ap
            # operating point at score >= 0.5
            keep = s >= 0.5
            out["recall@0.5score"] = float(tp[keep].sum() / max(n, 1))
            out["precision@0.5score"] = float(tp[keep].mean()) if keep.any() else float("nan")
        if abs(thr - 0.75) < 1e-6:
            out["AP75"] = ap
    out["AP50:95"] = float(np.mean(aps))
    out["n_gt"] = int(sum(len(g[0]) for g in gts))
    out["n_images"] = len(gts)
    return out
