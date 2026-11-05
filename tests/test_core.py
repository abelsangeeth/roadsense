import math

import numpy as np
import pytest
import torch
from PIL import Image

from roadsense.advisory import MU, advise
from roadsense.data.augment import random_affine, sign_flip
from roadsense.data.prepare import crop_sign
from roadsense.eval.calibration import ece, fit_temperature, threshold_for_risk
from roadsense.eval.corruptions import CORRUPTIONS
from roadsense.eval.detection_metrics import average_precision, box_iou, detection_report, match
from roadsense.eval.gradcam import _invert_affine, signnet_cam
from roadsense.labels import (FLIP_SELF, FLIP_SWAP, GTSDB_TO_GTSRB, GTSRB_CLASSES, IGNORE, RSCD_CLASSES,
                              SIGN_GROUP, compose, decompose)
from roadsense.models.signnet import SignNet
from roadsense.models.surfacenet import SurfaceCNN, joint_probs


# ------------------------------------------------------------------ labels
def test_label_spaces():
    assert len(GTSRB_CLASSES) == 43 and len(SIGN_GROUP) == 43
    assert sorted(GTSDB_TO_GTSRB.values()) == list(range(43)), "GTSDB map must be a bijection onto GTSRB"
    for a, b in FLIP_SWAP.items():
        assert FLIP_SWAP[b] == a
    assert not set(FLIP_SELF) & set(FLIP_SWAP)


@pytest.mark.parametrize("name", RSCD_CLASSES)
def test_decompose_compose_roundtrip(name):
    f, m, u = decompose(name)
    assert compose(f, max(m, 0), max(u, 0)) == name
    if name in ("fresh_snow", "melted_snow", "ice"):
        assert m == IGNORE and u == IGNORE
    if "gravel" in name or "mud" in name:
        assert u == IGNORE


def test_joint_probs_is_a_distribution():
    out = {"friction": torch.randn(5, 6), "material": torch.randn(5, 4), "unevenness": torch.randn(5, 3)}
    p = joint_probs(out)
    assert p.shape == (5, 27)
    assert torch.allclose(p.sum(1), torch.ones(5), atol=1e-5)


# ------------------------------------------------------------------ augmentation
def test_sign_flip_semantics():
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float).view(2, 3, 4, 4).repeat(50, 1, 1, 1)
    for cls, expect in [(1, {1}), (33, {33, 34}), (13, {13})]:  # speed limit never flips
        y = torch.full((100,), cls)
        torch.manual_seed(0)
        xf, yf = sign_flip(x, y, p=1.0)
        assert set(yf.tolist()) <= expect
        if cls == 1:
            assert torch.equal(xf, x)
        else:
            assert torch.equal(xf, x.flip(3))


def test_identity_affine():
    x = torch.rand(4, 3, 16, 16)
    y = random_affine(x)  # rot=0, scale=1, trans=0
    assert torch.allclose(x, y, atol=1e-5)


def test_crop_sign_centres_box():
    img = Image.new("RGB", (100, 100))
    img.paste((255, 0, 0), (40, 40, 60, 60))
    c = crop_sign(img, (40, 40, 60, 60), margin=0.15, size=26)
    assert c.shape == (26, 26, 3)
    assert c[13, 13, 0] == 255 and c[0, 0, 0] == 0  # sign in the centre, margin at the border


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_corruptions_shape_range_deterministic(name):
    x = torch.rand(3, 3, 48, 48)
    for s in range(3):
        a, b = CORRUPTIONS[name](x, s, seed=1), CORRUPTIONS[name](x, s, seed=1)
        assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
        assert torch.allclose(a, b)


# ------------------------------------------------------------------ models
def test_signnet_shapes_and_stn_identity_init():
    m = SignNet().eval()
    x = torch.rand(2, 3, 48, 48)
    logits, theta = m(x, return_theta=True)
    assert logits.shape == (2, 43)
    assert torch.allclose(theta, torch.tensor([[1., 0, 0], [0, 1, 0]]).expand(2, 2, 3), atol=1e-6)


def test_surface_heads():
    out = SurfaceCNN().eval()(torch.rand(2, 3, 128, 192))
    assert {k: v.shape[1] for k, v in out.items()} == {"friction": 6, "material": 4, "unevenness": 3}


def test_invert_affine():
    theta = torch.tensor([[[0.8, 0.1, 0.2], [-0.1, 0.9, -0.1]]])
    inv = _invert_affine(theta)
    pt = torch.tensor([0.3, -0.4, 1.0])
    fwd = theta[0] @ pt
    back = inv[0] @ torch.cat([fwd, torch.ones(1)])
    assert torch.allclose(back, pt[:2], atol=1e-6)


def test_gradcam_runs():
    cam, logits, theta = signnet_cam(SignNet(), torch.rand(2, 3, 48, 48))
    assert cam.shape == (2, 1, 48, 48) and cam.min() >= 0 and cam.max() <= 1 + 1e-6


# ------------------------------------------------------------------ metrics
def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[5, 0, 15, 10], [0, 0, 10, 10], [20, 20, 30, 30]], float)
    assert np.allclose(box_iou(a, b), [[50 / 150, 1.0, 0.0]])


def test_ap_perfect_and_half():
    gt = [(np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), np.zeros(2, int))]
    perfect = [(gt[0][0].copy(), np.array([0.9, 0.8]), np.zeros(2, int))]
    assert detection_report(perfect, gt)["AP50"] == pytest.approx(1.0)
    # one TP at top score, one FP, one GT missed -> P=1 up to R=0.5, then nothing
    half = [(np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float), np.array([0.9, 0.8]), np.zeros(2, int))]
    s, tp, n = match(half, gt, 0.5)
    assert average_precision(s, tp, n) == pytest.approx(51 / 101)


def test_duplicate_detection_is_false_positive():
    gt = [(np.array([[0, 0, 10, 10]], float), np.zeros(1, int))]
    dup = [(np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float), np.array([0.9, 0.8]), np.zeros(2, int))]
    s, tp, _ = match(dup, gt, 0.5)
    assert tp.tolist() == [True, False]


def test_classwise_matching():
    gt = [(np.array([[0, 0, 10, 10]], float), np.array([3]))]
    wrong = [(np.array([[0, 0, 10, 10]], float), np.array([0.9]), np.array([4]))]
    assert detection_report(wrong, gt, classwise=True)["AP50"] == 0.0
    assert detection_report(wrong, gt, classwise=False)["AP50"] == pytest.approx(1.0)


def test_calibration_tools():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    logits = torch.randn(2000, 5)
    logits[torch.arange(2000), torch.from_numpy(y)] += 2.0
    over = logits * 4  # overconfident
    yt = torch.from_numpy(y)
    t = fit_temperature(over, yt)
    grid = torch.linspace(0.5, 8, 751)
    nll = torch.stack([torch.nn.functional.cross_entropy(over / g, yt) for g in grid])
    assert abs(t - float(grid[nll.argmin()])) < 0.02  # LBFGS finds the NLL minimum
    p_over = over.softmax(1).numpy()
    p_cal = (over / t).softmax(1).numpy()
    assert ece(p_cal, y) < ece(p_over, y)
    thr = threshold_for_risk(p_cal, y, 0.05)
    keep = p_cal.max(1) >= thr
    assert (p_cal.argmax(1)[keep] != y[keep]).mean() <= 0.05 + 1e-9


# ------------------------------------------------------------------ advisory
def _surface(**p):
    f = {k: 0.0 for k in MU}
    f.update(p)
    return {"friction": f, "unevenness": None}


def test_advisory_physics():
    sign = [{"class_id": 5, "conf": 0.99}]  # 80 km/h
    dry = advise(sign, _surface(dry=1.0))
    assert dry["posted_limit_kmh"] == 80 and dry["recommended_kmh"] == 80 and dry["hazard_level"] == 0
    ice = advise(sign, _surface(ice=1.0))
    expect = 80 * math.sqrt(MU["ice"] / MU["dry"])
    assert abs(ice["recommended_kmh"] - expect) <= 2.5 and ice["hazard_level"] == 3
    speeds = [advise(sign, _surface(**{k: 1.0}))["recommended_kmh"] for k in ("dry", "wet", "water", "fresh_snow", "ice")]
    assert speeds == sorted(speeds, reverse=True)


def test_advisory_without_sign_keeps_previous_limit():
    a = advise([], _surface(wet=1.0), current_limit=50)
    assert a["posted_limit_kmh"] == 50 and a["recommended_kmh"] < 50


def test_advisory_road_damage():
    sign = [{"class_id": 7, "conf": 0.99}]  # 100 km/h
    pot = advise(sign, None, damage=[{"name": "pothole", "score": 0.9}])
    assert pot["hazard_level"] == 2 and pot["recommended_kmh"] == 75
    allig = advise(sign, None, damage=[{"name": "alligator crack", "score": 0.9}])
    assert allig["hazard_level"] == 1 and allig["recommended_kmh"] == 90
    crack = advise(sign, None, damage=[{"name": "transverse crack", "score": 0.9}])
    assert crack["hazard_level"] == 0 and crack["recommended_kmh"] == 100
    # damage never raises a speed already lowered by friction
    both = advise(sign, _surface(ice=1.0), damage=[{"name": "alligator crack", "score": 0.9}])
    assert both["recommended_kmh"] == advise(sign, _surface(ice=1.0))["recommended_kmh"]


def test_damage_detector_builds():
    from roadsense.models.detector import build_damage_detector
    m = build_damage_detector(pretrained=False).eval()
    with torch.no_grad():
        out = m([torch.rand(3, 300, 400)])[0]
    assert set(out) == {"boxes", "labels", "scores"} and m.roi_heads.box_predictor.cls_score.out_features == 5
