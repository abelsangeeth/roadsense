"""End-to-end RoadSense inference: scene -> signs + road surface -> driving advisory.

    rs = RoadSense.load("runs/release")
    result = rs.analyze(Image.open("frame.jpg"))           # full dashcam frame
    result = rs.analyze(Image.open("patch.jpg"), mode="surface")  # close-up road patch
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .advisory import advise
from .data.augment import normalize
from .data.prepare import SURFACE_HW, crop_sign
from .labels import FRICTION, GTSRB_CLASSES, MATERIAL, RSCD_CLASSES, SIGN_GROUP, UNEVENNESS
from .data.rdd import DAMAGE_CLASSES
from .models.detector import build_damage_detector, build_detector, set_eval_size
from .models.signnet import SignNet
from .models.surfacenet import build_surface_model, joint_probs

# Road region of a forward-facing dashcam frame used for the surface model: the lower
# centre, where the lane directly ahead of the bonnet is. RSCD patches are close-range
# views of the road ahead, so this is the closest dashcam analogue.
ROAD_ROI = (0.30, 0.62, 0.70, 0.92)  # x1, y1, x2, y2 as fractions of width / height


def _tensor(arr_u8: np.ndarray, dev) -> torch.Tensor:
    """uint8 HWC image or NHWC batch -> normalised float NCHW."""
    x = torch.from_numpy(np.ascontiguousarray(arr_u8)).to(dev)
    x = x.unsqueeze(0) if x.ndim == 3 else x
    return normalize(x.permute(0, 3, 1, 2).float().div(255))


def road_roi(img: Image.Image, roi=ROAD_ROI) -> Image.Image:
    w, h = img.size
    return img.crop((int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)))


class RoadSense:
    def __init__(self, signnet, surface, detector, calib: dict, dev, damage=None):
        self.signnet, self.surface, self.detector, self.damage = signnet, surface, detector, damage
        self.calib, self.dev = calib, dev
        self.T = calib.get("sign_temperature", 1.0)
        self.sign_thr = calib.get("sign_accept_threshold", 0.5)
        self.det_thr = calib.get("detector_threshold", 0.5)
        self.surface_T = calib.get("surface_joint_temperature", 1.0)
        self.damage_thr = calib.get("damage_threshold", 0.5)

    @classmethod
    def load(cls, release: str | Path, device: str | None = None) -> "RoadSense":
        release = Path(release)
        dev = torch.device(device or ("cuda" if torch.cuda.device_count() > 0 else "cpu"))  # is_available() can be True with 0 visible devices
        calib = json.loads((release / "calibration.json").read_text()) if (release / "calibration.json").exists() else {}

        ck = torch.load(release / "signnet.pt", map_location=dev, weights_only=False)
        c = ck["config"]
        signnet = SignNet(width=c.get("width", 32), use_stn=not c.get("no_stn"), use_se=not c.get("no_se"))
        signnet.load_state_dict(ck["model"])

        surface = None
        if (release / "surface.pt").exists():
            ck = torch.load(release / "surface.pt", map_location=dev, weights_only=False)
            surface = build_surface_model(ck["config"]["arch"], ck["config"]["head"], pretrained=False)
            surface.load_state_dict(ck["model"])
            surface.input_hw = tuple(ck["config"].get("hw", SURFACE_HW))
            surface = surface.to(dev).eval()

        if surface is not None and (release / "surface_ood.npz").exists():
            from .eval.fit_ood import MahalanobisGate
            surface.ood = MahalanobisGate.load(release / "surface_ood.npz")

        detector = None
        if (release / "detector.pt").exists():
            detector = build_detector(pretrained=False)
            detector.load_state_dict(torch.load(release / "detector.pt", map_location=dev, weights_only=False)["model"])
            detector = set_eval_size(detector.to(dev).eval(), 800)
        damage = None
        if (release / "damage.pt").exists():
            damage = build_damage_detector(pretrained=False)
            damage.load_state_dict(torch.load(release / "damage.pt", map_location=dev, weights_only=False)["model"])
            damage = set_eval_size(damage.to(dev).eval(), damage.test_size)
        return cls(signnet.to(dev).eval(), surface, detector, calib, dev, damage)

    # ------------------------------------------------------------------ parts
    @torch.no_grad()
    def classify_signs(self, img: Image.Image, boxes: np.ndarray):
        if len(boxes) == 0:
            return np.zeros((0, 43), np.float32), np.zeros((0, 48, 48, 3), np.uint8)
        crops = np.stack([crop_sign(img, b) for b in boxes])
        logits = self.signnet(_tensor(crops, self.dev))
        return (logits / self.T).softmax(1).cpu().numpy(), crops

    @torch.no_grad()
    def detect(self, img: Image.Image):
        x = torch.from_numpy(np.array(img)).to(self.dev).permute(2, 0, 1).float().div(255)
        out = self.detector([x])[0]
        keep = out["scores"] >= self.det_thr
        return out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()

    @torch.no_grad()
    def detect_damage(self, img: Image.Image) -> list[dict]:
        x = torch.from_numpy(np.array(img)).to(self.dev).permute(2, 0, 1).float().div(255)
        out = self.damage([x])[0]
        keep = out["scores"] >= self.damage_thr
        return [{"box": [round(float(v), 1) for v in b], "name": DAMAGE_CLASSES[int(l) - 1], "score": float(sc)}
                for b, l, sc in zip(out["boxes"][keep].cpu(), out["labels"][keep].cpu(), out["scores"][keep].cpu())]

    @torch.no_grad()
    def classify_surface(self, patch: Image.Image) -> dict:
        h, w = getattr(self.surface, "input_hw", SURFACE_HW)
        arr = np.asarray(patch.convert("RGB").resize((w, h), Image.BILINEAR))
        x = _tensor(arr, self.dev)
        out = self.surface(x.contiguous(memory_format=torch.channels_last))
        ood_score, in_domain = None, True
        if getattr(self.surface, "ood", None) is not None:
            from .eval.fit_ood import stat_features_float
            ood_score = float(self.surface.ood.score(stat_features_float(self.surface, x).double().cpu().numpy())[0])
            in_domain = ood_score <= self.surface.ood.threshold
        # temperature is fitted on log joint probabilities (works for flat and factorised heads)
        pj = (joint_probs(out).clamp_min(1e-9).log() / self.surface_T).softmax(-1)[0].cpu().numpy()
        res = {"class": RSCD_CLASSES[int(pj.argmax())], "confidence": float(pj.max()),
               "in_domain": bool(in_domain),
               "ood_ratio": None if ood_score is None else round(ood_score / self.surface.ood.threshold, 3),
               "top3": [(RSCD_CLASSES[i], float(pj[i])) for i in np.argsort(-pj)[:3]]}
        # Factor marginals from the joint distribution: valid for both head types.
        fr, ma, un = np.zeros(len(FRICTION)), np.zeros(len(MATERIAL)), np.zeros(len(UNEVENNESS))
        for p, name in zip(pj, RSCD_CLASSES):
            parts = name.split("_")
            if name in ("fresh_snow", "melted_snow", "ice"):
                fr[FRICTION.index(name)] += p
                continue
            fr[FRICTION.index(parts[0])] += p
            ma[MATERIAL.index(parts[1])] += p
            if len(parts) == 3:
                un[UNEVENNESS.index(parts[2])] += p
        res["friction"] = {k: float(v) for k, v in zip(FRICTION, fr)}
        ma_s, un_s = ma.sum(), un.sum()
        res["material"] = {k: float(v / ma_s) for k, v in zip(MATERIAL, ma)} if ma_s > 0.05 else None
        res["unevenness"] = {k: float(v / un_s) for k, v in zip(UNEVENNESS, un)} if un_s > 0.05 else None
        return res

    # ------------------------------------------------------------------ main
    def analyze(self, img: Image.Image, mode: str = "scene", current_limit: int | None = None) -> dict:
        """mode: 'scene' (dashcam frame), 'sign' (a single cropped sign), 'surface' (a road patch)."""
        img = img.convert("RGB")
        t0 = time.perf_counter()
        timings, signs, surface, damage = {}, [], None, []

        if mode == "sign":
            w, h = img.size
            m = 0.15 / 1.3  # treat the whole image as box + the training margin
            boxes = np.array([[w * m, h * m, w * (1 - m), h * (1 - m)]], np.float32)
            det_scores = np.ones(1, np.float32)
        elif mode == "scene" and self.detector is not None:
            boxes, det_scores = self.detect(img)
            timings["detect_ms"] = (time.perf_counter() - t0) * 1000
        else:
            boxes, det_scores = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

        t1 = time.perf_counter()
        probs, _ = self.classify_signs(img, boxes)
        timings["classify_ms"] = (time.perf_counter() - t1) * 1000
        for b, ds, p in zip(boxes, det_scores, probs):
            cid = int(p.argmax())
            signs.append({
                "box": [round(float(v), 1) for v in b], "det_score": float(ds),
                "class_id": cid, "name": GTSRB_CLASSES[cid], "group": SIGN_GROUP[cid],
                "conf": float(p[cid]), "accepted": bool(p[cid] >= self.sign_thr),
                "top3": [(GTSRB_CLASSES[i], float(p[i])) for i in np.argsort(-p)[:3]],
            })

        if self.surface is not None and mode in ("scene", "surface"):
            t2 = time.perf_counter()
            patch = road_roi(img) if mode == "scene" else img
            surface = self.classify_surface(patch)
            if mode == "scene":
                w, h = img.size
                surface["roi"] = [ROAD_ROI[0] * w, ROAD_ROI[1] * h, ROAD_ROI[2] * w, ROAD_ROI[3] * h]
            timings["surface_ms"] = (time.perf_counter() - t2) * 1000

        if self.damage is not None and mode == "scene":
            t3 = time.perf_counter()
            damage = self.detect_damage(img)
            timings["damage_ms"] = (time.perf_counter() - t3) * 1000

        accepted = [s for s in signs if s["accepted"]]
        # an out-of-domain surface reading (e.g. a dashcam view the surface model was never
        # trained on) is shown but never allowed to change the speed advice
        advisory = advise(accepted, surface if surface and surface["in_domain"] else None, current_limit, damage)
        if surface and not surface["in_domain"]:
            advisory["messages"].append("Road surface not assessed: the view is outside the surface model's training "
                                        "domain (it expects a road-facing camera). Use road-patch mode for a close-up.")
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return {"mode": mode, "image_size": list(img.size), "signs": signs, "surface": surface, "damage": damage,
                "advisory": advisory, "timings": {k: round(v, 1) for k, v in timings.items()}}
