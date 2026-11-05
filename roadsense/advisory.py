"""Driving advisory: fuse recognised signs with the predicted road-surface condition.

The rule is physical, not learned. Braking distance is d = v^2 / (2 g mu), so the speed
that keeps the *dry-road* stopping distance of the posted limit on a surface with
friction mu is

    v_safe = v_limit * sqrt(mu / mu_dry)

mu is the expectation of typical tyre-road friction coefficients under the friction
head's probabilities, so an uncertain "wet or ice?" prediction yields a cautious
in-between value instead of a hard switch. The coefficients are representative mid-points
of commonly cited ranges for passenger-car tyres; they are an engineering prior, not a
measurement, and the advisory is decision support, not a control signal.
"""
from __future__ import annotations

import math

from .labels import FRICTION, GTSRB_CLASSES, SIGN_GROUP, SPEED_LIMITS, UNEVENNESS

MU = {"dry": 0.75, "wet": 0.50, "water": 0.35, "melted_snow": 0.30, "fresh_snow": 0.22, "ice": 0.10}
MU_DRY = MU["dry"]
ROUGHNESS_FACTOR = {"smooth": 1.0, "slight": 0.9, "severe": 0.75}  # damaged surface: comfort + control margin
DEFAULT_LIMIT = None  # no speed sign seen -> report factors only

HAZARD_SIGNS = {18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31}


def expected_mu(friction_probs: dict[str, float]) -> float:
    return sum(p * MU[k] for k, p in friction_probs.items())


def advise(signs: list[dict], surface: dict | None, current_limit: int | None = None,
           damage: list[dict] | None = None) -> dict:
    """signs: accepted detections [{class_id, conf}]; surface: {"friction": {name: p}, "unevenness": {name: p} | None};
    damage: road-damage detections from the dashcam view [{name, score}], already thresholded.

    Returns posted / recommended speed, hazard level 0-3 and human-readable messages.
    """
    msgs: list[str] = []
    limit = current_limit
    speed_signs = [s for s in signs if s["class_id"] in SPEED_LIMITS]
    if speed_signs:
        limit = SPEED_LIMITS[max(speed_signs, key=lambda s: s["conf"])["class_id"]]
    if any(s["class_id"] in (6, 32) for s in signs):  # end-of-limit signs
        msgs.append("End of restriction sign: national limit applies.")
        if not speed_signs:
            limit = None

    hazard = 0
    for s in signs:
        cid = s["class_id"]
        if cid == 14:
            msgs.append("STOP sign ahead: prepare to stop.")
            hazard = max(hazard, 2)
        elif cid == 17:
            msgs.append("No entry.")
            hazard = max(hazard, 2)
        elif cid in HAZARD_SIGNS:
            msgs.append(f"Warning sign: {GTSRB_CLASSES[cid]}.")
            hazard = max(hazard, 1)

    out = {"posted_limit_kmh": limit, "recommended_kmh": limit, "mu": None, "braking_distance_ratio": None}
    if surface:
        f = surface["friction"]
        mu = expected_mu(f)
        ratio = MU_DRY / mu
        rough = 1.0
        if surface.get("unevenness"):
            rough = sum(p * ROUGHNESS_FACTOR[k] for k, p in surface["unevenness"].items())
        out["mu"] = round(mu, 3)
        out["braking_distance_ratio"] = round(ratio, 2)
        top = max(f, key=f.get)
        if top != "dry":
            msgs.append(f"{top.replace('_', ' ').capitalize()} surface (p={f[top]:.2f}): braking distance ~{ratio:.1f}x dry.")
        if top == "water":
            msgs.append("Standing water: aquaplaning risk, avoid sudden steering.")
        if surface.get("unevenness"):
            u = surface["unevenness"]
            if u.get("severe", 0) > 0.5:
                msgs.append("Severely damaged surface (potholes / heavy cracking).")
                hazard = max(hazard, 2)
            elif u.get("slight", 0) > 0.5:
                msgs.append("Uneven surface.")
                hazard = max(hazard, 1)
        if mu < 0.25:
            hazard = 3
        elif mu < 0.45:
            hazard = max(hazard, 2)
        elif mu < 0.65:
            hazard = max(hazard, 1)
        if limit is not None:
            v = limit * math.sqrt(mu / MU_DRY) * rough
            out["recommended_kmh"] = int(min(limit, 5 * round(v / 5)))
            if out["recommended_kmh"] < limit:
                msgs.append(f"Recommended {out['recommended_kmh']} km/h (posted {limit}).")
    # Road damage seen from the dashcam. Works where the surface model abstains (forward views),
    # so it is the only roughness signal for most scene frames. Single linear cracks are
    # reported but do not change the advice; alligator cracking and potholes do.
    if damage:
        kinds = {d["name"] for d in damage}
        rough = 1.0
        if "pothole" in kinds:
            n = sum(d["name"] == "pothole" for d in damage)
            msgs.append(f"Pothole{'s' if n > 1 else ''} ahead ({n}): slow down and keep your line.")
            hazard, rough = max(hazard, 2), ROUGHNESS_FACTOR["severe"]
        if "alligator crack" in kinds:
            msgs.append("Alligator cracking: broken surface ahead.")
            hazard, rough = max(hazard, 1), min(rough, ROUGHNESS_FACTOR["slight"])
        if kinds & {"longitudinal crack", "transverse crack"}:
            msgs.append("Cracked road surface.")
        if rough < 1.0 and limit is not None:
            v = int(min(out["recommended_kmh"] or limit, 5 * round(limit * rough / 5)))
            if v < (out["recommended_kmh"] or limit):
                out["recommended_kmh"] = v
                msgs.append(f"Recommended {v} km/h for the damaged surface (posted {limit}).")
    out["hazard_level"] = hazard
    out["hazard_label"] = ["low", "moderate", "high", "severe"][hazard]
    out["messages"] = msgs
    return out
