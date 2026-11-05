"""Label spaces for the three datasets and the maps between them.

* GTSRB  -- 43 sign classes (classification, the canonical id space for signs)
* GTSDB  -- the same 43 signs in full scenes (Roboflow names, alphabetical ids)
* RSCD   -- 27 road-surface classes, factorised into friction / material / unevenness
"""
from __future__ import annotations

GTSRB_CLASSES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5t", "Right-of-way at next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5t prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road",
    "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5t",
]

# Functional groups used by the advisory layer and in the per-group error analysis.
SPEED_LIMITS = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}
SIGN_GROUP = {}
for i in range(43):
    if i in SPEED_LIMITS:
        SIGN_GROUP[i] = "speed limit"
    elif i in (9, 10, 15, 16, 17):
        SIGN_GROUP[i] = "prohibitory"
    elif i in (11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
        SIGN_GROUP[i] = "danger"
    elif i in (33, 34, 35, 36, 37, 38, 39, 40):
        SIGN_GROUP[i] = "mandatory"
    elif i in (6, 32, 41, 42):
        SIGN_GROUP[i] = "derestriction"
    else:  # 12 priority road, 13 yield, 14 stop
        SIGN_GROUP[i] = "priority"

# Signs that mirror into another valid class under a horizontal flip (used for
# label-preserving / label-swapping flip augmentation).
FLIP_SELF = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]  # not 40: roundabout arrows have a handedness
FLIP_SWAP = {19: 20, 20: 19, 33: 34, 34: 33, 36: 37, 37: 36, 38: 39, 39: 38}

GTSDB_TO_GTSRB = {
    "animals": 31, "construction": 25, "cycles crossing": 29, "danger": 18,
    "no entry": 17, "pedestrian crossing": 27, "school crossing": 28, "snow": 30,
    "stop": 14, "bend": 21, "bend left": 19, "bend right": 20, "give way": 13,
    "go left": 34, "go left or straight": 37, "go right": 33,
    "go right or straight": 36, "go straight": 35, "keep left": 39, "keep right": 38,
    "no overtaking": 9, "no overtaking -trucks-": 10, "no traffic both ways": 15,
    "no trucks": 16, "priority at next intersection": 11, "priority road": 12,
    "restriction ends": 32, "restriction ends -overtaking -trucks--": 42,
    "restriction ends -overtaking-": 41, "restriction ends 80": 6, "road narrows": 24,
    "roundabout": 40, "slippery road": 23, "speed limit 100": 7, "speed limit 120": 8,
    "speed limit 20": 0, "speed limit 30": 1, "speed limit 50": 2, "speed limit 60": 3,
    "speed limit 70": 4, "speed limit 80": 5, "traffic signal": 26, "uneven road": 22,
}

# ---------------------------------------------------------------- road surface
FRICTION = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]
MATERIAL = ["asphalt", "concrete", "gravel", "mud"]
UNEVENNESS = ["smooth", "slight", "severe"]
IGNORE = -100  # PyTorch cross-entropy ignore_index: the attribute is undefined

RSCD_CLASSES = sorted(
    [f"{f}_{m}_{u}" for f in ("dry", "wet", "water") for m in ("asphalt", "concrete") for u in UNEVENNESS]
    + [f"{f}_{m}" for f in ("dry", "wet", "water") for m in ("gravel", "mud")]
    + ["fresh_snow", "melted_snow", "ice"]
)
assert len(RSCD_CLASSES) == 27


def decompose(rscd_label: str) -> tuple[int, int, int]:
    """'wet_asphalt_slight' -> (friction, material, unevenness) indices, IGNORE where undefined.

    Snow / ice cover the surface, so material and unevenness are not observable;
    gravel and mud have no unevenness grade in RSCD.
    """
    if rscd_label in ("fresh_snow", "melted_snow", "ice"):
        return FRICTION.index(rscd_label), IGNORE, IGNORE
    parts = rscd_label.split("_")
    f, m = FRICTION.index(parts[0]), MATERIAL.index(parts[1])
    u = UNEVENNESS.index(parts[2]) if len(parts) == 3 else IGNORE
    return f, m, u


def compose(f: int, m: int, u: int) -> str:
    """Inverse of `decompose` for the 27-way evaluation; ignores heads that do not apply."""
    fr = FRICTION[f]
    if fr in ("fresh_snow", "melted_snow", "ice"):
        return fr
    mat = MATERIAL[m]
    if mat in ("gravel", "mud"):
        return f"{fr}_{mat}"
    return f"{fr}_{mat}_{UNEVENNESS[u]}"
